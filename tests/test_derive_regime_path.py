"""Tests for derive_regime_path — the HMM → execution-grid bridge.

Three modes, each capturing a different narrative about how a historical
HMM state sequence should inform a future execution regime_path:

  - "current"    : constant at last HMM state
  - "historical" : block-mode downsample of history
  - "sample"     : forward sample from empirical transition matrix

Each mode has its own invariants. These tests verify the algorithmic
contract, not statistical properties. For statistical properties (does
regime-aware execution beat single-regime?), see
scripts/paired_test_regime_aware.py.
"""

from dataclasses import replace

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS
from extensions.regime import (
    RegimeParams,
    derive_regime_path,
    _empirical_transition_matrix,
)


# Synthetic regimes matching the shape of fit_hmm output. sigma/gamma/eta
# are dimensionless multipliers per the post-Yuhao-merge convention.
_SYNTHETIC_REGIMES = [
    RegimeParams(
        label="risk_on", sigma=0.9, gamma=0.9, eta=0.9, probability=0.6,
        state_vol=0.1, state_abs_ret=0.005, state_mean_ret=0.0,
    ),
    RegimeParams(
        label="risk_off", sigma=1.2, gamma=1.2, eta=1.2, probability=0.4,
        state_vol=0.3, state_abs_ret=0.02, state_mean_ret=0.0,
    ),
]


class TestCurrentMode:
    """mode="current" — constant at the last historical state."""

    def test_output_length_equals_N(self):
        state_seq = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=int)
        path = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq), mode="current",
        )
        assert path.shape == (DEFAULT_PARAMS.N,)
        assert path.dtype == np.int64 or path.dtype == int

    def test_all_values_equal_last_state(self):
        """Entire output must be the last state of the sequence."""
        state_seq = np.array([0, 0, 0, 0, 1], dtype=int)  # last = 1
        path = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq), mode="current",
        )
        assert np.all(path == 1)

    def test_insensitive_to_historical_pattern(self):
        """Output depends ONLY on last state — changing earlier history
        does not change output."""
        seq_a = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)  # alternating
        seq_b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=int)  # mostly 1s
        # Both end in 0 → same output
        path_a = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, seq_a), mode="current",
        )
        path_b = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, seq_b), mode="current",
        )
        np.testing.assert_array_equal(path_a, path_b)
        assert np.all(path_a == 0)


class TestHistoricalMode:
    """mode="historical" — block-mode downsample of historical sequence."""

    def test_output_length_equals_N(self):
        state_seq = np.array([0] * 100 + [1] * 100, dtype=int)
        path = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq), mode="historical",
        )
        assert path.shape == (DEFAULT_PARAMS.N,)

    def test_preserves_regime_proportions(self):
        """50/50 historical split → output ≈ 50/50 split.

        With state_seq = [0]*1000 + [1]*1000 and N=50, the first ~25
        blocks are all zeros, last ~25 blocks are all ones. The downsample
        must preserve this.
        """
        state_seq = np.array([0] * 1000 + [1] * 1000, dtype=int)
        params = replace(DEFAULT_PARAMS, N=50)
        path = derive_regime_path(
            params, (_SYNTHETIC_REGIMES, state_seq), mode="historical",
        )
        # First half should be 0, second half should be 1
        assert np.all(path[:25] == 0)
        assert np.all(path[25:] == 1)

    def test_short_sequence_falls_back_to_block_size_1(self):
        """When len(state_seq) < N, block_size is clamped to 1 so each
        output element maps to one input element."""
        state_seq = np.array([0, 1, 1, 0], dtype=int)  # length 4
        params = replace(DEFAULT_PARAMS, N=50)
        # Should not crash; block_size = max(4 // 50, 1) = 1
        path = derive_regime_path(
            params, (_SYNTHETIC_REGIMES, state_seq), mode="historical",
        )
        assert path.shape == (50,)
        assert path.min() >= 0 and path.max() <= 1

    def test_differs_from_current_when_history_is_not_constant(self):
        """Historical mode should produce non-constant output when the
        input had multiple regimes — distinguishes it from 'current'."""
        state_seq = np.array([0] * 500 + [1] * 500, dtype=int)
        params = replace(DEFAULT_PARAMS, N=10)
        path_hist = derive_regime_path(
            params, (_SYNTHETIC_REGIMES, state_seq), mode="historical",
        )
        path_curr = derive_regime_path(
            params, (_SYNTHETIC_REGIMES, state_seq), mode="current",
        )
        # Historical should show both regimes; current sees only last
        assert len(set(path_hist.tolist())) == 2, "historical should have both states"
        assert len(set(path_curr.tolist())) == 1, "current should be constant"


class TestSampleMode:
    """mode="sample" — forward sampling from transition matrix."""

    def test_output_length_equals_N(self):
        state_seq = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        rng = np.random.default_rng(42)
        path = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq),
            mode="sample", rng=rng,
        )
        assert path.shape == (DEFAULT_PARAMS.N,)

    def test_identity_transition_preserves_initial_state(self):
        """With an identity-like transition matrix (perfect persistence),
        the sampled path must stay in the starting state."""
        state_seq = np.array([0, 1, 1, 1, 1], dtype=int)  # ends at 1
        identity = np.eye(2)
        rng = np.random.default_rng(42)
        path = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq),
            mode="sample", transition_matrix=identity, rng=rng,
        )
        assert np.all(path == 1), (
            f"Identity transmat should lock state at 1; got {np.unique(path)}"
        )

    def test_uniform_transition_mixes_states(self):
        """With a fully-uniform transition matrix (0.5/0.5), sampled path
        should span both states at large N."""
        state_seq = np.array([0, 1, 0, 1], dtype=int)
        uniform = np.full((2, 2), 0.5)
        params = replace(DEFAULT_PARAMS, N=1000)
        rng = np.random.default_rng(42)
        path = derive_regime_path(
            params, (_SYNTHETIC_REGIMES, state_seq),
            mode="sample", transition_matrix=uniform, rng=rng,
        )
        fraction_1 = path.mean()
        # Stationary distribution of uniform 2-state is (0.5, 0.5)
        assert 0.4 < fraction_1 < 0.6, (
            f"Uniform transmat should produce ~50% state 1; got {fraction_1:.3f}"
        )

    def test_derives_transition_matrix_empirically_if_not_provided(self):
        """When transition_matrix=None, it must be estimated from
        state_sequence via empirical counts. Setup: 10 zeros followed by
        a long run of ones. Empirical transmat:
            0→0 count=9, 0→1 count=1  → P(0→*) = (0.9, 0.1)
            1→1 count=989, 1→0 count=0 → P(1→*) = (0.0, 1.0)
        Starting from state_seq[-1] = 1, path must be ALL ones (absorbing)."""
        state_seq = np.concatenate([
            np.zeros(10, dtype=int),
            np.ones(990, dtype=int),
        ])
        params = replace(DEFAULT_PARAMS, N=100)
        rng = np.random.default_rng(42)
        path = derive_regime_path(
            params, (_SYNTHETIC_REGIMES, state_seq),
            mode="sample", rng=rng,
        )
        # 1→1 is absorbing in the empirical transmat → path is all ones
        assert np.all(path == 1), (
            f"Empirical transmat should lock state at 1; got "
            f"{path.mean()*100:.1f}% ones"
        )

    def test_seed_determinism(self):
        """Same seed → same path."""
        state_seq = np.array([0, 1, 0, 1, 0], dtype=int)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        p1 = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq),
            mode="sample", rng=rng1,
        )
        p2 = derive_regime_path(
            DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq),
            mode="sample", rng=rng2,
        )
        np.testing.assert_array_equal(p1, p2)


class TestErrorPaths:

    def test_unknown_mode_raises(self):
        state_seq = np.array([0, 1], dtype=int)
        with pytest.raises(ValueError, match="Unknown mode"):
            derive_regime_path(
                DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq),
                mode="nonexistent",
            )

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="empty"):
            derive_regime_path(
                DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, np.array([], dtype=int)),
                mode="current",
            )

    def test_wrong_transition_matrix_shape_raises(self):
        state_seq = np.array([0, 1, 0, 1], dtype=int)
        bad_transmat = np.eye(3)  # 3x3 but only 2 states
        with pytest.raises(ValueError, match="shape"):
            derive_regime_path(
                DEFAULT_PARAMS, (_SYNTHETIC_REGIMES, state_seq),
                mode="sample", transition_matrix=bad_transmat,
            )


class TestEmpiricalTransitionMatrix:
    """Sanity check on the helper that powers sample mode's default transmat."""

    def test_persistent_sequence_produces_high_diagonal(self):
        seq = np.array([0] * 100 + [1] * 100, dtype=int)
        transmat = _empirical_transition_matrix(seq, n_states=2)
        # Diagonal should be near 1 (persistence)
        assert transmat[0, 0] > 0.95
        assert transmat[1, 1] > 0.95 or transmat[1].sum() == 0  # last row may be empty

    def test_alternating_sequence_produces_high_off_diagonal(self):
        seq = np.tile([0, 1], 50)
        transmat = _empirical_transition_matrix(seq, n_states=2)
        # Off-diagonal should dominate
        assert transmat[0, 1] > 0.95
        assert transmat[1, 0] > 0.95

    def test_rows_sum_to_one(self):
        seq = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0], dtype=int)
        transmat = _empirical_transition_matrix(seq, n_states=2)
        np.testing.assert_allclose(transmat.sum(axis=1), 1.0, atol=1e-12)
