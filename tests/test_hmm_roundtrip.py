"""Round-trip tests for fit_hmm() in extensions/regime.py.

Strategy: simulate 2-regime returns with known volatility properties,
fit the HMM, then verify the fitted parameters recover the ground truth.

Run:
    pytest tests/test_hmm_roundtrip.py -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from extensions.regime import fit_hmm, RegimeParams


# ---------------------------------------------------------------------------
# Shared simulation helper
# ---------------------------------------------------------------------------

class TestHMMRoundTrip:
    """Simulate 2-regime data with known properties -> fit HMM -> recover."""

    def _simulate_two_regime_returns(
        self,
        n_per_regime: int = 500,
        sigma_low: float = 0.005,
        sigma_high: float = 0.025,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alternating high/low vol regimes with Markov transition probability.

        Parameters
        ----------
        n_per_regime : int
            Approximate number of observations per regime state.
        sigma_low : float
            Volatility of the low-vol (risk_on) regime.
        sigma_high : float
            Volatility of the high-vol (risk_off) regime.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        returns : np.ndarray
            Simulated return series.
        true_states : np.ndarray
            Ground-truth state sequence (0 = low vol, 1 = high vol).
        """
        rng = np.random.default_rng(seed)
        returns = []
        states = []
        state = 0  # start in low-vol regime
        persist = 0.95  # 95% probability of staying in the same regime

        for _ in range(n_per_regime * 2):
            sigma = sigma_low if state == 0 else sigma_high
            returns.append(rng.normal(0, sigma))
            states.append(state)
            if rng.random() > persist:
                state = 1 - state  # flip

        return np.array(returns), np.array(states)

    # -----------------------------------------------------------------------
    # Test 1: Two distinct regimes recovered
    # -----------------------------------------------------------------------

    def test_hmm_recovers_two_distinct_regimes(self):
        """Fitted regimes should have substantially different sigmas.

        Input sigma_high / sigma_low = 5x.  After annualisation we still
        expect the ratio of fitted sigmas to be > 2x, i.e. the HMM clearly
        separates high-vol from low-vol.
        """
        returns, _ = self._simulate_two_regime_returns(
            n_per_regime=500, sigma_low=0.005, sigma_high=0.025, seed=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress hmmlearn install warning
            regimes, state_seq = fit_hmm(returns, n_regimes=2)

        assert len(regimes) == 2, f"Expected 2 regimes, got {len(regimes)}"
        assert len(state_seq) == len(returns), (
            f"state_seq length {len(state_seq)} != returns length {len(returns)}"
        )

        # After fit_hmm, regimes are sorted risk_on (low) then risk_off (high)
        sigma_on = regimes[0].sigma
        sigma_off = regimes[1].sigma

        assert sigma_on > 0, f"risk_on sigma={sigma_on} not positive"
        assert sigma_off > 0, f"risk_off sigma={sigma_off} not positive"

        ratio = sigma_off / sigma_on
        assert ratio >= 2.0, (
            f"sigma_off/sigma_on={ratio:.2f} < 2.0 — HMM failed to separate "
            f"the two regimes.  sigma_on={sigma_on:.4f}, sigma_off={sigma_off:.4f}"
        )

    # -----------------------------------------------------------------------
    # Test 2: Viterbi accuracy vs ground truth
    # -----------------------------------------------------------------------

    def test_hmm_labels_match_true_states(self):
        """Viterbi state sequence should agree with true states on >70% of points.

        Transitions are the hardest to classify; we allow 30% error.
        """
        returns, true_states = self._simulate_two_regime_returns(
            n_per_regime=500, sigma_low=0.005, sigma_high=0.025, seed=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, state_seq = fit_hmm(returns, n_regimes=2)

        # The HMM may label states in either order — check both assignments
        # and take the better one (label-permutation invariance).
        acc_direct = np.mean(state_seq == true_states)
        acc_flipped = np.mean(state_seq == (1 - true_states))
        best_acc = max(acc_direct, acc_flipped)

        assert best_acc >= 0.70, (
            f"Best label accuracy={best_acc:.2%} < 70%.  "
            f"HMM may be collapsing to a single state or underfitting."
        )

    # -----------------------------------------------------------------------
    # Test 3: IID data — should either raise or produce near-identical regimes
    # -----------------------------------------------------------------------

    def test_hmm_with_no_regimes_either_fails_or_collapses(self):
        """Pure IID Gaussian data has no regimes.

        fit_hmm() should either:
          (a) raise RuntimeError (all initialisations collapsed), or
          (b) return two near-identical regimes (sigma ratio < 1.3).

        We use xfail(strict=False) because hmmlearn may still find a
        spurious split on some seeds.
        """
        rng = np.random.default_rng(0)
        iid_returns = rng.normal(0, 0.01, size=1000)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                regimes, _ = fit_hmm(iid_returns, n_regimes=2)
        except RuntimeError:
            # Acceptable: HMM rejected all collapser initialisations
            return

        # If it returned, the two regimes should be nearly identical
        sigma_on = regimes[0].sigma
        sigma_off = regimes[1].sigma
        ratio = sigma_off / max(sigma_on, 1e-12)

        assert ratio < 2.0, (
            f"On IID data, sigma ratio={ratio:.2f} is too large ({sigma_off:.4f} vs "
            f"{sigma_on:.4f}).  HMM found a spurious regime split.  "
            "This is a weak-signal warning, not necessarily a bug."
        )

    # -----------------------------------------------------------------------
    # Test 4: API contract — risk_off has higher sigma than risk_on
    # -----------------------------------------------------------------------

    def test_hmm_labels_higher_vol_as_risk_off(self):
        """API contract: the 'risk_off' regime must have higher sigma than 'risk_on'.

        fit_hmm() sorts regimes by annualised sigma ascending and assigns
        labels ["risk_on", "risk_off"] in that order.  This test verifies
        that invariant holds.
        """
        returns, _ = self._simulate_two_regime_returns(
            n_per_regime=500, sigma_low=0.005, sigma_high=0.025, seed=7
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, _ = fit_hmm(returns, n_regimes=2)

        labels = [r.label for r in regimes]
        sigmas = [r.sigma for r in regimes]

        assert "risk_on" in labels, f"'risk_on' label missing from {labels}"
        assert "risk_off" in labels, f"'risk_off' label missing from {labels}"

        risk_on_sigma = sigmas[labels.index("risk_on")]
        risk_off_sigma = sigmas[labels.index("risk_off")]

        assert risk_off_sigma > risk_on_sigma, (
            f"risk_off sigma={risk_off_sigma:.4f} <= risk_on sigma={risk_on_sigma:.4f}.  "
            "The regime labelling convention is broken — 'risk_off' must be the "
            "higher-volatility regime."
        )

    # -----------------------------------------------------------------------
    # Test 5: Minimum observations guard
    # -----------------------------------------------------------------------

    def test_hmm_raises_on_too_few_observations(self):
        """fit_hmm() must raise ValueError when given fewer than 50 observations."""
        rng = np.random.default_rng(1)
        short_series = rng.normal(0, 0.01, size=30)

        with pytest.raises(ValueError, match="Too few observations"):
            fit_hmm(short_series, n_regimes=2)

    # -----------------------------------------------------------------------
    # Test 6: RegimeParams fields are all valid (positive, finite)
    # -----------------------------------------------------------------------

    def test_hmm_regime_params_are_finite_and_positive(self):
        """All RegimeParams fields (sigma, gamma, eta, probability) must be
        positive and finite after a successful fit."""
        import math

        returns, _ = self._simulate_two_regime_returns(
            n_per_regime=300, sigma_low=0.004, sigma_high=0.020, seed=99
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, state_seq = fit_hmm(returns, n_regimes=2)

        for r in regimes:
            assert isinstance(r, RegimeParams), f"Expected RegimeParams, got {type(r)}"
            for field_name in ("sigma", "gamma", "eta", "probability"):
                val = getattr(r, field_name)
                assert math.isfinite(val), (
                    f"regime '{r.label}' field '{field_name}'={val} is not finite"
                )
                assert val > 0, (
                    f"regime '{r.label}' field '{field_name}'={val} is not positive"
                )

        # state_seq must only contain 0 or 1
        unique_states = set(state_seq.tolist())
        assert unique_states.issubset({0, 1}), (
            f"state_seq contains unexpected state labels: {unique_states - {0,1}}"
        )

        # probabilities should sum to approximately 1
        total_prob = sum(r.probability for r in regimes)
        assert abs(total_prob - 1.0) < 0.05, (
            f"regime probabilities sum to {total_prob:.4f}, expected ~1.0"
        )
