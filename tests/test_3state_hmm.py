"""Tests for 3-state HMM support in extensions/regime.py.

Synthetic 3-regime data with known (sigma_low, sigma_mid, sigma_high).
Verifies:
  1. fit_hmm returns exactly 3 RegimeParams in ascending sigma order.
  2. Labels are ["risk_on", "neutral", "risk_off"].
  3. Sigma multipliers respect the ordering risk_on < neutral < risk_off.
  4. All fields are finite and positive.
  5. State sequence uses only labels {0, 1, 2}.
  6. ValueError for unsupported n_regimes (e.g. 4).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from extensions.regime import fit_hmm, RegimeParams


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def _simulate_three_regime_returns(
    n_per_regime: int = 600,
    sigma_low: float = 0.004,
    sigma_mid: float = 0.012,
    sigma_high: float = 0.030,
    persist: float = 0.97,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a 3-state Markov-switching returns series.

    States cycle: 0 (low vol) → 1 (mid vol) → 2 (high vol) with
    ``persist`` probability of staying in the current state and equal
    probability of transitioning to each other state.

    Returns
    -------
    returns : np.ndarray, shape (n_per_regime * 3,)
    true_states : np.ndarray, same length, values in {0, 1, 2}
    """
    rng = np.random.default_rng(seed)
    sigmas = [sigma_low, sigma_mid, sigma_high]
    n_states = 3
    returns = []
    states = []
    state = 0

    for _ in range(n_per_regime * n_states):
        returns.append(rng.normal(0.0, sigmas[state]))
        states.append(state)
        if rng.random() > persist:
            # Transition to one of the other two states at random
            other = [s for s in range(n_states) if s != state]
            state = int(rng.choice(other))

    return np.array(returns), np.array(states)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestThreeStateHMM:

    def test_returns_three_regimes(self):
        """fit_hmm with n_regimes=3 returns a list of exactly 3 RegimeParams."""
        returns, _ = _simulate_three_regime_returns(n_per_regime=600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, state_seq = fit_hmm(returns, n_regimes=3)

        assert len(regimes) == 3, f"Expected 3 regimes, got {len(regimes)}"
        assert len(state_seq) == len(returns)

    def test_labels_are_correct(self):
        """Labels must be exactly ['risk_on', 'neutral', 'risk_off'] in order."""
        returns, _ = _simulate_three_regime_returns(n_per_regime=600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, _ = fit_hmm(returns, n_regimes=3)

        labels = [r.label for r in regimes]
        assert labels == ["risk_on", "neutral", "risk_off"], (
            f"Got labels {labels}; expected ['risk_on', 'neutral', 'risk_off']"
        )

    def test_sigma_multipliers_are_ordered(self):
        """Sigma multipliers must be strictly increasing: risk_on < neutral < risk_off."""
        returns, _ = _simulate_three_regime_returns(
            n_per_regime=600,
            sigma_low=0.004,
            sigma_mid=0.012,
            sigma_high=0.030,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, _ = fit_hmm(returns, n_regimes=3)

        sigma_on = regimes[0].sigma
        sigma_neutral = regimes[1].sigma
        sigma_off = regimes[2].sigma

        assert sigma_on < sigma_neutral, (
            f"risk_on sigma ({sigma_on:.4f}) >= neutral sigma ({sigma_neutral:.4f})"
        )
        assert sigma_neutral < sigma_off, (
            f"neutral sigma ({sigma_neutral:.4f}) >= risk_off sigma ({sigma_off:.4f})"
        )

    def test_all_fields_finite_and_positive(self):
        """All RegimeParams numeric fields must be finite and positive."""
        returns, _ = _simulate_three_regime_returns(n_per_regime=500, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, state_seq = fit_hmm(returns, n_regimes=3)

        for r in regimes:
            assert isinstance(r, RegimeParams)
            for field in ("sigma", "gamma", "eta", "probability"):
                val = getattr(r, field)
                assert math.isfinite(val), (
                    f"Regime '{r.label}' field '{field}'={val} is not finite"
                )
                assert val > 0, (
                    f"Regime '{r.label}' field '{field}'={val} is not positive"
                )

        # State sequence must only contain {0, 1, 2}
        unique = set(state_seq.tolist())
        assert unique.issubset({0, 1, 2}), (
            f"state_seq contains unexpected labels: {unique - {0, 1, 2}}"
        )

    def test_probabilities_sum_to_one(self):
        """Regime stationary probabilities must sum to ~1.0."""
        returns, _ = _simulate_three_regime_returns(n_per_regime=500, seed=99)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, _ = fit_hmm(returns, n_regimes=3)

        total = sum(r.probability for r in regimes)
        assert abs(total - 1.0) < 0.05, (
            f"Probabilities sum to {total:.4f}, expected ~1.0"
        )

    def test_sigma_ratio_detects_separation(self):
        """risk_off sigma should be substantially higher than risk_on.

        With sigma_high/sigma_low = 7.5x, we expect the fitted ratio
        risk_off.sigma / risk_on.sigma to be >= 2.0, confirming the HMM
        separated the extreme regimes.
        """
        returns, _ = _simulate_three_regime_returns(
            n_per_regime=600,
            sigma_low=0.004,
            sigma_high=0.030,
            seed=11,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, _ = fit_hmm(returns, n_regimes=3)

        ratio = regimes[2].sigma / regimes[0].sigma
        assert ratio >= 2.0, (
            f"risk_off/risk_on sigma ratio={ratio:.2f} < 2.0 — "
            f"HMM may not have separated extreme regimes."
        )

    def test_unsupported_n_regimes_raises(self):
        """n_regimes=4 (or any value not in {2, 3}) must raise ValueError."""
        returns, _ = _simulate_three_regime_returns(n_per_regime=200, seed=0)
        with pytest.raises(ValueError, match="n_regimes=4"):
            fit_hmm(returns, n_regimes=4)

    def test_n_regimes_2_still_works(self):
        """Existing 2-state API must be unbroken after the 3-state extension."""
        returns, _ = _simulate_three_regime_returns(n_per_regime=400, seed=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, state_seq = fit_hmm(returns, n_regimes=2)

        assert len(regimes) == 2
        labels = [r.label for r in regimes]
        assert "risk_on" in labels
        assert "risk_off" in labels
        assert set(state_seq.tolist()).issubset({0, 1})
