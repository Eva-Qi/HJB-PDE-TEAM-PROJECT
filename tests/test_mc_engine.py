"""Tests for Monte Carlo SDE engine."""

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS, almgren_chriss_closed_form
from shared.cost_model import execution_cost
from montecarlo.sde_engine import (
    simulate_gbm_paths,
    simulate_execution,
    simulate_execution_with_control_variate,
)
from montecarlo.strategies import twap_trajectory, optimal_trajectory


class TestGBMPaths:
    """Test pure GBM path simulation.

    DELETED per 2026-04-19 audit (4 tests):
      test_initial_price       — literal assignment S[:,0] = S0
      test_shape               — numpy allocation
      test_positive_prices     — covered tighter by test_high_vol_positive_prices
      test_reproducibility     — tests numpy RNG, not project code
    Surviving: test_antithetic_mean_zero_noise (tight structural).
    """

    def test_antithetic_mean_zero_noise(self):
        """Antithetic variates: Z and -Z should give paths centered on E[S]."""
        S = simulate_gbm_paths(DEFAULT_PARAMS, n_paths=10000, antithetic=True)
        # Terminal prices should be centered around S0 (mu=0)
        mean_terminal = np.mean(S[:, -1])
        # With mu=0, E[S_T] ≈ S0 (for small sigma^2*T)
        rel_err = abs(mean_terminal - DEFAULT_PARAMS.S0) / DEFAULT_PARAMS.S0
        assert rel_err < 0.02, f"Mean terminal price {mean_terminal:.2f} too far from S0"


class TestExecutionSimulation:
    """Test execution cost simulation."""

    def test_twap_mean_cost_matches_deterministic(self):
        """MC mean cost for TWAP should approximate deterministic execution_cost.

        With mu=0, the stochastic price paths average out, so the MC mean
        cost should converge to the deterministic cost.
        """
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        det_cost = execution_cost(x_twap, DEFAULT_PARAMS)

        _, mc_costs = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=20000, antithetic=True
        )
        mc_mean = np.mean(mc_costs)

        rel_err = abs(mc_mean - det_cost) / abs(det_cost)
        assert rel_err < 0.05, (
            f"MC mean cost ({mc_mean:.2f}) differs from deterministic "
            f"({det_cost:.2f}) by {rel_err*100:.1f}%"
        )

    def test_optimal_mean_cost_near_closed_form(self):
        """MC mean cost for optimal should be near closed-form cost."""
        _, x_opt, cf_cost = almgren_chriss_closed_form(DEFAULT_PARAMS)

        _, mc_costs = simulate_execution(
            DEFAULT_PARAMS, x_opt, n_paths=20000, antithetic=True
        )
        mc_mean = np.mean(mc_costs)

        rel_err = abs(mc_mean - cf_cost) / abs(cf_cost)
        assert rel_err < 0.05, (
            f"MC optimal cost ({mc_mean:.2f}) differs from closed-form "
            f"({cf_cost:.2f}) by {rel_err*100:.1f}%"
        )

    # DELETED test_optimal_objective_less_than_twap_mc per 2026-04-19 audit
    # — duplicates test_optimal_beats_twap_exact in test_new_features.py AND
    # test_optimal_objective_less_than_twap in test_closed_form.py.

    def test_cost_has_positive_variance(self):
        """Execution cost should have nonzero variance (stochastic)."""
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        _, costs = simulate_execution(DEFAULT_PARAMS, x_twap, n_paths=1000)
        assert np.std(costs) > 0, "Cost distribution should have positive std"

    def test_antithetic_reduces_estimator_variance(self):
        """Antithetic should reduce the MC ESTIMATOR's variance across seeds.

        Methodological note (GLM audit + debug): the previous test measured
        `np.var(costs)` which is sample variance (bimodal for antithetic
        because half the samples come from +Z, half from -Z — sample spread
        doesn't shrink). The RIGHT metric is Var[mean_cost] across
        independent seeds, which captures estimator precision. For
        anti-symmetric payoffs, antithetic roughly halves this estimator
        variance.

        If this fails: likely the antithetic draw isn't actually ±Z, or
        the payoff is not anti-symmetric in Z (which for linear execution
        cost under GBM it should be, approximately).
        """
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        N_SEEDS = 20
        means_plain = []
        means_anti = []
        for seed in range(N_SEEDS):
            _, cp = simulate_execution(
                DEFAULT_PARAMS, x_twap, n_paths=2000,
                seed=seed, antithetic=False,
            )
            _, ca = simulate_execution(
                DEFAULT_PARAMS, x_twap, n_paths=2000,
                seed=seed, antithetic=True,
            )
            means_plain.append(float(np.mean(cp)))
            means_anti.append(float(np.mean(ca)))

        var_plain = float(np.var(means_plain, ddof=1))
        var_anti = float(np.var(means_anti, ddof=1))
        ratio = var_anti / var_plain if var_plain > 0 else float("inf")

        assert ratio < 0.5, (
            f"Estimator variance ratio (antithetic/plain) across "
            f"{N_SEEDS} seeds: Var[mean_anti]={var_anti:.4e}, "
            f"Var[mean_plain]={var_plain:.4e}, ratio={ratio:.3f}. "
            f"Expected < 0.5 (antithetic ~halves estimator variance for "
            f"anti-symmetric payoffs). If close to 1.0, antithetic is "
            f"providing no actual noise cancellation — the ±Z pairing or "
            f"payoff symmetry is broken."
        )


class TestControlVariate:
    """Test control variate variance reduction (previously untested)."""

    def test_control_variate_preserves_mean(self):
        """Control variate is UNBIASED by construction:
            E[C_cv] = E[C_opt] - β·(E[C_twap] - E[C_twap]) = E[C_opt]

        GLM verdict: old 10% tolerance "hides massive bias". The true
        property is ZERO expected bias — any deviation is sampling error.
        With 20k paths, CLT bound on the difference is ~0.5%, so 2% is
        a sharp but not fragile threshold.

        If this fails with > 2% shift, the CV implementation is likely
        using a biased β estimator or mismatched variate pairing.
        """
        x_opt = optimal_trajectory(DEFAULT_PARAMS)
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        _, costs_plain = simulate_execution(
            DEFAULT_PARAMS, x_opt, n_paths=20000, seed=42
        )
        _, costs_cv = simulate_execution_with_control_variate(
            DEFAULT_PARAMS, x_opt, x_twap, n_paths=20000, seed=42
        )

        rel_diff = abs(np.mean(costs_cv) - np.mean(costs_plain)) / abs(np.mean(costs_plain))
        assert rel_diff < 0.02, (
            f"CV mean shifted by {rel_diff*100:.2f}% (> 2% threshold). "
            f"CV should be unbiased by construction — large bias suggests "
            f"β-estimator issue or mispaired control variate."
        )

    def test_control_variate_reduces_variance(self):
        """CV should reduce variance compared to plain MC."""
        x_opt = optimal_trajectory(DEFAULT_PARAMS)
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        _, costs_plain = simulate_execution(
            DEFAULT_PARAMS, x_opt, n_paths=10000, seed=42
        )
        _, costs_cv = simulate_execution_with_control_variate(
            DEFAULT_PARAMS, x_opt, x_twap, n_paths=10000, seed=42
        )

        assert np.var(costs_cv) < np.var(costs_plain), (
            f"CV variance ({np.var(costs_cv):.2f}) should be less than "
            f"plain ({np.var(costs_plain):.2f})"
        )


class TestLogNormalGBM:
    """Test that log-normal GBM guarantees positive prices."""

    def test_high_vol_positive_prices(self):
        """Even with sigma=1.0, all prices must be positive."""
        from shared.params import ACParams
        high_vol = ACParams(
            S0=50.0, sigma=1.0, mu=0.0, X0=1e6,
            T=1.0, N=50, gamma=2.5e-7, eta=2.5e-6,
            alpha=1.0, lam=1e-3,
        )
        S = simulate_gbm_paths(high_vol, n_paths=5000, seed=42)
        assert np.all(S > 0), "Log-normal GBM must produce positive prices"
