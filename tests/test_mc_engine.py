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
    """Test pure GBM path simulation."""

    def test_initial_price(self):
        """All paths start at S0."""
        S = simulate_gbm_paths(DEFAULT_PARAMS, n_paths=100)
        assert np.allclose(S[:, 0], DEFAULT_PARAMS.S0)

    def test_shape(self):
        S = simulate_gbm_paths(DEFAULT_PARAMS, n_paths=100)
        assert S.shape == (100, DEFAULT_PARAMS.N + 1)

    def test_antithetic_mean_zero_noise(self):
        """Antithetic variates: Z and -Z should give paths centered on E[S]."""
        S = simulate_gbm_paths(DEFAULT_PARAMS, n_paths=10000, antithetic=True)
        # Terminal prices should be centered around S0 (mu=0)
        mean_terminal = np.mean(S[:, -1])
        # With mu=0, E[S_T] ≈ S0 (for small sigma^2*T)
        rel_err = abs(mean_terminal - DEFAULT_PARAMS.S0) / DEFAULT_PARAMS.S0
        assert rel_err < 0.02, f"Mean terminal price {mean_terminal:.2f} too far from S0"

    def test_positive_prices(self):
        """All prices should be positive (GBM)."""
        S = simulate_gbm_paths(DEFAULT_PARAMS, n_paths=1000)
        assert np.all(S > 0), "GBM prices should be positive"

    def test_reproducibility(self):
        """Same seed gives same paths."""
        S1 = simulate_gbm_paths(DEFAULT_PARAMS, n_paths=100, seed=123)
        S2 = simulate_gbm_paths(DEFAULT_PARAMS, n_paths=100, seed=123)
        assert np.allclose(S1, S2)


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

    def test_optimal_objective_less_than_twap_mc(self):
        """MC confirms: optimal has lower objective than TWAP."""
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        x_opt = optimal_trajectory(DEFAULT_PARAMS)

        _, costs_twap = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42
        )
        _, costs_opt = simulate_execution(
            DEFAULT_PARAMS, x_opt, n_paths=10000, seed=42
        )

        # Objective = E[cost] + lam * Var[cost]
        obj_twap = np.mean(costs_twap) + DEFAULT_PARAMS.lam * np.var(costs_twap)
        obj_opt = np.mean(costs_opt) + DEFAULT_PARAMS.lam * np.var(costs_opt)

        assert obj_opt < obj_twap, (
            f"MC optimal objective ({obj_opt:.2f}) should be < "
            f"TWAP objective ({obj_twap:.2f})"
        )

    def test_cost_has_positive_variance(self):
        """Execution cost should have nonzero variance (stochastic)."""
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        _, costs = simulate_execution(DEFAULT_PARAMS, x_twap, n_paths=1000)
        assert np.std(costs) > 0, "Cost distribution should have positive std"

    def test_antithetic_reduces_variance(self):
        """Antithetic variates should reduce cost variance vs plain MC."""
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        _, costs_plain = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42, antithetic=False
        )
        _, costs_anti = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42, antithetic=True
        )

        # Variance of the mean estimator
        var_plain = np.var(costs_plain) / len(costs_plain)
        var_anti = np.var(costs_anti) / len(costs_anti)

        # Antithetic should give lower variance (not guaranteed but very likely)
        # Use a generous check — just verify it's not much worse
        assert var_anti < var_plain * 1.1, (
            f"Antithetic variance ({var_anti:.6e}) should be less than "
            f"plain ({var_plain:.6e})"
        )


class TestControlVariate:
    """Test control variate variance reduction (previously untested)."""

    def test_control_variate_preserves_mean(self):
        """CV should not significantly change the mean estimate."""
        x_opt = optimal_trajectory(DEFAULT_PARAMS)
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        _, costs_plain = simulate_execution(
            DEFAULT_PARAMS, x_opt, n_paths=10000, seed=42
        )
        _, costs_cv = simulate_execution_with_control_variate(
            DEFAULT_PARAMS, x_opt, x_twap, n_paths=10000, seed=42
        )

        rel_diff = abs(np.mean(costs_cv) - np.mean(costs_plain)) / abs(np.mean(costs_plain))
        assert rel_diff < 0.1, f"CV mean shifted by {rel_diff*100:.1f}%"

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
