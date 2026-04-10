"""Tests for Heston stochastic volatility execution simulation.

Covers simulate_heston_execution: output shapes, initial conditions,
cost properties, reproducibility, and limiting behavior vs constant-vol MC.
"""

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS, ACParams
from extensions.heston import HestonParams
from montecarlo.sde_engine import simulate_execution, simulate_heston_execution
from montecarlo.strategies import twap_trajectory


# Reasonable Heston params for sigma ~ 0.3 (v0 = theta = 0.09)
HESTON_PARAMS = HestonParams(
    kappa=2.0,
    theta=0.09,
    xi=0.5,
    rho=-0.3,
    v0=0.09,
)

# Pre-compute TWAP trajectory (reused across tests)
TWAP_X = twap_trajectory(DEFAULT_PARAMS)

N_PATHS = 5000
SEED = 42


class TestHestonOutputShapes:
    """Verify array dimensions returned by simulate_heston_execution."""

    def test_heston_output_shapes(self):
        """S shape is (n_paths, N+1), v shape is (n_paths, N+1), costs shape is (n_paths,)."""
        S, v, costs = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=SEED,
        )
        N = DEFAULT_PARAMS.N
        assert S.shape == (N_PATHS, N + 1)
        assert v.shape == (N_PATHS, N + 1)
        assert costs.shape == (N_PATHS,)


class TestHestonInitialConditions:
    """Verify initial values are set correctly."""

    def test_heston_initial_conditions(self):
        """S[:,0] == S0 and v[:,0] == v0."""
        S, v, _ = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=SEED,
        )
        np.testing.assert_allclose(S[:, 0], DEFAULT_PARAMS.S0)
        np.testing.assert_allclose(v[:, 0], HESTON_PARAMS.v0)

    def test_heston_variance_starts_at_v0(self):
        """v[:,0] matches heston_params.v0 exactly."""
        _, v, _ = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=SEED,
        )
        assert np.all(v[:, 0] == HESTON_PARAMS.v0)


class TestHestonCostProperties:
    """Verify cost array is well-behaved."""

    def test_heston_costs_finite(self):
        """All costs are finite (no NaN or inf)."""
        _, _, costs = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=SEED,
        )
        assert np.all(np.isfinite(costs)), (
            f"Found non-finite costs: NaN={np.sum(np.isnan(costs))}, "
            f"inf={np.sum(np.isinf(costs))}"
        )

    def test_heston_costs_positive_variance(self):
        """Costs have positive variance (not degenerate)."""
        _, _, costs = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=SEED,
        )
        assert np.std(costs) > 0, "Cost distribution should have positive std"


class TestHestonPricePositivity:
    """Check price positivity under Euler-Maruyama (best effort)."""

    def test_heston_price_positive(self):
        """Most paths should have positive prices (Euler may rarely go negative)."""
        S, _, _ = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=SEED,
        )
        frac_positive = np.mean(S > 0)
        assert frac_positive > 0.99, (
            f"Only {frac_positive*100:.1f}% of price entries are positive "
            f"(min={S.min():.2f})"
        )


class TestHestonReproducibility:
    """Verify seed-based determinism."""

    def test_heston_reproducible(self):
        """Same seed gives identical results."""
        S1, v1, c1 = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=123,
        )
        S2, v2, c2 = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X, n_paths=N_PATHS, seed=123,
        )
        np.testing.assert_array_equal(S1, S2)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(c1, c2)


class TestHestonLimitingBehavior:
    """Compare Heston to constant-vol MC in limiting cases."""

    def test_heston_reduces_to_constant_vol(self):
        """When xi=0, kappa large, theta=v0=sigma^2, Heston should behave
        like constant-vol GBM. Mean cost should be within ~15% of the
        constant-vol MC mean cost."""
        sigma = DEFAULT_PARAMS.sigma
        degenerate_heston = HestonParams(
            kappa=100.0,       # very fast mean reversion
            theta=sigma ** 2,  # long-run var = sigma^2
            xi=0.0,            # zero vol-of-vol → deterministic variance
            rho=0.0,
            v0=sigma ** 2,     # start at long-run level
        )

        _, _, costs_heston = simulate_heston_execution(
            DEFAULT_PARAMS, degenerate_heston, TWAP_X,
            n_paths=20000, seed=SEED,
        )
        _, costs_const = simulate_execution(
            DEFAULT_PARAMS, TWAP_X, n_paths=20000, seed=SEED,
            antithetic=True, scheme="euler",
        )

        mean_h = np.mean(costs_heston)
        mean_c = np.mean(costs_const)
        rel_diff = abs(mean_h - mean_c) / abs(mean_c)
        assert rel_diff < 0.15, (
            f"Degenerate Heston mean cost ({mean_h:.2f}) differs from "
            f"constant-vol ({mean_c:.2f}) by {rel_diff*100:.1f}%"
        )

    def test_heston_mean_cost_reasonable(self):
        """Mean Heston cost should be in the same order of magnitude as
        constant-vol cost (between 0.5x and 3x)."""
        _, costs_const = simulate_execution(
            DEFAULT_PARAMS, TWAP_X, n_paths=10000, seed=SEED,
            antithetic=True, scheme="exact",
        )
        _, _, costs_heston = simulate_heston_execution(
            DEFAULT_PARAMS, HESTON_PARAMS, TWAP_X,
            n_paths=10000, seed=SEED,
        )

        mean_c = np.mean(costs_const)
        mean_h = np.mean(costs_heston)

        ratio = mean_h / mean_c
        assert 0.5 < ratio < 3.0, (
            f"Heston/constant-vol cost ratio = {ratio:.2f} "
            f"(Heston={mean_h:.2f}, const={mean_c:.2f}), expected 0.5-3.0"
        )
