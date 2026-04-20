"""Tests for new MC features: Sobol QMC, Milstein scheme, Z_extern injection."""

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS, ACParams, almgren_chriss_closed_form
from shared.cost_model import execution_cost
from montecarlo.sde_engine import (
    generate_normal_increments,
    simulate_execution,
)
from montecarlo.strategies import twap_trajectory, optimal_trajectory


# ---------------------------------------------------------------------------
# generate_normal_increments
# ---------------------------------------------------------------------------

class TestGenerateNormalIncrements:
    """Tests for the Z-generation factory.

    DELETED per 2026-04-19 audit (9 tests):
      - test_shape[pseudo/sobol/antithetic] — allocation theater
      - test_mean_near_zero[pseudo/sobol/antithetic] — CLT, not code
      - test_std_near_one[pseudo/sobol/antithetic] — CLT, not code
      - test_sobol_reproducible — tests scipy RNG, not project code
    Surviving: test_antithetic_exact_zero_mean (tight structural),
               test_invalid_method (error path).
    """

    def test_antithetic_exact_zero_mean(self):
        """Antithetic should have exactly zero mean (Z and -Z cancel)."""
        Z = generate_normal_increments(1000, 50, method="antithetic", seed=42)
        assert abs(Z.mean()) < 1e-10

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            generate_normal_increments(100, 50, method="invalid")


# ---------------------------------------------------------------------------
# Z_extern injection
# ---------------------------------------------------------------------------

class TestZExtern:
    """Tests for external Z injection into simulate_execution."""

    def test_z_extern_matches_internal(self):
        """Passing the same Z externally should give identical results."""
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        N = DEFAULT_PARAMS.N

        # Generate Z manually with same seed/method as internal default
        rng = np.random.default_rng(42)
        n_half = 5000
        Z_half = rng.standard_normal((n_half, N))
        Z = np.vstack([Z_half, -Z_half])

        # Internal generation (antithetic=True, seed=42)
        _, costs_internal = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42,
            antithetic=True, scheme="exact",
        )

        # External injection (antithetic=False because Z already has pairs)
        _, costs_external = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42,
            antithetic=False, Z_extern=Z, scheme="exact",
        )

        assert np.allclose(costs_internal, costs_external, rtol=1e-10)

    # DELETED test_sobol_z_into_execution: tests Sobol statistical
    # property (QMC accuracy), not code logic. Sobol running without
    # crash is covered by test_crypto_scale::test_sobol_works_at_crypto_scale.


# ---------------------------------------------------------------------------
# Milstein and Euler schemes
# ---------------------------------------------------------------------------

class TestSchemes:
    """Tests for Euler-Maruyama and Milstein discretization schemes."""

    def test_mean_cost_within_tolerance_exact(self):
        """Exact scheme should give mean cost within 5% of deterministic.

        DELETED [euler] and [milstein] parametrizations — duplicate of
        test_mc_engine::test_twap_mean_cost_matches_deterministic. Only
        keeping [exact] here because it's the reference scheme.
        """
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        true_cost = execution_cost(x_twap, DEFAULT_PARAMS)

        _, costs = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=20000, seed=42,
            antithetic=True, scheme="exact",
        )

        rel_err = abs(np.mean(costs) - true_cost) / true_cost
        assert rel_err < 0.05, (
            f"exact mean cost error {rel_err:.4%} exceeds 5%"
        )

    def test_milstein_closer_to_exact_than_euler(self):
        """Milstein should match exact more closely than Euler at coarse dt."""
        # Use fewer time steps to amplify discretization differences
        coarse_params = ACParams(
            S0=DEFAULT_PARAMS.S0, sigma=DEFAULT_PARAMS.sigma,
            mu=DEFAULT_PARAMS.mu, X0=DEFAULT_PARAMS.X0,
            T=DEFAULT_PARAMS.T, N=10,  # coarse: only 10 steps
            gamma=DEFAULT_PARAMS.gamma, eta=DEFAULT_PARAMS.eta,
            alpha=DEFAULT_PARAMS.alpha, lam=DEFAULT_PARAMS.lam,
        )
        x_twap = coarse_params.X0 * np.linspace(1, 0, 11)

        # Same Z for all three
        Z = generate_normal_increments(10000, 10, method="pseudo", seed=42)

        _, costs_exact = simulate_execution(
            coarse_params, x_twap, antithetic=False, Z_extern=Z, scheme="exact",
        )
        _, costs_euler = simulate_execution(
            coarse_params, x_twap, antithetic=False, Z_extern=Z, scheme="euler",
        )
        _, costs_milstein = simulate_execution(
            coarse_params, x_twap, antithetic=False, Z_extern=Z, scheme="milstein",
        )

        # Milstein mean should be closer to exact mean than Euler mean is
        err_euler = abs(np.mean(costs_euler) - np.mean(costs_exact))
        err_milstein = abs(np.mean(costs_milstein) - np.mean(costs_exact))

        assert err_milstein < err_euler, (
            f"Milstein error ({err_milstein:.2f}) should be < "
            f"Euler error ({err_euler:.2f})"
        )

    def test_invalid_scheme(self):
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        with pytest.raises(ValueError, match="Unknown scheme"):
            simulate_execution(
                DEFAULT_PARAMS, x_twap, n_paths=100, scheme="runge_kutta",
            )

    def test_optimal_beats_twap_exact(self):
        """Optimal trajectory has lower objective than TWAP (exact scheme).

        DELETED [euler] and [milstein] parametrizations — once [exact]
        passes, the other schemes passing is a consequence of param
        consistency, not a distinct bug class.
        """
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        x_opt = optimal_trajectory(DEFAULT_PARAMS)

        _, costs_twap = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42, scheme="exact",
        )
        _, costs_opt = simulate_execution(
            DEFAULT_PARAMS, x_opt, n_paths=10000, seed=42, scheme="exact",
        )

        obj_twap = np.mean(costs_twap) + DEFAULT_PARAMS.lam * np.var(costs_twap)
        obj_opt = np.mean(costs_opt) + DEFAULT_PARAMS.lam * np.var(costs_opt)

        assert obj_opt < obj_twap, (
            f"exact: optimal obj ({obj_opt:.2f}) should < TWAP ({obj_twap:.2f})"
        )
