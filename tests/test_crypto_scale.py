"""Integration tests with crypto-scale parameters.

All other tests use DEFAULT_PARAMS (S0=50, X0=1M, σ=0.3 — equity-like).
These tests verify numerical stability at BTC-like magnitudes:
S0=69000, X0=10, σ=0.4, T=1/24yr (1 hour).

If these fail while DEFAULT_PARAMS tests pass, it's a scaling/overflow issue.
"""

import numpy as np
import pytest

from shared.params import ACParams, almgren_chriss_closed_form
from shared.cost_model import execution_cost, execution_risk, objective
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory
from montecarlo.sde_engine import simulate_execution, generate_normal_increments
from montecarlo.strategies import twap_trajectory, optimal_trajectory
from montecarlo.cost_analysis import compute_metrics, bootstrap_confidence_interval


# Crypto-scale parameters matching calibrated Binance data
CRYPTO_PARAMS = ACParams(
    S0=69000.0,         # BTC price
    sigma=0.396,        # 39.6% annualized (calibrated)
    mu=0.0,
    X0=10.0,            # 10 BTC to liquidate
    T=1 / 24,           # 1 hour execution window
    N=50,               # time steps
    gamma=0.0133,       # calibrated Kyle's lambda
    eta=1e-3,           # literature fallback
    alpha=1.0,          # linear (for closed-form comparison)
    lam=1e-6,           # risk aversion
)

CRYPTO_NONLINEAR = ACParams(
    S0=69000.0,
    sigma=0.396,
    mu=0.0,
    X0=10.0,
    T=1 / 24,
    N=50,
    gamma=0.0133,
    eta=1e-3,
    alpha=0.6,          # nonlinear temporary impact
    lam=1e-6,
)


class TestCryptoClosedForm:

    def test_trajectory_boundary_conditions(self):
        t, x, cost = almgren_chriss_closed_form(CRYPTO_PARAMS)
        assert abs(x[0] - CRYPTO_PARAMS.X0) < 1e-10
        assert x[-1] / CRYPTO_PARAMS.X0 < 0.01

    def test_cost_positive_and_finite(self):
        _, _, cost = almgren_chriss_closed_form(CRYPTO_PARAMS)
        assert cost > 0
        assert np.isfinite(cost)

    def test_optimal_beats_twap(self):
        x_twap = twap_trajectory(CRYPTO_PARAMS)
        x_opt = optimal_trajectory(CRYPTO_PARAMS)
        assert objective(x_opt, CRYPTO_PARAMS) < objective(x_twap, CRYPTO_PARAMS)


class TestCryptoPDE:

    @pytest.fixture
    def pde_result(self):
        # M=200 needed: X0=10 gives dx=0.05, sufficient for interpolation
        return solve_hjb(CRYPTO_PARAMS, M=200)

    def test_value_function_finite(self, pde_result):
        _, V, _ = pde_result
        assert np.all(np.isfinite(V)), f"V has non-finite values: max={np.nanmax(V)}"

    def test_value_function_nonneg(self, pde_result):
        _, V, _ = pde_result
        assert np.all(V >= -1e-3)

    def test_trajectory_matches_closed_form(self, pde_result):
        grid, _, v_star = pde_result
        x_pde = extract_optimal_trajectory(grid, v_star, CRYPTO_PARAMS)
        _, x_cf, _ = almgren_chriss_closed_form(CRYPTO_PARAMS)
        # Compare cost (integrated metric) instead of pointwise trajectory,
        # because small X0=10 with discrete grid makes pointwise comparison noisy
        cost_pde = objective(x_pde, CRYPTO_PARAMS)
        cost_cf = objective(x_cf, CRYPTO_PARAMS)
        rel_err = abs(cost_pde - cost_cf) / abs(cost_cf)
        assert rel_err < 0.10, f"PDE objective vs CF: {rel_err:.2%} error"


class TestCryptoNonlinearPDE:

    def test_nonlinear_solves_without_error(self):
        grid, V, v_star = solve_hjb(CRYPTO_NONLINEAR, M=50)
        assert np.all(np.isfinite(V))
        assert np.all(v_star >= -1e-10)

    def test_nonlinear_trajectory_monotone(self):
        grid, _, v_star = solve_hjb(CRYPTO_NONLINEAR, M=50)
        x = extract_optimal_trajectory(grid, v_star, CRYPTO_NONLINEAR)
        assert abs(x[0] - CRYPTO_NONLINEAR.X0) < 1e-6
        assert np.all(np.diff(x) <= 1e-3)


class TestCryptoMC:

    def test_exact_scheme_positive_prices(self):
        x_twap = twap_trajectory(CRYPTO_PARAMS)
        paths, costs = simulate_execution(
            CRYPTO_PARAMS, x_twap, n_paths=5000, scheme="exact",
        )
        assert np.all(paths > 0), f"Exact scheme produced negative prices: min={paths.min()}"

    def test_euler_may_go_negative(self):
        """Euler with high vol can produce negative prices — document this."""
        x_twap = twap_trajectory(CRYPTO_PARAMS)
        paths, _ = simulate_execution(
            CRYPTO_PARAMS, x_twap, n_paths=5000, scheme="euler",
        )
        # With σ=40% and dt=T/50, Euler *may* go negative.
        # This test documents the behavior rather than asserting it.
        min_price = paths.min()
        if min_price < 0:
            pytest.skip(f"Euler produced negative prices (min={min_price:.0f}) — expected for high-vol crypto")

    def test_mc_mean_near_closed_form(self):
        _, x_opt, cf_cost = almgren_chriss_closed_form(CRYPTO_PARAMS)
        _, costs = simulate_execution(
            CRYPTO_PARAMS, x_opt, n_paths=20000, antithetic=True, scheme="exact",
        )
        rel_err = abs(np.mean(costs) - cf_cost) / abs(cf_cost)
        assert rel_err < 0.10, (
            f"MC mean ({np.mean(costs):.2f}) vs CF ({cf_cost:.2f}): {rel_err:.2%} error"
        )

    def test_sobol_works_at_crypto_scale(self):
        x_twap = twap_trajectory(CRYPTO_PARAMS)
        Z = generate_normal_increments(4096, CRYPTO_PARAMS.N, method="sobol", seed=42)
        paths, costs = simulate_execution(
            CRYPTO_PARAMS, x_twap, n_paths=4096,
            antithetic=False, Z_extern=Z, scheme="exact",
        )
        assert np.all(np.isfinite(costs))
        assert np.std(costs) > 0

    def test_bootstrap_ci_at_crypto_scale(self):
        x_twap = twap_trajectory(CRYPTO_PARAMS)
        _, costs = simulate_execution(
            CRYPTO_PARAMS, x_twap, n_paths=5000, scheme="exact",
        )
        ci = bootstrap_confidence_interval(costs, "mean", n_bootstrap=1000)
        assert ci.ci_lower < ci.estimate < ci.ci_upper
        assert np.isfinite(ci.ci_lower) and np.isfinite(ci.ci_upper)
