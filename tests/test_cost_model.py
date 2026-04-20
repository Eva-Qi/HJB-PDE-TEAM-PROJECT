"""Tests for cost model functions and cost analysis utilities."""

from dataclasses import replace

import numpy as np
import pytest

from shared.params import ACParams, DEFAULT_PARAMS
from shared.cost_model import (
    execution_cost,
    execution_fees,
    permanent_impact,
    temporary_impact,
)
from montecarlo.strategies import optimal_trajectory, twap_trajectory
from montecarlo.cost_analysis import compute_metrics, compare_strategies


class TestImpactFunctions:
    """Direct tests for impact functions (previously untested)."""

    def test_permanent_impact_linear(self):
        v = np.array([100, 200, 300])
        gamma = 1e-4
        result = permanent_impact(v, gamma)
        assert np.allclose(result, gamma * v)

    # DELETED test_permanent_impact_scalar per 2026-04-19 audit — degenerate
    # subset of test_permanent_impact_linear.

    def test_temporary_impact_linear(self):
        """alpha=1: h(v) = eta * v."""
        v = np.array([100, -50, 0])
        eta = 0.01
        result = temporary_impact(v, eta, alpha=1.0)
        assert np.allclose(result, eta * v)

    def test_temporary_impact_nonlinear(self):
        """alpha=0.5: h(v) = eta * |v|^0.5 * sign(v)."""
        eta = 0.01
        assert temporary_impact(100.0, eta, alpha=0.5) == pytest.approx(eta * 10.0)
        assert temporary_impact(0.0, eta, alpha=0.5) == pytest.approx(0.0)

    def test_temporary_impact_negative_v(self):
        """Negative v (buying) should give negative impact."""
        eta = 0.01
        result = temporary_impact(-100.0, eta, alpha=0.5)
        assert result < 0


class TestCostAnalysis:
    """Tests for compute_metrics and compare_strategies (previously untested)."""

    # DELETED test_constant_costs per 2026-04-19 audit — tests numpy.percentile
    # on a constant array (arithmetic identity, not code invariant).

    def test_cvar_geq_var(self):
        """CVaR >= VaR always (Expected Shortfall >= Value-at-Risk)."""
        rng = np.random.default_rng(42)
        costs = rng.normal(1000, 200, 10000)
        m = compute_metrics(costs)
        assert m.cvar_95 >= m.var_95

    def test_known_distribution(self):
        """For uniform [0, 1000]: mean≈500, VaR95≈950."""
        rng = np.random.default_rng(42)
        costs = rng.uniform(0, 1000, 100000)
        m = compute_metrics(costs)
        assert abs(m.mean - 500) < 10
        assert abs(m.var_95 - 950) < 10

    def test_compare_strategies(self):
        rng = np.random.default_rng(42)
        strategy_costs = {
            "TWAP": rng.normal(1000, 100, 1000),
            "Optimal": rng.normal(900, 150, 1000),
        }
        result = compare_strategies(strategy_costs)
        assert "TWAP" in result
        assert "Optimal" in result
        assert result["TWAP"].n_paths == 1000


class TestExchangeFees:
    """Fee term behavior. Binance taker = 7.5 bps per notional."""

    def test_zero_fee_is_no_op(self):
        """fee_bps=0 should give identical cost to before fees were added."""
        p = replace(DEFAULT_PARAMS, fee_bps=0.0)
        x = twap_trajectory(p)
        assert execution_fees(x, p) == 0.0

    def test_fee_constant_across_strategies_for_liquidation(self):
        """For any full liquidation (x[0]=X0 → x[N]=0) with monotone
        trajectory, sum(|n_k|) == X0, so fees are identical across
        TWAP / Optimal / any other valid trajectory."""
        p = replace(DEFAULT_PARAMS, fee_bps=7.5)
        x_twap = twap_trajectory(p)
        x_opt = optimal_trajectory(p)
        fee_twap = execution_fees(x_twap, p)
        fee_opt = execution_fees(x_opt, p)
        # Both should equal fee_bps/1e4 * S0 * X0
        expected = 7.5 / 1e4 * p.S0 * p.X0
        assert fee_twap == pytest.approx(expected, rel=1e-10)
        assert fee_opt == pytest.approx(expected, rel=1e-10)

    # ═══════════════════════════════════════════════════════════════
    # REWRITE: test_binance_fee_materially_erodes_reported_savings
    #   → test_execution_fees_matches_closed_form_identity
    # ═══════════════════════════════════════════════════════════════
    # GLM verdict: "Hardcodes specific parameters and asserts a narrative
    # ('fee is 20% to 200% of savings'). Tests the parameter choices,
    # not the code logic. Inherently fragile and will break if market
    # regime changes."
    #
    # Council:
    # - Contrarian: bounds [0.2x, 2.0x] are regime-dependent. Re-calibrate
    #   with Feb data or different X0 and the test breaks even though no
    #   code bug exists.
    # - First Principles: the REAL code invariant is the mathematical
    #   identity fee = (fee_bps/1e4) * S0 * sum(|n_k|). This is
    #   regime-independent and tests the actual formula.
    # - Outsider: the "narrative" value belongs in a notebook/report,
    #   not a unit test. Tests fail = code broke, not markets moved.
    def test_execution_fees_matches_closed_form_identity(self):
        """STRUCTURAL: execution_fees returns the closed-form identity
            fee = (fee_bps / 1e4) * S0 * sum_k |n_k|
        for any valid trajectory. Regime-independent — unlike the
        previous narrative-bound test which tested parameter choices
        rather than code logic.

        Bugs caught that the narrative bounds couldn't:
            • Wrong basis-point divisor (e.g. 1e2 instead of 1e4)
            • Using last price instead of S0
            • Summing n_k instead of |n_k| (would zero out on round-trip)
            • Forgetting the fee_bps multiplier entirely
        """
        p = replace(DEFAULT_PARAMS, fee_bps=12.3)  # arbitrary non-zero

        for traj_fn, name in [(twap_trajectory, "TWAP"),
                               (optimal_trajectory, "Optimal")]:
            x = traj_fn(p)
            n_k = x[:-1] - x[1:]
            expected = (p.fee_bps / 1e4) * p.S0 * float(np.sum(np.abs(n_k)))
            actual = execution_fees(x, p)
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-8), (
                f"{name}: fee identity broken. "
                f"Expected (fee_bps/1e4)·S0·Σ|n_k| = {expected:.6f}, "
                f"got {actual:.6f} (diff {abs(actual-expected):.2e})"
            )

        # Also verify linearity in fee_bps: doubling fee_bps doubles fee
        x = twap_trajectory(p)
        fee_1 = execution_fees(x, replace(p, fee_bps=7.5))
        fee_2 = execution_fees(x, replace(p, fee_bps=15.0))
        assert fee_2 == pytest.approx(2.0 * fee_1, rel=1e-12), (
            f"Fee should be linear in fee_bps: 2×7.5 should give 2×fee, "
            f"got fee(7.5)={fee_1:.2f}, fee(15.0)={fee_2:.2f}"
        )

        # Also verify linearity in S0
        fee_small = execution_fees(x, replace(p, S0=100.0))
        fee_large = execution_fees(x, replace(p, S0=200.0))
        assert fee_large == pytest.approx(2.0 * fee_small, rel=1e-12), (
            f"Fee should be linear in S0: doubling S0 should double fee"
        )

    def test_optimality_preserved_under_fees(self):
        """Fees don't change optimization direction: Optimal should
        still beat TWAP on objective after fees (fee is a constant
        added equally to both)."""
        p = replace(DEFAULT_PARAMS, fee_bps=7.5)
        x_twap = twap_trajectory(p)
        x_opt = optimal_trajectory(p)
        from shared.cost_model import objective
        assert objective(x_opt, p) < objective(x_twap, p)
