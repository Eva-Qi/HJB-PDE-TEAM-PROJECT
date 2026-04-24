"""Tests for Almgren-Chriss closed-form solution and cost model.

REWRITE HISTORY: 2026-04-18 — 5 theater tests in TestStrategies and
TestCostModel replaced after code-council audit. Each rewrite's
docstring names the economic invariant it tests. Original theater
tests (now removed):
    test_twap_boundary, test_twap_uniform, test_vwap_boundary,
    test_vwap_sums_to_x0, test_zero_inventory_zero_cost

These tests serve as acceptance criteria for P2 (PDE) and P3 (MC):
    - Their implementations must agree with closed-form within tolerance.
    - Optimal trajectory must beat TWAP on the objective (cost + λ·risk).
"""

from dataclasses import replace

import numpy as np
import pytest

from shared.params import ACParams, DEFAULT_PARAMS, almgren_chriss_closed_form
from shared.cost_model import execution_cost, execution_risk, objective
from montecarlo.strategies import twap_trajectory, vwap_trajectory, optimal_trajectory


class TestClosedForm:
    """Verify Almgren-Chriss closed-form solution properties."""

    def test_boundary_conditions(self):
        """x(0) = X0 and x(T) ≈ 0."""
        t, x, _ = almgren_chriss_closed_form(DEFAULT_PARAMS)
        assert abs(x[0] - DEFAULT_PARAMS.X0) < 1e-6, f"x(0) should be X0, got {x[0]}"
        assert abs(x[-1]) < 1e-3, f"x(T) should be ~0, got {x[-1]}"

    def test_monotonically_decreasing(self):
        """Inventory should decrease over time (we're liquidating)."""
        _, x, _ = almgren_chriss_closed_form(DEFAULT_PARAMS)
        diffs = np.diff(x)
        assert np.all(diffs <= 0), "Trajectory should be monotonically decreasing"

    def test_positive_cost(self):
        """Expected execution cost must be positive."""
        _, _, cost = almgren_chriss_closed_form(DEFAULT_PARAMS)
        assert cost > 0, f"Expected cost should be positive, got {cost}"

    def test_cost_consistency(self):
        """closed-form cost ≈ cost_model.execution_cost on the same trajectory."""
        _, x, cf_cost = almgren_chriss_closed_form(DEFAULT_PARAMS)
        model_cost = execution_cost(x, DEFAULT_PARAMS)
        rel_error = abs(cf_cost - model_cost) / cf_cost
        assert rel_error < 0.01, (
            f"Closed-form cost ({cf_cost:.2f}) and model cost ({model_cost:.2f}) "
            f"differ by {rel_error*100:.2f}%"
        )

    def test_time_grid(self):
        """Time grid should span [0, T] with N+1 points."""
        t, _, _ = almgren_chriss_closed_form(DEFAULT_PARAMS)
        assert len(t) == DEFAULT_PARAMS.N + 1
        assert abs(t[0]) < 1e-10
        assert abs(t[-1] - DEFAULT_PARAMS.T) < 1e-10

    def test_requires_linear_impact(self):
        """Should raise for nonlinear impact (alpha != 1)."""
        nonlinear = ACParams(
            S0=50, sigma=0.3, mu=0, X0=1e6, T=5/252, N=50,
            gamma=2.5e-7, eta=2.5e-6, alpha=0.8, lam=1e-6,
        )
        with pytest.raises(ValueError, match="alpha"):
            almgren_chriss_closed_form(nonlinear)


class TestOptimalVsTWAP:
    """Optimal trajectory must dominate TWAP on the objective function.

    Key economic insight:
        - TWAP minimizes execution cost (equal trades minimize sum(n_k^2))
        - Optimal INCREASES cost (front-loads → higher impact) to DECREASE risk
        - The risk reduction more than compensates → lower objective
    """

    def test_optimal_objective_less_than_twap(self):
        """Objective(optimal) < Objective(TWAP) — this is the optimality condition."""
        x_opt = optimal_trajectory(DEFAULT_PARAMS)
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        obj_opt = objective(x_opt, DEFAULT_PARAMS)
        obj_twap = objective(x_twap, DEFAULT_PARAMS)

        assert obj_opt < obj_twap, (
            f"Optimal objective ({obj_opt:.2f}) should be less than "
            f"TWAP objective ({obj_twap:.2f})"
        )

    def test_optimal_cost_greater_than_twap(self):
        """E[cost](optimal) >= E[cost](TWAP).

        For linear impact, TWAP minimizes execution cost because equal
        trades minimize sum(n_k^2). The optimal trajectory deliberately
        takes on more impact cost to reduce inventory risk.
        """
        x_opt = optimal_trajectory(DEFAULT_PARAMS)
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        cost_opt = execution_cost(x_opt, DEFAULT_PARAMS)
        cost_twap = execution_cost(x_twap, DEFAULT_PARAMS)

        assert cost_opt >= cost_twap, (
            f"Optimal cost ({cost_opt:.2f}) should be >= "
            f"TWAP cost ({cost_twap:.2f}) — front-loading increases impact"
        )

    def test_optimal_risk_less_than_twap(self):
        """Risk(optimal) < Risk(TWAP).

        Front-loaded liquidation reduces inventory exposure over time,
        lowering the total variance of execution cost.
        """
        x_opt = optimal_trajectory(DEFAULT_PARAMS)
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        risk_opt = execution_risk(x_opt, DEFAULT_PARAMS)
        risk_twap = execution_risk(x_twap, DEFAULT_PARAMS)

        assert risk_opt < risk_twap, (
            f"Optimal risk ({risk_opt:.2f}) should be less than "
            f"TWAP risk ({risk_twap:.2f}) because optimal front-loads"
        )


class TestStrategies:
    """Test strategy trajectory generators.

    Every rewrite below has a council-rationale comment explaining why
    the previous theater version (literal boundary / uniform-division
    / sum-tautology checks) was replaced.
    """

    # ═══════════════════════════════════════════════════════════════
    # REWRITE: test_twap_boundary + test_twap_uniform
    # → test_twap_minimizes_impact_cost_for_linear_impact
    # ═══════════════════════════════════════════════════════════════
    # Council:
    # - Contrarian: boundary check tests linspace[0]=1 and linspace[-1]=0.
    #   Uniform check tests that linspace gives equal spacing. Both are
    #   testing numpy's linspace, not our TWAP logic.
    # - First Principles: the ECONOMIC property of TWAP (for linear
    #   impact) is that it minimizes expected execution cost. sum(n_k^2)
    #   with sum(n_k)=X0 is minimized at n_k = X0/N (uniform).
    # - This test will ALSO fail if linspace breaks — one test covers
    #   both shapes and the economic invariant.
    def test_twap_minimizes_impact_cost_for_linear_impact(self):
        """For linear impact (alpha=1) with no risk term, TWAP minimizes
        execution cost among all valid liquidation trajectories.

        Invariant: among trajectories with x[0]=X0, x[-1]=0, and
        monotone decrease, the one with lowest execution cost is
        uniform (TWAP). Any front-loaded or back-loaded alternative
        pays more impact cost.
        """
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        # Build a front-loaded alternative: trade 80% in first half
        N = DEFAULT_PARAMS.N
        X0 = DEFAULT_PARAMS.X0
        half = N // 2
        x_front = np.concatenate([
            np.linspace(X0, X0 * 0.2, half + 1),
            np.linspace(X0 * 0.2, 0, N - half + 1)[1:],
        ])
        # Build a back-loaded alternative: trade 20% in first half
        x_back = np.concatenate([
            np.linspace(X0, X0 * 0.8, half + 1),
            np.linspace(X0 * 0.8, 0, N - half + 1)[1:],
        ])

        c_twap = execution_cost(x_twap, DEFAULT_PARAMS)
        c_front = execution_cost(x_front, DEFAULT_PARAMS)
        c_back = execution_cost(x_back, DEFAULT_PARAMS)

        assert c_twap < c_front, (
            f"TWAP cost ({c_twap:.2f}) should be < front-loaded "
            f"({c_front:.2f}) for linear impact"
        )
        assert c_twap < c_back, (
            f"TWAP cost ({c_twap:.2f}) should be < back-loaded "
            f"({c_back:.2f}) for linear impact"
        )

        # Also verifies the basic TWAP shape invariants the old tests
        # asserted (in case linspace ever breaks):
        assert len(x_twap) == N + 1
        assert abs(x_twap[0] - X0) < 1e-6
        assert abs(x_twap[-1]) < 1e-6

    # ═══════════════════════════════════════════════════════════════
    # REWRITE: test_vwap_boundary
    # → test_vwap_trades_more_in_high_volume_steps
    # ═══════════════════════════════════════════════════════════════
    # Council:
    # - Outsider: VWAP is defined as "trades proportional to volume".
    #   Not testing that defeats the test's only purpose.
    # - First Principles: the ONLY thing that differentiates VWAP from
    #   TWAP is the volume profile. Test that directly.
    def test_vwap_trades_more_in_high_volume_steps(self):
        """VWAP trade sizes should be positively correlated with the
        supplied volume profile.

        Invariant: if weights[k] is high, trade n_k should be large.
        Correlation is a robust way to test this across any profile
        shape.
        """
        # Synthetic profile: U-shape, highest at ends, low in middle
        N = DEFAULT_PARAMS.N
        custom_profile = np.linspace(1.0, 0.2, N // 2)
        custom_profile = np.concatenate([
            custom_profile, custom_profile[::-1][:N - len(custom_profile)],
        ])
        # Pad if odd
        if len(custom_profile) < N:
            custom_profile = np.concatenate([custom_profile,
                                             np.full(N - len(custom_profile),
                                                     custom_profile[-1])])
        custom_profile = custom_profile[:N]

        p = replace(DEFAULT_PARAMS, volume_profile=custom_profile)
        x = vwap_trajectory(p)
        n_k = x[:-1] - x[1:]
        corr = float(np.corrcoef(n_k, custom_profile)[0, 1])
        assert corr > 0.95, (
            f"VWAP trade sizes should correlate strongly with volume "
            f"profile, got corr={corr:.3f} — volume_profile may be "
            f"ignored or mis-applied."
        )

    # ═══════════════════════════════════════════════════════════════
    # REWRITE: test_vwap_sums_to_x0
    # → test_vwap_reduces_to_twap_under_flat_profile
    # ═══════════════════════════════════════════════════════════════
    # Council:
    # - Contrarian: x[0]-x[-1]=X0 is a tautology given x[0]=X0 and
    #   x[-1]=0 — both trivially true by construction. Zero
    #   information value.
    # - Expansionist: the interesting generalization property is that
    #   VWAP is a STRICT generalization of TWAP. If the profile is
    #   flat, the two must coincide exactly.
    def test_vwap_reduces_to_twap_under_flat_profile(self):
        """VWAP with a flat (uniform) volume profile must equal TWAP.

        Invariant: TWAP is the special case of VWAP where all weights
        are equal. If this fails, VWAP has a bug in the normalization
        or cumsum logic.
        """
        N = DEFAULT_PARAMS.N
        flat_profile = np.full(N, 1.0)  # all equal
        p_flat = replace(DEFAULT_PARAMS, volume_profile=flat_profile)
        x_vwap = vwap_trajectory(p_flat)
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        # rtol=1e-12 would fail on ~5e-10 floating-point cumsum drift,
        # which is well within double-precision limits — use 1e-8.
        np.testing.assert_allclose(
            x_vwap, x_twap, rtol=1e-8, atol=1e-4,
            err_msg="VWAP with flat profile should equal TWAP to "
                    "floating-point tolerance",
        )

    def test_vwap_schedule_changes_with_T(self):
        """VWAP must respect params.T. Schedules for T=1h vs T=24h differ.

        Before the T-aware fix, VWAP hardcoded hours_per_step = 24/N, so
        schedules were identical regardless of T. After the fix, a 1-hour
        execution samples a single hour bucket (degenerating to TWAP),
        while a 24-hour execution samples the full diurnal cycle.
        """
        p_1h = replace(DEFAULT_PARAMS, T=1.0 / (365.25 * 24), N=50)
        p_1d = replace(DEFAULT_PARAMS, T=1.0 / 365.25, N=50)
        x_1h = vwap_trajectory(p_1h)
        x_1d = vwap_trajectory(p_1d)

        # 1-hour execution → all steps in same hour bucket → shape ≈ TWAP
        x_twap = twap_trajectory(p_1h)
        # Same tolerance policy as test_vwap_reduces_to_twap_under_flat_profile
        np.testing.assert_allclose(
            x_1h, x_twap, rtol=1e-8, atol=1e-4,
            err_msg=("VWAP at T=1h should degenerate to TWAP (all steps within "
                     "a single hour bucket)"),
        )

        # 24-hour execution → VWAP sees full diurnal cycle → schedule ≠ TWAP
        assert not np.allclose(x_1d, x_twap, rtol=1e-3), (
            "VWAP at T=24h should differ from TWAP (volume profile has "
            "within-day variation)"
        )

        # And the two VWAP schedules themselves must differ
        assert not np.allclose(x_1h, x_1d, rtol=1e-6), (
            "VWAP(T=1h) and VWAP(T=24h) must differ — T-invariance is the bug"
        )


class TestCostModel:
    """Test cost model functions."""

    # ═══════════════════════════════════════════════════════════════
    # REWRITE: test_zero_inventory_zero_cost
    # → test_execution_cost_scales_quadratically_with_X0
    # ═══════════════════════════════════════════════════════════════
    # Council:
    # - Contrarian: 0*anything=0. Testing that is testing arithmetic.
    # - First Principles: the STRUCTURE of Almgren-Chriss cost is
    #   quadratic in X0 for linear impact — doubling X0 → 4x cost.
    #   That's the load-bearing property of the cost model.
    # - Outsider: "cost scales as X0^2" tells a reviewer the cost
    #   model has the right shape. "cost(0)=0" tells nothing.
    def test_execution_cost_scales_quadratically_with_X0(self):
        """For linear impact (alpha=1), execution cost scales as X0^2.

        Math: perm_cost ~ gamma * X0^2, temp_cost ~ eta/dt * X0^2
        (both come from sum(n_k^2) ~ X0^2/N for uniform trajectory).
        So total cost scales as X0^2, i.e. log-log slope = 2.0.

        Fit the exponent across THREE multipliers (0.5x, 1x, 2x) using
        log-log linear regression instead of a single ratio check.
        This catches X0^1.9 power laws that a single-ratio test misses.

        Invariant: if slope ever deviates from 2.0 by >1%, either the
        impact terms are miswired or the self-impact convention is broken.
        """
        X0_multipliers = [0.5, 1.0, 2.0]
        costs = []
        for mult in X0_multipliers:
            p = replace(DEFAULT_PARAMS, X0=mult * DEFAULT_PARAMS.X0)
            costs.append(execution_cost(twap_trajectory(p), p))
        slope = np.polyfit(np.log(X0_multipliers), np.log(costs), 1)[0]
        assert 1.98 < slope < 2.02, (
            f"Log-log slope of cost vs X0 should be 2.0 (strict quadratic); "
            f"got {slope:.4f}. Structural bug in cost model: impact terms "
            f"may not scale correctly with position size."
        )

    def test_risk_positive_for_nonzero_inventory(self):
        x = twap_trajectory(DEFAULT_PARAMS)
        assert execution_risk(x, DEFAULT_PARAMS) > 0

    def test_higher_lambda_more_aggressive(self):
        """Higher risk aversion → trades faster → more impact cost but less risk."""
        low_lam = ACParams(
            S0=50, sigma=0.3, mu=0, X0=1e6, T=5/252, N=50,
            gamma=2.5e-7, eta=2.5e-6, alpha=1.0, lam=1e-7,
        )
        high_lam = ACParams(
            S0=50, sigma=0.3, mu=0, X0=1e6, T=5/252, N=50,
            gamma=2.5e-7, eta=2.5e-6, alpha=1.0, lam=1e-5,
        )

        x_low = optimal_trajectory(low_lam)
        x_high = optimal_trajectory(high_lam)

        # Higher lambda → less risk (trades faster)
        risk_low = execution_risk(x_low, low_lam)
        risk_high = execution_risk(x_high, high_lam)
        assert risk_high < risk_low, "Higher lambda should produce lower risk"
