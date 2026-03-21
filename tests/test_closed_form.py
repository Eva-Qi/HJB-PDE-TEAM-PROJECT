"""Tests for Almgren-Chriss closed-form solution and cost model.

These tests serve as acceptance criteria for P2 (PDE) and P3 (MC):
    - Their implementations must agree with closed-form within tolerance.
    - Optimal trajectory must beat TWAP on expected cost.
"""

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
    """Test strategy trajectory generators."""

    def test_twap_boundary(self):
        x = twap_trajectory(DEFAULT_PARAMS)
        assert len(x) == DEFAULT_PARAMS.N + 1
        assert abs(x[0] - DEFAULT_PARAMS.X0) < 1e-6
        assert abs(x[-1]) < 1e-6

    def test_twap_uniform(self):
        """TWAP should trade equal amounts each step."""
        x = twap_trajectory(DEFAULT_PARAMS)
        n_k = x[:-1] - x[1:]
        expected_per_step = DEFAULT_PARAMS.X0 / DEFAULT_PARAMS.N
        assert np.allclose(n_k, expected_per_step, rtol=1e-10)

    def test_vwap_boundary(self):
        x = vwap_trajectory(DEFAULT_PARAMS)
        assert len(x) == DEFAULT_PARAMS.N + 1
        assert abs(x[0] - DEFAULT_PARAMS.X0) < 1e-6
        assert abs(x[-1]) < 1e-2  # May not be exactly 0 due to rounding

    def test_vwap_sums_to_x0(self):
        """Total shares traded should equal X0."""
        x = vwap_trajectory(DEFAULT_PARAMS)
        total_traded = x[0] - x[-1]
        assert abs(total_traded - DEFAULT_PARAMS.X0) / DEFAULT_PARAMS.X0 < 1e-6


class TestCostModel:
    """Test cost model functions."""

    def test_zero_inventory_zero_cost(self):
        """No inventory → no cost."""
        x = np.zeros(DEFAULT_PARAMS.N + 1)
        assert execution_cost(x, DEFAULT_PARAMS) == 0.0
        assert execution_risk(x, DEFAULT_PARAMS) == 0.0

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
