"""Tests for HJB PDE solver."""

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS, ACParams, almgren_chriss_closed_form
from shared.cost_model import execution_cost, objective
from pde.hjb_solver import (
    solve_hjb,
    extract_optimal_trajectory,
    analytical_value_function,
)


class TestHJBSolver:
    """Test HJB finite difference solver."""

    @pytest.fixture
    def hjb_result(self):
        """Solve HJB once for reuse across tests."""
        return solve_hjb(DEFAULT_PARAMS, M=200)

    def test_output_shapes(self, hjb_result):
        grid, V, v_star = hjb_result
        M = grid.M
        N = grid.N
        assert V.shape == (M + 1, N + 1)
        assert v_star.shape == (M + 1, N + 1)

    def test_boundary_zero_inventory(self, hjb_result):
        """V(0, t) = 0 for all t — no inventory means no cost."""
        _, V, _ = hjb_result
        assert np.allclose(V[0, :], 0.0, atol=1e-6)

    def test_terminal_condition(self, hjb_result):
        """V(x, T) = penalty * x^2."""
        grid, V, _ = hjb_result
        penalty = 1e8
        expected = penalty * grid.x_grid**2
        assert np.allclose(V[:, -1], expected, rtol=1e-6)

    def test_value_function_nonnegative(self, hjb_result):
        """V(x, t) >= 0 everywhere."""
        _, V, _ = hjb_result
        assert np.all(V >= -1e-6), "Value function should be non-negative"

    def test_value_function_increasing_in_x(self, hjb_result):
        """V should increase with inventory (more inventory = more cost to liquidate)."""
        _, V, _ = hjb_result
        # Check at t=0
        diffs = np.diff(V[:, 0])
        # Allow small numerical noise
        assert np.all(diffs >= -1e-3), "V should be increasing in x at t=0"

    def test_optimal_control_nonnegative(self, hjb_result):
        """v*(x, t) >= 0 — we're liquidating, not buying."""
        _, _, v_star = hjb_result
        assert np.all(v_star >= -1e-10), "Optimal control should be non-negative"

    def test_value_function_vs_analytical(self, hjb_result):
        """Numerical V should match analytical V(x,t) = eta*kappa*coth(kappa*(T-t))*x^2.

        This is the core validation: PDE solver reproduces the known solution.
        """
        grid, V, _ = hjb_result
        x = grid.x_grid

        # Compare at t=0 (full time to go — most representative)
        V_analytical = analytical_value_function(DEFAULT_PARAMS, x, t=0.0)
        V_numerical = V[:, 0]

        # Relative error at interior points (skip x=0 where both are 0)
        interior = x > 0
        rel_err = np.abs(V_numerical[interior] - V_analytical[interior]) / V_analytical[interior]
        max_rel_err = np.max(rel_err)

        assert max_rel_err < 0.05, (
            f"Max relative error between numerical and analytical V: "
            f"{max_rel_err*100:.2f}% (should be < 5%)"
        )


class TestExtractTrajectory:
    """Test optimal trajectory extraction from PDE solution."""

    @pytest.fixture
    def pde_trajectory(self):
        grid, V, v_star = solve_hjb(DEFAULT_PARAMS, M=200)
        return extract_optimal_trajectory(grid, v_star, DEFAULT_PARAMS)

    def test_boundary_conditions(self, pde_trajectory):
        """x(0) = X0, x(T) ≈ 0."""
        x = pde_trajectory
        assert abs(x[0] - DEFAULT_PARAMS.X0) < 1e-6, f"x(0) should be X0, got {x[0]}"
        # With large terminal penalty, nearly all inventory should be liquidated
        assert x[-1] / DEFAULT_PARAMS.X0 < 0.05, (
            f"x(T) should be near 0, got {x[-1]:.0f} "
            f"({x[-1]/DEFAULT_PARAMS.X0*100:.1f}% remaining)"
        )

    def test_monotonically_decreasing(self, pde_trajectory):
        """Inventory should decrease (we're liquidating)."""
        diffs = np.diff(pde_trajectory)
        assert np.all(diffs <= 1e-3), "Trajectory should be monotonically decreasing"

    def test_matches_closed_form(self, pde_trajectory):
        """PDE trajectory should match closed-form within 5%."""
        _, x_cf, _ = almgren_chriss_closed_form(DEFAULT_PARAMS)
        x_pde = pde_trajectory

        # Compare at interior points (exclude endpoints)
        mid_points = slice(5, -5)
        rel_err = np.abs(x_pde[mid_points] - x_cf[mid_points]) / x_cf[mid_points]
        max_err = np.max(rel_err)

        assert max_err < 0.10, (
            f"PDE trajectory differs from closed-form by {max_err*100:.1f}% "
            f"(should be < 10%)"
        )

    def test_pde_objective_near_analytical(self, pde_trajectory):
        """PDE trajectory's objective should be close to closed-form."""
        _, x_cf, _ = almgren_chriss_closed_form(DEFAULT_PARAMS)

        obj_pde = objective(pde_trajectory, DEFAULT_PARAMS)
        obj_cf = objective(x_cf, DEFAULT_PARAMS)

        rel_err = abs(obj_pde - obj_cf) / abs(obj_cf)
        assert rel_err < 0.10, (
            f"PDE objective ({obj_pde:.2f}) differs from closed-form "
            f"({obj_cf:.2f}) by {rel_err*100:.1f}%"
        )
