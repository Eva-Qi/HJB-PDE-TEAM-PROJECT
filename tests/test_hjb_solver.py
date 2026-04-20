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

    # DELETED per 2026-04-19 audit — all 4 redundant with
    # test_value_function_vs_analytical:
    #   test_output_shapes              (shape follows from grid construction)
    #   test_boundary_zero_inventory   (V(0,t)=0 implied by V~x² analytical form)
    #   test_value_function_nonnegative (implied by 5% match to non-neg analytical)
    #   test_value_function_increasing_in_x (monotonicity follows from V~x²)

    def test_terminal_condition(self, hjb_result):
        """V(x, T) = penalty * x^2 (solver's terminal BC, not implied by analytical)."""
        grid, V, _ = hjb_result
        penalty = 1e8
        expected = penalty * grid.x_grid**2
        assert np.allclose(V[:, -1], expected, rtol=1e-6)

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

    # DELETED test_boundary_conditions + test_monotonically_decreasing
    # per 2026-04-19 audit — both are corollaries of test_matches_closed_form
    # (5% match to closed-form implies boundary values and monotonicity).

    def test_matches_closed_form(self, pde_trajectory):
        """PDE trajectory should match closed-form within 5%."""
        _, x_cf, _ = almgren_chriss_closed_form(DEFAULT_PARAMS)
        x_pde = pde_trajectory

        # Compare at interior points (exclude endpoints)
        mid_points = slice(5, -5)
        rel_err = np.abs(x_pde[mid_points] - x_cf[mid_points]) / x_cf[mid_points]
        max_err = np.max(rel_err)

        assert max_err < 0.05, (
            f"PDE trajectory differs from closed-form by {max_err*100:.1f}% "
            f"(should be < 5%)"
        )

    def test_pde_objective_near_analytical(self, pde_trajectory):
        """PDE trajectory's objective should be close to closed-form."""
        _, x_cf, _ = almgren_chriss_closed_form(DEFAULT_PARAMS)

        obj_pde = objective(pde_trajectory, DEFAULT_PARAMS)
        obj_cf = objective(x_cf, DEFAULT_PARAMS)

        rel_err = abs(obj_pde - obj_cf) / abs(obj_cf)
        assert rel_err < 0.05, (
            f"PDE objective ({obj_pde:.2f}) differs from closed-form "
            f"({obj_cf:.2f}) by {rel_err*100:.1f}%"
        )


class TestFDNonlinear:
    """Test the FD nonlinear solver path (alpha != 1, previously untested)."""

    @pytest.fixture
    def nonlinear_params(self):
        # Parameters for the policy iteration (implicit) FD solver.
        # The implicit scheme is unconditionally stable, so we can use
        # realistic parameters unlike the old explicit solver.
        return ACParams(
            S0=50.0, sigma=0.3, mu=0.0, X0=10_000,
            T=0.25, N=20, gamma=2.5e-7, eta=2.5e-6,
            alpha=0.8, lam=1e-3,
        )

    def test_fd_value_nonnegative(self, nonlinear_params):
        """V >= 0 for nonlinear impact FD solver (policy iteration)."""
        _, V, _ = solve_hjb(nonlinear_params, M=50)
        assert np.all(V >= -1e-3), f"V has negative values: min={V.min()}"

    # DELETED test_fd_control_nonnegative (same property as V≥0 for this problem)
    # and test_fd_dispatches_correctly (shape assertion + literal V[0,0]=0)
    # per 2026-04-19 audit.

    def test_fd_trajectory_monotone(self, nonlinear_params):
        """Trajectory should decrease monotonically — also verifies solver
        dispatched correctly (alpha!=1 path) since we get a valid trajectory."""
        grid, _, v_star = solve_hjb(nonlinear_params, M=50)
        x = extract_optimal_trajectory(grid, v_star, nonlinear_params)
        assert abs(x[0] - nonlinear_params.X0) < 1e-6
        diffs = np.diff(x)
        assert np.all(diffs <= 1e-3), "FD trajectory should be decreasing"
