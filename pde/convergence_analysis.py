# ARCADIA CONTRIBUTION — pending review
# PDE convergence study: Richardson extrapolation on HJB solver
# Known issue: p=1.22/1.26 (>1), needs investigation
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.params import DEFAULT_PARAMS
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory



CONV_ALPHA = 1.5
CONV_X0 = 100000.0
CONV_S0 = 70000.0
CONV_SIGMA = 0.30
CONV_GAMMA = 2.5e-7
CONV_ETA = 2.5e-6
CONV_LAM = 4e-7
CONV_T = 0.25
CONV_TERMINAL_PENALTY = 1e4

GRID_LEVELS = [50, 100, 200, 400]


# Some helpers and fixes

def half_liquidation_time(t_traj: np.ndarray, x_traj: np.ndarray) -> float:
    half_x = 0.5 * x_traj[0]
    idx = np.where(x_traj <= half_x)[0]

    if len(idx) == 0:
        return float(t_traj[-1])

    return float(t_traj[idx[0]])

def first_step_trade(x_star: np.ndarray) -> float:
    x = np.asarray(x_star, dtype=float).reshape(-1)
    return float(x[0] - x[1])

def get_x_grid(grid, params) -> np.ndarray:
    if hasattr(grid, "x"):
        return np.asarray(grid.x, dtype=float)
    if hasattr(grid, "x_grid"):
        return np.asarray(grid.x_grid, dtype=float)
    if hasattr(grid, "inventory"):
        return np.asarray(grid.inventory, dtype=float)
    if isinstance(grid, dict):
        for key in ["x", "x_grid", "inventory"]:
            if key in grid:
                return np.asarray(grid[key], dtype=float)
    return np.linspace(0.0, params.X0, len(np.asarray(grid)) if np.ndim(grid) == 1 else 401)


def get_t_grid(grid, params, V) -> np.ndarray:
    if hasattr(grid, "t"):
        return np.asarray(grid.t, dtype=float)
    if hasattr(grid, "t_grid"):
        return np.asarray(grid.t_grid, dtype=float)
    if isinstance(grid, dict):
        for key in ["t", "t_grid", "time"]:
            if key in grid:
                return np.asarray(grid[key], dtype=float)

    V_arr = np.asarray(V)
    n0, n1 = V_arr.shape
    if n0 == params.N + 1:
        return np.linspace(0.0, params.T, n0)
    if n1 == params.N + 1:
        return np.linspace(0.0, params.T, n1)
    return np.linspace(0.0, params.T, params.N + 1)


def value_at_initial_state(grid, V, params) -> float:
    x_grid = get_x_grid(grid, params)
    t_grid = get_t_grid(grid, params, V)
    V_arr = np.asarray(V, dtype=float)

    if V_arr.shape == (len(t_grid), len(x_grid)):
        V0_slice = V_arr[0, :]
    elif V_arr.shape == (len(x_grid), len(t_grid)):
        V0_slice = V_arr[:, 0]
    else:
        raise ValueError(
            f"Could not interpret V shape {V_arr.shape} "
            f"against len(t_grid)={len(t_grid)}, len(x_grid)={len(x_grid)}"
        )

    return float(np.interp(params.X0, x_grid, V0_slice))


def trajectory_on_uniform_time(grid, x_star, params) -> tuple[np.ndarray, np.ndarray]:
    t_grid = get_t_grid(grid, params, np.zeros((params.N + 1, params.N + 1)))
    x_arr = np.asarray(x_star, dtype=float).reshape(-1)

    if len(t_grid) != len(x_arr):
        t_grid = np.linspace(0.0, params.T, len(x_arr))

    return t_grid, x_arr


def estimate_order(q_h, q_h2, q_h4) -> float:
    num = abs(q_h - q_h2)
    den = abs(q_h2 - q_h4)

    if den < 1e-14 or num < 1e-14:
        return np.nan

    return float(np.log2(num / den))


def richardson_extrapolation(q_h, q_h2, p) -> float:
    if not np.isfinite(p):
        return np.nan
    return float(q_h2 + (q_h2 - q_h) / (2.0**p - 1.0))


# Main
def build_params(M: int):
    # We refine BOTH inventory grid (M) and time steps (N=M)
    # so the study reflects O(dx) + O(dt).
    return replace(
        DEFAULT_PARAMS,
        alpha=CONV_ALPHA,
        X0=CONV_X0,
        S0=CONV_S0,
        sigma=CONV_SIGMA,
        gamma=CONV_GAMMA,
        eta=CONV_ETA,
        lam=CONV_LAM,
        T=CONV_T,
        N=M,
    )


def run_one_level(M: int) -> dict:
    params = build_params(M)

    grid, V, v_star = solve_hjb(
        params,
        M=M,
        terminal_penalty=CONV_TERMINAL_PENALTY,
    )

    x_star = extract_optimal_trajectory(grid, v_star, params)

    t_traj, x_traj = trajectory_on_uniform_time(grid, x_star, params)
    q0 = half_liquidation_time(t_traj, x_traj)

    return {
        "M": M,
        "params": params,
        "grid": grid,
        "V": V,
        "v_star": v_star,
        "x_star": x_star,
        "q0": q0,
        "t_traj": t_traj,
        "x_traj": x_traj,
    }


def compare_trajectories(coarse: dict, fine: dict) -> tuple[float, float]:
    t_c = coarse["t_traj"]
    x_c = coarse["x_traj"]

    t_f = fine["t_traj"]
    x_f = fine["x_traj"]

    x_f_on_c = np.interp(t_c, t_f, x_f)

    max_err = float(np.max(np.abs(x_c - x_f_on_c)))
    l2_err = float(np.sqrt(np.mean((x_c - x_f_on_c) ** 2)))

    return max_err, l2_err


def main():
    results = {}
    print("PDE convergence study for solve_hjb()")
    print(f"Grid levels: {GRID_LEVELS}")
    print(f"alpha = {CONV_ALPHA}  (nonlinear branch)")
    print(f"terminal_penalty = {CONV_TERMINAL_PENALTY}")

    for M in GRID_LEVELS:
        print(f"\n[RUN] M = {M}, N = {M}")
        results[M] = run_one_level(M)
        print(f"q0(M={M}) = {results[M]['q0']:.10f}")

    ######## Richardson on  q0 = V(0, X0)
    
    q50 = results[50]["q0"]
    q100 = results[100]["q0"]
    q200 = results[200]["q0"]
    q400 = results[400]["q0"]

    p_50_100_200 = estimate_order(q50, q100, q200)
    p_100_200_400 = estimate_order(q100, q200, q400)

    q_star_100 = richardson_extrapolation(q50, q100, p_50_100_200)
    q_star_200 = richardson_extrapolation(q100, q200, p_100_200_400)

    print("\nRichardson extrapolation on q0 = V(0, X0)")
    print(f"q50   = {q50:.10f}")
    print(f"q100  = {q100:.10f}")
    print(f"q200  = {q200:.10f}")
    print(f"q400  = {q400:.10f}")
    print(f"p(50,100,200)   = {p_50_100_200:.6f}")
    print(f"p(100,200,400)  = {p_100_200_400:.6f}")
    print(f"Richardson q* from (50,100)   = {q_star_100:.10f}")
    print(f"Richardson q* from (100,200)  = {q_star_200:.10f}")

    # Trajectory differences across refinements
    
    max_50_100, l2_50_100 = compare_trajectories(results[50], results[100])
    max_100_200, l2_100_200 = compare_trajectories(results[100], results[200])
    max_200_400, l2_200_400 = compare_trajectories(results[200], results[400])

    print("\nInventory trajectory differences")
    print(f"50 vs 100   : max = {max_50_100:.10f},  l2 = {l2_50_100:.10f}")
    print(f"100 vs 200  : max = {max_100_200:.10f},  l2 = {l2_100_200:.10f}")
    print(f"200 vs 400  : max = {max_200_400:.10f},  l2 = {l2_200_400:.10f}")

    # Error decay plot against finest solution (M=400)

    Ms = np.array([50, 100, 200], dtype=float)
    q_ref = q400

    scalar_errs = np.array([
        abs(q50 - q_ref),
        abs(q100 - q_ref),
        abs(q200 - q_ref),
    ], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.loglog(Ms, scalar_errs, marker="o", label="|q(M) - q(400)|")
    plt.xlabel("M")
    plt.ylabel("Error in q0")
    plt.title("PDE convergence study: scalar error vs grid size")
    plt.legend()
    plt.tight_layout()
    plt.show()



    # Plot

    plt.figure(figsize=(8, 5))
    for M in GRID_LEVELS:
        plt.plot(results[M]["t_traj"], results[M]["x_traj"], label=f"M={M}")
    plt.xlabel("Time")
    plt.ylabel("Inventory")
    plt.title("Optimal trajectories under grid refinement")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n=== Interpretation guide ===")
    print("If p is near 1, that supports first-order convergence consistent with O(dx)+O(dt).")
    print("If trajectory differences shrink as M increases, that supports numerical convergence.")
    print("If p is unstable or errors do not shrink, the solver or setup may need debugging.")


if __name__ == "__main__":
    main()

