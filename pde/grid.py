"""Finite difference grid setup and stability checks.

Adapted from MF796 HW3 grid construction. Key differences from BS-PDE:
    - State variable: inventory x ∈ [0, X0] instead of stock price S ∈ [0, S_max]
    - Time: backward from T to 0 (same as HW3)
    - No CFL issue for the basic 1D HJB (first-order in x, not second-order)
      BUT if we add diffusion terms (stochastic vol extension), CFL matters.

HW3 reference pattern:
    s_grid = np.linspace(0, s_max, M + 1)
    t_grid = np.linspace(0, T, N + 1)
    ht_max = hs**2 / (sigma**2 * s_max**2)  # CFL for explicit Euler
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.params import ACParams


@dataclass
class ExecutionGrid:
    """Finite difference grid for the HJB optimal execution problem.

    Attributes
    ----------
    x_grid : np.ndarray, shape (M+1,)
        Inventory grid from 0 to X0.
    t_grid : np.ndarray, shape (N+1,)
        Time grid from 0 to T.
    M : int
        Number of spatial (inventory) grid points.
    N : int
        Number of time steps.
    dx : float
        Inventory step size.
    dt : float
        Time step size.
    """

    x_grid: np.ndarray
    t_grid: np.ndarray
    M: int
    N: int
    dx: float
    dt: float


def build_grid(params: ACParams, M: int = 200) -> ExecutionGrid:
    """Build the finite difference grid for the execution problem.

    Parameters
    ----------
    params : ACParams
        Model parameters (X0, T, N determine the grid).
    M : int
        Number of inventory grid points (spatial resolution).

    Returns
    -------
    ExecutionGrid
    """
    x_grid = np.linspace(0, params.X0, M + 1)
    t_grid = np.linspace(0, params.T, params.N + 1)
    dx = params.X0 / M
    dt = params.dt

    return ExecutionGrid(
        x_grid=x_grid,
        t_grid=t_grid,
        M=M,
        N=params.N,
        dx=dx,
        dt=dt,
    )


def check_cfl(grid: ExecutionGrid, params: ACParams) -> dict:
    """Check CFL stability condition (relevant for 2D stochastic vol extension).

    For the basic 1D HJB with linear impact, the PDE is first-order in x
    and does not require a CFL condition. However, when we add diffusion
    (Part D: stochastic vol), the 2D PDE has second-order terms and CFL
    matters.

    HW3 pattern: ht_max = hs^2 / (sigma^2 * s_max^2)

    Returns
    -------
    dict
        {'stable': bool, 'dt': float, 'dt_max': float, 'cfl_ratio': float}
    """
    # For the basic problem: always stable (first-order PDE)
    # Placeholder for Part D extension
    return {
        "stable": True,
        "dt": grid.dt,
        "dt_max": float("inf"),  # No restriction for 1D first-order
        "cfl_ratio": 0.0,
        "note": "1D HJB is first-order; CFL not required. "
                "Will matter for 2D extension in Part D.",
    }
