"""Execution strategies: TWAP, VWAP, and optimal trajectories.

These generate deterministic inventory trajectories x(t) that are then
fed into the MC engine for cost simulation.
"""

from __future__ import annotations

import numpy as np

from shared.params import ACParams, HOURLY_VOLUME_PROFILE, almgren_chriss_closed_form


def twap_trajectory(params: ACParams) -> np.ndarray:
    """Time-Weighted Average Price: uniform liquidation.

    x(t_k) = X0 * (1 - k/N)

    Trades equal amount each period regardless of conditions.
    This is the simplest baseline.

    Returns
    -------
    np.ndarray, shape (N+1,)
        Inventory trajectory.
    """
    return params.X0 * np.linspace(1, 0, params.N + 1)


def vwap_trajectory(params: ACParams) -> np.ndarray:
    """Volume-Weighted Average Price: trade proportional to volume profile.

    Allocates shares to each time step proportional to expected volume.
    Uses HOURLY_VOLUME_PROFILE from Binance historical data.

    If params.volume_profile is provided, uses that instead.

    Returns
    -------
    np.ndarray, shape (N+1,)
        Inventory trajectory.
    """
    N = params.N

    if params.volume_profile is not None:
        weights = params.volume_profile
    else:
        # Map each time step to an hour and use the volume profile.
        # Assumes execution window spans a full 24h cycle starting at hour 0.
        # For N < 24, multiple hours collapse into single steps; for N >> 24,
        # multiple steps share the same hourly weight.
        hours_per_step = 24.0 / N
        weights = np.zeros(N)
        for k in range(N):
            hour = int((k * hours_per_step) % 24)
            weights[k] = HOURLY_VOLUME_PROFILE.get(hour, 1.0)

    # Normalize weights
    weights = weights / weights.sum()

    # Convert weights to cumulative trade fractions
    cum_traded = np.cumsum(weights)
    cum_traded = np.insert(cum_traded, 0, 0.0)  # prepend 0

    # Inventory trajectory
    x = params.X0 * (1.0 - cum_traded)
    return x


def optimal_trajectory(params: ACParams) -> np.ndarray:
    """Almgren-Chriss closed-form optimal trajectory.

    x*(t_k) = X0 * sinh(kappa * (T - t_k)) / sinh(kappa * T)

    For linear impact (alpha=1) only. This is the analytical benchmark
    that P2's PDE solver and P3's MC should reproduce.

    Returns
    -------
    np.ndarray, shape (N+1,)
    """
    _, x, _ = almgren_chriss_closed_form(params)
    return x
