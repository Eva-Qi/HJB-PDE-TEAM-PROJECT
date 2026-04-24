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


def _hourly_volume_weights(params: ACParams) -> np.ndarray:
    """Per-step volume weights for VWAP/POV, respecting params.T.

    If params.volume_profile is provided, returns it directly (caller owns
    normalisation). Otherwise samples HOURLY_VOLUME_PROFILE at the midpoint
    of each step, mapped from years via params.T.

    The execution span in hours is params.T * 365.25 * 24. Each step k is
    centred at (k + 0.5) * hours_per_step hours past params.execution_start_hour,
    and the UTC hour is that value mod 24. For T >= 24h the profile cycles
    and weights average out toward the daily mean.

    Returns
    -------
    np.ndarray, shape (N,)
        Un-normalised volume weights (caller normalises).
    """
    if params.volume_profile is not None:
        return np.asarray(params.volume_profile, dtype=float)

    N = params.N
    start_hour = float(getattr(params, "execution_start_hour", 0.0))
    T_hours = params.T * 365.25 * 24.0  # years -> hours
    hours_per_step = T_hours / N
    weights = np.zeros(N)
    for k in range(N):
        hour_float = (start_hour + (k + 0.5) * hours_per_step) % 24.0
        weights[k] = HOURLY_VOLUME_PROFILE.get(int(hour_float), 1.0)
    return weights


def vwap_trajectory(params: ACParams) -> np.ndarray:
    """Volume-Weighted Average Price: trade proportional to volume profile.

    Uses `_hourly_volume_weights(params)` which respects params.T and
    params.execution_start_hour. For T below one hour, all steps fall in
    the same hour bucket and VWAP reduces to TWAP. For T >= 1 day the
    profile cycles through 24 hours, weighting peaks and troughs correctly.

    If params.volume_profile is provided, uses that directly.

    Returns
    -------
    np.ndarray, shape (N+1,)
        Inventory trajectory.
    """
    weights = _hourly_volume_weights(params)

    # Normalize weights
    weights = weights / weights.sum()

    # Convert weights to cumulative trade fractions
    cum_traded = np.cumsum(weights)
    cum_traded = np.insert(cum_traded, 0, 0.0)  # prepend 0

    # Inventory trajectory
    x = params.X0 * (1.0 - cum_traded)
    return x


def pov_trajectory(params: ACParams, participation_rate: float = 0.10) -> np.ndarray:
    """Percentage-of-Volume: trade as a fraction of expected market volume.

    At each step k, trade min(n_pov, remaining_inventory) where
    n_pov = participation_rate * expected_volume[k].

    If params.volume_profile is provided, uses it for volume estimation.
    Otherwise uses HOURLY_VOLUME_PROFILE normalized so sum(volume_per_step) = 1.

    This is a common benchmark in institutional execution — trade visibility
    stays below participation_rate of market.

    Parameters
    ----------
    params : ACParams
    participation_rate : float
        Fraction of market volume to trade per step (e.g., 0.10 = 10%).
        The POV schedule is scaled so remaining inventory is fully liquidated
        by the final step (any residual is assigned to the last step).

    Returns
    -------
    np.ndarray, shape (N+1,)
        Inventory trajectory. x[0]=X0, x[N]=0.
    """
    N = params.N
    X0 = params.X0

    # Volume weights via shared helper — respects params.T + execution_start_hour
    weights = _hourly_volume_weights(params).astype(float)
    weights = weights / weights.sum()  # normalized step-volume fractions

    # Raw POV trades: participation_rate * volume_weight * some_total_market_volume
    # Since we only care about proportional shape, raw_trades[k] ∝ weights[k].
    # We then scale so the cumulative trade equals X0, with any residual flushed
    # to the last step to guarantee full liquidation.
    raw_trades = participation_rate * weights  # proportional schedule
    raw_total = raw_trades.sum()
    # Scale so total traded = X0
    scale = X0 / raw_total if raw_total > 0 else 1.0
    n_per_step = raw_trades * scale  # this sums to X0 by construction

    # Build inventory trajectory
    x = np.empty(N + 1)
    x[0] = X0
    for k in range(N):
        x[k + 1] = x[k] - n_per_step[k]

    # Clamp to zero (floating-point safety)
    x = np.maximum(x, 0.0)
    x[-1] = 0.0
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
