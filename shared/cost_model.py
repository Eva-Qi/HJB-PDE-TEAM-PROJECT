"""Almgren-Chriss execution cost model.

Core cost functions used by both PDE (Part B) and Monte Carlo (Part C).
All functions are pure math (numpy only) — no data dependencies.

Notation follows Almgren & Chriss (2001):
    x_k = remaining inventory after step k (x_0 = X0, x_N = 0)
    n_k = x_{k-1} - x_k = shares traded in step k (positive = selling)
    v_k = n_k / tau = trading rate in step k
    tau = dt = T / N = time step size

Cost components:
    Permanent impact: g(v) = gamma * v  (price shifts permanently)
    Temporary impact:  h(v) = eta * |v|^alpha * sign(v)  (instantaneous cost)
    Execution risk:    Var[cost] ∝ sigma^2 * sum(tau * x_k^2)
"""

from __future__ import annotations

import numpy as np

from shared.params import ACParams


def permanent_impact(v: np.ndarray | float, gamma: float) -> np.ndarray | float:
    """Permanent price impact: g(v) = gamma * v.

    Each trade permanently moves the price by g(v) * dt.
    """
    return gamma * v


def temporary_impact(
    v: np.ndarray | float, eta: float, alpha: float = 1.0
) -> np.ndarray | float:
    """Temporary price impact: h(v) = eta * |v|^alpha * sign(v).

    This is the per-share cost above midprice paid for immediacy.
    For alpha=1 (linear): h(v) = eta * v.
    """
    if abs(alpha - 1.0) < 1e-10:
        return eta * v
    return eta * np.abs(v) ** alpha * np.sign(v)


def execution_fees(x: np.ndarray, params: ACParams) -> float:
    """Exchange fees for a given inventory trajectory.

    Fee per trade = (fee_bps / 1e4) * S0 * |n_k|. For a full liquidation
    (x[0]=X0 → x[N]=0) with monotone trajectory, this collapses to
    (fee_bps / 1e4) * S0 * X0, i.e. a constant across strategies.

    Parameters
    ----------
    x : np.ndarray, shape (N+1,)
        Inventory trajectory.
    params : ACParams
        Model parameters. Uses params.fee_bps (0.0 → no fees).

    Returns
    -------
    float
        Total fees in dollars.
    """
    if params.fee_bps == 0.0:
        return 0.0
    n_k = x[:-1] - x[1:]
    return float((params.fee_bps / 1e4) * params.S0 * np.sum(np.abs(n_k)))


def execution_cost(x: np.ndarray, params: ACParams) -> float:
    """Total expected execution cost for a given inventory trajectory.

    Parameters
    ----------
    x : np.ndarray, shape (N+1,)
        Inventory trajectory: x[0]=X0, x[N]=0 (or close to 0).
    params : ACParams
        Model parameters.

    Returns
    -------
    float
        Total cost = permanent impact + temporary impact + exchange fees.

    Notes
    -----
    Cost = sum_{k=0}^{N-1} [ n_k * (gamma * sum_{j<k} n_j)
                              + tau * h(n_k/tau) * n_k/tau ]
           + (fee_bps / 1e4) * S0 * sum_k |n_k|

    Uses the Almgren-Chriss (2000) convention: trade k's permanent
    impact affects trades k+1, k+2, ... but NOT trade k itself.
    This matches the MC implementation-shortfall formula where S_k
    is the price *before* trade k executes.

    Exchange fees (params.fee_bps) are added on top of impact costs.
    Default fee_bps=0 preserves pre-fee cost behavior for legacy tests.
    """
    dt = params.dt
    n_k = x[:-1] - x[1:]  # trade list: shares sold each step

    # Permanent impact cost: trade k pays for cumulative impact of
    # trades 0..k-1 (excluding self-impact, per A&C convention)
    prior_cumulative_n = np.concatenate([[0.0], np.cumsum(n_k[:-1])])
    perm_cost = params.gamma * np.sum(n_k * prior_cumulative_n)

    # Temporary impact cost
    v_k = n_k / dt  # trading rate
    h_v = temporary_impact(v_k, params.eta, params.alpha)
    temp_cost = np.sum(h_v * n_k)

    # Exchange fees (constant for any full liquidation if fee_bps>0)
    fee_cost = execution_fees(x, params)

    return float(perm_cost + temp_cost + fee_cost)


def execution_risk(x: np.ndarray, params: ACParams) -> float:
    """Execution risk (variance of cost) for a given trajectory.

    Var[cost] = S0^2 * sigma^2 * sum_{k=0}^{N-1} tau * x_k^2

    Holding inventory x_k exposes us to dollar variance (S0*sigma)^2 * tau
    at each step. Total variance is the sum over all steps.

    Parameters
    ----------
    x : np.ndarray, shape (N+1,)
        Inventory trajectory.
    params : ACParams

    Returns
    -------
    float
        Execution variance.
    """
    dt = params.dt
    # Risk from holding inventory x_k during interval [t_k, t_{k+1}]
    # Uses dollar volatility (S0 * sigma_pct) so risk is in dollar² units
    return float(params.S0**2 * params.sigma**2 * dt * np.sum(x[:-1] ** 2))


def objective(x: np.ndarray, params: ACParams) -> float:
    """Almgren-Chriss objective: E[cost] + lambda * Var[cost].

    This is what the HJB PDE minimizes and what Monte Carlo estimates.

    Parameters
    ----------
    x : np.ndarray, shape (N+1,)
        Inventory trajectory.
    params : ACParams

    Returns
    -------
    float
        E[cost] + lambda * risk.
    """
    return execution_cost(x, params) + params.lam * execution_risk(x, params)
