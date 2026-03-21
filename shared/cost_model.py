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
        Total cost = permanent impact cost + temporary impact cost.

    Notes
    -----
    Cost = sum_{k=0}^{N-1} [ n_k * (gamma * sum_{j<=k} n_j)
                              + tau * h(n_k/tau) * n_k/tau ]
    Simplified for implementation:
        perm = gamma * sum_k(n_k * cumulative_n_k)
        temp = sum_k(eta * |n_k/tau|^alpha * n_k)
    """
    dt = params.dt
    n_k = x[:-1] - x[1:]  # trade list: shares sold each step

    # Permanent impact cost: each trade moves price by gamma * n_k
    # Subsequent trades suffer from prior permanent shifts
    cumulative_n = np.cumsum(n_k)
    perm_cost = params.gamma * np.sum(n_k * cumulative_n)

    # Temporary impact cost
    v_k = n_k / dt  # trading rate
    h_v = temporary_impact(v_k, params.eta, params.alpha)
    temp_cost = np.sum(h_v * n_k)

    return float(perm_cost + temp_cost)


def execution_risk(x: np.ndarray, params: ACParams) -> float:
    """Execution risk (variance of cost) for a given trajectory.

    Var[cost] = sigma^2 * sum_{k=0}^{N-1} tau * x_k^2

    Holding inventory x_k exposes us to price variance sigma^2 * tau
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
    return float(params.sigma**2 * dt * np.sum(x[:-1] ** 2))


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
