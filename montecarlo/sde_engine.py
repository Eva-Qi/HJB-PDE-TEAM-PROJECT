"""SDE discretization and path simulation for execution cost estimation.

Simulates the price process under execution with market impact:

    dS = mu*S*dt + sigma*S*dW - g(v)*dt

where g(v) = gamma * v is the permanent impact.

The execution cost for each path is computed as the realized P&L:
    Cost = sum_k n_k * S_k + h(v_k) * n_k
         = sum of (shares traded * execution price including temporary impact)

Methods:
    - Euler-Maruyama (first-order)
    - Milstein (second-order, optional extension)

Variance reduction:
    - Antithetic variates: for each path W, also simulate -W
    - Control variate: use TWAP expected cost as control
"""

from __future__ import annotations

import numpy as np

from shared.params import ACParams
from shared.cost_model import temporary_impact


def simulate_gbm_paths(
    params: ACParams,
    n_paths: int = 10000,
    seed: int = 42,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate GBM price paths (without impact — pure price dynamics).

    Uses exact log-normal discretization:
        S_{k+1} = S_k * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_k)

    Parameters
    ----------
    params : ACParams
    n_paths : int
        Number of Monte Carlo paths. If antithetic, must be even.
    seed : int
        Random seed for reproducibility.
    antithetic : bool
        If True, generates n_paths/2 paths and mirrors them.

    Returns
    -------
    np.ndarray, shape (n_paths, N+1)
        Price paths. paths[:, 0] = S0.
    """
    rng = np.random.default_rng(seed)
    N = params.N
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    if antithetic:
        n_half = n_paths // 2
        Z = rng.standard_normal((n_half, N))
        Z = np.vstack([Z, -Z])  # antithetic pairs
    else:
        Z = rng.standard_normal((n_paths, N))

    actual_paths = Z.shape[0]
    S = np.zeros((actual_paths, N + 1))
    S[:, 0] = params.S0

    # Exact log-normal simulation (guarantees positive prices)
    drift = (params.mu - 0.5 * params.sigma**2) * dt
    for k in range(N):
        S[:, k + 1] = S[:, k] * np.exp(drift + params.sigma * sqrt_dt * Z[:, k])

    return S


def simulate_execution(
    params: ACParams,
    trajectory_x: np.ndarray,
    n_paths: int = 10000,
    seed: int = 42,
    antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate execution cost across many price paths.

    For a given deterministic trajectory x(t), simulates the stochastic
    price process and computes realized execution cost per path.

    The price process includes permanent impact:
        S_{k+1} = S_k * (1 + mu*dt + sigma*sqrt(dt)*Z_k) - gamma*n_k

    Realized cost per path:
        C = sum_k [ n_k * (S_k + h(v_k)) ]
        = sum_k [ n_k * S_k  +  n_k * h(v_k) ]
                  ^^^^^^^^^      ^^^^^^^^^^^^^
                  market cost    temporary impact cost

    Parameters
    ----------
    params : ACParams
    trajectory_x : np.ndarray, shape (N+1,)
        Inventory trajectory (from strategies or PDE solver).
    n_paths : int
    seed : int
    antithetic : bool

    Returns
    -------
    price_paths : np.ndarray, shape (n_paths, N+1)
        Simulated price paths with permanent impact.
    costs : np.ndarray, shape (n_paths,)
        Realized execution cost for each path.
    """
    rng = np.random.default_rng(seed)
    N = params.N
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    # Trade list and rates (deterministic — same for all paths)
    n_k = trajectory_x[:-1] - trajectory_x[1:]  # shares sold each step
    v_k = n_k / dt  # trading rate

    # Temporary impact cost per share at each step (deterministic)
    h_k = temporary_impact(v_k, params.eta, params.alpha)

    # Generate random increments
    if antithetic:
        n_half = n_paths // 2
        Z = rng.standard_normal((n_half, N))
        Z = np.vstack([Z, -Z])
    else:
        Z = rng.standard_normal((n_paths, N))

    actual_paths = Z.shape[0]

    # Simulate price paths with permanent impact
    S = np.zeros((actual_paths, N + 1))
    S[:, 0] = params.S0
    costs = np.zeros(actual_paths)

    for k in range(N):
        # Implementation shortfall at step k:
        # IS_k = n_k * (S0 - S_k) + h_k * n_k
        # (S0 - S_k) captures price deterioration from vol + permanent impact
        # h_k * n_k is the temporary impact cost
        costs += n_k[k] * (params.S0 - S[:, k]) + h_k[k] * n_k[k]

        # Price evolution: exact log-normal GBM + permanent impact
        drift = (params.mu - 0.5 * params.sigma**2) * dt
        S[:, k + 1] = (
            S[:, k] * np.exp(drift + params.sigma * sqrt_dt * Z[:, k])
            - params.gamma * n_k[k]
        )

    return S, costs


def simulate_execution_with_control_variate(
    params: ACParams,
    trajectory_x: np.ndarray,
    twap_x: np.ndarray,
    n_paths: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """MC with control variate: use TWAP cost as control.

    The idea: Corr(cost_strategy, cost_TWAP) is high because both depend
    on the same price paths. Using TWAP cost as a control variate reduces
    the variance of the strategy cost estimator.

    C_cv = C_strategy - beta * (C_twap - E[C_twap])
    where beta = Cov(C_strategy, C_twap) / Var(C_twap)

    Parameters
    ----------
    params : ACParams
    trajectory_x : np.ndarray
        Strategy trajectory to estimate cost for.
    twap_x : np.ndarray
        TWAP trajectory (control).
    n_paths : int
    seed : int

    Returns
    -------
    price_paths : np.ndarray
        Price paths (from strategy simulation).
    costs_cv : np.ndarray
        Control-variate-adjusted costs.
    """
    from shared.cost_model import execution_cost

    # Simulate both strategies with the SAME random seed
    price_paths, costs_strategy = simulate_execution(
        params, trajectory_x, n_paths, seed, antithetic=True
    )
    _, costs_twap = simulate_execution(
        params, twap_x, n_paths, seed, antithetic=True
    )

    # E[C_twap] from the deterministic cost model (our "known" expectation)
    E_twap = execution_cost(twap_x, params)

    # Optimal beta
    cov = np.cov(costs_strategy, costs_twap)[0, 1]
    var_twap = np.var(costs_twap)
    beta = cov / var_twap if var_twap > 0 else 0.0

    # Adjusted costs
    costs_cv = costs_strategy - beta * (costs_twap - E_twap)

    return price_paths, costs_cv
