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
from scipy.stats.qmc import Sobol
from scipy.special import ndtri

from shared.params import ACParams
from shared.cost_model import temporary_impact


def generate_normal_increments(
    n_paths: int,
    n_steps: int,
    method: str = "pseudo",
    seed: int = 42,
) -> np.ndarray:
    """Generate standard normal increments for Monte Carlo simulation.

    Parameters
    ----------
    n_paths : int
        Number of paths (for Sobol, should be a power of 2).
    n_steps : int
        Number of time steps per path.
    method : "pseudo" | "sobol" | "antithetic"
        "pseudo"    : standard numpy pseudorandom
        "sobol"     : scrambled Sobol quasi-random (low discrepancy)
        "antithetic": pseudorandom with Z / -Z pairing
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Z : np.ndarray, shape (n_paths, n_steps)
        Standard normal increments.
    """
    if method == "pseudo":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_paths, n_steps))

    elif method == "antithetic":
        rng = np.random.default_rng(seed)
        n_half = n_paths // 2
        Z_half = rng.standard_normal((n_half, n_steps))
        return np.vstack([Z_half, -Z_half])

    elif method == "sobol":
        sampler = Sobol(d=n_steps, scramble=True, seed=seed)
        u = sampler.random(n_paths)
        Z = ndtri(np.clip(u, 1e-10, 1 - 1e-10))
        return Z

    else:
        raise ValueError(f"Unknown method: {method}")


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
    Z_extern: np.ndarray | None = None,
    scheme: str = "exact",
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate execution cost across many price paths.

    For a given deterministic trajectory x(t), simulates the stochastic
    price process and computes realized execution cost per path.

    Price SDE with market impact:
        dS = mu*S*dt + sigma*S*dW - gamma*v*dt

    Parameters
    ----------
    params : ACParams
    trajectory_x : np.ndarray, shape (N+1,)
        Inventory trajectory (from strategies or PDE solver).
    n_paths : int
    seed : int
    antithetic : bool
    Z_extern : np.ndarray, shape (n_paths, N) or None
        Pre-generated normal increments. If None, generated internally.
    scheme : "exact" | "euler" | "milstein"
        "exact"    : log-normal exact GBM step (no discretization error for GBM part)
        "euler"    : Euler-Maruyama discretization
        "milstein" : Milstein scheme (adds 0.5*sigma^2*S*(Z^2-1)*dt correction)

    Returns
    -------
    price_paths : np.ndarray, shape (n_paths, N+1)
        Simulated price paths with permanent impact.
    costs : np.ndarray, shape (n_paths,)
        Realized execution cost for each path.
    """
    if scheme not in ("exact", "euler", "milstein"):
        raise ValueError(f"Unknown scheme: {scheme}. Use 'exact', 'euler', or 'milstein'.")

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
    if Z_extern is not None:
        Z = Z_extern
    elif antithetic:
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
        # Accumulate implementation shortfall cost
        costs += n_k[k] * (params.S0 - S[:, k]) + h_k[k] * n_k[k]

        # Price step depends on scheme
        if scheme == "exact":
            # Exact log-normal: no discretization error for GBM part
            log_drift = (params.mu - 0.5 * params.sigma**2) * dt
            S[:, k + 1] = (
                S[:, k] * np.exp(log_drift + params.sigma * sqrt_dt * Z[:, k])
                - params.gamma * n_k[k]
            )

        elif scheme == "euler":
            # Euler-Maruyama: S_{k+1} = S_k + mu*S_k*dt + sigma*S_k*sqrt(dt)*Z_k
            S[:, k + 1] = (
                S[:, k]
                + params.mu * S[:, k] * dt
                + params.sigma * S[:, k] * sqrt_dt * Z[:, k]
                - params.gamma * n_k[k]
            )

        elif scheme == "milstein":
            # Milstein: Euler + 0.5*sigma^2*S*(Z^2-1)*dt correction
            # For dS = mu*S*dt + sigma*S*dW, sigma(S) = sigma*S, sigma'(S) = sigma
            # Correction = 0.5 * sigma * (sigma*S) * (Z^2 - 1) * dt
            #            = 0.5 * sigma^2 * S * (Z^2 - 1) * dt
            S[:, k + 1] = (
                S[:, k]
                + params.mu * S[:, k] * dt
                + params.sigma * S[:, k] * sqrt_dt * Z[:, k]
                + 0.5 * params.sigma**2 * S[:, k] * (Z[:, k]**2 - 1) * dt
                - params.gamma * n_k[k]
            )

    return S, costs


def simulate_execution_with_control_variate(
    params: ACParams,
    trajectory_x: np.ndarray,
    twap_x: np.ndarray,
    n_paths: int = 10000,
    seed: int = 42,
    Z_extern: np.ndarray | None = None,
    scheme: str = "exact",
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
    Z_extern : np.ndarray or None
        Pre-generated normal increments. If provided, both strategy and
        TWAP simulations use the same Z (required for correlation).
    scheme : "exact" | "euler" | "milstein"

    Returns
    -------
    price_paths : np.ndarray
        Price paths (from strategy simulation).
    costs_cv : np.ndarray
        Control-variate-adjusted costs.
    """
    from shared.cost_model import execution_cost

    if Z_extern is not None:
        # Use the same external Z for both — ensures correlation
        price_paths, costs_strategy = simulate_execution(
            params, trajectory_x, n_paths, seed,
            antithetic=False, Z_extern=Z_extern, scheme=scheme,
        )
        _, costs_twap = simulate_execution(
            params, twap_x, n_paths, seed,
            antithetic=False, Z_extern=Z_extern, scheme=scheme,
        )
    else:
        # Fall back to same-seed approach (original behavior)
        price_paths, costs_strategy = simulate_execution(
            params, trajectory_x, n_paths, seed,
            antithetic=True, scheme=scheme,
        )
        _, costs_twap = simulate_execution(
            params, twap_x, n_paths, seed,
            antithetic=True, scheme=scheme,
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
