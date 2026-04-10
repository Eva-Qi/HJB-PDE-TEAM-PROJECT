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

import warnings

import numpy as np
from scipy.stats.qmc import Sobol
from scipy.special import ndtri

from shared.params import ACParams
from shared.cost_model import temporary_impact
from extensions.heston import HestonParams
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory


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


def simulate_heston_execution(
    params: ACParams,
    heston_params: HestonParams,
    trajectory_x: np.ndarray,
    n_paths: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate execution cost under Heston stochastic volatility.

    Extends simulate_execution to handle stochastic variance. The price
    and variance processes are jointly simulated with correlated Brownian
    increments:

        dS = mu*S*dt + sqrt(v)*S*dW_S - gamma*v*dt
        dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
        corr(dW_S, dW_v) = rho

    Uses Euler-Maruyama with the Full Truncation scheme (Lord, Koekkoek &
    Van Dijk 2010) to handle the variance hitting zero. Full truncation
    clamps v at zero inside the drift and diffusion terms of the variance
    update, but allows the stored v itself to go slightly negative —
    minimizing discretization bias compared to reflection or absorption.

    Parameters
    ----------
    params : ACParams
        Almgren-Chriss execution parameters. Only S0, mu, X0, T, N, gamma,
        eta, alpha are used here; sigma is replaced by stochastic v.
    heston_params : HestonParams
        Heston stochastic volatility parameters (kappa, theta, xi, rho, v0).
    trajectory_x : np.ndarray, shape (N+1,)
        Inventory trajectory from a strategy (TWAP, VWAP, or optimal).
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    price_paths : np.ndarray, shape (n_paths, N+1)
        Simulated price paths with stochastic vol and permanent impact.
    variance_paths : np.ndarray, shape (n_paths, N+1)
        Simulated variance paths. May contain small negative values from
        the full truncation scheme — these are numerical artifacts and
        should be clamped if used downstream.
    costs : np.ndarray, shape (n_paths,)
        Realized implementation shortfall cost per path.
    """
    if 2 * heston_params.kappa * heston_params.theta < heston_params.xi ** 2:
        warnings.warn(
            f"Feller condition violated: 2*kappa*theta={2*heston_params.kappa*heston_params.theta:.4f} "
            f"< xi^2={heston_params.xi**2:.4f}. Variance process may hit zero.",
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    N = params.N
    dt = params.dt
    sqrt_dt = np.sqrt(dt)

    # Trade list and rates (deterministic — same for all paths)
    n_k = trajectory_x[:-1] - trajectory_x[1:]  # shares sold each step
    v_k = n_k / dt                              # trading rate
    h_k = temporary_impact(v_k, params.eta, params.alpha)

    # Build correlated normal increments for the two Brownian motions.
    # Z_S and Z_v are each N(0,1) with corr(Z_S, Z_v) = rho.
    rho = heston_params.rho
    W1 = rng.standard_normal((n_paths, N))
    W2 = rng.standard_normal((n_paths, N))
    Z_S = W1
    Z_v = rho * W1 + np.sqrt(1.0 - rho ** 2) * W2

    # Initialize state arrays
    S = np.zeros((n_paths, N + 1))
    var_paths = np.zeros((n_paths, N + 1))
    S[:, 0] = params.S0
    var_paths[:, 0] = heston_params.v0
    costs = np.zeros(n_paths)

    # Unpack Heston params for readability inside the loop
    kappa = heston_params.kappa
    theta = heston_params.theta
    xi = heston_params.xi

    for k in range(N):
        # Accumulate implementation shortfall cost at current step
        costs += n_k[k] * (params.S0 - S[:, k]) + h_k[k] * n_k[k]

        # Full truncation: clamp variance at zero before using it in
        # sqrt() and in the drift, but let var_paths itself evolve freely.
        v_plus = np.maximum(var_paths[:, k], 0.0)

        # Variance step (Euler-Maruyama, full truncation)
        var_paths[:, k + 1] = (
            var_paths[:, k]
            + kappa * (theta - v_plus) * dt
            + xi * np.sqrt(v_plus) * sqrt_dt * Z_v[:, k]
        )

        # Price step (Euler-Maruyama with stochastic vol + permanent impact)
        S[:, k + 1] = (
            S[:, k]
            + params.mu * S[:, k] * dt
            + np.sqrt(v_plus) * S[:, k] * sqrt_dt * Z_S[:, k]
            - params.gamma * n_k[k]
        )

    return S, var_paths, costs

def simulate_pde_optimal_execution(
    params: ACParams,
    n_paths: int = 10000,
    seed: int = 42,
    scheme: str = "exact",
    M: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate execution cost using the PDE-derived optimal trajectory.

    For nonlinear impact (alpha != 1), there is no closed-form Almgren-Chriss
    solution. This function chains P2's HJB PDE solver with P3's MC engine:

        1. Solve the HJB PDE to get the optimal control field v*(x, t).
        2. Extract the optimal inventory trajectory by walking forward
           from X0 following v*.
        3. Run MC simulation on that trajectory to get a cost distribution.

    Used for cross-validation: the MC mean cost should agree with
    `execution_cost(trajectory, params)` (deterministic formula) within ~5%.
    Large disagreements indicate a bug in either the PDE solver or MC engine.

    Parameters
    ----------
    params : ACParams
        Execution parameters. Supports both linear (alpha=1) and nonlinear
        (alpha != 1) temporary impact.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.
    scheme : "exact" | "euler" | "milstein"
        SDE discretization scheme for the price process.
    M : int
        PDE inventory grid resolution (number of interior points).

    Returns
    -------
    trajectory : np.ndarray, shape (N+1,)
        PDE-derived optimal inventory path.
    price_paths : np.ndarray, shape (n_paths, N+1)
        Simulated price paths under the optimal trajectory.
    costs : np.ndarray, shape (n_paths,)
        Realized implementation shortfall per path.
    """
    grid, _, v_star = solve_hjb(params, M=M)
    trajectory = extract_optimal_trajectory(grid, v_star, params)
    price_paths, costs = simulate_execution(
        params, trajectory, n_paths=n_paths, seed=seed, scheme=scheme
    )
    return trajectory, price_paths, costs