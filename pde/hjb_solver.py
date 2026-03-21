"""Hamilton-Jacobi-Bellman PDE solver for optimal execution.

For the value function V(x, t) = min expected cost-to-go from state (x, t):

    V_t + min_v { eta*v^2 + lam*sigma^2*x^2 - v*V_x } = 0

For LINEAR impact (alpha=1), V(x,t) = alpha(t)*x^2 where alpha(t) satisfies
the Riccati ODE (stable, exact):

    alpha'(t) = alpha(t)^2/eta - lam*sigma^2
    alpha(T) = terminal_penalty

Analytical solution: alpha(t) = eta * kappa * coth(kappa*(T-t))
where kappa = sqrt(lam*sigma^2/eta).

Optimal control: v*(x,t) = alpha(t)*x/eta = kappa*coth(kappa*(T-t))*x

For NONLINEAR impact (alpha != 1), uses explicit finite differences on the
full 2D grid (x, t). This approach also extends to stochastic vol (Part D).

KEY DIFFERENCES from HW3's Black-Scholes PDE:
    - HW3: second-order linear PDE -> tridiagonal matrix system A @ C
    - HJB: first-order nonlinear (V_x^2 term) -> pointwise optimization
    - HW3: state = stock price S; HJB: state = remaining inventory x
    - Both use backward induction from terminal condition
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from shared.params import ACParams
from pde.grid import ExecutionGrid, build_grid


def solve_hjb(
    params: ACParams,
    M: int = 200,
    terminal_penalty: float = 1e8,
) -> tuple[ExecutionGrid, np.ndarray, np.ndarray]:
    """Solve the HJB PDE for optimal execution.

    For linear impact (alpha=1): uses the Riccati ODE reduction (exact).
    For nonlinear impact: uses explicit finite differences on the 2D grid.

    Parameters
    ----------
    params : ACParams
        Model parameters.
    M : int
        Number of inventory grid points.
    terminal_penalty : float
        Penalty for leftover inventory at T.

    Returns
    -------
    grid : ExecutionGrid
        The computational grid.
    V : np.ndarray, shape (M+1, N+1)
        Value function V(x_i, t_j).
    v_star : np.ndarray, shape (M+1, N+1)
        Optimal trading rate v*(x_i, t_j).
    """
    if abs(params.alpha - 1.0) < 1e-10:
        return _solve_hjb_riccati(params, M, terminal_penalty)
    else:
        return _solve_hjb_fd(params, M, terminal_penalty)


def _solve_hjb_riccati(
    params: ACParams,
    M: int,
    terminal_penalty: float,
) -> tuple[ExecutionGrid, np.ndarray, np.ndarray]:
    """Solve HJB via Riccati ODE reduction (linear impact, alpha=1).

    For linear impact, V(x,t) = alpha(t)*x^2 where alpha satisfies:
        d(alpha)/d(tau) = -alpha^2/eta + lam*sigma^2
    with alpha(tau=0) = terminal_penalty, tau = T - t (time remaining).

    This is numerically stable regardless of penalty size.
    """
    grid = build_grid(params, M)
    eta = params.eta
    lam = params.lam
    sigma = params.sigma
    T = params.T
    N = grid.N
    x = grid.x_grid

    # Solve Riccati ODE backward: tau = T - t, alpha(tau=0) = penalty
    # d(alpha)/d(tau) = -alpha^2/eta + lam*sigma^2
    def riccati_rhs(tau, alpha):
        return -alpha**2 / eta + lam * sigma**2

    tau_span = (0.0, T)
    tau_eval = T - grid.t_grid[::-1]  # tau values corresponding to t_grid (reversed)

    sol = solve_ivp(
        riccati_rhs,
        tau_span,
        [terminal_penalty],
        t_eval=tau_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success, f"Riccati ODE solve_ivp failed: {sol.message}"

    # alpha_values[j] = alpha at time t_grid[j]
    # sol.y[0] is indexed by tau; we need to reverse to get time ordering
    alpha_values = sol.y[0][::-1]  # now indexed by t_grid

    # Reconstruct V(x, t) = alpha(t) * x^2
    V = np.zeros((M + 1, N + 1))
    v_star = np.zeros((M + 1, N + 1))

    for j in range(N + 1):
        V[:, j] = alpha_values[j] * x**2
        # v*(x, t) = V_x / (2*eta) = 2*alpha(t)*x / (2*eta) = alpha(t)*x/eta
        v_star[:, j] = alpha_values[j] * x / eta

    # Enforce boundary: V(0, t) = 0, v*(0, t) = 0
    V[0, :] = 0.0
    v_star[0, :] = 0.0

    return grid, V, v_star


def _solve_hjb_fd(
    params: ACParams,
    M: int,
    terminal_penalty: float,
) -> tuple[ExecutionGrid, np.ndarray, np.ndarray]:
    """Solve HJB via explicit finite differences (general nonlinear impact).

    For nonlinear temporary impact h(v) = eta*|v|^alpha*sign(v), the
    optimal v* must be found numerically at each grid point via grid search.

    The Hamiltonian is:
        H(v) = eta*|v|^(alpha+1) - v*V_x + lam*sigma^2*x^2
    Note: eta*|v|^alpha*v = eta*|v|^(alpha+1) since v >= 0 for liquidation.
    """
    grid = build_grid(params, M)
    N = grid.N
    dt = grid.dt
    dx = grid.dx
    x = grid.x_grid
    eta = params.eta
    alpha_pow = params.alpha
    lam = params.lam
    sigma = params.sigma

    V = np.zeros((M + 1, N + 1))
    v_star = np.zeros((M + 1, N + 1))

    # Terminal condition
    V[:, N] = terminal_penalty * x**2

    n_candidates = 100  # grid search resolution for optimal v*

    def _find_optimal_v(V_x_val, x_i, eta, alpha_pow, lam, sigma, dt, n_cand):
        """Find v* minimizing Hamiltonian via grid search."""
        v_max = x_i / dt  # can't sell more than remaining inventory
        v_candidates = np.linspace(0, v_max, n_cand)
        # H(v) = eta*|v|^alpha*v - v*V_x + lam*sigma^2*x^2
        # Note: eta*|v|^alpha*v = eta*|v|^(alpha+1) for v >= 0
        H_vals = (
            eta * np.abs(v_candidates) ** alpha_pow * v_candidates
            - v_candidates * V_x_val
            + lam * sigma**2 * x_i**2
        )
        best_idx = np.argmin(H_vals)
        return v_candidates[best_idx], H_vals[best_idx]

    for j in range(N, 0, -1):
        for i in range(1, M + 1):
            x_i = x[i]

            # V_x via central differences (one-sided at boundary)
            if i < M:
                V_x = (V[i + 1, j] - V[i - 1, j]) / (2.0 * dx)
            else:
                V_x = (V[i, j] - V[i - 1, j]) / dx

            v_opt, H_min = _find_optimal_v(
                V_x, x_i, eta, alpha_pow, lam, sigma, dt, n_candidates
            )
            v_star[i, j] = v_opt
            V[i, j - 1] = V[i, j] + dt * H_min

        # Boundary
        V[0, j - 1] = 0.0
        v_star[0, j] = 0.0

    # v_star at t=0
    for i in range(1, M + 1):
        x_i = x[i]
        if i < M:
            V_x = (V[i + 1, 0] - V[i - 1, 0]) / (2.0 * dx)
        else:
            V_x = (V[i, 0] - V[i - 1, 0]) / dx
        v_star[i, 0], _ = _find_optimal_v(
            V_x, x_i, eta, alpha_pow, lam, sigma, dt, n_candidates
        )

    return grid, V, v_star


def extract_optimal_trajectory(
    grid: ExecutionGrid,
    v_star: np.ndarray,
    params: ACParams,
) -> np.ndarray:
    """Extract the optimal inventory path from the optimal control field.

    Starting from x(0) = X0, integrate forward using the optimal control:
        x(t_{k+1}) = x(t_k) - v*(x(t_k), t_k) * dt

    Uses linear interpolation on v_star to handle non-grid-aligned x values.

    Parameters
    ----------
    grid : ExecutionGrid
    v_star : np.ndarray, shape (M+1, N+1)
    params : ACParams

    Returns
    -------
    np.ndarray, shape (N+1,)
        Optimal inventory trajectory x*(t).
    """
    N = grid.N
    dt = grid.dt
    x_grid = grid.x_grid
    dx = grid.dx

    trajectory = np.zeros(N + 1)
    trajectory[0] = params.X0

    for k in range(N):
        x_curr = trajectory[k]

        # Linear interpolation of v_star at current x
        idx = x_curr / dx
        i_lo = int(np.floor(idx))
        i_lo = max(0, min(i_lo, grid.M - 1))
        i_hi = i_lo + 1

        w = (x_curr - x_grid[i_lo]) / dx if dx > 0 else 0.0
        w = max(0.0, min(1.0, w))

        v = (1 - w) * v_star[i_lo, k] + w * v_star[i_hi, k]

        trajectory[k + 1] = max(0.0, x_curr - v * dt)

    return trajectory


def analytical_value_function(
    params: ACParams,
    x: np.ndarray,
    t: float,
) -> np.ndarray:
    """Analytical value function for validation (linear impact only).

    V(x, t) = eta * kappa * coth(kappa*(T-t)) * x^2

    Parameters
    ----------
    params : ACParams
    x : np.ndarray
        Inventory grid.
    t : float
        Current time.

    Returns
    -------
    np.ndarray
        V(x, t) for each x.
    """
    kappa = params.kappa
    tau = params.T - t
    if tau < 1e-15:
        return np.full_like(x, 1e8) * x**2
    return params.eta * kappa / np.tanh(kappa * tau) * x**2
