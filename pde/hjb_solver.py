"""Hamilton-Jacobi-Bellman PDE solver for optimal execution.

For the value function V(x, t) = min expected cost-to-go from state (x, t):

    V_t + min_v { eta*v^2 + lam*S0^2*sigma^2*x^2 - v*V_x } = 0

For LINEAR impact (alpha=1), V(x,t) = alpha(t)*x^2 where alpha(t) satisfies
the Riccati ODE (stable, exact):

    alpha'(t) = alpha(t)^2/eta - lam*S0^2*sigma^2
    alpha(T) = terminal_penalty

Analytical solution: alpha(t) = eta * kappa * coth(kappa*(T-t))
where kappa = sqrt(lam*S0^2*sigma^2/eta).

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
        d(alpha)/d(tau) = -alpha^2/eta + lam*S0^2*sigma^2
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

    S0 = params.S0
    # Solve Riccati ODE backward: tau = T - t, alpha(tau=0) = penalty
    # d(alpha)/d(tau) = -alpha^2/eta + lam*S0^2*sigma^2
    def riccati_rhs(tau, alpha):
        return -alpha**2 / eta + lam * S0**2 * sigma**2

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
    """Solve HJB via policy iteration / Howard's algorithm (general nonlinear impact).

    For nonlinear temporary impact h(v) = eta*|v|^alpha*sign(v), the HJB is:

        V_t + min_v { eta*|v|^(alpha+1) + lam*S0^2*sigma^2*x^2 - v*V_x } = 0

    Policy iteration alternates between:
      1. Policy evaluation: for fixed v*, solve the LINEAR PDE
         V_t + eta*|v*|^(alpha+1) + lam*S0^2*sigma^2*x^2 - v*V_x = 0
         backward in time using implicit Euler with upwind differencing.
      2. Policy improvement: update v* by minimizing the Hamiltonian given V.
         FOC: v* = (V_x / (eta*(alpha+1)))^(1/alpha) for v >= 0, V_x > 0.

    Implicit Euler with upwind differencing produces a diagonally dominant
    tridiagonal system (diagonal >= 1 + dt*v/dx > 0), guaranteeing unconditional
    stability and non-singularity regardless of dt, penalty size, or alpha.

    Convergence is typically quadratic (Howard 1960).
    """
    grid = build_grid(params, M)
    N = grid.N
    dt = grid.dt
    dx = grid.dx
    x = grid.x_grid
    eta = params.eta
    ap = params.alpha  # alpha exponent
    lam = params.lam
    sigma = params.sigma

    # We solve for unknowns at indices 1..M (i=0 is boundary V=0)
    n_int = M

    V = np.zeros((M + 1, N + 1))
    v_star = np.zeros((M + 1, N + 1))

    # Terminal condition
    V[:, N] = terminal_penalty * x**2

    # --- Initialize policy: TWAP-like v*(x, t) = x / T ---
    for j in range(N + 1):
        v_star[:, j] = x / params.T
    v_star[0, :] = 0.0

    # --- Helper: compute V_x using backward (upwind) differences ---
    def compute_Vx(V_col):
        """dV/dx via backward differences at indices 1..M.

        Upwind scheme for v >= 0: information flows from higher to lower x.
        """
        Vx = np.zeros(M + 1)
        Vx[1:M + 1] = (V_col[1:M + 1] - V_col[0:M]) / dx
        return Vx

    # --- Helper: optimal control from V_x ---
    def optimal_control(Vx, x_arr):
        """v* = (V_x / (eta*(alpha+1)))^(1/alpha), v >= 0.

        From FOC of H(v) = eta*v^(alpha+1) - v*V_x:
            dH/dv = eta*(alpha+1)*v^alpha - V_x = 0
        """
        v_opt = np.zeros_like(Vx)
        pos = Vx > 0
        v_opt[pos] = (np.maximum(Vx[pos] - params.gamma * (params.X0 - x_arr[pos]), 1e-30) / (eta * (ap + 1.0))) ** (1.0 / ap)
        # Cap at x/dt — cannot sell more than remaining inventory per step
        v_max = np.maximum(x_arr / dt, 0.0)
        v_opt = np.minimum(v_opt, v_max)
        v_opt = np.maximum(v_opt, 0.0)
        return v_opt

    # --- Policy iteration (Howard's algorithm) ---
    max_policy_iter = 50
    tol_policy = 1e-8

    for iteration in range(max_policy_iter):
        V_old = V.copy()

        # --- Policy evaluation: implicit Euler backward sweep ---
        #
        # PDE: V_t + source(x,t) - v*(x,t)*V_x = 0
        #   where source = eta*|v*|^(alpha+1) + lam*S0^2*sigma^2*x^2
        #
        # Backward step from t_{j+1} to t_j (implicit in V^j):
        #   (V^j - V^{j+1}) / dt = v*(x, t_j) * V_x^j - source(x, t_j)
        #
        # With upwind backward difference V_x^j_i = (V^j_i - V^j_{i-1})/dx:
        #   V^j_i - V^{j+1}_i = dt * v_i * (V^j_i - V^j_{i-1})/dx - dt * src_i
        #
        # Rearranging:
        #   V^j_i * (1 + dt*v_i/dx) - V^j_{i-1} * (dt*v_i/dx) = V^{j+1}_i - dt*src_i
        #
        # This is a lower-bidiagonal system (trivially solved by forward substitution)
        # with diagonal = 1 + dt*v_i/dx >= 1 (always positive, always non-singular).

        V[:, N] = terminal_penalty * x**2  # re-apply terminal

        for j in range(N - 1, -1, -1):
            # Policy and source at time step j (interior points 1..M)
            v_j = v_star[1:M + 1, j]  # shape (M,)
            src_j = eta * np.abs(v_j) ** (ap + 1.0) + params.gamma * v_j * (params.X0 - x[1:M + 1]) + lam * params.S0**2 * sigma**2 * x[1:M + 1]**2

            # Courant numbers (always >= 0)
            c = dt * v_j / dx  # shape (M,)

            # RHS: V^{j+1} at interior - dt * source
            rhs = V[1:M + 1, j + 1] - dt * src_j

            # Forward substitution for lower-bidiagonal system:
            #   (1 + c_i) * V_i - c_i * V_{i-1} = rhs_i
            # with V_0 = 0 (boundary)
            V_prev = 0.0  # V[0, j] = 0 boundary
            for i in range(n_int):
                V_val = (rhs[i] + c[i] * V_prev) / (1.0 + c[i])
                V[i + 1, j] = V_val
                V_prev = V_val

            V[0, j] = 0.0  # boundary

        # --- Policy improvement: update v* from new V ---
        v_star_new = np.zeros_like(v_star)
        for j in range(N + 1):
            Vx = compute_Vx(V[:, j])
            v_star_new[:, j] = optimal_control(Vx, x)
        v_star_new[0, :] = 0.0

        # Check convergence (sup-norm relative change in V)
        V_norm = np.max(np.abs(V))
        if V_norm > 0:
            rel_change = np.max(np.abs(V - V_old)) / V_norm
        else:
            rel_change = 0.0

        v_star = v_star_new

        if rel_change < tol_policy:
            break

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
