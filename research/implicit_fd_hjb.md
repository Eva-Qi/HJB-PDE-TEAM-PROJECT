# Implicit Finite Difference / Crank-Nicolson for HJB Optimal Execution

**MF796 Course Project — Research Note**
*Generated: 2026-03-22*

---

## 1. Problem Statement

### 1.1 The HJB PDE for Nonlinear Almgren-Chriss

The value function `V(t, x)` for the Almgren-Chriss optimal execution problem satisfies:

```
V_t + min_{v >= 0} { eta * |v|^(alpha+1) + lambda * sigma^2 * x^2 - v * V_x } = 0
```

with terminal condition:

```
V(T, x) = 0    for x = 0  (liquidation complete, no residual penalty)
V(T, x) = M    for x > 0  (large penalty for incomplete liquidation)
```

where:
- `x` = remaining inventory (state variable, x in [0, X0])
- `t` = time (backward from T to 0)
- `v` = trading rate control (v >= 0, we are selling)
- `eta` = temporary impact coefficient
- `alpha` = impact exponent (alpha=1: linear, alpha<1: concave, alpha>1: convex)
- `lambda` = risk aversion
- `sigma` = volatility

### 1.2 The Optimal Control in Closed Form (Pointwise Minimizer)

For fixed `(t, x)`, the Hamiltonian minimization over `v` gives the first-order condition:

```
(alpha+1) * eta * v^alpha = V_x
```

Solving:

```
v*(t, x) = ( V_x(t, x) / ((alpha+1) * eta) )^(1/alpha)
```

This is only valid when `V_x >= 0` (which holds because selling faster reduces future inventory exposure). When `V_x < 0`, the optimal rate is `v* = 0` (stop trading).

### 1.3 Why Explicit Finite Differences Fail for alpha != 1

**For alpha = 1 (linear impact):** The HJB reduces to a Riccati ODE along characteristics. The explicit FD is stable because the PDE is effectively first-order advection with a known velocity field; the Riccati ODE can be solved directly.

**For alpha != 1 (nonlinear impact):**

1. **The optimal velocity `v*(t,x)` can be arbitrarily large near `x=0` or when `V_x` is large.** In explicit Euler, the advection term `-v* * V_x` contributes `dt * v* * (V_x / dx)`. When `v*` is large (e.g., near terminal time with residual inventory), the local Courant number `dt * v* / dx` can exceed 1 and the scheme blows up.

2. **The CFL condition for explicit upwinding requires:**
   ```
   dt <= dx / max_v*(t,x)
   ```
   But `v*` is data-dependent and can spike to infinity near `t = T` with any remaining inventory. This forces either astronomically small `dt` or the scheme diverges.

3. **The power-law nonlinearity amplifies errors.** For `alpha < 1` (sublinear impact, realistic for large trades), `v^(alpha+1)` is concave in `v`, and small perturbations in `V_x` produce large perturbations in the optimal control, which then feeds back into the explicit update — causing oscillation and blowup.

4. **Verification via existing code:** The project's `pde/grid.py` already notes "No CFL issue for the basic 1D HJB" — this is true only for the linear case where the Riccati ODE sidesteps FD entirely. For nonlinear alpha, the FD solver must confront the CFL problem directly.

**Conclusion:** For alpha != 1, the explicit FD solver requires dt proportional to dx / v_max. Since v_max is problem-dependent and can diverge, the explicit scheme is impractical. An implicit or semi-implicit scheme is mandatory.

---

## 2. Approach Options

### 2.1 Fully Implicit (Backward Euler) with Nonlinear Solver

Discretize time backward: given `V^{n+1}` at time `t_{n+1}`, solve for `V^n` at time `t_n`:

```
(V^n - V^{n+1}) / dt + min_v { eta*|v|^(alpha+1) + lambda*sigma^2*x^2 - v*(V^n_x) } = 0
```

The optimal control `v*` now depends on `V^n_x` (the unknown), making this a nonlinear system. Solving it requires a nonlinear iteration at each time step:

- **Newton's method**: Linearize around current iterate. Converges quadratically if good initial guess. Can fail for non-smooth Hamiltonians.
- **Fixed-point iteration**: Freeze `v*` from previous iterate, solve linear system. Converges if contraction.
- **Policy iteration (Howard's algorithm)**: See Section 2.3. Generally preferred.

**Pros:** Unconditionally stable (arbitrary dt). Consistent. Monotone if upwinding is used.
**Cons:** Must solve a nonlinear system at each time step. Higher per-step cost.

### 2.2 Crank-Nicolson (CN) Time Stepping

Average the spatial operator between `t_n` and `t_{n+1}`:

```
(V^n - V^{n+1}) / dt + 0.5 * H(V^n_x, x) + 0.5 * H(V^{n+1}_x, x) = 0
```

where `H(p, x) = min_v { eta*|v|^(alpha+1) + lambda*sigma^2*x^2 - v*p }`.

**Pros:** Second-order in time (vs first-order for backward Euler). Achieves O(dt^2 + dx^2) convergence.
**Cons:**

1. **Crank-Nicolson is NOT monotone.** The Barles-Souganidis theorem (1991) requires monotonicity for guaranteed convergence to the viscosity solution. CN can produce spurious oscillations near non-smooth solutions (e.g., the terminal penalty kink).

2. **Rannacher correction:** Apply 2-4 fully implicit steps at the initial time (backward from t=T) before switching to CN. This eliminates oscillations near the terminal condition. Also known as "Crank-Nicolson-Rannacher" (CNR) timestepping.

3. **For this specific HJB:** The terminal condition has a corner (V(T,x>0) = large constant). CN will create spurious Gibbs-like oscillations near x=0 at t=T unless smoothed or corrected.

**Recommendation for HJB:** Use CN only with Rannacher correction; otherwise use fully implicit.

### 2.3 Policy Iteration (Howard's Algorithm) — RECOMMENDED INNER SOLVER

Policy iteration decouples the nonlinear minimization from the linear PDE solve at each time step. This is the standard approach in computational finance for HJB PDEs (Forsyth & Labahn, 2007).

**Algorithm for one backward time step:**

```
Given V^{n+1} (known), solve for V^n:

1. Initialize: v^(0)(x_j) = v_prev(x_j)  [control from previous time step]

2. POLICY EVALUATION:
   With v^(k) fixed, solve the LINEAR system:
   (V^n - V^{n+1}) / dt + eta*|v^(k)|^(alpha+1) + lambda*sigma^2*x_j^2
                        - v^(k) * (D_x V^n)_j = 0
   where D_x is the upwind finite difference of V^n.

3. POLICY IMPROVEMENT:
   v^(k+1)(x_j) = ( (D_x V^n)_j / ((alpha+1)*eta) )^(1/alpha)
   (clamp to [0, v_max])

4. Convergence check: if ||v^(k+1) - v^(k)||_inf < tol, STOP.
   Otherwise k <- k+1, go to step 2.
```

**Why this works:**
- Step 2 is a linear tridiagonal system (upwind differencing of the advection term). Solved in O(M) by scipy.linalg.solve_banded or scipy.sparse.linalg.spsolve.
- Step 3 is a pointwise closed-form update (no optimization needed).
- The sequence V^(k) is monotone decreasing (improvements at every step).
- Convergence in finitely many steps for the discrete problem (Forsyth & Labahn, 2007).
- Typically 3-10 iterations per time step for well-behaved problems.

### 2.4 Semi-Implicit Scheme

Treat the advection term partially explicitly (using V^{n+1}) and partially implicitly:

```
(V^n - V^{n+1}) / dt - v*(D_x V^{n+1}) + eta*|v*|^(alpha+1) + lambda*sigma^2*x^2 = 0
```

where `v* = (D_x V^{n+1} / ((alpha+1)*eta))^(1/alpha)` is computed explicitly from the known `V^{n+1}`.

**Pros:** Only one linear solve per time step. No inner iteration. Simple to implement.
**Cons:**
- Only first-order in time.
- Stability requires moderate dt (better than explicit but not unconditionally stable).
- The control v* may be stale (computed from previous time level), causing accuracy issues near the terminal condition.
- Not guaranteed monotone in general.

This approach is reasonable for quick prototyping or when speed matters over accuracy.

### 2.5 Semi-Lagrangian Scheme (Forsyth's Approach for Optimal Execution)

Forsyth (2011, "A HJB Approach to Optimal Trade Execution") uses a semi-Lagrangian (SL) discretization that is:
- Essentially form-independent of the price impact function.
- Monotone and consistent by construction.
- Unconditionally stable.
- Converges to the viscosity solution under a comparison principle.

The SL method traces characteristics backward:
```
V(t_n, x) = min_v { dt * (eta*|v|^(alpha+1) + lambda*sigma^2*x^2)
                    + V(t_{n+1}, x - v*dt) }
```

The term `V(t_{n+1}, x - v*dt)` is interpolated from the grid. This reduces the PDE to a sequence of 1D optimization problems at each grid point (independent for each `x_j`).

**Pros:** Clean, robust, directly handles nonlinear impact. Proven to converge for this exact problem.
**Cons:** Requires interpolation (cubic spline recommended for accuracy). The optimization over `v` at each node is a 1D scalar minimization — use scipy.optimize.minimize_scalar or the closed-form expression when it exists.

---

## 3. Recommended Approach

### Primary Recommendation: Fully Implicit + Policy Iteration (Howard)

For the MF796 project's nonlinear HJB with `alpha != 1`, the **fully implicit backward Euler time stepping with Howard's policy iteration** is the recommended approach. Reasons:

1. **Unconditionally stable**: No CFL constraint on dt. Use the same dt grid as the linear case.

2. **Monotone**: Upwind differencing on the advection term `-v * V_x` ensures the scheme satisfies the Barles-Souganidis monotonicity condition, guaranteeing convergence to the viscosity solution.

3. **Fast inner loop**: Each policy iteration step solves a tridiagonal linear system in O(M). Convergence in ~5 iterations per time step means total cost is O(5 * M * N) — same order as the explicit scheme but now stable.

4. **Clean convergence theory**: Forsyth & Labahn (2007) prove convergence of the policy iteration for fully implicit HJB discretizations in finance. No pathological cases for this scalar control problem.

5. **Directly comparable to existing Riccati solver**: For alpha=1, the policy iteration will recover the same trajectory as the Riccati ODE (useful for validation).

### Secondary Option: Semi-Lagrangian (if policy iteration is too complex)

If the full policy iteration implementation is too involved, the semi-Lagrangian approach (Section 2.5) is the simplest alternative that is still provably convergent. It requires a 1D scalar minimize at each grid point, which is cheap and uses scipy.optimize.minimize_scalar.

### Avoid Pure Crank-Nicolson Without Rannacher Fix

The terminal condition `V(T, x>0) = large penalty` creates a kink at `x=0`. Pure CN will oscillate here and pollute the entire solution. Use CN only with 2-4 Rannacher steps at the start (stepping backward from t=T).

---

## 4. Implementation Outline

### 4.1 Grid and Setup

```python
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Grid
M = 400        # inventory grid points (increased from 200 for nonlinear)
N = 100        # time steps
X0 = params.X0
T  = params.T
dx = X0 / M
dt = T / N

x = np.linspace(0, X0, M + 1)   # shape (M+1,)
t = np.linspace(0, T,  N + 1)   # shape (N+1,)
```

### 4.2 Terminal Condition

```python
PENALTY = 1e12    # large number: cost of not liquidating

V = np.zeros(M + 1)          # V[j] = V(T, x_j)
V[1:] = PENALTY              # x > 0 at terminal time: unacceptable
V[0] = 0.0                   # x = 0: fully liquidated, zero cost
```

### 4.3 Fully Implicit Backward Step with Howard's Policy Iteration

```python
def optimal_v(Vx: np.ndarray, eta: float, alpha: float) -> np.ndarray:
    """Pointwise optimal trading rate from first-order condition.

    v*(x) = (Vx / ((alpha+1)*eta))^(1/alpha)
    Clamped to [0, v_max].
    """
    v_max = 1e8   # physical upper bound on trading rate
    ratio = np.maximum(Vx, 0.0) / ((alpha + 1) * eta)
    return np.minimum(ratio ** (1.0 / alpha), v_max)


def upwind_Vx(V: np.ndarray, v: np.ndarray, dx: float) -> np.ndarray:
    """Upwind finite difference for V_x.

    Since v >= 0 and we are liquidating (V_x >= 0 expected),
    the advection direction is leftward (x decreases as we sell).
    Use backward difference: (V[j] - V[j-1]) / dx  for the interior.

    Convention:
      - v > 0 means we sell => inventory x decreases => characteristics
        move LEFT => upwind direction is RIGHT => BACKWARD difference in x.
    """
    Vx = np.zeros_like(V)
    # Interior: backward difference (upwind for leftward advection)
    Vx[1:] = (V[1:] - V[:-1]) / dx
    Vx[0]  = 0.0    # boundary: no selling at x=0
    return Vx


def build_implicit_system(V_next: np.ndarray, v: np.ndarray,
                           x: np.ndarray, params, dt: float, dx: float):
    """Build the linear system A * V_now = rhs for one backward time step.

    The implicit discretization of:
      (V^n - V^{n+1}) / dt + eta*v^(alpha+1) + lam*sigma^2*x^2 - v*(V^n_x) = 0

    With upwind: (V^n_x)_j = (V^n_j - V^n_{j-1}) / dx   [backward diff]

    Rearranging:
      V^n_j / dt - v_j * (V^n_j - V^n_{j-1}) / dx =
      V^{n+1}_j / dt - eta*v_j^(alpha+1) - lam*sigma^2*x_j^2

    => V^n_j * (1/dt - v_j/dx) + V^n_{j-1} * (v_j/dx) = RHS_j

    This is a bidiagonal (lower) system: only V_j and V_{j-1} appear.
    Special case: j=0 boundary: V^n_0 = 0 (already liquidated).
    """
    M = len(x) - 1
    eta   = params.eta
    alpha = params.alpha
    lam   = params.lam
    sigma = params.sigma

    # RHS
    source = eta * v**(alpha + 1) + lam * sigma**2 * x**2
    rhs = V_next / dt - source

    # Main diagonal: coefficient of V^n_j
    diag_main = np.ones(M + 1) / dt - v / dx   # shape (M+1,)
    # Sub-diagonal: coefficient of V^n_{j-1}  (only j=1..M)
    diag_sub  = v[1:] / dx                      # shape (M,)

    # Build sparse bidiagonal system
    A = sparse.diags(
        [diag_sub, diag_main],
        offsets=[-1, 0],
        shape=(M + 1, M + 1),
        format='csc'
    )

    # Enforce boundary: V(t, x=0) = 0
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1.0
    rhs[0]  = 0.0
    A = A.tocsc()

    return A, rhs


def howard_step(V_next: np.ndarray, x: np.ndarray, params,
                dt: float, dx: float,
                tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """One backward time step using Howard's policy iteration.

    Returns V_now (approximation of V at t_n given V at t_{n+1}).
    """
    V = V_next.copy()   # initial guess for V^n

    for iteration in range(max_iter):
        # Policy evaluation: compute current V_x and optimal v
        Vx = upwind_Vx(V, np.zeros_like(V), dx)  # use current V for Vx
        v  = optimal_v(Vx, params.eta, params.alpha)

        # Build and solve linear system
        A, rhs = build_implicit_system(V_next, v, x, params, dt, dx)
        V_new = spsolve(A, rhs)

        # Convergence check
        err = np.max(np.abs(V_new - V))
        V = V_new
        if err < tol:
            break

    return V


def solve_hjb_implicit(params, M: int = 400) -> tuple:
    """Full backward HJB solve using fully implicit + Howard iteration.

    Returns
    -------
    V_grid : np.ndarray, shape (N+1, M+1)
        Value function on the full grid.
    v_grid : np.ndarray, shape (N+1, M+1)
        Optimal trading rate on the full grid.
    x : np.ndarray, shape (M+1,)
    t : np.ndarray, shape (N+1,)
    """
    X0 = params.X0
    T  = params.T
    N  = params.N
    dx = X0 / M
    dt = T / N
    x  = np.linspace(0, X0, M + 1)
    t  = np.linspace(0, T,  N + 1)

    # Terminal condition
    PENALTY = 1e10
    V = np.zeros(M + 1)
    V[1:] = PENALTY

    V_grid = np.zeros((N + 1, M + 1))
    v_grid = np.zeros((N + 1, M + 1))
    V_grid[N, :] = V

    # Backward sweep
    for n in range(N - 1, -1, -1):
        V = howard_step(V, x, params, dt, dx)
        # Extract optimal control at this time level
        Vx = upwind_Vx(V, np.zeros_like(V), dx)
        v  = optimal_v(Vx, params.eta, params.alpha)
        V_grid[n, :] = V
        v_grid[n, :] = v

    return V_grid, v_grid, x, t
```

### 4.4 Extracting the Optimal Trajectory

```python
def simulate_optimal_path(V_grid, v_grid, x, t, params):
    """Simulate the optimal inventory path starting from X0.

    At each time step, interpolate the optimal control from v_grid.
    """
    from scipy.interpolate import interp1d

    N = len(t) - 1
    x_path = np.zeros(N + 1)
    v_path = np.zeros(N + 1)
    x_path[0] = params.X0

    for n in range(N):
        # Interpolate v*(t_n, x_current) from grid
        v_interp = interp1d(x, v_grid[n, :], kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        v_now = float(v_interp(x_path[n]))
        v_now = max(0.0, v_now)   # no buying
        v_path[n] = v_now

        # Euler step: x decreases at rate v
        dt = params.dt
        x_path[n + 1] = max(0.0, x_path[n] - v_now * dt)

    return x_path, v_path
```

### 4.5 Alternative: Semi-Lagrangian (Simpler but Less Accurate Near Kinks)

```python
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def sl_step(V_next: np.ndarray, x: np.ndarray, params, dt: float) -> np.ndarray:
    """One backward step via semi-Lagrangian scheme.

    V(t_n, x_j) = min_{v>=0} { dt*(eta*v^(alpha+1) + lam*sig^2*x_j^2)
                                + V(t_{n+1}, x_j - v*dt) }
    """
    X0    = params.X0
    eta   = params.eta
    alpha = params.alpha
    lam   = params.lam
    sigma = params.sigma

    cs = CubicSpline(x, V_next, extrapolate=True)
    V_now = np.zeros_like(x)

    for j, xj in enumerate(x):
        if xj == 0.0:
            V_now[j] = 0.0
            continue

        def cost(v):
            if v < 0:
                return 1e15
            x_dep = xj - v * dt
            x_dep = np.clip(x_dep, 0, X0)
            return dt * (eta * v**(alpha + 1) + lam * sigma**2 * xj**2) + cs(x_dep)

        # Use closed-form v* as starting bracket
        # v* = (V_x / ((alpha+1)*eta))^(1/alpha); rough estimate:
        v_guess = (V_next[j] / (dt * (alpha + 1) * eta + 1e-12)) ** (1.0 / alpha)
        v_guess = np.clip(v_guess, 0, xj / dt)

        res = minimize_scalar(cost, bounds=(0, xj / dt + 1e-8),
                              method='bounded',
                              options={'xatol': 1e-10})
        V_now[j] = res.fun

    return V_now
```

---

## 5. Key References

### Foundational Papers

1. **Forsyth, P. A. & Labahn, G. (2007)**
   "Numerical Methods for Controlled Hamilton-Jacobi-Bellman PDEs in Finance"
   *Journal of Computational Finance*, 11(2), 1-44.
   [PDF](https://cs.uwaterloo.ca/~paforsyt/hjb.pdf)
   - THE reference for HJB in finance. Proves convergence of policy iteration for fully implicit upwind schemes. Covers monotonicity, consistency, viscosity solution convergence.

2. **Forsyth, P. A. (2011)**
   "A Hamilton-Jacobi-Bellman Approach to Optimal Trade Execution"
   *Applied Numerical Mathematics*, 61(2), 241-265.
   [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168927410001789)
   - Applies Forsyth-Labahn framework specifically to optimal execution. Uses semi-Lagrangian scheme. Works for any price impact function. DIRECTLY RELEVANT.

3. **Barles, G. & Souganidis, P. E. (1991)**
   "Convergence of Approximation Schemes for Fully Nonlinear Second-Order Equations"
   *Asymptotic Analysis*, 4(3), 271-283.
   [PDF via Moll](https://benjaminmoll.com/wp-content/uploads/2021/04/barles-souganidis.pdf)
   - Proves: monotone + consistent + stable => convergence to viscosity solution.
   - The theoretical foundation for any HJB finite difference scheme.

4. **Almgren, R. & Chriss, N. (2001)**
   "Optimal Execution of Portfolio Transactions"
   *Journal of Risk*, 3(2), 5-39.
   [PDF](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
   - Original paper. Linear impact, closed-form solution.

5. **Forsyth, P. A. & Vetzal, K. R. (2012)**
   "Optimal Asset Allocation for Retirement Saving: Deterministic vs. Stochastic Policies"
   Related: same group's work on policy iteration for HJB in portfolio optimization.

### Penalty Method Papers (Alternative to Policy Iteration)

6. **Reisinger, C. & Rotaetxe Arto, J. H. (2017)**
   "A Penalty Method for the Numerical Solution of Hamilton-Jacobi-Bellman (HJB) Equations in Finance"
   *SIAM Journal on Numerical Analysis*, 49(1), 213-231.
   [SIAM](https://epubs.siam.org/doi/10.1137/100797606)
   - Penalty approximation O(1/rho) per step; iterative solver converges in finite steps.

7. **Smears, I. & Suli, E. (2013)**
   "Discontinuous Galerkin Finite Element Approximation of Nondivergence Form Elliptic Equations with Cordès Coefficients"
   Related monotone scheme work.

### Lecture Notes and Tutorials

8. **Moll, B. (2019)**
   "Lecture 3: Hamilton-Jacobi-Bellman Equations"
   Princeton ECO 521 Notes.
   [PDF](https://benjaminmoll.com/wp-content/uploads/2019/07/Lecture3_ECO521.pdf)
   - Clear derivation of upwind implicit scheme for HJB. Economics context but immediately applicable.

9. **Forsyth, P. A.**
   "Lecture 1: Numerical Methods for Hamilton-Jacobi-Bellman Equations in Finance"
   Amsterdam Winter School Slides.
   [PDF](https://staff.fnwi.uva.nl/p.j.c.spreij/winterschool/9forsythslides.pdf)
   - Slides covering policy iteration steps with worked examples.

10. **Achdou, Y., Han, J., Lasry, J.-M., Lions, P.-L., & Moll, B. (2022)**
    "Income and Wealth Distribution in Macroeconomics"
    *Review of Economic Studies*, 89(1), 45-86.
    Related appendix: numerical methods for HJB via upwind implicit FD.
    [PDF appendix](https://benjaminmoll.com/wp-content/uploads/2020/02/HACT_Numerical_Appendix.pdf)

### Code Repositories

11. **anriseth/HJBSolver.jl** (Julia)
    [GitHub](https://github.com/anriseth/HJBSolver.jl)
    General HJB solver using policy iteration + implicit upwind.

12. **joshuapjacob/almgren-chriss-optimal-execution** (Python)
    [GitHub](https://github.com/joshuapjacob/almgren-chriss-optimal-execution)
    Linear impact only; useful for reference implementation structure.

13. **fediskhakov/CompEcon** (Python/Jupyter)
    Policy iteration notebook:
    [GitHub](https://github.com/fediskhakov/CompEcon/blob/main/43_policy_iter.ipynb)

---

## 6. Gotchas and Pitfalls

### 6.1 Upwinding Direction Must Match Sign of v

The advection term is `-v * V_x`. Since `v >= 0` (selling only), the advection moves `x` leftward (decreasing). The upwind direction for a leftward advection is the **backward difference**:

```
D_x^- V_j = (V_j - V_{j-1}) / dx
```

Using forward differences instead will introduce a spurious anti-diffusion and destabilize even the implicit scheme.

**If v can be negative (buying back):** Must use a flux-splitting or conditional upwinding:
```
D_x V_j = v_j^+ * (V_j - V_{j-1})/dx + v_j^- * (V_{j+1} - V_j)/dx
where v^+ = max(v,0), v^- = min(v,0)
```
For pure liquidation (no buying), `v >= 0` always, so backward difference suffices.

### 6.2 Terminal Condition Penalty Must Be Consistent

The penalty `V(T, x>0) = M_penalty` effectively enforces `x(T) = 0`. If the penalty is too small, the solver may prefer non-zero terminal inventory. If too large, it can cause ill-conditioning of the linear system at the first backward step.

**Practical range:** `M_penalty = 1e6 * eta * X0^(alpha+1)` — scale with the problem's natural cost units.

**Soft terminal condition alternative:** Instead of a hard penalty, use a smooth terminal cost:
```
V(T, x) = c * x^2
```
where `c` is chosen large enough. This avoids the kink and makes CN more accurate without Rannacher.

### 6.3 CFL in Policy Iteration Does NOT Disappear Completely

The implicit scheme is unconditionally stable for the linear system at each policy iteration step. However, **accuracy** still requires that `v* * dt / dx` is not too large (otherwise the numerical diffusion dominates). A practical guideline:

```
dt / dx * max_v_expected <= 10   (accuracy condition, not stability)
```

Choose `dx` and `dt` such that this holds at the expected trading rates. For typical AC parameters, `max_v ~ X0 / T` (TWAP rate), so `dt/dx ~ T/N * M/X0 = M/(X0*N)`. With `M=400, N=100, X0=1e6, T=0.25`, this is fine.

### 6.4 Policy Iteration May Stall on First Step

At `t = T - dt` (first backward step), `V_next` has a large kink. The initial control guess `v^(0)` based on `Vx` of a kinky function can overshoot. Fix:

- Clamp `v^(k)` to `[0, x_j/dt]` at every iteration (cannot sell more than remaining inventory in one step).
- Use 2-3 extra Howard iterations at the first backward step.
- Initialize `v^(0)` with a physical guess: `v^(0)(x_j) = x_j / (T - t_n)` (TWAP rate).

### 6.5 alpha < 1 (Concave Impact) Causes Infinite Optimal Rate

For `alpha < 1`, the first-order condition `v* = (V_x / ((alpha+1)*eta))^(1/alpha)` grows faster than `V_x` when `V_x` is large (because `1/alpha > 1`). Near the terminal condition where `V_x` is effectively infinite, `v*` diverges.

**Fixes:**
1. Clamp `v* <= x_j / dt` (physical bound: cannot deplete more inventory than available).
2. Add a small regularization: `eta_reg * v^2 / 2` term (penalizes excessive rate).
3. Use a soft terminal condition (reduces the spike in `V_x`).

### 6.6 Boundary Conditions at x = X0

At `x = X0` (maximum inventory), there is an incoming boundary (no inflow from higher inventory). The linear system should just use the interior equation without modification at `j = M`. The solution there will naturally reflect that the problem is defined on `[0, X0]`.

If the trader might start with more than `X0` inventory, extend the grid. Otherwise, the solver assumes `V(t, x > X0)` is not needed.

### 6.7 Convergence Tolerance for Howard Iteration

Too tight a tolerance (`tol = 1e-12`) wastes time on already-converged iterations. Too loose (`tol = 1e-3`) propagates policy errors backward in time and corrupts the value function.

**Recommended:** `tol = max(1e-8, 1e-6 * |V_next|_inf / (params.eta * params.X0^(alpha+1)))` — relative tolerance scaled to the problem magnitude.

### 6.8 Crank-Nicolson Spurious Oscillations Near Terminal Condition

If CN is used without Rannacher correction, the oscillations near `t=T, x~0` will have amplitude proportional to the penalty jump and will propagate backward in time, corrupting the entire solution. Always apply 2-4 fully implicit steps first:

```python
# Rannacher correction: first 4 steps are fully implicit
for n in range(N-1, N-5, -1):
    V = howard_step(V, ...)

# Remaining steps: Crank-Nicolson
for n in range(N-5, -1, -1):
    V = cn_step(V, ...)
```

### 6.9 Validation Against alpha = 1 (Riccati Solution)

Before trusting the nonlinear solver, validate it against the closed-form Riccati solution for `alpha = 1`:

1. Set `alpha = 1` in the implicit solver.
2. Compare `v*(t, X0 * sinh(kappa*(T-t)) / sinh(kappa*T))` from the grid against the closed-form.
3. Expect O(dx + dt) convergence for backward Euler; O(dx + dt^2) for CN.

If the policy iteration does not recover the linear case, the upwinding direction or boundary condition implementation is wrong.

### 6.10 Scipy Sparse Matrix Format

Use `csc` format for `spsolve` (column operations are efficient). Use `lil` or `coo` format for construction, then convert:

```python
A_csc = A.tocsc()
V_new = spsolve(A_csc, rhs)
```

For the bidiagonal structure of the upwind implicit system, `scipy.linalg.solve_banded` with `(1, 0)` bandwidth is slightly faster than sparse direct:

```python
from scipy.linalg import solve_banded
# ab[0] = sub-diagonal (shifted), ab[1] = main diagonal
ab = np.zeros((2, M + 1))
ab[0, 1:] = diag_sub      # sub-diagonal (j-1 coeff, stored at j)
ab[1, :]  = diag_main     # main diagonal
V_new = solve_banded((1, 0), ab, rhs)
```

---

## 7. Summary Decision Table

| Scheme | Stability | Order (time) | Monotone | Cost/step | Recommendation |
|---|---|---|---|---|---|
| Explicit Euler | Conditional (CFL) | O(dt) | Yes | O(M) | **Avoid for alpha!=1** |
| Backward Euler + Howard | Unconditional | O(dt) | Yes | O(k * M) | **USE THIS** |
| Crank-Nicolson (pure) | Unconditional | O(dt^2) | No | O(k * M) | Avoid (oscillates) |
| CN + Rannacher (4 steps) | Unconditional | O(dt^2) | Approx | O(k * M) | OK if high accuracy needed |
| Semi-implicit | Conditional (mild) | O(dt) | Partial | O(M) | Prototype only |
| Semi-Lagrangian | Unconditional | O(dt) | Yes | O(M * scalar opt) | Alternative to Howard |

**Legend:** k = Howard iterations per step (typically 3-10)

---

*Research compiled from: Forsyth & Labahn (2007), Forsyth (2011), Barles & Souganidis (1991), Moll (2019 lecture notes), Reisinger & Rotaetxe Arto (2017), and related literature on numerical HJB methods in computational finance.*
