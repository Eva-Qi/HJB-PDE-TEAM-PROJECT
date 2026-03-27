# Gap Research: Heston 1D PDE Implementation, Sensitivity Analysis, PDE–MC Cross-Validation

**Date:** 2026-03-27
**Scope:** Implementation gaps not covered by existing theoretical derivation.
**What is NOT repeated here:** separable ansatz derivation, α≠1 failure analysis, ADI for full 2D, Heston characteristic function/FFT, Feller condition basics.

---

## 1. Step-by-Step Implementation of the 1D PDE for A(v,t)

### 1.1 The PDE (reference)

```
A_t − A²/η + λv + κ(θ−v)A_v + ½ξ²v·A_vv = 0
```

Terminal condition: A(v, T) = 0 (zero cost at horizon).
The nonlinear term is `−A²/η` (Riccati-type quadratic).

---

### 1.2 Spatial Grid for the v Dimension

**Grid type:** Uniform grid is adequate for a 1D problem in v alone (no x dimension here).
Non-uniform grids improve accuracy in the full 2D Heston option-pricing PDE because of the discontinuity in delta at K. In the 1D A(v,t) problem there is no such kink, so a uniform grid on [0, v_max] is the practical default.

**v_max choice:** The literature on Heston PDE numerics (In 't Hout & Foulon 2010, Haentjens & In 't Hout 2012) uses v_max = 1.0 to 5.0 for typical equity parameters (ξ ≈ 0.3–0.6, θ ≈ 0.04). For a stochastic-volatility execution model, the variance process lives in the same range; v_max = 5·θ is a safe rule of thumb that keeps truncation error negligible. At v_max, the mean-reversion term κ(θ − v) is strongly negative, so A is nearly flat; a Dirichlet condition A(v_max, t) = 0 (or a linear extrapolation) works well.

**Recommended grid size:** 100–200 uniform points in v is sufficient for convergence to 3–4 significant figures. Time steps: 200–500 steps over [0, T]. Refine by doubling both and checking relative change < 0.1%.

---

### 1.3 Boundary Conditions

**At v = 0:**
The PDE is degenerate parabolic at v = 0 (the diffusion coefficient ½ξ²v vanishes). By the Fichera classification, no explicit boundary condition is needed when the convection term κ(θ − v) at v = 0 equals κθ > 0 (pointing inward). The PDE itself degenerates to an ODE:

```
A_t − A²/η + λ·0 + κθ·A_v = 0
```

In practice, simply apply the PDE at the first interior node (i = 0 treated as a boundary node using a one-sided finite difference for A_v). Many implementations use a Neumann condition ∂A/∂v = 0 at v = 0 as a convenient approximation when κθ is large and the Feller condition holds.

**At v = v_max:**
The diffusion is large but A ≈ 0 at this end (since the execution cost converges). Use Dirichlet A(v_max, t) = 0, which is consistent with the terminal condition and supported by exponential decay of A for large v.

---

### 1.4 Time Discretization: Backward Integration with Crank-Nicolson

The PDE is solved backward in time from t = T to t = 0. Introduce τ = T − t so the equation becomes:

```
A_τ + A²/η − λv − κ(θ−v)A_v − ½ξ²v·A_vv = 0
```

with A(v, 0) = 0 (initial condition in τ).

Apply Crank-Nicolson in τ: for each time step from τ^n to τ^{n+1} = τ^n + Δτ,

```
(A^{n+1} − A^n)/Δτ = ½ [ L(A^{n+1}) + L(A^n) ]
```

where L(A) = λv + κ(θ−v)A_v + ½ξ²v·A_vv − A²/η is the spatial-nonlinear operator.

The nonlinear term A²/η prevents direct solution of the resulting system. Two practical approaches:

---

### 1.5 Handling the Nonlinear A² Term

#### Option A: Picard / Lagged-Coefficient Linearization (recommended for simplicity)

Replace A² at time level n+1 with A^n · A^{n+1}:

```
A²/η  →  A^n · A^{n+1} / η
```

This makes the system linear in A^{n+1}. The error introduced is O(Δτ), but since Crank-Nicolson is already O(Δτ²) in the linear terms, this limits overall accuracy to O(Δτ). To restore second-order accuracy, do **two Picard sweeps per time step**:

1. Set A^{(0)} = A^n.
2. Solve the linearized system to get A^{(1)}.
3. Update A^{(0)} ← A^{(1)}, solve again to get A^{(2)}.
4. Accept A^{(2)} as A^{n+1}.

Convergence criterion: max|A^{(k+1)} − A^{(k)}| < 1e-8.

#### Option B: Newton Iteration per Time Step (higher cost, quadratic convergence)

At each time step, define residual F(A^{n+1}) = 0 and iterate Newton steps. The Jacobian is tridiagonal with an extra diagonal contribution −A^{n+1}/η from the nonlinear term. For a well-behaved problem, Newton converges in 3–5 iterations. This is overkill unless η is very small (making the nonlinear term stiff).

**Practical recommendation:** Use Picard with 2–3 sweeps. If |η| < 0.1 and the solution A becomes large, switch to Newton.

---

### 1.6 Spatial Discretization (Finite Differences)

On a uniform grid v_i = i·Δv, i = 0, 1, ..., N, use centered differences:

```
A_v  ≈  (A_{i+1} − A_{i-1}) / (2Δv)          [second-order]
A_vv ≈  (A_{i+1} − 2A_i + A_{i-1}) / (Δv²)   [second-order]
```

At i = 0, use a forward difference for A_v (one-sided, first-order):
```
A_v  ≈  (A_1 − A_0) / Δv
```

The resulting system at each time step is tridiagonal (or banded with Picard linearization). Use `scipy.linalg.solve_banded` or `numpy.linalg.solve` for small N, or `scipy.sparse.linalg.spsolve` for large N.

---

### 1.7 Python Code Sketch

```python
import numpy as np
from scipy.linalg import solve_banded

def solve_A_pde(kappa, theta, xi, eta, lam, T, N_v=150, N_t=300):
    """
    Solve A_t - A^2/eta + lam*v + kappa*(theta-v)*A_v + 0.5*xi^2*v*A_vv = 0
    backward from t=T to t=0, terminal condition A(v,T) = 0.
    Returns: v_grid (N_v+1,), A_grid (N_v+1, N_t+1)
    """
    v_max = max(5.0 * theta, 1.0)
    dv = v_max / N_v
    dt = T / N_t                         # forward tau step
    v = np.linspace(0, v_max, N_v + 1)  # v_0=0, v_N=v_max

    A = np.zeros(N_v + 1)               # terminal: A(v,T)=0

    def build_tridiag(A_prev):
        """Build tridiagonal (ab format for solve_banded) for Crank-Nicolson step."""
        N = N_v + 1
        lower = np.zeros(N)
        diag  = np.ones(N)
        upper = np.zeros(N)
        rhs   = np.zeros(N)

        for i in range(1, N - 1):
            vi = v[i]
            diff_coef  = 0.5 * xi**2 * vi
            conv_coef  = kappa * (theta - vi)

            # Spatial coefficients (centered differences)
            c_minus = diff_coef / dv**2 - conv_coef / (2 * dv)
            c_zero  = -2 * diff_coef / dv**2
            c_plus  = diff_coef / dv**2 + conv_coef / (2 * dv)

            # Picard linearization: A^2/eta -> A_prev[i] * A^{n+1}[i] / eta
            nl_coef = A_prev[i] / eta   # from nonlinear term, moves to LHS

            # LHS: (I - dt/2 * L_linear + dt/2 * nl_diag)
            lower[i] = -dt/2 * c_minus
            diag[i]  =  1 - dt/2 * c_zero + dt/2 * nl_coef
            upper[i] = -dt/2 * c_plus

            # RHS: (I + dt/2 * L_linear - dt/2 * nl_diag) * A_prev + dt*source
            rhs[i] = (A_prev[i]
                      + dt/2 * (c_minus*A_prev[i-1] + c_zero*A_prev[i] + c_plus*A_prev[i+1])
                      - dt/2 * nl_coef * A_prev[i]
                      + dt * lam * vi)

        # BC at v=0: Neumann A_v=0 → A[0]=A[1]
        diag[0] = 1; upper[0] = -1; rhs[0] = 0
        # BC at v=v_max: Dirichlet A=0
        diag[-1] = 1; lower[-1] = 0; rhs[-1] = 0

        # Pack into banded format (2 sub, 2 super for solve_banded)
        ab = np.zeros((3, N))
        ab[0, 1:] = upper[:-1]
        ab[1, :]  = diag
        ab[2, :-1] = lower[1:]
        return ab, rhs

    A_store = np.zeros((N_v + 1, N_t + 1))
    A_store[:, -1] = A  # terminal condition

    for step in range(N_t - 1, -1, -1):
        # Two Picard sweeps
        A_iter = A.copy()
        for _ in range(2):
            ab, rhs = build_tridiag(A_iter)
            A_iter = solve_banded((1, 1), ab, rhs)
        A = A_iter
        A_store[:, step] = A

    return v, A_store


# Retrieve A at t=0 (A_store[:, 0]) and interpolate at current variance v0
# Execution cost ≈ A(v0, 0) * x^2
```

**Notes:**
- The source term `lam * vi` enters the RHS at each time step.
- The Picard linearization makes the tridiagonal matrix time-step cost O(N_v), total O(N_v × N_t).
- For typical parameters (N_v=150, N_t=300), runtime is under 1 second in Python.

---

### 1.8 Grid Size Recommendations Summary

| Parameter | Minimum | Recommended |
|---|---|---|
| N_v (v grid points) | 50 | 150–200 |
| N_t (time steps) | 100 | 300–500 |
| v_max | 3θ | 5θ (or 5.0 if θ is small) |
| Picard sweeps/step | 1 | 2–3 |

Convergence check: double N_v and N_t independently, verify change in A(v_0, 0) < 0.1%.

---

## 2. Sensitivity Analysis Methodology

### 2.1 Parameters to Sweep

The model has four primary parameters: α (market impact exponent), η (liquidity/impact scale), λ (risk aversion), σ or ξ (volatility of variance). We also treat κ (mean reversion speed) and θ (long-run variance) as secondary.

Relevant parameter ranges based on the Almgren-Chriss literature and stochastic volatility calibration:

| Parameter | Low | Base | High | Physical meaning |
|---|---|---|---|---|
| α | 0.5 | 1.0 | 1.5 | Impact convexity (α=1 is linear) |
| η | 0.05 | 0.1 | 0.5 | Temporary impact coefficient |
| λ | 1e-5 | 1e-4 | 1e-3 | Risk aversion |
| ξ | 0.1 | 0.3 | 0.6 | Vol-of-vol |
| κ | 0.5 | 2.0 | 5.0 | Mean reversion speed |
| θ | 0.01 | 0.04 | 0.16 | Long-run variance |

These ranges cover realistic equity market microstructure values. The base case should match calibrated parameters from the project's data.

---

### 2.2 Analysis Methods: Which to Use

**One-at-a-time (OAT):**
Vary each parameter individually (low → high), hold all others fixed. Result is one value at each extreme per parameter. Produces the tornado chart directly. Fast: 2 × N_params model evaluations.
**Limitation:** cannot detect interaction effects.

**Latin Hypercube Sampling (LHS):**
Sample all parameters simultaneously with stratified random design. N_samples = 4/3 × N_params is a minimum; 200–500 is practical for 6 parameters. Enables Sobol sensitivity indices and correlation analysis.
**Use when:** interactions are suspected (e.g., η and λ jointly drive execution cost in a nonlinear way).

**Response Surface (Heatmap):**
Fix 4 parameters at base, vary 2 parameters over a grid (e.g., 10×10). Produces a 2D cost surface. Recommended for the most important pair of parameters identified by the tornado chart.

**Recommended workflow:**
1. Run OAT → produce tornado chart → identify top 2–3 parameters.
2. Run 2D grid sweep on the top pair → produce heatmap.
3. Optionally run LHS with N=200 → compute rank correlation coefficients for validation.

---

### 2.3 Tornado Chart: Step-by-Step

For each parameter p_i with low/base/high values:

1. Compute cost C_base at all base parameters.
2. Compute C_low_i = cost with p_i set to low, others at base.
3. Compute C_high_i = cost with p_i set to high, others at base.
4. Record swing = C_high_i − C_low_i (can be negative if cost decreases with parameter increase).
5. Sort parameters by |swing| descending.
6. Plot horizontal diverging bars centered on C_base.

```python
import matplotlib.pyplot as plt
import numpy as np

def tornado_chart(param_names, base_cost, low_costs, high_costs):
    swings = np.array(high_costs) - np.array(low_costs)
    order = np.argsort(np.abs(swings))[::-1]  # largest swing on top

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(param_names) + 2))
    for rank, idx in enumerate(order):
        lo = low_costs[idx] - base_cost
        hi = high_costs[idx] - base_cost
        ax.barh(rank, hi, left=base_cost, color='steelblue', alpha=0.7)
        ax.barh(rank, lo, left=base_cost, color='tomato', alpha=0.7)

    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels([param_names[i] for i in order])
    ax.axvline(base_cost, color='black', linewidth=1.2, linestyle='--')
    ax.set_xlabel('Execution Cost')
    ax.set_title('Tornado Chart: Parameter Sensitivity')
    plt.tight_layout()
    return fig
```

---

### 2.4 Response Surface (Heatmap)

For the top-two parameters (e.g., η and λ):

```python
import numpy as np
import matplotlib.pyplot as plt

eta_vals = np.linspace(0.05, 0.5, 20)
lam_vals  = np.logspace(-5, -3, 20)  # log scale often better for lambda
costs = np.zeros((20, 20))

for i, eta in enumerate(eta_vals):
    for j, lam in enumerate(lam_vals):
        costs[i, j] = compute_cost(eta=eta, lam=lam, **base_params)

fig, ax = plt.subplots()
cs = ax.contourf(lam_vals, eta_vals, costs, levels=20, cmap='RdYlGn_r')
plt.colorbar(cs, ax=ax, label='Execution Cost')
ax.set_xscale('log')
ax.set_xlabel('λ (risk aversion)')
ax.set_ylabel('η (impact coefficient)')
ax.set_title('Response Surface: Cost vs (η, λ)')
```

---

## 3. PDE vs Monte Carlo Cross-Validation Diagnostics

### 3.1 Common Causes of PDE–MC Disagreement

When PDE and MC results differ, the causes fall into four categories:

| Category | Typical magnitude | How to detect |
|---|---|---|
| MC time-discretization bias | O(Δt) for Euler, O(Δt²) for Milstein | Refine Δt by 2×, check if gap closes |
| MC sampling variance | O(1/√N) | Report ±2σ confidence interval; if PDE is inside CI, consistent |
| PDE spatial discretization error | O(Δv²) for centered differences | Refine N_v by 2×; check convergence rate |
| Formulation mismatch | Arbitrary | See section 3.3 below |

The most common trap is confusing MC sampling noise with a real discrepancy. Always compute MC confidence intervals first.

---

### 3.2 Systematic Diagnostic Protocol

**Step 1: Confirm MC confidence interval.**
Run MC with N = 10,000 paths and N = 100,000 paths. The 95% CI should shrink by roughly √10 ≈ 3.2×. Report: `MC_estimate ± 1.96 * std / sqrt(N)`. If the PDE value falls inside the CI for both runs, there is no disagreement.

**Step 2: Richardson extrapolation on PDE.**
Run PDE with (N_v, N_t) and (2N_v, 2N_t). Extrapolated value:

```
A_extrap = A_fine + (A_fine − A_coarse) / (2^p − 1)
```

where p = 2 for Crank-Nicolson centered differences. If A_extrap differs from A_fine by < 0.1%, the PDE is converged.

**Step 3: Richardson extrapolation on MC.**
Run MC with time steps Δt and Δt/2 using the same random seed (common random numbers, CRN). Extrapolate similarly. This isolates discretization bias from sampling noise.

**Step 4: Reduce to a simple limit.**
Set ξ = 0 (deterministic variance). The PDE reduces to a simple ODE in t; MC reduces to a deterministic path. If they agree here but disagree at ξ > 0, the issue is in how the stochastic variance is implemented.

**Step 5: Check boundary condition consistency.**
Verify that the MC simulation respects the same constraints as the PDE:
- Does MC allow v to go negative? (Euler-Maruyama for CIR can; use Milstein with reflection or QE scheme by Andersen.)
- Does MC compute the same cost functional as the PDE terminal condition?

---

### 3.3 Method of Manufactured Solutions (MMS)

MMS is applicable here and is the most rigorous code-verification technique.

**Procedure:**
1. Choose an arbitrary smooth function A_exact(v, t) — e.g., A_exact(v,t) = (T−t)·v·exp(−v).
2. Substitute A_exact into the PDE and compute the residual R(v,t) analytically. The PDE becomes:
   ```
   A_t − A²/η + λv + κ(θ−v)A_v + ½ξ²v·A_vv = R(v,t)  ≠ 0
   ```
3. Add R(v,t) as a source term to your numerical PDE solver (do not change the solver code, only add the source).
4. Solve numerically. The numerical solution should converge to A_exact.
5. Compute L2 error vs grid resolution. For Crank-Nicolson centered differences, expect O(Δv² + Δt²) convergence.

If the observed convergence rate matches the expected order, the code is correct. If it is O(Δv) instead of O(Δv²), there is a bug in the spatial discretization (typically a BC or one-sided difference error).

**Why this is valuable:** MMS tests the solver independent of whether the PDE correctly models the physical problem. It separates coding errors from modeling errors.

**Practical A_exact suggestion for this PDE:**
```python
def A_exact(v, t, T):
    return (T - t) * v * np.exp(-v)

def source_term(v, t, T, kappa, theta, xi, eta, lam):
    tau = T - t
    A = tau * v * np.exp(-v)
    A_t = -v * np.exp(-v)                         # dA/d(T-t) sign: A_tau = +v*exp(-v)
    A_v = tau * (1 - v) * np.exp(-v)
    A_vv = tau * (v - 2) * np.exp(-v)
    # Residual = PDE_operator(A_exact)
    R = A_t - A**2/eta + lam*v + kappa*(theta-v)*A_v + 0.5*xi**2*v*A_vv
    return R   # add -R as source in modified solver
```

---

### 3.4 Formulation Mismatch Checklist

If Steps 1–3 above show no numerical error but PDE and MC still disagree, check:

- [ ] **Sign convention:** is the cost functional integrated forward or backward? Both must use the same convention.
- [ ] **Discount factor:** if there is a discount rate r, both PDE (via the PDE term) and MC (via the payoff discounting) must handle it identically. For execution cost problems, r = 0 is common — verify this is consistent.
- [ ] **Variance process:** is MC simulating dv = κ(θ−v)dt + ξ√v dW exactly matching the PDE coefficients?
- [ ] **Terminal condition:** PDE terminal condition A(v,T)=0 must match MC payoff at T.
- [ ] **Cost functional:** the PDE value function is E[∫ λv x² dt + x² dS_permanent] — verify MC integrates the same functional.
- [ ] **x(t) path:** if MC simulates an optimal trajectory, verify it uses the same feedback control A(v,t) from the PDE, not a different approximation.

---

## Summary: Implementation Priorities

1. **PDE solver:** uniform grid in v (N_v=150), Crank-Nicolson in time (N_t=300), Picard linearization with 2 sweeps/step, Neumann BC at v=0, Dirichlet A=0 at v_max=5θ. Runtime ~0.5s in Python. Validate with MMS before using.

2. **Sensitivity analysis:** OAT tornado chart first (12 model runs for 6 parameters). Then 2D heatmap for (η, λ) which are the dominant pair per Almgren-Chriss theory. Use LHS with N=200 only if interaction terms matter.

3. **PDE–MC cross-validation:** start with confidence intervals, then Richardson extrapolation on both sides, then the formulation mismatch checklist. MMS for code correctness. The most likely culprit in practice is MC Euler-Maruyama bias at coarse Δt — use Milstein or the QE scheme for the CIR variance process.

---

## Sources

- [Crank-Nicolson method (Wikipedia)](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method)
- [Solving nonlinear ODE and PDE problems — Langtangen](http://hplgit.github.io/num-methods-for-PDEs/doc/pub/nonlin/sphinx/index.html)
- [ADI finite difference schemes for the Heston-Hull-White PDE — In 't Hout et al.](https://arxiv.org/pdf/1111.4087)
- [Numerical impact of variance boundary conditions — IMACM Wuppertal](https://www.imacm.uni-wuppertal.de/fileadmin/imacm/preprints/2023/imacm_23_11.pdf)
- [Option Pricing under Heston using ADI schemes — Imperial College](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Reports/ADIHeston_Report.pdf)
- [Solving Heston 2-factor PDE in Python — Smolski](https://antonismolski.medium.com/solving-heston-2-factor-pde-in-python-with-code-a312786f8990)
- [Optimal Portfolio Execution Strategies and Sensitivity to Price Impact Parameters — SIAM](https://epubs.siam.org/doi/10.1137/080715901)
- [Optimal Execution of Portfolio Transactions — Almgren & Chriss](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [Simulation Analysis of Optimal Execution Based on Almgren-Chriss Framework](https://www.researchgate.net/publication/326794614_The_Simulation_Analysis_of_Optimal_Execution_Based_on_Almgren-Chriss_Framework)
- [Sensitivity analysis OAT vs Latin Hypercube — Analytica Blog](https://analytica.com/blog/sensitivity-analysis-one-at-a-time-or-all-together/)
- [Latin hypercube sampling and sensitivity analysis of a Monte Carlo model — McKay et al.](https://www.sciencedirect.com/article/abs/pii/0020710188900670)
- [Monte Carlo Methods in Financial Engineering — Glasserman (UH PDF)](https://www.bauer.uh.edu/spirrong/Monte_Carlo_Methods_In_Financial_Enginee.pdf)
- [Convergence of MC simulations involving the mean-reverting square root process — Higham & Mao](https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P57.pdf)
- [Richardson extrapolation convergence (1.4.1)](http://www.lifelong-learners.com/pde/com/SYL/s1node17.php)
- [Code Verification by the Method of Manufactured Solutions — Roache (ResearchGate)](https://www.researchgate.net/publication/278408318_Code_Verification_by_the_Method_of_Manufactured_Solutions)
- [Verify Simulations with MMS — COMSOL Blog](https://www.comsol.com/blogs/verify-simulations-with-the-method-of-manufactured-solutions)
- [Tornado chart in Python — Python Graph Gallery](https://python-graph-gallery.com/web-tornado-chart/)
- [Analytic approach to degenerate parabolic PDE for Heston model — Canale et al.](https://onlinelibrary.wiley.com/doi/10.1002/mma.4363)
- [IEOR E4603 Monte Carlo SDEs — Columbia Haugh](https://www.columbia.edu/~mh2078/MonteCarlo/MCS_SDEs.pdf)
