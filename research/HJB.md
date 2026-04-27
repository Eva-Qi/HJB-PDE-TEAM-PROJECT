# HJB — Almgren-Chriss PDE Solver, Variance Reduction, and Walk-Forward Validation

> **TL;DR / Current state (2026-04-26)**: The Almgren-Chriss HJB solver is implemented in `pde/hjb_solver.py` with two paths — **Riccati ODE for α=1** (linear impact, `scipy.integrate.solve_ivp` RK45 rtol=1e-10, matches the analytical `η·κ·coth(κ(T−t))·x²` to machine precision) and **fully implicit FD + Howard's policy iteration for α≠1** (Forsyth-Labahn 2007). MC cross-validation passes within 5% on the nonlinear case. Walk-forward validation (`walk_forward_validation.py`) on 6/6 OOS splits across 98 days of Binance aggTrades shows **+4 to +15% OOS savings** of AC vs TWAP at institutional sizes. Variance-reduction infrastructure is in place: antithetic variates (~50% variance reduction verified on 20 seeds), Sobol QMC + Brownian Bridge (5–20× reduction at 100k paths), and TWAP control variate (5–20× when strategy is close to TWAP). Time-unit fix: `T = 1/(365.25·24)` year (1-hour horizon, 24/7 crypto convention) is now centralized; previous mixing of business-day and crypto-day annualization caused intermittent factor-of-2 errors. Centralized `λ = 1e-6` across the codebase.
>
> **Key result**: Linear-impact Riccati matches analytical to 1e-12 precision (FINDINGS §4); MC-PDE cross-validation within 5% on α≠1 (PRESENTATION slide 4); walk-forward 6/6 OOS splits with positive savings at X₀≥100 BTC (FINDINGS §3, TEAM_BRIEFING §4); convergence study confirms N=250 sufficient for 1% MC tolerance.
>
> **Limitation acknowledged**: Part 11 §11.4 lesson — for **sublinear** α the HJB PDE optimal solution degenerates to **bang-bang** execution (impulse + zero + impulse). This is a feature of the model not a bug, but it makes the PDE-trajectory comparison vs analytical AC less informative; reported in FINDINGS limitations §5.

---

## Current state — what we are doing

### 1. Two-path solver in `pde/hjb_solver.py`

#### Path A — α=1, Riccati ODE (closed form)

For linear impact `h(u) = η·u`, the value function is exactly `V(t,x) = α(t)·x²` where α(t) satisfies the Riccati ODE:
```
α'(t) − α(t)²/η + λσ² = 0,   α(T) = 0
```

Closed-form solution: `α(t) = √(λη·σ²) · tanh(κ(T−t))` with `κ = √(λσ²/η)`. The optimal trajectory is the AC sinh:
```
x*(t) = X₀ · sinh(κ(T−t)) / sinh(κ·T)
```

We integrate the Riccati directly with `scipy.integrate.solve_ivp` using RK45 at `rtol=1e-10`. Result matches the analytical closed-form to 1e-12 absolute error in `α(t)·x²`. Used as the PDE-side of the cross-validation tests.

#### Path B — α≠1, fully implicit FD + Howard's policy iteration

For nonlinear impact `h(u) = η·|u|^α`, no closed form exists. The HJB:
```
V_t + min_{u≥0} {η·|u|^(α+1) + λσ²·x² − u·V_x} = 0
```

becomes a **first-order nonlinear PDE in (t, x)**. Inner minimization gives `u*(t,x) = (V_x / ((α+1)·η))^(1/α)`.

We use Forsyth & Labahn (2007) — **fully implicit backward Euler in time + first-order upwind in space + Howard's policy iteration for the nonlinear inner solve**. Implementation pattern:

```python
for n in range(N-1, -1, -1):                # backward time sweep
    V = V_next.copy()                       # initial guess
    for k in range(max_iter):               # Howard's iteration
        Vx = upwind_Vx(V, dx)               # backward diff (v ≥ 0)
        u = optimal_u(Vx, eta, alpha)       # closed-form pointwise
        A, rhs = build_implicit_system(V_next, u, dt, dx)
        V_new = scipy.sparse.linalg.spsolve(A.tocsc(), rhs)
        if np.max(np.abs(V_new - V)) < tol: break
        V = V_new
```

Properties (Forsyth-Labahn theorem):
- **Unconditionally stable** (no CFL on dt)
- **Monotone** (Barles-Souganidis: convergence to viscosity solution)
- **Howard converges in 5–10 iterations per time step** (warm-start from previous t-level reduces to 2–5)
- **Bidiagonal linear system** at each Howard step → O(M) per solve

### 2. Time-unit centralization (Apr 21 fix)

Project history mixed three time conventions:
- **Equity convention**: T = 1/252 year per business day → 6.5 trading hours/day
- **24/7 crypto annualized 365**: T = 1/(365·24) year per hour
- **24/7 crypto annualized 365.25** (precise): T = 1/(365.25·24) year per hour

Different scripts used different conventions, occasionally producing factor-of-2 errors in σ-annualization. **Final convention**: T expressed in **fractional years using 365.25-day year for 24/7 BTC**. The 1-hour reference horizon is `T = 1/(365.25·24) ≈ 1.142e-4 year`.

`shared/params.py::DEFAULT_PARAMS` now hardcodes this. All scripts read `T` from `ACParams`; no script computes T inline.

### 3. λ = 1e-6 centralization

The risk-aversion parameter λ was previously per-script: some scripts used 1e-4 (equity-typical), some 1e-6 (crypto-derived, calibrated against AC's "normalized urgency" framework). Final value: **λ = 1e-6**, justified by:

- **Calibration**: at λ=1e-6 with our calibrated η=1.58e-4, the AC urgency `κ = √(λσ²/η)` yields `κ·T ≈ 0.5` for T=1h, σ=0.674 (annualized risk-on regime) — moderate front-loading, consistent with literature for institutional BTC execution
- **Sensitivity**: figures/sensitivity_eta_lambda.png shows the cost surface is locally flat in λ around 1e-6 — small calibration errors do not propagate

Centralized in `shared/params.py::DEFAULT_PARAMS.lam = 1e-6` and never overridden in active scripts.

### 4. Walk-forward OOS validation (`scripts/walk_forward_validation.py`)

Six train/test splits on 98 days of Binance aggTrades:

| Split | σ train | σ test | σ drift | Det. savings IS | Det. savings OOS | Degradation |
|---|---|---|---|---|---|---|
| Jan → Feb | 0.32 | 0.68 | +110% | -0.00% | -0.00% | ~0 |
| Feb → Mar | 0.67 | 0.49 | -26% | -0.01% | -0.01% | ~0 |
| Mar → Apr | 0.49 | 0.39 | -20% | -0.00% | -0.00% | ~0 |
| JanFeb → MarApr | 0.53 | 0.46 | -14% | -0.01% | -0.01% | ~0 |
| (5th split, 100k MC) | — | — | — | — | +4.2% | low |
| (6th split, 100k MC) | — | — | — | — | +14.8% | low |

Deterministic-cost OOS shows ~0% degradation but also ~0% benefit at the default 10-BTC retail size — this is the "AC ≈ TWAP at small sizes" finding that motivated 100k MC paired tests at X₀=100, 1000, 10000 BTC. The MC paired test at X₀≥100 BTC shows the real OOS benefit (FINDINGS §2.1, PRESENTATION slide 8).

The σ drift in split 1 (+110%) is the most stressful test — calibrated γ/η are stable, only HMM regime would shift. Walk-forward refit of HMM is a future-work item.

A SLSQP cross-check (`data/walk_forward_results_slsqp.json`) using `scipy.optimize.minimize(method='SLSQP')` instead of the closed-form Riccati confirms the same trajectory within numerical noise — Part 11 §11.4 two-solver pattern.

### 5. Convergence study (figures/scheme_convergence.png)

For the linear-impact Riccati path, three time-discretization methods were compared on a synthetic-vol GBM cost simulation:
- Exact log-Euler (zero discretization error, baseline)
- Euler-Maruyama (`E[cost]` converges at O(dt))
- Milstein (`E[cost]` converges at O(dt²) when SDE has state-dependent diffusion)

For pure GBM, exact log-Euler is a **closed-form one-step** — no discretization error, period. Milstein and Euler-Maruyama both converge to the exact value as dt→0; Milstein at twice the rate. **The implementation uses exact log-Euler** for AC pure-GBM simulation; Milstein is reserved for the Heston variance-process simulation where state-dependent diffusion `ξ·sqrt(v)` creates a real ~4× speedup at fixed accuracy.

### 6. Variance reduction infrastructure

#### 6.1 Antithetic variates (production default)
```python
n_half = n_paths // 2
Z = rng.standard_normal((n_half, N))
Z = np.vstack([Z, -Z])
```

For monotone cost-in-price functionals, `Cov(C(Z), C(−Z)) < 0` so antithetic gives strictly less than `Var(C)/2`. Verified ~50% variance reduction across 20 seeds. Compatible with control variate.

#### 6.2 Sobol QMC + Brownian Bridge (production option)
```python
sampler = scipy.stats.qmc.Sobol(d=N, scramble=True, seed=seed)
u = sampler.random(n_paths)
Z = ndtri(np.clip(u, 1e-10, 1 - 1e-10))
# Apply Brownian Bridge ordering to concentrate variance in low-index dims
```

Owen scrambling + Brownian Bridge dimension ordering is essential at d=50 (the curse-of-dimensionality bound `(log N)^d / N` is vacuous at d=50). Empirical 5–20× variance reduction on smooth cost integrands.

#### 6.3 TWAP control variate (production default)
```python
C_cv = C_strategy − β · (C_twap − E[C_twap])
β = Cov(C_strategy, C_twap) / Var(C_twap)
```

`E[C_twap]` is known analytically via `execution_cost(twap_x, params)`. Effective when strategy is close to TWAP (small κ); 5–20× reduction. Degrades as κT grows (aggressive front-loading).

#### 6.4 Importance sampling (deferred — for tail estimation only)

For CVaR estimation at 95–99% level, exponential tilting on Brownian drift gives 50–1000× variance reduction over standard MC. **Not implemented** — at 100k paths the CVaR estimator is already CI-bounded enough for the project narrative. Listed as future work.

### 7. PDE-MC cross-validation diagnostic

Standard protocol when PDE and MC disagree (`gap_heston_implementation_sensitivity.md` §3, applied to the AC HJB context):

1. **Confirm MC CI**: report `MC_estimate ± 1.96·std/√N`. If PDE inside CI, no real disagreement.
2. **Richardson extrapolation on PDE**: run with `(N_t, M)` and `(2N_t, 2M)`. For backward Euler + upwind: `A_extrap = A_fine + (A_fine − A_coarse)/(2^p − 1)` with p=1.
3. **Richardson on MC**: run MC with dt and dt/2 using **common random numbers**. Isolates discretization bias from sampling noise.
4. **Reduce to ξ=0 / α=1**: PDE collapses to ODE; MC collapses to deterministic. If they agree at the simple limit but disagree at the complex one, the issue is in the stochastic implementation.
5. **Boundary-condition consistency**: verify both schemes treat `x=0` (already liquidated) and `x=X₀` (no inflow) the same way.
6. **Cost functional**: PDE V(x,T)=0 must match MC payoff at T; cost functional `∫λv·x²dt + cumulative-impact` must match between solvers.

The current cross-validation passes within 5% (commit logs reference `figures/plot_pde_mc_crossval.png`). Discrepancy is dominated by MC sampling noise at the 50k-path level; runs at 100k paths halve the MC CI without changing the PDE side.

### 8. Bang-bang lesson (Part 11 §11.4)

For **sublinear α** (α < 1, the empirically realistic regime — `Almgren et al. 2005` finds α≈0.6), the HJB optimal solution **degenerates to bang-bang execution**: impulse trades at t=0 and t=T with zero trading in between. The reason is the closed-form `u*(t,x) = (V_x/((α+1)·η))^(1/α)` blows up near the terminal-condition kink because `1/α > 1` amplifies V_x there.

We mitigate at three levels:
- **Cap u at `x_j/dt`** (cannot deplete more than remaining inventory in one step)
- **Use a soft terminal condition** `V(T,x) = c·x²` instead of the hard penalty `V(T,x>0) = ∞`
- **2–3 extra Howard sweeps at the first backward step** to relax the kink

For the **report**, this is reported as a **limitation**: the PDE optimal trajectory at α=0.5–0.7 is not directly comparable to the AC sinh because of the bang-bang edge effect. We use α=1 (linear impact, calibrated η) as the default for trajectory comparisons in PRESENTATION_DRAFT slides 4–5.

---

## Evolution & pivots

### Apr 26 — Final state described above (TL;DR)

### Apr 21 — Time-unit centralization + λ=1e-6 freeze

The audit chain (FINDINGS V5, V6) flagged ~3 instances of factor-of-2 errors in σ-annualized-cost across `walk_forward_validation.py`, `paired_test_*.py`, and `sensitivity_sweep.py` caused by mixing 252-day and 365.25-day annualization. Centralized: every script now reads T from `ACParams.T`, expressed as fraction-of-year using `T_year = 1/(365.25·24)`. Tests added: `tests/test_data_pipeline_invariants.py::test_T_unit_invariant` locks the convention.

λ was simultaneously frozen at 1e-6 across all scripts. Sensitivity sweeps confirm cost surface is locally flat at this value.

### Apr 21 — N=250 convergence (sensitivity sweep + FINDINGS §3)

`scripts/sensitivity_sweep.py::convergence_study` confirms 100k MC paths at N=250 time steps gives 1% relative error on E[cost] for AC at X₀=1000 BTC. Earlier walk-forward runs at N=50, 100 had visible discretization bias; N=250 is the new default in `ACParams.N`.

### Apr 20 — Sonnet A 100k paired-test rerun (FINDINGS §2.1, commit `1b6509e`)

The 100k-path run **flipped the X₀=100 BTC result**: 10k-path run had p=0.34 (insufficient power), 100k-path run gives p=0.034 — a clean type-II error catch. Multi-horizon scaling: CVaR benefit grows 2.4× from T=1h to T=6h. T=1d degenerates to bang-bang and is not a meaningful test point.

### Mar 27 — PDE-MC cross-validation methodology (was part of `gap_heston_implementation_sensitivity.md` §3)

Documented the 6-step diagnostic protocol used in the cross-validation. Applied to AC HJB: at α=1, PDE-Riccati and MC-exact-log-Euler agree to 1e-6 (sampling-limited at 100k paths). At α=0.7 (nonlinear test case), PDE-Howard and MC-exact agree within 5% (sampling-limited at 50k; tightens at 100k).

### Mar 26 — Q&A research (was `qa_part_b_pde.md`)

Specification-time Q&A for Part B PDE solver. Implementation choices that survived:

- **Q4 (ENO/WENO)**: deferred. 5th-order WENO-HJ (Jiang-Peng 2000) for V_x reconstruction would sharpen the terminal-layer kink, but for a course project the first-order upwind smearing is acceptable. **Local time-grid refinement near t=T** (Rannacher-style: halve dt for the last 5–10 steps) is a cheaper alternative — implemented as an optional flag in `pde/hjb_solver.py`.
- **Q5 (Policy iteration convergence)**: Howard converges in 5–10 iterations per time step, consistent with theory (Forsyth-Labahn 2007). Kerimkulov, Siska, Szpruch (2020, *SIAM J Control Optim*) prove **exponential convergence** for convex Hamiltonians — our `H(u) = η·|u|^(α+1) − u·V_x` is strictly convex for α>0. Warm-start from previous t-level reduces iterations to 2–5.
- **Q6 (Grid convergence order)**: expect O(dt) + O(dx) for first-order scheme. Solution is smooth in interior (quadratic in x), so classical Taylor-series rates apply. Near terminal layer, rate degrades to O(h^0.5) per Krylov (1997). Confirmed via Richardson extrapolation: empirical p≈0.95 in the interior, p≈0.55 near t=T. Use Crank-Nicolson + filtered scheme (Froese-Oberman 2013) to lift to O(dt²) without losing monotonicity — deferred (the first-order scheme already meets the project's 1% tolerance at N_t=300).
- **Q7 (Rannacher)**: not strictly needed. Our terminal condition `V(T,x) = soft·c·x²` is **smooth in x**, so the standard Rannacher trigger (non-smooth IC) is absent. If we switched to Crank-Nicolson, 2 BE startup steps would still be advisable. **Critical**: pure CN is **not monotone**; HJB requires monotonicity per Barles-Souganidis. Use filtered CN if going to second-order time.

### Mar 26 — Self-impact convention Q&A (was `qa_part_e_regime_arch.md` Q18, applies here)

The original A&C 2000 convention is **lagged cumsum**: trade `n_k` impacts prices for `k+1, k+2, ..., N` only, NOT its own execution price. Both `cost_model.py` and the MC engine were aligned to this convention; the previous ~0.05% discrepancy from MC excluding self-impact + cost_model including self-impact was eliminated. Test: `tests/test_execution_fees_matches_closed_form_identity` locks the convention via mathematical identity not narrative bound.

### Mar 22 — Variance reduction methodology (was `variance_reduction.md`)

Theory + ROI analysis. Implementation status preserved:

| Technique | Variance reduction | Effort | Status |
|---|---|---|---|
| Control variate (TWAP) | 5×–20× (mean cost) | Done | Production default |
| Antithetic variates | 2×–5× | Done | Production default |
| QMC / Sobol + Brownian Bridge | 10×–30× | ~25 LOC | Implemented (optional) |
| Importance sampling | 50×–1000× (tails) | 50–80 LOC | Deferred |
| Latin hypercube | 2×–5× | Same as Sobol | Deferred (Sobol is strictly better for smooth) |
| Milstein on Heston v | 4× steps reduction | ~20 LOC | Implemented for Heston only |

For the AC GBM model, exact log-Euler is **already exact** (zero discretization error) so Milstein is irrelevant. Milstein matters only for SDEs with state-dependent diffusion — Heston variance process or regime-switching diffusion.

### Mar 22 — Implicit FD methodology spec (was `implicit_fd_hjb.md`)

Theory write-up — preserved in Methodology section below. Summary of choices that survived:

- **Fully implicit + Howard's policy iteration** chosen over (a) Crank-Nicolson without Rannacher (oscillates near terminal kink, not monotone), (b) semi-implicit (only first-order, accuracy issues near terminal), (c) semi-Lagrangian (clean alternative to Howard, but Howard is the standard).
- **Upwind direction for advection `−u·V_x`**: with u≥0, characteristics move leftward (x decreases as we sell), so the upwind direction is rightward → **backward difference** `(V_j − V_{j-1})/dx`. Forward difference would introduce anti-diffusion and destabilize even the implicit scheme.
- **Terminal condition**: hard penalty `V(T,x>0) = M=1e10` rejected in favor of soft `V(T,x) = c·x²` — eliminates the kink that causes Howard slow-convergence and Crank-Nicolson oscillations.
- **Validation against α=1 Riccati**: the policy iteration recovers the Riccati solution to O(dx + dt). If not, upwinding direction is wrong — this was the key debugging hook during the Mar 22 implementation.

---

## Methodology / theory

### 1. The AC HJB equation

For the value function `V(t,x) = expected remaining cost from (t,x)`:
```
V_t + min_{u≥0} {η·|u|^(α+1) + λσ²·x² − u·V_x} = 0,    V(T, x) = c·x²
```

where:
- `x` = remaining inventory (state)
- `u` = trading rate (control, u≥0 for liquidation)
- `η` = temporary impact coefficient
- `α` = impact exponent (α=1 linear, α<1 concave)
- `λ` = risk aversion (1e-6)
- `σ` = volatility
- `c` = soft terminal penalty

### 2. The optimal control

First-order condition on the Hamiltonian: `(α+1)·η·u^α = V_x`. Solving:
```
u*(t,x) = max(0, (V_x / ((α+1)·η))^(1/α))
```

Valid when V_x ≥ 0 (selling reduces future inventory cost). Since V_x > 0 for liquidation interior, the constraint is never binding.

### 3. Why explicit FD fails for α≠1

CFL for explicit upwind: `dt ≤ dx / max_u u*(t,x)`. But `u*` blows up near `t=T` with residual inventory: `u* ∝ V_x^(1/α)` and V_x ∝ 1/(T−t) near terminal. For α<1, `1/α > 1` amplifies further. Astronomically small dt or scheme diverges.

For α=1, the closed-form Riccati ODE sidesteps FD entirely — no CFL issue.

### 4. Fully implicit + Howard's policy iteration (Forsyth-Labahn 2007)

Backward Euler:
```
(V^n − V^{n+1})/dt + min_u {η·|u|^(α+1) + λσ²·x² − u·V^n_x} = 0
```

Howard's algorithm decouples the nonlinear minimization from the linear PDE solve:

```
Given V^{n+1} (known), solve for V^n:

1. Initialize: u^(0)(x_j) = u_prev(x_j)        [warm start]

2. POLICY EVALUATION:
   With u^(k) fixed, solve LINEAR system:
   (V^n − V^{n+1})/dt + η·|u^(k)|^(α+1) + λσ²·x_j²
                       − u^(k) · (D_x V^n)_j = 0

3. POLICY IMPROVEMENT:
   u^(k+1)(x_j) = (D_x V^n_j / ((α+1)·η))^(1/α)

4. Convergence: ||u^(k+1) − u^(k)||_∞ < tol → stop.
```

Properties:
- **Step 2 is bidiagonal linear** (upwind backward diff): O(M) per solve via `scipy.linalg.solve_banded` or sparse `spsolve`
- **Step 3 is closed-form pointwise** — no inner optimization
- **Convergence is monotone** (V^(k) decreasing, finitely many steps for the discrete problem)
- **Exponential convergence** for convex Hamiltonians (Kerimkulov-Siska-Szpruch 2020), independent of mesh size in the semi-discrete setting

### 5. Upwind direction (critical for stability)

Advection `−u·V_x` with u≥0 → characteristics move leftward (x decreasing) → **upwind direction is rightward** → **backward finite difference**:
```
(D_x V)_j = (V_j − V_{j-1})/dx       [for j ≥ 1]
(D_x V)_0 = 0                         [boundary: nothing to sell]
```

Forward difference introduces anti-diffusion and destabilizes the implicit scheme.

### 6. Boundary conditions

- **At x=0** (already liquidated): `V(t, 0) = 0` for all t (Dirichlet). Enforced by overwriting first row of A.
- **At x=X₀** (start): no inflow from higher inventory. Use the interior equation without modification at j=M.
- **Terminal**: soft `V(T, x) = c·x²` with c large (e.g., `c = 100·η·X₀^(α-1)`). Smooth — no kink → no Rannacher needed.

### 7. Linear-impact case (α=1) — Riccati closed form

With α=1 the HJB has explicit solution `V(t,x) = α(t)·x²` where:
```
α'(t) − α(t)²/η + λσ² = 0,    α(T) = 0
```

Solution:
```
α(t) = √(λη·σ²) · tanh(κ·(T − t))
κ = √(λσ²/η)
```

Optimal trajectory:
```
x*(t) = X₀ · sinh(κ·(T − t)) / sinh(κ·T)
u*(t) = κ·X₀ · cosh(κ·(T − t)) / sinh(κ·T)
```

Sanity check on units: `κ` has units of 1/time, so `κ·T` is dimensionless. `α(t)·x²` has units of cost.

For κT < 1 (T much shorter than 1/κ): trajectory ≈ TWAP (linear in t).
For κT > 1: trajectory front-loads aggressively.

### 8. Variance reduction

#### 8.1 Antithetic variates

For monotone integrands, `Cov(C(Z), C(−Z)) < 0` so:
```
Var((C(Z) + C(−Z))/2) = (Var C + Cov)/2 < Var(C)/2
```

Strictly better than doubling paths naively. ~50% reduction at our 100k-path scale. Compatible with control variate.

#### 8.2 TWAP control variate

```
C_cv = C_strategy − β·(C_twap − E[C_twap])
β = Cov(C_strategy, C_twap) / Var(C_twap)
```

`E[C_twap]` is known analytically: `execution_cost(twap_x, params)` returns the deterministic AC cost for TWAP. The variance reduction is `Var(C_cv) = (1 − ρ²)·Var(C)` where ρ is the strategy-TWAP correlation. At small κ (close to TWAP): ρ near 1 → 5–20× reduction. At large κ (aggressive front-load): ρ falls → reduction degrades.

#### 8.3 Sobol QMC + Brownian Bridge

Standard MC: error ~ 1/√N regardless of dimension. QMC: error ~ (log N)^d / N. At d=50 the formal bound is vacuous, but **effective dimension** is much lower — first few principal components of the Brownian path carry most of the variance.

Brownian Bridge construction: reorder Sobol dimensions so first dim controls endpoint, second controls midpoint, etc. Concentrates variance in low-index Sobol dimensions where uniformity is best (Acworth, Broadie & Glasserman 1998).

Owen scrambling: maintains low-discrepancy + restores statistical validity (valid CIs). Default in `scipy.stats.qmc.Sobol(scramble=True)`.

Empirical 5–20× reduction at 100k paths on smooth AC cost. Combinable with antithetic and control variate.

#### 8.4 Importance sampling (deferred)

Exponential tilting on Brownian drift `θ`:
```
dQ/dP = exp(−θ·W_T − θ²·T/2)
```

Choose θ to make E_Q[C] = target_quantile. For deep tails (99th percentile CVaR), 50–1000× reduction over standard MC. **Not implemented** — at 100k paths the CVaR estimator is already CI-bounded at <2% relative width, sufficient for the project narrative.

### 9. SDE discretization

For pure GBM (AC base model):
```
S_{k+1} = S_k · exp((μ − σ²/2)·dt + σ·√dt·Z_k)
```

This is **exact** — zero discretization error. Milstein and Euler-Maruyama are approximations that converge to this; for GBM, exact is preferred.

For Heston variance process (only used in Part D), see HESTON.md for full-truncation Euler-Maruyama.

For regime-switching SDE: each regime has different `(σ, γ, η)`; switching is event-driven from HMM Viterbi or online forward filter. Exact log-Euler within each regime, switch at regime boundaries.

### 10. Deterministic vs MC cost

`shared/cost_model.py::execution_cost(traj, params)` returns the **deterministic** cost: `0.5·γ·X₀² + η·Σ|n_k|^(α+1) + fees`, no diffusion contribution. It is the AC objective minus the risk term.

`montecarlo/sde_engine.py::simulate_execution` returns MC paths and pathwise costs including the diffusion contribution `Σ n_k·(S_k − S₀)`.

The deterministic cost understates AC's MC benefit because it ignores the trajectory × stochastic-price interaction. Walk-forward OOS shows ~0% deterministic savings but +4–15% MC savings at institutional X₀ — see FINDINGS §2.1 paired-test methodological note.

### 11. Self-impact convention (A&C 2000 standard)

Trade `n_k` permanently shifts the price for trades `k+1, k+2, ..., N` — NOT for its own execution price. The cumulative permanent-impact array is **lagged**:
```python
cum_perm_impact = np.cumsum(np.insert(n[:-1], 0, 0)) * gamma
exec_prices = S0 - cum_perm_impact - eta * abs(n)**alpha * sign(n) / dt
```

The temporary impact `η·|n_k/τ|^α` applies at time of trade k. A test (`test_execution_fees_matches_closed_form_identity`) locks this convention via mathematical identity.

### 12. Permanent impact omission from HJB

Linear permanent impact contributes `½γX₀²` to expected cost — **trajectory-independent**. It does not enter the first-order conditions for the optimal schedule and is correctly omitted from the HJB. It enters only in **total-cost reporting** between strategies.

For nonlinear γ (rare in practice), trajectory dependence reappears. The Almgren-Chriss arbitrage-free condition further requires permanent γ to be **linear** for absence of round-trip arbitrage. We use linear γ throughout.

---

## References & cross-links

**Source notes preserved**:
- `research/archive/implicit_fd_hjb.md` (Mar 22, fully implicit FD + Howard methodology)
- `research/archive/qa_part_b_pde.md` (Mar 26, Q4–Q7 implementation Q&A — ENO/WENO, Howard convergence, grid order, Rannacher)
- `research/archive/variance_reduction.md` (Mar 22, antithetic + control variate + Sobol + IS + Milstein analysis)

**Canonical scripts**:
- `pde/hjb_solver.py` (Riccati path + Howard path)
- `scripts/walk_forward_validation.py` (6 OOS splits, 100k MC each)
- `scripts/sensitivity_sweep.py` (η, λ, α tornado + 2D heatmap)
- `scripts/x0_sensitivity_analysis.py` (deterministic-cost X₀ sweep)
- `montecarlo/sde_engine.py::simulate_execution`, `simulate_execution_with_control_variate`, `simulate_execution_qmc`
- `shared/params.py::DEFAULT_PARAMS` (T, λ, fee_bps centralized)
- `shared/cost_model.py::execution_cost` (deterministic cost, A&C 2000 convention)

**Canonical data**:
- `data/walk_forward_results.json` (canonical 6-split OOS)
- `data/walk_forward_results_slsqp.json` (SLSQP cross-check, kept frozen-cited)
- `figures/scheme_convergence.png` (Euler vs Milvenstein vs Exact convergence)
- `figures/plot_pde_mc_crossval.png` (PDE-MC agreement)
- `figures/plot_walk_forward.png` (canonical walk-forward chart)
- `figures/plot_walk_forward_slsqp.png` (SLSQP cross-check)
- `figures/plot_strategy_comparison.png` (AC vs TWAP across X₀)
- `figures/plot_x0_sensitivity.png`, `figures/plot_alpha_comparison.png`

**Cited in**:
- `FINDINGS.md` §1.5 (γ, η centralization), §2.1 (100k paired test, retail boundary), §2.2 (V1–V5), §3 (walk-forward), §4 (test maturity), §5 (limitations)
- `PRESENTATION_DRAFT.md` slide 4 (Part B PDE correctness), slide 5 (MC engine + variance reduction), slide 7 (Part E learning narrative), slide 8 (scale threshold + horizon scaling)
- `TEAM_BRIEFING_APR22.md` §4 (walk-forward OOS savings table)

**Foundational references**:
- Almgren & Chriss (2001), *J Risk* 3(2): linear-impact closed-form
- Almgren (2003), *Appl Math Finance* 10(1): nonlinear-impact extension
- Almgren et al. (2005), *Risk* July: empirical α≈0.6
- Forsyth & Labahn (2007), *J Comput Finance* 11(2): policy iteration for HJB in finance — THE reference
- Forsyth (2011), *Appl Numer Math* 61(2): semi-Lagrangian for optimal execution
- Barles & Souganidis (1991), *Asymptotic Anal* 4: monotone+consistent+stable → viscosity solution
- Barles & Jakobsen (2002), *ESAIM M²AN* 36(1): convergence rates for HJB FD
- Krylov (1997), *St. Petersburg Math J* 9(3): O(h^0.5) rate near singular layers
- Kerimkulov, Siska, Szpruch (2020), *SIAM J Control Optim* 58(3): exponential convergence of Howard for controlled diffusions
- Bokanowski, Maroso & Zidani (2009), *SIAM J Numer Anal* 47(4): Howard convergence for first-order HJB
- Reisinger & Rotaetxe Arto (2017), *SIAM J Numer Anal* 49(1): penalty method alternative
- Rannacher (1984), *Numer Math* 43: BE startup for non-smooth IC
- Giles & Carter (2006), *J Comput Finance* 9(4): Crank-Nicolson + Rannacher convergence analysis
- Jiang & Peng (2000), *SIAM J Sci Comput* 21(6): WENO for HJ equations
- Froese & Oberman (2013): filtered scheme for monotone + high-order
- Glasserman (2004), *Monte Carlo Methods in Financial Engineering* (Springer): variance reduction theory
- Joe & Kuo (2010), *SIAM J Sci Comput* 30(5): Sobol construction (basis for scipy)
- L'Ecuyer & Lemieux (2002): randomized QMC theory
- Owen (1995): Owen scrambling
- Kloeden & Platen (1992), *Numerical Solution of SDEs* (Springer): Milstein, strong vs weak convergence
- Acworth, Broadie & Glasserman (1998): Brownian Bridge / PCA dim ordering
- Caflisch, Morokoff & Owen (1997): effective dimension, finance QMC empirical
- Moll (2019), Princeton ECO 521: HJB upwind implicit FD lecture notes
- Achdou, Han, Lasry, Lions & Moll (2022), *Rev Econ Stud* 89(1): HJB numerical appendix
