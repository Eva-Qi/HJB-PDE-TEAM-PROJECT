# MF796 Technical Summary — For Expert Review

## Project Title
Optimal Execution Under Stochastic Volatility and Market Regimes: A Computational Approach with Real Order Book Data

## Objective
Minimize the Almgren-Chriss mean-variance execution objective for liquidating X0 shares over horizon [0, T]:

```
min_{v(t)} E[C(v)] + λ Var[C(v)]
```

where C(v) is the implementation shortfall under strategy v(t), λ is risk aversion.

---

## Part A — Market Impact Calibration (COMPLETE)

### Data
- Source: Binance BTCUSDT aggTrades (data.binance.vision)
- 5,020,555 trades over 5 days (Mar 17-21, 2026)
- Timestamp resolution: microsecond

### Estimated Parameters

| Parameter | Value | Method |
|-----------|-------|--------|
| S₀ | $68,918 | Last trade price |
| σ (annualized) | 0.3956 | Realized vol from 5-min VWAP log returns, annualized by √(365.25×24×3600/300) |
| γ (permanent impact) | 1.33×10⁻² | Kyle's lambda = Cov(ΔP, signed_flow) / Var(signed_flow), Welford online algorithm |
| η (temporary impact) | 1×10⁻³ | Literature fallback (trade-level power-law regression too noisy) |
| α (impact exponent) | 0.6 | Literature: Almgren et al. (2005), consistent with square-root law α≈0.5 |

### Methods Used
- **Kyle's lambda**: Welford online covariance/variance accumulator for numerical stability on 5M+ observations
- **Realized volatility**: Sample std of log returns (ddof=1) on 5-min VWAP buckets
- **Temporary impact**: Attempted quantile-bucketed regression (slippage ~ η·qty^α in log-log space); fell back to literature values due to noise in trade-level data (no order book depth available)

### Open Question for Part A
- η/α calibration from trade data alone is noisy. Would order book depth snapshots (Tardis.dev) meaningfully improve the power-law regression?
- Is Kyle's lambda the right estimator for γ, or should we use the Cont-Kukanov-Stoikov (2014) OFI-based approach?

---

## Part B — HJB PDE Solver (COMPLETE)

### Formulation
The value function V(x, t) satisfies the Hamilton-Jacobi-Bellman equation:

```
V_t + min_{v≥0} { η|v|^(α+1) + λS₀²σ²x² − v·V_x } = 0
```

with V(0,t) = 0, V(x,T) = penalty·x², where x = remaining inventory.

The optimal control (FOC for α=1):
```
v*(x, t) = V_x / (2η)
```

General FOC (any α):
```
v*(x, t) = (V_x / (η(α+1)))^(1/α)
```

### Solver 1: Riccati ODE (linear impact, α=1)

V(x,t) = A(t)·x² reduces the HJB to a scalar Riccati ODE:
```
dA/dτ = −A²/η + λS₀²σ²     (τ = T−t, time remaining)
```

Solved via `scipy.integrate.solve_ivp` (RK45, rtol=1e-10). Validated against analytical solution:
```
A(τ) = η·κ·coth(κτ),    κ = √(λS₀²σ²/η)
```

Relative error < 5% across entire (x, t) grid.

### Solver 2: Policy Iteration / Howard's Algorithm (nonlinear impact, α≠1)

For α≠1, no closed-form exists. We use **Howard's policy iteration**:

1. **Initialize**: v*(x,t) = x/(T−t) (TWAP-like)
2. **Policy evaluation**: Fix v*, HJB becomes linear advection PDE:
   ```
   V_t + v*·V_x = η|v*|^(α+1) + λS₀²σ²x²
   ```
   Solved backward in time with **fully implicit Euler + upwind differencing** (unconditionally stable, diagonal ≥ 1). Forward substitution on lower bidiagonal system (no matrix factorization needed).
3. **Policy improvement**: Update v* via FOC
4. **Iterate** until ‖V_new − V_old‖∞ / ‖V‖∞ < 10⁻⁸ (typically 5-10 iterations, quadratic convergence)

### Why Implicit Euler, Not Crank-Nicolson
Crank-Nicolson (θ=0.5) produces diagonal entries 1 − 0.5·dt·v/dx which can vanish when advection velocity is large near terminal time. Fully implicit (θ=1) gives 1 + dt·v/dx ≥ 1 always — unconditionally non-singular.

### Open Questions for Part B
- Is the upwind scheme sufficient, or should we use ENO/WENO for sharper resolution near the terminal layer?
- Should we implement Rannacher time-stepping (2 implicit Euler steps before switching to CN) as a compromise?
- Grid convergence analysis: what is the observed convergence order? (Need M=50,100,200,400 comparison)

---

## Part C — Monte Carlo Engine (COMPLETE)

### Price Process
Exact log-normal GBM with permanent impact:
```
S_{k+1} = S_k · exp((μ − σ²/2)dt + σ√dt·Z_k) − γ·n_k
```

Exact log-normal ensures S > 0 always (unlike Euler-Maruyama which can produce negative prices at high σ).

### Cost Computation
Implementation shortfall per path:
```
IS = Σ_k [ n_k·(S₀ − S_k) + h(v_k)·n_k ]
```

where h(v) = η|v|^α·sign(v) is temporary impact per share.

### Variance Reduction
1. **Antithetic variates**: For each Z, simulate −Z. Variance reduction 2-5×.
2. **Control variate**: TWAP cost as control (known E[C_TWAP] from deterministic model). β = Cov(C_strategy, C_TWAP)/Var(C_TWAP). Variance reduction 5-20×.

### Risk Metrics
- VaR₉₅ and CVaR₉₅ (Expected Shortfall) from empirical cost distribution
- Strategy comparison: TWAP vs VWAP vs Optimal (mean, std, VaR, CVaR)

### Strategies Implemented
| Strategy | Trajectory | Properties |
|----------|-----------|------------|
| TWAP | x(t) = X₀(1 − t/T) | Min impact cost, max risk |
| VWAP | Proportional to 24h volume profile | Slightly better than TWAP |
| Optimal | x(t) = X₀·sinh(κ(T−t))/sinh(κT) | Min objective (cost + λ·risk) |

### Open Questions for Part C
- Should we add Quasi-Monte Carlo (Sobol sequences) for 10-30× additional variance reduction? (P3 teammate considering this)
- Bootstrap confidence intervals for VaR/CVaR estimates?
- How does MC expected cost relate to deterministic `execution_cost()`? There's a small systematic bias from permanent impact self-inclusion convention (documented, <0.05% magnitude).

---

## Part D — Stochastic Volatility Extension (NOT YET IMPLEMENTED)

### Formulation
Replace constant σ with Heston stochastic variance:
```
dS_t = √v_t · S_t · dW_S − γ·u_t·dt
dv_t = κ(θ − v_t)dt + ξ√v_t · dW_v
corr(dW_S, dW_v) = ρ
```

### 2D HJB
Value function becomes V(x, v, t):
```
V_t + min_{u≥0} { η·u² − u·V_x } + λ·v·x²
    + κ(θ−v)·V_v + ½ξ²v·V_vv = 0
```

Note: no V_xx or V_xv terms because x evolves deterministically.

### Separable Ansatz (Key Simplification)
The 2D PDE admits V(x,v,t) = A(v,t)·x² with optimal u* = A·x/η. Substituting:
```
A_t − A²/η + λ·v + κ(θ−v)·A_v + ½ξ²v·A_vv = 0
```

This is a 1D nonlinear PDE for A(v,t) — much simpler than the full 2D problem.

### Implementation Plan (from research)
1. Implement `simulate_heston_execution()` in MC engine (Cholesky for correlated BM, full-truncation for negative variance)
2. Solve 1D PDE for A(v,t) via Crank-Nicolson (or our existing implicit Euler)
3. Calibrate Heston params from Deribit BTC options (or use literature values: κ=3, θ=0.09, ξ=0.8, ρ≈0, v₀=σ²)

### Research Completed
- `research/heston_execution.md` (28KB): Full 2D HJB derivation, separable ansatz proof, ADI scheme comparison, Cholesky SDE simulation, crypto parameter ranges
- `extensions/heston.py`: Heston characteristic function and FFT call pricing already implemented (from HW2)
- `extensions/heston.py`: `HestonParams` dataclass defined

### What Still Needs Research for Part D
- **Feller condition violation in crypto**: 2κθ > ξ² is often violated for BTC. How does our PDE solver handle v→0? Need boundary treatment at v=0 (reflecting/absorbing?).
- **Separable ansatz validity**: Does V = A(v,t)·x² hold exactly, or only for linear temporary impact (α=1)? If α≠1, does the 2D PDE still separate?
- **Calibration data source**: Deribit BTC options are ideal but require API access or data purchase. Can we use implied vol surface from free sources?
- **Comparison metric**: How do we quantify "stochastic vol changes the trajectory"? Is it the difference in expected cost, or the change in trajectory shape, or the change in VaR/CVaR?

---

## Part E — Regime-Aware Execution (NOT YET IMPLEMENTED)

### Formulation
Market alternates between K=2 hidden states (risk-on / risk-off). Each regime k has parameters (σ_k, γ_k, η_k). Solve per-regime optimal trajectories and switch based on current regime posterior.

### HMM Specification
```
Hidden state: S_t ∈ {0 (risk-on), 1 (risk-off)}
Transition: A[i,j] = P(S_t=j | S_{t-1}=i)
Emission: features_t | S_t=k ~ N(μ_k, Σ_k)
```

Features: [realized_vol, log_spread_bps, OFI, log_volume]

### Per-Regime Execution
For each regime k:
1. Estimate (σ_k, γ_k, η_k) from Viterbi-labeled data
2. Construct ACParams_k
3. Solve HJB → optimal trajectory v*_k(x, t)

Two switching strategies:
- **Hard**: Use v*_k where k = argmax P(S_t=k | data)
- **Soft**: Blend v* = Σ_k P(S_t=k | data) · v*_k (posterior-weighted)

### Implementation Plan
1. `pip install hmmlearn`
2. Fit GaussianHMM on Binance trade features
3. Viterbi decode → per-regime parameter estimation
4. `regime_aware_params()` = `dataclasses.replace(base_params, sigma=σ_k, ...)`
5. Call `solve_hjb()` twice → two trajectories → compare

### Research Completed
- `research/regime_hmm.md` (29KB): Full hmmlearn tutorial with code, BIC model selection, Baum-Welch derivation, online filtering via forward algorithm, per-regime param estimation pipeline

### What Still Needs Research for Part E
- **Feature engineering**: Our data has price/qty/side. We need to compute realized_vol and volume features at appropriate frequency (hourly? 5-min?). Spread is not available from aggTrades alone (need order book).
- **Regime stability**: With only 5 days of data, how stable is the HMM fit? Do we need 30+ days?
- **Transition during execution**: If regime switches mid-execution, do we re-plan from current inventory x(t), or commit to the initial plan? The research suggests re-planning but this requires online HMM filtering.
- **Statistical significance**: How do we test that regime-conditional execution is significantly better than unconditional? Need a backtest framework or paired MC test.

---

## Cross-Cutting Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Units convention | Dollar-denominated: risk = S₀²σ²·Σx², κ = √(λS₀²σ²/η) | Teammate-flagged; ensures kappa balances cost vs risk in correct units |
| GBM discretization | Exact log-normal exp(…) not Euler-Maruyama | Audit-flagged; guarantees positive prices at any σ |
| Nonlinear HJB solver | Policy iteration + implicit Euler (not CN) | CN diagonal can vanish; implicit Euler unconditionally stable |
| Permanent impact in HJB | Omitted (only temp + risk) | Perm impact ≈ 0.5γX₀² is trajectory-independent for linear γ |
| MC cost definition | Implementation shortfall: Σ n_k(S₀−S_k) + h_k·n_k | Differs from cost_model by γΣn_k² (self-impact); <0.05% magnitude |
| Temporary impact fallback | α=0.6, η=1e-3 from literature | Trade-level regression too noisy without order book depth |

## Test Coverage
- 53 tests, 0 skipped, 0 failures
- Closed-form ↔ PDE ↔ MC triangular validation
- Nonlinear FD solver tested (policy iteration path)
- Control variate, antithetic, log-normal GBM all tested
- Impact functions, cost analysis metrics directly tested

## Repository
https://github.com/Eva-Qi/HJB-PDE-TEAM-PROJECT
