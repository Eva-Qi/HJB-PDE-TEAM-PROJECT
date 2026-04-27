# HESTON — Stochastic Volatility for Optimal Execution

> **TL;DR / Current state (2026-04-26)**: We calibrate Heston **under the Q-measure from Deribit BTC option implied-vol surfaces** (commits `fbc5fe9`, `bc7a47d`), not from spot returns. P-measure spot calibration was structurally broken for κ and ρ — 24-bar overlapping rolling windows produce return autocorrelation ≈0.985, so the GMM-on-realized-variance estimator collapses (κ recovered as 20.0 vs true 2.0, clipped at the upper bound; ρ recovered as ≈+0.02, statistical noise). Switching to Carr-Madan FFT on Deribit options resolves both: κ=9.09 (std=0.01), ρ=−0.385 (std=0.001) over a 2-day stability window, IV-surface RMSE=0.0086, round-trip <4% on synthetic options. Heston Carr-Madan FFT beats the 3/2 model MC by 33.9% RMSE on the same surface (0.098 vs 0.132), which justifies the model choice. The downstream paired test (`paired_test_heston_qmeasure.py`, 100k paths) finds Heston-Q reduces CVaR₉₅ vs constant-vol by **4.79% (p<0.0001)** — Part D's anchor positive finding.
>
> **Key result**: κ=9.09, ρ=−0.385, ξ=2.04, θ=0.229, v₀=0.162; IV RMSE=0.0086; Heston beats 3/2 by 33.9% RMSE; Heston-Q vs const-vol CVaR₉₅ Δ = −4.79%, p<0.0001 (FINDINGS §1.4, §6).
>
> **Status**: stable. Deribit-Q is canonical; IBIT (cross-source) used only as robustness check (`compare_heston_ibit_vs_deribit.py`, `heston_cross_source_comparison.json`).

---

## Current state — what we are doing

### 1. Q-measure calibration via Deribit options (canonical)

**Pipeline**: Deribit `get_book_summary_by_currency` → bid/ask IV per strike-maturity → mid IV (mark IV when bid/ask both missing) → filter calls in delta ∈ [0.1, 0.9] and tenor ∈ [7, 180] days (106 options after filtering on the 2026-04-20 chain) → BS-imply mid prices → minimize SSE between Heston Carr-Madan FFT prices and market mid prices over `(κ, θ, ξ, ρ, v₀)`. Optimizer: `scipy.optimize.minimize` with `method='L-BFGS-B'`, multi-start over 10 random initial points to avoid local minima.

Implemented in `scripts/qmeasure_heston_time_series.py` (production pipeline) and `scripts/deribit_qmeasure_time_series.py` (Deribit-only single-snapshot fit). Cross-source robustness check via `compare_heston_ibit_vs_deribit.py`.

### 2. Calibrated parameters and stability

| Parameter | Value | 2-day std | Notes |
|---|---|---|---|
| κ (mean-reversion speed) | **9.09** | 0.01 | High-end of equity range, consistent with crypto vol mean-reversion |
| θ (long-run variance) | **0.229** | — | √θ = 0.479 → 47.9% long-run vol |
| ξ (vol-of-vol) | **2.04** | 0.005 | Down from 3.00 ceiling-clipped P-measure value |
| ρ (leverage correlation) | **−0.385** | 0.001 | Real leverage effect; was −0.001 (noise) under P-measure |
| v₀ (initial variance) | **0.162** | — | √v₀ = 0.402 → 40.2% spot vol |
| **Feller condition** | **OK** (margin +0.009) | — | `2κθ = 4.16`, `ξ² = 4.16` — barely satisfied |
| **IV surface fit RMSE** | **0.0086** | — | 0.86% absolute IV error; max residual 3.2% |

Round-trip test (simulate from these params → Carr-Madan FFT calibrate back) recovers all 5 parameters within 4% error. Compare with the original P-measure round-trip from §1.4 of FINDINGS: κ recovered as 20.0 (clip-at-ceiling, 900% error), ρ as +0.02 (sign-flip noise). The Deribit Q-measure resolves both.

### 3. Why P-measure failed and Q-measure works

**P-measure failure mode** (FINDINGS §1.4, original 2026-04-18 finding): `calibrate_heston_from_spot` ran 24-bar overlapping rolling windows on 5-min log returns. With window=24 and step=1, consecutive realized-variance observations share 23/24 of their data — autocorrelation ≈0.985. The GMM moment conditions identifying κ via `Cov(v_t, v_{t+h}) = (ξ²/2κ)·exp(−κh)` get crushed: the empirical autocorrelation is dominated by the rolling-window overlap, not by Heston dynamics. Symptoms:
- κ optimizer hits the upper bound (20.0) on every retry — the loss surface is flat in κ
- ρ correlation between log-return increments and variance increments collapses to noise because both series are smoothed identically by the rolling window

**Why Q-measure works**: option prices are a **single-time-stamp snapshot** of the risk-neutral distribution. There is no rolling window to introduce spurious autocorrelation. Each option strike-maturity gives one independent constraint on the joint `(κ, θ, ξ, ρ, v₀)` distribution, and 106 filtered options provide enough constraints that the optimizer finds a unique global minimum.

**Trade-off**: Q-measure ≠ P-measure (risk premium adjustment). For execution simulation we want P-measure (realized cost under physical dynamics). However:
- The structural mismatch is small for κ, θ, ξ (typically 10–30% gap)
- ρ is approximately measure-invariant at short horizons
- Conservative direction: Q-measure tends to overestimate κ and ξ (risk premium loading on vol), so execution cost simulated under Q-params is slightly **conservative** for tail risk — acceptable for the project narrative

This is documented as a limitation in PRESENTATION_DRAFT slide 10.

### 4. Model selection — Heston vs 3/2 model

On the same Deribit BTC IV surface (2026-04-20):

| Model | Calibration | RMSE (full grid) | Calibration cost |
|---|---|---|---|
| Heston (Carr-Madan FFT) | 5 params | **0.098** | ~5 s, FFT closed form |
| 3/2 model (Monte Carlo) | 5 params | 0.132 | ~30 s, MC + degenerate CF at z₀≈533 |

**Heston wins by 33.9% RMSE**. The 3/2 model's characteristic function is degenerate at the BTC vol level (`z₀ = 2κθ/ξ² ≈ 533`, far from the regime where the Lewis-Lipton CF is well-behaved), forcing fallback to direct MC pricing — slower and noisier.

This justifies the model choice **empirically**, not as a default. Reported in PRESENTATION_DRAFT slide 6.

### 5. Heston-Q execution paired test (`paired_test_heston_qmeasure.py`)

Setup:
- 100k MC paths, common-random-numbers across the two strategies
- Strategy A: AC-optimal trajectory under **constant volatility** σ = √θ ≈ 0.48
- Strategy B: AC-optimal trajectory under **Heston-Q dynamics** (variance path simulated via full-truncation Euler-Maruyama using calibrated κ, θ, ξ, ρ, v₀)
- T = 1 hour, X₀ = 1000 BTC, λ = 1e-6 (centralized constant)

Result:
- Const-vol CVaR₉₅ = 74,222
- Heston-Q CVaR₉₅ = 71,016
- **Δ = −4.79%, p < 0.0001** (paired t-test on common-random-numbers diffs)
- Robust at both 50k and 100k paths

The Heston-adapted strategy speeds up execution when variance is high and slows down when low; this regime-aware behavior reduces left-tail outcomes. See `data/paired_heston_qmeasure_results.json` for full distribution.

### 6. Tardis BTC options pipeline (cross-check, kept active)

`scripts/qmeasure_heston_time_series.py` consumes the 12 BTC `tardis_deribit_options_*.json` files via glob to produce a longitudinal κ, ρ, ξ time series. This is the project's two-solver cross-check pattern (Part 11 §11.4) — Tardis snapshots provide an independent provenance for the same Deribit chain. ETH Tardis files are out of scope and were archived to `audits/snapshots/` (Decision 1, 2026-04-27).

### 7. Numerical scheme — Heston SDE simulation

Full-truncation Euler-Maruyama (Lord, Koekkoek & Van Dijk 2010), recommended over reflection because it preserves drift sign without distorting:

```python
v_pos = np.maximum(0.0, v_k)
v_{k+1} = np.maximum(0.0,
    v_k + κ·(θ − v_pos)·dt + ξ·sqrt(v_pos)·dW_v
)
```

**Why not Milstein for the variance process**: the Milstein correction `+ (1/4)·ξ²·(dW²−dt)` reduces strong-error order from O(dt^0.5) to O(dt), but for the **execution cost estimator** only weak convergence matters (we report E[cost] under a fixed strategy, not pathwise reproduction). Full-truncation Euler-Maruyama with dt=1/(365.25·24·12) (5-min bars over 1 year-equivalent) is sufficient for weak convergence at 100k paths.

**Cholesky for correlated Brownians**:
```
dW_S = Z₁ · √dt
dW_v = (ρ·Z₁ + √(1−ρ²)·Z₂) · √dt
```

Antithetic variates negate **both** `Z₁` and `Z₂` to preserve correlation structure ρ.

### 8. Z-injection deterministic test (test suite)

`tests/test_heston_simulation.py::TestHestonZInjection`, `TestHestonFullTruncationAlgorithm` patch `np.random` with known Z arrays and assert that the Heston Euler step matches a hand-calculated value within 1e-12. This catches bugs that stochastic tests with loose tolerances cannot — wrong truncation floor, asymmetric `max(v, 0)` in drift vs diffusion, Cholesky sign flip. See FINDINGS §4.

### 9. Walk-forward / OOS robustness

Heston calibration is currently snapshot-based (single chain), not walk-forward. The 2-day stability test (std(κ)=0.01, std(ρ)=0.001 across two snapshots) is a sanity check, not a true OOS test. Listed as an open limitation in PRESENTATION_DRAFT slide 10 — multi-day longitudinal pulls would require historical Deribit IV surfaces (only currently retrievable from Tardis paid feeds).

---

## Evolution & pivots

### Apr 26 — Final state described above (TL;DR)

### Apr 21 — Deribit Q-measure calibration (FINDINGS §1.4, §6)

Commit `fbc5fe9` (Sonnet B): integrated Deribit BTC option chain pipeline, Carr-Madan FFT, multi-start L-BFGS-B. This is the resolution of the κ/ρ unreliability documented in FINDINGS §1.4 (original).

Commit `bc7a47d` (Sonnet C): added 7-day stability snapshots, bid/ask vs mid IV ablation, OI-weighting test. The 2-day stability std(κ)=0.01 came from this batch.

Commit `4f4b794`: Phase 5 — `paired_test_heston_qmeasure.py` against constant-vol baseline using Q-measure params. CVaR₉₅ Δ = −4.79% result is from this commit, robust at 50k and 100k paths.

### Apr 20 — IV-fit visualization and cross-source check (`figures/iv_fit_heatmap.png`, `data/heston_cross_source_comparison.json`)

`heston_calibrate_ibit.py` calibrated Heston to BlackRock iShares Bitcoin Trust (IBIT) options as a cross-source robustness check on the BTC narrative. Results within 5% of Deribit-BTC κ and ρ. Confirms the Deribit fit is not Deribit-specific. `heston_ibit_smile_20260424.png` shows the IBIT smile fit visually.

### Mar 27 — Implementation gaps (was `gap_heston_implementation_sensitivity.md`)

This was a working note on the **1D PDE for `A(v,t)`** under the separable ansatz `V(x,v,t) = A(v,t)·x²` plus sensitivity-analysis methodology and PDE-MC cross-validation diagnostics. Key practical recipes preserved here:

- **1D PDE grid**: uniform on `v ∈ [0, v_max]` with `v_max = 5θ` (or 5.0 if θ is small); N_v = 150–200 points; Crank-Nicolson with **Picard linearization** of the nonlinear `A²/η` term (2 sweeps per step); Neumann `A_v = 0` at v=0 (Feller-borderline) and Dirichlet `A = 0` at v_max.
- **Picard linearization**: replace `A^{n+1}_j² / η` with `A^n_j · A^{n+1}_j / η` to make the per-step system tridiagonal-linear. Two sweeps restore O(dt²) Crank-Nicolson accuracy.
- **MMS verification**: pick A_exact(v,t) = (T−t)·v·exp(−v), substitute into the PDE to compute the residual analytically, add it as a source term, verify the solver converges to A_exact at the expected order.
- **Sensitivity analysis ranges (OAT tornado then 2D heatmap on η×λ)**: α ∈ [0.5, 1.5], η ∈ [0.05, 0.5], λ ∈ [1e-5, 1e-3], ξ ∈ [0.1, 0.6], κ ∈ [0.5, 5.0], θ ∈ [0.01, 0.16]. Note: the project's actual final calibration has ξ=2.04, well above this range — the Mar 27 sensitivity bounds were drafted before crypto-Heston was calibrated. The 2D heatmap in `figures/sensitivity_eta_lambda.png` uses the actual range η ∈ [η_lit/3, 3η_lit].

The final project does not actually use the separable-ansatz 1D PDE — Part D's Heston work is purely calibration + MC paired test, not HJB-with-stochastic-vol. The 1D PDE machinery is preserved in `pde/hjb_solver.py::solve_hjb_heston` for completeness but not exercised in the report.

### Mar 26 — Q&A research (was `qa_part_cd_mc_heston.md`)

Specification-time Q&A for Parts C and D. Implementation choices that survived:

- **Q7 (Self-impact convention)**: Both conventions (cumsum-includes-self vs MC-no-self-impact) differ by `γ·Σn_k² = O(γX²/N)`, vanishing as N→∞. Neither is "wrong" per the original A&C paper. We document our convention (lagged cumsum, A&C 2000 standard) and accept the O(γX²/N) discrepancy — see the HJB note for the same Q&A.
- **Q8 (Sobol QMC at d=50)**: Use `scipy.stats.qmc.Sobol(d=50, scramble=True)` (Owen scrambling). Apply Brownian Bridge dimension ordering to concentrate variance in low-index Sobol dimensions. With smooth integrand (no digital payoffs), 5–20× variance reduction over pseudo-random MC. Implemented but not used in production — the AC + Heston-Q pipeline uses pseudo-random + antithetic + control variate, which is sufficient at 100k paths.
- **Q9 (Bootstrap for CVaR₉₅)**: Standard nonparametric percentile bootstrap (B=2000+) is correct for IID MC paths. BCa (`scipy.stats.bootstrap` with `method='BCa'`) for skewed distributions. Subsampling reserved for heavy-tailed crypto only — at 100k paths, the empirical tail is dense enough that standard bootstrap is consistent.
- **Q10 (Separable ansatz for α≠1)**: **CRITICAL** — `V(x,v,t) = A(v,t)·x²` works iff α=1 (linear temporary impact). For α≠1 the HJB contains `|V_x|^((α+1)/α)`, which is not quadratic in x; separation fails. No closed form is known for Heston + power-law impact. Options: (a) restrict to α=1 (default for Heston work), (b) full 2D PDE on (x,v) grid, (c) RL approach (FlowOE 2025). We chose (a). For the regime-aware α-extension we use a 1D HJB without Heston, not the 2D Heston-HJB.
- **Q11 (Feller violation at v=0)**: Use full-truncation Euler-Maruyama for MC (Lord et al. 2010) and Neumann `∂V/∂v = 0` at v=0 for any PDE solver. Crypto Heston frequently violates Feller (`2κθ < ξ²`); the calibrated value `2·9.09·0.229 = 4.16 ≈ ξ² = 4.16` is borderline-OK. Fichera classification: when Feller violated, a boundary condition **must** be imposed (the Fichera function is negative at v=0).
- **Q12 (Calibration without options)**: Time-series GMM on realized-variance moments (Azencott, Ren & Timofeyev 2017) is feasible but identifies P-measure parameters; for Q-measure execution simulation you need options. The original 2026-04-18 audit finding (P-measure round-trip broken) settled this in favor of Q-measure for the project. Listed as a limitation: P-measure cross-check (GMM on realized variance) was deferred.

### Mar 22 — Initial methodology spec (was `heston_execution.md`)

Theory write-up — preserved in the Methodology section below. Summary of key choices:

- **2D HJB for V(x, v, t)**: full PDE has no V_xx (inventory has no diffusion) and no V_xv cross term — simpler than option-pricing Heston PDEs.
- **Reduced 1D PDE under separable ansatz** (Section 2.4 of original note): V = A(v,t)·x² → A_t − A²/η + λv + κ(θ−v)A_v + ½ξ²v·A_vv = 0. ADI splitting (Douglas-Rachford) is unnecessary — no cross derivative.
- **MC simulation pattern**: pre-generate correlated Z arrays via Cholesky, full-truncation EM for v, exact log-Euler for S, antithetic by negating both Z₁ and Z₂.
- **Crypto vs equity Heston ranges**: crypto θ, v₀ are 3–5× higher; ξ is higher (vol-of-vol is itself high); ρ is weaker (often near zero or weakly negative). Calibrated ρ=−0.385 falls within the documented BTC range [−0.4, +0.2].

---

## Methodology / theory

### 1. Why stochastic vol changes the execution problem

In standard AC the risk cost is `λσ²·∫x(t)²dt` with constant σ. The optimal trajectory is deterministic — sinh-shape with `κ = √(λσ²/η)`.

Under Heston, σ² → v_t (CIR variance process). Consequences:
1. **Variance is a second state variable**: the optimal policy is a surface `u*(x, v, t)` not a curve `u*(x, t)`. When variance spikes, two competing pressures: trade faster (reduce risk exposure) but recognize variance may mean-revert.
2. **Risk penalty is stochastic**: market risk cost `λ·v_t·x_t²` is itself uncertain. Agent must hedge variance fluctuations in the value function.
3. **Leverage effect (ρ<0)**: price drops co-move with rising vol — a "double hit" for liquidation. Asymmetric urgency: trade faster when both inventory AND variance are high.
4. **No closed-form trajectory**: PDE has variable coefficients depending on `(x, v)`; numerical 2D PDE or simulation required.

### 2. Heston dynamics

```
dS_t = μ·S_t·dt + sqrt(v_t)·S_t·dW_S
dv_t = κ·(θ − v_t)·dt + ξ·sqrt(v_t)·dW_v
corr(dW_S, dW_v) = ρ·dt
```

For execution, μ is set to zero over short horizons. Permanent impact enters as a separate term on S (linear γ).

| Parameter | Heston role | Execution role |
|---|---|---|
| κ | Mean-reversion speed of variance | Fast κ → vol returns to θ quickly; AC can rely on average vol |
| θ | Long-run variance | Baseline risk; equivalent to σ² in constant-vol AC |
| ξ | Vol-of-vol | Uncertainty about future vol; higher ξ → more hedging motive in HJB |
| ρ | Price-vol correlation | Negative ρ amplifies urgency at high variance |
| v₀ | Initial variance | Starting point of 2D state |

### 3. 2D HJB formulation

Value function:
```
V(x, v, t) = inf_{u_s, t≤s≤T} E_t[ ∫_t^T (η·u_s² + λ·v_s·x_s²) ds + φ·x_T² ]
```

Since inventory `x` has no diffusion (only drift `dx = −u·dt`), no `V_xx` and no `V_xv` cross term. The HJB:

```
V_t + min_{u≥0} {η·u² − u·V_x} + λv·x²
    + κ(θ−v)·V_v + ½ξ²v·V_vv = 0
```

Inner minimization: `u* = max(0, V_x / (2η))`. For liquidation V_x>0 everywhere — constraint never binding.

Substituting back:
```
V_t − V_x²/(4η) + λv·x² + κ(θ−v)·V_v + ½ξ²v·V_vv = 0
```

Nonlinear PDE due to V_x² term.

### 4. Separable ansatz V = A(v,t)·x² (works iff α=1)

With V = A(v,t)·x²: V_x = 2A·x, V_xx = 2A, V_t = A_t·x², etc. Substituting and dividing by x²:
```
A_t − A²/η + λv + κ(θ−v)·A_v + ½ξ²v·A_vv = 0
```

This is a 1D nonlinear PDE in (v, t) — much cheaper than 2D.

Optimal control: `u*(x, v, t) = A(v, t)·x / η` — same structure as constant-vol AC but A depends on v.

**Why ansatz fails for α≠1**: with `h(u) = η·|u|^(α+1)`, inner minimization gives `u* ∝ |V_x|^(1/α)`, substituting yields `|V_x|^((α+1)/α)` term in the PDE. With V = A(v,t)·x², `|V_x|^((α+1)/α) = const·A^((α+1)/α)·|x|^((α+1)/α)`, which is not quadratic in x unless α=1. See Q10 above.

### 5. Numerical methods for the 2D PDE

**Why explicit Euler fails**: CFL `dt ≤ dv²/(ξ²v_max)`. For ξ=0.5, v_max=4θ, dv=0.01, θ=0.09: dt_max ≈ 2.8e-4. Thousands of time steps per day. Impractical.

**Crank-Nicolson on the 1D reduced PDE**: tridiagonal at each step. Nonlinear A²/η term handled via Picard linearization (2 sweeps per step) or Newton (3–5 inner iterations, used only if η small).

For the rare case of full 2D (α≠1), **ADI Douglas-Rachford** suffices because there is no V_xv cross term — Craig-Sneyd unnecessary, Hundsdorfer-Verwer overkill.

### 6. SDE simulation

**Full-truncation Euler-Maruyama (Lord et al. 2010)**:
```python
v_pos_k = max(0, v_k)
v_{k+1} = max(0, v_k + κ(θ-v_pos_k)·dt + ξ·sqrt(v_pos_k)·dW_v)
```

**Milstein correction** (only useful when paths matter, not just E[cost]):
```python
v_{k+1} += (1/4)·ξ²·(dW_v² - dt)
```

**QE scheme (Andersen 2007)** for tight Feller violation: approximates non-central χ² of v. More accurate near v=0 but more code. For 2κθ/ξ² > 1 (our case is borderline at ≈1.0), full-truncation is sufficient.

**Cholesky for ρ**:
```
dW_S = Z₁·sqrt(dt)
dW_v = (ρ·Z₁ + sqrt(1-ρ²)·Z₂)·sqrt(dt)
```

**Antithetic variates**: negate both Z₁ and Z₂ to preserve ρ.

### 7. Carr-Madan FFT for option pricing

Heston has a closed-form characteristic function `φ_T(u)` for `log(S_T/S_0)`. Carr & Madan (1999) pricing:
```
C(K) = exp(−α·k)/π · ∫_0^∞ exp(−i·v·k) · ψ(v) dv
ψ(v) = exp(−rT)·φ_T(v − (α+1)i) / (α² + α − v² + i(2α+1)v)
```

Computed via FFT over a log-strike grid. Used in `extensions/heston.py::fft_call_price` for the calibration objective.

### 8. Calibration methodology

**Vega-weighted least squares on IV (preferred)**:
```
min_{κ,θ,ξ,ρ,v₀} Σ_i w_i·(σ_heston(K_i,T_i) − σ_imp(K_i,T_i))²
where w_i ∝ vega(K_i, T_i)
```

Vega-weighting focuses calibration on the liquid ATM region. Alternative: relative-error-on-prices (`(C_heston − C_market)² / C_market²`) prevents near-ATM dominance.

**Bounds** (`scipy.optimize.minimize` with `L-BFGS-B`):
```
κ ∈ [0.1, 20.0]
θ ∈ [0.001, 1.0]    # 3% to 100% vol
ξ ∈ [0.01, 3.0]
ρ ∈ [-0.999, 0.999]
v₀ ∈ [0.001, 2.0]
```

**Feller soft penalty** (encourages but doesn't enforce — crypto often violates):
```
penalty = max(0, ξ² - 2κθ) · 1000
```

### 9. Crypto Heston ranges (literature priors)

| Parameter | Equities (S&P 500) | BTC (Deribit) | ETH (Deribit) | Our calibrated |
|---|---|---|---|---|
| v₀ | 0.03–0.06 | 0.08–0.25 | 0.10–0.35 | **0.162** ✓ |
| θ | 0.03–0.05 | 0.07–0.18 | 0.09–0.22 | **0.229** (slightly high) |
| κ | 1–4 | 2–8 | 2–10 | **9.09** (high end) |
| ξ | 0.2–0.5 | 0.5–1.5 | 0.6–1.8 | **2.04** (high end) |
| ρ | −0.8 to −0.5 | −0.3 to +0.1 | −0.4 to +0.2 | **−0.385** (low end of range) |
| Feller? | Usually OK | Often violated | Often violated | **Borderline OK** |

Key crypto differences: higher v₀, θ (3–5× more volatile than equities); higher ξ; weaker leverage. Our ρ=−0.385 is on the strong-negative end of the BTC range, suggesting BTC has been more equity-like during 2026-Q1 than the historical 2022–2024 baseline. Possible explanation: spot ETF flows tying BTC to risk-asset β.

---

## References & cross-links

**Source notes preserved**:
- `research/archive/heston_execution.md` (Mar 22, theory + execution problem)
- `research/archive/qa_part_cd_mc_heston.md` (Mar 26, Q7–Q12 implementation Q&A)
- `research/archive/gap_heston_implementation_sensitivity.md` (Mar 27, 1D PDE + sensitivity + PDE-MC cross-val)

**Canonical scripts**:
- `scripts/qmeasure_heston_time_series.py` (production Q-measure pipeline)
- `scripts/deribit_qmeasure_time_series.py` (single-snapshot Deribit fit)
- `scripts/paired_test_heston_qmeasure.py` (Heston-Q vs const-vol paired test)
- `scripts/compare_heston_ibit_vs_deribit.py` (cross-source robustness)
- `scripts/heston_calibrate_ibit.py` (BlackRock IBIT cross-check)
- `scripts/plot_iv_surface_fit.py` (IV surface heatmap)
- `extensions/heston.py::fft_call_price`, `calibrate_heston` (core pricing/calibration)
- `montecarlo/sde_engine.py::simulate_heston_paths`, `simulate_heston_execution` (MC simulation)

**Canonical data**:
- `data/heston_qmeasure_time_series.json` (longitudinal κ/ρ snapshots)
- `data/heston_pmeasure_vs_qmeasure.json` (P vs Q comparison, 167KB; consumed by `paired_test_heston_qmeasure.py`)
- `data/paired_heston_qmeasure_results.json` (CVaR Δ result)
- `data/heston_qmeasure_ibit_20260424.json` (IBIT cross-check)
- `data/heston_cross_source_comparison.json` (Deribit vs IBIT)
- `data/deribit_btc_option_chain_20260420.json`, `..._20260421.json` (raw chains)
- `data/tardis_deribit_options_*.json` × 12 (BTC; cross-check pipeline, KEPT in `data/`)
- `figures/heston_qmeasure_time_series.png` (Apr 25 longitudinal plot, canonical)

**Cited in**:
- `FINDINGS.md` §1.4, §6, §9 confidence summary (Heston Q-measure resolution, model selection, anchor positive finding)
- `PRESENTATION_DRAFT.md` slide 6 (Part D anchor: Q-measure calibration table, model selection, paired-test CVaR result)

**Foundational references**:
- Almgren & Chriss (2001), *J Risk* 3(2): baseline framework
- Heston (1993), *RFS* 6(2): characteristic function
- Cartea, Jaimungal & Penalva (2015), *Algorithmic and HFT* (Cambridge): HJB optimal execution, ch. 6–7, 9
- Almgren (2012), *SIAM J Fin Math* 3(1): stochastic vol AC extension
- Gatheral (2006), *The Volatility Surface* (Wiley): Heston, calibration, Feller
- in 't Hout & Foulon (2010), *IJNAM* 7(2): ADI for Heston
- Lord, Koekkoek & Van Dijk (2010), *Quant Finance* 10(2): full-truncation Euler-Maruyama
- Andersen (2007), *J Comput Finance* 11(3): QE scheme
- Glasserman (2004), *Monte Carlo Methods in Financial Engineering* (Springer): Cholesky correlated BMs, antithetic
- Carr & Madan (1999), *J Comput Finance* 2(4): FFT option pricing
- Matic, Packham & Schoutens (2021), *Rev Deriv Res*: BTC Heston calibration on Deribit; Feller often violated
- Azencott, Ren & Timofeyev (2017), arXiv 1706.04566: realized-vol P-measure GMM (deferred)
- Krishnaswamy et al. (2018), preprint: GMM Heston P-measure
- Clayton (2020), SSRN 3718697: trinomial-tree time-series Heston
- Lewis (2000), *Option Valuation under Stochastic Volatility*: 3/2 model
- Acerbi & Tasche (2002), arXiv cond-mat/0105191: Expected Shortfall (CVaR)
- Canale, Mininni, Rhandi (2017), *Math Methods Appl Sci*: degenerate-parabolic Heston PDE analysis (Fichera)
- Feehan & Pop (2023), CPAA: viscosity solutions for Heston PDE
- Brokmann et al. (2024), *Math Finance*: nonlinear price impact, linear strategies
