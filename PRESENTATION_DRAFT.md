# MF796 Term Project — Presentation Draft

**Title**: Optimal Execution Under Stochastic Volatility — Calibration, Extension, and Regime Awareness on Binance BTCUSDT

**Date**: 2026-04-21 | **Team**: Eva-Qi, bp (no-322), Yuhao

This draft consolidates the defensible findings from the full audit chain. Use Part D as narrative anchor (cleanest positive result). Part E is a "methodological learning" story — four successive audits revealed successive layers of bias.

---

## Slide 1 — Problem Statement

**Almgren-Chriss optimal execution** on real Binance BTCUSDT:
- Liquidate X₀ BTC over horizon T, minimize E[cost] + λ·Var[cost]
- Extensions: Heston stochastic vol (Part D), HMM regime-switching (Part E)
- Data: 98 days aggTrades (2026-01-01 → 2026-04-08, 130M trades) + Deribit option chain

**Key question**: Does the theoretical framework produce measurable execution improvements on real crypto-execution data?

---

## Slide 2 — Headline findings

1. ✅ **Numerical correctness** — PDE matches analytical to machine precision; MC-PDE-closed-form cross-validated within 5%
2. ✅ **Calibration methodology fixed** — tick-level γ was producing NEGATIVE values (-0.0113); aggregated cascade gives γ=+1.48
3. ✅ **AC benefit at institutional size** — significant at X₀ ≥ 100 BTC (p=0.034 at 100k paths), ~50% MC cost reduction at X₀=1000
4. ✅ **Heston Q-measure via Deribit** — κ=9.09, ρ=-0.385 (leverage recovered), round-trip <4% error; beats 3/2 model 33.9% RMSE
5. ✅ **Multi-horizon scaling** — CVaR benefit grows 2.4x per 6x horizon (1h → 6h)
6. ⚠️ **Regime-aware unresolved** — V1–V5 each revealed a new bias layer; V6 pending extended data window

---

## Slide 3 — Part A: Calibration

**Methodology correction** (audit finding):

| Parameter | Original (tick-level) | Corrected (1-min aggregated) |
|---|---|---|
| γ (Kyle's λ) | **−0.0113 (WRONG SIGN)** | +1.48 (R²=0.179, n=141,120) |
| η | literature fallback 1e-3 | 1.58e-4 (R²=0.13) |
| α | 0.04 (out of range) | 0.441 (in literature range) |
| σ | 0.396 (GK, 5-min OHLC) | unchanged |
| fee_bps | not modeled | 7.5 (Binance taker) |

**Root cause of original bug**: tick-by-tick price.diff() dominated by bid-ask bounce mean-reversion → Cov(Δp, flow) comes out negative. Aggregated bar-level regression is the correct methodology.

---

## Slide 4 — Part B: HJB PDE Solver

**Correctness** (all verified):
- **Linear impact (α=1)**: Riccati ODE via `scipy.integrate.solve_ivp` RK45 rtol=1e-10 matches analytical `η·κ·coth(κ(T-t))·x²` to machine precision
- **Nonlinear impact (α≠1)**: policy iteration (implicit FD), unconditionally stable, converges in 10-20 iterations
- **Self-impact convention**: fixed mid-audit (trade k doesn't see own permanent impact)
- **PDE ↔ MC cross-validation**: within 5% on non-linear impact test

---

## Slide 5 — Part C: Monte Carlo Engine

**Key features**:
- 3 schemes: exact log-normal (preferred for GBM), Euler-Maruyama, Milstein (verified milstein-closer-to-exact-than-euler at coarse dt)
- Variance reduction: antithetic (estimator variance ~50% reduction verified on 20 seeds), Sobol QMC, control variate
- Bootstrap CI (both IID and block; IID verified sufficient — AR(1) of paired diffs = 0.0023)
- **Z-injection deterministic tests**: Heston Full Truncation matches hand-calculated Euler step within 1e-12 — catches bugs stochastic tests with loose tolerances cannot

---

## Slide 6 — Part D (anchor): Heston via Deribit

### Q-measure calibration from option IV surface

| Parameter | P-measure (spot GK) | **Q-measure (Deribit IV)** | Recovery (round-trip) |
|---|---|---|---|
| κ | 16.17 (clipped to ceiling) | **9.09** | **3.7% error** (was 900%) |
| θ | 0.256 | 0.229 | 0.1% |
| ξ | 3.00 (clipped at bound) | 2.04 | 0.4% |
| **ρ** | −0.001 (NOISE) | **−0.385** (real leverage) | **<0.1%** |
| v₀ | 0.097 | 0.162 | <0.1% |
| Feller | **VIOLATED** | **OK** (barely, +0.009 margin) | — |

### Fit quality visual (figures/iv_fit_heatmap.png)
- 106 filtered calls (Δ ∈ [0.1, 0.9], T ∈ [7, 180] days)
- **RMSE = 0.0086** (0.86% relative IV error)
- Max residual = 3.2%

### Model selection
- **Heston (Carr-Madan FFT) vs 3/2 model (MC)**:
  - Heston RMSE: 0.098 (full grid)
  - 3/2 RMSE: 0.132 (200-path MC; CF is degenerate at z₀≈533)
  - **Heston wins 33.9%** → model choice defensible, not arbitrary

### Part D causal chain (clean)
1. Deribit option chain → market IV surface
2. `fft_call_price` + L-BFGS-B multi-start → Q-measure Heston params
3. Round-trip on synthetic chain → all 5 params recovered <4% error
4. Paired MC test vs const-vol → **CVaR_95 reduced by 4.79%** (p<0.0001) on 100k paths

### Q-measure time stability (2 snapshots)
- std(κ) = 0.010, std(ρ) = 0.001, std(ξ) = 0.005 over 2 days — very stable

---

## Slide 7 — Part E (learning narrative): Regime-aware execution

**Four paired-test iterations, each exposing a new bias layer**:

| Version | Setup | Result | Bias revealed |
|---|---|---|---|
| V1 | Pre-Yuhao, mean cost, 7 days | p=0.73, "cosmetic" | σ·1e-8 magic-constant scaling → regime params differ by 8 orders of magnitude |
| V2 | Yuhao fix, mean cost, 7 days | p=0.84 | 7-day sample too short — HMM only finds 8% σ spread |
| V3 | Yuhao fix, mean cost, 98 days | p=0.84 | **Wrong metric** — AC optimizes `cost + λ·risk`, not cost alone |
| V4 | Yuhao fix, VaR/CVaR, 98 days | **CVaR-14.0%**, p<0.0001 | **Wrong params** — Yuhao multipliers overestimate γ by 41-469%, underestimate η by 79-90% |
| V5 | True per-regime calibration, 98 days | **CVaR+227%**, p<0.0001 | **Wrong data** — risk_off sub-sample (5.5% of bars ~1500 trades) insufficient → η falls back to literature |
| V6 | Pending — extended to 280 days (Worker I in flight) | TBD | — |

**Honest position**: Part E regime-aware benefit on this data window is currently **unresolved**. Each iteration surfaced a successively subtler methodological issue. The audit discipline itself is the contribution — clean positive findings from flawed methodology mislead; the V4→V5 reversal is the kind of result proper audit must surface.

---

## Slide 8 — Bonus findings

### Scale threshold (Sonnet A, 100k paths)
- X₀=1-10 BTC: AC indistinguishable from TWAP
- **X₀=100 BTC: AC significantly beats TWAP (p=0.034)** — retail boundary
- X₀=1000 BTC: ~50% MC cost reduction, p<0.0001
- (10k paths missed X₀=100 due to insufficient power — this was a type-II error in earlier analysis)

### Horizon scaling
- T=1h: CVaR benefit |−59,593|
- T=6h: CVaR benefit |−142,266| (2.4x larger)
- T=1d: AC degenerates (κT → bang-bang execution)

### Bivariate HMM experiments (both failed, generalizable finding)
- Univariate log-return: 250% σ spread
- + F&G sentiment: 45% (diluted)
- + CoinMetrics FlowIn: 46% (also diluted)
- + log FlowIn: 48% (also diluted)
- + FlowIn + FlowOut: 48% (diluted)

**Generalized**: daily-frequency features structurally mismatched with 5-min vol regimes; HMM is pulled by daily structure, loses 5-min vol regime separation. Not a feature-quality problem; a frequency-mismatch problem.

### Fee domination
- 10 BTC liquidation: fees ≈ 50% of pre-fee AC-vs-TWAP savings
- Post-fee reported savings %: ~3% (down from pre-fee 5.22% headline)

### WRDS MarketPsych negative result
- BU subscription limited to mpsych_sample (ends 2024-10-20, all-NULL crypto columns)
- Free alternative.me F&G used instead (98 days, spread 5-61)

---

## Slide 9 — Test suite maturity

### Evolution
- Starting: 129 tests, ~40% theater (shape checks, literal assignments, RNG determinism)
- After: **162+ tests** — structural, Z-injection, round-trip, data-pipeline invariants
- 16 theater tests rewritten, 57 deleted (code-council audit), 1 genuine bug (`test_ci_level` wrong direction assertion)

### Categories now present
- **Z-injection deterministic** — patch RNG, hand-calculate Euler step, assert <1e-12 match
- **Round-trip calibration** — simulate from known params, fit back, check recovery
- **Data-pipeline invariants** — γ-sign invariant, dtype safety, proxy disclaimer, NaN rejection
- **Structural identity tests** — mathematical identities not narrative bounds

---

## Slide 10 — Limitations & Future work

### Open / unresolved
- **Part E regime-aware net benefit** — V4 was Yuhao multiplier artifact, V5 has η fallback, V6 pending extended data
- **Deribit Q-measure time series** — only current-snapshot API, multi-day stability requires longitudinal pulls
- **bid/ask IV** — Deribit `get_book_summary_by_currency` returns mark_iv only; OI-weighted loss used as proxy
- **Tardis L2 spot data** — requires academic/paid access; Binance futures bookDepth used as proxy (>0.99 correlation)

### What would improve the project
- Longer data window (2024+2025) to resolve V5 risk_off sub-sample
- Joint calibration with regime-interaction term (vs sub-sample split)
- Multi-venue execution (routing Binance + Coinbase) for large orders
- Deribit options as live hedging signal during execution

---

## Appendix — all commits in this audit

```
cb2e6fa Test bivariate HMM with CoinMetrics exchange FlowIn — does microstructure feature help?
6bab5b1 Part E V5: paired regime test with TRUE per-regime params (replaces Yuhao multipliers)
fbc5fe9 Heston upgrade: bid/ask+OI weighting, 3/2 model MC comparison, 7-day snapshot infra
bc7a47d Sonnet C: 3-state HMM support + per-regime impact refit (Tasks 8, 12)
1b6509e Sonnet A: 100k hi-res paired reruns + multi-horizon + block bootstrap
3180840 GLM workers D+E+F output: bookDepth, Coinbase, CoinMetrics, IV/tail plots
4f4b794 Phase 5: paired Heston vs const-vol test using Q-measure Deribit params
9ddbbde Implement Q-measure Heston calibration via Deribit BTC option IV surface
[earlier commits: Yuhao merge, derive_regime_path 3-mode, theater test rewrites, aggregated calibration cascade, fee modeling]
```

---

*Draft generated 2026-04-21 from live audit state. See FINDINGS.md for detailed per-finding methodology and commit-level traceability.*
