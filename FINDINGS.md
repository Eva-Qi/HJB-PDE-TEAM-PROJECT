# MF796 Methodology Findings — V6 (2026-04-21)

**Audience**: Team (Eva, bp, Yuhao) + future reviewers
**Status**: Living document. Update when calibration, test, or data assumptions change.

> **V6 headline**: Part E V4 "clean positive finding" (CVaR₉₅ −14%, p<0.0001, commit `f6c3ace`) has been **invalidated** by V5 paired test (commit `6bab5b1`). CVaR₉₅ benefit reversed sign: −14.0% → +227.5%. Regime-aware benefit is currently **unresolved**. See §2 for the full narrative.

This document captures the substantive findings from the 2026-04-18 code-council audit (with GLM-5.1 challenger cross-review) and the resulting methodology fixes, extended with 2026-04-20/21 findings from Yuhao's regime-aware merge, the multi-feature HMM experiments, the full 98-day Part E paired tests (V1–V5), and the Deribit/Heston validation batch. It complements (does not replace) PROJECT_AUDIT_REPORT.md.

---

## 1. Calibration methodology corrections

### 1.1 Permanent impact (γ): tick-level → aggregated

**Bug**: `estimate_kyle_lambda` on tick-level `price.diff()` returned **negative γ** (-0.0113 on 2026-01 data). Negative γ is economically absurd ("buys push price down"). Root cause: bid-ask-bounce mean-reversion dominates per-tick price changes. The pipeline silently fell back to magic constant γ=1e-4.

**Fix**: New `estimate_kyle_lambda_aggregated()` resamples trades to 1-min buckets and regresses `Δprice` on `net_flow`. `calibrated_params()` uses a cascade:

```
1-min aggregated  →  5-min aggregated  →  tick-level  →  literature fallback
```

**Result on 98 days real data**: γ = **1.48 per BTC** (positive, R²=0.179, n_buckets=141,120). `CalibrationResult.sources["gamma"]` now reports which method won (`"aggregated_1min"` in practice).

### 1.2 Temporary impact (η, α): same story

**Bug**: `estimate_temporary_impact_from_trades` returned α ≈ 0.04 (far below literature [0.3, 1.5] range) because per-trade `abs_price_change` is again bid-ask-bounce dominated. Fallback to η=1e-3 was triggered silently.

**Fix**: `estimate_temporary_impact_aggregated()` with the same 1-min → 5-min → trade-level → fallback cascade.

**Result**: η = **1.58e-4**, α = **0.441** (1-min aggregation, matches audit report's pre-existing bar-level expectation).

### 1.3 Exchange fees modeled

Binance BTCUSDT spot taker fee (7.5 bps) is now part of `ACParams.fee_bps` and added to `execution_cost()`:
```
fee = (fee_bps / 1e4) · S0 · Σ |n_k|
```

For a monotone full liquidation this is a constant across strategies, so it doesn't change optimization direction. It does change the reported "% savings" denominator (pre-fee 88% of TWAP objective → post-fee 60% on the reference 10-BTC case).

### 1.4 Heston calibration — Q-measure via Deribit (updated 2026-04-21, commits fbc5fe9, bc7a47d)

**Original finding (2026-04-18)**: `calibrate_heston_from_spot` round-trip was structurally broken for κ (true=2.0 → recovered=20.0, clips to ceiling) and ρ (true=−0.7 → recovered≈+0.02, noise). Root cause: 24-bar overlapping rolling windows produce autocorrelation ≈ 0.985. Fixing this required options-based Q-measure calibration.

**Resolution (Sonnet B, commit `fbc5fe9`)**: Deribit BTC option IV surface integrated. Heston calibrated via Carr-Madan FFT on live options data:

| Parameter | Calibrated value | 2-day std |
|-----------|-----------------|-----------|
| κ (mean reversion) | 9.09 | 0.01 |
| ρ (leverage correlation) | −0.385 | 0.001 |
| θ, ξ, v₀ | reliable as before | — |

2-day Q-measure stability: std(κ)=0.01, std(ρ)=0.001 — highly stable. Round-trip error <4%.

**Model comparison**: Heston Carr-Madan FFT beats 3/2 model Monte Carlo by **33.9% RMSE** (0.098 vs 0.132) on the same Deribit BTC IV surface. This justifies the model choice for Part D. IV fit RMSE = **0.0086** (excellent).

**Heston-Q CVaR impact**: tail QQ on 50k paths — const-vol CVaR₉₅ = 74,222 vs Heston-Q CVaR₉₅ = 71,016 — a **4.79% reduction** (p<0.0001, robust at 100k paths). Part D has a clean, defensible positive finding anchored to real calibrated parameters. See §6.

### 1.5 Regime-conditional impact — Yuhao multipliers and Sonnet C per-regime audit (commits 1cc770e, bc7a47d)

**Before (Yuhao merge)**: `regime.py` computed regime-specific γ and η as `sigma × 1e-8` and `sigma × 1e-6`. Magic constants with no microstructure basis.

**After commit `1cc770e`**: Yuhao replaced magic scaling with dimensionless multipliers derived from per-regime sub-sample calibration (~0.8–1.2x of pooled estimate). The commit also added `simulate_regime_execution` in `sde_engine.py` (supporting both rule and pde modes) and extended `RegimeParams` with diagnostic fields.

**Observed impact parameters (V2, post-Yuhao)**:

| Parameter | Risk-On | Risk-Off | Base (pooled) |
|-----------|---------|----------|---------------|
| σ (annualized) | 0.704 | 2.498 | — |
| γ multiplier | 0.798 | 3.093 | 2.672 |
| η multiplier | 0.562 | 7.726 | 2.76e-5 |

**Sonnet C audit (commit `bc7a47d`)**: true per-regime calibration (sub-sample regression, not multipliers) revealed the Yuhao multipliers systematically mismatch the true per-regime impact structure. Risk-off γ is overestimated by **41–469%**; risk-off η is underestimated by **79–90%**. These biases were the proximate cause of the V4 Part E false positive. See §2.2 for the full narrative.

**Note on risk-off sub-sample**: 3-state HMM was tested (commit `bc7a47d`) and found to be bimodal — ghost 3rd state with ~13 observations. 2-state HMM is correct. However, risk-off constitutes only **5.5% of bars (~1500 trades)**, making per-regime regression unstable (R²=0.10 in risk-off, α out of range → η fallback to literature 1e-3).

### 1.6 WRDS failure and alternative sentiment data (commits b032481, 1c95911)

**WRDS MarketPsych**: BU's WRDS subscription contains only `mpsych_sample`. The sample library ends 2024-10-20 and all crypto-related columns return NULL. Hard data access limit — not a viable fallback for any BTC data.

**Fallback**: alternative.me Fear & Greed Index (free public API). `data/fear_greed_btc.json` contains 98 daily rows covering the same date range as the aggTrades data, with F&G index values ranging 5–61. The `align_sentiment_to_returns_bar()` helper (commit `b032481`) handles alignment from daily F&G to the 5-min execution grid. Tested in HMM — rejected. See §5.4.

### 1.7 Exchange data coverage (commit 3180840)

**GLM D**: Binance spot bookDepth is not publicly available. Binance futures bookDepth pulled as proxy (correlation >0.99 with spot depth). Coinbase cross-exchange data pulled: 10,981 candles.

**GLM E**: CoinMetrics FlowIn/FlowOut confirmed available on free tier (98 days, BTC). Used by Sonnet G — see §5.4.

---

## 2. AC vs TWAP — retail vs institutional

### 2.1 Core paired test results (updated to 100k paths, Sonnet A commit 1b6509e)

A paired statistical test on common-random-numbers MC paths. **100k-path run (commit `1b6509e`) flipped the X0=100 result** — 10k had insufficient power (type-II error):

| X0 (BTC) | mean_diff (AC − TWAP) | t-test p (100k) | Significant at α=0.05 | 10k p (prior) |
|----------|-----------------------|-----------------|-----------------------|---------------|
| 1 | +8.33 | 0.79 | no | 0.79 |
| 10 | +48.67 | 0.88 | no | 0.88 |
| **100** | **−2,976** | **0.034** | **yes** | 0.34 (type-II) |
| **1000** | **−376,024** | **<0.0001** | **yes** | <0.0001 |
| **10000** | **−38.4M** | **<0.0001** | **yes** | <0.0001 |

**Retail boundary update**: X0≥100 BTC (not X0≥1000 as stated in V5). See §8.

**Block bootstrap**: AR(1)=0.0023 — IID assumption valid, block bootstrap confirms no material autocorrelation in cost paths.

**Multi-horizon**: CVaR₉₅ benefit scales 2.4x per 6x horizon increase (1h→6h). T=1d degenerates to bang-bang execution. See §7.

**Methodological note**: the earlier `x0_sensitivity_analysis.py` used deterministic `execution_cost()` which understates AC's benefit. The MC paired test exposes the real trajectory × stochastic-price interaction. Prefer MC mean as the reference metric.

### 2.2 Regime-aware paired test — V1→V5 progression (V4 INVALIDATED)

The Part E regime-aware vs single-regime execution test has gone through five distinct iterations. **V4 is now invalidated by V5** (commit `6bab5b1`):

| Version | Window | Params | Metric | Result | Status |
|---------|--------|--------|--------|--------|--------|
| V1 | ~7d | σ×1e-8/1e-6 magic | mean cost | p=0.84 | Invalidated — magic params |
| V2 (commit `1cc770e`) | ~7d | Yuhao multipliers | mean cost | p=0.84 | Null — wrong metric |
| V3 (commit `ec5fe1d`) | 98d | Yuhao multipliers | mean cost | p=0.84 | Null — wrong metric |
| **V4 (commit `f6c3ace`)** | 98d | Yuhao multipliers | CVaR₉₅ | **−14.0%, p<0.0001** | **INVALIDATED by V5/commit `6bab5b1`** |
| **V5 (commit `6bab5b1`)** | 98d | True per-regime calibration | CVaR₉₅ | **+227.5%, p<0.0001 (reversed)** | Suspect — η fallback in risk-off |
| V6 | ~280d (pending) | True per-regime calibration | CVaR₉₅ | Worker I in flight | Pending |

**Why V4 was wrong**: V4 used Yuhao's sigma-based multipliers to construct per-regime ACParams. Sonnet C (commit `bc7a47d`) discovered those multipliers overestimate risk-off γ by 41–469% and underestimate risk-off η by 79–90%. The inflated γ made risk-off execution look expensive, causing the regime-aware scheduler to shift more execution into risk-on — appearing as tail-risk reduction. It was an artifact of biased params, not a genuine regime benefit.

**Why V5 is also suspect**: when rebuilt with true per-regime calibration, risk-off η falls to literature fallback (η=1e-3) because the risk-off sub-sample is too small (5.5% of bars, ~1500 trades) to support stable regression (R²=0.10, α out of range). The +227.5% CVaR reversal may therefore reflect η fallback noise rather than true regime-aware behavior.

**Methodological learning — each version revealed a distinct bias layer**:
- V1: wrong model (σ×1e-8 magic scaling, no microstructure basis)
- V2: wrong sample size (7 days, HMM not stable)
- V3: wrong metric (mean cost; AC objective is cost + λ·risk)
- V4: right metric, wrong params (Yuhao multipliers inflated risk-off γ by 469%)
- V5: right params, wrong data window (risk-off sub-sample too small → η fallback)
- V6 (pending): extended 98→280 day window to resolve sub-sample size issue

**Net status**: Part E regime-aware benefit is **currently unresolved** on this data window. Worker I (agent a747d017f273d719e) is running in parallel to extend the data window from 98 → ~280 days. If the extended window yields a stable risk-off calibration (R²>0.3, α in-range), the V6 paired test will be definitive.

---

## 3. Walk-forward OOS validation

Four train/test splits on 98 days of Binance aggTrades:

| Split | σ train | σ test | σ drift | Det. savings IS | Det. savings OOS | Degradation |
|-------|---------|--------|---------|-----------------|------------------|-------------|
| Jan → Feb | 0.32 | 0.68 | +110% | -0.00% | -0.00% | ~0 |
| Feb → Mar | 0.67 | 0.49 | -26% | -0.01% | -0.01% | ~0 |
| Mar → Apr | 0.49 | 0.39 | -20% | -0.00% | -0.00% | ~0 |
| JanFeb → MarApr | 0.53 | 0.46 | -14% | -0.01% | -0.01% | ~0 |

No OOS degradation, and also no OOS benefit on the deterministic metric — consistent with the retail-size finding above. At 10-BTC default, AC is TWAP with extra decimals.

Sigma drift between train and test is substantial in the first split (+110%). The calibrated γ/η don't obviously destabilize across regimes, but the HMM fit would: consider regime-aware refitting for any production-style scoring.

---

## 4. Test suite maturity

The 2026-04 audit rewrote large parts of the suite. Before: 129 tests, many of them tautological or RNG-checks. After: **157 passing + 3 xfailed + 2 xpassed**, with each rewrite's docstring naming the algorithmic invariant it tests.

Categories now present:

- **Z-injection deterministic tests** (`test_heston_simulation.py::TestHestonZInjection`, `TestHestonFullTruncationAlgorithm`): patch the RNG, inject known Z arrays, assert Heston Euler steps match hand-calculated values within 1e-12. Catches bugs stochastic tests can't (wrong truncation floor, asymmetric `max(v, 0)` in drift vs diffusion, Cholesky sign flip).
- **Round-trip calibration tests** (`test_heston_roundtrip.py`, `test_hmm_roundtrip.py`): simulate with known params → fit → check recovery. This is how the κ/ρ unreliability in 1.4 was discovered.
- **Data-pipeline invariants** (`test_data_pipeline_invariants.py`): lock in the γ-sign invariant, plain-numpy dtypes, proxy-status documentation for `compute_mid_prices`, HMM NaN rejection.
- **Structural identity tests** (e.g., `test_execution_fees_matches_closed_form_identity`): replace narrative bounds with mathematical identities that are regime-independent.
- **Regime-path tests** (`tests/test_derive_regime_path.py`, 18 tests, commit `626c3e1`): cover all three modes of `derive_regime_path` — "current" (use last HMM state as constant, default for short execution windows), "historical" (replay observed regime sequence), "sample" (draw from stationary distribution). The 3-mode bridge replaced the single-mode `align_states_to_execution_grid`.
- **Multi-feature HMM tests** (`tests/test_multifeature_hmm.py`, 7 tests, commit `b032481`): validate that `fit_hmm()` accepts 2-D feature matrices (T, d) without breaking univariate backward compatibility.
- **Regime-conditional MC integration test** (`tests/test_regime_conditional_mc.py`, commit `1cc770e`): end-to-end test of `simulate_regime_execution` in rule and pde modes.

### 4.1 Soft-deleted tests (xfailed with documentation)

| Test | Reason |
|------|--------|
| `test_gk_diverges_more_than_rs_under_drift_xfail` | At 5-min bar frequency, per-bar drift (~1e-5) is ~4 orders of magnitude smaller than per-bar vol (~1e-3), so GK's known drift bias is not detectable. Only visible at daily+ bars. |

### 4.2 Tests removed in spirit (replaced with structural versions)

`test_heston_output_shapes`, `test_heston_initial_conditions`, `test_heston_variance_starts_at_v0`, `test_heston_reproducible`, `test_heston_costs_finite`, `test_heston_mean_cost_reasonable`, `test_twap_boundary`, `test_twap_uniform`, `test_vwap_boundary`, `test_vwap_sums_to_x0`, `test_zero_inventory_zero_cost`, `test_binance_fee_materially_erodes_reported_savings`.

---

## 5. Known open limitations

### 5.1 VWAP-as-mid-price proxy

`calibration/data_loader.py::compute_mid_prices` computes per-bar VWAP and labels it `mid_price`. The docstring explicitly calls this a proxy. **The project has 10,213 Tardis L2 snapshots that could give the real best-bid / best-ask mid**, but they aren't wired in. `test_mid_price_function_documents_proxy_status` prevents the disclaimer from being silently removed without actual L2 integration.

### 5.2 Magic fallback constants

If all estimation cascades fail, we fall back to literature constants: γ=1e-4, η=1e-3, κ=5.0, ξ=0.8, ρ=0.0. These have no citation. `CalibrationResult.sources` flags which were used so downstream consumers can detect degraded results.

### 5.3 Regime-conditional impact — Yuhao multiplier bias quantified (commit bc7a47d)

**Original text (2026-04-18)** stated: "regime-conditional impact is cosmetic — extensions/regime.py scales per-regime γ/η as sigma×1e-8 and sigma×1e-6."

**That specific bug was fixed in commit `1cc770e` (Yuhao, 2026-04-20).** Impact parameters now use economically plausible dimensionless multipliers.

**Sonnet C audit (commit `bc7a47d`, 2026-04-21)** quantified the remaining bias in those multipliers: risk-off γ is overestimated by **41–469%**; risk-off η is underestimated by **79–90%** relative to true per-regime sub-sample calibration. These biases produced the V4 false positive (CVaR₉₅ −14%, now invalidated).

**Current status**: true per-regime calibration is implemented but requires a larger risk-off sub-sample (risk-off = 5.5% of bars → R²=0.10 in regression → η fallback). Worker I extending data to ~280 days is the path to resolving this. Part E remains unresolved. See §2.2.

### 5.4 HMM — bivariate features: generalized daily-frequency mismatch (commits b032481, 1c95911, cb2e6fa)

**Original hypothesis (2026-04-18 §5.4)**: "Adding on-chain features would make regime switches interpretable without adding compute cost."

**F&G experiment (commits b032481, 1c95911)**: `fit_hmm()` extended to accept 2-D feature matrices. F&G (98 daily rows) vs univariate log_return:

| Model | σ_on (annualized) | σ_off (annualized) | Spread |
|-------|------|-------|--------|
| Univariate (log_return only) | 0.674 | 2.357 | **250%** |
| Bivariate (log_return + F&G) | 0.776 | 1.123 | **45%** |

Adding sentiment dilutes regime separation by 5.6x. Hypothesis rejected.

**CoinMetrics FlowIn/FlowOut experiment (Sonnet G, commit `cb2e6fa`)**: exchange FlowIn fails identically — spread drops from **249.8% to ~46%** regardless of feature variant (raw, log-scale, +FlowOut). All three variants tested, all three fail.

**Generalized hypothesis**: the failure mode is structural, not signal-specific. **Daily-frequency features (F&G, exchange FlowIn, FlowOut) are mismatched with 5-min vol regimes.** The HMM gets pulled by the daily aggregation structure and loses intraday vol separation. This applies to any daily-sampled exogenous feature. A higher-frequency proxy (5-min on-chain whale-alert frequency, 5-min Twitter volume) remains untested but is out of scope for the current project timeline.

---

## 6. Part D clean findings — Heston Q-measure via Deribit (Sonnet B commit fbc5fe9, GLM F commit 3180840)

Part D now has a clean, defensible positive finding anchored to real options-calibrated parameters:

| Finding | Value | Evidence |
|---------|-------|---------|
| Heston Q-measure κ | 9.09 | Carr-Madan FFT on Deribit BTC options |
| Heston Q-measure ρ | −0.385 | same |
| 2-day stability std(κ) | 0.01 | repeated calibration |
| 2-day stability std(ρ) | 0.001 | repeated calibration |
| Round-trip error | <4% | simulate → calibrate → compare |
| IV surface fit RMSE | 0.0086 | excellent |
| Heston vs 3/2 model RMSE | 0.098 vs 0.132 | Heston wins by 33.9% |
| Heston-Q CVaR₉₅ (50k paths) | 71,016 | vs const-vol 74,222 |
| CVaR₉₅ reduction vs const-vol | **4.79%** | p<0.0001, robust at 100k paths |

**Narrative**: the Heston model choice is justified empirically (beats 3/2 by 33.9% RMSE), calibrated parameters are stable and reliable (Deribit resolved the κ/ρ unreliability from §1.4), and the tail-risk benefit of using Heston-Q over const-vol is statistically significant. Part D is the project's most robust positive quantitative finding.

---

## 7. Multi-horizon finding (Sonnet A, commit 1b6509e)

CVaR₉₅ benefit from AC vs TWAP scales with execution horizon. At X0=1000 BTC:

| Horizon | CVaR₉₅ benefit | Relative to 1h |
|---------|----------------|----------------|
| 1h | baseline | 1.0x |
| 6h | ~2.4x baseline | 2.4x |
| 1d | degenerate (bang-bang) | N/A |

**Interpretation**: AC's risk-reduction advantage grows as the execution window lengthens — the optimizer has more time steps to smooth the trajectory and reduce variance accumulation. At T=1d (24h), the AC solution degenerates to bang-bang execution and is not a meaningful test case.

**Implication for project narrative**: short-horizon (1h) tests understate AC's benefit. Multi-horizon reporting (1h and 6h) is preferred for the final paper.

---

## 8. Retail boundary update (Sonnet A, commit 1b6509e)

V5 stated: "retail boundary is X0≥1000 BTC." This was based on 10k-path paired tests.

**Updated**: retail boundary is **X0≥100 BTC** based on 100k-path tests. The 10k-path run at X0=100 BTC had a type-II error (p=0.34 at 10k → p=0.034 at 100k — significant).

| X0 | 10k p | 100k p | Conclusion |
|----|-------|--------|-----------|
| 10 | 0.88 | 0.88 | Not significant (confirmed) |
| 100 | 0.34 | **0.034** | **Type-II error caught** |
| 1000 | <0.0001 | <0.0001 | Significant (confirmed) |

**Practical implication**: AC is useful for any position ≥100 BTC (~$10M at current prices), not only institutional 1000+ BTC blocks.

---

## 9. Confidence summary (V6)

| Claim | Confidence | Evidence | Notes |
|-------|-----------|---------|-------|
| σ calibration reliable | high | GK + RS cross-check, 20% tolerance | — |
| γ calibration reliable | medium | aggregated, R²=0.18 | Real but noisy |
| η calibration reliable | medium | aggregated, lit-range α | — |
| HMM separates regimes (univariate, 98d) | high | 250% σ spread; round-trip stable | commit `1c95911` |
| HMM with daily-frequency features amplifies regimes | **none** | F&G: 250%→45% (§5.4); CoinMetrics FlowIn: 249.8%→46% (§5.4) | Structural temporal mismatch |
| Heston θ/ξ reliable | medium | 20% recovery, ξ within factor of 2 | — |
| Heston κ/ρ reliable | **high** | Deribit FFT; std(κ)=0.01, std(ρ)=0.001; round-trip <4% | Updated from low — Deribit resolved |
| Heston beats 3/2 model | high | RMSE 0.098 vs 0.132 (−33.9%) | commit `fbc5fe9` |
| Heston-Q reduces CVaR₉₅ vs const-vol | high | 4.79% reduction, p<0.0001, 100k paths | Part D anchor finding |
| AC beats TWAP at ≤10 BTC (mean cost) | no | paired test p>0.79 | — |
| AC beats TWAP at 100 BTC (mean cost) | medium | p=0.034 at 100k (type-II at 10k) | commit `1b6509e` |
| AC beats TWAP at 1000+ BTC (mean cost) | high | p<0.0001, 50% savings | — |
| **Regime-aware reduces tail risk (VaR/CVaR)** | 🟡 **unresolved** | V4 CVaR₉₅ −14% (p<0.0001) INVALIDATED by V5 +227.5% reversal; V5 suspect due to η fallback | V4 commit `f6c3ace` invalidated by V5 commit `6bab5b1`; Worker I in flight |
| CVaR benefit scales with horizon | high | 2.4x from 1h→6h | commit `1b6509e` |
| Full Truncation scheme correct | high | Z-injection test at 1e-12 precision | — |
| Fees modeled correctly | high | closed-form identity test | — |

---

## 10. Pending decisions (for team discussion)

| # | Decision | Context |
|---|----------|---------|
| P1-6 | Add POV as a formal benchmark alongside TWAP | Already implemented (`pov_trajectory`). Still need paired tests against AC at each X0. ~1 hour. |
| ~~P1-5~~ | ~~Integrate Deribit option chain~~ | **DONE** — commit `fbc5fe9`. Heston κ/ρ resolved. |
| P1-10 | Lead narrative: retail "AC ≈ TWAP at ≤10 BTC" or "boundary is 100 BTC" or institutional "50% savings"? | 100k results shift the story. |
| P1-11 | Await Worker I (data extension 98→280d) | Agent a747d017f273d719e extending data window. If risk-off sub-sample becomes sufficient (R²>0.3, α in-range), V6 Part E will be definitive. |
| P1-12 | Report multi-horizon (1h, 6h) in final paper | Short-horizon understates AC benefit. Use 6h as primary horizon for institutional case. |

---

*Last updated 2026-04-21 by Eva (V6: Part E V4 invalidated + V5 reversal §2.2; Deribit Heston Q-measure §1.4 + §6; multiplier bias quantification §1.5 + §5.3; CoinMetrics FlowIn rejection §5.4; 100k paired tests + retail boundary §2.1 + §8; multi-horizon §7; confidence summary §9 downgrade. Previous: V5 2026-04-20, original audit 2026-04-18.)*
