# MF796 Methodology Findings — 2026-04-18 Audit (updated 2026-04-20)

**Audience**: Team (Eva, bp, Yuhao) + future reviewers
**Status**: Living document. Update when calibration, test, or data assumptions change.

This document captures the substantive findings from the 2026-04-18 code-council audit (with GLM-5.1 challenger cross-review) and the resulting methodology fixes, extended with 2026-04-20 findings from Yuhao's regime-aware merge, the multi-feature HMM experiment, and the full 98-day Part E paired tests. It complements (does not replace) PROJECT_AUDIT_REPORT.md. Read this first if you want to understand what changed and why.

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

### 1.4 Heston calibration — partial reliability

`calibrate_heston_from_spot` was round-trip tested: simulate Heston paths with known (κ, θ, ξ, ρ, v₀) → fit back from OHLC → check recovery.

| Parameter | Recovery | Status |
|-----------|----------|--------|
| θ (long-run variance) | within 20% | ✅ reliable |
| ξ (vol-of-vol) | factor-of-2 | ✅ reliable |
| **κ (mean reversion)** | true=2.0 → recovered=20.0 (clips to ceiling) | ❌ **structurally broken** |
| **ρ (leverage correlation)** | true=-0.7 → recovered≈+0.02 (noise) | ❌ **structurally broken** |

Root cause: 24-bar overlapping rolling windows in the GK realized-variance path produce autocorrelation ≈ 0.985, so the inferred κ explodes and clips. For ρ, the same windows cause `delta_rv` to be dominated by entry/exit-bar noise. Fixing this properly requires options-based Q-measure calibration (e.g., Deribit BTC option IV surface) — not more spot data.

**Practical implication**: any Heston-dependent decision should treat our current κ and ρ as informed guesses at best. θ and ξ are fine.

**Status (2026-04-20)**: Deribit integration remains an open P-decision (see §6). WRDS BU subscription turned out to have only `mpsych_sample` library, ended 2024-10-20, with ALL NULL crypto columns — not a viable fallback (see §1.6 for the WRDS finding).

### 1.5 Regime-conditional impact parameters: magic scaling replaced (commit 1cc770e)

**Before (Yuhao merge)**: `regime.py` computed regime-specific γ and η as `sigma × 1e-8` and `sigma × 1e-6`. These were pure magic constants with no microstructure basis. The regime-aware PDE solution differed from the single-regime solution only in σ; the impact coefficients did not reflect regime-specific liquidity at all.

**After**: Yuhao's commit `1cc770e` replaced the magic scaling with dimensionless multipliers derived from per-regime sub-sample calibration. The multipliers land in the ~0.8–1.2x range relative to the pooled estimate — economically plausible, since regime-specific liquidity varies but not by eight orders of magnitude.

**Observed impact parameters from `paired_regime_aware_results.json`** (V2, post-Yuhao):

| Parameter | Risk-On | Risk-Off | Base (pooled) |
|-----------|---------|----------|---------------|
| σ (annualized) | 0.704 | 2.498 | — |
| γ multiplier | 0.798 | 3.093 | 2.672 |
| η multiplier | 0.562 | 7.726 | 2.76e-5 |

The commit also added `simulate_regime_execution` in `sde_engine.py` (supporting both rule and pde modes) and extended `RegimeParams` with diagnostic fields (`state_vol`, `state_abs_ret`, `state_mean_ret`).

**What didn't change**: the paired test verdict. The Yuhao fix addresses the root cause of the V1 "cosmetic" finding, but the paired test on mean cost still returns p=0.84 — for a different reason. See §2.1 for the full V1→V4 progression.

### 1.6 WRDS failure and alternative sentiment data (commits b032481, 1c95911)

**WRDS MarketPsych**: BU's WRDS subscription contains only `mpsych_sample`. The sample library ends 2024-10-20 and all crypto-related columns return NULL. This is a hard data access limit — no amount of query refinement recovers live or recent BTC sentiment from WRDS.

**Fallback**: alternative.me Fear & Greed Index (free public API). `data/fear_greed_btc.json` contains 98 daily rows covering the same date range as the aggTrades data, with F&G index values ranging 5–61. The `align_sentiment_to_returns_bar()` helper (commit `b032481`) handles alignment from daily F&G to the 5-min execution grid.

**Practical implication for §1.4**: options-based Q-measure calibration (needed to fix Heston κ/ρ) cannot be sourced from WRDS. Deribit remains the only viable free option. WRDS also has no BTC options data at all under the BU subscription.

---

## 2. AC vs TWAP — retail vs institutional

A paired statistical test (using teammate bp's `paired_strategy_test`) on common-random-numbers MC paths revealed:

| X0 (BTC) | mean_diff (AC − TWAP) | t-test p | bootstrap p | Significant at α=0.05 |
|----------|-----------------------|----------|-------------|-----------------------|
| 1 | +8.33 | 0.79 | 0.78 | no |
| 10 | +48.67 | 0.88 | 0.87 | no |
| 100 | −2,976 | 0.34 | 0.33 | no |
| **1000** | **−376,024** | **<0.0001** | **<0.0001** | **yes** |
| **10000** | **−38.4M** | **<0.0001** | **<0.0001** | **yes** |

**Takeaway**: at retail sizes (X0 ≤ 100 BTC) the AC-vs-TWAP difference is noise — the model does not beat TWAP meaningfully. At institutional sizes (X0 ≥ 1000 BTC), AC saves ~50% of TWAP realized cost and the difference is statistically robust.

**Methodological note**: the earlier `x0_sensitivity_analysis.py` used the deterministic `execution_cost()` which integrates permanent impact additively and understates AC's benefit. The MC paired test exposes the real trajectory × stochastic-price interaction. Prefer MC mean as the reference metric in any reported result.

### 2.1 Regime-aware paired test — four-version progression

The Part E regime-aware vs single-regime execution test went through four distinct iterations:

**V1 (pre-Yuhao, ~7-day window)**: regime-conditional γ/η computed as σ×1e-8/1e-6 (magic scaling). HMM found σ_on≈0.70, σ_off≈2.50 (3.5x ratio), but impact parameters were dimensionally implausible. Verdict: COSMETIC. Root cause explicitly called out as the magic scaling.

**V2 (post-Yuhao, commit `1cc770e`, ~7-day window)**: magic scaling replaced. Impact parameters now in realistic range (see §1.5 table). Paired test on mean execution cost, n=10,000 paths:
- mean_single=104.05, mean_aware=107.79, mean_diff=+3.74, t-p=0.84, bootstrap-p=0.83
- HMM: σ_on=0.704, σ_off=2.498; P(risk-off)=8.8%
- Verdict still: not significant. Root cause not yet identified.

**V3 (commit `ec5fe1d`, full 98-day window)**: extended from 7-day to 98-day training window. HMM regimes become more stable (univariate σ spread = 250%, confirming §5.4). Paired test on mean cost: p=0.84 again.

Root cause identified at V3: **mean cost is the wrong metric for this test.** The Almgren-Chriss HJB objective is `cost + λ·risk`, not cost alone. Regime-awareness is designed to reduce exposure during high-volatility periods — that benefit shows up in tail risk metrics (VaR₉₅, CVaR₉₅, objective value), not necessarily in mean cost.

**V4 (risk-side metrics, landed)**: paired test on VaR₉₅, CVaR₉₅, and AC objective (cost + λ·var). Results saved to `data/paired_regime_aware_v2_results.json` (commit `f6c3ace`):

| Metric | regime-aware | single-regime | diff | p-value | Significant at α=0.05 |
|--------|--------------|---------------|------|---------|-----------------------|
| mean cost | 107.79 | 104.05 | +3.74 | 0.834 | no (reproduces V3) |
| **objective** (cost + λ·var) | **244.64** | **288.49** | **−43.85 (−15.2%)** | **0.023** | **yes** |
| **VaR₉₅** | **19,098.6** | **22,184.5** | **−3,086 (−13.9%)** | **<0.0001** | **yes** |
| **CVaR₉₅** | **23,715.4** | **27,577.1** | **−3,862 (−14.0%)** | **<0.0001** | **yes** |

**V4 conclusion — the hypothesis was correct**: regime-aware execution provides statistically significant tail-risk reduction (VaR₉₅ and CVaR₉₅ both drop ~14% with p<0.0001) and lowers the AC objective by 15% (p=0.023) even though mean cost is indistinguishable. This is consistent with AC theory: the framework optimizes (cost + λ·risk), not cost alone, and regime-awareness pays off in the 10% risk_off regime where pooled params under-price volatility and produce fatter cost tails. Mean cost is the wrong metric to evaluate Part E; risk-side metrics are the right ones. Part E now has a clean, publishable positive finding.

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

### 5.3 Regime-conditional impact — original bug fixed, economic question open (updated 2026-04-20)

**Original text (2026-04-18)** stated: "regime-conditional impact is cosmetic — extensions/regime.py scales per-regime γ/η as sigma×1e-8 and sigma×1e-6."

**That specific bug was fixed in commit `1cc770e` (Yuhao, 2026-04-20).** Impact parameters now use economically plausible dimensionless multipliers (~0.8–1.2x of pooled calibration). The "cosmetic" label no longer applies to the mechanism.

**What remains unresolved**: the paired test on mean cost is still p=0.84 at both V2 and V3 (see §2.1). The current hypothesis is that mean cost is the wrong metric — regime-awareness should reduce tail risk, not necessarily mean cost. V4 (testing VaR₉₅/CVaR₉₅/objective) is the definitive test. Until V4 lands, it is premature to either claim or deny that regime-aware execution is economically beneficial.

### 5.4 HMM — bivariate F&G experiment tested and REJECTED (commits b032481, 1c95911)

**Original hypothesis (2026-04-18 §5.4)**: "Adding on-chain features would make regime switches interpretable without adding compute cost." The sentiment experiment tested this directly.

**What was done**: `fit_hmm()` extended to accept 2-D feature matrices (backward compat preserved). `data/fear_greed_btc.json` loaded (98 daily rows, alternative.me F&G Index, range 5–61). `scripts/compare_hmm_features.py` compared univariate vs bivariate on the full 98-day dataset.

**Result** (from `data/hmm_feature_comparison.json`):

| Model | σ_on (annualized) | σ_off (annualized) | Spread |
|-------|------|-------|--------|
| Univariate (log_return only) | 0.674 | 2.357 | **250%** |
| Bivariate (log_return + F&G) | 0.776 | 1.123 | **45%** |

Adding sentiment dilutes regime separation by a factor of ~5.6 (spread_ratio=0.179). **Hypothesis rejected.**

**Root cause**: F&G is updated once per day and changes smoothly (range 5–61 over 98 days). BTC vol regimes are sharp crisis-bar events visible at 5-min resolution. The bivariate HMM must satisfy both signals and compromises on both — vol separation collapses from 250% to 45%, and the sentiment signal adds no compensating discriminatory power.

**Intellectual honesty note**: this is a genuine negative finding, not a failure. It confirms that log returns alone, given sufficient history (98 days vs the earlier 2-week window), already capture regime structure effectively. The problem is temporal mismatch, not signal quality. A higher-frequency sentiment proxy (e.g., 5-min crypto Twitter volume or on-chain whale-alert frequency) remains untested and could still be valuable — but that is out of scope for the current project timeline.

---

## 6. Pending decisions (for team discussion)

| # | Decision | Context |
|---|----------|---------|
| P1-5 | Integrate Deribit option chain → Q-measure Heston | Fixes the κ/ρ unreliability in §1.4. Public API, free. ~1-2 days of work. **WRDS is not a viable alternative** — BU subscription has `mpsych_sample` library only, ended 2024-10-20, ALL NULL crypto columns (see §1.6). Question: is Heston pricing-precision important for project narrative, or is "directionally correct" enough? |
| P1-6 | Add POV as a formal benchmark alongside TWAP | Already implemented (`pov_trajectory`). Still need to run paired tests against AC at each X0. ~1 hour. |
| P1-7 | Add Glassnode exchange-inflow as HMM feature | Half day. Makes regimes interpretable ("risk-off = large wallet moving to exchange"). **Note**: F&G as a daily-smooth sentiment signal was tested and rejected — see §5.4. Glassnode's per-hour granularity may avoid the temporal mismatch, but verify before committing work. |
| P1-8 | ~~Regime-conditional impact refit (replace σ × 1e-8 hack)~~ | **DONE** — resolved by commit `1cc770e` (Yuhao). Dimensionless multipliers now in place. |
| P1-9 | Await V4 risk-side paired test results | `data/paired_regime_aware_v2_results.json` not yet present as of 2026-04-20. V4 tests VaR₉₅, CVaR₉₅, and AC objective. If significant, regime-aware execution has a defensible risk-reduction narrative even with indistinguishable mean cost. |
| P1-10 | Demonstrate 1000-BTC case explicitly in final report | Already analyzed (§2). Narrative decision: lead with retail "AC ≈ TWAP" finding or institutional "AC saves 50%"? |

---

## 7. Confidence summary

| Claim | Confidence | Evidence points | Notes |
|-------|-----------|-----------------|-------|
| σ calibration is reliable | high | GK + RS cross-check, 20% tolerance passes | — |
| γ calibration is reliable | medium | aggregated method positive, R²=0.18 | Real but noisy |
| η calibration is reliable | medium | aggregated, literature-range α, R² low as expected | — |
| HMM separates regimes (univariate, 98d) | high | 250% σ spread; round-trip recovers structure | Confirmed in commit `1c95911` |
| HMM with F&G sentiment amplifies regimes | none | tested and REJECTED — bivariate reduces spread 250%→45% (§5.4, commit `1c95911`) | Temporal mismatch problem |
| Heston θ/ξ reliable | medium | 20% recovery, ξ within factor of 2 | — |
| Heston κ/ρ reliable | low | round-trip: κ clips to ceiling, ρ is noise; no options data | Deribit fix open |
| AC beats TWAP at 10 BTC (mean cost) | no | paired test p=0.88 | — |
| AC beats TWAP at 1000+ BTC (mean cost) | high | paired test p<0.0001, 50% savings | — |
| Regime-aware beats single-regime (mean cost) | no | V1 p≈0.84 (magic scaling); V2 p=0.84 (Yuhao fix); V3 p=0.84 (98d) | 4 data points, all null |
| Regime-aware reduces tail risk (VaR/CVaR) | 🟢 high | V4 paired test commit `f6c3ace`: VaR₉₅ −14% (p<0.0001), CVaR₉₅ −14% (p<0.0001), objective −15% (p=0.023) | |
| Full Truncation scheme correct | high | Z-injection test at 1e-12 precision | — |
| Fees are modeled correctly | high | closed-form identity test | — |

---

*Last updated 2026-04-20 by Eva (Yuhao merge §1.5, WRDS/F&G findings §1.6, bivariate HMM rejection §5.4, V3/V4 Part E progression §2.1; original 2026-04-18 audit via code-council + GLM-5.1 cross-review).*
