# MF796 Methodology Findings — 2026-04-18 Audit

**Audience**: Team (Eva, bp, Yuhao) + future reviewers
**Status**: Living document. Update when calibration, test, or data assumptions change.

This document captures the substantive findings from the 2026-04-18 code-council audit (with GLM-5.1 challenger cross-review) and the resulting methodology fixes. It complements (does not replace) PROJECT_AUDIT_REPORT.md. Read this first if you want to understand what changed and why.

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

### 5.3 Regime-conditional impact is cosmetic

`extensions/regime.py` scales per-regime γ/η as `sigma × 1e-8` and `sigma × 1e-6`. These are magic scalings, not actual microstructure-derived regime-specific estimates. The regime-aware PDE solution differs by σ only; the impact parameters don't reflect regime-specific liquidity.

### 5.4 HMM uses log returns only

Fitted 2-state HMM cleanly separates a high-vol and low-vol regime (5x σ ratio, 88% / 12% stationary split), but the regimes are noise-driven rather than economically interpretable. Adding on-chain features (exchange inflow, whale count from Glassnode free tier) would make the regime switches interpretable without adding compute cost.

---

## 6. Pending decisions (for team discussion)

| # | Decision | Context |
|---|----------|---------|
| P1-5 | Integrate Deribit option chain → Q-measure Heston | Fixes the κ/ρ unreliability in 1.4. Public API, free. ~1-2 days of work. Is Heston pricing-precision important for this project's narrative, or is execution-side κ/ρ "directionally correct" enough? |
| P1-6 | Add POV as a formal benchmark alongside TWAP | Already implemented (`pov_trajectory`). Still need to run paired tests against AC at each X0. ~1 hour. |
| P1-7 | Add Glassnode exchange-inflow as HMM feature | Half day. Makes regimes interpretable ("risk-off = large wallet moving to exchange"). Not grade-critical. |
| P1-8 | Regime-conditional impact refit (replace σ × 1e-8 hack) | ~3 hours. Refit `estimate_kyle_lambda_aggregated` + `estimate_temporary_impact_aggregated` on regime-tagged sub-samples. Makes regime-aware execution economically meaningful rather than cosmetic. |
| P1-9 | Demonstrate 1000-BTC case explicitly in final report | Already analyzed (section 2). Decision is narrative-level: lead with retail "AC is TWAP" finding or institutional "AC saves 50%" finding? |

---

## 7. Confidence summary

| Claim | Confidence | Rationale |
|-------|-----------|-----------|
| σ calibration is reliable | 🟢 high | GK + RS cross-check, 20% tolerance passes |
| γ calibration is reliable | 🟡 medium | aggregated method returns positive sensible value but R² = 0.18 — real but noisy |
| η calibration is reliable | 🟡 medium | aggregated returns literature-range α, R² low as expected |
| HMM separates regimes | 🟢 high | round-trip test recovers regime structure |
| Heston θ/ξ reliable | 🟡 medium | 20% recovery, xi within factor of 2 |
| Heston κ/ρ reliable | 🔴 low | round-trip shows κ clips and ρ is noise |
| AC beats TWAP at 10 BTC | 🔴 no | paired test p=0.88 |
| AC beats TWAP at 1000+ BTC | 🟢 high | paired test p < 0.0001, 50% savings |
| Full Truncation scheme correct | 🟢 high | Z-injection test at 1e-12 precision |
| Fees are modeled correctly | 🟢 high | closed-form identity test |

---

*Last updated 2026-04-18 by Eva (via code-council audit + GLM-5.1 cross-review + parallel Sonnet worker pipeline).*
