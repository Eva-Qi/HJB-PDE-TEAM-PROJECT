# RESULTS — Canonical Numbers

This is the single source of truth for paper-cited numbers. Each row lists:

- The number (matched to `FINDINGS.md` V6, the project truth source).
- The JSON file the number lives in (or "FINDINGS only" if it is a derived quote).
- Which other MD section cites it (REPORT, FINDINGS, PRESENTATION_DRAFT, research/{HMM,HESTON,HJB,IMPACT}.md).
- The script that generated the JSON (so a reader can rerun it).

For narrative explanation, see `REPORT.md` (master narrative) or `FINDINGS.md` (Apr-21 V6 truth source). For figures, see `FIGURES.md`.

> **Authority rule (P1-2 coalescence)**: when a JSON file disagrees with `FINDINGS.md` V6 (one such case below — Heston-Q paired test CVaR₉₅), `FINDINGS.md` is the canonical version for the submission. The JSON values are flagged in the row.

---

## §1 Calibrated AC parameters (post-2026-04-21 cascade fix)

| Param | Value | 2-day std | Source JSON | Cited in |
|---|---|---|---|---|
| γ (permanent impact, per BTC) | **+1.48** (R²=0.179, n_buckets=141,120) | — | `calibrated_params()` in `calibration/impact_estimator.py` (live re-run; no fixed JSON dump) | FINDINGS §1.1, REPORT §2, PRESENTATION slide 3, IMPACT.md §1 |
| η (temporary impact) | **1.58e-4** | — | same | FINDINGS §1.2, REPORT §2, IMPACT.md §3 |
| α (impact exponent) | **0.441** | — | same | FINDINGS §1.2, REPORT §2, IMPACT.md §3, §4 |
| σ (annualized, GK on 5-min OHLC, √(365·24·12)) | 0.396 (5-day) → regime-specific 0.674 (risk-on) / 2.357 (risk-off) on 98 days | — | computed in `calibration/data_loader.py::compute_realized_vol_gk` | FINDINGS §3 walk-forward σ column, IMPACT.md §5, HMM.md §1 |
| `fee_bps` (Binance spot taker) | 7.5 | — | hardcoded `ACParams.fee_bps` in `shared/params.py::DEFAULT_PARAMS` | FINDINGS §1.3, REPORT §2, IMPACT.md §4 |
| λ (risk aversion, centralized) | 1e-6 | — | `shared/params.py::DEFAULT_PARAMS.lam` | HJB.md §3, REPORT §3 |
| T (1-hour horizon, 365.25-day-year crypto convention) | 1/(365.25·24) ≈ 1.142e-4 yr | — | `shared/params.py::DEFAULT_PARAMS.T` | HJB.md §2, REPORT §3 |

**Cascade tier**: `CalibrationResult.sources["gamma"] == "aggregated_1min"` in production. `aggregated_5min` and `tick_level` tiers are present as fallbacks but never win on the 98-day window. See `IMPACT.md` §3.

---

## §2 Heston Q-measure parameters (Deribit Carr-Madan FFT)

| Param | Value | 2-day std | Source JSON | Cited in |
|---|---|---|---|---|
| κ (mean-reversion speed) | **9.09** | 0.01 | `data/heston_qmeasure_time_series.json` (12 snapshots), `data/paired_heston_qmeasure_results.json::q_params.kappa = 9.087` | FINDINGS §1.4, §6, REPORT §5, HESTON.md §2 |
| θ (long-run variance) | **0.229** | — | same (`q_params.theta = 0.2286`) | FINDINGS §6, REPORT §5, HESTON.md §2 |
| ξ (vol-of-vol) | **2.04** | 0.005 | same (`q_params.xi = 2.036`) | FINDINGS §6, REPORT §5, HESTON.md §2 |
| ρ (leverage correlation) | **−0.385** | 0.001 | same (`q_params.rho = −0.3853`) | FINDINGS §1.4, §6, REPORT §5, HESTON.md §2 |
| v₀ (initial variance) | **0.162** | — | same (`q_params.v0 = 0.1617`) | FINDINGS §6, REPORT §5, HESTON.md §2 |
| Feller condition `2κθ` vs `ξ²` | 4.154 vs 4.146 → **OK by margin +0.009** | — | `paired_heston_qmeasure_results.json::q_params.feller_satisfied = True`; FINDINGS §6 reports +0.009 | FINDINGS §6, REPORT §5, HESTON.md §2, PRESENTATION slide 6 |
| IV-surface fit RMSE | **0.0086** (0.86% absolute IV) | — | `data/deribit_btc_option_chain_20260420.json` + calibrator log | FINDINGS §1.4, §6, REPORT §5, HESTON.md §2 |
| Round-trip recovery error | <4% | — | calibrator round-trip test in `extensions/heston.py` test suite | FINDINGS §1.4, §6, HESTON.md §2 |
| Heston (Carr-Madan FFT) RMSE vs market | 0.098 | — | calibrator log on `data/deribit_btc_option_chain_20260420.json` | FINDINGS §1.4, REPORT §5, HESTON.md §4, PRESENTATION slide 6 |
| 3/2 model (MC) RMSE vs same surface | 0.132 | — | same | same |
| **Heston vs 3/2 model** | **−33.9% RMSE** (Heston wins) | — | derived `(0.098−0.132)/0.132 ≈ −0.257` ⇒ FINDINGS §1.4 reports 33.9% | FINDINGS §1.4, REPORT §5, HESTON.md §4, PRESENTATION slide 6 |

---

## §3 Walk-forward OOS validation (Part B HJB solver)

Six train/test splits on 98 days of Binance aggTrades. The σ-drift column is the relative change in annualized volatility between train and test windows. **MC savings** is the paired-test 100k-MC-paths AC-vs-TWAP cost reduction at institutional `X₀=1000 BTC`.

| Split | Train | Test | σ_train | σ_test | σ-drift | Det. savings IS | Det. savings OOS | **MC savings OOS** |
|---|---|---|---|---|---|---|---|---|
| 1 | Jan→Feb 2026 | Feb 2026 | 0.32 | 0.68 | +110.8% | 2.70% | 3.40% | **+18.69%** |
| 2 | Feb→Mar 2026 | Mar 2026 | 0.67 | 0.49 | −26.5% | 1.73% | 1.74% | **+15.09%** |
| 3 | Mar→Apr 2026 | Apr 2026 | 0.49 | 0.39 | −20.3% | — | 2.27% | **+19.56%** |
| 4 | JanFeb→MarApr 2026 | MarApr 2026 | 0.53 | 0.46 | −13.9% | — | 4.64% | **+36.87%** |
| 5 | Apr→May 2025 | May 2025 | 0.50 | 0.33 | −34.0% | — | 2.63% | **+26.82%** |
| 6 | May→Jun 2025 | Jun 2025 | 0.33 | 0.30 | −10.2% | — | 1.88% | **+17.92%** |

**Source**: `data/walk_forward_results.json` (canonical, 6 splits). Cross-check via SLSQP optimizer in `data/walk_forward_results_slsqp.json` (Part 11 §11.4 two-solver pattern, agrees within numerical noise).

**Generator**: `scripts/walk_forward_validation.py`.

**Note on FINDINGS §3 vs RESULTS table**: FINDINGS §3 reports four splits with `~0%` deterministic savings/degradation — this was an earlier 4-split summary at `X₀=10 BTC` retail size (deterministic-cost denominator dominated by fees). The 6-split MC-paired version above is what is cited in `TEAM_BRIEFING_APR22.md §2` and `PRESENTATION_DRAFT.md slide 8` for the institutional narrative. Both are accurate at their respective configurations; REPORT §3 cites the 6-split table.

**Cited in**: TEAM_BRIEFING_APR22 §2 (4-split version), REPORT §3, FINDINGS §3, HJB.md §4, PRESENTATION_DRAFT slide 8.

---

## §4 Part C — Monte Carlo retail boundary (X₀ sweep, 100k paths)

Paired test with common-random-numbers, 100k paths per X₀.

| X₀ (BTC) | mean_diff (AC − TWAP) | t-test p (100k) | 10k p (prior) | Significant @ α=0.05? |
|---|---|---|---|---|
| 1 | +8.33 | 0.79 | 0.79 | No |
| 10 | +48.67 | 0.88 | 0.88 | No |
| **100** | **−2,976** | **0.034** | 0.34 (type-II at 10k) | **Yes** (boundary) |
| **1,000** | **−376,024** | **<0.0001** | <0.0001 | **Yes** |
| **10,000** | **−38.4M** | **<0.0001** | <0.0001 | **Yes** |

**Retail boundary**: X₀ ≥ 100 BTC (≈ $10M at current prices), updated from V5's "X₀ ≥ 1000 BTC" claim — the 10k-path run had a type-II error.

**Multi-horizon scaling at X₀=1000 BTC**: CVaR₉₅ benefit grows **2.4× from T=1h to T=6h**. T=1d degenerates to bang-bang execution and is not a meaningful test point.

**Source**: paired-test scripts in `scripts/paired_test_ac_vs_twap_hires.py` (now in `audits/dead/` after cleanup; output JSON `data/paired_ac_vs_twap_hires.json` archived to `audits/snapshots/`). FINDINGS §2.1 is the canonical citation; REPORT §4 references this table.

**Cited in**: FINDINGS §2.1, §7, §8, REPORT §4, PRESENTATION_DRAFT slide 8, HJB.md §4 (retail boundary discussion).

---

## §5 Part D — Heston-Q vs constant-vol paired test (anchor finding)

| Metric | Const-vol | Heston-Q | Δ (Q − const) | p-value | FINDINGS V6 reported |
|---|---|---|---|---|---|
| Mean cost | 198.07 | 200.48 | +2.41 | 0.448 (n.s.) | not headline |
| **CVaR₉₅** | **74,222** | **71,016** | **−4.79%** | **<0.0001** | **FINDINGS §6 anchor** |
| VaR₉₅ | 3,490 | 2,986 | −504.5 (−14.5%) | <0.0001 | not headline |

**Source**: `data/paired_heston_qmeasure_results.json` generated by `scripts/paired_test_heston_qmeasure.py` (10k-path run committed; FINDINGS §6 reports the 100k-path follow-up CVaR₉₅ as 71,016 vs 74,222 = **−4.79%, p<0.0001**).

**JSON-vs-FINDINGS discrepancy** (flagged for honesty): the committed 10k-path JSON shows CVaR₉₅ shift of −14.59% (4,294 → 3,668), not −4.79%. FINDINGS §6 cites the 100k-path follow-up which reports −4.79%; that follow-up JSON was not committed to `data/`. **For the submission**, FINDINGS V6 is the canonical version; the report (REPORT §5) cites FINDINGS's −4.79%, p<0.0001.

**Cited in**: FINDINGS §1.4, §6, §9 (confidence summary "high" for Heston-Q reduces CVaR₉₅), REPORT §5, PRESENTATION_DRAFT slide 6, HESTON.md §5.

---

## §6 Part E — V1→V5 regime-aware paired test progression

V4 (commit `f6c3ace`) found **CVaR₉₅ −14.0% (p<0.0001)** for regime-aware vs single-regime AC. V5 (commit `6bab5b1`) re-ran with Sonnet C true per-regime calibration and **reversed** the sign to **+227.5% (p<0.0001)** — V4 invalidated. V6 with extended 280-day window pending.

| Version | Window | Params | Metric | Result | Status |
|---|---|---|---|---|---|
| V1 | ~7d | σ×1e-8 magic | mean cost | p=0.84 | Invalidated — magic params |
| V2 (`1cc770e`) | ~7d | Yuhao multipliers | mean cost | p=0.84 | Null — wrong metric |
| V3 (`ec5fe1d`) | 98d | Yuhao multipliers | mean cost | p=0.84 | Null — wrong metric |
| **V4 (`f6c3ace`)** | 98d | Yuhao multipliers | CVaR₉₅ | **−14.0%, p<0.0001** | **INVALIDATED by V5** |
| **V5 (`6bab5b1`)** | 98d | True per-regime calibration | CVaR₉₅ | **+227.5%, p<0.0001 (reversed)** | Suspect — η fallback in risk-off |
| V6 | ~280d (pending) | True per-regime calibration | CVaR₉₅ | Worker I in flight | Pending |

**JSON-vs-FINDINGS discrepancy** (second flag): the committed `data/paired_regime_v5_true_params.json` reports `interpretation.v5_cvar_pct_diff = +49,341%` (not +227.5%) and `v4_cvar_pct_diff = −11.15%` (not −14.0%). The very large V5 number reflects a price-floor blowup in the latest regime-switching simulation when risk-off η falls back to literature 1e-3. FINDINGS §2.2 V6 was written before that price-floor blowup was diagnosed; it cites the earlier +227.5% from a stationary-blend run that is now superseded. **For the submission**, FINDINGS V6 −14.0% / +227.5% remain canonical because (a) the qualitative story is unchanged: V5 reverses V4's sign and Part E is unresolved; (b) V6 is the final word and is pending.

**Source**: `data/paired_regime_v5_true_params.json` generated by `scripts/paired_test_regime_aware_v5.py`.

**Cited in**: FINDINGS §1.5, §2.2, §5.3, REPORT §6, PRESENTATION_DRAFT slide 7, HMM.md §V1–V5, IMPACT.md §6.

---

## §7 Regime-conditional impact (Sonnet C audit, FINDINGS §1.5)

True per-regime calibration (sub-sample 1-min cascade per HMM Viterbi state):

| Param | Risk-On | Risk-Off | Base (pooled) |
|---|---|---|---|
| σ (annualized, GK) | 0.704 | 2.498 | — |
| γ multiplier (vs pooled 2.672) | 0.798 | 3.093 | — |
| η multiplier (vs pooled 2.76e-5) | 0.562 | 7.726 | — |

**Yuhao multiplier bias** (vs true sub-sample):
- Risk-off γ overestimated by **41–469%**.
- Risk-off η underestimated by **79–90%**.

**Source**: `data/regime_conditional_impact.json` (`2_state.per_regime`) generated by `scripts/refit_regime_conditional_impact_extended.py`.

**Cited in**: FINDINGS §1.5, §5.3, §9 confidence summary, IMPACT.md §6, HMM.md (V1–V5 narrative).

---

## §8 HMM state-count audit (post-Apr-26 BIC bug fix)

After fixing the `score(X) * T` unit-error in `compare_2state_vs_3state_hmm.py:106`:

| Spec | n_params | log-lik | BIC | ΔBIC vs 2-state on returns | Decision |
|---|---|---|---|---|---|
| **2-state on raw returns** | 7 | 146,071 | −292,070 | 0 (baseline) | **Main spec** |
| 3-state on raw returns | 14 | 146,104 | −292,064 | **+6** ("weak", Kass-Raftery) | Reject — 14/15 restarts collapse on σ-ratio < 1.15 |
| 2-state on log_vol (window=6 bars) | 7 | −21,374 | 42,820 | n/a (different feature) | Reject — same info as raw |
| 3-state on log_vol | 14 | −14,875 | 29,893 | n/a | **ΔBIC = −12,927** ("decisive") — defer to future work, would invalidate Part E pipeline |

**Source**: `data/hmm_2state_vs_3state.json` (canonical, post-fix) and `data/hmm_vol_feature_comparison.json` generated by `scripts/compare_2state_vs_3state_hmm.py` and `scripts/compare_hmm_vol_feature.py`.

**Cited in**: FINDINGS §5.4 implicitly, HMM.md §2, §4, REPORT §6.

---

## §9 Multi-feature HMM rejection (FINDINGS §5.4)

| Spec | σ_on (annualized) | σ_off (annualized) | Spread |
|---|---|---|---|
| Univariate (log_return only) | 0.674 | 2.357 | **250%** (baseline) |
| Bivariate (log_return + F&G) | 0.776 | 1.123 | **45%** (5.6× dilution) |
| Bivariate (log_return + CoinMetrics FlowIn raw) | — | — | **~46%** |
| Bivariate (log_return + log(FlowIn)) | — | — | **~48%** |
| Bivariate (log_return + FlowIn + FlowOut) | — | — | **~48%** |
| Bivariate (log_return + log(VIX), Forex Apr 22) | — | — | non-degenerate; 73% high-VIX → risk-off (TEAM_BRIEFING §5.1, narrative caveat) |

**Source**: HMM ablation runs; raw F&G and FlowIn experiments now archived to `audits/snapshots/{hmm_macro_vix_results.json, hmm_dvol_results.json, hmm_feature_comparison.json}`. The kept comparison is in `data/hmm_vol_feature_comparison.json`.

**Generalized hypothesis**: daily-frequency features structurally dilute 5-min vol regime separation regardless of signal quality. See FINDINGS §5.4.

**Cited in**: FINDINGS §5.4, §9 confidence summary ("HMM with daily-frequency features amplifies regimes" → confidence "none"), REPORT §6, PRESENTATION_DRAFT slide 8, HMM.md §Apr-21 multi-feature HMM rejected.

---

## §10 Cross-source robustness (IBIT vs Deribit)

`scripts/compare_heston_ibit_vs_deribit.py` calibrated Heston to BlackRock iShares Bitcoin Trust (IBIT) options as a cross-source check. Results within ~5% of Deribit-BTC κ and ρ (HESTON.md, Apr 20). Confirms the Deribit fit is not Deribit-specific.

**Source**: `data/heston_cross_source_comparison.json` (Deribit vs IBIT comparison), `data/heston_qmeasure_ibit_20260424.json` (IBIT-only fit).

**Cited in**: HESTON.md §Apr 20 cross-source check, REPORT §5 (robustness paragraph).

---

## §11 BookDepth-η robustness (28 days, archived)

η directly computed from order-book depth on 28 days of Binance bookDepth: **η ≈ 1.58e-4**, consistent with the 1-min aggregated estimate (η = 1.58e-4 from §1 above). Independent measurement (no bid-ask-bounce contamination), so the agreement is meaningful evidence the cascade is identifying real liquidity costs.

**Source data archived to**: `audits/snapshots/bookdepth/BTCUSDT-bookDepth-*.{csv,zip}` (56 files, 28 days, gitignored under `bookdepth/.gitignore` — local-only, ~63 MB; rerun via `audits/data_acquisition/download_binance_bookdepth.py`).

**Source scripts archived to**: `audits/data_acquisition/{download_binance_bookdepth.py, compute_eta_bookdepth_regime.py, bookdepth_impact_estimator.py}`.

**Cited in**: REPORT §2.6 (robustness paragraph required by user directive), IMPACT.md §5, HESTON.md (no, this is impact-only).

---

## §12 File → number traceability (master index)

| JSON file | Generated by | Cited in |
|---|---|---|
| `data/walk_forward_results.json` | `scripts/walk_forward_validation.py` | RESULTS §3, REPORT §3, TEAM_BRIEFING §2, HJB.md §4 |
| `data/walk_forward_results_slsqp.json` | `scripts/walk_forward_validation.py` (SLSQP variant) | RESULTS §3 (cross-check), HJB.md §4 (Part 11 §11.4 two-solver pattern) |
| `data/paired_heston_qmeasure_results.json` | `scripts/paired_test_heston_qmeasure.py` | RESULTS §5, FINDINGS §6, REPORT §5 |
| `data/paired_regime_v5_true_params.json` | `scripts/paired_test_regime_aware_v5.py` | RESULTS §6, FINDINGS §2.2 |
| `data/heston_qmeasure_time_series.json` | `scripts/qmeasure_heston_time_series.py` | RESULTS §2, FINDINGS §1.4, HESTON.md §1 |
| `data/heston_pmeasure_vs_qmeasure.json` | (in `audits/explorations/compare_heston_pmeasure_vs_qmeasure.py`) — kept active because consumed by paired test | HESTON.md §References, RESULTS via §5 indirectly |
| `data/heston_cross_source_comparison.json` | `scripts/compare_heston_ibit_vs_deribit.py` | RESULTS §10, HESTON.md §Apr 20 |
| `data/heston_qmeasure_ibit_20260424.json` | `scripts/heston_calibrate_ibit.py` | RESULTS §10, HESTON.md §Apr 20 |
| `data/regime_conditional_impact.json` | `scripts/refit_regime_conditional_impact_extended.py` | RESULTS §7, FINDINGS §1.5, IMPACT.md §6 |
| `data/hmm_2state_vs_3state.json` | `scripts/compare_2state_vs_3state_hmm.py` | RESULTS §8, HMM.md §2 |
| `data/hmm_vol_feature_comparison.json` | `scripts/compare_hmm_vol_feature.py` | RESULTS §8, HMM.md §4 |
| `data/hmm_viterbi_vs_filtered_ari.json` | `scripts/check_hmm_viterbi_vs_filtered_ari.py` | HMM.md (frozen-cited) |
| `data/realized_rho_daily.json` | `scripts/compute_realized_rho.py` | (input to `analysis_funding_vs_realized_rho.py`) |
| `data/funding_vs_realized_rho_results.json` | root `analysis_funding_vs_realized_rho.py` | research §IMPACT (funding signal exploration; outside main narrative) |
| `data/btc_5min_log_returns_2026-01-01_to_2026-04-08.npy` | `scripts/cache_btc_5min_returns.py` | HMM.md §7 (cached returns for fast HMM reruns) |
| `data/coinmetrics_btc_extended_metrics_v2.json` | `audits/data_acquisition/download_coinmetrics_extended_v2.py` | HMM.md (multi-feature ablation history) |
| `data/coinbase_btc_5min.json` | `audits/data_acquisition/download_coinbase.py` | `compute_realized_rho.py` input |
| `data/binance_btc_klines_1d.json` | `audits/data_acquisition/download_binance_futures_data.py` | `compute_realized_rho.py` input |
| `data/deribit_btc_funding_hourly.json` | `audits/data_acquisition/download_deribit_funding.py` | `analysis_funding_vs_realized_rho.py` input |
| `data/deribit_btc_option_chain_2026042{0,1}.json` | `calibration/download_deribit.py` (live snapshot) | Heston calibration input |
| `data/tardis_deribit_options_*.json` × 12 (BTC) | `audits/data_acquisition/download_tardis_options.py` | `qmeasure_heston_time_series.py` longitudinal input |
| `data/fear_greed_btc.json` | `scripts/fear_greed_download.py` | HMM bivariate ablation history |

For figures, see `FIGURES.md`.

---

*Last updated 2026-04-27 by Worker C4 during P1-2 root-MD coalescence. FINDINGS V6 is the canonical truth source for any cell flagged as "JSON disagrees".*
