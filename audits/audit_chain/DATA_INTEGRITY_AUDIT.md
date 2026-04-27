# Data Integrity Audit — 2026-04-21

**Audience**: Project team (P1–P3) and course presentation reviewers.
**Method**: Code Council Part 9 Quant Pipeline Audit (7-layer traps + 8-step diagnostic).
**Scope**: Read-only. No source files modified.

---

## Executive Summary

| Severity | Count |
|----------|-------|
| Critical (P0) | 4 |
| Material (P1) | 6 |
| Informational (P2) | 5 |

**Bottom line**: The pipeline is well-instrumented but has four issues that can mislead a presentation audience if not addressed. The most dangerous is that `n_regime_trades = 0` for every regime in the stored results file, meaning all per-regime calibrations in `regime_conditional_impact.json` ran on zero trades — they are pure literature fallback, not data-driven estimates. The second critical issue is the positive `rho` (+0.61) in the P-measure time-series data, which is the wrong sign for any stochastic vol model fit to BTC. The third is that `walk_forward_results.json` sets `used_linear_approx: true` on all four splits while reporting `alpha` values of 0.30–0.66 — a linear AC approximation is only valid at `alpha=1`, so these results are economically inconsistent. The fourth is the overflow/divide warnings produced during Q-measure calibration that are stored in the results file but never analyzed.

---

## Part 1 — NaN Handling

### 1.1 Calibration Layer

**`calibration/data_loader.py` — `compute_ohlc`**

- Uses `resample().ohlc().dropna()`. Bars with zero trades produce NaN OHLC (open=close=NaN from no price). `dropna()` silently removes them. This is correct behavior but undocumented — a single-trade bar is kept (open=high=low=close=that price), which feeds through to GK estimator as `u-d=0`, `cc=0`, and GK variance = 0 for that bar. Net effect: GK vol is slightly downward-biased on thin-volume days.
- There is no guard against single-trade bars in `compute_ohlc`. The GK estimator (`estimate_realized_vol_gk`) clips to `valid = (o>0)&(h>0)&(l>0)&(c>0)` so these bars pass. With only one trade in a bar, H=L=O=C, so GK var per bar = 0. This is mathematically valid but represents microstructure noise rather than true realized vol.

**`calibration/data_loader.py` — `compute_mid_prices` (VWAP proxy)**

- The proxy disclaimer is present in the docstring (test `TestMidPriceProxyDocumented` would pass). No mid-price NaN produced unless a bar has zero trades (empty bucket → notional=0, volume=0 → 0/0 = NaN). The subsequent `dropna()` handles this. No issue here beyond the known proxy limitation.

**`calibration/impact_estimator.py` — `calibrated_params_per_regime`**

- Trades before the first 5-min bar are dropped via `merged.dropna(subset=["regime"])`. This is correct. However, the function uses `float(trades_df["price"].iloc[-1])` for S0 in fallback paths — this references the **full** dataset's last price, not the regime-specific sub-sample's last price. Under normal conditions this is fine, but in backtest replays with historical regimes, it can create a stale S0 reference.

**`calibration/impact_estimator.py` — `estimate_realized_vol_rs`**

- Line 450: `rs_var_per_bar = np.clip(rs_var_per_bar, 0, None)`. Negative RS variance is a real signal (microstructure noise or data error), but the code silently clips to 0 without warning. The GK estimator does not clip. This asymmetry means RS will always be ≤ GK even on normal data, inflating the apparent drift gap (`drift_gap = abs(sigma-sigma_rs)/sigma`) and potentially triggering the >10% warning spuriously.

### 1.2 Simulation Layer

**`montecarlo/sde_engine.py` — `simulate_heston_execution`**

- Full Truncation scheme: `v_plus = np.maximum(var_paths[:, k], 0.0)`. The stored `var_paths` can go negative, and the docstring explicitly says "may contain small negative values". There is **no downstream clamp** applied when `var_paths` is returned. Callers that use `var_paths` directly (e.g., for analytics or plotting) will see negative variance entries without warning.

**`montecarlo/sde_engine.py` — `simulate_regime_execution`**

- `regime_path` is validated for length (must equal `base_params.N`) but not for valid label values. If a label outside `{0, 1}` appears (e.g., from a 3-state HMM), `_regime_params_from_label` silently returns `risk_off_params` for any label ≠ 0. This is a silent fallback, not a validation error.

### 1.3 Regime Layer

**`extensions/regime.py` — `fit_hmm` / `_validate_returns`**

- NaN/Inf input causes a hard `ValueError` with a clear message. This is correct and the test `test_hmm_rejects_nan_input_with_clear_error` locks it in.
- **Multi-feature path**: `align_sentiment_to_returns_bar` drops rows where either log_return OR sentiment is NaN. This is documented. However, the drop is asymmetric by design — if sentiment is NaN for a day but returns are valid, that entire day's 5-min bars are excluded from HMM training. For a daily sentiment series over 98 days aligned to 28,223 5-min bars, any missing sentiment day removes ~288 bars. The code does not report how many bars were dropped.

**`extensions/regime.py` — `_fit_hmm_rolling_vol` fallback**

- State sequence alignment: `state_sequence[:window-1] = states_feature[0]`. This forward-fills the first `window-1` observations with the first post-window state. If the market opens in a very different regime than the feature-space state for bar 0, this is a look-ahead contamination for the first `window-1` bars. With `window=min(20, max(10, len/20))`, this is typically 10–20 bars. For a 5-min bar series, this affects the first 50–100 minutes.

---

## Part 2 — Clips / Bounds / Fallbacks

### 2.1 Alpha bound [0.3, 1.5]

The acceptance window for alpha is `0.3 <= alpha <= 1.5`. From `alpha_estimation_results.json`:

- **5-min aggregation on 26 out of 98 days** produces `alpha_5min < 0.3` (values as low as −0.023 on 2026-03-27 and 0.046 on 2026-03-13). These days fail the 5-min acceptance check.
- **1-min aggregation is more stable**: all 98 days produce `alpha_1min` in range [0.18, 1.11], though 2026-02-09 (0.179) is marginally below the 0.3 threshold, meaning even 1-min can fail.
- Consequence: The alpha bound causes many per-regime calibrations to fall back to `alpha=0.6`. This is not documented prominently in results files.

### 2.2 R² threshold 0.05 (gamma) and 0.05 (impact)

- For gamma, the cascade threshold is `r_squared >= 0.01`. From `alpha_estimation_results.json`, gamma R² at 1-min is 0.08–0.33 across days, so this threshold rarely activates on the full dataset. For per-regime sub-samples (much smaller), R² can be below 0.01.
- For impact, `r_squared >= 0.05`. On 5-min, many days show `r2_5min < 0.05` (e.g., 2026-03-10: 0.006, 2026-03-13: 0.003). When both freq fail, fallback `eta=1e-3, alpha=0.6` silently activates.

### 2.3 Fallback constants — prevalence

From `regime_conditional_impact.json`:
- **All five regime entries (2-state risk_on, 2-state risk_off, 3-state risk_on, 3-state neutral, 3-state risk_off) have `n_regime_trades = 0`.**
- This means every "per-regime calibrated" result in this file ran `calibrated_params_per_regime` on an empty sub-sample and hit the `len(sub) < 200` early-exit fallback.
- The "true" per-regime params shown in `paired_regime_v5_true_params.json` are labeled as calibrated, but `risk_off.eta_source = "fallback"` and `risk_off.alpha_source = "fallback"` with `eta=1e-3, alpha=0.6`.
- **Summary**: risk_off regime impact parameters (eta, alpha) are **never empirically calibrated** — always fallback. They are used in all regime-aware paired tests and walk-forward analysis.

### 2.4 Heston calibration — kappa clip [0.5, 20.0] and xi clip [0.1, 3.0]

`calibrate_heston_from_spot` clips kappa and xi with documented fallbacks. The P-measure time series shows kappa = 2.17–2.19, xi = 0.021–0.028 (well within bounds). The Q-measure calibration uses L-BFGS-B bounds `kappa in [0.1, 10]`, `xi in [0.01, 3.0]`. From `heston_pmeasure_vs_qmeasure.json`, Q-measure result: kappa=9.09, xi=2.04 — kappa is close to the upper bound of 10.0. A param hitting its bound is a fit quality warning.

### 2.5 GK vs RS drift gap >10% warning

The code emits a warning if `abs(GK - RS)/GK > 0.10`, but the warning is stored in the `CalibrationResult.warnings` list and never surfaced to the user unless they inspect the object. The walk_forward_results.json records `sigma_drift_pct` values of 110.8% (Split-1), 26.5% (Split-2), 20.3% (Split-3), and 13.9% (Split-4). These represent train_sigma vs test_sigma drift, not GK vs RS drift. The GK vs RS drift is a separate measure not stored in any results file.

---

## Part 3 — Normalization / Scaling

### 3.1 Regime multiplier semantics (fixed in v5, but beware v4 artifacts)

The comment at the top of `extensions/regime.py` documents: "sigma, gamma, eta in RegimeParams are dimensionless multipliers (~0.8–1.2x)." This was a fix from the prior audit which found magic constants 8 orders of magnitude below calibrated values.

From `regime_conditional_impact.json`:
- `risk_off` sigma_mult = 2.50 (2-state) or 2.85 (3-state). These are 2.5×–2.85× multipliers — substantially more than the "~0.8–1.2x" range mentioned in the comment. The comment is misleading for risk_off; it should say "can range up to ~3x."
- `risk_on` sigma_mult = 0.70 (2-state), which is within normal range.

### 3.2 Walk-forward: `used_linear_approx: true` on all splits with non-linear alpha

All four splits in `walk_forward_results.json` report `used_linear_approx: true`. The reported alpha values are:
- Split-1: alpha=0.661
- Split-2: alpha=0.337
- Split-3: alpha=0.301
- Split-4: alpha=0.520

The Almgren-Chriss closed-form solution is only valid for `alpha=1` (linear temporary impact). For `alpha != 1`, the PDE solver is required. Using linear approximation with alpha=0.30–0.66 introduces a modeling inconsistency. The OOS cost differences (0.001–0.003 pp degradation) may be artifacts of the linear approximation rather than true out-of-sample generalization.

### 3.3 V5 "true params" objective function is 657% worse than single-regime

From `paired_regime_v5_true_params.json`, V5 results:
- `objective.aware = 2185` vs `objective.single = 288` → 657% worse
- `var_95.aware = 73974` vs `var_95.single = 22184` → 233% worse
- `mean_cost.aware = -21.4` vs `mean_cost.single = 104.1` → negative (regime-aware gets "paid" on average)

A negative mean cost combined with 657% worse risk objective indicates the risk_off parameters (`sigma=1.17, gamma=1.45, eta=1e-3`) are producing numerically pathological behavior. The `eta=1e-3` fallback combined with `alpha=0.6` (both literature fallbacks) appears to be the driver. This result should not be presented as a valid regime-aware cost comparison.

---

## Part 4 — Proxy Variables

### 4.1 VWAP as mid-price proxy

`compute_mid_prices` uses VWAP as a proxy for true mid-price (bid/ask midpoint). The docstring acknowledges this. `load_orderbook_snapshots` is `NotImplementedError`. This is a known limitation, documented, and acceptable. Test `TestMidPriceProxyDocumented` enforces the disclaimer.

### 4.2 Q-measure `rho` positive sign — economic anomaly

From `heston_qmeasure_time_series.json`, the P-measure (spot-based) calibration produces `rho = +0.61`. From `heston_pmeasure_vs_qmeasure.json`, the Q-measure calibration produces `rho = -0.385`.

- Positive `rho` in a stochastic vol model for an equity-like asset (BTC) contradicts the leverage effect: when price falls, vol should rise, implying **negative** rho.
- The P-measure `rho` estimation uses `corr(log_returns, delta_rv)` where `rv` is rolling GK variance. GK variance is a lagged rolling mean, which smooths and delays the variance response. Delta of a smoothed series correlates with future returns, not past returns, potentially reversing the sign.
- This is a **proxy variable trap** (9.5): the rv proxy changes the sign of the estimated correlation. The P-measure rho should be treated as suspect and not presented alongside Q-measure rho without this caveat.

### 4.3 gamma for P-measure from `calibrate_heston_from_spot`

The spot-based calibration uses `Var(rv) / theta` to estimate xi, and uses ACF lag-1 of the **overlapping** rolling window series. Overlapping windows inflate autocorrelation, causing `rho_1` to overestimate the true lag-1 autocorrelation, causing kappa to be underestimated. This is a known bias in the overlapping window estimator. Code comment at line 1280 acknowledges the issue but does not correct for it.

---

## Part 5 — Unit Consistency

### 5.1 Gamma units

From `alpha_estimation_results.json`, overall gamma = 2.67 (base_params in regime_conditional_impact.json). From walk_forward splits, gamma = 1.87e-5 to 2.44e-5. These differ by ~5 orders of magnitude. Investigating:

- `alpha_estimation_results.json` stores gamma from the full 98-day aggregated calibration, where the regression regresses `delta_price` vs `net_flow` in BTC units. With BTC at $88K and flows in BTC, Kyle's lambda has units of $/BTC/BTC = $/BTC². This gives gamma ≈ 2.67 when flows are in BTC.
- Walk_forward gamma (1.87e-5) appears to be from a different unit convention (e.g., flows normalized by ADV, or quantities in USD).
- The exact unit derivation is not documented consistently across files. Two different results files report gamma differing by 5 orders of magnitude for the same market. **This is a unit mismatch or universe mismatch requiring clarification.**

### 5.2 Eta units

Base `eta = 2.76e-5` (regime_conditional_impact.json, full calibration). Walk_forward eta = 8.16e-5 to 2.44e-4. Risk_off fallback eta = 1e-3. These differ by up to 36x. The 1-min aggregated regression returns eta as `exp(intercept)` from `log(|return|) = alpha * log(|net_flow|) + log(eta)` — this is a dimensionless return per unit flow, not dollars. The AC cost model `temporary_impact(v, eta, alpha) = eta * v^alpha` treats eta as having units consistent with `cost_per_share = eta * v^alpha`, but v is in BTC/year and cost is in $/BTC. Unit homogeneity should be formally checked.

### 5.3 DVOL units

`deribit_dvol_btc_daily.json` stores DVOL values (open, high, low, close) in implied volatility points (e.g., open=45.07). These are in **percent** not decimal. If any code reads DVOL and uses it as a decimal vol input without dividing by 100, it would overstate sigma by 100x. There is no code path visible that directly consumes this file (it appears to be stored for reference). Informational.

### 5.4 Fear & Greed Index scale

`fear_greed_btc.json` values range 20–44 (0–100 scale). When used as a sentiment feature in `align_sentiment_to_returns_bar`, these are in absolute index units, not z-scored or normalized. The HMM feature matrix would have column 0 (log_return, ~1e-3 scale) and column 1 (sentiment, ~20–80 scale). The ~40,000x scale difference means sentiment dominates the GaussianHMM covariance fit and effectively absorbs all regime separation signal. This explains the `hmm_bivariate_coinmetrics_results.json` finding that bivariate HMM dilutes the regime spread from 250% to 45% — the sentiment column overwhelms the returns column. The sentiment should be z-scored before inclusion.

---

## Part 6 — Cross-File Consistency

### 6.1 `n_regime_trades = 0` in regime_conditional_impact.json

All five per-regime entries record `n_regime_trades = 0`. This means the state sequence was produced (HMM ran) but the trade-level merge assigned zero trades to each regime bucket. This could happen if:
1. The `bar_timestamps` index used in `calibrated_params_per_regime` was tz-naive while trade timestamps were tz-aware, causing the `merge_asof` to fail silently.
2. The trade data was not available at calibration time and a stub was used.

Either way, the per-regime calibration in that file is not data-driven. The `sources` fields show `gamma: "aggregated_1min"` for risk_on (not "insufficient_data"), which is inconsistent with `n_regime_trades = 0`. This suggests the reporting and the calibration code paths are misaligned — possibly the calibration ran on a different dataset than what produced the zero trade counts.

### 6.2 BIC preference for 3-state HMM with collapse flag

`hmm_2state_vs_3state.json` reports:
- `three_state.collapsed = true` with note that the 3-state solution has a "phantom state" with <1% occupancy.
- Yet `preferred_model = "3-state"` based on `delta_bic = -4138933` (lower BIC).

Preferring a collapsed 3-state model over a stable 2-state model is statistically questionable. The BIC difference is enormous (likely because n=28,223 observations), but the model is functionally degenerate. The presentation should clarify that 3-state is preferred by BIC arithmetic but not functionally meaningful.

### 6.3 Q-measure time series has only 2 data points

`heston_qmeasure_time_series.json` contains only 2 entries (2026-04-20 and 2026-04-21). With 2 data points, there is no "time series" to analyze — the file describes a snapshot comparison, not a time series. Any analysis framing this as a "per-month Feller status" check would be misleading.

### 6.4 Overflow warnings in Q-measure calibration

`heston_pmeasure_vs_qmeasure.json` contains 100+ stored `"overflow encountered in power"` and `"invalid value encountered in divide"` warnings from the FFT/characteristic function evaluations during L-BFGS-B optimization. These are from degenerate parameter proposals during multi-start search (expected behavior in the loss landscape). The final calibrated parameters satisfy Feller (`feller_satisfied: true`, margin = 0.009). The warnings are stored but never analyzed — they should either be suppressed (they are expected) or summarized (e.g., "N of M optimizer starts produced warnings").

### 6.5 Regime label semantics consistency

"risk_on = low vol" is consistent across all modules:
- `extensions/regime.py`: sorts states by ascending vol, labels 0=risk_on, 1=risk_off.
- `montecarlo/sde_engine.py` `_regime_params_from_label`: 0 → risk_on, 1 → risk_off.
- `regime_conditional_impact.json`: regime_idx 0 = risk_on, lower sigma_mult.
- No inconsistency found.

### 6.6 Walk-forward date ranges have no overlap gaps

Splits 1–3 use non-overlapping test windows (Jan→Feb, Feb→Mar, Mar→Apr). Split 4 uses a combined train (Jan–Mar) and overlapping test (Mar–Apr). Split-3's test (2026-03-29 to 2026-04-08) overlaps with Split-4's train (ends 2026-03-01) — they don't overlap. No data leakage between splits detected.

---

## Part 7 — Test Coverage Gaps

### 7.1 No test for `n_regime_trades > 0`

`test_data_pipeline_invariants.py` does not assert that `calibrated_params_per_regime` produces regimes with `n_regime_trades > 0`. A test should verify that the merge_asof alignment produces non-zero trade counts for each regime when real data is available.

### 7.2 No test for Feller condition enforcement

`simulate_heston_execution` warns on Feller violation but continues. No test asserts that the warning fires or that downstream variance paths remain bounded. The variance paths can go deeply negative and this is not tested.

### 7.3 No test for sentiment normalization

No test checks that the sentiment feature is z-scored (or explicitly documented as not needing to be) before bivariate HMM. The scale mismatch (9.3) is untested.

### 7.4 No test for `derive_regime_path(mode='historical')` with T < N

When `T < N` (historical state sequence shorter than execution grid), the `_block_mode` function clips to the last block. This is a silent fill. No test exercises this edge case.

### 7.5 `TestFallbackConstantsExplicit` does not test eta/alpha fallbacks

The test checks `gamma=1e-4` is returned as None by `estimate_kyle_lambda`. It does not check that `eta=1e-3, alpha=0.6` literature fallback fires when both freq aggregations fail. There is no regression test for the fallback activation logic in `calibrated_params`.

---

## Critical Issues — Detailed Findings

### CRITICAL-1: `n_regime_trades = 0` — All Per-Regime Calibrations Are Spurious

**Location**: `data/regime_conditional_impact.json`, all entries; `data/paired_regime_v5_true_params.json` (risk_off fallback)

**Finding**: Every per-regime entry in `regime_conditional_impact.json` has `n_regime_trades: 0`. This means `calibrated_params_per_regime` was called but the `merge_asof` between trade-level data and regime state_sequence produced zero trade assignments for every regime. The reported "calibrated" per-regime parameters for sigma, gamma in the 2-state and 3-state experiments are from a `compute_ohlc` of an empty DataFrame — or they are from a separate code path that did not use regime-filtered trades.

**Why it matters**: The entire regime-conditional analysis rests on these parameters. If they are produced from zero trades, the per-regime gamma/eta estimates are either NaN (replaced by fallback 1e-4/1e-3) or came from the full-dataset calibration mislabeled as per-regime.

**Proposed fix**: Verify the `calibrated_params_per_regime` call site and add assertion `assert len(sub) > 0 for sub in regime_subs`. If the tz alignment is the cause, ensure both `bar_timestamps` and `trades_df.timestamp` are UTC-aware before `merge_asof`.

**Effort**: 1 hour to diagnose, 2 hours to fix and re-run.

---

### CRITICAL-2: V5 "True Params" Results Are Numerically Pathological

**Location**: `data/paired_regime_v5_true_params.json` (v5_results section)

**Finding**: The risk_off regime uses literature fallbacks `eta=1e-3, alpha=0.6` (no empirical calibration possible, as documented). With these fallbacks, the regime-aware optimizer produces `objective.aware = 2185` vs `objective.single = 288` — the regime-aware strategy is 657% worse by the risk-adjusted objective, and `mean_cost.aware = -21.4` (negative cost, i.e., profitable). A negative mean cost in a liquidation problem is physically impossible under normal market conditions; it means the cost model assigns negative values to some execution paths, which is a sign of parameter-scale incompatibility between risk_off and risk_on regimes.

**Why it matters**: If this result is presented as a regime-aware comparison, it will appear to show regime-awareness is harmful. The true issue is that the risk_off fallback params (eta=1e-3) are out of scale with risk_on params (eta=1.49e-4), making the risk-adjusted objective numerically dominated by the risk_off paths.

**Proposed fix**: For presentation, either (a) remove the V5 analysis entirely and present only V4 (which uses consistent multiplier scaling), or (b) add a prominent caveat that V5 uses unverified fallback parameters for risk_off and the results are not interpretable. Do not present V5 as a valid calibration comparison.

**Effort**: No code fix needed — disclosure caveat only.

---

### CRITICAL-3: P-Measure rho Sign Error from Overlapping Window Proxy

**Location**: `data/heston_qmeasure_time_series.json` (rho = +0.61 both dates), `extensions/heston.py` lines 1326–1350

**Finding**: The P-measure Heston calibration (`calibrate_heston_from_spot`) estimates rho via `corr(log_returns, delta_rv)` where `rv` is a **rolling overlapping GK variance mean** (window=24 bars). Three issues compound:

1. Rolling means create temporal autocorrelation in the rv series, making delta_rv a lagged-return proxy.
2. The log_return alignment clips the last `n_rho` bars to match delta_rv length — this alignment is off by one bar in the indexing (lines 1329–1332: `lr_aligned[1:]` vs `delta_rv[:n_rho]`).
3. Positive rho (+0.61) contradicts the leverage effect for BTC (the Q-measure gives rho = −0.385, which is economically correct).

**Why it matters**: The P-measure rho is wrong in sign. If used as a prior or for cross-validation against Q-measure, the comparison appears to show P vs Q "disagree" by ~1.0 in rho — but the disagreement is methodological, not economic. Presenting both without this caveat is misleading.

**Proposed fix**: Either (a) replace rolling-mean rv with point-in-time per-bar GK variance for the rho estimation (not the smoothed series), or (b) explicitly label P-measure rho as "suspect due to overlapping window bias" in any presentation. The Q-measure rho (−0.385) is the reliable estimate.

**Effort**: 30 min to implement point-in-time rv for rho, 1 hour to re-run.

---

### CRITICAL-4: Walk-Forward Uses Linear Approximation for Non-Linear Alpha

**Location**: `data/walk_forward_results.json` (all 4 splits: `used_linear_approx: true`)

**Finding**: All four walk-forward splits use the linear AC approximation (`used_linear_approx: true`) but calibrate `alpha` values ranging from 0.301 to 0.661. The AC closed-form solution `x*(t) = X0 * sinh(kappa*(T-t)) / sinh(kappa*T)` is only valid for alpha=1. For alpha != 1, the PDE solver must be used. Using the wrong functional form makes the "optimal" trajectory incorrect, and the OOS savings metrics (0.001–0.004 pp) are not meaningful as model validation — they reflect the approximation error as much as the true strategy performance.

**Why it matters**: This is the primary walk-forward validation shown in the project. If the strategy is computed with the wrong optimizer, the OOS comparison (optimal vs TWAP) is not a valid test of the AC strategy.

**Proposed fix**: Re-run walk-forward with PDE solver for splits where alpha != 1.0 (all four splits). Alternatively, document explicitly that linear approximation is used as a computational shortcut and compare against the PDE solution on one split to quantify approximation error.

**Effort**: 4–8 hours to re-run with PDE solver.

---

## Material Issues — Detailed Findings

### MATERIAL-1: Sentiment Feature Not Z-Scored Before Bivariate HMM

**Location**: `extensions/regime.py` — `align_sentiment_to_returns_bar`; `data/hmm_bivariate_coinmetrics_results.json`

**Finding**: Fear & Greed index (range 0–100) and on-chain FlowIn (range ~$800M–$4.6B USD) are passed directly to the multivariate GaussianHMM without normalization. Log_return is on the scale 1e-3. The 40,000x–4,600,000x scale difference means the HMM covariance is dominated by the sentiment/flow column, effectively destroying the regime signal from log_returns. This is confirmed by `hmm_bivariate_coinmetrics_results.json`: bivariate regime spread drops to 45–48% of the univariate spread.

**Severity**: The bivariate experiment correctly concludes that daily sentiment "dilutes" the 5-min regime signal. But the finding conflates two effects: (a) semantic mismatch between daily and 5-min frequencies, and (b) scale mismatch from missing z-scoring. Effect (b) alone could explain the dilution.

### MATERIAL-2: Risk_off Impact Never Empirically Calibrated

**Location**: `data/regime_conditional_impact.json`, `data/paired_regime_v5_true_params.json`

**Finding**: All `risk_off` regime entries in the stored results have `eta_source = "fallback"` and `alpha_source = "fallback"`. The warnings explain: alpha=0.261 (5-min) and alpha=0.266 (1-min) are below the 0.3 acceptance threshold. This is reproducible from the per-day alpha data: many days in volatile periods (Feb/Mar) show alpha_5min < 0.3. The risk_off regime (high-vol periods) has the most volatile and extreme microstructure — exactly the periods where alpha falls below threshold. The system systematically cannot calibrate the most important regime.

**Recommendation**: Lower the acceptance floor to 0.2 for the risk_off sub-sample (where alpha below 0.3 is physically meaningful — high-vol markets can have near-linear impact), or use the 1-min estimate even if slightly below the 0.3 threshold rather than falling back to 0.6.

### MATERIAL-3: 3-State HMM Phantom State Used in Analysis

**Location**: `data/hmm_2state_vs_3state.json`, `data/regime_conditional_impact.json` (3_state section)

**Finding**: The 3-state HMM has `collapsed: true` with a "neutral" state having probability 0.0545 and `sigma_mult = 0.775` — nearly identical to risk_on (`sigma_mult = 0.761`). The two-state BIC is preferred by 4.1M BIC units. Yet the 3-state results are stored alongside 2-state results with equal weight. Any analysis using 3-state parameters treats a phantom state as a real regime.

### MATERIAL-4: Heston CF Overflow During Q-Measure Calibration

**Location**: `data/heston_pmeasure_vs_qmeasure.json` (warnings list)

**Finding**: 100+ overflow/invalid-divide warnings stored from the Q-measure calibration run. These are from degenerate multi-start proposals (expected). However, the final kappa=9.09 is at 91% of the upper bound (10.0). A parameter at 91% of its bound is a red flag — the true optimum may be at kappa > 10, meaning the L-BFGS-B result is constrained by the bound, not the data. The Feller margin is only 0.009 (`feller_lhs=4.154, feller_rhs=4.146`), meaning the Q-measure calibration is barely Feller-satisfying. A small perturbation to the data could violate Feller.

### MATERIAL-5: Daily Alpha Instability (alpha_1min std = 0.197)

**Location**: `data/alpha_estimation_results.json`

**Finding**: Per-day alpha_1min has `mean=0.365, std=0.197`. The range spans [0.179, 1.106] — a 6x spread. This means the power-law exponent changes dramatically day-to-day. Using a single pooled alpha (0.441 overall) in the AC model papers over this volatility. The pooled alpha is relatively stable (SE=0.003) but the per-day variability suggests alpha is regime-dependent or market-condition-dependent. This invalidates the walk-forward assumption that a "trained" alpha is transferable across windows.

### MATERIAL-6: `simulate_regime_execution` Price SDE is Euler Not Exact

**Location**: `montecarlo/sde_engine.py`, line 714

**Finding**: `s_next = s_prev - perm_impact*dt + sigma*s_prev*np.sqrt(dt)*z`. This is the Euler-Maruyama approximation for GBM. The base `simulate_execution` function supports `scheme="exact"` (log-normal discretization with no discretization error). The regime-switching path always uses Euler. For large `sigma*sqrt(dt)` (relevant for risk_off regime with sigma_mult=2.5), Euler has meaningful discretization bias. Negative prices are possible (no guard), though rare.

---

## What This Audit Does NOT Cover

1. **Actual trade data CSV files**: The fixture tests were not run. The audit is static code and JSON analysis only.
2. **PDE solver (`pde/hjb_solver.py`)**: Grid resolution, boundary conditions, and convergence were not audited.
3. **Deribit option chain data** (`tardis_deribit_options_*.json`): Individual option chain completeness, bid-ask spread distributions, and OI data quality were not checked.
4. **`coinbase_btc_5min.json` and `coinmetrics_btc_extended_metrics.json`**: Content not inspected.
5. **`orderbook_alpha_results.json`**: Content not inspected.
6. **Multi-path simulation variance**: The 10,000-path MC standard errors were not computed; whether p-values are stable across seeds was not tested.
7. **End-to-end pipeline execution**: The audit is based on code reading and JSON inspection, not running the pipeline.
8. **ETH option chain data**: The `tardis_deribit_options_ETH_*.json` files were not inspected; the audit focuses on BTC.
