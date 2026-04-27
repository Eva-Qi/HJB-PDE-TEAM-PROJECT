# Audit Verification — Opus U, 2026-04-21

**Scope**: Independent re-audit of `DATA_INTEGRITY_AUDIT.md` produced by Sonnet R.
**Method**: Read actual source and JSON files, verify every claim, look for missed issues. CRITICAL-1 and CRITICAL-4 are being fixed in parallel by Opus S and T — spot-checked only for framing, not re-verified in depth.

## TL;DR

- **Audit quality: medium-high.** Sonnet R correctly surfaced most real problems, but made **two factual misattributions** that change the fix direction, and **missed four issues** that are at least as severe as what they flagged.
- **Critical issues confirmed**: 2/4 confirmed as stated, 1/4 refined (misattributed root cause), 1/4 **misdiagnosed** (the reported "bug" is a reporting placeholder, not a calibration defect).
- **Material issues confirmed**: 5/6 confirmed, 1/6 refined.
- **New issues found by this verification: 4.**

---

## Confirmations / refinements of Sonnet R findings

### CRITICAL-1 (`n_regime_trades = 0`) — REFINED, root cause misdiagnosed
Sonnet R read this as "`merge_asof` produced zero trade assignments", and proposed tz alignment as the fix. **Incorrect.** In `scripts/refit_regime_conditional_impact.py` lines 169–173:

```python
"n_regime_trades": int(len(trades[
    trades["timestamp"].isin(trades["timestamp"])
]) if False else 0),  # placeholder; actual count from cal metadata
```

The field is a **hard-coded `0` placeholder** — the `if False else 0` branch is dead code that always returns 0. Meanwhile `sources.gamma = "aggregated_1min"` in the same JSON row proves the per-regime regression actually ran and produced real estimates. `calibrated_params_per_regime` (`calibration/impact_estimator.py:485`) does tz-localize both sides (`tz_localize("UTC")`) before `merge_asof`, so the tz-alignment hypothesis is wrong.

**Actual severity: Material (not Critical).** It is a reporting-bug that made Sonnet R's audit worse, not a calibration failure. The per-regime gamma/sigma numbers are real; only risk_off eta/alpha are fallback (which Sonnet R also captured as MATERIAL-2 — the same finding from a different angle). Opus S's fix should focus on replacing the placeholder with the real count (already done correctly in `refit_regime_conditional_impact_extended.py:240`), **not** on re-running calibration against suspicion that it never ran.

### CRITICAL-2 (V5 pathology) — CONFIRMED but root cause refined
Verified from `data/paired_regime_v5_true_params.json`:
- `objective.aware=2185, objective.single=288` → 657% confirmed.
- `mean_cost.aware=-21.4` confirmed.
- `risk_off.eta_source="fallback", alpha_source="fallback"` confirmed.

**Real driver Sonnet R missed**: Reading `scripts/paired_test_regime_aware_v5.py` lines 407, 378:

```python
costs_v5_aware = w_on * costs_v5_riskon + w_off * costs_v5_riskoff
```

This is **not a regime-aware simulation**. It is a stationary-weighted average of two *unconditional* simulations, each running with one regime's params on every bar for the whole horizon. `costs_v5_riskoff` runs sigma=1.17 (117% annualised vol) for the entire horizon — extreme GBM path divergence interacts with `s_prev - temp_impact` in `_compute_execution_cost_step` (sde_engine.py:517) to produce negative exec_px when the stochastic drift pushes `s_prev` below `temp_impact`. The negative mean cost is therefore a direct consequence of (a) the ill-scaled fallback η and (b) the blended-simulation design — not of regime-awareness itself.

**Fix**: Sonnet R's proposal ("do not present V5") is fine as a cover-up but misses that the underlying `paired_test_regime_aware_v5.py` does not implement true regime-switching. A proper fix uses `simulate_regime_execution` with a regime path, not two separate full-horizon runs.

### CRITICAL-3 (P-measure ρ sign flip) — MISATTRIBUTED
Sonnet R claimed `heston_qmeasure_time_series.json` shows `rho = +0.61` from a P-measure spot-based estimate inside `calibrate_heston_from_spot`, and blamed overlapping-window bias at lines 1326–1350 of `extensions/heston.py`.

**This is wrong on the file attribution.** `scripts/qmeasure_heston_time_series.py:203` calls `calibrate_heston_from_options`, not `calibrate_heston_from_spot`. So the `rho=+0.61` on 20260420–21 is a **Q-measure options-based** estimate from Tardis snapshots, not a P-measure spot-based estimate.

The real contradiction is between **two Q-measure calibrations**:
- `heston_qmeasure_time_series.json` (Tardis Deribit): `rho = +0.61`
- `heston_pmeasure_vs_qmeasure.json` q_measure block (Deribit chain 20260420): `rho = -0.385`
- `heston_pmeasure_vs_qmeasure.json` p_measure block (spot): `rho = -0.0014` (near zero, not +0.61)

**Three separate findings hide here**:
1. Two Q-measure runs on BTC options for essentially the same date produce opposite-sign ρ. This is a **calibration reproducibility failure** — different option filters, weights, or initial guesses flip the sign.
2. The spot-based ρ is ≈ 0, not +0.61 — the leverage-effect anomaly Sonnet R worried about does not actually appear in the spot calibration that is run.
3. Sonnet R's proposed fix (replace overlapping rolling-mean with per-bar point-in-time rv) is aimed at a bug that does not affect the stored results. The off-by-one alignment at `extensions/heston.py:1327–1331` is real (`lr_for_rho = lr_aligned[1:]` vs `delta_rv = np.diff(rv_aligned)` — lr leads Δrv by one bar) but the consequence in stored data is ρ≈0, not ρ=+0.61.

**Correct fix direction**: Investigate why `scripts/qmeasure_heston_time_series.py` produces ρ=+0.61 while the other Q-measure code path produces ρ=−0.385 on the same asset. Options filtering, weighting (uniform vs OI), and initial-guess stability are the likely culprits. Sonnet R's rolling-window diagnosis is unrelated.

### CRITICAL-4 — CONFIRMED (deferred to Opus T). One amplification: in all four splits `is_savings_pct` and `oos_det_savings_pct` are **negative** — AC-optimal underperforms TWAP even in-sample. Linear approximation is part of the story, but the cross-file γ unit inconsistency (see new finding N-1) is also a candidate. Opus T should verify γ units in the optimizer input match the optimizer's expected scale before declaring the linear approximation the culprit.

### MATERIAL-1 to MATERIAL-6 — confirmed
M1 (sentiment scaling), M2 (risk_off never empirically calibrated), M3 (phantom 3-state), M4 (kappa at 91% of bound + razor-thin Feller margin 0.009), M5 (alpha instability), M6 (Euler SDE) all verified. Refinement on M6 below.

**M6 refinement**: `simulate_regime_execution` at `montecarlo/sde_engine.py:714` uses a **hybrid** SDE — arithmetic permanent-impact drift plus multiplicative noise:
```python
s_next = s_prev - perm_impact*dt + sigma*s_prev*np.sqrt(dt)*z
```
This is dimensionally consistent but is neither a pure arithmetic Brownian nor a pure GBM. Negative prices are possible when `sigma*sqrt(dt)*z < -(1 - perm_impact*dt/s_prev)`. With sigma=1.17 (risk_off) and 50 steps over 1 hour, P(`s_next < 0` on at least one step) is non-trivial, which directly feeds the pathology in CRITICAL-2.

### Informational (I1–I5) — confirmed

---

## New findings missed by Sonnet R

### N-1 (CRITICAL): γ unit mismatch propagates into walk-forward optimizer
Sonnet R flagged the 5-order-of-magnitude γ gap in Part 5.1 but described it as "requires clarification". Verified more sharply:

- `calibration/impact_estimator.py:123 estimate_kyle_lambda_aggregated`: regresses `Δprice ($)` on `net_flow (BTC)` → γ in **$/BTC**. This is what produces γ≈2.67 in `regime_conditional_impact.json`.
- `scripts/walk_forward_validation.py:168–175`: regresses **`return (dimensionless)`** on `net_flow (BTC)` → γ in **1/BTC** (return per BTC). This is what produces γ≈1.9e-5 to 2.4e-5 in `walk_forward_results.json`.

Both are then fed into the same `ACParams` and HJB solver, which expects a single consistent convention for the permanent-impact term `γ·v`. Using `1/BTC` as if it were `$/BTC` (or vice versa) silently produces a trajectory optimized against a permanent-impact term that is 5 orders of magnitude off. **This is the likely explanation for why AC-optimal underperforms TWAP on all four splits** — γ is effectively zeroed out, so the optimizer cannot see any permanent-impact trade-off and collapses onto (approximately) TWAP with a small extra penalty.

**Fix**: Standardize on one convention (recommend `Δprice` regression, consistent with `regime_conditional_impact.json`). Rerun walk-forward.

### N-2 (MATERIAL): `paired_regime_aware_results.json` has schema collision between η multipliers and η absolutes
File contents (`data/paired_regime_aware_results.json`):
```json
"regime_etas": { "risk_on": 0.56, "risk_off": 7.72, "base": 2.76e-05 }
```
`base` is an absolute η (consistent with all other files); `risk_on` and `risk_off` are multipliers. A downstream reader cannot distinguish the two. The file's own `verdict` field calls out a "sigma*1e-8 / sigma*1e-6 scaling hack" — a historical artifact that likely related to this schema confusion. If any downstream consumer treats `risk_on=0.56` as an absolute η, it produces eta 4+ orders of magnitude too high.

**Fix**: Rename to `regime_eta_multipliers` and drop `base` from the same dict (put `base_eta` at the top level).

### N-3 (MATERIAL): Silent label-out-of-range fallback in `_regime_params_from_label`
`montecarlo/sde_engine.py:483–494`:
```python
if regime_label == 0:
    return risk_on_params
return risk_off_params
```
Any integer ≠ 0 (including e.g. 2 from a 3-state HMM Viterbi output, or −1 from an error state) silently routes to `risk_off_params`. Sonnet R noted this in Section 1.2 but did not elevate it. Given that the 3-state results **are** stored alongside 2-state (MATERIAL-3), there is a concrete scenario where a 3-state Viterbi sequence gets fed into a 2-regime simulator and every "neutral" bar becomes a risk_off bar, inflating cost estimates.

**Fix**: Replace with `if regime_label not in (0, 1): raise ValueError(...)`.

### N-4 (MATERIAL): Pre-existing `compute_mid_prices` backfill method deprecation in walk_forward
`scripts/walk_forward_validation.py:167`:
```python
dp_bar = (flow_clean["return"] * flow_clean["return"].shift(1).fillna(method="bfill")).values
```
`fillna(method="bfill")` is deprecated in pandas ≥ 2.1; this emits a FutureWarning on every split. The computed `dp_bar` variable is **then overwritten on the next line** by `dp_bars = flow_clean["return"].values` — so the expression is dead code, but it also computes a product of consecutive returns (not a `Δprice`), which is conceptually wrong if anyone were to read it later. **Remove the dead line**.

---

## Assessment of Sonnet R's proposed fixes

| Finding | Sonnet R fix | Verdict |
|---|---|---|
| CRITICAL-1 | Add tz alignment check + assertion on n_trades > 0 | **Wrong direction.** Real bug is the placeholder literal `0` in the reporting code. Fix is a 1-line change in `refit_regime_conditional_impact.py:169–173`, not tz logic. |
| CRITICAL-2 | Disclose caveat; do not present V5 | Partially correct. Also need to note that the "regime-aware" simulation is stationary-blend of two unconditional runs, not true regime switching. |
| CRITICAL-3 | Replace overlapping rolling-mean rv with per-bar rv | **Unrelated to the observed anomaly.** Real task: reconcile two Q-measure calibrations that produce opposite-sign ρ. |
| CRITICAL-4 | Re-run with PDE solver | Correct but insufficient. Also verify γ unit convention (N-1) before blaming linear approximation. |
| MATERIAL-1 | Z-score sentiment | Correct. |
| MATERIAL-2 | Lower alpha floor to 0.2 for risk_off | Correct and simple. |
| MATERIAL-3 | Flag phantom state | Correct. |
| MATERIAL-4 | Flag kappa-at-bound | Correct. |
| MATERIAL-5 | Note alpha instability | Correct, but a further action (regime-conditional α rather than pooled) is worth proposing. |
| MATERIAL-6 | Switch to exact scheme | Correct; also add explicit `np.maximum(s_next, 0)` guard because current code admits negative prices. |

---

## What this verification does NOT cover

- **PDE solver correctness** (`pde/hjb_solver.py`) — grid resolution, boundary conditions, convergence.
- **Actual rerun of the pipeline** — everything here is read-only inspection of code and JSON outputs. Any claim about simulation behaviour is inferred from code reading, not from executing.
- **ETH options data** (`tardis_deribit_options_ETH_*.json`) — the new BTC/ETH comparison branch is out of scope.
- **Coinmetrics on-chain feature pipeline** — bivariate HMM inputs were not read at source level.
- **Feller enforcement in the Heston simulator** — we noted it is warned-but-not-enforced; we did not trace the downstream consequences.
- **Test theatre detection on post-2026-04-19 tests** — a brief scan did not flag new tautologies, but a full audit of the test additions since the previous cleanup is still warranted.

Author: Opus U, 2026-04-21.
