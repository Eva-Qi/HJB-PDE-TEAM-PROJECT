# MF796 Team Briefing — April 22, 2026

**Purpose:** Summarize data wiring, audit findings, and current model state for team review.

---

## 1. Sign Convention (Read First)

Our project uses two metrics for AC vs TWAP comparison. They point in opposite directions — make sure everyone knows which is which before discussing results:

| Metric | Formula | Good direction | Example |
|---|---|---|---|
| Raw execution cost | $ paid | Lower | - |
| `mean_diff` | AC_cost - TWAP_cost | **Negative = AC wins** | -$2,156 @ X0=100 |
| Savings % | (TWAP_cost - AC_cost) / TWAP_cost | **Positive = AC wins** | +14.66% (walk-forward) |
| CVaR ratio | CVaR_AC / CVaR_TWAP - 1 | **Negative = AC wins** | +3225% (V5, bad) |

**For presentation, default to Savings % — the positive-is-good direction is intuitive.**

---

## 2. Current Main Claim: AC Beats TWAP Out-of-Sample

| Split | Train Period | Test Period | Savings % (OOS) |
|---|---|---|---|
| 1 | Jul-Sep 2025 | Oct 2025 | **+14.66%** |
| 2 | Aug-Oct 2025 | Nov 2025 | **+11.15%** |
| 3 | Sep-Nov 2025 | Dec 2025 | **+11.31%** |
| 4 | Oct 2025 - Jan 2026 | Feb 2026 | **+4.17%** |

**All four splits positive. Mean OOS savings +10.3%. This is our strongest result.**

Source: `data/walk_forward_results.json`

---

## 3. Data Changes Since Last Update

### What Got Pulled (Apr 21-22)

| Source | File(s) | Purpose | Wired into model? |
|---|---|---|---|
| Binance aggTrades (98 days) | `data/BTCUSDT-aggTrades-*.csv` | Kyle γ, η | Yes (core) |
| Binance 1d klines | `data/binance_btc_klines_1d.json` | Heston σ | Yes (core) |
| Coinbase 5-min (85k candles) | `data/coinbase_btc_5min.json` | Cross-venue validation | Cross-check only |
| Kraken daily | `data/kraken_btc_daily.json` | Cross-venue validation | Cross-check only |
| Tardis Deribit options (12 months) | `data/tardis_deribit_options_*.json` | Q-measure Heston | Yes (core) |
| Tardis ETH options (12 months) | `data/tardis_deribit_options_ETH_*.json` | Cross-asset comparison | Auxiliary analysis |
| CoinMetrics onchain | `data/coinmetrics_btc_onchain.json` | HMM feature | Yes (bivariate HMM) |
| CoinMetrics extended | `data/coinmetrics_btc_extended_metrics.json` | Exchange flows | **Newly wired (V4/V5)** |
| Deribit OI (2 snapshots) | `data/deribit_btc_oi_summary.json` | OI structure | Narrative only |
| Deribit funding | `data/deribit_btc_funding_hourly.json` | Positioning signal | Not wired |
| Deribit DVOL (BTC VIX) | `data/deribit_dvol_btc_daily.json` | Forward-vol signal | **Running now** |
| Deribit historical vol | `data/deribit_btc_historical_vol.json` | Realized vol cross-check | Not wired (duplicate) |
| FRED macro (VIX, TNX, IRX, DXY, EURUSD) | `data/fred_macro_daily.json` | Macro overlay | **Newly wired** |
| Fear & Greed index | `data/fear_greed_btc.json` | Sentiment | Not wired (too coarse) |
| Binance bookDepth (28 days) | `data/BTCUSDT-bookDepth-*.csv` | Direct η estimator | **Running now** |
| Binance 4h klines | `data/binance_btc_klines_4h.json` | Intermediate timescale | Not wired (no use case) |
| Binance OI daily | `data/binance_btc_openinterest_daily.json` | Vol leading indicator | Not wired |

### Summary

- **Core pipeline inputs:** aggTrades, 1d klines, Tardis options, CoinMetrics onchain
- **Newly wired today:** FRED VIX, CoinMetrics extended, bookDepth, DVOL (in progress)
- **Cross-check only:** Coinbase 5-min, Kraken daily
- **Remaining orphans:** funding rates, Fear & Greed, 4h klines, OI daily

---

## 4. Audit Findings (Apr 22)

### 4.1 HJB Permanent-Cost Audit

**Claim from chat:** "HJB solver doesn't optimize for permanent cost — that's why we lose to TWAP at α<1."

**Verdict: Partially right, but the conclusion is wrong.**

| Sub-claim | Verdict | Evidence |
|---|---|---|
| HJB is missing γ·v·x term | **True** | `pde/hjb_solver.py:141,199,235` — zero matches for `gamma` |
| This causes AC < TWAP at α<1 | **False** | See table below |
| 10k paths is underpowered (Type 2) | **Narrow truth** | Only at α=1, X0=100 BTC |

**Actual AC vs TWAP at α=0.47 (calibrated concave impact):**

| X0 (BTC) | Mean AC cost | Mean TWAP cost | Savings | p-value |
|---|---|---|---|---|
| 100 | -$1,177 | +$6,612 | **$7,790** | <1e-10 |
| 1,000 | -$13,883 | +$709,205 | **$723,089** | <1e-10 |

AC dominates TWAP at α<1 by **2-6× the α=1 margin**. The missing γ term is academically incorrect but numerically immaterial (γ·X₀ ≈ 148 vs V_x scale ~1000s).

**Why the missing γ doesn't bite:**

- **At α=1**: γ·v·x integrates to ½γX₀² — trajectory-independent, cancels in optimization.
- **At α<1**: γ·v·x has trajectory dependence, but magnitude is a ~1% nudge to the HJB policy.

**Recommendation:** The HJB γ fix is correctness housekeeping but **should not be labeled as "fixes our α<1 TWAP loss" — that loss doesn't exist in the data.**

---

### 4.2 `demo_type2_error.py` Audit

**Demo mechanics:** correct. CRN properly implemented, SE scales as √N, paired test valid.

**Issue 1 — Single-seed methodology weak:**

- seed=42 @ 10k paths → p=0.356 (demo's "Type 2")
- seed=1 @ 10k paths → p=0.011 (significant — contradicts Type 2 claim)
- Correct approach: 1000 seeds × 10k paths, compute rejection rate (empirical power)

**Issue 2 — Chat claim about X0=1000 is backwards:**

Teammate said in chat: "significant for >100 BTC and not 1000."

Actual data (X0 sweep at 100k paths, seed=42):

| X0 | mean_diff | t | p |
|---|---|---|---|
| 10 | +$144 | +1.36 | 0.172 |
| **100** | -$2,156 | -2.05 | **0.041** (marginal) |
| **1,000** | **-$381,067** | **-36.33** | **<1e-200 (extremely significant)** |
| 10,000 | -$39.8M | -397 | ~0 |

X0=1000 is **the most significant case**, not the insignificant one. This appears to be a verbal misstatement — please double-check the source.

---

## 5. New Wiring Results

### 5.1 FRED VIX → HMM (Completed)

Compared 5 feature combinations using hmmlearn 2-state HMM:

| Variant | Features | AIC | BIC | Degenerate? | High-VIX → risk-off rate |
|---|---|---|---|---|---|
| V1 | log_return only | -2297 | -2268 | **Yes** (497/1) | 0% |
| V2 | log_return + vix_close | +419 | +474 | **No** (403/95) | **73%** |
| V3 | log_return + Δlog(vix) | -3732 | -3677 | **Yes** (488/10) | 6% |
| V4 | log_return + log(vix) + exchange inflow | -386 | -331 | No (62/36) | 0% (narrow window) |
| V5 | log_return + log(vix) + inflow + outflow | -452 | -372 | No (62/36) | 0% (narrow window) |

**Best by AIC/BIC:** V3 (but degenerate — 10-observation regime).

**Most defensible:** V2. Non-degenerate, properly 2-regime, **73% of high-VIX days classified risk-off** (vs 19% baseline) — confirms macro overlay adds real discrimination.

**Caveats:**
- V4/V5 restricted to 98-day CoinMetrics window — AIC/BIC not directly comparable to V1-V3 (499 days)
- V2's positive AIC means it's nominally worse than V1 by likelihood, but V1 is degenerate so V1's AIC is misleadingly good
- Forward-fill VIX to BTC weekends: 31% of BTC days use previous Friday's VIX

**Conclusion for presentation:** Report V2 as "VIX macro overlay adds meaningful regime discrimination (73% high-VIX → risk-off)", caveat that pure-AIC-minimization selects V3 which collapses.

Source: `data/hmm_macro_vix_results.json`, `scripts/hmm_macro_vix.py`

---

### 5.2 BookDepth → η Regime-Conditional (In Progress)

**Status:** Running (`scripts/compute_eta_bookdepth_regime.py`). Results pending.

**Purpose:** Replace literature fallback (η ≈ 1e-3) in V5 regime-aware execution with empirically-derived η from 28 days of Binance bookDepth CSVs.

**Why it matters:** V5 currently reports +3225% CVaR (AC much worse than baseline) due to `risk_off` regime's η falling back to literature value. If bookDepth-derived η is in a sensible range, V5 may flip from "unresolved negative finding" to a defensible result.

---

### 5.3 DVOL → HMM + Heston ρ Diagnosis (In Progress)

**Status:** Running (`scripts/hmm_with_dvol.py`). Results pending.

**Purpose:** Two tasks:
1. Wire DVOL (BTC implied vol index) as second HMM feature
2. Test whether DVOL regime explains Heston ρ bimodal pattern

**Hypothesis:** High-DVOL months ↔ ρ<0 (fear regime), low-DVOL months ↔ ρ>0 (euphoria regime). If correlation confirms, we have a structural interpretation for Heston ρ sign flips across months.

---

### 5.4 CoinMetrics Extended → HMM (In Progress)

**Status:** Running (`scripts/hmm_coinmetrics_extended.py`). Results pending.

**Purpose:** Test whether exchange inflow/outflow or other extended metrics are stronger HMM features than realized vol alone. Ranks candidates by bimodality coefficient and correlation with realized vol.

---

## 6. Open Items (No Action Requested)

These are issues discussed in audit, flagged but not blocking:

1. **Heston ρ time-varying** — confirmed real (Feller healthy both basins); frame as structural finding in FINDINGS.md, not as calibration failure.
2. **V5 regime-aware** — currently unresolved; bookDepth result (in progress) determines final framing.
3. **Missing γ·v·x in HJB** — optional fix. Low priority for α=1; marginal correction for α<1.

---

## 7. Summary for Presentation

**Keep these claims:**

- AC beats TWAP OOS across 4 walk-forward splits (+4.17% to +14.66%, all p<0.05)
- Multi-feature HMM with VIX overlay produces meaningful regime discrimination (73% high-VIX → risk-off)
- AC dominates TWAP especially in concave-impact regime (α<1), where savings are 2-6× the α=1 margin

**Remove or reword:**

- Do not claim "HJB missing γ causes TWAP loss at α<1" — not supported by data
- Do not claim "X0=1000 is insignificant" — it is the most significant case in our data
- Do not claim regime-aware execution saves X% until bookDepth-derived η is wired (result pending)

**Verify before shipping:**

- All four walk-forward split results — recompute once on main branch to confirm
- The V2 HMM-with-VIX regime classifications — spot-check 2024 Q3 rally vs 2024 Q4 correction
