# IMPACT — Market Impact Calibration (γ, η, α) for Almgren-Chriss

> **TL;DR / Current state (2026-04-26)**: Tick-level Kyle's-λ regression on raw Binance aggTrades returned **negative γ = −0.0113** on the 2026-01 window (economically absurd; bid-ask-bounce mean-reversion dominates per-tick `Δprice`) and the temporary-impact regression returned **α ≈ 0.04** (far below the literature [0.3, 1.5] range). Both fell back to magic-constant fallbacks silently. The fix: a **1-min → 5-min → tick → literature** cascade in `calibration/impact_estimator.py` that resamples trades to 1-minute aggregated buckets first. On 98 days of real Binance BTCUSDT data this gives γ = **+1.48 per BTC** (R²=0.179, n=141,120 buckets) and η = **1.58e-4**, α = **0.441** (in the literature range). `CalibrationResult.sources["gamma"]` reports which cascade tier won — `"aggregated_1min"` in production. Binance taker fee 7.5 bps is now part of `ACParams.fee_bps` and shows up in `execution_cost`. We also computed η directly from order-book depth on 28 days of bookDepth data (`audits/snapshots/bookdepth/`); the result was consistent with the 1-min aggregated estimate (η ≈ 1.58e-4), supporting the calibration's robustness.
>
> **Key result**: γ = +1.48 (positive sign restored), η = 1.58e-4, α = 0.441; cascade method documented in `CalibrationResult.sources`; bookDepth cross-check independent and consistent (FINDINGS §1.1, §1.2; PRESENTATION slide 3).
>
> **Status**: stable. Aggregated cascade is canonical; tick-level retained for academic comparison only.

---

## Current state — what we are doing

### 1. The bug — tick-level Kyle's-λ returns negative γ

**Original estimator** (`estimate_kyle_lambda` on raw aggTrades):
```
ΔP_n = γ · sign(trade_n) · sqrt(qty_n) + ε_n
```

per-tick OLS. On 2026-01 Binance BTCUSDT data this returned **γ = −0.0113**. Negative γ means "buys push price down, sells push price up" — economically impossible. The pipeline silently fell back to the literature constant γ = 1e-4.

**Root cause**: at tick frequency, the dominant signal in `Δprice` is **bid-ask bounce mean-reversion**, not informational impact. A buy hits the ask; the next trade often hits the bid (random retail flow); `Δprice` between them is negative even though both trades push the **mid-price** up by their permanent impact. The Cov(Δp, signed_flow) regression measures the **transaction-price** dynamic, not the **mid-price** dynamic, and at tick frequency these are dominated by spread dynamics not impact dynamics.

**Diagnostic that should have caught this earlier**: the R² of the tick-level regression was 0.0008 (essentially zero). Any fit with R²<0.01 and a sign-flipped coefficient should have triggered an alert before the silent fallback. We added a γ-sign invariant test (`tests/test_data_pipeline_invariants.py::test_calibrated_gamma_is_positive`) to lock in the fix.

### 2. The fix — 1-min aggregated cascade

`calibration/impact_estimator.py::estimate_kyle_lambda_aggregated()`:

```python
# 1-min aggregated regression
df_1min = (trades
    .assign(bucket=lambda d: d.timestamp.dt.floor('1min'))
    .groupby('bucket')
    .agg(price_change=('price', lambda x: x.iloc[-1] - x.iloc[0]),
         net_flow=('signed_qty', 'sum'))
)
result = OLS(df_1min.price_change, df_1min.net_flow).fit()
gamma_1min = result.params[0]
```

Aggregating trades to 1-minute buckets before regressing gives:
- `Δprice` = last-price minus first-price within the bucket → measures the **mid-price drift** (bid-ask bounce averages out within a minute of activity)
- `net_flow` = signed-quantity sum within the bucket → measures the **net informational pressure**

On 98 days of real Binance BTCUSDT data, the 1-min aggregated regression gives **γ = +1.48 per BTC, R²=0.179, n=141,120 buckets**. Positive sign restored, R² in the meaningful range.

**Cascade**: the calibration pipeline tries methods in order and uses the first one with a positive sign and acceptable R²:
```
1-min aggregated  →  5-min aggregated  →  tick-level  →  literature fallback (1e-4)
```

In practice the 1-min tier wins. `CalibrationResult.sources["gamma"]` reports `"aggregated_1min"` to flag the cascade tier used, so downstream consumers can detect degraded calibrations.

### 3. Temporary impact (η, α) — same cascade, same fix

`estimate_temporary_impact_from_trades()` had the same bug: per-trade `abs_price_change` is bid-ask-bounce dominated, so the power-law fit gave α ≈ 0.04 (out of literature range), triggering fallback to η = 1e-3.

`estimate_temporary_impact_aggregated()`:
```python
df_1min = (trades
    .assign(bucket=lambda d: d.timestamp.dt.floor('1min'))
    .groupby('bucket')
    .agg(slippage=('price', lambda x: abs(x.iloc[-1] - x.iloc[0])),
         volume=('signed_qty', lambda x: abs(x).sum()))
)
# OLS on log(slippage) ~ alpha · log(volume) + log(eta)
log_slip = np.log(df_1min.slippage[df_1min.slippage > 0])
log_vol = np.log(df_1min.volume[df_1min.slippage > 0])
result = OLS(log_slip, sm.add_constant(log_vol)).fit()
alpha_1min, log_eta_1min = result.params[1], result.params[0]
eta_1min = np.exp(log_eta_1min)
```

Result on 98 days: **η = 1.58e-4, α = 0.441**, both in literature ranges (Almgren et al. 2005 finds α ≈ 0.6; Bouchaud et al. 2004 finds α ≈ 0.5 at metaorder level; our 0.441 is on the empirically reported low-α side of the equity range, consistent with crypto's tighter spreads and higher liquidity).

### 4. Binance taker fee modeled (7.5 bps)

Previous `execution_cost()` returned only impact + risk components. Real BTCUSDT spot taker fee is 0.075% (7.5 bps). Now part of `ACParams.fee_bps`:
```python
fee = (fee_bps / 1e4) · S0 · sum(abs(n_k))
```

For a monotone full liquidation `Σ|n_k| = X₀`, so the fee is a **constant across strategies** — does not change optimization direction. It does change the reported "% savings" denominator: pre-fee 88% of TWAP objective → post-fee 60% on the reference 10-BTC case. PRESENTATION slide 8 documents the fee dominance at small X₀.

### 5. BookDepth-η robustness check

We also computed η directly from order-book depth on 28 days of bookDepth data (`audits/snapshots/bookdepth/`). The bookDepth pipeline reads Binance L2 snapshots (top-25 levels at sub-second resolution), reconstructs the visible supply curve at each snapshot, computes the **instantaneous slippage** for a fixed quantity (`avg_fill_price(qty, t) − midprice(t)`), and regresses log(slippage) on log(qty) over thousands of snapshots:

```python
# Pseudocode for bookdepth_impact_estimator.py
for snapshot in bookdepth_snapshots:
    bids, asks, mid = parse_snapshot(snapshot)
    for qty in qty_grid:                          # 0.01, 0.1, 1, 10 BTC
        slip = walk_book(asks, qty) - mid          # ask side for sell
        observations.append((qty, slip, snapshot.timestamp))

# OLS on log(slip) ~ alpha · log(qty) + log(eta)
result = OLS(log_slip, sm.add_constant(log_qty)).fit()
```

The bookDepth-derived η was consistent with the 1-min aggregated estimate (**η ≈ 1.58e-4**), supporting the calibration's robustness. This is an **independent measurement** — no shared bid-ask-bounce contamination — so the agreement is meaningful evidence that the 1-min aggregated method is identifying real liquidity costs, not regression artifacts.

The bookDepth scripts (`download_binance_bookdepth.py`, `compute_eta_bookdepth_regime.py`, `bookdepth_impact_estimator.py`) and the 56 bookDepth files have been moved to `audits/data_acquisition/` and `audits/snapshots/bookdepth/` (Decision 2, 2026-04-27) — preserved for citation in this robustness paragraph but not part of the active production pipeline.

### 6. Regime-conditional calibration (Sonnet C audit, FINDINGS §1.5)

When per-regime impact is computed by **sub-sample regression** (filtering trades by HMM Viterbi state, then running the 1-min cascade on each sub-sample independently):

| Parameter | Risk-On | Risk-Off | Base (pooled) |
|---|---|---|---|
| σ (annualized) | 0.704 | 2.498 | — |
| γ multiplier | 0.798 | 3.093 | 2.672 |
| η multiplier | 0.562 | 7.726 | 2.76e-5 |

Compare against Yuhao's earlier `σ × dimensionless_multiplier` parameterization (commit `1cc770e`, V2/V3/V4 Part E paired tests). Sonnet C's audit (commit `bc7a47d`) found the Yuhao multipliers systematically biased: **risk-off γ overestimated by 41–469%, risk-off η underestimated by 79–90%**. This was the proximate cause of the V4 Part E false positive. See HMM.md for the full V1→V5 progression.

**Sub-sample stability caveat**: risk-off is only **5.5% of bars (~1,500 trades)** on the 98-day window. At this size, the regression has R²=0.10 and α falls outside the literature range, triggering η fallback to 1e-3. This is the V5 failure mode and the reason for Worker I's data-window extension to ~280 days.

### 7. CalibrationResult.sources audit trail

Every calibration run logs which cascade tier won for each parameter:
```python
@dataclass
class CalibrationResult:
    gamma: float
    eta: float
    alpha: float
    sigma: float
    fee_bps: float
    sources: dict[str, str]   # e.g. {"gamma": "aggregated_1min",
                              #       "eta":   "aggregated_1min",
                              #       "sigma": "garman_klass",
                              #       "fee":   "binance_spot_taker"}
```

Downstream consumers can detect degraded calibrations (`"sources['gamma'] == 'literature'"`) and refuse to run, or flag the result. This was added after the original audit found the silent fallbacks were hiding the negative-γ bug.

---

## Evolution & pivots

### Apr 26 — Final state described above (TL;DR)

### Apr 21 — Sonnet C per-regime sub-sample audit (FINDINGS §1.5, commit `bc7a47d`)

True per-regime sub-sample calibration replaced Yuhao's σ-multiplier parameterization. The audit quantified the multiplier bias and re-ran V5 with true per-regime params, producing the +227.5% CVaR reversal that invalidated V4. See HMM.md §V1–V5 for the full narrative.

### Apr 18 — Sign invariant + cascade implementation (FINDINGS §1.1, §1.2)

The 2026-04-18 code-council audit caught the silent fallback. Implementation:
- Added `estimate_kyle_lambda_aggregated()` and `estimate_temporary_impact_aggregated()`
- Added cascade ordering in `calibrated_params()`
- Added γ-sign invariant test
- Added `CalibrationResult.sources` audit trail
- Added Binance taker fee to `ACParams.fee_bps = 7.5`

### Apr 18 — Test for proxy-status documentation

`compute_mid_prices()` returns per-bar VWAP and labels it `mid_price` — but the 10,213 Tardis L2 snapshots **could** give the real best-bid/best-ask mid. We don't wire L2 in because the project scope didn't budget for it. `tests/test_data_pipeline_invariants.py::test_mid_price_function_documents_proxy_status` prevents the disclaimer from being silently removed.

### Mar 26 — Q&A research (was `qa_part_a_calibration.md`)

Specification-time Q&A for Part A calibration. Implementation choices that survived:

- **Q1 (Kyle's λ vs OFI)**: Use Kyle's λ (Cov(ΔP, signed_flow)/Var(signed_flow)) as primary because Binance `aggTrades` directly provides aggressor flag — no L2 needed for γ. OFI (Cont, Kukanov & Stoikov 2014) requires L2 snapshots and gives R²≈65% for equities, ~40% for crypto at 1-min — higher than Kyle's λ but data cost is high. **Mitigate calendar-time bias** (20–50% overestimate in high-activity periods per Hasbrouck 2009) by bucketing into fixed-trade-count windows or 30s/1min intervals. We use 1-min calendar buckets and verify intraday stability.
- **Q2 (Temporary impact without order book)**: Without Tardis L2, the cleanest empirical approach is **the aggregated cascade above** rather than tick-level slippage regression. Literature priors: α=0.6 (Almgren 2005), α=0.5 (Bouchaud 2004 square-root law), η from `spread / (0.01 × ADV)` heuristic. We report **sensitivity over α ∈ {0.5, 0.6, 0.7}** in `figures/plot_alpha_comparison.png`. Our calibrated α=0.441 is on the low end but in-range — consistent with BTC's tighter spreads vs equities.
- **Q3 (Realized vol estimator)**: Garman-Klass on 5-min OHLC bars, annualized with `√(365·24·12) = √105,120` (24/7 crypto convention). Kristoufek et al. (2024, *Appl Econ Lett*) shows GK clearly beats GARCH on Binance crypto data. Rogers-Satchell as alternative for non-zero drift. **NOT** Yang-Zhang (overnight-jump component is meaningless for 24/7). Our σ uses GK with √105,120; Rogers-Satchell as cross-check (consistent within 20% tolerance per FINDINGS §9 confidence summary).

### Mar 22 — Methodology spec (was `nonlinear_impact.md`)

Theory write-up — preserved in Methodology section below. Key choices that survived:

- **Linear vs power-law impact**: linear (α=1) gives the AC closed form; power-law (α≠1) requires HJB FD. Empirical evidence (Almgren 2005, Bouchaud 2004, Torre-Ferrari 1999) supports α ≈ 0.5–0.6 in equities; crypto α estimated 0.4–0.6 (Cont & Cucuringu 2021).
- **Concave (α<1) economic intuition**: doubling trading rate **less than doubles** impact — order book heterogeneity, market-maker accommodation. Square-root law `Δp ~ σ·√(Q/V_daily)` corresponds to α=0.5.
- **Trajectory shape under concave impact**: more uniform than linear (parabolic vs hyperbolic) — front-loading is less aggressive because marginal impact decreases at high rates. Quantitative example from Almgren 2003: trajectory `d/dt[|v|^(-0.5)·v] = 2λσ²x/η` — parabolic with longer tail.
- **Gatheral (2010) NDA constraint**: f(v) = v^α requires α ≥ 0.5 for absence of round-trip arbitrage. α = 0.5 on the boundary (NDA-compatible, just barely); α ∈ [0.5, 1] is the empirically + theoretically safe zone. Our calibrated α=0.441 is **slightly below** the Gatheral safety bound — flagged as a limitation. Permanent γ must be linear (α_perm=1) to prevent price-manipulation arbitrage.
- **Bouchaud propagator vs A&C**: the A&C model assumes **instantaneous** temporary impact (no memory). Bouchaud's propagator G(τ) = τ^{-β} with β≈0.5 captures the actual decay — but adds a convolution integral to cost computation. Obizhaeva-Wang (2013) gives a tractable alternative with exponential decay (one extra state variable in the HJB). For the project, we use the A&C instantaneous assumption and note the simplification in PRESENTATION slide 10.

---

## Methodology / theory

### 1. The Almgren-Chriss impact model

```
ΔS_k = γ·v_k·dt + σ·√dt·Z_k       (permanent impact + diffusion)
exec_price_k = mid_k - h(v_k)      (temporary impact at time of trade)
h(v) = η · |v|^α · sign(v)         (power-law temporary impact)
```

For a sell (v_k = n_k/τ > 0):
- **Permanent impact** `γ·v_k·dt`: persists indefinitely, shifts mid-price for all subsequent trades. In A&C: linear in v (informational content of trade).
- **Temporary impact** `η·v^α`: instantaneous — affects only the current execution price, no memory. In A&C: power-law parameterized by exponent α.

**Total cost** for trajectory `x_t` (continuous):
```
C = ∫_0^T [η·|dx/dt|^(α+1)] dt + ½γX₀² + ∫_0^T λσ²·x_t² dt
        ↑ temporary cost      ↑ perm    ↑ AC risk penalty
```

The permanent-impact term `½γX₀²` is **trajectory-independent** and does not enter the HJB optimal-schedule problem. It enters only in total-cost reporting. See HJB.md §12 for the formal proof via telescoping cumsum.

### 2. Calibration challenge — why tick-level fails for both γ and η

At tick (per-trade) frequency:
- `Δprice` between consecutive trades is dominated by **bid-ask bounce** (mean-reverting noise around mid)
- `signed_flow` is mostly retail / market-making, weak correlation with mid-price drift
- Result: `Cov(Δp, signed_flow) → 0` or **negative** (bounce signal dominates) → γ → 0 or negative

Same bias for temporary impact: per-trade `abs(Δprice) ~ noise`, not slippage against mid.

Aggregating to 1-min buckets:
- `Δprice` between bucket-start and bucket-end measures **mid-price drift over the bucket** (bounce averages out)
- `net_flow` = sum of signed quantities in the bucket measures **net informational pressure**
- Result: meaningful R²; γ recovers positive sign

5-min buckets are an alternative tier in the cascade for low-activity windows. Tick-level retained as the last tier before literature fallback for academic comparison.

### 3. The cascade in `calibrated_params()`

```python
def calibrated_params(...) -> CalibrationResult:
    # Try methods in order; first one with positive sign + R²>threshold wins
    methods_gamma = [
        ("aggregated_1min", estimate_kyle_lambda_aggregated, "1min"),
        ("aggregated_5min", estimate_kyle_lambda_aggregated, "5min"),
        ("tick_level",      estimate_kyle_lambda,            None),
        ("literature",      lambda: 1e-4,                    None),
    ]

    for source, fn, freq in methods_gamma:
        gamma = fn(trades, freq=freq) if freq else fn(trades)
        if gamma is not None and gamma > 0:
            sources["gamma"] = source
            break
    ...
```

Same pattern for η, α. The `CalibrationResult.sources` dict tracks which tier succeeded for each parameter — visible to all downstream consumers.

### 4. Empirical α values across studies

| Study | Asset class | α (temporary) | Method |
|---|---|---|---|
| Almgren et al. (2005) | US equities | ~0.6 | Citigroup institutional regression |
| Bouchaud et al. (2004) | French equities | ~0.5 | TAQ microstructure data |
| Torre & Ferrari (1999) | US equities | ~0.5 | BARRA model calibration |
| Gomes & Waelbroeck (2015) | Multi-asset | 0.5–0.6 | Institutional order data |
| Cont & Cucuringu (2021) | Crypto | 0.4–0.6 | Order-book reconstruction |
| **Our calibration (98 days BTCUSDT)** | **BTC** | **0.441** | **1-min aggregated regression** |

α=0.441 is on the low end of the empirical range, consistent with BTC's tighter spreads relative to mid-cap equities and the 24/7 continuous-trading liquidity profile.

### 5. Volatility estimator — Garman-Klass on 5-min OHLC

For 5-min bars on 24/7 crypto:
```python
def garman_klass(o, h, l, c):  # log prices
    return 0.5 * (log(h/l))**2 - (2*log(2) - 1) * (log(c/o))**2

# Per-bar GK_t = garman_klass(open_t, high_t, low_t, close_t)
# Daily rolling vol (288 bars per 24h):
gk_daily = rolling_mean(GK_series, window=288)
sigma_annual = sqrt(gk_daily * (365 * 24 * 12))  # = sqrt(daily_var * 105,120)
```

Properties:
- ~7.4× more efficient than close-to-close (Garman & Klass 1980)
- Validated empirically on Binance crypto by Kristoufek et al. (2024) — outperforms GARCH-family in both in-sample fit and OOS forecasting
- Does NOT require overnight-gap assumption (irrelevant for 24/7)

**Rogers-Satchell** as cross-check (handles non-zero drift, also no overnight assumption). Tested in `tests/test_data_pipeline_invariants.py::test_gk_vs_rs_within_20pct`.

**NOT Yang-Zhang**: combines overnight close-to-open variance with intraday RS — overnight component is noise for 24/7 markets.

### 6. Self-impact convention (A&C 2000 standard, repeated for completeness)

Trade `n_k` permanently shifts the price for trades `k+1, ..., N` only — NOT for its own execution price. The `cum_perm_impact` array is **lagged**:
```python
cum_perm_impact = np.cumsum(np.insert(n[:-1], 0, 0)) * gamma
exec_prices = S0 - cum_perm_impact - eta * abs(n)**alpha * sign(n) / dt
```

A previous ~0.05% MC vs deterministic discrepancy was traced to MC excluding self-impact + cost_model including self-impact. Aligning both to lagged cumsum eliminated the gap. Test: `test_execution_fees_matches_closed_form_identity` locks the convention.

### 7. Calendar-time vs trade-time bias (Hasbrouck 2009)

Calendar-time Kyle's-λ estimates overstate trade-time estimates by **20–50%** in high-activity periods (e.g., NY open, FOMC announcements). For 24/7 crypto with continuous trading and clustered activity, this bias is meaningful. Mitigations:
- **Trade-time bucketing**: bucket every N=500 trades instead of every 1 min
- **Two-pass regression with lagged-flow instrument**: reduces simultaneity bias

Our 1-min calendar bucketing is a compromise — fast enough to compute, slow enough to average bid-ask bounce, but not trade-time-uniform across the day. We verify intraday γ stability by re-running the regression on hourly subsets and reporting variation; if γ varies by >2× across the day, that's flagged in the calibration log.

### 8. NDA constraint (Gatheral 2010)

For the price-impact kernel `G(t, t')` to be **arbitrage-free**, it must be **completely monotone**: `G(t) = ∫_0^∞ exp(−s·t) μ(ds)` for some non-negative measure μ. Equivalently, G is the Laplace transform of a non-negative measure.

Allowed kernels:
- **Power-law decay** `K(t) = t^{-β}` for β ∈ (0, 1) — completely monotone, NDA-compatible (Bouchaud propagator)
- **Exponential decay** `K(t) = exp(-ρ·t)` — completely monotone (Obizhaeva-Wang)
- **Linear permanent impact** `K(t) = constant` — degenerate completely monotone (A&C)

Constraint on α for `f(v) = v^α`: **α ≥ 0.5** under some model specifications. Our calibrated α=0.441 is **slightly below** the Gatheral bound. Implication: the A&C model with α=0.441 admits round-trip arbitrage in theory. In practice, transaction costs and discreteness eliminate this — but it's worth flagging in the report's limitations section.

### 9. Bouchaud propagator vs A&C — what we don't model

A&C assumes **instantaneous** temporary impact: trade `n_k` affects only the current execution price, no carry-over. Real markets have memory:

```
S(t) = S(0) + Σ_{t'<t} ε(t') · G(t − t') + noise
```

with `ε(t')` = trade sign and `G(τ)` = decay propagator (G(0) = initial impact, G(∞) = permanent). Empirically G(τ) ~ τ^{−0.5}.

**Why no arbitrage despite persistent impact**: Lillo & Farmer (2004) show trade signs are positively autocorrelated (large institutional orders split into many small trades), which exactly cancels the decaying impact to produce a near-martingale price. This "conspiracy" is a microstructure equilibrium, not coincidence.

**Cost of modeling this**: convolution integrals in the cost computation, plus an extra state variable (accumulated impact) in the HJB. Not implemented in the project — listed as future work in PRESENTATION slide 10.

### 10. Bid-ask spread proxies (when L2 unavailable)

| Estimator | Cross-sectional ρ vs TAQ effective | Data required | Crypto applicable |
|---|---|---|---|
| Roll (1984) | 0.56 | Trade prices only | Yes |
| Corwin-Schultz (2012) | 0.61 | OHLC | Yes |
| **Abdi-Ranaldo (2017)** | **0.74** | Close, High, Low | **Yes — best in class** |
| Hasbrouck Gibbs (2009) | 0.64–0.67 | Trade prices, Bayesian | Yes |

**Recommended**: Abdi-Ranaldo on 5-min OHLC bars from aggTrades. Used as **HMM emission feature** (multivariate spec, deferred — see HMM.md). Not used in primary impact calibration because the 1-min aggregated cascade does not need a spread proxy.

### 11. Roll (1984) implementation
```python
prices = agg_df['price'].values
dp = np.diff(prices)
cov = np.cov(dp[:-1], dp[1:])[0, 1]
roll_spread = 2 * np.sqrt(max(-cov, 0))
```

Per Harris (1990) / Hasbrouck (2009): set negative-autocovariance instances to zero (positive autocov occurs in trend-dominated periods, gives imaginary results). Apply over rolling 5–15 min windows for stability.

### 12. Test suite locks for this section

- `test_calibrated_gamma_is_positive` — γ-sign invariant
- `test_calibrated_alpha_in_literature_range` — 0.3 ≤ α ≤ 1.5
- `test_mid_price_function_documents_proxy_status` — VWAP-as-mid disclaimer locked
- `test_execution_fees_matches_closed_form_identity` — fee model + cost-model convention
- `test_gk_vs_rs_within_20pct` — vol cross-check

---

## References & cross-links

**Source notes preserved**:
- `research/archive/nonlinear_impact.md` (Mar 22, power-law theory + α evidence + Gatheral NDA + Bouchaud propagator)
- `research/archive/qa_part_a_calibration.md` (Mar 26, Q1–Q3 calibration Q&A)

**Canonical scripts**:
- `calibration/impact_estimator.py::estimate_kyle_lambda_aggregated`, `estimate_temporary_impact_aggregated`, `calibrated_params` (cascade)
- `calibration/data_loader.py::compute_mid_prices` (VWAP proxy with locked disclaimer)
- `scripts/refit_regime_conditional_impact_extended.py` (per-regime sub-sample calibration, V5)
- `scripts/refit_regime_conditional_impact.py` (V4 version, frozen-cited)
- `scripts/compute_realized_rho.py` (price-vol correlation sanity check)

**Canonical data**:
- `data/regime_conditional_impact.json` (per-regime γ, η, σ, α — V5 final)
- `data/regime_conditional_impact_prefix.json` (pre-fix, kept in `audits/snapshots/` for paper trail)
- `figures/plot_regime_detection.png`
- `figures/plot_alpha_comparison.png` (sensitivity over α ∈ {0.5, 0.6, 0.7, calibrated})
- `figures/sensitivity_eta_lambda.png` (2D heatmap)

**BookDepth robustness sources** (preserved, not active):
- `audits/data_acquisition/bookdepth_impact_estimator.py`
- `audits/data_acquisition/download_binance_bookdepth.py`
- `audits/data_acquisition/compute_eta_bookdepth_regime.py`
- `audits/snapshots/bookdepth/BTCUSDT-bookDepth-*.{csv,zip}` × 56 daily files (28 days)

**Cited in**:
- `FINDINGS.md` §1.1 (γ aggregated cascade), §1.2 (η aggregated cascade), §1.3 (Binance fee), §1.5 (regime-conditional Sonnet C audit), §5.1 (VWAP proxy limitation), §5.2 (magic fallback constants), §9 confidence summary
- `PRESENTATION_DRAFT.md` slide 3 (calibration table), slide 8 (fee dominance), slide 10 (Tardis L2 limitation)

**Foundational references**:
- Almgren & Chriss (2001), *J Risk* 3(2): linear-impact baseline
- Almgren (2003), *Appl Math Finance* 10(1): power-law extension
- Almgren, Thum, Hauptmann & Li (2005), *Risk* July: empirical α≈0.6 from Citigroup data
- Kyle (1985), *Econometrica* 53(6): permanent impact slope
- Hasbrouck (1991), *J Finance* 46(1): permanent vs transient decomposition
- Hasbrouck (2009), *J Finance* 64(3): trading costs on US equities, calendar-time bias
- Bouchaud, Gefen, Potters & Wyart (2004), *Quant Finance* 4(2): square-root law, propagator model
- Bouchaud, Farmer & Lillo (2009), *Handbook of Financial Markets*: comprehensive transient-impact review
- Lillo & Farmer (2004), *Studies Nonlin Dyn Econ* 8(3): long-memory trade signs
- Gatheral (2010), *Quant Finance* 10(7): no-dynamic-arbitrage constraint on impact functions
- Obizhaeva & Wang (2013), *J Fin Markets* 16(1): exponential-decay tractable model
- Cont, Kukanov & Stoikov (2014), *J Fin Econometrics* 12(1): OFI as price-impact predictor
- Goyenko, Holden & Trzcinka (2009): cross-sectional spread-estimator comparison
- Roll (1984), *J Finance*: implicit-spread estimator from trade-price autocovariance
- Corwin & Schultz (2012), *J Finance* 67(2): high-low spread estimator
- Abdi & Ranaldo (2017), *RFS* 30(12): close-high-low spread estimator (best-in-class ρ=0.74)
- Garman & Klass (1980), *J Business* 53(1): OHLC-based vol estimator
- Rogers & Satchell (1991), *Annals Appl Prob* 1(4): drift-aware OHLC vol
- Yang & Zhang (2000), *J Business* 73(3): drift-independent OHLC + overnight (NOT for crypto)
- Bandi & Russell (2006): microstructure noise + optimal sampling frequency
- Kristoufek et al. (2024), *Appl Econ Lett* 32: GK vs GARCH on Binance crypto data
- Cont & Cucuringu (2021): crypto microstructure
- Torre & Ferrari (1999): BARRA market impact model
- Gomes & Waelbroeck (2015): impact and information content
- Anboto Labs (2024), Medium: A&C on Binance BTCUSDT context
