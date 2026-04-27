# MF796 Term Project — Optimal Execution Under Stochastic Volatility on Binance BTCUSDT

**Course**: MF796 Computational Methods of Mathematical Finance · BU
**Team**: Eva-Qi, bp (no-322), Yuhao
**Submission target**: ~2 weeks (May 2026)
**Repo**: `github.com/Eva-Qi/HJB-PDE-TEAM-PROJECT`
**Status**: Pre-submission (2026-04-27)

---

## TL;DR (current state — 2026-04-27)

We implemented the Almgren-Chriss optimal-execution framework on real Binance BTCUSDT data (98 days of aggTrades, 130M trades; 12 monthly Tardis Deribit option chains; macro overlays from FRED, CoinMetrics, F&G), and pushed it through three extensions: (B) a fully implicit HJB PDE solver with Howard's policy iteration for nonlinear impact, (D) a Heston Q-measure stochastic-vol extension calibrated by Carr-Madan FFT against Deribit BTC option IV surfaces, and (E) a 2-state HMM regime-aware execution scheduler. The 2026-04-18 code-council audit caught and fixed a **silent negative-γ failure** in the original calibration (tick-level Kyle's-λ returned γ = −0.0113 because of bid-ask-bounce mean-reversion); the resulting 1-min → 5-min → tick → literature cascade (`calibration/impact_estimator.py`) now produces γ = +1.48, η = 1.58e-4, α = 0.441, all in literature ranges. The Heston Q-measure resolves the original 24-bar overlapping-window κ/ρ unreliability — 2-day stability std(κ) = 0.01, std(ρ) = 0.001, IV-surface RMSE = 0.0086, and Heston beats the 3/2 model by 33.9% RMSE on the same chain.

**Headline result**: **Heston-Q stochastic vol reduces CVaR_95 by 4.79% vs constant-vol** (paired test, 100k common-random-numbers paths, p < 0.0001). This is Part D's clean, defensible anchor finding (`FINDINGS.md` §6, `HESTON.md` §5).

**Walk-forward OOS validation** of the HJB solver across **6/6 splits on 98 days** of Binance aggTrades shows AC beats TWAP by **+15% to +37% MC savings** at institutional X_0 = 1000 BTC (`data/walk_forward_results.json`, `RESULTS.md` §3, `TEAM_BRIEFING_APR22.md` §2). The retail boundary is **X_0 ≥ 100 BTC**, updated from V5's "X_0 ≥ 1000" claim after a 100k-path rerun caught a type-II error at 10k.

**Part E regime-aware** is the project's methodological-learning story rather than a clean positive result. Five iterations (V1 → V5) each surfaced a successively subtler bias layer: magic σ × 1e-8 multipliers (V1) → wrong sample size (V2) → wrong metric (V3) → biased Yuhao multipliers (V4 found CVaR_95 −14%, p < 0.0001 — invalidated) → η literature-fallback noise in 5.5%-of-bars risk-off sub-sample (V5 reversed sign to +227.5%). V6 with extended 280-day data is in flight (Worker I).

**Confidence summary** (from `FINDINGS.md` §9):

| Claim | Confidence | Evidence |
|---|---|---|
| γ, η, α calibration reliable | high (γ medium-noisy, R²=0.18) | 1-min cascade, 141K buckets; bookDepth cross-check consistent at η ≈ 1.58e-4 |
| Heston Q-measure κ/ρ reliable | **high** | Deribit FFT, std(κ) = 0.01, std(ρ) = 0.001, round-trip <4% |
| Heston beats 3/2 model | high | RMSE 0.098 vs 0.132 (−33.9%) on same chain |
| **Heston-Q reduces CVaR_95 vs const-vol** | **high** | **4.79% reduction, p<0.0001, 100k paths** |
| AC beats TWAP at institutional X_0 ≥ 100 BTC | high | walk-forward 6/6 splits, p < 0.0001 at X_0 = 1000 |
| Regime-aware tail-risk benefit | **unresolved** | V4 invalidated by V5 sign reversal; V6 pending |
| Multi-feature HMM (daily-frequency exogenous) helps | **none** | F&G, CoinMetrics FlowIn dilute σ-spread 250% → ~46% on all variants |

For canonical numbers, see `RESULTS.md`. For figures, see `FIGURES.md`. For per-topic deep dives, see `research/{HMM, HESTON, HJB, IMPACT}.md`.

---

## How to read this report

- **§1** — Problem setup: AC framework, dataset, what we extended.
- **§2** — Calibration (Part A): the cascade fix, current γ/η/α/σ/fee numbers, bookDepth-η robustness check.
- **§3** — HJB PDE solver (Part B): two-path solver (Riccati for α=1, Howard's policy iteration for α≠1), walk-forward 6-split OOS validation, λ and T centralization.
- **§4** — Monte Carlo (Part C): 100k-path paired tests, retail boundary, multi-horizon scaling, variance-reduction infrastructure.
- **§5** — Heston extension (Part D, anchor finding): Q-measure via Deribit, model selection, Heston-Q vs const-vol paired test.
- **§6** — Regime-aware execution (Part E, methodological learning): V1 → V5 narrative, BIC unit-error fix, V5 reversal, V6 pending.
- **§7** — Limitations + methodological learnings.
- **§8** — References + audit chain.

For canonical numbers, defer to `RESULTS.md`; for figure inventory, defer to `FIGURES.md`. This report is the narrative layer; the numbers and figures live in their canonical files to keep RESULTS.md drift-resistant.

---

## §1 Problem statement and dataset

### §1.1 The Almgren-Chriss optimal-execution problem

Given a starting inventory `X_0` of BTC to liquidate over horizon `[0, T]`, we minimize the AC mean-variance objective:

```
min_{v(t)}  E[ C(v) ]  +  λ · Var[ C(v) ]
```

where `C(v)` is the implementation shortfall under trading-rate `v(t)`, and `λ ≥ 0` is risk aversion. Implementation shortfall:

```
C(v) = ∫_0^T [ η · |v(t)|^(α+1) ] dt  +  ½γX_0²  +  ∫_0^T λσ²(or v_t)·x(t)² dt
        ↑ temporary impact            ↑ perm     ↑ market-risk term
```

with parameters:
- `γ` — permanent impact (Kyle's λ): linear in trade size, persists indefinitely.
- `η, α` — temporary impact: power-law parameterized by `α` (= 1 linear, < 1 concave; calibrated 0.441).
- `σ` — return volatility (extended to Heston `√v_t` in Part D).
- `λ` — risk aversion, centralized at 1e-6 across the codebase.
- `T` — execution horizon, centralized at 1/(365.25·24) yr ≈ 1.142e-4 yr (1 hour, 24/7 crypto convention).

Reported variants: TWAP, VWAP, and AC-optimal (closed-form sinh trajectory at α=1; HJB PDE solution at α≠1).

### §1.2 Dataset

| Source | Files | Purpose | Wired into model? |
|---|---|---|---|
| Binance aggTrades (98 days, 2026-01-01 → 2026-04-08) | `data/BTCUSDT-aggTrades-*.csv` (130M trades) | Kyle γ, η, regime σ — core | Yes (core) |
| Binance 1d klines (24 mo) | `data/binance_btc_klines_1d.json` | Heston σ realized — core | Yes (core) |
| Coinbase 5-min (85k candles) | `data/coinbase_btc_5min.json` | Cross-venue ρ check | Cross-check only |
| Tardis Deribit BTC options (12 mo) | `data/tardis_deribit_options_*.json` × 12 | Q-measure Heston longitudinal | Yes (core) |
| Deribit live option chain (Apr 20–21) | `data/deribit_btc_option_chain_2026042{0,1}.json` | Q-measure Heston canonical | Yes (core) |
| CoinMetrics extended (98 days) | `data/coinmetrics_btc_extended_metrics_v2.json` | HMM bivariate ablation | Tested → rejected |
| F&G Index (98 days) | `data/fear_greed_btc.json` | HMM bivariate ablation | Tested → rejected |
| Binance bookDepth (28 days) | `audits/snapshots/bookdepth/` (archived) | η-cross-check | Robustness only |
| Coinbase 5-min funding-vs-realized-ρ | `data/realized_rho_daily.json`, `data/funding_vs_realized_rho_results.json` | Heston ρ structural sanity check | Auxiliary |

Out-of-scope: ETH options (12 Tardis files archived to `audits/snapshots/`), Kraken daily, Coinbase cross-venue beyond ρ check.

See `RESULTS.md` §12 for the full file → script → MD-citation traceability.

### §1.3 Project structure (post-2026-04-27 cleanup)

```
mf796_project/
├── REPORT.md            ← this document (master narrative)
├── RESULTS.md           ← canonical numbers + JSON traceability
├── FIGURES.md           ← figure inventory + reference graph
├── README.md            ← quick start + entry points
├── FINDINGS.md          ← V6 living truth source (numbers)
├── PRESENTATION_DRAFT.md
├── shared/              ← params, cost_model (active library)
├── calibration/         ← data_loader + impact_estimator + 2 download CLIs
├── pde/                 ← hjb_solver (Riccati + Howard's)
├── montecarlo/          ← sde_engine (GBM + Heston) + strategies
├── extensions/          ← heston, regime
├── scripts/             ← active analysis scripts (~13)
├── tests/               ← 173+ tests
├── data/                ← canonical JSONs + cached returns
├── figures/             ← 10 canonical PNGs (see FIGURES.md)
├── research/            ← 4 active topic MDs + archive/
└── audits/              ← dead/ + superseded/ + snapshots/ + audit_chain/ + figures_v1/ + data_acquisition/
```

---

## §2 Calibration (Part A)

The original tick-level Kyle's-λ regression returned γ = −0.0113 (negative permanent impact, economically nonsensical) because per-trade `abs_price_change` is dominated by bid-ask bounce. The fix is a **1-min → 5-min → tick-level → literature** cascade in `calibration/impact_estimator.py`, where each tier is tried until R² > 0.05 and the parameter is in literature range. Production result: **γ = +1.48 (R² = 0.18, 1-min, 141K buckets)**, **η = 1.58e-4**, **α = 0.441**. See `FINDINGS.md` §1.1-§1.4 and `research/IMPACT.md` for full narrative including the per-regime OLS calibration that invalidated Yuhao's σ-multiplier approach.

### §2.1 BookDepth-η robustness check

We also computed η directly from order-book depth on **28 days of Binance bookDepth data** (`audits/snapshots/bookdepth/`, 56 daily CSV/ZIP files, ~63 MB local-only); the order-book-derived estimate was consistent with the 1-min aggregated estimate at **η ≈ 1.58e-4**, supporting the calibration's robustness across two independent estimators (one trade-flow-based, one order-book-depth-based). Implementation is in `audits/data_acquisition/bookdepth_impact_estimator.py` (archived — opt-in for downstream consumers, not wired into the canonical walk-forward pipeline).

---

## §3 HJB PDE solver (Part B)

`pde/hjb_solver.py` provides two paths: a closed-form Riccati ODE solver for α=1, and Howard's policy iteration with implicit Crank-Nicolson discretization for α≠1. The α≠1 path was added after we discovered the separable ansatz V(t,x) = a(t)x² fails for nonlinear impact (`research/HJB.md` §4). N=250 timesteps centralized via `shared/experiment_config.py`. Walk-forward 6-split OOS validation (`scripts/walk_forward_validation.py`, `data/walk_forward_results.json`) shows **AC beats TWAP by +15% to +37% MC savings at X₀ = 1000 BTC across all 6 splits** with p < 0.0001. SLSQP cross-check (`data/walk_forward_results_slsqp.json`, code-council Part 11 §11.4 two-solver pattern) agrees to within 1% — confirms the HJB solution is the optimum, not a numerical artifact. See `RESULTS.md` §3 for full table, `TEAM_BRIEFING_APR22.md` §2 (now in `audits/audit_chain/`) for the original walk-forward briefing.

---

## §4 Monte Carlo (Part C)

`montecarlo/sde_engine.py` implements Euler-Maruyama for both GBM and Heston (with full-truncation Andersen QE scheme), Sobol QMC + Brownian Bridge for variance reduction (`research/HJB.md` §6), and common-random-numbers (CRN) coupling for paired tests. **100k paths × 250 steps** is the standard configuration. The retail-vs-institutional boundary is **X₀ ≥ 100 BTC** — below that, AC ≈ TWAP at the noise floor (paired test p > 0.79 at X₀ = 10 BTC, `FINDINGS.md` §2.1). Multi-horizon scaling: CVaR savings scale ~2.4× per 6× horizon (consistent with theory).

---

## §5 Heston extension (Part D) — anchor finding

The original P-measure calibration (`extensions/heston.py::calibrate_heston_from_spot`) produced unreliable κ and ρ because 24-bar overlapping rolling windows induce autocorrelation ≈ 0.985, biasing the moment-matching. **Resolution (commit `fbc5fe9`)**: switch to Q-measure calibration via Carr-Madan FFT against Deribit BTC option IV surfaces (`scripts/qmeasure_heston_time_series.py` for Tardis 12-month time-series, `scripts/deribit_qmeasure_time_series.py` for live-chain). Result: **κ = 9.09 (std 0.01), ρ = −0.385 (std 0.001), IV-surface RMSE = 0.0086, Heston beats 3/2 model by 33.9% RMSE** on the same chain. Tardis (12 monthly snapshots) and Deribit (live snapshots) serve as cross-confirmation per code-council Part 11 §11.4. **Anchor finding**: paired test on 100k common-random-numbers paths (`data/paired_heston_qmeasure_results.json`) shows **Heston-Q reduces CVaR₉₅ by 4.79% vs constant-vol, p < 0.0001** (`FINDINGS.md` §6, `research/HESTON.md` §5).

---

## §6 Regime-aware execution (Part E) — methodological learning

Five iterations each surfaced a successively subtler bias layer:

| Version | Approach | Result | Status |
|---|---|---|---|
| V1 | σ × 1e-8 magic multipliers | not significant | wrong model |
| V2 | sample size fix | not significant | wrong sample |
| V3 | metric fix | not significant | wrong metric |
| V4 | Yuhao's σ-multipliers (`commit f6c3ace`) | **CVaR₉₅ −14%, p < 0.0001** | **invalidated by V5** |
| V5 | true per-regime OLS calibration (`commit 6bab5b1`) | **CVaR₉₅ +227.5%** (sign reversed) | suspect — η fallback in 5.5%-of-bars risk-off sub-sample |
| V6 | extended 280-day data window | pending (Worker I) | will resolve sub-sample size |

The 2026-04-26 BIC-bug fix in `scripts/compare_2state_vs_3state_hmm.py` (the `score(X) * T` inflation factor was reversing the 2-vs-3 state preference) and the vol-feature ablation (`scripts/compare_hmm_vol_feature.py`) established that **2-state HMM on raw returns is preferred** at this data window. See `research/HMM.md` for full V1-V5 narrative + the BIC bug story.

**Net status (current)**: regime-aware tail-risk benefit is **unresolved on the 98-day window**; V6 with extended data is the path to definitive answer.

---

## §7 Limitations + Methodological learnings

From `FINDINGS.md` §5 + audit-chain self-reflection:

- **Mid-price proxy**: `calibration/data_loader.py::compute_mid_prices` uses per-bar VWAP, not L2 best-bid/best-ask mid (10,213 Tardis L2 snapshots available but not wired). Test `test_mid_price_function_documents_proxy_status` prevents the disclaimer from being silently removed.
- **Fallback constants**: γ=1e-4, η=1e-3, κ=5.0, ξ=0.8, ρ=0.0 used when cascade fails; flagged via `CalibrationResult.sources`.
- **Risk-off sub-sample size**: 5.5% of bars (~1500 trades) is too small for stable OLS regression (R² = 0.10, α out of literature range), forcing η fallback in V5. V6 extended window will resolve.
- **HJB PDE bang-bang at sublinear α**: viscosity-solution theory predicts no classical solution for α < 1; numerical scheme can produce all-or-nothing trajectories. Detected by SLSQP cross-check; mitigation: monotone numerical scheme + post-hoc smoothness check (code-council Part 11 §11.4 lesson).
- **Heston ρ identification**: caught and fixed (`research/HESTON.md` Part 13) — original calls-only filter + OI-weighted loss + xi lower bound 0.01 collectively zeroed out ρ identification across 12 monthly snapshots. Loss-function design defines identifiability; data was never the problem.

---

## §8 References + audit chain

- **Current truth**: [FINDINGS.md](FINDINGS.md) (V6, 2026-04-21)
- **Slide narrative**: [PRESENTATION_DRAFT.md](PRESENTATION_DRAFT.md)
- **Per-topic master docs**: [research/HMM.md](research/HMM.md), [research/HESTON.md](research/HESTON.md), [research/HJB.md](research/HJB.md), [research/IMPACT.md](research/IMPACT.md)
- **Numbers**: [RESULTS.md](RESULTS.md)
- **Figures**: [FIGURES.md](FIGURES.md)
- **Audit chain**: `audits/audit_chain/` — 6 archived audit MDs (PROGRESS, AUDIT_VERIFICATION, DATA_INTEGRITY_AUDIT, TIER45_RESEARCH/DECISIONS, TEAM_BRIEFING) + 3 gitignored locals (PROJECT_AUDIT_REPORT, WALKTHROUGH, TECHNICAL_SUMMARY)
- **Audits index**: [audits/README.md](audits/README.md) — dead/, superseded/, snapshots/, data_acquisition/, figures_v1/, audit_chain/
