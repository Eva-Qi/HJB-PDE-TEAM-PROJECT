# Audits

This directory archives scripts and data artifacts that are NOT part of the
active execution pipeline but are preserved for git history, reproducibility,
and audit trail.

Files here are excluded from GitHub language stats via `.gitattributes`
(`audits/** linguist-generated=true`) so the repo language breakdown reflects
only the active codebase.

---

## Snapshots (`snapshots/`)

Pre-fix data snapshots preserved for methodology / before-after citation.

| File | Purpose | Linked Fix |
|---|---|---|
| heston_qmeasure_time_series_PREFIX.json | Pre-3-bug-fix Heston Q-measure ρ time series | research/hmm_state_count_audit_apr26.md (Heston bug fix narrative) |
| hmm_2state_vs_3state_PREFIX.json | Pre-BIC-bug-fix HMM 2-vs-3 state with score(X)*T inflation | research/hmm_state_count_audit_apr26.md |
| paired_*_preTfix.json | Pre-T-unit-fix paired test results (T=1/24=15d bug) | superseded; cite for "before/after" if narrative needs it |
| paired_regime_v5_true_params_preOptionC.json | Pre-Option-C V5 params | Option C = OLS-based regime impact wiring (2026-04-24) |
| walk_forward_results_N50.json | Walk-forward at N=50 | pre-N=250 convergence; cite for N convergence narrative |
| walk_forward_results_N250_Tfix.json | Intermediate N=250 + T-fix snapshot | superseded by current walk_forward_results.json |

---

## Superseded scripts (`superseded/`)

Earlier iterations replaced by canonical versions.

| File | Replaced by | Why moved |
|---|---|---|
| paired_test_regime_aware_v2.py | paired_test_regime_aware_v5.py | V5 fixes V2's regime-multiplier issue (Option C, 2026-04-24) |
| paired_test_regime_multihorizon.py | paired_test_regime_aware_v5.py | multi-horizon ablation absorbed into V5 |
| aggregated_alpha_estimation.py | alpha_estimation_full.py | full version covers same test space |
| aggregated_alpha_v2.py | alpha_estimation_full.py | "_v2" iteration superseded |
| refit_regime_conditional_impact.py | refit_regime_conditional_impact_extended.py | extended version is canonical (Apr 27 P1-2 coalescence) |

---

## Explorations (`explorations/`)

Alternative paths investigated; not in the final report.

| File | Purpose | Why not used |
|---|---|---|
| qmeasure_heston_eth_time_series.py | ETH parallel of BTC pipeline | Project narrative focused on BTC; ETH is cross-check, not main story |
| compare_heston_eth_vs_btc.py | ETH vs BTC parameter comparison | Same reason |
| compare_heston_variants.py | Various Heston spec variants | Earlier exploration before settling on Q-measure spec |
| compare_heston_pmeasure_vs_qmeasure.py | P vs Q measure comparison | superseded by paired_test_heston_qmeasure.py (cited) |

---

## Demos (`demos/`)

Uncited demo scripts.

| File | Purpose |
|---|---|
| demo_multifeature_hmm.py | Multifeature HMM demo (multifeature ablated out per audit doc) |
| arcadia_regime_demo.py | 29-LOC standalone demo of regime API |

---

## Frozen (`frozen/`)

Cited but not actively run. Preserved for paper reproducibility.

| File | Why frozen |
|---|---|
| paired_test_regime_aware.py | V1 regime-aware, cited in research/regime_hmm.md as method origin |
| paired_test_x0_sensitivity.py | x0 sensitivity sweep, cited in research/regime_hmm.md |

---

## Dead (`dead/`)

Zero importers + zero cited output. Retired during Apr 27 P1-2 coalescence.

| File | Origin dir | Reason |
|---|---|---|
| bivariate_hmm_coinmetrics.py | scripts/ | TIER45 NEGLECT, multi-feature HMM rejected |
| hmm_coinmetrics_extended.py | scripts/ | superseded by compare_hmm_vol_feature |
| hmm_macro_vix.py | scripts/ | FINDINGS §5.4 macro HMM rejected |
| compare_hmm_features.py | scripts/ | superseded by compare_hmm_vol_feature.py (post Apr 26 BIC fix) |
| compute_eta_bookdepth_regime.py | scripts/ | bookdepth pipeline opt-in, not wired |
| orderbook_alpha_estimation.py | scripts/ | superseded by alpha_estimation_full.py |
| kyle_gamma_monthly_time_series.py | scripts/ | not cited in canonical narrative |
| run_calibrated_pipeline.py | scripts/ | early prototype, superseded by walk_forward_validation |
| generate_all_plots.py | scripts/ | omnibus plotting, no MD citations |
| full_comparison.py | scripts/ | early plot script |
| scheme_comparison.py | scripts/ | early plot script |
| convergence_study.py | scripts/ | early convergence plot, superseded by tests/ |
| diagnose_pde_mc_discrepancy.py | scripts/ | one-shot diagnostic |
| demo_type2_error.py | scripts/ | type-II demo, no MD citation |
| run_multi_x0_comparison.py | scripts/ | superseded by paired MC test (FINDINGS §2.1) |
| paired_test_ac_vs_twap_hires.py | scripts/ | superseded by paired_test_regime_aware_v5 + paired_test_heston_qmeasure |
| paired_test_heston_vs_const.py | scripts/ | superseded by paired_test_heston_qmeasure |
| paired_test_heston_multihorizon.py | scripts/ | docstring-only "import", not real importer |
| diag/diag_mergeasof.py | scripts/diag/ | one-shot diagnostic |
| diag/diagnose_n_sensitivity.py | scripts/diag/ | one-shot diagnostic |

---

## Data Acquisition (`data_acquisition/`)

Download scripts archived during Apr 27 cleanup. Data files they produced are either KEPT in `data/` (used by ACTIVE pipeline) or moved to `audits/snapshots/`. Scripts kept here for "how we obtained the data" provenance.

| Script | Output data location |
|---|---|
| download_kraken.py | `audits/snapshots/kraken_btc_daily.json` |
| download_coinmetrics.py (v0) | `audits/snapshots/coinmetrics_btc_onchain.json` |
| download_coinmetrics_extended.py (v1) | `audits/snapshots/coinmetrics_btc_extended_metrics.json` |
| download_coinmetrics_extended_v2.py | `data/coinmetrics_btc_extended_metrics_v2.json` (KEPT — used by HMM) |
| download_tardis_eth_options.py | `audits/snapshots/tardis_deribit_options_ETH_*.json` × 12 |
| download_tardis_options.py | `data/tardis_deribit_options_*.json` × 12 BTC (KEPT — cross-check) |
| download_binance_bookdepth.py | `audits/snapshots/bookdepth/BTCUSDT-bookDepth-*.{csv,zip}` × 56 |
| download_binance_aggTrades_backfill.py | `data/BTCUSDT-aggTrades-*.csv` (KEPT — gitignored, used by walk_forward) |
| download_binance_futures_data.py | `data/binance_btc_klines_1d.json` (KEPT — used by compute_realized_rho) |
| download_coinbase.py | `data/coinbase_btc_5min.json` (KEPT — used by compute_realized_rho) |
| download_deribit_dvol.py | `audits/snapshots/deribit_dvol_btc_daily.json` |
| download_deribit_funding.py | `data/deribit_btc_funding_hourly.json` (KEPT — used by analysis_funding_vs_realized_rho) |
| download_deribit_extended.py | `audits/snapshots/deribit_btc_historical_vol.json` |
| download_fred_macro.py | `audits/snapshots/fred_macro_daily.json` |
| extract_deribit_oi.py | `audits/snapshots/deribit_btc_oi_summary.json` |
| cross_exchange_analysis.py | (no output) |
| bookdepth_impact_estimator.py | (library — used by retired compute_eta_bookdepth_regime.py) |

KEPT in `calibration/`: `data_loader.py`, `impact_estimator.py`, `download_deribit.py` (runtime dep), `download_binance.py` (CLI hint).
