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
