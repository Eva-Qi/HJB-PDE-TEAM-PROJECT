# Audit Snapshots

Files in this directory are FROZEN audit-trail data: pre-fix snapshots
preserved for reproducibility / methodology section citations. They are
NOT consumed by the active pipeline.

## Snapshots

| File | Purpose | Linked Fix |
|---|---|---|
| heston_qmeasure_time_series_PREFIX.json | Pre-3-bug-fix Heston Q-measure ρ time series | research/hmm_state_count_audit_apr26.md (Heston bug fix narrative) |
| hmm_2state_vs_3state_PREFIX.json | Pre-BIC-bug-fix HMM 2-vs-3 state with score(X)*T inflation | research/hmm_state_count_audit_apr26.md |
| paired_*_preTfix.json | Pre-T-unit-fix paired test results (T=1/24=15d bug) | (mark as superseded; cite for "before/after" if narrative needs it) |
| paired_regime_v5_true_params_preOptionC.json | Pre-Option-C v5 params | (Option C = OLS-based regime impact wiring) |
| walk_forward_results_N50.json | Walk-forward at N=50 (pre-N=250 convergence) | (cite for N convergence narrative if needed) |
| walk_forward_results_N250_Tfix.json | Intermediate N=250 + T-fix snapshot | superseded by current walk_forward_results.json |
