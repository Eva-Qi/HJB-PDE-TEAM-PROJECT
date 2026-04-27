# FIGURES — Inventory + Reference Graph

Each figure lists its source script, the JSON dependency it draws from, and which MD section cites it. Figures in `figures/` are committed and canonical. Figures in `audits/figures_v1/` are superseded but preserved for paper trail. The 14 root-level `plot_*.png` are gitignored Apr-10 wave-3 artifacts not part of the committed repo.

---

## §1 Canonical figures (`figures/`)

| File | Source script | JSON / data dep | Cited in |
|---|---|---|---|
| `iv_fit_heatmap.png` | `scripts/plot_iv_surface_fit.py` | `data/deribit_btc_option_chain_20260420.json` | PRESENTATION_DRAFT slide 6, REPORT §5, HESTON.md §2 |
| `heston_qmeasure_time_series.png` (Apr 25, 237 KB, canonical) | `scripts/qmeasure_heston_time_series.py` | `data/heston_qmeasure_time_series.json` (12 snapshots) | FINDINGS §1.4, REPORT §5, HESTON.md §References |
| `heston_variants_iv_fit.png` | `scripts/plot_iv_surface_fit.py` (variant comparison mode) | `data/deribit_btc_option_chain_20260420.json` | HESTON.md (variant comparison narrative), REPORT §5 |
| `sensitivity_alpha.png` | `scripts/sensitivity_sweep.py` | live re-run of `execution_cost` over α-grid | FINDINGS §8 (multi-horizon and α scaling), REPORT §3, HJB.md §References |
| `sensitivity_lambda.png` | `scripts/sensitivity_sweep.py` | live re-run over λ-grid | REPORT §3, HJB.md §3 (λ-centralization rationale) |
| `sensitivity_T.png` | `scripts/sensitivity_sweep.py` | live re-run over T-grid | REPORT §3, HJB.md §References |
| `sensitivity_x0.png` | `scripts/sensitivity_sweep.py` | live re-run over X₀-grid | REPORT §3, FINDINGS §8 |
| `tail_qq_heston_vs_const.png` | `scripts/plot_tail_qq.py` | `data/paired_heston_qmeasure_results.json` | FINDINGS §1.4, §6, REPORT §5, HESTON.md §5 |
| `heston_cross_source_comparison.png` | `scripts/compare_heston_ibit_vs_deribit.py` | `data/heston_cross_source_comparison.json` | RESULTS §10, REPORT §5, HESTON.md §Apr 20 |
| `heston_ibit_smile_20260424.png` | `scripts/heston_calibrate_ibit.py` | `data/heston_qmeasure_ibit_20260424.json` | HESTON.md §Apr 20, REPORT §5 |

10 figures committed. Filename convention: snake_case with no version suffix when canonical.

---

## §2 Archived figures (`audits/figures_v1/`)

| File | Reason archived | Replaced by |
|---|---|---|
| `heston_qmeasure_timeseries.png` (Apr 20, 117 KB — different filename: `_timeseries` no underscore between time and series) | Superseded by Apr 25 file with the underscore | `figures/heston_qmeasure_time_series.png` |
| `dvol_vs_rho_scatter.png` | DVOL HMM experiment frozen (FINDINGS §5.4 rejected); no MD citation in the active narrative | (no replacement) |

Both moved during the 2026-04-27 P1-2 cleanup pass (Worker C2). See `audits/README.md` §"Figures v1" for the full table.

---

## §3 Local-only / gitignored (root)

14 PNGs at the repo root are gitignored via `.gitignore:16` (`plot_*.png`). They are Apr-10 Wave-3 artifacts from `scripts/generate_all_plots.py` (now in `audits/dead/`) and from earlier diagnostic scripts. Listed for completeness:

```
plot_alpha_comparison.png
plot_heston_vs_constant.png
plot_multi_x0_costs.png
plot_multi_x0_savings.png
plot_multi_x0_trajectories.png
plot_pde_mc_crossval.png
plot_regime_detection.png
plot_scheme_convergence.png
plot_strategy_comparison.png
plot_walk_forward_slsqp.png
plot_walk_forward.png
plot_x0_sensitivity.png
calibrated_pipeline_results.png   (gitignored as exact filename)
full_comparison.png               (untracked, generator in audits/dead/)
```

The HJB.md and FINDINGS.md references to e.g. `figures/plot_walk_forward.png` and `figures/plot_pde_mc_crossval.png` are **legacy citations** from before the canonical `figures/` directory was set up. The current canonical figures for those topics live in `figures/` proper or are in committed-but-not-yet-regenerated state. Re-running `scripts/sensitivity_sweep.py` and `scripts/walk_forward_validation.py` will produce updated PNGs in `figures/` when the final paper assembly happens.

---

## §4 Reference graph

Where each canonical figure is consumed:

| Figure | REPORT.md | FINDINGS.md | PRESENTATION_DRAFT.md | research/*.md |
|---|---|---|---|---|
| `iv_fit_heatmap.png` | §5 | — | slide 6 | HESTON.md §2 |
| `heston_qmeasure_time_series.png` | §5 | §1.4 | — | HESTON.md (Refs) |
| `heston_variants_iv_fit.png` | §5 | — | — | HESTON.md (variants) |
| `sensitivity_alpha.png` | §3 | §8 | — | HJB.md (Refs), HESTON.md §gap_heston_implementation_sensitivity |
| `sensitivity_lambda.png` | §3 | — | — | HJB.md §3 |
| `sensitivity_T.png` | §3 | — | — | HJB.md (Refs) |
| `sensitivity_x0.png` | §3 | §8 | — | (none) |
| `tail_qq_heston_vs_const.png` | §5 | §1.4, §6 | — | HESTON.md §5 |
| `heston_cross_source_comparison.png` | §5 | — | — | HESTON.md §Apr 20 |
| `heston_ibit_smile_20260424.png` | §5 | — | — | HESTON.md §Apr 20 |

**Orphan figures** (in `figures/` but with no MD citation): **0** after this audit. All 10 canonical figures are referenced.

---

## §5 Figures that exist *as references in MD* but live as gitignored root PNGs

These are scripts whose figures were generated earlier but the canonical filenames in MDs point to `figures/<name>.png` which does not exist yet at that path. To resolve before submission: re-run the script and save into `figures/`.

| MD reference | Current location | Action |
|---|---|---|
| `figures/scheme_convergence.png` (HJB.md §5) | root `plot_scheme_convergence.png` (gitignored) | re-run `scripts/sensitivity_sweep.py::convergence_study` and save into `figures/` before final paper build |
| `figures/plot_pde_mc_crossval.png` (HJB.md §7) | root `plot_pde_mc_crossval.png` (gitignored) | re-run cross-val script and save into `figures/` |
| `figures/plot_walk_forward.png`, `figures/plot_walk_forward_slsqp.png` (HJB.md Refs) | root `plot_walk_forward.png`, `plot_walk_forward_slsqp.png` (gitignored) | re-run `scripts/walk_forward_validation.py` plotting block |
| `figures/plot_strategy_comparison.png` (HJB.md Refs) | root `plot_strategy_comparison.png` (gitignored) | re-run plot script (currently in `audits/dead/full_comparison.py` — un-archive or rewrite) |
| `figures/plot_x0_sensitivity.png` (HJB.md Refs) | root `plot_x0_sensitivity.png` (gitignored) | re-run `scripts/x0_sensitivity_analysis.py` plotting block |
| `figures/plot_alpha_comparison.png` (HJB.md, IMPACT.md Refs) | root `plot_alpha_comparison.png` (gitignored) | re-run script and save into `figures/` |
| `figures/plot_regime_detection.png` (IMPACT.md Refs) | root `plot_regime_detection.png` (gitignored) | re-run regime detection plot block |
| `figures/sensitivity_eta_lambda.png` (IMPACT.md, HESTON.md, HJB.md) | not yet generated | run the 2D η×λ heatmap block in `scripts/sensitivity_sweep.py` |

**Submission checklist**: before final `md2pdf` build of REPORT.md, regenerate the 8 missing-from-`figures/` PNGs and commit. None of them are blocking the cleanup; they are paper-build chores.

---

*Last updated 2026-04-27 by Worker C4 during P1-2 root-MD coalescence.*
