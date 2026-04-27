# Audits

This directory archives scripts and data artifacts that are not part of the
active execution pipeline but are preserved for git history, reproducibility,
or audit trail.

<!-- Worker A: append data-snapshot table section here -->

## Superseded scripts

| File | Replaced by | Why moved |
|---|---|---|
| paired_test_regime_aware_v2.py | paired_test_regime_aware_v5.py | V5 fixes V2's regime-multiplier issue (Option C, 2026-04-24) |
| paired_test_regime_multihorizon.py | paired_test_regime_aware_v5.py | multi-horizon ablation absorbed into V5 |
| aggregated_alpha_estimation.py | alpha_estimation_full.py | full version covers same test space |
| aggregated_alpha_v2.py | alpha_estimation_full.py | "_v2" iteration superseded |

## Explorations (not in final report)

| File | Purpose | Why not used |
|---|---|---|
| qmeasure_heston_eth_time_series.py | ETH parallel of BTC pipeline | Project narrative focused on BTC; ETH is cross-check, not main story |
| compare_heston_eth_vs_btc.py | ETH vs BTC parameter comparison | Same reason |
| compare_heston_variants.py | Various Heston spec variants | Earlier exploration before settling on Q-measure spec |
| compare_heston_pmeasure_vs_qmeasure.py | P vs Q measure comparison | superseded by paired_test_heston_qmeasure.py (cited) |

## Demos

| File | Purpose |
|---|---|
| demo_multifeature_hmm.py | Multifeature HMM demo (multifeature ablated out per audit doc) |
| arcadia_regime_demo.py | 29-LOC standalone demo of regime API |

## Frozen (cited, not actively run)

| File | Why frozen |
|---|---|
| paired_test_regime_aware.py | V1 regime-aware, cited in research/regime_hmm.md as method origin |
| paired_test_x0_sensitivity.py | x0 sensitivity sweep, cited in research/regime_hmm.md |
