# MF796 Term Project: Optimal Execution Under Stochastic Volatility on Binance BTCUSDT

**Status (2026-04-27)**: Pre-submission. Active master docs:

- **[REPORT.md](REPORT.md)** — master narrative (TL;DR + per-Part findings + Limitations)
- **[RESULTS.md](RESULTS.md)** — canonical numbers + JSON-file → MD-citation mapping
- **[FIGURES.md](FIGURES.md)** — figure inventory + reference graph
- **[FINDINGS.md](FINDINGS.md)** — V6 living truth source (numbers, audit fixes)
- **[PRESENTATION_DRAFT.md](PRESENTATION_DRAFT.md)** — slide narrative

For per-topic deep dives:
- [research/HMM.md](research/HMM.md) — Hidden Markov regime detection
- [research/HESTON.md](research/HESTON.md) — Stochastic volatility extension
- [research/HJB.md](research/HJB.md) — HJB PDE solver
- [research/IMPACT.md](research/IMPACT.md) — Market impact calibration

For historical/audit material: [audits/README.md](audits/README.md).

---

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/ -v   # 173+ tests
```

```python
from shared.experiment_config import T_1H, LAM, N_STEPS, SEED  # canonical constants
from shared.params import ACParams, almgren_chriss_closed_form
from shared.cost_model import execution_cost, objective
from montecarlo.strategies import twap_trajectory, optimal_trajectory

# Closed-form optimal trajectory
params = ACParams(...)  # see calibration/impact_estimator.py for canonical calibration
t, x_opt, cost = almgren_chriss_closed_form(params)
print(f"Optimal cost: {cost:.2f}")

# Compare with TWAP
x_twap = twap_trajectory(params)
print(f"TWAP cost:    {execution_cost(x_twap, params):.2f}")
```

## Project Structure (post-2026-04-27 cleanup)

| Directory | Part | Status |
|-----------|------|--------|
| `shared/` | — | ✅ Active (params, cost_model, experiment_config) |
| `calibration/` | A | ✅ Active (data_loader, impact_estimator + 2 download CLIs) |
| `pde/` | B | ✅ Active (HJB solver: Riccati for α=1, Howard's policy iteration for α≠1) |
| `montecarlo/` | C | ✅ Active (Euler-Maruyama, Heston Andersen QE, Sobol+BB QMC, CRN paired) |
| `extensions/` | D, E | ✅ Active (Heston Q-measure FFT calibration, 2-state HMM regime detection) |
| `scripts/` | All | ✅ ~13 active analysis scripts (paired tests, walk-forward, calibration time-series) |
| `tests/` | All | ✅ 173+ tests passing |
| `research/` | All | ✅ 4 master docs (HMM/HESTON/HJB/IMPACT) + archive/ |
| `audits/` | All | 📁 dead/ + superseded/ + snapshots/ + data_acquisition/ + audit_chain/ + figures_v1/ |

## Canonical pipeline entry points

| Goal | Script |
|---|---|
| Walk-forward OOS validation (Part B headline) | `scripts/walk_forward_validation.py` |
| Heston Q-measure calibration time-series | `scripts/qmeasure_heston_time_series.py` (Tardis), `scripts/deribit_qmeasure_time_series.py` (live chain) |
| Part D paired test (Heston-Q vs const-vol) | `scripts/paired_test_heston_qmeasure.py` |
| Part E V5 regime-aware paired test | `scripts/paired_test_regime_aware_v5.py` |
| HMM 2-vs-3 state audit (post BIC bug fix) | `scripts/compare_2state_vs_3state_hmm.py`, `scripts/compare_hmm_vol_feature.py` |
| Per-regime OLS impact refit | `scripts/refit_regime_conditional_impact_extended.py` |
| Realized-ρ vs funding-rate analysis | `analysis_funding_vs_realized_rho.py` (root) + `scripts/compute_realized_rho.py` |

## Headline result

**Heston-Q stochastic volatility reduces CVaR₉₅ by 4.79% vs constant-vol** (paired test, 100k common-random-numbers paths, p < 0.0001). See REPORT.md §5 for narrative, RESULTS.md for full numbers, `data/paired_heston_qmeasure_results.json` for raw output.

**Walk-forward OOS**: AC beats TWAP by **+15% to +37% MC savings at X₀ = 1000 BTC across 6/6 splits**, p < 0.0001. Retail boundary: X₀ ≥ 100 BTC (below that, AC ≈ TWAP at noise floor).

**Regime-aware (Part E)**: methodological-learning story — V4 finding invalidated by V5 sign reversal; V6 with extended 280-day window pending.

## Tests

```bash
pytest tests/ -v
```
