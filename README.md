# MF796 Term Project: Optimal Execution Under Stochastic Volatility

Almgren-Chriss optimal execution with PDE + Monte Carlo methods, calibrated on real Binance order book data.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from shared.params import DEFAULT_PARAMS, almgren_chriss_closed_form
from shared.cost_model import execution_cost, objective
from montecarlo.strategies import twap_trajectory, optimal_trajectory

# Closed-form optimal trajectory
t, x_opt, cost = almgren_chriss_closed_form(DEFAULT_PARAMS)
print(f"Optimal cost: {cost:.2f}")

# Compare with TWAP
x_twap = twap_trajectory(DEFAULT_PARAMS)
print(f"TWAP cost:    {execution_cost(x_twap, DEFAULT_PARAMS):.2f}")
```

## Project Structure

| Directory | Owner | Part | Status |
|-----------|-------|------|--------|
| `shared/` | All | — | **Ready** (params, cost model, plotting) |
| `calibration/` | P1 | A | Skeleton (data loading + impact estimation) |
| `pde/` | P2 | B | Skeleton (HJB finite difference solver) |
| `montecarlo/` | P3 | C | Partial (strategies implemented, MC engine skeleton) |
| `extensions/` | All | D, E | Skeleton (Heston CF ready, calibration + HMM skeleton) |

## How to Start Working

### P1 (Data & Calibration)
1. Implement `calibration/data_loader.py` — load Binance trade + OB data
2. Implement `calibration/impact_estimator.py` — estimate gamma, eta, alpha
3. Deliver: `calibrated_params() -> ACParams`

### P2 (PDE)
1. Use `DEFAULT_PARAMS` to develop
2. Implement `pde/hjb_solver.py` — HJB finite difference solver
3. Validate: PDE trajectory vs `almgren_chriss_closed_form()` within 1%

### P3 (Monte Carlo)
1. Use `DEFAULT_PARAMS` + `strategies.py` (TWAP, optimal already work)
2. Implement `montecarlo/sde_engine.py` — Euler-Maruyama simulation
3. Validate: MC expected cost vs closed-form within 5%

## Interface Contracts

```
P1 delivers: calibrated_params() -> ACParams
P2 delivers: solve_hjb(params) -> (grid, V, v_star)
             extract_optimal_trajectory(grid, v_star, params) -> x_trajectory
P3 delivers: simulate_execution(params, trajectory, n_paths) -> (paths, costs)

Cross-validation: |cost_PDE - mean(cost_MC)| / cost_PDE < 5%
```

## Tests

```bash
cd mf796_project
pytest tests/ -v
```
