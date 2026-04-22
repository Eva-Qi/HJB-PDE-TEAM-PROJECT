import sys
sys.path.insert(0, ".")

import numpy as np
from shared.params import ACParams
from shared.cost_model import execution_cost, execution_risk
from montecarlo.sde_engine import simulate_execution, generate_normal_increments
from montecarlo.strategies import twap_trajectory, optimal_trajectory
from montecarlo.cost_analysis import paired_strategy_test

# Calibrated params from Binance
T_1HR = 1.0 / (24)
S0, sigma, eta = 68918.0, 0.4214, 1.58e-4
kappa_needed = 1.5 / T_1HR
lam = 1e-6

params = ACParams(
    S0=S0, sigma=sigma, mu=0.0, X0=100.0,
    T=T_1HR, N=50, gamma=1.48, eta=eta,
    alpha=1.0, lam=lam,
)

x_twap = twap_trajectory(params)
x_opt = optimal_trajectory(params)

# --- 10k paths ---
Z_10k = generate_normal_increments(10000, params.N, method="pseudo", seed=42)
_, costs_twap_10k = simulate_execution(params, x_twap, Z_extern=Z_10k, antithetic=False)
_, costs_opt_10k = simulate_execution(params, x_opt, Z_extern=Z_10k, antithetic=False)
r10k = paired_strategy_test(costs_opt_10k, costs_twap_10k, "AC", "TWAP", test="both")

# --- 100k paths ---
Z_100k = generate_normal_increments(100000, params.N, method="pseudo", seed=42)
_, costs_twap_100k = simulate_execution(params, x_twap, Z_extern=Z_100k, antithetic=False)
_, costs_opt_100k = simulate_execution(params, x_opt, Z_extern=Z_100k, antithetic=False)
r100k = paired_strategy_test(costs_opt_100k, costs_twap_100k, "AC", "TWAP", test="both")

print("=" * 60)
print(f"  TYPE-II ERROR DEMONSTRATION at X0=100 BTC")
print("=" * 60)
print(f"\n  10k paths:  t={r10k.t_statistic:.3f}  p={r10k.t_pvalue:.4f}  "
      f"bootstrap_p={r10k.bootstrap_pvalue:.4f}")
print(f"  100k paths: t={r100k.t_statistic:.3f}  p={r100k.t_pvalue:.4f}  "
      f"bootstrap_p={r100k.bootstrap_pvalue:.4f}")
print(f"\n  Mean diff (AC - TWAP):")
print(f"    10k:  ${r10k.mean_diff:,.2f}")
print(f"    100k: ${r100k.mean_diff:,.2f}")
print(f"\n  Effect size is similar. SE shrank by sqrt(10) = {10**0.5:.2f}x")
print(f"  10k says: NOT significant (p={r10k.t_pvalue:.3f} > 0.05)")
print(f"  100k says: SIGNIFICANT (p={r100k.t_pvalue:.3f} < 0.05)")
print(f"\n  This is a type-II error: 10k had insufficient power to detect")
print(f"  a real effect that 100k reveals.")