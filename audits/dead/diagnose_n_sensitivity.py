"""Diagnose why N=50 vs N=250 gives wildly different savings for alpha < 1.

Shows three things:
  1. The PDE trajectory is bang-bang (all-or-nothing), not smooth
  2. N=50 and N=250 produce opposite bang-bang directions (front vs back)
  3. This is because gamma is missing from the HJB Hamiltonian

Run:  python scripts/diagnose_n_sensitivity.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from dataclasses import replace
from shared.params import ACParams
from shared.cost_model import execution_cost, execution_risk
from montecarlo.strategies import twap_trajectory
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory


# Split-1 approximate params
BASE = ACParams(
    S0=69000.0, sigma=0.32, mu=0.0, X0=10.0,
    T=1 / (365.25 * 24), N=50,
    gamma=1.48, eta=1.58e-4, alpha=0.66, lam=1e-6,
)


def decompose_cost(x_traj, params):
    """Split execution_cost into permanent and temporary components."""
    dt = params.T / params.N
    n_k = x_traj[:-1] - x_traj[1:]
    v_k = n_k / dt
    prior_cumulative = np.concatenate([[0.0], np.cumsum(n_k[:-1])])
    perm = params.gamma * np.sum(n_k * prior_cumulative)
    temp = params.eta * np.sum(np.abs(v_k) ** params.alpha * n_k)
    total = execution_cost(x_traj, params)
    return perm, temp, total


def analyze(label, params):
    """Run PDE + TWAP and print full diagnostic."""
    x_twap = twap_trajectory(params)
    grid, _, v_star = solve_hjb(params, M=200)
    x_opt = extract_optimal_trajectory(grid, v_star, params)
    n_k = x_opt[:-1] - x_opt[1:]

    perm_t, temp_t, total_t = decompose_cost(x_twap, params)
    perm_o, temp_o, total_o = decompose_cost(x_opt, params)
    savings = (total_t - total_o) / abs(total_t) * 100

    # Where does trading happen?
    max_step = int(np.argmax(n_k))
    nonzero = int(np.sum(n_k > 0.01))

    print(f"\n  {label}")
    print(f"  {'-' * 65}")
    print(f"  Inventory path:  t=0: {x_opt[0]:.1f} BTC → "
          f"t=T/2: {x_opt[params.N // 2]:.1f} BTC → "
          f"t=T: {x_opt[-1]:.1f} BTC")
    print(f"  Nonzero trades:  {nonzero} out of {params.N} steps")
    print(f"  Largest trade:   {n_k[max_step]:.2f} BTC at step {max_step} "
          f"({100 * max_step / params.N:.0f}% through horizon)")
    print(f"  TWAP cost:       ${total_t:>10,.2f}  (perm ${perm_t:,.2f} + temp ${temp_t:,.2f})")
    print(f"  PDE cost:        ${total_o:>10,.2f}  (perm ${perm_o:,.2f} + temp ${temp_o:,.2f})")
    print(f"  Savings:         {savings:>+.1f}%")


# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PROBLEM: PDE produces bang-bang control for alpha < 1")
print("=" * 70)

print("\n  ── Same PDE, different N ──")
analyze("N=50  (coarse)", replace(BASE, N=50))
analyze("N=250 (fine)",   replace(BASE, N=250))

print(f"\n  N=50 dumps at the START.  N=250 dumps at the END.")
print(f"  Same PDE solution — trajectory extraction lands on different")
print(f"  sides of a bang-bang boundary depending on step size.")

# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ROOT CAUSE: compare alpha < 1 vs alpha = 1")
print("=" * 70)

print("\n  ── alpha=0.66 (nonlinear — bang-bang) ──")
analyze("N=250, α=0.66", replace(BASE, N=250, alpha=0.66))

print("\n  ── alpha=1.0 (linear — smooth, closed-form) ──")
analyze("N=250, α=1.00", replace(BASE, N=250, alpha=1.0))

print(f"\n  At alpha=1, the trajectory is smooth (sinh curve).")
print(f"  At alpha<1, the weaker penalty |v|^1.66 instead of |v|^2")

print("\n" + "=" * 70)
print("  GRID RESOLUTION TEST: does larger M fix it?")
print("=" * 70)
for M in [200, 500, 1000]:
    p = replace(BASE, N=250)
    grid, _, v_star = solve_hjb(p, M=M)
    x_opt = extract_optimal_trajectory(grid, v_star, p)
    n_k = x_opt[:-1] - x_opt[1:]
    nonzero = int(np.sum(n_k > 0.01))
    max_step = int(np.argmax(n_k))
    _, _, total = decompose_cost(x_opt, p)
    _, _, total_t = decompose_cost(twap_trajectory(p), p)
    sav = (total_t - total) / abs(total_t) * 100
    print(f"  M={M:>5}:  nonzero trades={nonzero:>3}  "
          f"largest at step {max_step:>3}  savings={sav:>+.1f}%")
    
print("\n" + "=" * 70)
print("  FIX: direct optimization (no PDE)")
print("=" * 70)
from scipy.optimize import minimize

for N in [50, 250]:
    p = replace(BASE, N=N)
    
    def neg_obj(trades):
        x = np.concatenate([[p.X0], p.X0 - np.cumsum(trades)])
        return execution_cost(x, p) + p.lam * execution_risk(x, p)
    
    x0_trades = np.full(N, p.X0 / N)
    cons = {"type": "eq", "fun": lambda t: np.sum(t) - p.X0}
    bds = [(0, p.X0)] * N
    res = minimize(neg_obj, x0_trades, method="SLSQP",
                   bounds=bds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    
    x_opt = np.concatenate([[p.X0], p.X0 - np.cumsum(res.x)])
    perm, temp, total = decompose_cost(x_opt, p)
    _, _, total_t = decompose_cost(twap_trajectory(p), p)
    sav = (total_t - total) / abs(total_t) * 100
    nonzero = int(np.sum(res.x > 0.01))
    
    print(f"\n  N={N}:  {nonzero} nonzero trades  "
          f"savings={sav:>+.1f}%  (perm=${perm:,.2f} temp=${temp:,.2f})")
    print(f"    First 5: {res.x[:5].round(4)}")
    print(f"    Path: {x_opt[0]:.1f} → {x_opt[N//2]:.1f} → {x_opt[-1]:.1f}")
