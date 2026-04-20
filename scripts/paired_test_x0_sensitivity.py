"""Paired statistical test: AC vs TWAP across X0 using common random numbers.

Uses the `paired_strategy_test` from montecarlo.cost_analysis (added by
teammate bp in commit 795d8d8) to formally test whether the ~0-1%
AC-vs-TWAP savings observed in x0_sensitivity_analysis.py are
statistically significant or within simulation noise.

Simulates both strategies on the SAME Z-paths for each X0 so the
per-path cost difference is a clean paired observation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shared.params import ACParams
from shared.cost_model import execution_cost, objective
from montecarlo.strategies import twap_trajectory, optimal_trajectory
from montecarlo.sde_engine import simulate_execution
from montecarlo.cost_analysis import paired_strategy_test


# Calibrated params matching scripts/x0_sensitivity_analysis.py
BASE_PARAMS = ACParams(
    S0=69000.0, sigma=0.396, mu=0.0, X0=10.0,
    T=1 / 24, N=50,
    gamma=1.48, eta=1.58e-4, alpha=1.0, lam=1e-6,
    fee_bps=7.5,
)

X0_VALUES = [1.0, 10.0, 100.0, 1000.0, 10000.0]
N_PATHS = 10_000
SEED = 42


def run_one(x0: float) -> dict:
    p = replace(BASE_PARAMS, X0=x0)
    x_twap = twap_trajectory(p)
    x_opt = optimal_trajectory(p)

    # Common random numbers: same seed for both simulations
    _, costs_twap = simulate_execution(
        p, x_twap, n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
    )
    _, costs_opt = simulate_execution(
        p, x_opt, n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
    )

    # Paired test: AC vs TWAP. Positive mean_diff → AC costs MORE than TWAP.
    result = paired_strategy_test(
        costs_a=costs_opt, costs_b=costs_twap,
        label_a="AC_Optimal", label_b="TWAP",
        test="both", n_bootstrap=5_000, seed=SEED,
    )

    # Also compute the deterministic objective gap for reference
    obj_twap = objective(x_twap, p)
    obj_opt = objective(x_opt, p)

    return {
        "X0": x0,
        "mean_cost_twap": float(np.mean(costs_twap)),
        "mean_cost_ac": float(np.mean(costs_opt)),
        "mean_diff_ac_minus_twap": result.mean_diff,
        "t_pvalue": result.t_pvalue,
        "bootstrap_pvalue": result.bootstrap_pvalue,
        "n_paths": result.n_paths,
        "significant_at_0.05": (
            (result.t_pvalue is not None and result.t_pvalue < 0.05)
            and (result.bootstrap_pvalue is not None
                 and result.bootstrap_pvalue < 0.05)
        ),
        "det_obj_twap": obj_twap,
        "det_obj_ac": obj_opt,
    }


def main() -> None:
    rows = []
    print("\nRunning paired AC-vs-TWAP tests across X0 range...")
    print("=" * 90)
    for x0 in X0_VALUES:
        r = run_one(x0)
        rows.append(r)
        print(
            f"X0={r['X0']:>8.0f}  "
            f"mean(AC)={r['mean_cost_ac']:>12.2f}  "
            f"mean(TWAP)={r['mean_cost_twap']:>12.2f}  "
            f"diff={r['mean_diff_ac_minus_twap']:>+12.2f}"
        )
    print("=" * 90)
    print()
    print("Significance table (H_0: E[C_AC - C_TWAP] = 0)")
    print("-" * 90)
    print(
        f"{'X0(BTC)':>10} {'mean_diff':>14} "
        f"{'t_pvalue':>12} {'boot_pvalue':>14} {'significant':>14} "
        f"{'det_obj_Δ':>14}"
    )
    print("-" * 90)
    for r in rows:
        det_delta = r["det_obj_ac"] - r["det_obj_twap"]
        sig = "YES" if r["significant_at_0.05"] else "no"
        tp = f"{r['t_pvalue']:.4f}" if r['t_pvalue'] is not None else "—"
        bp = f"{r['bootstrap_pvalue']:.4f}" if r['bootstrap_pvalue'] is not None else "—"
        print(
            f"{r['X0']:>10.0f} {r['mean_diff_ac_minus_twap']:>+14.2f} "
            f"{tp:>12} {bp:>14} {sig:>14} {det_delta:>+14.2f}"
        )
    print("-" * 90)
    print()
    print("Interpretation:")
    print("  • mean_diff > 0  → AC costs MORE than TWAP on average (AC worse)")
    print("  • mean_diff < 0  → AC costs LESS than TWAP on average (AC better)")
    print("  • significant=YES → both t and bootstrap reject H_0 at α=0.05")
    print("  • det_obj_Δ  = deterministic objective gap (E[cost] + λ·risk)")
    print()

    out_path = Path(__file__).resolve().parent.parent / "data" / "paired_test_results.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
