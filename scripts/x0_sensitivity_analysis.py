"""X0 Sensitivity Analysis: At what order size does AC beat TWAP meaningfully?

Sweeps X0 in [1, 10, 100, 1000, 10000] BTC, computes execution cost and
objective for TWAP, VWAP, POV, and optimal (AC) trajectories.

Key question: at what X0 does AC save > 1% on objective post-fee vs TWAP?
"""

from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from pathlib import Path

# Allow running from scripts/ without package install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.impact_estimator import calibrated_params
from shared.cost_model import execution_cost, execution_risk, objective, execution_fees
from montecarlo.strategies import (
    twap_trajectory,
    vwap_trajectory,
    pov_trajectory,
    optimal_trajectory,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
X0_VALUES = [1, 10, 100, 1000, 10000]
POV_RATE = 0.15          # 15% participation — ensures liquidation in N=50 steps
OUTPUT_JSON = str(Path(__file__).resolve().parent.parent / "data" / "x0_sensitivity_results.json")
OUTPUT_PLOT = str(Path(__file__).resolve().parent.parent / "plot_x0_sensitivity.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_savings(baseline: float, improved: float) -> float:
    """Percent savings: positive = improved is cheaper."""
    if abs(baseline) < 1e-12:
        return 0.0
    return (baseline - improved) / abs(baseline) * 100.0


def run_one(x0: float, base_params) -> dict:
    """Compute metrics for a single X0 value."""
    import copy
    p = copy.copy(base_params)
    p.X0 = x0

    # Closed-form optimal trajectory requires alpha=1 (linear impact).
    # If calibrated alpha != 1, use alpha=1 copy for AC trajectory only.
    # All cost evaluations use the original alpha (p), not the alpha=1 copy.
    if abs(p.alpha - 1.0) < 1e-10:
        p_ac = p
    else:
        p_ac = copy.copy(p)
        p_ac.alpha = 1.0

    x_twap = twap_trajectory(p)
    x_vwap = vwap_trajectory(p)
    x_pov  = pov_trajectory(p, participation_rate=POV_RATE)
    x_ac   = optimal_trajectory(p_ac)  # uses alpha=1 copy if needed

    results = {}
    for name, x in [("TWAP", x_twap), ("VWAP", x_vwap), ("POV", x_pov), ("AC", x_ac)]:
        cost_nofee = execution_cost(x, p) - execution_fees(x, p)
        cost_withfee = execution_cost(x, p)
        risk = execution_risk(x, p)
        obj_nofee = cost_nofee + p.lam * risk
        obj_withfee = cost_withfee + p.lam * risk
        fees = execution_fees(x, p)
        results[name] = {
            "cost_nofee": cost_nofee,
            "cost_withfee": cost_withfee,
            "risk": risk,
            "obj_nofee": obj_nofee,
            "obj_withfee": obj_withfee,
            "fees": fees,
        }

    # Savings: AC vs TWAP
    twap_cost = results["TWAP"]["cost_withfee"]
    twap_obj  = results["TWAP"]["obj_withfee"]
    ac_cost   = results["AC"]["cost_withfee"]
    ac_obj    = results["AC"]["obj_withfee"]

    return {
        "X0": x0,
        "S0": p.S0,
        "strategies": results,
        "savings_cost_pct": pct_savings(twap_cost, ac_cost),
        "savings_obj_pct":  pct_savings(twap_obj,  ac_obj),
        "fees_pct_of_twap_cost": results["TWAP"]["fees"] / abs(twap_cost) * 100
            if abs(twap_cost) > 1e-12 else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nLoading calibrated params (X0=10 BTC base)...")
    cal = calibrated_params(trades_path=DATA_DIR)
    base_params = cal.params
    print(f"  S0={base_params.S0:.2f}, sigma={base_params.sigma:.4f}, "
          f"eta={base_params.eta:.4e}, gamma={base_params.gamma:.4e}, "
          f"alpha={base_params.alpha:.4f}, fee_bps={base_params.fee_bps}")
    print(f"  kappa={base_params.kappa:.4f}, kappa*T={base_params.kappa*base_params.T:.4f}\n")

    all_results = []
    for x0 in X0_VALUES:
        r = run_one(x0, base_params)
        all_results.append(r)

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    header = (
        f"{'X0(BTC)':>10} | "
        f"{'AC_cost':>14} | "
        f"{'TWAP_cost':>14} | "
        f"{'AC_obj':>14} | "
        f"{'TWAP_obj':>14} | "
        f"{'%sav_cost':>10} | "
        f"{'%sav_obj':>9} | "
        f"{'fees_pct':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in all_results:
        x0 = r["X0"]
        ac   = r["strategies"]["AC"]
        twap = r["strategies"]["TWAP"]
        line = (
            f"{x0:>10,.0f} | "
            f"{ac['cost_withfee']:>14,.2f} | "
            f"{twap['cost_withfee']:>14,.2f} | "
            f"{ac['obj_withfee']:>14,.2f} | "
            f"{twap['obj_withfee']:>14,.2f} | "
            f"{r['savings_cost_pct']:>9.3f}% | "
            f"{r['savings_obj_pct']:>8.3f}% | "
            f"{r['fees_pct_of_twap_cost']:>8.3f}%"
        )
        print(line)
    print(sep)

    # ------------------------------------------------------------------
    # Answer the key question
    # ------------------------------------------------------------------
    print("\nAt what X0 does AC beat TWAP > 1% on objective (post-fee)?")
    threshold_found = False
    for r in all_results:
        if r["savings_obj_pct"] > 1.0:
            print(f"  --> X0 = {r['X0']:,} BTC: {r['savings_obj_pct']:.3f}% savings on objective")
            if not threshold_found:
                threshold_found = True
    if not threshold_found:
        print("  --> None of the tested X0 values reach > 1% savings on objective post-fee.")
        # Find the breakeven point
        for r in all_results:
            print(f"  X0={r['X0']:>8,.0f}: {r['savings_obj_pct']:.4f}%")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    # Convert numpy floats to Python floats for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    x0_arr = np.array(X0_VALUES, dtype=float)
    ac_costs   = [r["strategies"]["AC"]["cost_withfee"]   for r in all_results]
    twap_costs = [r["strategies"]["TWAP"]["cost_withfee"] for r in all_results]
    vwap_costs = [r["strategies"]["VWAP"]["cost_withfee"] for r in all_results]
    pov_costs  = [r["strategies"]["POV"]["cost_withfee"]  for r in all_results]
    sav_obj    = [r["savings_obj_pct"]                    for r in all_results]
    fees_pct   = [r["fees_pct_of_twap_cost"]              for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("X0 Sensitivity Analysis: AC vs Benchmark Strategies", fontsize=13)

    # --- Subplot 1: costs by X0 (log-log) ---
    ax = axes[0]
    ax.loglog(x0_arr, ac_costs,   "o-", color="#1D9E75", label="AC (Optimal)", linewidth=2)
    ax.loglog(x0_arr, twap_costs, "s--", color="#378ADD", label="TWAP",         linewidth=2)
    ax.loglog(x0_arr, vwap_costs, "^--", color="#F59E0B", label="VWAP",         linewidth=2)
    ax.loglog(x0_arr, pov_costs,  "D--", color="#D85A30", label="POV (15%)",    linewidth=2)
    ax.set_xlabel("X0 (BTC)")
    ax.set_ylabel("Execution Cost (USD, post-fee)")
    ax.set_title("Cost by Order Size")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(x0_arr)
    ax.set_xticklabels([str(int(v)) for v in x0_arr])

    # --- Subplot 2: % savings AC vs TWAP on objective ---
    ax = axes[1]
    colors_bar = ["#d0d0d0" if s <= 1.0 else "#1D9E75" for s in sav_obj]
    bars = ax.bar([str(int(v)) for v in x0_arr], sav_obj, color=colors_bar, edgecolor="black", linewidth=0.7)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label=">1% threshold")
    ax.set_xlabel("X0 (BTC)")
    ax.set_ylabel("% Savings on Objective (AC vs TWAP)")
    ax.set_title("AC vs TWAP: Objective Savings")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, s in zip(bars, sav_obj):
        ax.text(bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0) + 0.02 * max(abs(v) for v in sav_obj),
                f"{s:.2f}%", ha="center", va="bottom", fontsize=9)

    # --- Subplot 3: fees as % of total cost ---
    ax = axes[2]
    ax.semilogx(x0_arr, fees_pct, "o-", color="#8B5CF6", linewidth=2, markersize=8)
    ax.set_xlabel("X0 (BTC)")
    ax.set_ylabel("Fees as % of TWAP Total Cost")
    ax.set_title("Fee Drag by Order Size")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(x0_arr)
    ax.set_xticklabels([str(int(v)) for v in x0_arr])
    for xi, yi in zip(x0_arr, fees_pct):
        ax.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
