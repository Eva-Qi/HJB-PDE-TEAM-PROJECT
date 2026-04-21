"""High-resolution paired test: AC vs TWAP across X0 + multi-horizon analysis.

Two experiments in one script:

  1. Hi-res X0 sweep (Task 1): Reproduces paired_test_x0_sensitivity.py at
     100_000 paths (10x upgrade) to confirm significance holds at tighter CI.

  2. Multi-horizon test (Task 2): AC vs TWAP at T = 1h, 6h, 1d (10_000 paths
     each — 100k not needed; the horizon sweep is the scientific question).
     Hypothesis: regime-awareness / AC benefit scales with horizon.

Both use common random numbers (same seed for AC and TWAP within each cell).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shared.params import ACParams
from shared.cost_model import execution_cost, objective
from montecarlo.strategies import twap_trajectory, optimal_trajectory
from montecarlo.sde_engine import simulate_execution
from montecarlo.cost_analysis import paired_strategy_test


# ---------------------------------------------------------------------------
# Shared base parameters (match paired_test_x0_sensitivity.py exactly)
# ---------------------------------------------------------------------------
BASE_PARAMS = ACParams(
    S0=69000.0, sigma=0.396, mu=0.0, X0=10.0,
    T=1 / 24, N=50,
    gamma=1.48, eta=1.58e-4, alpha=1.0, lam=1e-6,
    fee_bps=7.5,
)

SEED = 42


# ===========================================================================
# Task 1: Hi-res X0 sweep (100k paths)
# ===========================================================================
X0_VALUES = [1.0, 10.0, 100.0, 1000.0, 10000.0]
N_PATHS_HIRES = 100_000


def run_x0_hires(x0: float) -> dict:
    """Run paired AC-vs-TWAP test at 100k paths for one X0 value."""
    p = replace(BASE_PARAMS, X0=x0)
    x_twap = twap_trajectory(p)
    x_opt = optimal_trajectory(p)

    # Common random numbers
    _, costs_twap = simulate_execution(
        p, x_twap, n_paths=N_PATHS_HIRES, seed=SEED,
        antithetic=False, scheme="exact",
    )
    _, costs_opt = simulate_execution(
        p, x_opt, n_paths=N_PATHS_HIRES, seed=SEED,
        antithetic=False, scheme="exact",
    )

    result = paired_strategy_test(
        costs_a=costs_opt, costs_b=costs_twap,
        label_a="AC_Optimal", label_b="TWAP",
        test="both", n_bootstrap=5_000, seed=SEED,
    )

    obj_twap = objective(x_twap, p)
    obj_opt = objective(x_opt, p)

    return {
        "X0": x0,
        "n_paths": N_PATHS_HIRES,
        "mean_cost_twap": float(np.mean(costs_twap)),
        "mean_cost_ac": float(np.mean(costs_opt)),
        "mean_diff_ac_minus_twap": result.mean_diff,
        "t_statistic": result.t_statistic,
        "t_pvalue": result.t_pvalue,
        "bootstrap_pvalue": result.bootstrap_pvalue,
        "significant_at_0.05": (
            (result.t_pvalue is not None and result.t_pvalue < 0.05)
            and (result.bootstrap_pvalue is not None
                 and result.bootstrap_pvalue < 0.05)
        ),
        "det_obj_twap": obj_twap,
        "det_obj_ac": obj_opt,
    }


# ===========================================================================
# Task 2: Multi-horizon test (T = 1h, 6h, 1d) at 10k paths
# ===========================================================================

# Horizon definitions: label, T in years (1 year = 1 unit), N steps
HORIZONS = [
    {"label": "1h",  "T": 1 / 24,       "N": 50},
    {"label": "6h",  "T": 6 / 24,       "N": 100},
    {"label": "1d",  "T": 1.0,          "N": 250},
]

N_PATHS_HORIZON = 10_000
LAM = 1e-6


def _var_95(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 95))


def _cvar_95(arr: np.ndarray) -> float:
    v = np.percentile(arr, 95)
    tail = arr[arr >= v]
    return float(np.mean(tail)) if len(tail) > 0 else float(v)


def _objective_fn(arr: np.ndarray, lam: float = LAM) -> float:
    return float(np.mean(arr) + lam * np.var(arr))


def paired_bootstrap_stat(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    stat_fn,
    n_bootstrap: int = 5_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Joint paired bootstrap for a scalar statistic; returns (stat_a, stat_b, pvalue)."""
    n = len(costs_a)
    rng = np.random.default_rng(seed)
    sa = stat_fn(costs_a)
    sb = stat_fn(costs_b)
    obs_diff = sa - sb

    idx = rng.integers(0, n, size=(n_bootstrap, n))
    ba = costs_a[idx]
    bb = costs_b[idx]
    boot_diff = np.array([stat_fn(ba[b]) - stat_fn(bb[b]) for b in range(n_bootstrap)])
    boot_diff_shifted = boot_diff - np.mean(boot_diff)
    pval = float(np.mean(np.abs(boot_diff_shifted) >= abs(obs_diff)))
    return sa, sb, pval


def _safe_optimal_trajectory(p) -> tuple[np.ndarray, bool]:
    """Return (x_opt, is_degenerate). Falls back to bang-bang if kappa*T overflows.

    At kappa*T >> 1, sinh(kappa*T) overflows to inf. The AC optimal trajectory
    in that limit is a bang-bang schedule: trade everything instantly at t=0.
    We detect the overflow and return a front-loaded bang-bang proxy instead,
    flagging the result as degenerate so callers can note it.
    """
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        x_opt = optimal_trajectory(p)

    if not np.all(np.isfinite(x_opt)):
        # Bang-bang limit: all shares in first step, nothing remaining
        x_bang = np.zeros(p.N + 1)
        x_bang[0] = p.X0
        return x_bang, True

    return x_opt, False


def run_horizon(horizon: dict) -> list[dict]:
    """Run AC vs TWAP at one horizon; return rows for mean_cost, objective, CVaR.

    Note: at T=1d with BTC-calibrated params, kappa*T ~ 2174, causing sinh
    overflow.  The AC optimal degenerates to a bang-bang (all shares in step 1).
    We detect and flag this as 'ac_degenerate=True' in the output.
    """
    label = horizon["label"]
    T = horizon["T"]
    N = horizon["N"]

    p = replace(BASE_PARAMS, T=T, N=N)
    x_twap = twap_trajectory(p)
    x_opt, ac_degenerate = _safe_optimal_trajectory(p)

    _, costs_twap = simulate_execution(
        p, x_twap, n_paths=N_PATHS_HORIZON, seed=SEED,
        antithetic=False, scheme="exact",
    )
    _, costs_opt = simulate_execution(
        p, x_opt, n_paths=N_PATHS_HORIZON, seed=SEED,
        antithetic=False, scheme="exact",
    )

    # Mean cost (t + bootstrap)
    res_mean = paired_strategy_test(
        costs_a=costs_opt, costs_b=costs_twap,
        label_a="AC", label_b="TWAP",
        test="both", n_bootstrap=5_000, seed=SEED,
    )

    # Objective
    obj_ac, obj_twap_v, p_obj = paired_bootstrap_stat(
        costs_opt, costs_twap,
        stat_fn=lambda c: _objective_fn(c, lam=LAM),
        n_bootstrap=5_000, seed=SEED + 1,
    )

    # CVaR
    cvar_ac, cvar_twap_v, p_cvar = paired_bootstrap_stat(
        costs_opt, costs_twap,
        stat_fn=_cvar_95,
        n_bootstrap=5_000, seed=SEED + 2,
    )

    degen_note = ("AC optimal degenerate: kappa*T overflow (bang-bang limit). "
                  "Results reflect AC=immediate execution vs TWAP."
                  if ac_degenerate else "")

    rows = [
        {
            "T_label": label,
            "T": T,
            "metric": "mean_cost",
            "val_ac": float(np.mean(costs_opt)),
            "val_twap": float(np.mean(costs_twap)),
            "diff_ac_minus_twap": float(np.mean(costs_opt) - np.mean(costs_twap)),
            "t_pvalue": res_mean.t_pvalue,
            "bootstrap_pvalue": res_mean.bootstrap_pvalue,
            "significant_at_0.05": (
                res_mean.t_pvalue is not None and res_mean.t_pvalue < 0.05
                and res_mean.bootstrap_pvalue is not None and res_mean.bootstrap_pvalue < 0.05
            ),
            "ac_degenerate": ac_degenerate,
            "note": degen_note,
        },
        {
            "T_label": label,
            "T": T,
            "metric": "objective",
            "val_ac": obj_ac,
            "val_twap": obj_twap_v,
            "diff_ac_minus_twap": obj_ac - obj_twap_v,
            "t_pvalue": None,
            "bootstrap_pvalue": p_obj,
            "significant_at_0.05": p_obj < 0.05,
            "ac_degenerate": ac_degenerate,
            "note": degen_note,
        },
        {
            "T_label": label,
            "T": T,
            "metric": "cvar_95",
            "val_ac": cvar_ac,
            "val_twap": cvar_twap_v,
            "diff_ac_minus_twap": cvar_ac - cvar_twap_v,
            "t_pvalue": None,
            "bootstrap_pvalue": p_cvar,
            "significant_at_0.05": p_cvar < 0.05,
            "ac_degenerate": ac_degenerate,
            "note": degen_note,
        },
    ]
    return rows


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    t0 = time.time()

    # ── Task 1: Hi-res X0 sweep ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 1: AC vs TWAP — Hi-res X0 sweep (100k paths)")
    print("=" * 90)

    x0_rows = []
    for x0 in X0_VALUES:
        t_start = time.time()
        print(f"  X0={x0:>8.0f}  running {N_PATHS_HIRES:,} paths...", end="", flush=True)
        r = run_x0_hires(x0)
        x0_rows.append(r)
        print(f"  done ({time.time() - t_start:.1f}s)"
              f"  diff={r['mean_diff_ac_minus_twap']:+.2f}"
              f"  t_p={r['t_pvalue']:.4f}"
              f"  boot_p={r['bootstrap_pvalue']:.4f}"
              f"  sig={r['significant_at_0.05']}")

    print()
    print("Hi-res significance table (H_0: E[C_AC - C_TWAP] = 0)")
    print("-" * 90)
    print(f"{'X0(BTC)':>10} {'mean_diff':>14} {'t_pvalue':>12} "
          f"{'boot_pvalue':>14} {'significant':>14} {'det_obj_Δ':>14}")
    print("-" * 90)
    for r in x0_rows:
        det_delta = r["det_obj_ac"] - r["det_obj_twap"]
        sig = "YES" if r["significant_at_0.05"] else "no"
        tp = f"{r['t_pvalue']:.4f}" if r["t_pvalue"] is not None else "—"
        bp = f"{r['bootstrap_pvalue']:.4f}" if r["bootstrap_pvalue"] is not None else "—"
        print(f"{r['X0']:>10.0f} {r['mean_diff_ac_minus_twap']:>+14.2f} "
              f"{tp:>12} {bp:>14} {sig:>14} {det_delta:>+14.2f}")
    print("-" * 90)

    # ── Task 2: Multi-horizon test ───────────────────────────────────────────
    print()
    print("=" * 90)
    print("TASK 2: AC vs TWAP — Multi-horizon test (10k paths each)")
    print("Hypothesis: AC benefit scales with horizon (more regime shift opportunity)")
    print("=" * 90)

    horizon_rows = []
    for hz in HORIZONS:
        t_start = time.time()
        print(f"  T={hz['label']:>4}  running {N_PATHS_HORIZON:,} paths...", end="", flush=True)
        rows = run_horizon(hz)
        horizon_rows.extend(rows)
        print(f"  done ({time.time() - t_start:.1f}s)")

    print()
    print("Multi-horizon significance table")
    print("-" * 90)
    header = (
        f"{'T':>6} {'Metric':<14} {'AC':>14} {'TWAP':>14} "
        f"{'diff(AC-TWAP)':>14} {'p-value':>10} {'sig':>6}"
    )
    print(header)
    print("-" * 90)
    for r in horizon_rows:
        tp = f"{r['bootstrap_pvalue']:.4f}" if r["bootstrap_pvalue"] is not None else "—"
        sig = "YES *" if r["significant_at_0.05"] else "no"
        degen_flag = " [degen]" if r.get("ac_degenerate") else ""
        val_ac_str = f"{r['val_ac']:>14.4f}" if np.isfinite(r["val_ac"]) else f"{'DEGEN':>14}"
        diff_str = (f"{r['diff_ac_minus_twap']:>+14.4f}"
                    if np.isfinite(r["diff_ac_minus_twap"]) else f"{'DEGEN':>+14}")
        print(
            f"{r['T_label']:>6} {r['metric']:<14} {val_ac_str} "
            f"{r['val_twap']:>14.4f} {diff_str} "
            f"{tp:>10} {sig:>6}{degen_flag}"
        )
    print("-" * 90)

    # Narrative: look for horizon scaling
    print()
    print("HORIZON SCALING NARRATIVE:")
    mean_cost_by_horizon = {
        r["T_label"]: r["diff_ac_minus_twap"]
        for r in horizon_rows if r["metric"] == "mean_cost"
    }
    for label, diff in mean_cost_by_horizon.items():
        if np.isfinite(diff):
            direction = "AC saves" if diff < 0 else "AC costs MORE"
            print(f"  T={label}: mean_diff={diff:+.4f}  → {direction} vs TWAP")
        else:
            print(f"  T={label}: AC degenerate (kappa*T overflow) — bang-bang limit")
    finite_diffs = [(lbl, d) for lbl, d in mean_cost_by_horizon.items() if np.isfinite(d)]
    diffs = [d for _, d in finite_diffs]
    if len(diffs) >= 2:
        if all(diffs[i] <= diffs[i + 1] for i in range(len(diffs) - 1)):
            print("  Pattern (finite horizons): AC advantage INCREASES with horizon (supports hypothesis)")
        elif all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1)):
            print("  Pattern (finite horizons): AC advantage DECREASES with horizon (contra hypothesis)")
        else:
            print("  Pattern (finite horizons): Non-monotone across horizons")
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(__file__).resolve().parent.parent / "data" / "paired_ac_vs_twap_hires.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "description": "Hi-res AC vs TWAP (100k paths X0 sweep + multi-horizon 10k)",
            "seed": SEED,
        },
        "x0_sweep_100k": x0_rows,
        "multi_horizon_10k": horizon_rows,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
