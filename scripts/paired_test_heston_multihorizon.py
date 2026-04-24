"""High-resolution paired test: Heston Q-measure vs const-vol at 100k paths.

Task 1 (hi-res): Reruns paired_test_heston_qmeasure.py at 100_000 paths.
Verifies that the 4.63% CVaR reduction finding holds at tighter CI.

Reads Heston Q/P params from data/heston_pmeasure_vs_qmeasure.json (same
source as the original). Uses 50k paths if memory/time is a concern
(documented below).

Scope: const-vol vs Heston-Q (comparison B only — the key test).
We skip comparison A (const vs P) and C (P vs Q) since those are secondary
and the runtime budget is for the hi-res verification.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shared.params import ACParams
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution, simulate_heston_execution
from montecarlo.cost_analysis import paired_strategy_test
from extensions.heston import HestonParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# 100k paths is the goal (Task 1). Heston simulation is O(n_paths * n_steps)
# and each path has a variance sub-process. At 100k x 50 steps this is
# manageable in <5 min on a laptop. We keep 100k.
N_PATHS = 100_000
SEED = 42
BOOTSTRAP_REPS = 5_000
LAM = 1e-6

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARAMS_JSON = DATA_DIR / "heston_pmeasure_vs_qmeasure.json"
OUT_JSON = DATA_DIR / "paired_heston_multihorizon_hires.json"


def _make_base_params(sigma: float) -> ACParams:
    return ACParams(
        S0=69000.0,
        sigma=sigma,
        mu=0.0,
        X0=10.0,
        T=1.0/(365.25*24),
        N=250,
        gamma=1.48,
        eta=1.58e-4,
        alpha=1.0,
        lam=LAM,
        fee_bps=0.0,
    )


def _var_95(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 95))


def _cvar_95(arr: np.ndarray) -> float:
    v = np.percentile(arr, 95)
    tail = arr[arr >= v]
    return float(np.mean(tail)) if len(tail) > 0 else float(v)


def _objective(arr: np.ndarray, lam: float = LAM) -> float:
    return float(np.mean(arr) + lam * np.var(arr))


def paired_bootstrap_stat(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    stat_fn,
    n_bootstrap: int = 5_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Joint paired bootstrap for a scalar stat; returns (stat_a, stat_b, pvalue)."""
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


def main() -> None:
    t0 = time.time()
    print()
    print("=" * 80)
    print(f"Heston Q-measure hi-res test ({N_PATHS:,} paths)")
    print("=" * 80)

    # ── Load Heston params ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading Heston params from {PARAMS_JSON.name}...")
    with open(PARAMS_JSON) as f:
        heston_data = json.load(f)

    qm = heston_data["q_measure"]
    q_params = HestonParams(
        kappa=qm["kappa"],
        theta=qm["theta"],
        xi=qm["xi"],
        rho=qm["rho"],
        v0=qm["v0"],
    )

    sigma_base = float(np.sqrt(q_params.theta))
    base_params = _make_base_params(sigma=sigma_base)
    x_twap = twap_trajectory(base_params)

    print(f"  Q-measure: κ={q_params.kappa:.4f} θ={q_params.theta:.4f} "
          f"ξ={q_params.xi:.4f} ρ={q_params.rho:.4f} v₀={q_params.v0:.4f}")
    print(f"  sigma_base = {sigma_base:.6f}")

    # ── Feller check ───────────────────────────────────────────────────────
    q_feller_lhs = 2 * q_params.kappa * q_params.theta
    q_feller_rhs = q_params.xi ** 2
    print(f"  Feller check Q: 2κθ={q_feller_lhs:.4f}  ξ²={q_feller_rhs:.4f}  "
          f"margin={(q_feller_lhs - q_feller_rhs):.4f} "
          f"({'OK' if q_feller_lhs >= q_feller_rhs else 'VIOLATED'})")

    # ── Simulate ──────────────────────────────────────────────────────────
    print(f"\n[2/4] Simulating {N_PATHS:,} paths (common random numbers, seed={SEED})...")

    print("  (a) Constant-vol simulation...", end="", flush=True)
    t1 = time.time()
    _, costs_const = simulate_execution(
        base_params,
        x_twap,
        n_paths=N_PATHS,
        seed=SEED,
        antithetic=False,
        scheme="exact",
    )
    print(f"  done ({time.time() - t1:.1f}s)")

    print("  (b) Heston Q-measure simulation...", end="", flush=True)
    t1 = time.time()
    with warnings.catch_warnings(record=True) as q_warns:
        warnings.simplefilter("always")
        _, var_paths_Q, costs_heston_Q = simulate_heston_execution(
            base_params,
            q_params,
            x_twap,
            n_paths=N_PATHS,
            seed=SEED,
        )
    print(f"  done ({time.time() - t1:.1f}s)")
    if q_warns:
        print(f"      [Q-measure warnings: {len(q_warns)}]")

    neg_var_frac = float(np.mean(var_paths_Q < 0))
    near_zero_frac = float(np.mean(var_paths_Q < 1e-4))
    print(f"  Variance collapse check: neg={neg_var_frac:.3%}  near-zero={near_zero_frac:.3%}")

    print(f"\n  mean(const-vol)  = {np.mean(costs_const):.6f}")
    print(f"  mean(Heston-Q)   = {np.mean(costs_heston_Q):.6f}")

    # ── Paired tests ──────────────────────────────────────────────────────
    print(f"\n[3/4] Running paired tests (bootstrap reps={BOOTSTRAP_REPS:,})...")

    # Mean cost
    res_mean = paired_strategy_test(
        costs_a=costs_const,
        costs_b=costs_heston_Q,
        label_a="const-vol",
        label_b="Heston-Q",
        test="both",
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED,
    )
    mean_const = float(np.mean(costs_const))
    mean_hq = float(np.mean(costs_heston_Q))
    p_mean = res_mean.bootstrap_pvalue

    # Objective
    obj_const, obj_hq, p_obj = paired_bootstrap_stat(
        costs_const, costs_heston_Q,
        stat_fn=lambda c: _objective(c, lam=LAM),
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + 1,
    )

    # VaR_95
    var_const, var_hq, p_var = paired_bootstrap_stat(
        costs_const, costs_heston_Q,
        stat_fn=_var_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + 2,
    )

    # CVaR_95
    cvar_const, cvar_hq, p_cvar = paired_bootstrap_stat(
        costs_const, costs_heston_Q,
        stat_fn=_cvar_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + 3,
    )

    # CVaR reduction percentage
    cvar_reduction_pct = (cvar_const - cvar_hq) / abs(cvar_const) * 100 if cvar_const != 0 else float("nan")

    # ── Print results ──────────────────────────────────────────────────────
    ALPHA = 0.05
    def sig_label(p): return "YES *" if p < ALPHA else "no"

    print()
    print("=" * 80)
    print(f"RESULTS — const-vol vs Heston-Q  ({N_PATHS:,} paths)")
    print("=" * 80)
    print(f"  N paths = {N_PATHS:,}   bootstrap reps = {BOOTSTRAP_REPS:,}   lambda = {LAM:.1e}")
    print()
    header = (
        f"  {'Metric':<14} {'const-vol':>14} {'Heston-Q':>14} "
        f"{'diff(C-HQ)':>12} {'p-value':>10} {'sig@0.05':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows_data = [
        ("mean cost",  mean_const,  mean_hq,  mean_const  - mean_hq,  p_mean),
        ("objective",  obj_const,   obj_hq,   obj_const   - obj_hq,   p_obj),
        ("VaR_95",     var_const,   var_hq,   var_const   - var_hq,   p_var),
        ("CVaR_95",    cvar_const,  cvar_hq,  cvar_const  - cvar_hq,  p_cvar),
    ]
    for metric, a, b, diff, pval in rows_data:
        print(
            f"  {metric:<14} {a:>14.4f} {b:>14.4f} "
            f"{diff:>+12.4f} {pval:>10.4f} {sig_label(pval):>10}"
        )
    print()
    print(f"  CVaR_95 reduction (const→Heston-Q): {cvar_reduction_pct:+.2f}%")
    print(f"  (Original 10k finding was -4.63%; at {N_PATHS:,} paths: "
          f"{cvar_reduction_pct:+.2f}%)")
    print()
    print(f"  t_statistic (mean cost): {res_mean.t_statistic:.4f}")
    print(f"  t_pvalue    (mean cost): {res_mean.t_pvalue:.6f}")
    print("=" * 80)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")

    # ── Save JSON ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving results to {OUT_JSON.name}...")
    out = {
        "version": "heston_hires_v1",
        "n_paths": N_PATHS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "lambda": LAM,
        "seed": SEED,
        "sigma_base": float(sigma_base),
        "q_params": {
            "kappa": float(q_params.kappa),
            "theta": float(q_params.theta),
            "xi": float(q_params.xi),
            "rho": float(q_params.rho),
            "v0": float(q_params.v0),
            "feller_lhs": float(q_feller_lhs),
            "feller_rhs": float(q_feller_rhs),
            "feller_satisfied": bool(q_feller_lhs >= q_feller_rhs),
        },
        "variance_collapse_check": {
            "negative_fraction": float(neg_var_frac),
            "near_zero_fraction": float(near_zero_frac),
        },
        "summary_means": {
            "const_vol": float(mean_const),
            "heston_Q": float(mean_hq),
        },
        "results": {
            "mean_cost": {
                "const_vol": float(mean_const),
                "heston_Q": float(mean_hq),
                "diff": float(mean_const - mean_hq),
                "p_value_bootstrap": float(p_mean),
                "t_statistic": float(res_mean.t_statistic) if res_mean.t_statistic is not None else None,
                "t_pvalue": float(res_mean.t_pvalue) if res_mean.t_pvalue is not None else None,
                "significant": bool(p_mean < ALPHA),
            },
            "objective": {
                "const_vol": float(obj_const),
                "heston_Q": float(obj_hq),
                "diff": float(obj_const - obj_hq),
                "p_value": float(p_obj),
                "significant": bool(p_obj < ALPHA),
            },
            "var_95": {
                "const_vol": float(var_const),
                "heston_Q": float(var_hq),
                "diff": float(var_const - var_hq),
                "p_value": float(p_var),
                "significant": bool(p_var < ALPHA),
            },
            "cvar_95": {
                "const_vol": float(cvar_const),
                "heston_Q": float(cvar_hq),
                "diff": float(cvar_const - cvar_hq),
                "pct_reduction": float(cvar_reduction_pct),
                "p_value": float(p_cvar),
                "significant": bool(p_cvar < ALPHA),
            },
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
