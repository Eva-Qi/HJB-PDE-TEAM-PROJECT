"""High-resolution regime paired test V4 + block bootstrap comparison.

Task 1 (hi-res): Reruns paired_test_regime_aware_v2.py at 100_000 paths.
Verifies VaR/CVaR/objective findings (-14% headline if significant) hold.

Task 3 (block bootstrap): Re-runs the V4 regime test with block bootstrap
and compares IID vs block p-values. Reports if materially different.

All simulations use common random numbers (Z_shared drawn once for 100k paths).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration.data_loader import load_trades, compute_mid_prices
from calibration.impact_estimator import calibrated_params
from extensions.regime import fit_hmm, regime_aware_params
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution
from montecarlo.cost_analysis import (
    paired_strategy_test,
    paired_bootstrap_test_block,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = sorted(DATA_DIR.glob("BTCUSDT-aggTrades-2026-*.csv"))

N_PATHS = 100_000
SEED = 42
N_REGIMES = 2
BOOTSTRAP_REPS = 5_000

X0 = 10.0
T = 1.0 / 24
N_STEPS = 50
LAM = 1e-6


# ---------------------------------------------------------------------------
# Bootstrap helpers (mirrors paired_test_regime_aware_v2.py)
# ---------------------------------------------------------------------------

def _var_95(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 95))


def _cvar_95(arr: np.ndarray) -> float:
    v = np.percentile(arr, 95)
    tail = arr[arr >= v]
    return float(np.mean(tail)) if len(tail) > 0 else float(v)


def _objective(arr: np.ndarray, lam: float = LAM) -> float:
    return float(np.mean(arr) + lam * np.var(arr))


def paired_bootstrap_stat_test(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    stat_fn,
    n_bootstrap: int = 5_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap for a scalar population stat; returns (stat_a, stat_b, pvalue)."""
    assert costs_a.shape == costs_b.shape
    n = len(costs_a)
    rng = np.random.default_rng(seed)
    stat_a = stat_fn(costs_a)
    stat_b = stat_fn(costs_b)
    observed_diff = stat_a - stat_b
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    ba = costs_a[idx]
    bb = costs_b[idx]
    boot_diff = np.array([stat_fn(ba[b]) - stat_fn(bb[b]) for b in range(n_bootstrap)])
    boot_diff_shifted = boot_diff - np.mean(boot_diff)
    p_value = float(np.mean(np.abs(boot_diff_shifted) >= abs(observed_diff)))
    return stat_a, stat_b, p_value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    print("\n" + "=" * 80)
    print("Paired test V4 hi-res + block bootstrap: Regime-Aware vs Single-Regime")
    print(f"N_PATHS={N_PATHS:,}   BOOTSTRAP_REPS={BOOTSTRAP_REPS:,}   seed={SEED}")
    print("=" * 80)

    if not DATA_FILES:
        raise FileNotFoundError(
            f"No BTCUSDT-aggTrades-2026-*.csv files found in {DATA_DIR}."
        )

    # ── Step 1: load data ─────────────────────────────────────────────────
    print(f"\n[1/7] Loading {len(DATA_FILES)} CSV files (full 98-day window)...")
    trades = load_trades(DATA_DIR, start="2026-01-01", end="2026-04-08")
    print(f"      {len(trades):,} trades  "
          f"{trades['timestamp'].iloc[0].date()} → "
          f"{trades['timestamp'].iloc[-1].date()}")

    # ── Step 2: calibrate base (pooled) params ────────────────────────────
    print("[2/7] Calibrating base (pooled) params...")
    cal = calibrated_params(
        trades_path=str(DATA_FILES[0]),
        X0=X0, T=T, N=N_STEPS, lam=LAM,
    )
    base_params = cal.params
    print(f"      S0={base_params.S0:.2f}  sigma={base_params.sigma:.6f}  "
          f"gamma={base_params.gamma:.4e}  eta={base_params.eta:.4e}")

    # ── Step 3: compute 5-min log returns, fit HMM ───────────────────────
    print("[3/7] Computing 5-min returns & fitting 2-state Gaussian HMM...")
    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    returns = mid["log_return"].to_numpy()
    returns = returns[np.isfinite(returns)]
    print(f"      Returns series: {len(returns):,} observations")

    regimes, state_seq = fit_hmm(returns, n_regimes=N_REGIMES)
    reg_dict = {r.label: r for r in regimes}
    risk_on = reg_dict["risk_on"]
    risk_off = reg_dict["risk_off"]

    sigma_ratio = risk_off.sigma / risk_on.sigma
    print(f"      risk_on  sigma={risk_on.sigma:.6f}  prob={risk_on.probability:.4f}")
    print(f"      risk_off sigma={risk_off.sigma:.6f}  prob={risk_off.probability:.4f}")
    print(f"      sigma ratio (off/on): {sigma_ratio:.2f}x")

    # ── Step 4: build per-regime ACParams ─────────────────────────────────
    print("[4/7] Building per-regime ACParams...")
    params_riskon = regime_aware_params(base_params, risk_on)
    params_riskoff = regime_aware_params(base_params, risk_off)

    # ── Step 5: simulate on common random Z ──────────────────────────────
    print(f"[5/7] Simulating {N_PATHS:,} paths (common random numbers)...", end="", flush=True)
    t1 = time.time()
    rng = np.random.default_rng(SEED)
    Z_shared = rng.standard_normal((N_PATHS, N_STEPS))

    x_twap_base = twap_trajectory(base_params)
    x_twap_riskon = twap_trajectory(params_riskon)
    x_twap_riskoff = twap_trajectory(params_riskoff)

    _, costs_single = simulate_execution(
        base_params, x_twap_base,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )
    _, costs_riskon = simulate_execution(
        params_riskon, x_twap_riskon,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )
    _, costs_riskoff = simulate_execution(
        params_riskoff, x_twap_riskoff,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )

    w_on = risk_on.probability
    w_off = risk_off.probability
    total_w = w_on + w_off
    w_on /= total_w
    w_off /= total_w
    costs_aware = w_on * costs_riskon + w_off * costs_riskoff
    print(f"  done ({time.time() - t1:.1f}s)")

    print(f"      mean_single  = {np.mean(costs_single):.6f}")
    print(f"      mean_aware   = {np.mean(costs_aware):.6f}")

    # ── Step 6: run all four IID bootstrap tests (V4 reproduction) ────────
    print(f"\n[6/7] Running V4 paired tests — IID bootstrap (reps={BOOTSTRAP_REPS:,})...")

    # (a) Mean cost
    print("  (a) Mean cost test...")
    result_mean = paired_strategy_test(
        costs_a=costs_aware,
        costs_b=costs_single,
        label_a="RegimeAware",
        label_b="SingleRegime",
        test="both",
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED,
    )
    mean_aware = float(np.mean(costs_aware))
    mean_single = float(np.mean(costs_single))
    mean_diff = result_mean.mean_diff
    p_mean_iid = result_mean.bootstrap_pvalue

    # (b) Objective
    print("  (b) Objective test...")
    obj_aware_full, obj_single_full, p_obj_iid = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=lambda c: _objective(c, lam=LAM),
        n_bootstrap=BOOTSTRAP_REPS, seed=SEED + 1,
    )

    # (c) VaR_95
    print("  (c) VaR_95 test...")
    var_aware_full, var_single_full, p_var_iid = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=_var_95,
        n_bootstrap=BOOTSTRAP_REPS, seed=SEED + 2,
    )

    # (d) CVaR_95
    print("  (d) CVaR_95 test...")
    cvar_aware_full, cvar_single_full, p_cvar_iid = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=_cvar_95,
        n_bootstrap=BOOTSTRAP_REPS, seed=SEED + 3,
    )

    # Compute CVaR reduction percentage headline
    cvar_reduction_pct = (
        (cvar_single_full - cvar_aware_full) / abs(cvar_single_full) * 100
        if cvar_single_full != 0 else float("nan")
    )

    # ── Step 7: block bootstrap on mean cost diff ─────────────────────────
    print(f"\n[7/7] Block bootstrap comparison (mean cost) ...")

    # Default block size: n^(1/3)
    block_size_default = max(1, int(np.ceil(N_PATHS ** (1.0 / 3.0))))
    print(f"      Block size (n^1/3 heuristic): {block_size_default}")

    # Run block bootstrap for mean cost diff
    p_mean_block_default = paired_bootstrap_test_block(
        costs_a=costs_aware,
        costs_b=costs_single,
        block_size=block_size_default,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED,
    )

    # Also test with block_size = 10 (economic: ~10-step clusters)
    p_mean_block_10 = paired_bootstrap_test_block(
        costs_a=costs_aware,
        costs_b=costs_single,
        block_size=10,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED,
    )

    # Check serial correlation in diff array (AR1 coefficient)
    diff_arr = costs_aware - costs_single
    diff_centered = diff_arr - diff_arr.mean()
    ar1_numerator = float(np.dot(diff_centered[:-1], diff_centered[1:]))
    ar1_denom = float(np.dot(diff_centered, diff_centered))
    ar1_coeff = ar1_numerator / ar1_denom if ar1_denom != 0 else 0.0

    # ── Print results ──────────────────────────────────────────────────────
    ALPHA = 0.05
    def sig_label(p): return "YES *" if p < ALPHA else "no"

    print()
    print("=" * 90)
    print(f"RESULTS — Regime-Aware vs Single-Regime (Pooled)  [N={N_PATHS:,}]")
    print("=" * 90)
    print(f"  N paths = {N_PATHS:,}   bootstrap reps = {BOOTSTRAP_REPS:,}   lambda = {LAM:.1e}")
    print()
    header = (
        f"  {'Metric':<18} {'mean(aware)':>14} {'mean(single)':>14} "
        f"{'diff(A-B)':>12} {'IID p-val':>12} {'sig@0.05':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    results_rows = [
        ("mean cost",   mean_aware,      mean_single,      mean_diff,                           p_mean_iid),
        ("objective",   obj_aware_full,  obj_single_full,  obj_aware_full  - obj_single_full,   p_obj_iid),
        ("VaR_95",      var_aware_full,  var_single_full,  var_aware_full  - var_single_full,   p_var_iid),
        ("CVaR_95",     cvar_aware_full, cvar_single_full, cvar_aware_full - cvar_single_full,  p_cvar_iid),
    ]
    for metric, ea, eb, diff, pval in results_rows:
        print(
            f"  {metric:<18} {ea:>14.6f} {eb:>14.6f} "
            f"{diff:>+12.6f} {pval:>12.4f} {sig_label(pval):>10}"
        )

    print()
    print(f"  CVaR_95 reduction (single→aware): {cvar_reduction_pct:+.2f}%")
    print(f"  (V2 10k finding: check if -14% holds at {N_PATHS:,} paths: "
          f"{cvar_reduction_pct:+.2f}%)")

    print()
    print("─" * 90)
    print("BLOCK BOOTSTRAP COMPARISON (mean cost H_0: E[D]=0)")
    print("─" * 90)
    print(f"  AR(1) coefficient of D_i = costs_aware[i] - costs_single[i]: {ar1_coeff:.6f}")
    print(f"  (|AR1| < 0.05 suggests IID is fine; larger → block bootstrap matters)")
    print()
    print(f"  {'Method':<35} {'p-value':>12} {'sig@0.05':>10}")
    print(f"  {'-'*57}")
    print(f"  {'IID bootstrap (standard)':<35} {p_mean_iid:>12.4f} {sig_label(p_mean_iid):>10}")
    print(f"  {'Block bootstrap (block=n^1/3='+ str(block_size_default)+')':<35} {p_mean_block_default:>12.4f} {sig_label(p_mean_block_default):>10}")
    print(f"  {'Block bootstrap (block=10)':<35} {p_mean_block_10:>12.4f} {sig_label(p_mean_block_10):>10}")
    print()

    # Assess materiality
    max_abs_diff = max(
        abs(p_mean_iid - p_mean_block_default),
        abs(p_mean_iid - p_mean_block_10),
    )
    if max_abs_diff > 0.05:
        print(f"  MATERIAL DIFFERENCE: max |p_IID - p_block| = {max_abs_diff:.4f} > 0.05")
        print("  Block bootstrap gives meaningfully different inference. Use block version.")
    else:
        print(f"  NOT MATERIAL: max |p_IID - p_block| = {max_abs_diff:.4f} <= 0.05")
        print("  IID and block bootstrap agree. Serial correlation does not drive results.")

    print()
    print("  Regime parameters:")
    print(f"    risk_on  sigma={risk_on.sigma:.6f}  prob={risk_on.probability:.4f}")
    print(f"    risk_off sigma={risk_off.sigma:.6f}  prob={risk_off.probability:.4f}")
    print(f"    sigma ratio (off/on): {sigma_ratio:.2f}x")
    print("=" * 90)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")

    # ── Save JSON ──────────────────────────────────────────────────────────
    out_path = DATA_DIR / "paired_regime_multihorizon_hires.json"
    out = {
        "version": "v4_hires_blockbootstrap",
        "n_paths": N_PATHS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "lambda": LAM,
        "seed": SEED,
        "regime_sigmas": {
            "risk_on": float(risk_on.sigma),
            "risk_off": float(risk_off.sigma),
            "ratio": float(sigma_ratio),
        },
        "stationary_probs": {
            "risk_on": float(risk_on.probability),
            "risk_off": float(risk_off.probability),
        },
        "weights_used": {
            "risk_on": float(w_on),
            "risk_off": float(w_off),
        },
        "results_iid_bootstrap": {
            "mean_cost": {
                "aware": float(mean_aware),
                "single": float(mean_single),
                "diff": float(mean_diff),
                "p_value": float(p_mean_iid),
                "significant": bool(p_mean_iid < ALPHA),
                "t_statistic": float(result_mean.t_statistic) if result_mean.t_statistic is not None else None,
                "t_pvalue": float(result_mean.t_pvalue) if result_mean.t_pvalue is not None else None,
            },
            "objective": {
                "aware": float(obj_aware_full),
                "single": float(obj_single_full),
                "diff": float(obj_aware_full - obj_single_full),
                "p_value": float(p_obj_iid),
                "significant": bool(p_obj_iid < ALPHA),
            },
            "var_95": {
                "aware": float(var_aware_full),
                "single": float(var_single_full),
                "diff": float(var_aware_full - var_single_full),
                "p_value": float(p_var_iid),
                "significant": bool(p_var_iid < ALPHA),
            },
            "cvar_95": {
                "aware": float(cvar_aware_full),
                "single": float(cvar_single_full),
                "diff": float(cvar_aware_full - cvar_single_full),
                "pct_reduction_single_to_aware": float(cvar_reduction_pct),
                "p_value": float(p_cvar_iid),
                "significant": bool(p_cvar_iid < ALPHA),
            },
        },
        "block_bootstrap_comparison": {
            "ar1_coefficient_of_diff": float(ar1_coeff),
            "iid_p_value": float(p_mean_iid),
            "block_p_value_n_cube_root": float(p_mean_block_default),
            "block_size_n_cube_root": int(block_size_default),
            "block_p_value_size10": float(p_mean_block_10),
            "block_size_10": 10,
            "max_abs_difference_iid_vs_block": float(max_abs_diff),
            "materially_different": bool(max_abs_diff > 0.05),
        },
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
