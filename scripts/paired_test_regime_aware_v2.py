"""Paired statistical test V4: regime-aware vs single-regime on RISK-SIDE metrics.

V1-V3 tested mean cost (H_0: E[C_aware - C_single] = 0) and got p > 0.8.
This script tests THREE metrics that should capture the regime-aware benefit:

  a) Mean cost (reproduces V3 as a sanity check, expect p≈0.84)
  b) Objective = E[cost] + lambda * Var[cost]  (bootstrap comparison)
  c) VaR_95 = 95th percentile of cost distribution (bootstrap comparison)
  d) CVaR_95 = E[cost | cost >= VaR_95] (bootstrap comparison)

Design rationale:
    AC optimizes (E[cost] + lambda * Var[cost]).  Regime-aware params
    use a LOWER sigma in risk_on (~90% of time) and are better tuned for
    risk_off (~10% of time where sigma is 3.5x normal).  Mean cost averages
    over the mixing distribution and washes out the benefit.  The tail risk
    metrics (VaR, CVaR) and the variance-penalized objective should reveal
    whether regime-awareness reduces tail cost / variance.

Bootstrap methodology for scalar population metrics:
    We cannot pair observations of VaR (VaR is a population statistic, not
    per-path).  Instead we use the percentile bootstrap on the DIFFERENCE
    of statistics:  resample (costs_aware, costs_single) JOINTLY (same
    bootstrap indices to preserve pairing), compute stat on each resample,
    and test if the distribution of delta = stat_aware - stat_single is
    centred at 0.  This is a paired bootstrap test for any statistic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration.data_loader import load_trades, compute_mid_prices
from calibration.impact_estimator import calibrated_params
from extensions.regime import fit_hmm, regime_aware_params
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution
from montecarlo.cost_analysis import paired_strategy_test, _compute_statistic

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = sorted(DATA_DIR.glob("BTCUSDT-aggTrades-2026-*.csv"))

N_PATHS = 10_000
SEED = 42
N_REGIMES = 2
BOOTSTRAP_REPS = 5_000

# Execution parameters (same as V3)
X0 = 10.0
T = 1.0/(365.25*24)   # 1-hour horizon
N_STEPS = 250
LAM = 1e-6


# ─── Bootstrap helpers ────────────────────────────────────────────────────────

def _var_95(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 95))


def _cvar_95(arr: np.ndarray) -> float:
    v = np.percentile(arr, 95)
    tail = arr[arr >= v]
    return float(np.mean(tail)) if len(tail) > 0 else float(v)


def _objective(arr: np.ndarray, lam: float = LAM) -> float:
    """E[cost] + lam * Var[cost]  — the AC objective."""
    return float(np.mean(arr) + lam * np.var(arr))


def paired_bootstrap_stat_test(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    stat_fn,
    n_bootstrap: int = 5_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap test for a scalar population statistic.

    Jointly resamples both arrays (preserving pairing), computes the
    statistic on each resample, and tests H_0: stat(A) = stat(B).

    Returns
    -------
    stat_a : float   — point estimate on full sample
    stat_b : float   — point estimate on full sample
    p_value : float  — two-sided bootstrap p-value
    """
    assert costs_a.shape == costs_b.shape
    n = len(costs_a)
    rng = np.random.default_rng(seed)

    # Point estimates on full samples
    stat_a = stat_fn(costs_a)
    stat_b = stat_fn(costs_b)
    observed_diff = stat_a - stat_b

    # Bootstrap: resample JOINTLY (same indices) to preserve pairing
    # Vectorised: shape (n_bootstrap, n)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    boot_a = costs_a[idx]   # (n_bootstrap, n)
    boot_b = costs_b[idx]   # (n_bootstrap, n)

    # Compute stat on each bootstrap resample
    boot_diff = np.array([
        stat_fn(boot_a[b]) - stat_fn(boot_b[b])
        for b in range(n_bootstrap)
    ])

    # Shift null: centre boot_diff at 0 (null world)
    boot_diff_shifted = boot_diff - np.mean(boot_diff)
    p_value = float(np.mean(np.abs(boot_diff_shifted) >= abs(observed_diff)))

    return stat_a, stat_b, p_value


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 72)
    print("Paired test V4: Regime-Aware vs Single-Regime — RISK-SIDE METRICS")
    print("=" * 72)

    if not DATA_FILES:
        raise FileNotFoundError(
            f"No BTCUSDT-aggTrades-2026-*.csv files found in {DATA_DIR}."
        )

    # ── Step 1: load data ─────────────────────────────────────────────────
    print(f"\n[1/6] Loading {len(DATA_FILES)} CSV files (full 98-day window)...")
    trades = load_trades(
        DATA_DIR,
        start="2026-01-01",
        end="2026-04-08",
    )
    print(f"      {len(trades):,} trades  "
          f"{trades['timestamp'].iloc[0].date()} → "
          f"{trades['timestamp'].iloc[-1].date()}")

    # ── Step 2: calibrate base (pooled) params ────────────────────────────
    print("[2/6] Calibrating base (pooled) params...")
    cal = calibrated_params(
        trades_path=str(DATA_FILES[0]),
        X0=X0, T=T, N=N_STEPS, lam=LAM,
    )
    base_params = cal.params
    print(f"      S0={base_params.S0:.2f}  sigma={base_params.sigma:.6f}  "
          f"gamma={base_params.gamma:.4e}  eta={base_params.eta:.4e}")

    # ── Step 3: compute 5-min log returns, fit HMM ───────────────────────
    print("[3/6] Computing 5-min returns & fitting 2-state Gaussian HMM...")
    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    returns = mid["log_return"].to_numpy()
    returns = returns[np.isfinite(returns)]
    print(f"      Returns series: {len(returns):,} observations")

    regimes, state_seq = fit_hmm(returns, n_regimes=N_REGIMES)
    reg_dict = {r.label: r for r in regimes}
    risk_on  = reg_dict["risk_on"]
    risk_off = reg_dict["risk_off"]

    sigma_ratio = risk_off.sigma / risk_on.sigma
    print(f"      risk_on  sigma={risk_on.sigma:.6f}  prob={risk_on.probability:.4f}")
    print(f"      risk_off sigma={risk_off.sigma:.6f}  prob={risk_off.probability:.4f}")
    print(f"      sigma ratio (off/on): {sigma_ratio:.2f}x")

    # ── Step 4: build per-regime ACParams ─────────────────────────────────
    print("[4/6] Building per-regime ACParams...")
    params_riskon  = regime_aware_params(base_params, risk_on)
    params_riskoff = regime_aware_params(base_params, risk_off)

    # ── Step 5: simulate on common random Z ──────────────────────────────
    print(f"[5/6] Simulating {N_PATHS:,} paths (common random numbers)...")
    rng = np.random.default_rng(SEED)
    Z_shared = rng.standard_normal((N_PATHS, N_STEPS))

    x_twap_base    = twap_trajectory(base_params)
    x_twap_riskon  = twap_trajectory(params_riskon)
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

    # Weighted composite (stationary-probability-weighted average)
    w_on  = risk_on.probability
    w_off = risk_off.probability
    total_w = w_on + w_off
    w_on  /= total_w
    w_off /= total_w
    costs_aware = w_on * costs_riskon + w_off * costs_riskoff

    print(f"      mean_single  = {np.mean(costs_single):.4f}")
    print(f"      mean_aware   = {np.mean(costs_aware):.4f}")

    # ── Step 6: run all four tests ────────────────────────────────────────
    print(f"\n[6/6] Running paired tests (bootstrap reps={BOOTSTRAP_REPS:,})...")

    # (a) Mean cost — use existing paired_strategy_test (V3 reproduction)
    print("  (a) Mean cost test (V3 reproduction)...")
    result_mean = paired_strategy_test(
        costs_a=costs_aware,
        costs_b=costs_single,
        label_a="RegimeAware",
        label_b="SingleRegime",
        test="both",
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED,
    )
    mean_aware  = float(np.mean(costs_aware))
    mean_single = float(np.mean(costs_single))
    mean_diff   = result_mean.mean_diff
    # Use bootstrap p-value as primary (distribution-free, robust)
    p_mean      = result_mean.bootstrap_pvalue

    # (b) Objective = E[cost] + LAM * Var[cost]
    print("  (b) Objective test (bootstrap on E[cost]+λ·Var[cost])...")
    obj_aware_full  = _objective(costs_aware,  lam=LAM)
    obj_single_full = _objective(costs_single, lam=LAM)
    obj_aware_bs, obj_single_bs, p_obj = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=lambda c: _objective(c, lam=LAM),
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + 1,
    )

    # (c) VaR_95
    print("  (c) VaR_95 test (bootstrap)...")
    var_aware_full, var_single_full, p_var = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=_var_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + 2,
    )

    # (d) CVaR_95
    print("  (d) CVaR_95 test (bootstrap)...")
    cvar_aware_full, cvar_single_full, p_cvar = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=_cvar_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + 3,
    )

    # ── Print results table ───────────────────────────────────────────────
    ALPHA = 0.05

    def sig_label(p): return "YES *" if p < ALPHA else "no"

    print()
    print("=" * 90)
    print("RESULTS — Regime-Aware vs Single-Regime (Pooled)")
    print("=" * 90)
    print(f"  N paths = {N_PATHS:,}   bootstrap reps = {BOOTSTRAP_REPS:,}   lambda = {LAM:.1e}")
    print()
    header = (
        f"{'Metric':<18} {'mean(aware)':>14} {'mean(single)':>14} "
        f"{'diff(A-B)':>12} {'p-value':>10} {'sig@0.05':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = [
        ("mean cost",  mean_aware,       mean_single,      mean_diff,
         p_mean,  sig_label(p_mean)),
        ("objective",  obj_aware_full,   obj_single_full,  obj_aware_full  - obj_single_full,
         p_obj,   sig_label(p_obj)),
        ("VaR_95",     var_aware_full,   var_single_full,  var_aware_full  - var_single_full,
         p_var,   sig_label(p_var)),
        ("CVaR_95",    cvar_aware_full,  cvar_single_full, cvar_aware_full - cvar_single_full,
         p_cvar,  sig_label(p_cvar)),
    ]

    for metric, est_a, est_b, diff, pval, sl in rows:
        print(
            f"  {metric:<16} {est_a:>14.4f} {est_b:>14.4f} "
            f"{diff:>+12.4f} {pval:>10.4f} {sl:>10}"
        )

    print()
    print("  Regime parameters:")
    print(f"    risk_on  sigma={risk_on.sigma:.6f}  prob={risk_on.probability:.4f}")
    print(f"    risk_off sigma={risk_off.sigma:.6f}  prob={risk_off.probability:.4f}")
    print(f"    sigma ratio (off/on): {sigma_ratio:.2f}x")
    print()

    # ── Interpretation ────────────────────────────────────────────────────
    significant_metrics = [m for m, _, _, _, p, _ in rows if p < ALPHA]

    print("INTERPRETATION:")
    if not significant_metrics:
        print(
            "  None of the four metrics reject H_0 at alpha=0.05.\n"
            "\n"
            "  Part E narrative: The regime-aware composite (stationary-probability-\n"
            "  weighted average of per-regime params) is statistically indistinguishable\n"
            "  from the single-regime pooled calibration on ALL metrics — mean cost,\n"
            "  AC objective, VaR_95, and CVaR_95.  Two reasons explain this:\n"
            "\n"
            "  1. WEIGHTING DILUTION: The risk_off regime (high-sigma, where regime-\n"
            "     awareness helps most) has low stationary probability (~10%). The\n"
            "     weighted composite reflects 90% risk_on behaviour, masking the\n"
            "     tail benefit.\n"
            "\n"
            "  2. COMPOSITE CONSTRUCTION: The weighted-average cost conflates what\n"
            "     regime-awareness IS (knowing which regime you're in NOW) with a\n"
            "     long-run average.  True regime-aware benefit is conditional: GIVEN\n"
            "     risk_off, the regime-aware strategy saves cost vs pooled.  That\n"
            "     conditional benefit is invisible in an unconditional test.\n"
            "\n"
            "  Recommended narrative: Cite these null results honestly. Then show the\n"
            "  CONDITIONAL analysis: restrict paths to risk_off realizations and re-run\n"
            "  the comparison — that is where regime-awareness has its largest effect."
        )
    else:
        print(f"  Metrics that reject H_0: {', '.join(significant_metrics)}")
        for metric, est_a, est_b, diff, pval, sl in rows:
            if pval < ALPHA:
                direction = "LOWER" if diff < 0 else "HIGHER"
                print(
                    f"  {metric}: regime-aware is {direction} "
                    f"({est_a:.4f} vs {est_b:.4f}, diff={diff:+.4f}, p={pval:.4f})"
                )
        print()
        if any(pval < ALPHA and diff < 0 for _, _, _, diff, pval, _ in rows):
            print(
                "  Part E narrative: Regime-awareness provides statistically significant\n"
                "  REDUCTION in risk-side metrics. The benefit is real but concentrated\n"
                "  in the tail (VaR/CVaR) rather than mean cost — consistent with the\n"
                "  theory that regime detection helps most when sigma is elevated."
            )

    print("=" * 90)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out = {
        "version": "v4",
        "n_paths": N_PATHS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "lambda": LAM,
        "regime_sigmas": {
            "risk_on":  float(risk_on.sigma),
            "risk_off": float(risk_off.sigma),
            "ratio":    float(sigma_ratio),
        },
        "stationary_probs": {
            "risk_on":  float(risk_on.probability),
            "risk_off": float(risk_off.probability),
        },
        "results": {
            "mean_cost": {
                "aware":   float(mean_aware),
                "single":  float(mean_single),
                "diff":    float(mean_diff),
                "p_value": float(p_mean),
                "significant": bool(p_mean < ALPHA),
                "t_statistic": float(result_mean.t_statistic) if result_mean.t_statistic is not None else None,
                "t_pvalue":    float(result_mean.t_pvalue)    if result_mean.t_pvalue    is not None else None,
            },
            "objective": {
                "aware":   float(obj_aware_full),
                "single":  float(obj_single_full),
                "diff":    float(obj_aware_full - obj_single_full),
                "p_value": float(p_obj),
                "significant": bool(p_obj < ALPHA),
            },
            "var_95": {
                "aware":   float(var_aware_full),
                "single":  float(var_single_full),
                "diff":    float(var_aware_full - var_single_full),
                "p_value": float(p_var),
                "significant": bool(p_var < ALPHA),
            },
            "cvar_95": {
                "aware":   float(cvar_aware_full),
                "single":  float(cvar_single_full),
                "diff":    float(cvar_aware_full - cvar_single_full),
                "p_value": float(p_cvar),
                "significant": bool(p_cvar < ALPHA),
            },
        },
        "weights_used": {
            "risk_on":  float(w_on),
            "risk_off": float(w_off),
        },
    }

    out_path = DATA_DIR / "paired_regime_aware_v2_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
