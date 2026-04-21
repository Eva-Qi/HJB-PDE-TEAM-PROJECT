"""Paired statistical test V5: TRUE per-regime calibrated params vs Yuhao multipliers.

V4 used Yuhao-style sigma-based multipliers to build regime-aware ACParams.
Sonnet C (commit bc7a47d) found those multipliers are 41-469% off on gamma
and 79-90% off on eta vs the TRUE per-regime calibration stored in
data/regime_conditional_impact.json.

V5 compares:
  - V4 baseline: Yuhao-multiplier regime params (regime_aware_params from extensions/regime.py)
  - V5 new:      TRUE per-regime calibrated params from calibrated_params_per_regime
                 (stored in data/regime_conditional_impact.json, 2-state keys)

Both use the SAME simulation infrastructure, base ACParams, and shared Z matrix
(common random numbers, seed=42, N_PATHS=10_000) so that the only thing that
changes between V4 and V5 is the regime-specific gamma/eta/alpha/sigma values.

Risk_off caveat: eta and alpha for risk_off fell back to literature values
(eta=1e-3, alpha=0.6) because the impact regression had r²=0.10 and
alpha=0.261 (out-of-window). This is documented below and recorded in JSON.

Question: Does the -14% CVaR finding (V4) hold / strengthen / weaken / flip
when we use TRUE per-regime params?
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
from montecarlo.cost_analysis import paired_strategy_test
from shared.params import ACParams

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = sorted(DATA_DIR.glob("BTCUSDT-aggTrades-2026-*.csv"))

N_PATHS = 10_000
SEED = 42
N_REGIMES = 2
BOOTSTRAP_REPS = 5_000
ALPHA_STAT = 0.05

# Execution parameters (identical to V4 / V3 / V2 / V1)
X0 = 10.0
T = 1.0 / 24   # 1-hour horizon
N_STEPS = 50
LAM = 1e-6


# ─── Bootstrap helpers (copied from V4 to keep V5 self-contained) ─────────────

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

    Returns stat_a, stat_b, p_value (two-sided).
    """
    assert costs_a.shape == costs_b.shape
    n = len(costs_a)
    rng = np.random.default_rng(seed)

    stat_a = stat_fn(costs_a)
    stat_b = stat_fn(costs_b)
    observed_diff = stat_a - stat_b

    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_a = costs_a[idx]
    boot_b = costs_b[idx]

    boot_diff = np.array([
        stat_fn(boot_a[b]) - stat_fn(boot_b[b])
        for b in range(n_bootstrap)
    ])

    # Null-shift (centred bootstrap)
    boot_diff_shifted = boot_diff - np.mean(boot_diff)
    p_value = float(np.mean(np.abs(boot_diff_shifted) >= abs(observed_diff)))
    return stat_a, stat_b, p_value


# ─── Run all four tests for a given (costs_aware, costs_single) pair ──────────

def _run_four_tests(
    costs_aware: np.ndarray,
    costs_single: np.ndarray,
    label: str,
    seed_offset: int = 0,
) -> dict:
    """Run the four paired tests and return a results dict."""
    # (a) Mean cost — use existing paired_strategy_test
    result_mean = paired_strategy_test(
        costs_a=costs_aware,
        costs_b=costs_single,
        label_a="RegimeAware",
        label_b="SingleRegime",
        test="both",
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + seed_offset,
    )
    mean_aware  = float(np.mean(costs_aware))
    mean_single = float(np.mean(costs_single))
    p_mean      = result_mean.bootstrap_pvalue
    mean_diff   = result_mean.mean_diff

    # (b) Objective
    obj_aware  = _objective(costs_aware)
    obj_single = _objective(costs_single)
    obj_a_bs, obj_b_bs, p_obj = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=lambda c: _objective(c, lam=LAM),
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + seed_offset + 1,
    )

    # (c) VaR_95
    var_aware, var_single, p_var = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=_var_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + seed_offset + 2,
    )

    # (d) CVaR_95
    cvar_aware, cvar_single, p_cvar = paired_bootstrap_stat_test(
        costs_aware, costs_single,
        stat_fn=_cvar_95,
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED + seed_offset + 3,
    )

    return {
        "mean_cost": {
            "aware":   mean_aware,
            "single":  mean_single,
            "diff":    float(mean_diff),
            "pct_diff": float(100 * mean_diff / mean_single) if mean_single != 0 else None,
            "p_value": float(p_mean),
            "significant": bool(p_mean < ALPHA_STAT),
        },
        "objective": {
            "aware":   obj_aware,
            "single":  obj_single,
            "diff":    float(obj_aware - obj_single),
            "pct_diff": float(100 * (obj_aware - obj_single) / obj_single) if obj_single != 0 else None,
            "p_value": float(p_obj),
            "significant": bool(p_obj < ALPHA_STAT),
        },
        "var_95": {
            "aware":   var_aware,
            "single":  var_single,
            "diff":    float(var_aware - var_single),
            "pct_diff": float(100 * (var_aware - var_single) / var_single) if var_single != 0 else None,
            "p_value": float(p_var),
            "significant": bool(p_var < ALPHA_STAT),
        },
        "cvar_95": {
            "aware":   cvar_aware,
            "single":  cvar_single,
            "diff":    float(cvar_aware - cvar_single),
            "pct_diff": float(100 * (cvar_aware - cvar_single) / cvar_single) if cvar_single != 0 else None,
            "p_value": float(p_cvar),
            "significant": bool(p_cvar < ALPHA_STAT),
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 80)
    print("Paired test V5: TRUE per-regime params vs Yuhao multipliers")
    print("=" * 80)

    if not DATA_FILES:
        raise FileNotFoundError(
            f"No BTCUSDT-aggTrades-2026-*.csv files found in {DATA_DIR}."
        )

    # ── Step 1: load per-regime JSON (true calibrated params) ────────────
    print("\n[1/7] Loading true per-regime calibrated params from JSON...")
    json_path = DATA_DIR / "regime_conditional_impact.json"
    with open(json_path) as f:
        regime_json = json.load(f)

    two_state = regime_json["2_state"]
    base_json = two_state["base_params"]

    # Extract per-regime true params
    per_regime = {r["label"]: r for r in two_state["per_regime"]}
    riskon_true  = per_regime["risk_on"]
    riskoff_true = per_regime["risk_off"]

    # Document data limitation (risk_off fallback)
    riskoff_caveats = riskoff_true.get("warnings", [])
    riskoff_sources = riskoff_true.get("sources", {})
    riskoff_eta_source   = riskoff_sources.get("eta", "unknown")
    riskoff_alpha_source = riskoff_sources.get("alpha", "unknown")

    print(f"  risk_on  true_gamma={riskon_true['true_gamma']:.6f}  "
          f"true_eta={riskon_true['true_eta']:.6e}  "
          f"true_alpha={riskon_true['true_alpha']:.4f}  "
          f"true_sigma={riskon_true['true_sigma']:.6f}")
    print(f"  risk_off true_gamma={riskoff_true['true_gamma']:.6f}  "
          f"true_eta={riskoff_true['true_eta']:.6e}  "
          f"true_alpha={riskoff_true['true_alpha']:.4f}  "
          f"true_sigma={riskoff_true['true_sigma']:.6f}")
    if riskoff_eta_source == "fallback":
        print(f"  *** CAVEAT: risk_off eta/alpha are LITERATURE FALLBACK values "
              f"(eta=1e-3, alpha=0.6) — impact regression failed (r²=0.10, "
              f"alpha out-of-window). See warnings in JSON.")

    # ── Step 2: load trade data ───────────────────────────────────────────
    print(f"\n[2/7] Loading {len(DATA_FILES)} CSV files...")
    trades = load_trades(
        DATA_DIR,
        start="2026-01-01",
        end="2026-04-08",
    )
    print(f"      {len(trades):,} trades  "
          f"{trades['timestamp'].iloc[0].date()} → "
          f"{trades['timestamp'].iloc[-1].date()}")

    # ── Step 3: calibrate base (pooled) params ────────────────────────────
    print("[3/7] Calibrating base (pooled) params...")
    cal = calibrated_params(
        trades_path=str(DATA_FILES[0]),
        X0=X0, T=T, N=N_STEPS, lam=LAM,
    )
    base_params = cal.params
    print(f"      S0={base_params.S0:.2f}  sigma={base_params.sigma:.6f}  "
          f"gamma={base_params.gamma:.4e}  eta={base_params.eta:.4e}")

    # ── Step 4: fit HMM (for V4 Yuhao-multiplier baseline) ───────────────
    print("[4/7] Computing 5-min returns & fitting 2-state HMM (V4 baseline)...")
    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    returns = mid["log_return"].to_numpy()
    returns = returns[np.isfinite(returns)]
    print(f"      Returns series: {len(returns):,} observations")

    regimes, state_seq = fit_hmm(returns, n_regimes=N_REGIMES)
    reg_dict = {r.label: r for r in regimes}
    risk_on_hmm  = reg_dict["risk_on"]
    risk_off_hmm = reg_dict["risk_off"]

    sigma_ratio_hmm = risk_off_hmm.sigma / risk_on_hmm.sigma
    print(f"      risk_on  sigma_mult={risk_on_hmm.sigma:.4f}  "
          f"gamma_mult={risk_on_hmm.gamma:.4f}  "
          f"eta_mult={risk_on_hmm.eta:.4f}  "
          f"prob={risk_on_hmm.probability:.4f}")
    print(f"      risk_off sigma_mult={risk_off_hmm.sigma:.4f}  "
          f"gamma_mult={risk_off_hmm.gamma:.4f}  "
          f"eta_mult={risk_off_hmm.eta:.4f}  "
          f"prob={risk_off_hmm.probability:.4f}")

    # ── Step 5: build ACParams (V4 Yuhao mult, V5 true per-regime) ───────
    print("\n[5/7] Building per-regime ACParams (V4 Yuhao mult + V5 true)...")

    # V4: Yuhao multipliers (identical to V4 script)
    v4_params_riskon  = regime_aware_params(base_params, risk_on_hmm)
    v4_params_riskoff = regime_aware_params(base_params, risk_off_hmm)

    print(f"  V4 risk_on  sigma={v4_params_riskon.sigma:.6f}  "
          f"gamma={v4_params_riskon.gamma:.4e}  "
          f"eta={v4_params_riskon.eta:.4e}  "
          f"alpha={v4_params_riskon.alpha:.4f}")
    print(f"  V4 risk_off sigma={v4_params_riskoff.sigma:.6f}  "
          f"gamma={v4_params_riskoff.gamma:.4e}  "
          f"eta={v4_params_riskoff.eta:.4e}  "
          f"alpha={v4_params_riskoff.alpha:.4f}")

    # V5: TRUE per-regime calibrated values (absolute, not multipliers)
    v5_params_riskon = ACParams(
        S0=base_params.S0,
        sigma=riskon_true["true_sigma"],
        mu=0.0,
        X0=X0,
        T=T,
        N=N_STEPS,
        gamma=riskon_true["true_gamma"],
        eta=riskon_true["true_eta"],
        alpha=riskon_true["true_alpha"],
        lam=LAM,
        fee_bps=0.0,
    )
    v5_params_riskoff = ACParams(
        S0=base_params.S0,
        sigma=riskoff_true["true_sigma"],
        mu=0.0,
        X0=X0,
        T=T,
        N=N_STEPS,
        gamma=riskoff_true["true_gamma"],
        eta=riskoff_true["true_eta"],
        alpha=riskoff_true["true_alpha"],
        lam=LAM,
        fee_bps=0.0,
    )

    print(f"  V5 risk_on  sigma={v5_params_riskon.sigma:.6f}  "
          f"gamma={v5_params_riskon.gamma:.4e}  "
          f"eta={v5_params_riskon.eta:.4e}  "
          f"alpha={v5_params_riskon.alpha:.4f}")
    print(f"  V5 risk_off sigma={v5_params_riskoff.sigma:.6f}  "
          f"gamma={v5_params_riskoff.gamma:.4e}  "
          f"eta={v5_params_riskoff.eta:.4e}  "
          f"alpha={v5_params_riskoff.alpha:.4f}")

    # ── Step 6: simulate on common random Z ──────────────────────────────
    print(f"\n[6/7] Simulating {N_PATHS:,} paths (common random numbers, seed={SEED})...")
    rng = np.random.default_rng(SEED)
    Z_shared = rng.standard_normal((N_PATHS, N_STEPS))

    # Single-regime baseline (shared between V4 and V5)
    x_twap_base = twap_trajectory(base_params)
    _, costs_single = simulate_execution(
        base_params, x_twap_base,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )

    # V4: Yuhao-multiplier regime-aware composite
    x_twap_v4_riskon  = twap_trajectory(v4_params_riskon)
    x_twap_v4_riskoff = twap_trajectory(v4_params_riskoff)

    _, costs_v4_riskon = simulate_execution(
        v4_params_riskon, x_twap_v4_riskon,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )
    _, costs_v4_riskoff = simulate_execution(
        v4_params_riskoff, x_twap_v4_riskoff,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )

    # Stationary probability weights from HMM (same as V4)
    w_on  = risk_on_hmm.probability
    w_off = risk_off_hmm.probability
    total_w = w_on + w_off
    w_on  /= total_w
    w_off /= total_w

    costs_v4_aware = w_on * costs_v4_riskon + w_off * costs_v4_riskoff

    # V5: TRUE per-regime params composite (same weights)
    x_twap_v5_riskon  = twap_trajectory(v5_params_riskon)
    x_twap_v5_riskoff = twap_trajectory(v5_params_riskoff)

    _, costs_v5_riskon = simulate_execution(
        v5_params_riskon, x_twap_v5_riskon,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )

    # Check for NaN in V5 simulations
    v5_riskon_nan = int(np.sum(~np.isfinite(costs_v5_riskon)))
    if v5_riskon_nan > 0:
        print(f"  *** WARNING: V5 risk_on has {v5_riskon_nan} NaN paths!")

    _, costs_v5_riskoff = simulate_execution(
        v5_params_riskoff, x_twap_v5_riskoff,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )

    v5_riskoff_nan = int(np.sum(~np.isfinite(costs_v5_riskoff)))
    if v5_riskoff_nan > 0:
        print(f"  *** WARNING: V5 risk_off has {v5_riskoff_nan} NaN paths!")

    costs_v5_aware = w_on * costs_v5_riskon + w_off * costs_v5_riskoff

    v5_aware_nan = int(np.sum(~np.isfinite(costs_v5_aware)))
    if v5_aware_nan > 0:
        print(f"  *** WARNING: V5 aware composite has {v5_aware_nan} NaN paths! "
              f"Clamping NaN to prevent bootstrap crash.")
        # Replace NaN with the finite mean (conservative fallback)
        finite_mask = np.isfinite(costs_v5_aware)
        fallback = float(np.mean(costs_v5_aware[finite_mask])) if finite_mask.any() else 0.0
        costs_v5_aware = np.where(np.isfinite(costs_v5_aware), costs_v5_aware, fallback)

    print(f"  costs_single  mean={np.mean(costs_single):.4f}  "
          f"std={np.std(costs_single):.4f}")
    print(f"  costs_v4_aware mean={np.mean(costs_v4_aware):.4f}  "
          f"std={np.std(costs_v4_aware):.4f}")
    print(f"  costs_v5_aware mean={np.mean(costs_v5_aware):.4f}  "
          f"std={np.std(costs_v5_aware):.4f}")

    # ── Step 7: run all four tests for V4 and V5 ─────────────────────────
    print(f"\n[7/7] Running paired bootstrap tests (reps={BOOTSTRAP_REPS:,})...")
    print("  Running V4 tests (Yuhao mult)...")
    res_v4 = _run_four_tests(costs_v4_aware, costs_single,
                             label="V4", seed_offset=0)

    print("  Running V5 tests (true per-regime)...")
    res_v5 = _run_four_tests(costs_v5_aware, costs_single,
                             label="V5", seed_offset=100)

    # ── Print side-by-side comparison table ──────────────────────────────
    def _pstr(p: float) -> str:
        if p < 0.0001:
            return "<.0001"
        return f"{p:.4f}"

    def _sig(p: float) -> str:
        return "YES *" if p < ALPHA_STAT else "no"

    print()
    print("=" * 100)
    print("SIDE-BY-SIDE RESULTS: V4 (Yuhao mult) vs V5 (true per-regime)")
    print("=" * 100)
    print(f"  N_paths={N_PATHS:,}   bootstrap_reps={BOOTSTRAP_REPS:,}   lambda={LAM:.1e}")
    print()

    header = (
        f"{'Metric':<14} "
        f"{'V4 aware':>10} {'V4 single':>10} {'V4 diff':>9} {'V4 %diff':>8} {'V4 p':>8} {'V4 sig':>7}  "
        f"{'V5 aware':>10} {'V5 single':>10} {'V5 diff':>9} {'V5 %diff':>8} {'V5 p':>8} {'V5 sig':>7}"
    )
    print(header)
    print("-" * len(header))

    metric_keys = [
        ("mean_cost", "mean cost"),
        ("objective", "objective"),
        ("var_95",    "VaR_95"),
        ("cvar_95",   "CVaR_95"),
    ]

    for key, label in metric_keys:
        v4 = res_v4[key]
        v5 = res_v5[key]
        v4_pct = f"{v4['pct_diff']:+.1f}%" if v4["pct_diff"] is not None else "N/A"
        v5_pct = f"{v5['pct_diff']:+.1f}%" if v5["pct_diff"] is not None else "N/A"
        print(
            f"  {label:<14} "
            f"{v4['aware']:>10.2f} {v4['single']:>10.2f} {v4['diff']:>+9.2f} "
            f"{v4_pct:>8} {_pstr(v4['p_value']):>8} {_sig(v4['p_value']):>7}  "
            f"{v5['aware']:>10.2f} {v5['single']:>10.2f} {v5['diff']:>+9.2f} "
            f"{v5_pct:>8} {_pstr(v5['p_value']):>8} {_sig(v5['p_value']):>7}"
        )

    print()

    # ── Interpretation ────────────────────────────────────────────────────
    v4_cvar_pct = res_v4["cvar_95"]["pct_diff"]
    v5_cvar_pct = res_v5["cvar_95"]["pct_diff"]
    v4_cvar_p   = res_v4["cvar_95"]["p_value"]
    v5_cvar_p   = res_v5["cvar_95"]["p_value"]

    print("INTERPRETATION:")
    print(f"  V4 CVaR_95 diff: {v4_cvar_pct:+.1f}%  (p={_pstr(v4_cvar_p)})")
    print(f"  V5 CVaR_95 diff: {v5_cvar_pct:+.1f}%  (p={_pstr(v5_cvar_p)})")
    print()

    if v5_cvar_pct is not None:
        if abs(v5_cvar_pct) < 2.0 and v5_cvar_p >= ALPHA_STAT:
            print(
                "  VERDICT: V5 CVaR effect is NEGLIGIBLE / NON-SIGNIFICANT.\n"
                "  True per-regime params do NOT reproduce the V4 -14% CVaR finding.\n"
                "  Yuhao multipliers inflated the regime difference; the true params\n"
                "  are more similar to the single-regime pooled baseline."
            )
        elif v5_cvar_pct < 0 and abs(v5_cvar_pct) >= 10 and v5_cvar_p < ALPHA_STAT:
            if abs(v5_cvar_pct) > abs(v4_cvar_pct) * 1.15:
                print(
                    "  VERDICT: V5 CVaR reduction is STRONGER than V4.\n"
                    "  True per-regime params reveal even LARGER regime benefit;\n"
                    "  Yuhao multipliers UNDERSTATED the gain."
                )
            elif abs(v5_cvar_pct) < abs(v4_cvar_pct) * 0.85:
                print(
                    "  VERDICT: V5 CVaR reduction is WEAKER than V4 but still significant.\n"
                    "  True per-regime params show real benefit, but Yuhao multipliers\n"
                    "  OVERstated the magnitude. The -14% in V4 was inflated."
                )
            else:
                print(
                    "  VERDICT: V5 CVaR reduction is SIMILAR to V4 (within ±15%).\n"
                    "  Yuhao multiplier was 'good enough' — quantitative gamma/eta\n"
                    "  differences don't propagate materially to tail risk metrics."
                )
        elif v5_cvar_pct > 0:
            print(
                "  VERDICT: V5 CVaR FLIPPED DIRECTION (positive = worse).\n"
                "  True per-regime params INCREASE CVaR vs single regime.\n"
                "  The V4 -14% finding was an artifact of the multiplier approach."
            )
        else:
            print(
                "  VERDICT: V5 CVaR shows some reduction but is not significant (p≥0.05).\n"
                "  The V4 finding is WEAKENED with true params."
            )

    print()
    print("  Risk_off data limitation:")
    if riskoff_eta_source == "fallback":
        print(
            "  risk_off eta and alpha used LITERATURE FALLBACK (eta=1e-3, alpha=0.6).\n"
            "  The impact regression produced r²=0.10 and alpha=0.261 (out-of-window\n"
            "  bounds), indicating insufficient risk_off sub-sample trades for reliable\n"
            "  calibration. Fix: collect more data during risk_off periods, or use a\n"
            "  Bayesian prior that pulls toward the pooled calibration."
        )
    else:
        print(
            "  risk_off eta/alpha were successfully estimated from data (no fallback)."
        )

    print("=" * 100)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out = {
        "version": "v5",
        "description": "Paired test V5: TRUE per-regime calibrated params vs Yuhao multipliers",
        "n_paths": N_PATHS,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "lambda": LAM,
        "base_params": {
            "S0":    float(base_params.S0),
            "sigma": float(base_params.sigma),
            "gamma": float(base_params.gamma),
            "eta":   float(base_params.eta),
            "alpha": float(base_params.alpha),
        },
        "stationary_probs": {
            "risk_on":  float(w_on),
            "risk_off": float(w_off),
        },
        "v4_yuhao_params": {
            "risk_on": {
                "sigma": float(v4_params_riskon.sigma),
                "gamma": float(v4_params_riskon.gamma),
                "eta":   float(v4_params_riskon.eta),
                "alpha": float(v4_params_riskon.alpha),
            },
            "risk_off": {
                "sigma": float(v4_params_riskoff.sigma),
                "gamma": float(v4_params_riskoff.gamma),
                "eta":   float(v4_params_riskoff.eta),
                "alpha": float(v4_params_riskoff.alpha),
            },
        },
        "v5_true_params": {
            "risk_on": {
                "sigma": float(v5_params_riskon.sigma),
                "gamma": float(v5_params_riskon.gamma),
                "eta":   float(v5_params_riskon.eta),
                "alpha": float(v5_params_riskon.alpha),
                "source": "calibrated (aggregated_1min)",
            },
            "risk_off": {
                "sigma": float(v5_params_riskoff.sigma),
                "gamma": float(v5_params_riskoff.gamma),
                "eta":   float(v5_params_riskoff.eta),
                "alpha": float(v5_params_riskoff.alpha),
                "eta_source": riskoff_eta_source,
                "alpha_source": riskoff_alpha_source,
                "caveats": riskoff_caveats,
            },
        },
        "v4_results": res_v4,
        "v5_results": res_v5,
        "interpretation": {
            "v4_cvar_pct_diff": v4_cvar_pct,
            "v5_cvar_pct_diff": v5_cvar_pct,
            "v4_cvar_p": float(v4_cvar_p),
            "v5_cvar_p": float(v5_cvar_p),
        },
        "caveats": {
            "risk_off_eta_alpha_source": riskoff_eta_source,
            "risk_off_warnings": riskoff_caveats,
            "note": (
                "risk_off eta=1e-3, alpha=0.6 are literature fallback values. "
                "The impact regression failed for regime 1 (r²=0.10, alpha out-of-window). "
                "Sufficient risk_off trade data would be needed for true calibration."
                if riskoff_eta_source == "fallback" else
                "No fallback used — all params estimated from data."
            ),
        },
    }

    out_path = DATA_DIR / "paired_regime_v5_true_params.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
