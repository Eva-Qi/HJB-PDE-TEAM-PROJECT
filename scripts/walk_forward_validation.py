"""Walk-forward out-of-sample (OOS) validation for Almgren-Chriss optimal execution.

Rolling train/test windows over the 98-day BTCUSDT dataset:
  Split 1: train Jan 01-28  → test Jan 29 – Feb 25  (28d train, 28d test)
  Split 2: train Feb 01-28  → test Mar 01-28         (28d train, 28d test)
  Split 3: train Mar 01-28  → test Mar 29 – Apr 08   (28d train, ~11d test)
  Split 4: train Jan 01 – Mar 01 → test Mar 02 – Apr 08  (60d train, 38d test)

For each split:
  1. Calibrate ACParams on TRAIN window (window-specific, not full dataset).
  2. Compute AC-optimal trajectory and TWAP trajectory from train params.
  3. Evaluate both in TEST window:
     - Swap sigma → test sigma; keep gamma/eta/alpha from train calibration.
     - Simulate with MC (n_paths=5000).
  4. Report: train vs test sigma (regime drift), IS vs OOS % savings vs TWAP.

Output:
  - Printed table to stdout
  - data/walk_forward_results.json
  - plot_walk_forward.png (project root)

Run:
  python scripts/walk_forward_validation.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from calibration.data_loader import _load_single_csv, compute_ohlc  # noqa: F401 (compute_ohlc used indirectly)
from calibration.impact_estimator import (
    estimate_realized_vol_gk,
    estimate_kyle_lambda,
    estimate_temporary_impact_aggregated,
)
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory
from shared.cost_model import execution_cost
from shared.params import ACParams, almgren_chriss_closed_form


# ── run constants ─────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
N_PATHS  = 5000
X0       = 10.0                       # BTC to liquidate
T        = 1.0 / (365.25 * 24)        # 1-hour execution horizon (years)
N_STEPS  = 250
LAM      = 1e-6

# (label, train_start, train_end, test_start, test_end) — all inclusive
SPLITS = [
    ("Split-1 Jan→Feb",       "2026-01-01", "2026-01-28", "2026-01-29", "2026-02-25"),
    ("Split-2 Feb→Mar",       "2026-02-01", "2026-02-28", "2026-03-01", "2026-03-28"),
    ("Split-3 Mar→Apr",       "2026-03-01", "2026-03-28", "2026-03-29", "2026-04-08"),
    ("Split-4 JanFeb→MarApr", "2026-01-01", "2026-03-01", "2026-03-02", "2026-04-08"),
    # New splits using 2025 aggTrades backfill (91 days, Apr-Jun 2025)
    ("Split-5 Apr→May 2025",  "2025-04-01", "2025-04-28", "2025-04-29", "2025-05-27"),
    ("Split-6 May→Jun 2025",  "2025-05-01", "2025-05-28", "2025-05-29", "2025-06-27"),
]

# Literature fallbacks (used when estimation fails on short window)
# Units: gamma in $/BTC (Kyle's lambda), eta in $/BTC^alpha, alpha dimensionless.
# gamma ≈ 2.5 $/BTC is the BTCUSDT bar-level ballpark from calibrated_params()
# and regime_conditional_impact.json. Previous 1e-4 was a legacy 1/BTC-convention
# value that produced a permanent-impact term ~5 orders of magnitude too small.
FALLBACK_GAMMA = 2.5
FALLBACK_ETA   = 1e-3
FALLBACK_ALPHA = 0.6


def load_window_aggregated(start: str, end: str, ohlc_freq: str = "5min"):
    """Stream-aggregate a date window into OHLC bars and signed-flow buckets.

    Loads one day at a time and aggregates in place — never holds more than
    one day of raw tick data in memory at once.  Returns two DataFrames:
      ohlc_df  — resampled OHLC bars for sigma estimation
      flow_df  — 1-min signed-flow buckets for eta/alpha estimation
      last_price — float, last tick price in the window (for S0)
      n_ticks    — int, total tick count for logging

    This is memory-safe even for 60-day windows (max ~350 MB per day raw,
    but we release each day after aggregating).
    """
    start_date = pd.Timestamp(start).normalize()
    end_date   = pd.Timestamp(end).normalize()

    ohlc_list  = []
    flow_list  = []
    last_price = None
    n_ticks    = 0

    day = start_date
    while day <= end_date:
        fname = DATA_DIR / f"BTCUSDT-aggTrades-{day.strftime('%Y-%m-%d')}.csv"
        if not fname.exists():
            day += pd.Timedelta(days=1)
            continue

        df_day = _load_single_csv(fname)
        n_ticks += len(df_day)
        last_price = float(df_day["price"].iloc[-1])

        # OHLC aggregation
        df_day_idx = df_day.set_index("timestamp").sort_index()
        o = df_day_idx["price"].resample(ohlc_freq).ohlc()
        vol = df_day_idx["quantity"].resample(ohlc_freq).sum()
        o["volume"] = vol
        ohlc_list.append(o.dropna())

        # 1-min signed-flow buckets
        df_day_idx["signed_flow"] = df_day_idx["quantity"] * df_day_idx["side"]
        flow = df_day_idx.resample("1min").agg(
            net_flow=("signed_flow", "sum"),
            first_price=("price", "first"),
            last_price_b=("price", "last"),
            n_trades=("price", "count"),
        ).dropna()
        flow = flow[flow["n_trades"] > 0]
        flow["return"] = (flow["last_price_b"] - flow["first_price"]) / flow["first_price"]
        # Keep first_price/last_price_b so gamma regression can use dprice ($)
        # on net_flow (BTC) → gamma in $/BTC (the convention ACParams.gamma expects,
        # matching calibration/impact_estimator.estimate_kyle_lambda_aggregated).
        flow_list.append(flow[["net_flow", "return", "first_price", "last_price_b"]])

        del df_day, df_day_idx
        day += pd.Timedelta(days=1)

    if not ohlc_list:
        raise FileNotFoundError(f"No CSV files found for window {start} – {end}")

    ohlc_df = pd.concat(ohlc_list).sort_index()
    flow_df = pd.concat(flow_list).sort_index()

    return ohlc_df, flow_df, last_price, n_ticks


# ── calibration helpers ───────────────────────────────────────────────────────

def calibrate_from_aggregated(ohlc_df, flow_df, last_price: float) -> dict:
    """Calibrate AC params from pre-aggregated OHLC + flow DataFrames.

    Memory-safe: operates on aggregated data only (no raw tick DataFrame).
    Returns dict with keys: sigma, gamma, eta, alpha, S0, warnings, sources.

    gamma (Kyle's lambda) is estimated from 1-min bar returns vs net flow,
    which is the next-best approximation when we don't keep tick-level data.
    """
    from scipy import stats as scipy_stats

    w = []
    src = {}

    # sigma — Garman-Klass from 5-min OHLC
    sigma = estimate_realized_vol_gk(ohlc_df, freq_seconds=300.0, annualize=True)
    src["sigma"] = "estimated"

    # gamma — Kyle's lambda at 1-min bar resolution: Δprice ($) ≈ γ · net_flow (BTC)
    #
    # Units: dp_bars is in $ (last - first within bucket), nf_bars is in BTC,
    # so γ = dp/flow is in $/BTC. This MUST match ACParams.gamma convention,
    # which the cost model (shared/cost_model.execution_cost) uses as
    #   perm_cost = γ · Σ n_k · (Σ prior n_k)     [$/BTC · BTC · BTC = $]
    # and the SDE engine (montecarlo/sde_engine) uses as
    #   s_next = s_prev - γ · n_k                  [$/BTC · BTC = $ price drop].
    #
    # Equivalent to calibration/impact_estimator.estimate_kyle_lambda_aggregated
    # (Cov(Δprice, flow) / Var(flow) via OLS). Previous implementation regressed
    # `return` (dimensionless) on `net_flow` (BTC) → γ in 1/BTC (~1.9e-5),
    # five orders of magnitude too small, zeroing out the permanent-impact term
    # in the optimizer and collapsing AC-optimal onto TWAP. See N-1 in
    # AUDIT_VERIFICATION.md.
    try:
        flow_clean = flow_df.dropna()
        dp_bars = (flow_clean["last_price_b"] - flow_clean["first_price"]).values  # $ per bucket
        nf_bars = flow_clean["net_flow"].values                                     # BTC per bucket
        mask = np.isfinite(dp_bars) & np.isfinite(nf_bars) & (nf_bars != 0)
        if mask.sum() >= 10:
            # Kyle's lambda: regress Δprice ($) on net_flow (BTC) — OLS slope → $/BTC
            slope, intercept, r, p, se = scipy_stats.linregress(nf_bars[mask], dp_bars[mask])
            gamma_bar = float(slope)
            if gamma_bar > 0:
                gamma = gamma_bar
                src["gamma"] = "estimated_bar"
                w.append(f"[bar-level gamma] lambda={gamma:.4f} $/BTC  r²={r**2:.3f}")
            else:
                w.append(f"Kyle bar-level: negative gamma ({gamma_bar:.4f} $/BTC), using fallback")
                gamma = FALLBACK_GAMMA
                src["gamma"] = "fallback"
        else:
            w.append(f"Kyle bar-level: insufficient data ({mask.sum()} bars), using fallback")
            gamma = FALLBACK_GAMMA
            src["gamma"] = "fallback"
    except Exception as exc:
        w.append(f"Kyle bar-level failed: {exc}, using fallback")
        gamma = FALLBACK_GAMMA
        src["gamma"] = "fallback"

    # eta, alpha — from pre-computed 1-min flow buckets
    eta = alpha = None

    try:
        flow_clean = flow_df.dropna()
        abs_flow   = flow_clean["net_flow"].abs().values
        abs_ret    = flow_clean["return"].abs().values
        mask2 = (abs_flow > 0) & (abs_ret > 0) & np.isfinite(abs_flow) & np.isfinite(abs_ret)
        af, ar = abs_flow[mask2], abs_ret[mask2]
        if len(af) >= 20:
            from scipy import stats as scipy_stats2
            slope2, intercept2, r2, p2, se2 = scipy_stats2.linregress(np.log(af), np.log(ar))
            a_est = float(slope2)
            e_est = float(np.exp(intercept2))
            r2_val = float(r2**2)
            if 0.3 <= a_est <= 1.5 and r2_val >= 0.05:
                eta, alpha = e_est, a_est
                src["eta"] = src["alpha"] = "1min_flow"
                w.append(f"[1min_flow] alpha={a_est:.3f} r²={r2_val:.3f} n={len(af)}")
            else:
                w.append(f"[1min_flow] alpha={a_est:.3f} r²={r2_val:.3f} rejected")
        else:
            w.append(f"[1min_flow] insufficient buckets ({len(af)})")
    except Exception as exc:
        w.append(f"[1min_flow] failed: {exc}")

    if eta is None:
        eta, alpha = FALLBACK_ETA, FALLBACK_ALPHA
        src["eta"] = src["alpha"] = "fallback"
        w.append("temporary impact: all methods failed, using literature fallback")

    return dict(sigma=sigma, gamma=gamma, eta=eta, alpha=alpha, S0=last_price,
                warnings=w, sources=src)


def build_params(cal: dict, sigma_override: float | None = None,
                 s0_override: float | None = None,
                 force_linear: bool = False) -> ACParams:
    """Build ACParams from a calibration dict, optionally overriding sigma/S0.

    ``force_linear`` retained for backward-compat but should generally be False —
    non-linear alpha is handled via the HJB PDE solver downstream.
    """
    alpha = 1.0 if force_linear else cal["alpha"]
    return ACParams(
        S0=s0_override if s0_override is not None else cal["S0"],
        sigma=sigma_override if sigma_override is not None else cal["sigma"],
        mu=0.0,
        X0=X0, T=T, N=N_STEPS,
        gamma=cal["gamma"],
        eta=cal["eta"],
        alpha=alpha,
        lam=LAM,
        fee_bps=7.5,
    )


def compute_optimal_trajectory(params: ACParams) -> tuple[np.ndarray, str, str | None]:
    """Compute the AC optimal inventory trajectory.

    Uses closed-form when alpha is very close to 1 (|alpha-1| < 0.01); otherwise
    solves the HJB PDE via policy iteration (Howard's algorithm). Raising a
    ``ValueError`` from the closed-form path falls through to the PDE solver.

    Returns
    -------
    x_opt : np.ndarray, shape (N+1,)
        Optimal inventory trajectory.
    method : str
        "closed_form" or "hjb_pde".
    note : str | None
        Optional diagnostic (e.g., convergence warning) — currently always None
        but reserved for future use.
    """
    # Keep the cheap path when alpha is effectively 1 (back-compat & speed)
    if abs(params.alpha - 1.0) < 0.01:
        _, x_opt, _ = almgren_chriss_closed_form(params)
        return x_opt, "closed_form", None

    # Non-linear impact: HJB PDE solver handles arbitrary alpha
    try:
        grid, _V, v_star = solve_hjb(params, M=200)
        x_opt = extract_optimal_trajectory(grid, v_star, params)
    except ValueError:
        # Fallback: if the closed-form path is accidentally reached (|alpha-1|<1e-10
        # but not <0.01) we'd hit this; treat as bug and re-raise.
        raise

    # Validate monotonicity and boundary conditions
    # (x[0] == X0 by construction in extract_optimal_trajectory; x[-1] should be ~0)
    return x_opt, "hjb_pde", None


# ── per-split logic ──────────────────────────────────────────────────────────

def run_split(label: str, train_start: str, train_end: str,
              test_start: str, test_end: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"  Train: {train_start} – {train_end}   Test: {test_start} – {test_end}")
    print(f"{'='*62}")

    result = dict(label=label,
                  train_start=train_start, train_end=train_end,
                  test_start=test_start, test_end=test_end,
                  error=None)

    # ── stream-aggregate train window (memory-safe: no full tick DataFrame) ────
    try:
        tr_ohlc, tr_flow, tr_last_price, tr_n = load_window_aggregated(train_start, train_end)
        print(f"  Train window: {tr_n:,} ticks  →  {len(tr_ohlc):,} OHLC bars  {len(tr_flow):,} 1-min buckets")
    except Exception as e:
        result["error"] = f"Train load: {e}"
        print(f"  [ERROR] {result['error']}")
        return result

    # ── calibrate on TRAIN window ────────────────────────────────────────────
    try:
        train_cal = calibrate_from_aggregated(tr_ohlc, tr_flow, tr_last_price)
        del tr_ohlc, tr_flow      # free memory before loading test window
        for wm in train_cal["warnings"]:
            print(f"  [WARN] {wm}")
        print(f"  Train  sigma={train_cal['sigma']:.4f}  gamma={train_cal['gamma']:.2e}"
              f"  eta={train_cal['eta']:.2e}  alpha={train_cal['alpha']:.3f}")
    except Exception as e:
        result["error"] = f"Train calibration: {e}"
        print(f"  [ERROR] {result['error']}")
        return result

    # ── stream-aggregate test window ─────────────────────────────────────────
    try:
        te_ohlc, te_flow, te_first_price, te_n = load_window_aggregated(test_start, test_end)
        print(f"  Test  window: {te_n:,} ticks  →  {len(te_ohlc):,} OHLC bars  {len(te_flow):,} 1-min buckets")
        test_sigma  = estimate_realized_vol_gk(te_ohlc, freq_seconds=300.0, annualize=True)
        del te_ohlc, te_flow      # free memory
        sigma_drift = abs(test_sigma - train_cal["sigma"]) / train_cal["sigma"] * 100.0
        print(f"  Test   sigma={test_sigma:.4f}   sigma_drift={sigma_drift:.1f}%")
    except Exception as e:
        result["error"] = f"Test load/sigma: {e}"
        print(f"  [ERROR] {result['error']}")
        return result

    # ── build params ─────────────────────────────────────────────────────────
    # Keep the calibrated (possibly non-linear) alpha — HJB PDE handles it.
    train_params = build_params(train_cal, force_linear=False)
    test_params  = build_params(train_cal,
                                sigma_override=test_sigma,
                                s0_override=te_first_price,
                                force_linear=False)

    # ── trajectories from TRAIN params ───────────────────────────────────────
    # AC optimal: closed-form iff alpha ≈ 1; otherwise HJB PDE (Howard's policy
    # iteration).  This replaces the previous invalid "linear_approx" fallback
    # which forced alpha=1 and invoked the closed-form on non-linear data.
    method = None
    method_note = None
    try:
        x_twap = twap_trajectory(train_params)
        x_opt, method, method_note = compute_optimal_trajectory(train_params)
        print(f"  AC trajectory method: {method}  (alpha={train_params.alpha:.3f})")

        # Sanity checks on the trajectory
        if x_opt[0] <= 0 or abs(x_opt[0] - X0) > 1e-6:
            raise ValueError(f"x_opt[0]={x_opt[0]:.4f} != X0={X0}")
        if not np.all(np.diff(x_opt) <= 1e-9):
            n_bad = int(np.sum(np.diff(x_opt) > 1e-9))
            print(f"  [WARN] x_opt non-monotone at {n_bad} steps")
        if x_opt[-1] > X0 * 1e-3:
            print(f"  [WARN] x_opt[-1]={x_opt[-1]:.6f} (expected ≈ 0); "
                  f"HJB residual — trajectory still used")
    except Exception as e:
        result["error"] = f"Trajectory ({method or 'unknown'}): {e}"
        print(f"  [ERROR] {result['error']}")
        return result

    # ── in-sample deterministic costs ────────────────────────────────────────
    # Both trajectories fixed; only impact params matter (sigma ∉ execution_cost).
    is_cost_opt  = execution_cost(x_opt,  train_params)
    is_cost_twap = execution_cost(x_twap, train_params)
    is_sav = (is_cost_twap - is_cost_opt) / abs(is_cost_twap) * 100.0 if is_cost_twap else 0.0
    print(f"  IS    cost_AC={is_cost_opt:.4f}  cost_TWAP={is_cost_twap:.4f}  savings={is_sav:.2f}%")

    # ── OOS deterministic costs (train trajectory + test impact/sigma params) ──
    # Primary OOS metric: evaluate same trajectories under TEST-period parameters.
    # This isolates whether the trained strategy still beats TWAP when sigma and S0
    # have shifted, without the MC shortfall S0-drift contamination.
    oos_det_opt  = execution_cost(x_opt,  test_params)
    oos_det_twap = execution_cost(x_twap, test_params)
    oos_det_sav  = ((oos_det_twap - oos_det_opt) / abs(oos_det_twap) * 100.0
                    if oos_det_twap else 0.0)
    print(f"  OOS(det) cost_AC={oos_det_opt:.4f}  cost_TWAP={oos_det_twap:.4f}  savings={oos_det_sav:.2f}%")

    # ── OOS MC simulation (for variance / confidence interval) ──────────────
    # MC uses same test_params. Costs here are implementation shortfall under
    # GBM paths; the mean should approximate oos_det_* above within MC noise.
    #
    # CRN (common random numbers): generate Z once and pass via Z_extern to BOTH
    # simulate_execution calls. Implicit seed-sharing (passing seed=42 to each call
    # and relying on internal rng init to produce identical Z) is fragile —
    # any future refactor that adds an rng call before Z generation breaks
    # CRN silently and inflates mean_diff variance.
    try:
        rng_oos = np.random.default_rng(42)
        n_half = N_PATHS // 2
        Z_half = rng_oos.standard_normal((n_half, test_params.N))
        Z_oos = np.vstack([Z_half, -Z_half])  # antithetic pairs, total = N_PATHS
        _, costs_opt_oos  = simulate_execution(test_params, x_opt,  n_paths=N_PATHS, Z_extern=Z_oos)
        _, costs_twap_oos = simulate_execution(test_params, x_twap, n_paths=N_PATHS, Z_extern=Z_oos)

        oos_mc_opt  = float(np.mean(costs_opt_oos))
        oos_mc_twap = float(np.mean(costs_twap_oos))
        oos_mc_sav  = ((oos_mc_twap - oos_mc_opt) / abs(oos_mc_twap) * 100.0
                       if oos_mc_twap else 0.0)
        oos_mc_opt_std  = float(np.std(costs_opt_oos) / np.sqrt(len(costs_opt_oos)))
        oos_mc_twap_std = float(np.std(costs_twap_oos) / np.sqrt(len(costs_twap_oos)))
        print(f"  OOS(MC)  cost_AC={oos_mc_opt:.4f}±{oos_mc_opt_std:.4f}  "
              f"cost_TWAP={oos_mc_twap:.4f}±{oos_mc_twap_std:.4f}  savings={oos_mc_sav:.2f}%")
        mc_ok = True
    except Exception as e:
        oos_mc_opt = oos_mc_twap = oos_mc_sav = oos_mc_opt_std = oos_mc_twap_std = None
        mc_ok = False
        print(f"  [WARN] MC simulation failed: {e}")

    print(f"  Degradation (det): {is_sav:.2f}% → {oos_det_sav:.2f}%  (Δ={is_sav - oos_det_sav:+.2f} pp)")

    result.update(dict(
        train_sigma=round(train_cal["sigma"], 6),
        test_sigma=round(test_sigma, 6),
        sigma_drift_pct=round(sigma_drift, 2),
        gamma=train_cal["gamma"],
        eta=train_cal["eta"],
        alpha=train_cal["alpha"],
        # Trajectory method used — "closed_form" iff |alpha-1|<0.01, else "hjb_pde".
        # Replaces the previous `used_linear_approx` flag which was always True
        # and reflected a broken fallback (invalid closed-form on non-linear data).
        ac_method=method,
        ac_method_note=method_note,
        used_linear_approx=False,  # retained for backward-compat; always False now
        # in-sample
        is_cost_opt=round(is_cost_opt, 6),
        is_cost_twap=round(is_cost_twap, 6),
        is_savings_pct=round(is_sav, 4),
        # OOS deterministic (primary metric)
        oos_det_cost_opt=round(oos_det_opt, 6),
        oos_det_cost_twap=round(oos_det_twap, 6),
        oos_det_savings_pct=round(oos_det_sav, 4),
        savings_degradation_pp=round(is_sav - oos_det_sav, 4),
        # OOS MC (secondary — for confidence)
        oos_mc_cost_opt=round(oos_mc_opt, 6) if mc_ok else None,
        oos_mc_cost_twap=round(oos_mc_twap, 6) if mc_ok else None,
        oos_mc_savings_pct=round(oos_mc_sav, 4) if mc_ok else None,
    ))
    return result


# ── output helpers ────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    hdr = (f"{'Split':<28}  {'α':>6}  {'method':>10}  {'σ_train':>8}  {'σ_test':>8}  {'σ_drift':>8}"
           f"  {'IS%sav':>8}  {'OOS%sav':>8}  {'Δpp':>8}")
    sep = "─" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in results:
        if r.get("error"):
            print(f"  {'  '+r['label']:<26}  [FAILED: {r['error'][:55]}]")
            continue
        print(
            f"  {r['label']:<26}"
            f"  {r['alpha']:>6.3f}"
            f"  {r.get('ac_method', '?'):>10}"
            f"  {r['train_sigma']:>8.4f}"
            f"  {r['test_sigma']:>8.4f}"
            f"  {r['sigma_drift_pct']:>7.1f}%"
            f"  {r['is_savings_pct']:>7.2f}%"
            f"  {r['oos_det_savings_pct']:>7.2f}%"
            f"  {r['savings_degradation_pp']:>+7.2f}"
        )
    print(sep)
    print("α = calibrated temporary-impact exponent")
    print("method = AC trajectory solver (closed_form iff |α-1|<0.01, else hjb_pde)")
    print("σ_train/σ_test = GK annualized vol | σ_drift = |Δσ|/σ_train")
    print("IS%sav = deterministic savings vs TWAP on train params")
    print("OOS%sav = deterministic savings when test-period sigma/S0 applied")
    print("Δpp = IS - OOS (positive = degradation out-of-sample)\n")


def plot_results(results: list[dict], out_path: str) -> None:
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print("[WARN] No valid results to plot.")
        return

    labels      = [r["label"].replace(" ", "\n") for r in valid]
    x           = np.arange(len(valid))
    is_sav      = [r["is_savings_pct"]      for r in valid]
    oos_sav     = [r["oos_det_savings_pct"] for r in valid]
    train_sig   = [r["train_sigma"]     for r in valid]
    test_sig    = [r["test_sigma"]      for r in valid]

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 1, hspace=0.50)

    # Panel 1 — IS vs OOS savings
    ax1   = fig.add_subplot(gs[0])
    width = 0.35
    b_is  = ax1.bar(x - width/2, is_sav,  width, label="In-Sample (train, deterministic)",
                    color="#2E86AB", alpha=0.88, edgecolor="white")
    b_oos = ax1.bar(x + width/2, oos_sav, width, label="Out-of-Sample (test, deterministic)",
                    color="#E84855", alpha=0.88, edgecolor="white")

    for bar in b_is:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                 f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color="#1a5f7a")
    for bar in b_oos:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.03,
                 f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color="#a0001e")

    ax1.set_title("Walk-Forward Validation — AC Optimal vs TWAP % Savings",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("% Cost Savings vs TWAP")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2 — sigma regime drift
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, train_sig, "o-", label="Train σ (GK annualized)",
             color="#2E86AB", lw=2, ms=7)
    ax2.plot(x, test_sig,  "s--", label="Test σ (GK annualized)",
             color="#E84855", lw=2, ms=7)

    for i, (ts, es) in enumerate(zip(train_sig, test_sig)):
        ax2.annotate("", xy=(i, es), xytext=(i, ts),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.3))
        mid   = (ts + es) / 2
        drift = abs(es - ts) / ts * 100
        ax2.text(i + 0.07, mid, f"{drift:.1f}%", fontsize=8, color="gray", va="center")

    ax2.set_title("Sigma Regime Drift — Train vs Test Window", fontsize=12,
                  fontweight="bold", pad=10)
    ax2.set_ylabel("Annualized Volatility (GK)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("MF796 Walk-Forward OOS Validation")
    print(f"X0={X0} BTC | T=1h | N={N_STEPS} | lam={LAM:.0e} | n_paths={N_PATHS}")

    results = []
    for label, tr_s, tr_e, te_s, te_e in SPLITS:
        try:
            r = run_split(label, tr_s, tr_e, te_s, te_e)
        except Exception as exc:
            r = dict(label=label,
                     train_start=tr_s, train_end=tr_e,
                     test_start=te_s,  test_end=te_e,
                     error=f"Unhandled: {exc}")
            print(f"  [UNHANDLED] {r['error']}")
        results.append(r)

    print_table(results)

    out_json = PROJECT_ROOT / "data" / "walk_forward_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved → {out_json}")

    plot_results(results, str(PROJECT_ROOT / "plot_walk_forward.png"))


if __name__ == "__main__":
    main()
