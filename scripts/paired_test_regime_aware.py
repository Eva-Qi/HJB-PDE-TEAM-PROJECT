"""Paired statistical test: regime-aware vs single-regime (pooled) execution.

Tests whether HMM-based regime detection produces statistically different
MC realized cost from using single pooled-calibrated parameters on the SAME
paths (common random numbers).

Design choice — regime_aware strategy:
    We compute execution costs separately under ACParams for each regime
    (risk_on, risk_off), then take a STATIONARY-PROBABILITY-WEIGHTED average
    as the "regime_aware" cost per path. Rationale: in steady state, a trader
    using regime-aware params faces risk_on conditions ~P(risk_on) of the time
    and risk_off conditions ~P(risk_off) of the time. The weighted average is
    the expected cost under the long-run mixing distribution — a conservative,
    theory-justified composite. Both simulations run on the SAME Z-paths
    (common random numbers) so per-path differences are clean paired
    observations with minimal noise from MC sampling.

Null hypothesis: E[C_regime_aware - C_single] = 0
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration.data_loader import load_trades, compute_ohlc, compute_mid_prices
from calibration.impact_estimator import calibrated_params
from extensions.regime import fit_hmm, regime_aware_params
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution
from montecarlo.cost_analysis import paired_strategy_test

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Use ALL 98 days (Jan-Apr 2026). Earlier version used 7 days and found
# weak regime separation (~8% sigma spread) → paired test p=0.84. With
# Yuhao's sigma*1e-8 fix + full data, univariate HMM finds 250% spread
# (see scripts/compare_hmm_features.py output) so p should now reject.
DATA_FILES = sorted(DATA_DIR.glob("BTCUSDT-aggTrades-2026-*.csv"))

N_PATHS = 10_000
SEED = 42
N_REGIMES = 2
BOOTSTRAP_REPS = 5_000

# Execution parameters matching the project's calibrated setup
X0 = 10.0          # 10 BTC
T = 1.0 / 24      # 1 hour horizon
N_STEPS = 50
LAM = 1e-6


def load_one_month_data():
    """Load full 98-day window of aggTrades (2026-01-01 to 2026-04-08)."""
    if not DATA_FILES:
        raise FileNotFoundError(
            f"No BTCUSDT-aggTrades-2026-*.csv files found in {DATA_DIR}. "
            "Run `python -m calibration.download_binance` first."
        )
    from calibration.data_loader import load_trades
    trades = load_trades(
        DATA_DIR,
        start="2026-01-01",
        end="2026-04-08",
    )
    return trades


def compute_log_returns(trades):
    """Compute 5-min log returns from trade data."""
    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    returns = mid["log_return"].to_numpy()
    # Validate
    bad = ~np.isfinite(returns)
    if bad.sum() > 0:
        returns = returns[~bad]
    return returns


def main() -> None:
    print("\n" + "=" * 72)
    print("Paired test: Regime-Aware vs Single-Regime (Pooled) Execution")
    print("=" * 72)

    # ── Step 1: load data ─────────────────────────────────────────────────
    print(f"\n[1/6] Loading data from {len(DATA_FILES)} CSV files...")
    trades = load_one_month_data()
    print(f"      Loaded {len(trades):,} trades from "
          f"{trades['timestamp'].iloc[0].date()} to "
          f"{trades['timestamp'].iloc[-1].date()}")

    # ── Step 2: calibrate base (single-regime) params ────────────────────
    print("[2/6] Calibrating base (pooled) params via calibrated_params()...")
    # Pass a single CSV file to avoid loading all 90+ days — fast but representative
    cal = calibrated_params(
        trades_path=str(DATA_FILES[0]),
        X0=X0,
        T=T,
        N=N_STEPS,
        lam=LAM,
    )
    base_params = cal.params
    print(f"      S0={base_params.S0:.2f}  sigma={base_params.sigma:.4f}  "
          f"gamma={base_params.gamma:.4e}  eta={base_params.eta:.4e}")

    # ── Step 3: compute 5-min log returns, fit HMM ───────────────────────
    print("[3/6] Computing 5-min log returns and fitting 2-state Gaussian HMM...")
    returns = compute_log_returns(trades)
    print(f"      Returns series length: {len(returns)}")
    regimes, state_seq = fit_hmm(returns, n_regimes=N_REGIMES)

    # Ensure risk_on = low-vol, risk_off = high-vol (fit_hmm guarantees order)
    reg_dict = {r.label: r for r in regimes}
    risk_on = reg_dict["risk_on"]
    risk_off = reg_dict["risk_off"]

    print(f"      risk_on : sigma={risk_on.sigma:.4f}  gamma={risk_on.gamma:.4e}  "
          f"eta={risk_on.eta:.4e}  prob={risk_on.probability:.4f}")
    print(f"      risk_off: sigma={risk_off.sigma:.4f}  gamma={risk_off.gamma:.4e}  "
          f"eta={risk_off.eta:.4e}  prob={risk_off.probability:.4f}")

    # ── Step 4: build per-regime ACParams ─────────────────────────────────
    print("[4/6] Building per-regime ACParams...")
    params_riskon = regime_aware_params(base_params, risk_on)
    params_riskoff = regime_aware_params(base_params, risk_off)

    # ── Step 5: simulate costs on common random Z paths ───────────────────
    print(f"[5/6] Simulating {N_PATHS:,} paths per strategy (common random numbers)...")

    # Pre-generate the shared Z array — used by ALL three simulations
    rng = np.random.default_rng(SEED)
    Z_shared = rng.standard_normal((N_PATHS, N_STEPS))

    # TWAP trajectory is the same shape-of-execution for all strategies
    x_twap_base = twap_trajectory(base_params)
    x_twap_riskon = twap_trajectory(params_riskon)
    x_twap_riskoff = twap_trajectory(params_riskoff)

    # Strategy A: single-regime — base params, TWAP trajectory
    _, costs_single = simulate_execution(
        base_params, x_twap_base,
        n_paths=N_PATHS, seed=SEED,
        antithetic=False, scheme="exact",
        Z_extern=Z_shared,
    )

    # Strategy B: regime-aware — run risk_on and risk_off separately on same Z,
    # then combine with stationary-probability weights.
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

    # Weighted composite: E_stationary[cost] = sum_r P(regime=r) * cost_r
    w_on = risk_on.probability
    w_off = risk_off.probability
    # Re-normalise in case they don't sum exactly to 1
    total_w = w_on + w_off
    w_on /= total_w
    w_off /= total_w
    costs_aware = w_on * costs_riskon + w_off * costs_riskoff

    print(f"      mean_single  = {np.mean(costs_single):.4f}")
    print(f"      mean_riskon  = {np.mean(costs_riskon):.4f}  (weight={w_on:.4f})")
    print(f"      mean_riskoff = {np.mean(costs_riskoff):.4f}  (weight={w_off:.4f})")
    print(f"      mean_aware   = {np.mean(costs_aware):.4f}")

    # ── Step 6: paired statistical test ──────────────────────────────────
    print(f"[6/6] Running paired_strategy_test (bootstrap n={BOOTSTRAP_REPS:,})...")
    result = paired_strategy_test(
        costs_a=costs_aware,
        costs_b=costs_single,
        label_a="RegimeAware",
        label_b="SingleRegime",
        test="both",
        n_bootstrap=BOOTSTRAP_REPS,
        seed=SEED,
    )

    # ── Print results table ───────────────────────────────────────────────
    mean_single = float(np.mean(costs_single))
    mean_aware = float(np.mean(costs_aware))
    sig = (
        result.t_pvalue is not None and result.t_pvalue < 0.05
        and result.bootstrap_pvalue is not None and result.bootstrap_pvalue < 0.05
    )

    print()
    print("=" * 72)
    print("RESULTS — Regime-Aware vs Single-Regime (Pooled) Execution")
    print("=" * 72)
    print(f"  N paths              : {N_PATHS:,}")
    print(f"  mean_single          : {mean_single:>14.4f}")
    print(f"  mean_aware           : {mean_aware:>14.4f}")
    print(f"  mean_diff (A-B)      : {result.mean_diff:>+14.4f}")
    print(f"  t_statistic          : {result.t_statistic:>14.4f}")
    print(f"  t_pvalue             : {result.t_pvalue:>14.6f}")
    print(f"  bootstrap_pvalue     : {result.bootstrap_pvalue:>14.6f}")
    print(f"  significant @ 0.05   : {'YES' if sig else 'no':>14}")
    print()
    print("  Regime parameters:")
    print(f"    risk_on  sigma={risk_on.sigma:.4f}  prob={risk_on.probability:.4f}")
    print(f"    risk_off sigma={risk_off.sigma:.4f}  prob={risk_off.probability:.4f}")
    print()

    if sig and result.mean_diff < 0:
        verdict = ("REGIME-AWARE IS BETTER: statistically significant cost reduction. "
                   "HMM regime detection provides real value.")
    elif sig and result.mean_diff > 0:
        verdict = ("REGIME-AWARE IS WORSE: statistically significant cost increase. "
                   "Per-regime params (sigma*1e-8 / sigma*1e-6 magic constants) "
                   "hurt execution relative to pooled calibration.")
    else:
        verdict = ("COSMETIC: not significant at alpha=0.05. "
                   "Regime-awareness provides no statistically detectable "
                   "benefit over single pooled params. The sigma*1e-8 / sigma*1e-6 "
                   "scaling hack likely drives this null result.")

    print("  Verdict:", verdict)
    print("=" * 72)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out = {
        "n_paths": N_PATHS,
        "mean_single": mean_single,
        "mean_aware": mean_aware,
        "mean_diff": result.mean_diff,
        "t_statistic": result.t_statistic,
        "t_pvalue": result.t_pvalue,
        "bootstrap_pvalue": result.bootstrap_pvalue,
        "significant_at_0.05": sig,
        "regime_sigmas": {
            "risk_on": risk_on.sigma,
            "risk_off": risk_off.sigma,
        },
        "stationary_probs": {
            "risk_on": risk_on.probability,
            "risk_off": risk_off.probability,
        },
        "regime_gammas": {
            "risk_on": risk_on.gamma,
            "risk_off": risk_off.gamma,
            "base": base_params.gamma,
        },
        "regime_etas": {
            "risk_on": risk_on.eta,
            "risk_off": risk_off.eta,
            "base": base_params.eta,
        },
        "weights_used": {
            "risk_on": w_on,
            "risk_off": w_off,
        },
        "verdict": verdict,
    }
    out_path = DATA_DIR / "paired_regime_aware_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
