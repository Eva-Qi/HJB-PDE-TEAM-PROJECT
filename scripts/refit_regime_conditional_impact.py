"""Per-regime impact calibration (Task #12).

Loads 98-day BTCUSDT aggTrades, fits both 2-state and 3-state HMMs,
then for each regime sub-samples the trades where that regime is active
and re-estimates (γ, η, α, σ) directly.

Also computes Yuhao's multiplier-based approximation:
    γ_yuhao = multiplier × γ_base
    η_yuhao = multiplier × η_base

and reports how far it deviates from the true per-regime estimates.

Saves results to data/regime_conditional_impact.json

Usage:
    python scripts/refit_regime_conditional_impact.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from calibration.data_loader import load_trades, compute_mid_prices
from calibration.impact_estimator import calibrated_params, calibrated_params_per_regime
from extensions.regime import fit_hmm, regime_aware_params

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = sorted(DATA_DIR.glob("BTCUSDT-aggTrades-2026-*.csv"))

X0 = 10.0
T_HORIZON = 1.0 / 24   # 1-hour execution window
N_STEPS = 50
LAM = 1e-6


def _regime_label_to_idx(label: str, n_regimes: int) -> int:
    """Map regime label to canonical integer index."""
    if n_regimes == 2:
        return {"risk_on": 0, "risk_off": 1}[label]
    return {"risk_on": 0, "neutral": 1, "risk_off": 2}[label]


def main() -> None:
    print("\n" + "=" * 72)
    print("Per-Regime Impact Calibration (Task #12)")
    print("=" * 72)

    if not DATA_FILES:
        raise FileNotFoundError(
            f"No BTCUSDT-aggTrades-2026-*.csv files found in {DATA_DIR}."
        )

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {len(DATA_FILES)} CSV files...")
    trades = load_trades(DATA_DIR, start="2026-01-01", end="2026-04-08")
    print(f"      {len(trades):,} trades loaded  "
          f"{trades['timestamp'].iloc[0].date()} → "
          f"{trades['timestamp'].iloc[-1].date()}")

    # ── 2. Calibrate base (pooled) params ─────────────────────────────────
    print("[2/5] Calibrating base (pooled) params from first CSV file...")
    base_cal = calibrated_params(
        trades_path=str(DATA_FILES[0]),
        X0=X0, T=T_HORIZON, N=N_STEPS, lam=LAM,
    )
    base_params = base_cal.params
    print(f"      S0={base_params.S0:.2f}  sigma={base_params.sigma:.6f}  "
          f"gamma={base_params.gamma:.4e}  eta={base_params.eta:.4e}  "
          f"alpha={base_params.alpha:.4f}")

    # ── 3. Build 5-min return bars and fit HMMs ───────────────────────────
    print("[3/5] Computing 5-min returns & fitting HMMs...")
    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    mid = mid[np.isfinite(mid["log_return"])]

    bar_timestamps = mid["timestamp"]  # DatetimeIndex or column of bar starts
    returns = mid["log_return"].to_numpy()
    print(f"      {len(returns):,} 5-min return bars")

    import warnings
    results_by_n = {}

    for n_regimes in (2, 3):
        print(f"\n  Fitting {n_regimes}-state HMM...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes, state_seq = fit_hmm(returns, n_regimes=n_regimes)

        for r in regimes:
            print(
                f"    {r.label:<12} sigma_mult={r.sigma:.4f}  "
                f"state_vol={r.state_vol:.6f}  prob={r.probability:.4f}"
            )

        results_by_n[n_regimes] = {
            "regimes": regimes,
            "state_seq": state_seq,
        }

    # ── 4. Per-regime calibration ─────────────────────────────────────────
    print("\n[4/5] Running per-regime impact calibration...")

    # bar_timestamps needs to be a proper DatetimeIndex aligned to state_seq
    if isinstance(bar_timestamps, pd.Series):
        bar_ts_index = pd.DatetimeIndex(bar_timestamps.values)
    else:
        bar_ts_index = pd.DatetimeIndex(bar_timestamps)

    output_sections = {}

    for n_regimes in (2, 3):
        state_seq = results_by_n[n_regimes]["state_seq"]
        regimes = results_by_n[n_regimes]["regimes"]

        print(f"\n  === {n_regimes}-state HMM ===")

        per_regime_cal = calibrated_params_per_regime(
            trades_df=trades,
            state_sequence=state_seq,
            bar_timestamps=bar_ts_index,
            X0=X0,
            T=T_HORIZON,
            N=N_STEPS,
            lam=LAM,
        )

        rows = []
        for r in regimes:
            idx = _regime_label_to_idx(r.label, n_regimes)
            cal = per_regime_cal.get(idx)

            if cal is None:
                print(f"    {r.label}: no calibration result (regime not in data?)")
                continue

            # True per-regime params
            true_gamma = cal.params.gamma
            true_eta = cal.params.eta
            true_alpha = cal.params.alpha
            true_sigma = cal.params.sigma

            # Yuhao's approximation: base × multiplier
            yuhao_params = regime_aware_params(base_params, r)
            yuhao_gamma = yuhao_params.gamma
            yuhao_eta = yuhao_params.eta

            # Relative error
            def _rel_err(true_val: float, approx_val: float) -> float:
                if abs(true_val) < 1e-15:
                    return float("nan")
                return (approx_val - true_val) / abs(true_val)

            re_gamma = _rel_err(true_gamma, yuhao_gamma)
            re_eta = _rel_err(true_eta, yuhao_eta)

            row = {
                "label": r.label,
                "regime_idx": idx,
                "sigma_mult": float(r.sigma),
                "n_regime_trades": int(len(trades[
                    trades["timestamp"].isin(
                        trades["timestamp"]
                    )
                ]) if False else 0),  # placeholder; actual count from cal metadata
                # True per-regime estimates
                "true_sigma": float(true_sigma),
                "true_gamma": float(true_gamma),
                "true_eta": float(true_eta),
                "true_alpha": float(true_alpha),
                # Yuhao multiplier approximation
                "yuhao_gamma": float(yuhao_gamma),
                "yuhao_eta": float(yuhao_eta),
                # Relative errors
                "rel_err_gamma": float(re_gamma) if not np.isnan(re_gamma) else None,
                "rel_err_eta": float(re_eta) if not np.isnan(re_eta) else None,
                # Calibration source metadata
                "sources": cal.sources,
                "warnings": cal.warnings,
            }
            rows.append(row)

            warn_note = " (FALLBACK)" if "fallback" in " ".join(cal.warnings).lower() else ""
            print(
                f"    {r.label:<12}  "
                f"true γ={true_gamma:.4e}  η={true_eta:.4e}  α={true_alpha:.3f}  "
                f"σ={true_sigma:.4f}{warn_note}"
            )
            print(
                f"    {'':12}  "
                f"yuhao γ={yuhao_gamma:.4e}  η={yuhao_eta:.4e}  "
                f"Δγ={re_gamma:+.0%}  Δη={re_eta:+.0%}"
            )

        output_sections[f"{n_regimes}_state"] = {
            "base_params": {
                "sigma": float(base_params.sigma),
                "gamma": float(base_params.gamma),
                "eta": float(base_params.eta),
                "alpha": float(base_params.alpha),
            },
            "per_regime": rows,
        }

    # ── 5. Save JSON ──────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")
    out_path = DATA_DIR / "regime_conditional_impact.json"
    out_path.write_text(json.dumps(output_sections, indent=2))
    print(f"      Saved: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("Base (pooled) params:")
    print(
        f"  gamma={base_params.gamma:.4e}  eta={base_params.eta:.4e}  "
        f"alpha={base_params.alpha:.3f}  sigma={base_params.sigma:.6f}"
    )

    for n_regimes in (2, 3):
        section = output_sections[f"{n_regimes}_state"]
        print(f"\n{n_regimes}-state per-regime estimates:")
        print(
            f"  {'Regime':<12} {'true γ':>12} {'yuhao γ':>12} "
            f"{'Δγ':>8} {'true η':>12} {'yuhao η':>12} {'Δη':>8}"
        )
        print("  " + "-" * 78)
        for row in section["per_regime"]:
            re_g = f"{row['rel_err_gamma']:+.0%}" if row["rel_err_gamma"] is not None else "N/A"
            re_e = f"{row['rel_err_eta']:+.0%}" if row["rel_err_eta"] is not None else "N/A"
            print(
                f"  {row['label']:<12} "
                f"{row['true_gamma']:>12.4e} "
                f"{row['yuhao_gamma']:>12.4e} "
                f"{re_g:>8} "
                f"{row['true_eta']:>12.4e} "
                f"{row['yuhao_eta']:>12.4e} "
                f"{re_e:>8}"
            )

    print("=" * 72)


if __name__ == "__main__":
    main()
