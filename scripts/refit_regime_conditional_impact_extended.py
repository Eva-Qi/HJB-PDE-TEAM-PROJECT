"""Per-regime impact calibration on extended 280-day window (2025-07 → 2026-04).

Loads all BTCUSDT aggTrades from 2025-07-01 through 2026-04-08 (~280 days),
fits both 2-state and 3-state HMMs, then calibrates (γ, η, α, σ) per regime.

Compared to the 98-day version (refit_regime_conditional_impact.py):
  - risk_off sub-sample is ~3× larger → η should now be truly estimated
  - Yuhao relative error comparison is included for continuity

Saves results to data/regime_conditional_impact_extended.json

Usage:
    python scripts/refit_regime_conditional_impact_extended.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from calibration.data_loader import load_trades, compute_mid_prices
from calibration.impact_estimator import calibrated_params, calibrated_params_per_regime
from extensions.regime import fit_hmm, regime_aware_params

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

X0 = 10.0
T_HORIZON = 1.0/(365.25*24)   # 1-hour execution window
N_STEPS = 250
LAM = 1e-6

# Extended window: 2025-07-01 → 2026-04-08 (~280 days)
START_DATE = "2025-07-01"
END_DATE   = "2026-04-08"


def _regime_label_to_idx(label: str, n_regimes: int) -> int:
    if n_regimes == 2:
        return {"risk_on": 0, "risk_off": 1}[label]
    return {"risk_on": 0, "neutral": 1, "risk_off": 2}[label]


def main() -> None:
    print("\n" + "=" * 72)
    print("Per-Regime Impact Calibration — Extended 280-day Window")
    print(f"Date range: {START_DATE} → {END_DATE}")
    print("=" * 72)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading trades from {DATA_DIR} ...")
    trades = load_trades(DATA_DIR, start=START_DATE, end=END_DATE)
    n_total = len(trades)
    date_min = trades["timestamp"].iloc[0].date()
    date_max = trades["timestamp"].iloc[-1].date()
    print(f"      {n_total:,} trades loaded  {date_min} → {date_max}")

    # Count unique days
    n_days = trades["timestamp"].dt.date.nunique()
    print(f"      {n_days} unique trading days")

    # ── 2. Calibrate base (pooled) params ─────────────────────────────────
    print("\n[2/5] Calibrating base (pooled) params from full extended sample...")
    # Use all data for pooled calibration (no file path — pass DataFrame path trick)
    # We call each estimator directly instead of calibrated_params() file-based API
    from calibration.impact_estimator import (
        estimate_realized_vol_gk,
        estimate_realized_vol_rs,
        estimate_kyle_lambda_aggregated,
        estimate_temporary_impact_aggregated,
    )
    from calibration.data_loader import compute_ohlc
    from shared.params import ACParams

    ohlc_all = compute_ohlc(trades, freq="5min")
    sigma_base = estimate_realized_vol_gk(ohlc_all, freq_seconds=300.0, annualize=True)

    gamma_base, g_diag = estimate_kyle_lambda_aggregated(trades, freq="1min")
    if not (gamma_base > 0 and g_diag["r_squared"] >= 0.01):
        gamma_base, g_diag = estimate_kyle_lambda_aggregated(trades, freq="5min")
    if gamma_base <= 0:
        gamma_base = 1e-4

    eta_base, alpha_base, imp_diag = estimate_temporary_impact_aggregated(trades, freq="1min")
    if not (0.3 <= alpha_base <= 1.5 and imp_diag["r_squared"] >= 0.05):
        eta_base, alpha_base, imp_diag = estimate_temporary_impact_aggregated(trades, freq="5min")

    S0_base = float(trades["price"].iloc[-1])
    base_params = ACParams(
        S0=S0_base, sigma=sigma_base, mu=0.0,
        X0=X0, T=T_HORIZON, N=N_STEPS,
        gamma=gamma_base, eta=eta_base, alpha=alpha_base,
        lam=LAM, fee_bps=7.5,
    )
    print(f"      S0={S0_base:.2f}  sigma={sigma_base:.6f}  "
          f"gamma={gamma_base:.4e}  eta={eta_base:.4e}  alpha={alpha_base:.4f}")

    # ── 3. Build 5-min return bars and fit HMMs ───────────────────────────
    print("\n[3/5] Computing 5-min returns & fitting HMMs...")
    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    mid = mid[np.isfinite(mid["log_return"])]

    bar_timestamps = mid["timestamp"]
    returns = mid["log_return"].to_numpy()
    print(f"      {len(returns):,} 5-min return bars over {n_days} days")

    if isinstance(bar_timestamps, pd.Series):
        bar_ts_index = pd.DatetimeIndex(bar_timestamps.values)
    else:
        bar_ts_index = pd.DatetimeIndex(bar_timestamps)

    results_by_n: dict = {}

    for n_regimes in (2, 3):
        print(f"\n  Fitting {n_regimes}-state HMM...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                regimes, state_seq = fit_hmm(returns, n_regimes=n_regimes)
            except RuntimeError as e:
                # On 190-day extended window hmmlearn can fail (all inits
                # collapse to single state when vol distribution is close
                # to unimodal). Fallback to rolling-vol-feature EM which
                # uses a different initialization scheme.
                print(f"    hmmlearn failed ({str(e)[:80]}) — "
                      f"falling back to rolling-vol EM")
                from extensions.regime import _fit_hmm_rolling_vol, RegimeParams
                means, stds, A, pi, state_seq_raw = _fit_hmm_rolling_vol(
                    returns, n_regimes=n_regimes,
                )
                # Build RegimeParams minimally — borrow logic from fit_hmm
                # by calling it on synthetic data that should NOT fail,
                # then monkey-patching; simpler: just bypass and rebuild.
                order = np.argsort(stds)
                state_seq = np.array([
                    int(np.where(order == s)[0][0]) for s in state_seq_raw
                ], dtype=int)
                labels = ["risk_on", "neutral", "risk_off"] if n_regimes == 3 else ["risk_on", "risk_off"]
                overall_vol = float(np.std(returns, ddof=1))
                regimes = []
                for i, lab in enumerate(labels):
                    mask = state_seq == i
                    if mask.sum() < 2:
                        state_vol = overall_vol
                    else:
                        state_vol = float(np.std(returns[mask], ddof=1))
                    sigma_mult = state_vol / max(overall_vol, 1e-12)
                    regimes.append(RegimeParams(
                        label=lab, sigma=sigma_mult, gamma=sigma_mult, eta=sigma_mult,
                        probability=float(mask.mean()),
                        state_vol=state_vol,
                        state_abs_ret=float(np.mean(np.abs(returns[mask]))) if mask.any() else 0.0,
                        state_mean_ret=float(np.mean(returns[mask])) if mask.any() else 0.0,
                    ))

        for r in regimes:
            mask_r = (state_seq == _regime_label_to_idx(r.label, n_regimes))
            n_bars_regime = int(mask_r.sum())
            print(
                f"    {r.label:<12} sigma_mult={r.sigma:.4f}  "
                f"state_vol={r.state_vol:.6f}  prob={r.probability:.4f}  "
                f"n_bars={n_bars_regime:,}"
            )

        results_by_n[n_regimes] = {
            "regimes": regimes,
            "state_seq": state_seq,
        }

    # ── 4. Per-regime calibration ─────────────────────────────────────────
    print("\n[4/5] Running per-regime impact calibration...")

    output_sections: dict = {}

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

            true_gamma = cal.params.gamma
            true_eta   = cal.params.eta
            true_alpha = cal.params.alpha
            true_sigma = cal.params.sigma

            yuhao_params = regime_aware_params(base_params, r)
            yuhao_gamma  = yuhao_params.gamma
            yuhao_eta    = yuhao_params.eta

            def _rel_err(true_val: float, approx_val: float) -> float | None:
                if abs(true_val) < 1e-15:
                    return None
                return (approx_val - true_val) / abs(true_val)

            re_gamma = _rel_err(true_gamma, yuhao_gamma)
            re_eta   = _rel_err(true_eta, yuhao_eta)

            # Count trades in this regime
            # We need to re-do the merge_asof to get actual trade counts
            # (mirrors what calibrated_params_per_regime does internally)
            bar_label_df = pd.DataFrame(
                {"bar_ts": bar_ts_index, "regime": state_seq}
            ).sort_values("bar_ts").reset_index(drop=True)
            trades_sorted = trades.sort_values("timestamp").reset_index(drop=True)
            merged_check = pd.merge_asof(
                trades_sorted[["timestamp"]],
                bar_label_df.rename(columns={"bar_ts": "timestamp"}),
                on="timestamp",
                direction="backward",
            )
            merged_check = merged_check.dropna(subset=["regime"])
            merged_check["regime"] = merged_check["regime"].astype(int)
            n_regime_trades = int((merged_check["regime"] == idx).sum())

            row = {
                "label": r.label,
                "regime_idx": idx,
                "sigma_mult": float(r.sigma),
                "n_regime_trades": n_regime_trades,
                "true_sigma": float(true_sigma),
                "true_gamma": float(true_gamma),
                "true_eta": float(true_eta),
                "true_alpha": float(true_alpha),
                "yuhao_gamma": float(yuhao_gamma),
                "yuhao_eta": float(yuhao_eta),
                "rel_err_gamma": float(re_gamma) if re_gamma is not None else None,
                "rel_err_eta": float(re_eta) if re_eta is not None else None,
                "sources": cal.sources,
                "warnings": cal.warnings,
            }
            rows.append(row)

            is_fallback = "fallback" in " ".join(str(w) for w in cal.warnings).lower()
            eta_note = " (FALLBACK)" if is_fallback else " (ESTIMATED)"
            print(
                f"    {r.label:<12}  n_trades={n_regime_trades:,}  "
                f"true γ={true_gamma:.4e}  η={true_eta:.4e}  α={true_alpha:.3f}  "
                f"σ={true_sigma:.4f}{eta_note}"
            )
            if re_gamma is not None and re_eta is not None:
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
            "metadata": {
                "start_date": START_DATE,
                "end_date": END_DATE,
                "n_total_trades": n_total,
                "n_days": n_days,
                "n_bars": int(len(returns)),
            },
        }

    # ── 5. Save JSON ──────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")
    out_path = DATA_DIR / "regime_conditional_impact_extended.json"
    out_path.write_text(json.dumps(output_sections, indent=2))
    print(f"      Saved: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY — Extended 280-day window")
    print("=" * 72)
    print(f"Total trades: {n_total:,}  ({n_days} days)")
    print("Base (pooled) params:")
    print(
        f"  gamma={base_params.gamma:.4e}  eta={base_params.eta:.4e}  "
        f"alpha={base_params.alpha:.3f}  sigma={base_params.sigma:.6f}"
    )

    for n_regimes in (2, 3):
        section = output_sections[f"{n_regimes}_state"]
        print(f"\n{n_regimes}-state per-regime estimates:")
        print(
            f"  {'Regime':<12} {'n_trades':>10} {'true γ':>12} "
            f"{'true η':>12} {'true α':>8} {'η source':>15}"
        )
        print("  " + "-" * 75)
        for row in section["per_regime"]:
            eta_src = row["sources"].get("eta", "?")
            print(
                f"  {row['label']:<12} "
                f"{row['n_regime_trades']:>10,} "
                f"{row['true_gamma']:>12.4e} "
                f"{row['true_eta']:>12.4e} "
                f"{row['true_alpha']:>8.4f} "
                f"{eta_src:>15}"
            )

    print("=" * 72)


if __name__ == "__main__":
    main()
