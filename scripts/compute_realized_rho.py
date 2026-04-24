"""Compute realized (physical) ρ from Coinbase 5-min OHLCV + Binance 1d klines.

Formula:
    r_t      = daily log return = log(close_t / close_{t-1})
    RV_t     = realized variance = sum of squared 5-min log returns on day t
    σ_t      = sqrt(RV_t)
    Δσ_t     = σ_t - σ_{t-1}
    rho_hat  = rolling corr(r_t, Δσ_t) over N-day window

Output: data/realized_rho_daily.json with daily rho series + monthly aggregates
compared against Q-measure ρ from heston_qmeasure_time_series.json.

The *realized ρ* is a P-measure (physical) estimate of the spot-vol
correlation. Compare against Q-measure ρ (from option prices) to check
whether Q ρ bimodal flip is real physical phenomenon vs option-pricing
artifact (skew premium).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"


def load_coinbase_5min() -> pd.DataFrame:
    with open(DATA / "coinbase_btc_5min.json") as f:
        bars = json.load(f)
    df = pd.DataFrame(bars)
    df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    df["date"] = df["dt"].dt.date
    df["log_ret"] = np.log(df["close"]).diff()
    return df


def load_binance_1d() -> pd.DataFrame:
    with open(DATA / "binance_btc_klines_1d.json") as f:
        d = json.load(f)
    if isinstance(d, dict):
        data = d.get("data", d.get("klines", d))
    else:
        data = d
    if not isinstance(data, list):
        data = list(data)
    df = pd.DataFrame(data)
    # Schema: date (YYYY-MM-DD string), open, high, low, close, volume, trades
    df["dt"] = pd.to_datetime(df["date"], utc=True)
    df["date"] = df["dt"].dt.date
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df = df.sort_values("dt").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"]).diff()
    # Garman-Klass daily variance as RV proxy (when no 5-min available)
    df["rv_gk"] = (
        0.5 * (np.log(df["high"] / df["low"])) ** 2
        - (2 * np.log(2) - 1) * (np.log(df["close"] / df["open"])) ** 2
    )
    return df


def main() -> None:
    print("Loading Coinbase 5-min …")
    cb = load_coinbase_5min()
    print(f"  5-min rows: {len(cb):,}  range: {cb['dt'].iloc[0]} → {cb['dt'].iloc[-1]}")

    daily_cb = cb.groupby("date").agg(
        ret_d=("log_ret", lambda x: x.dropna().sum()),
        rv=("log_ret", lambda x: (x.dropna() ** 2).sum()),
        n_5min=("log_ret", "count"),
    ).reset_index()
    daily_cb["sigma"] = np.sqrt(daily_cb["rv"])
    daily_cb["source"] = "coinbase_5min"
    print(f"  daily rows (coinbase): {len(daily_cb)}")

    # Binance 1d for 2025-05/06 coverage
    print("\nLoading Binance 1d …")
    bn = load_binance_1d()
    print(f"  daily rows (binance): {len(bn)}  range: {bn['date'].iloc[0]} → {bn['date'].iloc[-1]}")

    bn_only = bn[~bn["date"].isin(daily_cb["date"])].copy()
    print(f"  binance-only days (pre-Jul 2025): {len(bn_only)}")

    bn_only = bn_only.rename(columns={"rv_gk": "rv"})[["date", "log_ret", "rv"]]
    bn_only = bn_only.rename(columns={"log_ret": "ret_d"})
    bn_only["n_5min"] = np.nan
    bn_only["sigma"] = np.sqrt(bn_only["rv"])
    bn_only["source"] = "binance_1d_garman_klass"

    # Stitch
    daily = pd.concat([bn_only, daily_cb], ignore_index=True, sort=False)
    daily = daily.sort_values("date").reset_index(drop=True)
    # dedupe by date (shouldn't be any)
    daily = daily.drop_duplicates(subset=["date"])
    print(f"\n  combined daily rows: {len(daily)}")
    print(f"  coverage: {daily['date'].iloc[0]} → {daily['date'].iloc[-1]}")

    daily["d_sigma"] = daily["sigma"].diff()

    for w in (10, 20, 30, 60):
        daily[f"rho_{w}d"] = daily["ret_d"].rolling(w).corr(daily["d_sigma"])

    daily.to_csv("/tmp/realized_rho_daily.csv", index=False)

    # Compare with Q ρ
    with open(DATA / "heston_qmeasure_time_series.json") as f:
        qrho = json.load(f)

    print("\n" + "=" * 80)
    print("Monthly comparison: Q ρ (option chain) vs Realized ρ (30d rolling mean)")
    print("=" * 80)
    print(f"{'month':<12}{'Q_rho':>10}{'realized_rho_30d':>20}{'realized_rho_60d':>20}{'n_days':>8}")
    print("-" * 72)

    rows = []
    for qr in qrho:
        month = qr["date"]
        y, mo, _ = month.split("-")
        mask = daily["date"].astype(str).str.startswith(f"{y}-{mo}-")
        sub = daily[mask]
        n = len(sub)
        m30 = sub["rho_30d"].mean()
        m60 = sub["rho_60d"].mean()
        print(
            f"{month:<12}{qr['rho']:>+10.4f}"
            f"{m30 if not np.isnan(m30) else 0.0:>+20.4f}"
            f"{m60 if not np.isnan(m60) else 0.0:>+20.4f}"
            f"{n:>8d}"
        )
        rows.append(
            {
                "month": month,
                "q_rho": qr["rho"],
                "realized_rho_30d": None if np.isnan(m30) else float(m30),
                "realized_rho_60d": None if np.isnan(m60) else float(m60),
                "n_days": int(n),
            }
        )

    # Correlation of monthly series
    clean = [r for r in rows if r["realized_rho_30d"] is not None]
    if len(clean) >= 3:
        q = np.array([r["q_rho"] for r in clean])
        r30 = np.array([r["realized_rho_30d"] for r in clean])
        r60 = np.array([r["realized_rho_60d"] for r in clean if r["realized_rho_60d"] is not None])
        print(f"\n  Pearson r(Q_rho, realized_30d) = {np.corrcoef(q, r30)[0,1]:+.4f}  (n={len(clean)})")
        if len(r60) >= 3:
            q60 = np.array([r["q_rho"] for r in clean if r["realized_rho_60d"] is not None])
            print(f"  Pearson r(Q_rho, realized_60d) = {np.corrcoef(q60, r60)[0,1]:+.4f}  (n={len(r60)})")

    # Write JSON
    out = {
        "source": "coinbase_5min (RV from sq log returns) + binance_1d (Garman-Klass for pre-2025-07)",
        "method": "rolling corr(daily_log_return, daily_sigma_change)",
        "windows_days": [10, 20, 30, 60],
        "coverage": {
            "start": str(daily["date"].iloc[0]),
            "end": str(daily["date"].iloc[-1]),
            "n_days": len(daily),
        },
        "daily_rho": [
            {
                "date": str(r["date"]),
                "ret_d": None if np.isnan(r["ret_d"]) else float(r["ret_d"]),
                "sigma": None if np.isnan(r["sigma"]) else float(r["sigma"]),
                "rho_30d": None if np.isnan(r["rho_30d"]) else float(r["rho_30d"]),
                "rho_60d": None if np.isnan(r["rho_60d"]) else float(r["rho_60d"]),
                "source": r["source"],
            }
            for _, r in daily.iterrows()
        ],
        "monthly_q_vs_realized": rows,
    }
    out_path = DATA / "realized_rho_daily.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
