"""Compute Kyle's lambda γ per calendar month using Binance aggTrades CSVs.

Produces a 12-month γ time series (2025-05 through 2026-04) that can be
analyzed for stability and correlated with Heston ρ monthly series.

Strategy per month:
  - Prefer days 10-14 of the month (mid-month sample, avoids month-end effects).
  - If fewer than 5 days are available in that range, fall back to the first
    5 available days in the month.
  - Load day-by-day, concat, then release memory before moving to next month.

Estimator used: estimate_kyle_lambda_aggregated() from calibration/impact_estimator.py
  Aggregates to 1-min buckets: Δprice vs net_signed_flow regression.
  Returns γ in $/BTC.

Output: data/kyle_gamma_monthly.json

Run:
  python scripts/kyle_gamma_monthly_time_series.py
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration.data_loader import _load_single_csv, compute_ohlc
from calibration.impact_estimator import (
    estimate_kyle_lambda_aggregated,
    estimate_realized_vol_gk,
)

DATA_DIR = PROJECT_ROOT / "data"
OUT_FILE = PROJECT_ROOT / "data" / "kyle_gamma_monthly.json"

# 12 months: 2025-05 through 2026-04
MONTHS = [
    "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10",
    "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
]

# Preferred mid-month sample window (days 10-14 inclusive)
PREFERRED_DAY_START = 10
PREFERRED_DAY_END   = 14
TARGET_DAYS         = 5


def list_month_csvs(month: str) -> list[Path]:
    """Return sorted list of all aggTrades CSVs for a given YYYY-MM month."""
    pattern = f"BTCUSDT-aggTrades-{month}-*.csv"
    return sorted(DATA_DIR.glob(pattern))


def pick_sample_days(month: str) -> list[Path]:
    """Pick up to TARGET_DAYS CSV files for the month.

    Preference order:
      1. Days 10-14 (mid-month, avoids month-open/close distortions).
      2. First TARGET_DAYS available if mid-month window has <5 files.
    """
    all_csvs = list_month_csvs(month)
    if not all_csvs:
        return []

    # Filter to preferred window (days 10-14)
    preferred = [
        p for p in all_csvs
        if PREFERRED_DAY_START <= int(p.stem.split("-")[-1]) <= PREFERRED_DAY_END
    ]

    if len(preferred) >= TARGET_DAYS:
        return preferred[:TARGET_DAYS]

    # Fallback: first TARGET_DAYS available in the full month
    return all_csvs[:TARGET_DAYS]


def load_days(csv_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate a list of aggTrades CSV files.

    Memory note: caller should del + gc.collect() after use.
    """
    frames = []
    for p in csv_paths:
        df_day = _load_single_csv(p)
        frames.append(df_day)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_realized_vol(trades_df: pd.DataFrame) -> float | None:
    """Compute annualized Garman-Klass realized vol from trade data."""
    try:
        ohlc = compute_ohlc(trades_df, freq="5min")
        return estimate_realized_vol_gk(ohlc, freq_seconds=300.0, annualize=True)
    except Exception:
        return None


def process_month(month: str) -> dict | None:
    """Compute Kyle γ for one calendar month.

    Returns a result dict, or None if the month has no data.
    Prints per-month progress to stdout.
    """
    csv_paths = pick_sample_days(month)

    if not csv_paths:
        print(f"[{month}] SKIP — no CSV files found")
        return None

    days_used = [p.stem.replace("BTCUSDT-aggTrades-", "") for p in csv_paths]
    print(f"[{month}] Loading {len(csv_paths)} days: {days_used}")

    # Load all selected days into one DataFrame
    trades = load_days(csv_paths)
    n_trades = len(trades)
    mean_price = float(trades["price"].mean())

    print(f"[{month}]   n_trades={n_trades:,}  mean_price=${mean_price:,.0f}")

    # Realized vol (Garman-Klass, annualized)
    rvol = compute_realized_vol(trades)

    # Kyle γ — try 1-min, then 5-min
    gamma = None
    r_squared = None
    n_buckets = None
    freq_used = None
    error_msg = None

    for freq in ("1min", "5min"):
        try:
            g, diag = estimate_kyle_lambda_aggregated(trades, freq=freq)
            if g > 0 and diag["r_squared"] >= 0.01:
                gamma     = g
                r_squared = diag["r_squared"]
                n_buckets = diag["n_buckets"]
                freq_used = freq
                print(
                    f"[{month}]   gamma={g:.4f} $/BTC  r²={r_squared:.4f} "
                    f"n_buckets={n_buckets}  freq={freq}"
                )
                break
            else:
                print(
                    f"[{month}]   [{freq}] γ={g:.4e} r²={diag['r_squared']:.4f} "
                    f"rejected (negative γ or low R²)"
                )
        except Exception as exc:
            print(f"[{month}]   [{freq}] failed: {exc}")
            error_msg = str(exc)

    if gamma is None:
        print(f"[{month}]   WARNING: all gamma methods failed — recording null")

    result = {
        "month":                month,
        "days_used":            days_used,
        "n_days":               len(days_used),
        "gamma_dollar_per_BTC": gamma,
        "gamma_freq":           freq_used,
        "n_trades":             n_trades,
        "n_buckets":            n_buckets,
        "mean_price":           round(mean_price, 2),
        "realized_vol_ann":     round(rvol, 6) if rvol is not None else None,
        "r_squared":            round(r_squared, 6) if r_squared is not None else None,
        "error":                error_msg,
    }

    # Release memory before processing next month
    del trades
    gc.collect()

    return result


def main() -> None:
    print("=" * 65)
    print("Kyle γ monthly time series  —  2025-05 through 2026-04")
    print(f"Data dir : {DATA_DIR}")
    print(f"Output   : {OUT_FILE}")
    print("=" * 65)

    results = []
    for month in MONTHS:
        rec = process_month(month)
        if rec is not None:
            results.append(rec)
        print()

    # Summary table
    print("=" * 65)
    print(f"{'Month':<10} {'γ ($/BTC)':>12} {'RVol ann':>10} {'R²':>8} {'N_trades':>10}")
    print("-" * 65)
    for r in results:
        g_str   = f"{r['gamma_dollar_per_BTC']:.4f}" if r["gamma_dollar_per_BTC"] is not None else "  FAILED"
        rv_str  = f"{r['realized_vol_ann']:.3f}"     if r["realized_vol_ann"]     is not None else "     N/A"
        r2_str  = f"{r['r_squared']:.4f}"            if r["r_squared"]            is not None else "     N/A"
        nt_str  = f"{r['n_trades']:,}"
        print(f"{r['month']:<10} {g_str:>12} {rv_str:>10} {r2_str:>8} {nt_str:>10}")
    print("=" * 65)

    # Stability check
    gammas = [r["gamma_dollar_per_BTC"] for r in results if r["gamma_dollar_per_BTC"] is not None]
    if len(gammas) >= 2:
        ratio = max(gammas) / min(gammas)
        print(f"\nγ range: {min(gammas):.4f} – {max(gammas):.4f} $/BTC")
        print(f"Max/min ratio: {ratio:.1f}x  {'(>10x: NOT stable)' if ratio > 10 else '(<= 10x: plausibly stable)'}")

    # Write JSON
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {len(results)} months → {OUT_FILE}")


if __name__ == "__main__":
    main()
