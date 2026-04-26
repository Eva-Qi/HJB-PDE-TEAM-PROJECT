"""Compute 5-min BTCUSDT log-returns once, cache to .npy.

Loads aggTrades day-by-day (NOT a global concat) to keep peak memory low,
resamples each day's mid-prices to 5-min, and concatenates the 5-min series.
Then computes log-returns and saves to a .npy file.

Output: data/btc_5min_log_returns_<start>_to_<end>.npy

Re-run only when the date window changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from calibration.data_loader import _load_single_csv

DATA_DIR = PROJECT_ROOT / "data"
START = "2026-01-01"
END = "2026-04-08"  # inclusive


def main() -> None:
    start_ts = pd.Timestamp(START, tz="UTC")
    end_ts = pd.Timestamp(END, tz="UTC") + pd.Timedelta(days=1)

    days = pd.date_range(start_ts, end_ts - pd.Timedelta(days=1), freq="D")
    print(f"Window: {START} → {END} ({len(days)} days)")

    closes = []
    n_files = 0
    for d in days:
        fp = DATA_DIR / f"BTCUSDT-aggTrades-{d.strftime('%Y-%m-%d')}.csv"
        if not fp.exists():
            print(f"  MISSING: {fp.name}")
            continue
        df = _load_single_csv(fp)
        df_idx = df.set_index("timestamp").sort_index()
        c = df_idx["price"].resample("5min").last().dropna()
        closes.append(c)
        n_files += 1
        if n_files % 10 == 0:
            print(f"  loaded {n_files}/{len(days)} days")

    print(f"\nLoaded {n_files} days. Concatenating 5-min closes...")
    close_series = pd.concat(closes).sort_index()

    # Boundary filter (in case of slop at edges)
    close_series = close_series[(close_series.index >= start_ts) & (close_series.index < end_ts)]

    log_ret = np.log(close_series).diff().dropna()
    log_ret = log_ret[np.isfinite(log_ret)]
    rets = log_ret.to_numpy()
    print(f"  {len(rets):,} 5-min log-returns")
    print(f"  mean={rets.mean():.2e}, std={rets.std():.2e}, min={rets.min():.4f}, max={rets.max():.4f}")

    out_path = DATA_DIR / f"btc_5min_log_returns_{START}_to_{END}.npy"
    np.save(out_path, rets)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
