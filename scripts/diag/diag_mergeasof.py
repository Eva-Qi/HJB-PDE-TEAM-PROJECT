"""Diagnose merge_asof alignment in calibrated_params_per_regime.
Uses ONE day of data to be fast.
"""
import sys, os
sys.path.insert(0, "/Users/evanolott/Desktop/MF796-COURSE PROJECT/mf796_project")
os.chdir("/Users/evanolott/Desktop/MF796-COURSE PROJECT/mf796_project")

import numpy as np
import pandas as pd
from calibration.data_loader import _load_single_csv, compute_mid_prices
from pathlib import Path

# Load ONE day
trades = _load_single_csv(Path("data/BTCUSDT-aggTrades-2026-01-02.csv"))
print(f"Loaded {len(trades)} trades from 1 day")
print(f"trades.timestamp dtype: {trades['timestamp'].dtype}")
print(f"trades.timestamp.dt.tz: {trades['timestamp'].dt.tz}")
print(f"First 3 ts: {trades['timestamp'].head(3).tolist()}")

# Build 5-min mid_prices
mid = compute_mid_prices(trades, freq="5min").copy()
mid["log_return"] = np.log(mid["mid_price"]).diff()
mid = mid.dropna()
mid = mid[np.isfinite(mid["log_return"])]
print(f"\n{len(mid)} 5-min bars")
print(f"mid.timestamp dtype: {mid['timestamp'].dtype}")
print(f"mid.timestamp.dt.tz: {mid['timestamp'].dt.tz}")

# Simulate what the refit script does:
bar_timestamps = mid["timestamp"]
# Line 114 in script: bar_ts_index = pd.DatetimeIndex(bar_timestamps.values)
# `.values` on tz-aware Series returns numpy datetime64 array WITHOUT tz
bar_ts_index_v1 = pd.DatetimeIndex(bar_timestamps.values)
print(f"\n[v1] pd.DatetimeIndex(bar_timestamps.values).tz = {bar_ts_index_v1.tz}")
bar_ts_index_v2 = pd.DatetimeIndex(bar_timestamps)
print(f"[v2] pd.DatetimeIndex(bar_timestamps).tz = {bar_ts_index_v2.tz}")

# Now simulate calibrated_params_per_regime's logic
state_sequence = np.zeros(len(mid), dtype=int)  # dummy regime labels
state_sequence[len(state_sequence) // 2 :] = 1  # second half = regime 1

bar_ts = pd.DatetimeIndex(bar_ts_index_v1)  # what script passes
print(f"\nBefore tz_localize: bar_ts.tz = {bar_ts.tz}")
if bar_ts.tz is None:
    bar_ts = bar_ts.tz_localize("UTC")
else:
    bar_ts = bar_ts.tz_convert("UTC")
print(f"After tz handling: bar_ts.tz = {bar_ts.tz}")

bar_label_df = pd.DataFrame({"bar_ts": bar_ts, "regime": state_sequence}).sort_values("bar_ts").reset_index(drop=True)
print(f"\nbar_label_df.dtypes:\n{bar_label_df.dtypes}")

trades_sorted = trades.sort_values("timestamp").reset_index(drop=True)
trade_ts = trades_sorted["timestamp"].copy()
if trade_ts.dt.tz is None:
    trade_ts = trade_ts.dt.tz_localize("UTC")
else:
    trade_ts = trade_ts.dt.tz_convert("UTC")
trades_sorted = trades_sorted.copy()
trades_sorted["timestamp"] = trade_ts
print(f"trades_sorted.timestamp.dt.tz: {trades_sorted['timestamp'].dt.tz}")

merged = pd.merge_asof(
    trades_sorted,
    bar_label_df.rename(columns={"bar_ts": "timestamp"}),
    on="timestamp",
    direction="backward",
)
print(f"\nmerged.head():")
print(merged.head(5))
n_total = len(merged)
n_with_regime = merged["regime"].notna().sum()
print(f"\nmerged total rows: {n_total}")
print(f"rows with non-NaN regime: {n_with_regime}")
print(f"Unique regime labels: {merged['regime'].dropna().unique()}")

merged = merged.dropna(subset=["regime"])
merged["regime"] = merged["regime"].astype(int)
for r in sorted(merged["regime"].unique()):
    n = (merged["regime"] == r).sum()
    print(f"  regime {r}: {n} trades")
