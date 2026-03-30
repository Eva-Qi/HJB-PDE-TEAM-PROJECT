"""Load and preprocess Binance order book and trade data.

Data sources (in priority order):
    1. QC_Trade_Platform processed data: ~/Desktop/QC_Trade_Platform/data/
    2. Binance historical CSV: data.binance.vision (free, download manually)
    3. Raw snapshots from Tardis.dev

Expected data formats:
    Trades CSV: timestamp, price, qty, is_buyer_maker
    Order book snapshots: timestamp, bids[(price, qty)...], asks[(price, qty)...]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Column names for Binance aggTrades CSVs (no header row in the raw files)
AGG_TRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


def load_trades(
    path: str | Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load trade data from Binance aggTrades CSV(s).

    Expected columns: timestamp, price, qty, is_buyer_maker
    Returns DataFrame with columns: timestamp, price, quantity, side
    where side = +1 (buy) or -1 (sell).

    Parameters
    ----------
    path : str or Path
        Path to a single CSV file, or a directory containing multiple
        BTCUSDT-aggTrades-*.csv files. If a directory, all matching
        CSVs are loaded and concatenated.
    start, end : str, optional
        ISO datetime strings for filtering (e.g. '2026-03-15 00:00:00').
    """
    path = Path(path)

    if path.is_dir():
        csv_files = sorted(path.glob("*-aggTrades-*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No aggTrades CSV files found in {path}. "
                "Run `python -m calibration.download_binance` first."
            )
        dfs = [_load_single_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = _load_single_csv(path)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter by time range if specified
    if start is not None:
        start_ts = pd.Timestamp(start)
        df = df[df["timestamp"] >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        df = df[df["timestamp"] <= end_ts]

    return df.reset_index(drop=True)


def _load_single_csv(filepath: Path) -> pd.DataFrame:
    """Load a single Binance aggTrades CSV and normalize columns.

    Returns DataFrame with columns: timestamp, price, quantity, side.
    """
    df = pd.read_csv(
        filepath,
        header=None,
        names=AGG_TRADES_COLUMNS,
        dtype={
            "agg_trade_id": np.int64,
            "price": np.float64,
            "quantity": np.float64,
            "first_trade_id": np.int64,
            "last_trade_id": np.int64,
            "timestamp": np.int64,
            "is_buyer_maker": bool,
            "is_best_match": bool,
        },
    )

    # Binance aggTrades use microsecond timestamps (16 digits, not 13)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)

    # Side convention: is_buyer_maker=True means taker is selling
    # side = +1 for buyer-initiated (taker buy), -1 for seller-initiated
    df["side"] = np.where(df["is_buyer_maker"], -1, 1)

    return df[["timestamp", "price", "quantity", "side"]]


def compute_ohlc(
    trades: pd.DataFrame,
    freq: str = "5min",
) -> pd.DataFrame:
    """Resample trade data into OHLC bars.

    Uses all tick prices within each bar — much more information than
    just VWAP. Required for Garman-Klass volatility estimator.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with columns: timestamp, price.
    freq : str
        Resampling frequency (default: '5min').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    df = trades.set_index("timestamp")

    ohlc = df["price"].resample(freq).ohlc()
    volume = df["quantity"].resample(freq).sum()

    result = ohlc.copy()
    result["volume"] = volume
    result = result.dropna()
    result = result.reset_index()

    return result


def load_orderbook_snapshots(
    path: str | Path,
    depth: int = 20,
) -> pd.DataFrame:
    """Load order book snapshots.

    Returns DataFrame with columns:
        timestamp, bid_prices (list), bid_qtys (list),
        ask_prices (list), ask_qtys (list), mid_price

    Parameters
    ----------
    path : str or Path
        Path to snapshot data.
    depth : int
        Number of levels per side to retain.
    """
    raise NotImplementedError(
        "P1: implement order book snapshot loading. "
        "Can use Tardis.dev normalized format or Binance depth snapshots."
    )


def compute_mid_prices(
    trades: pd.DataFrame,
    freq: str = "5min",
) -> pd.DataFrame:
    """Compute mid-price proxy from trade data by resampling.

    Since we don't have order book data yet, we approximate mid price
    as the VWAP within each time bucket. This is a reasonable proxy
    for liquid instruments like BTCUSDT.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with columns: timestamp, price, quantity.
    freq : str
        Resampling frequency (default: '5min').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, mid_price, volume.
        Indexed by time bucket.
    """
    df = trades.set_index("timestamp")

    # VWAP per bucket as mid-price proxy
    notional = (df["price"] * df["quantity"]).resample(freq).sum()
    volume = df["quantity"].resample(freq).sum()
    vwap = notional / volume

    result = pd.DataFrame({
        "mid_price": vwap,
        "volume": volume,
    }).dropna()

    result = result.reset_index()
    return result
