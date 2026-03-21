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


def load_trades(
    path: str | Path,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load trade data from Binance CSV.

    Expected columns: timestamp, price, qty, is_buyer_maker
    Returns DataFrame with columns: timestamp, price, quantity, side
    where side = +1 (buy) or -1 (sell).

    Parameters
    ----------
    path : str or Path
        Path to CSV file or directory of CSVs.
    start, end : str, optional
        ISO datetime strings for filtering.
    """
    raise NotImplementedError(
        "P1: implement trade data loading. "
        "See data.binance.vision for BTCUSDT aggTrades format."
    )


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


def compute_mid_prices(snapshots: pd.DataFrame) -> pd.Series:
    """Extract mid prices from order book snapshots.

    mid = (best_bid + best_ask) / 2
    """
    raise NotImplementedError("P1: extract mid prices from snapshots.")


def walk_the_book_slippage(
    snapshots: pd.DataFrame,
    quantities: np.ndarray,
    side: str = "sell",
    fill_ratio: float = 0.5,
) -> np.ndarray:
    """Estimate slippage by walking the order book for various quantities.

    This is a simplified version of QC_Trade_Platform's WalkTheBook._walk_levels().
    For each (snapshot, quantity) pair, walks through book levels consuming
    fill_ratio of each level's liquidity, computing VWAP slippage in bps.

    Parameters
    ----------
    snapshots : pd.DataFrame
        Order book snapshots with bid/ask prices and quantities.
    quantities : np.ndarray
        Array of order sizes to simulate.
    side : str
        'buy' or 'sell'.
    fill_ratio : float
        Fraction of each level's liquidity assumed available (0.3-0.7).

    Returns
    -------
    np.ndarray
        Slippage in bps for each (snapshot, quantity) combination.
        Shape: (len(snapshots), len(quantities)).
    """
    raise NotImplementedError(
        "P1: implement walk-the-book slippage estimation. "
        "Reference: QC_Trade_Platform/src/orderbook/walk_the_book.py _walk_levels()"
    )
