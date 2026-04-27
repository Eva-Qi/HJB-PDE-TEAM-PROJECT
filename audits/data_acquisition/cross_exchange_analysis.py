"""Cross-exchange BTC daily close price comparison.

Compares Binance.US (aggTrades), Coinbase (5-min OHLCV), and Kraken (daily OHLCV)
for the last 30 trading days and flags any day where the max cross-exchange spread
exceeds 50 bps.

Usage:
    python -m calibration.cross_exchange_analysis
    python -m calibration.cross_exchange_analysis --days 60
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_binance_daily(days: int = 30) -> dict[str, float]:
    """Load Binance daily close (last aggTrade price of each day)."""
    binance_daily: dict[str, float] = {}
    ref_dt = datetime(2026, 4, 22, tzinfo=timezone.utc)
    for i in range(days):
        dt = ref_dt - timedelta(days=i + 1)
        date_str = dt.strftime("%Y-%m-%d")
        csv_path = DATA_DIR / f"BTCUSDT-aggTrades-{date_str}.csv"
        if not csv_path.exists():
            continue
        last_price = None
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) >= 2:
                    try:
                        last_price = float(row[1])
                    except ValueError:
                        pass
        if last_price is not None:
            binance_daily[date_str] = last_price
    return binance_daily


def load_coinbase_daily() -> dict[str, float]:
    """Load Coinbase daily close from 5-min OHLCV JSON (last candle of each day)."""
    fpath = DATA_DIR / "coinbase_btc_5min.json"
    if not fpath.exists():
        return {}
    with open(fpath) as f:
        cb_data = json.load(f)

    # time may be ms or seconds
    per_day: dict[str, dict] = {}
    for row in cb_data:
        t = row["time"]
        dt = datetime.utcfromtimestamp(t / 1000) if t > 1e12 else datetime.utcfromtimestamp(t)
        date_str = dt.strftime("%Y-%m-%d")
        if date_str not in per_day or t > per_day[date_str]["ts"]:
            per_day[date_str] = {"ts": t, "close": row["close"]}
    return {d: v["close"] for d, v in per_day.items()}


def load_kraken_daily() -> dict[str, float]:
    """Load Kraken daily close from JSON."""
    fpath = DATA_DIR / "kraken_btc_daily.json"
    if not fpath.exists():
        return {}
    with open(fpath) as f:
        kr_data = json.load(f)
    result: dict[str, float] = {}
    for row in kr_data:
        dt = datetime.utcfromtimestamp(row["time"])
        date_str = dt.strftime("%Y-%m-%d")
        result[date_str] = row["close"]
    return result


def run_analysis(days: int = 30) -> None:
    print(f"Loading data for cross-exchange comparison (last {days} days)...")

    binance = load_binance_daily(days=days)
    coinbase = load_coinbase_daily()
    kraken = load_kraken_daily()

    print(f"  Binance: {len(binance)} days")
    print(f"  Coinbase: {len(coinbase)} days total")
    print(f"  Kraken: {len(kraken)} days total")

    # Build date list
    ref_dt = datetime(2026, 4, 22, tzinfo=timezone.utc)
    dates = sorted([
        (ref_dt - timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(days)
    ])

    print()
    header = f"{'date':<12} {'binance_us':>12} {'coinbase':>12} {'kraken':>12} {'max_spread_bps':>16}"
    print(header)
    print("-" * len(header))

    table_rows = []
    flagged = []

    for date in dates:
        b = binance.get(date)
        c = coinbase.get(date)
        k = kraken.get(date)

        prices = [p for p in [b, c, k] if p is not None]
        if len(prices) < 2:
            continue

        spread_bps = (max(prices) - min(prices)) / min(prices) * 10000
        table_rows.append((date, b, c, k, spread_bps))

        b_s = f"{b:>12.2f}" if b is not None else "         N/A"
        c_s = f"{c:>12.2f}" if c is not None else "         N/A"
        k_s = f"{k:>12.2f}" if k is not None else "         N/A"
        flag = "  *** FLAG" if spread_bps > 50 else ""
        print(f"{date:<12} {b_s} {c_s} {k_s} {spread_bps:>14.1f}{flag}")

        if spread_bps > 50:
            flagged.append((date, spread_bps))

    print()
    if flagged:
        print("FLAGGED DATES (max_spread_bps > 50):")
        for d, s in sorted(flagged, key=lambda x: -x[1]):
            print(f"  {d}  {s:.1f} bps  <-- potential cross-exchange inefficiency")
    else:
        print("No dates with max_spread_bps > 50.")

    print()
    print("Top 5 largest spreads:")
    for d, b, c, k, s in sorted(table_rows, key=lambda x: -x[4])[:5]:
        b_s = f"{b:.2f}" if b is not None else "N/A"
        c_s = f"{c:.2f}" if c is not None else "N/A"
        k_s = f"{k:.2f}" if k is not None else "N/A"
        print(f"  {d}  binance={b_s}  coinbase={c_s}  kraken={k_s}  spread={s:.1f} bps")


def main():
    parser = argparse.ArgumentParser(description="Cross-exchange BTC daily price comparison")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    args = parser.parse_args()
    run_analysis(days=args.days)


if __name__ == "__main__":
    main()
