"""Download Binance BTCUSDT aggTrades data from data.binance.vision.

This is a free, no-API-key-required data source. Files come as daily ZIPs
containing a single CSV with columns:
    agg_trade_id, price, quantity, first_trade_id, last_trade_id,
    timestamp, is_buyer_maker, is_best_match

Usage:
    python -m calibration.download_binance
    python -m calibration.download_binance --days 5 --symbol BTCUSDT
    python -m calibration.download_binance --start 2025-07-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"

# See data_loader.py for AGG_TRADES_COLUMNS column definitions

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_day(
    symbol: str,
    date: datetime,
    out_dir: Path,
    timeout: int = 60,
) -> Path | None:
    """Download one day of aggTrades data.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. "BTCUSDT".
    date : datetime
        The date to download.
    out_dir : Path
        Directory to save the extracted CSV.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    Path or None
        Path to the saved CSV, or None if download failed.
    """
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}"
    zip_url = f"{BASE_URL}/{symbol}/{filename}.zip"
    csv_path = out_dir / f"{filename}.csv"

    # Skip if already downloaded
    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"  [skip] {csv_path.name} already exists")
        return csv_path

    print(f"  [GET]  {zip_url}")
    try:
        resp = requests.get(zip_url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [FAIL] {date_str}: {e}")
        return None

    # Extract the CSV from the ZIP
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Typically contains one file: BTCUSDT-aggTrades-YYYY-MM-DD.csv
            names = zf.namelist()
            csv_name = [n for n in names if n.endswith(".csv")]
            if not csv_name:
                print(f"  [FAIL] {date_str}: no CSV found in ZIP (contents: {names})")
                return None
            data = zf.read(csv_name[0])
    except zipfile.BadZipFile:
        print(f"  [FAIL] {date_str}: corrupted ZIP")
        return None

    csv_path.write_bytes(data)
    size_mb = len(data) / (1024 * 1024)
    print(f"  [OK]   {csv_path.name} ({size_mb:.1f} MB)")
    return csv_path


def download_recent(
    symbol: str = "BTCUSDT",
    days: int = 7,
    out_dir: Path | None = None,
) -> list[Path]:
    """Download the most recent N days of aggTrades data.

    Starts from yesterday (today's file is usually not yet available)
    and works backward. Skips dates that fail (e.g., too recent).

    Parameters
    ----------
    symbol : str
        Trading pair.
    days : int
        Number of days to attempt (will download up to this many).
    out_dir : Path, optional
        Output directory. Defaults to project data/.

    Returns
    -------
    list[Path]
        Paths to successfully downloaded CSV files.
    """
    if out_dir is None:
        out_dir = DEFAULT_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {symbol} aggTrades ({days} days) -> {out_dir}/")

    downloaded = []
    # Start from 2 days ago to be safe (today and yesterday may not be ready)
    start_date = datetime.utcnow() - timedelta(days=2)

    for i in range(days):
        date = start_date - timedelta(days=i)
        result = download_day(symbol, date, out_dir)
        if result is not None:
            downloaded.append(result)

    print(f"\nDone: {len(downloaded)}/{days} files downloaded")
    return downloaded


def download_date_range(
    symbol: str = "BTCUSDT",
    start: str = "",
    end: str = "",
    out_dir: Path | None = None,
) -> list[Path]:
    """Download aggTrades data for an explicit date range (inclusive).

    Parameters
    ----------
    symbol : str
        Trading pair.
    start : str
        Start date in ISO format YYYY-MM-DD (inclusive).
    end : str
        End date in ISO format YYYY-MM-DD (inclusive).
    out_dir : Path, optional
        Output directory. Defaults to project data/.

    Returns
    -------
    list[Path]
        Paths to successfully downloaded CSV files.
    """
    if out_dir is None:
        out_dir = DEFAULT_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    if start_dt > end_dt:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    total_days = (end_dt - start_dt).days + 1
    print(
        f"Downloading {symbol} aggTrades {start} → {end} "
        f"({total_days} days) -> {out_dir}/"
    )

    downloaded = []
    failed = []
    current = start_dt

    while current <= end_dt:
        result = download_day(symbol, current, out_dir)
        if result is not None:
            downloaded.append(result)
        else:
            failed.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(
        f"\nDone: {len(downloaded)}/{total_days} files downloaded"
        + (f"  ({len(failed)} failed: {', '.join(failed[:5])}{'...' if len(failed) > 5 else ''})" if failed else "")
    )
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance BTCUSDT aggTrades from data.binance.vision"
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Number of recent days to download (mutually exclusive with --start/--end)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format (use with --end for explicit range)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (use with --start for explicit range)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: project data/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    # Validate arg combinations
    has_range = args.start is not None or args.end is not None
    has_days = args.days is not None

    if has_range and has_days:
        parser.error("--days is mutually exclusive with --start/--end")

    if has_range:
        if args.start is None or args.end is None:
            parser.error("--start and --end must both be provided together")
        downloaded = download_date_range(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            out_dir=out_dir,
        )
    else:
        days = args.days if has_days else 7
        downloaded = download_recent(
            symbol=args.symbol,
            days=days,
            out_dir=out_dir,
        )

    if not downloaded:
        print("\nNo files downloaded. Check your network or try different dates.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
