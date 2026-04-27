"""
Backfill BTCUSDT aggTrades daily CSVs from Binance Vision.

URL pattern:
  https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/
      BTCUSDT-aggTrades-YYYY-MM-DD.zip

Usage:
  python calibration/download_binance_aggTrades_backfill.py \
      --start 2025-04-01 --end 2025-06-30

Defaults fill the gap before the existing 2025-07-01 data.
"""

import argparse
import io
import os
import time
import zipfile
from datetime import date, timedelta

import requests

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = (
    "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/"
    "BTCUSDT-aggTrades-{date}.zip"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SLEEP_BETWEEN = 0.75  # seconds — be polite to Binance Vision


# ── Helpers ────────────────────────────────────────────────────────────────────
def iter_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def csv_path(day: date) -> str:
    fname = f"BTCUSDT-aggTrades-{day}.csv"
    return os.path.abspath(os.path.join(DATA_DIR, fname))


def already_exists(day: date) -> bool:
    return os.path.isfile(csv_path(day))


def download_day(session: requests.Session, day: date) -> int | None:
    """
    Download and unzip one day's aggTrades CSV.
    Returns number of bytes written, or None on failure.
    """
    url = BASE_URL.format(date=day)
    try:
        resp = session.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"  FAILED  {day}  (network error: {exc})")
        return None

    if resp.status_code == 404:
        print(f"  MISS    {day}  (404 — not available on Vision)")
        return None

    if resp.status_code != 200:
        print(f"  FAILED  {day}  (HTTP {resp.status_code})")
        return None

    # Unzip in-memory
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            # Expect exactly one CSV inside
            csv_names = [n for n in names if n.endswith(".csv")]
            if not csv_names:
                print(f"  FAILED  {day}  (zip contains no CSV: {names})")
                return None
            target_name = csv_names[0]
            raw = zf.read(target_name)
    except zipfile.BadZipFile as exc:
        print(f"  FAILED  {day}  (bad zip: {exc})")
        return None

    out_path = csv_path(day)
    with open(out_path, "wb") as f:
        f.write(raw)

    size_kb = len(raw) / 1024
    print(f"  GOT     {day}  ({size_kb:,.0f} KB → {out_path})")
    return len(raw)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backfill BTCUSDT aggTrades CSVs from Binance Vision."
    )
    parser.add_argument(
        "--start",
        default="2025-04-01",
        help="First date to download, YYYY-MM-DD (default: 2025-04-01)",
    )
    parser.add_argument(
        "--end",
        default="2025-06-30",
        help="Last date to download, YYYY-MM-DD (default: 2025-06-30)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if start > end:
        parser.error(f"--start {start} is after --end {end}")

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Backfill range: {start} → {end}")
    print(f"Output dir    : {os.path.abspath(DATA_DIR)}")
    print("-" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": "mf796-research/1.0"})

    total_days = 0
    skipped = 0
    downloaded = 0
    failed = 0
    total_bytes = 0

    for day in iter_dates(start, end):
        total_days += 1
        if already_exists(day):
            print(f"  SKIP    {day}  (already in data/)")
            skipped += 1
            continue

        nbytes = download_day(session, day)
        if nbytes is not None:
            downloaded += 1
            total_bytes += nbytes
        else:
            failed += 1

        time.sleep(SLEEP_BETWEEN)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"Done. {total_days} calendar days in range.")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped}  (already existed)")
    print(f"  Failed/Miss: {failed}  (404 or error)")
    print(f"  Disk added : {total_bytes / 1_048_576:.1f} MB")


if __name__ == "__main__":
    main()
