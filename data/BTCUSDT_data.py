# ARCADIA CONTRIBUTION — pending review
# Binance aggTrades download + live depth snapshot collection
import os
import time
import json
import argparse
import zipfile
from io import BytesIO
from datetime import datetime, timedelta

import requests


BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT"
BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def daterange(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def download_aggtrades(start_date: str, end_date: str, out_dir: str) -> None:
    """
    Download daily BTCUSDT aggTrades zip files from Binance Vision
    and extract them into out_dir/aggTrades/.
    """
    agg_dir = os.path.join(out_dir, "aggTrades")
    ensure_dir(agg_dir)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    session = requests.Session()

    for day in daterange(start_dt, end_dt):
        day_str = day.strftime("%Y-%m-%d")
        zip_name = f"BTCUSDT-aggTrades-{day_str}.zip"
        csv_name = f"BTCUSDT-aggTrades-{day_str}.csv"
        zip_path = os.path.join(agg_dir, zip_name)
        csv_path = os.path.join(agg_dir, csv_name)
        url = f"{BINANCE_VISION_BASE}/{zip_name}"

        if os.path.exists(csv_path):
            print(f"[SKIP] {csv_name} already extracted.")
            continue

        print(f"[DOWNLOAD] {url}")
        resp = session.get(url, timeout=30)

        if resp.status_code != 200:
            print(f"[FAIL] Could not download {zip_name} (status {resp.status_code})")
            continue

        with open(zip_path, "wb") as f:
            f.write(resp.content)

        try:
            with zipfile.ZipFile(BytesIO(resp.content)) as zf:
                zf.extractall(agg_dir)
            print(f"[OK] Extracted {csv_name}")
        except Exception as e:
            print(f"[FAIL] Could not extract {zip_name}: {e}")
            continue


def collect_live_depth_snapshots(
    seconds: int,
    interval: float,
    out_dir: str,
    limit: int = 50,
) -> None:
    """
    Collect live BTCUSDT order book snapshots from Binance depth endpoint.
    Saves each snapshot as a JSON file into out_dir/depth_snapshots_live/.
    """
    snap_dir = os.path.join(out_dir, "depth_snapshots_live")
    ensure_dir(snap_dir)

    n = int(seconds / interval)
    session = requests.Session()

    print(f"[INFO] Collecting {n} live snapshots, interval={interval}s, limit={limit}")

    for i in range(n):
        ts_ms = int(time.time() * 1000)
        params = {"symbol": "BTCUSDT", "limit": limit}

        try:
            resp = session.get(BINANCE_DEPTH_URL, params=params, timeout=10)
            resp.raise_for_status()
            payload = resp.json()

            record = {
                "local_timestamp_ms": ts_ms,
                "symbol": "BTCUSDT",
                "limit": limit,
                "lastUpdateId": payload.get("lastUpdateId"),
                "bids": payload.get("bids", []),
                "asks": payload.get("asks", []),
            }

            out_file = os.path.join(snap_dir, f"BTCUSDT_depth_{ts_ms}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(record, f)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"[OK] Saved snapshot {i + 1}/{n}")

        except Exception as e:
            print(f"[WARN] Snapshot {i + 1}/{n} failed: {e}")

        if i < n - 1:
            time.sleep(interval)


def write_readme(out_dir: str, start_date: str, end_date: str, seconds: int, interval: float) -> None:
    """
    Save a short note describing what this script collected.
    """
    readme_path = os.path.join(out_dir, "DATA_COLLECTION_NOTE.txt")
    text = f"""Project data collection note

1) Historical aggTrades downloaded:
   BTCUSDT
   Date range: {start_date} to {end_date}
   Source: Binance Vision public daily aggTrades

2) Live depth snapshots collected:
   Symbol: BTCUSDT
   Duration: {seconds} seconds
   Interval: {interval} seconds
   Source: Binance /api/v3/depth live endpoint

Important:
- This script DOES NOT recreate historical book_snapshot_25 for past dates.
- If the project specifically requires historical order-book snapshots for 2026-03-17 to 2026-03-21,
  you still need Tardis.dev exports or files already collected by a teammate.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", type=str, default="2026-03-17")
    parser.add_argument("--end_date", type=str, default="2026-03-21")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--collect_live_depth", action="store_true")
    parser.add_argument("--live_seconds", type=int, default=300)
    parser.add_argument("--live_interval", type=float, default=1.0)
    parser.add_argument("--depth_limit", type=int, default=50)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("\n=== Step 1: Download historical aggTrades ===")
    download_aggtrades(args.start_date, args.end_date, args.out_dir)

    if args.collect_live_depth:
        print("\n=== Step 2: Collect live depth snapshots ===")
        collect_live_depth_snapshots(
            seconds=args.live_seconds,
            interval=args.live_interval,
            out_dir=args.out_dir,
            limit=args.depth_limit,
        )
    else:
        print("\n[INFO] Live depth collection skipped.")

    write_readme(
        out_dir=args.out_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        seconds=args.live_seconds,
        interval=args.live_interval,
    )

    print("\n[DONE] Data collection finished.")


if __name__ == "__main__":
    main()
