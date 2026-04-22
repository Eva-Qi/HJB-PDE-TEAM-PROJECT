"""Download Kraken XBTUSD daily OHLCV from public REST API.

Endpoint: https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1440&since=<unix_sec>
Returns up to 720 candles per request (720 days ~= 2 years), so 2023-01-01 → today fits
in 2 calls (covering ~840 days).

Response structure:
    {
        "error": [],
        "result": {
            "XXBTZUSD": [[time, open, high, low, close, vwap, volume, count], ...],
            "last": <int>
        }
    }

Usage:
    python -m calibration.download_kraken
    python -m calibration.download_kraken --start 2023-01-01 --end 2026-04-22
    python -m calibration.download_kraken --out data/kraken_btc_daily.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://api.kraken.com/0/public/OHLC"
PAIR = "XBTUSD"
RESPONSE_KEY = "XXBTZUSD"
INTERVAL_DAILY = 1440  # minutes

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUT_FILE = DEFAULT_DATA_DIR / "kraken_btc_daily.json"


def fetch_ohlcv(since_unix: int) -> tuple[list[list], int | None]:
    """Fetch up to 720 daily candles from Kraken since given unix timestamp.

    Parameters
    ----------
    since_unix : int
        Unix timestamp (seconds) for the start of the fetch window.

    Returns
    -------
    candles : list[list]
        Raw candle rows: [time, open, high, low, close, vwap, volume, count].
    last : int or None
        Kraken's `last` cursor for pagination (None on error).
    """
    params = urllib.parse.urlencode({
        "pair": PAIR,
        "interval": INTERVAL_DAILY,
        "since": since_unix,
    })
    url = f"{API_URL}?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "mf796-project/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())

    errors = body.get("error", [])
    if errors:
        raise RuntimeError(f"Kraken API error: {errors}")

    result = body["result"]
    candles = result.get(RESPONSE_KEY, [])
    last = result.get("last")
    return candles, last


def download_daily(
    start: str = "2023-01-01",
    end: str | None = None,
    out_file: Path | None = None,
) -> Path:
    """Download Kraken XBTUSD daily candles and save as JSON.

    Parameters
    ----------
    start : str
        Start date YYYY-MM-DD (inclusive).
    end : str, optional
        End date YYYY-MM-DD (inclusive). Defaults to today.
    out_file : Path, optional
        Output JSON path. Defaults to data/kraken_btc_daily.json.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    if out_file is None:
        out_file = DEFAULT_OUT_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end is None:
        end_dt = datetime.now(tz=timezone.utc)
    else:
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    since_unix = int(start_dt.timestamp())
    end_unix = int(end_dt.timestamp())

    # Kraken daily interval: each page covers at most 720 days.
    # We need to page in 720-day chunks from start to end.
    CHUNK_DAYS = 720
    CHUNK_SECS = CHUNK_DAYS * 86400

    print(f"Fetching Kraken {PAIR} daily OHLCV: {start} → {end or 'today'}")

    all_candles: list[dict] = []
    page = 0
    current_since = since_unix

    while current_since < end_unix:
        page += 1
        print(f"  [page {page}] since={current_since} ({datetime.utcfromtimestamp(current_since).strftime('%Y-%m-%d')})")
        candles, last = fetch_ohlcv(current_since)

        if not candles:
            print("  [done] No candles returned for this window.")
            # Advance window manually if Kraken returns nothing
            current_since += CHUNK_SECS
            time.sleep(0.5)
            continue

        added = 0
        max_t = current_since
        for row in candles:
            t = int(row[0])
            max_t = max(max_t, t)
            if t <= end_unix:
                all_candles.append({
                    "time": t,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "vwap": float(row[5]),
                    "volume": float(row[6]),
                    "count": int(row[7]),
                })
                added += 1

        print(f"    Got {len(candles)} candles, {added} in range. last_ts={last}")

        # Advance: use Kraken's `last` cursor if available and advancing,
        # otherwise jump manually by one chunk.
        if last is not None and last > current_since:
            current_since = last
        else:
            current_since = max_t + 86400  # next day after the last candle

        if current_since >= end_unix:
            break

        time.sleep(0.5)  # be polite

    # Deduplicate and sort by time
    seen: set[int] = set()
    unique: list[dict] = []
    for c in sorted(all_candles, key=lambda x: x["time"]):
        if c["time"] not in seen:
            seen.add(c["time"])
            unique.append(c)

    print(f"\n  Total candles: {len(unique)}")
    if unique:
        first_dt = datetime.utcfromtimestamp(unique[0]["time"]).strftime("%Y-%m-%d")
        last_dt = datetime.utcfromtimestamp(unique[-1]["time"]).strftime("%Y-%m-%d")
        print(f"  Date range: {first_dt} → {last_dt}")

    out_file.write_text(json.dumps(unique, indent=2))
    print(f"  Saved to {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Download Kraken XBTUSD daily OHLCV data"
    )
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default=None, help="Output JSON file path")
    args = parser.parse_args()

    out_file = Path(args.out) if args.out else None
    download_daily(start=args.start, end=args.end, out_file=out_file)


if __name__ == "__main__":
    main()
