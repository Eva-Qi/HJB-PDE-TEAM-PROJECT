"""Pull Deribit BTC DVOL (Deribit Volatility Index) historical time series.

DVOL is Deribit's 30-day implied volatility index — the crypto analog of VIX.
Comparing our fitted Heston sqrt(theta) across monthly Tardis snapshots to the
DVOL value on the same date provides an independent validation of the
Q-measure calibration quality.

Endpoint: public/get_volatility_index_data (no auth, no API key).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "data" / "deribit_dvol_btc_daily.json"

BASE = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def _ms(date_iso: str) -> int:
    dt = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch(start: str, end: str, resolution: str = "1D") -> dict:
    params = {
        "currency": "BTC",
        "start_timestamp": _ms(start),
        "end_timestamp": _ms(end),
        "resolution": resolution,
    }
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-05-01")
    p.add_argument("--end", default="2026-04-21")
    p.add_argument("--resolution", default="1D", help="1, 60, 3600, 43200, 1D")
    args = p.parse_args()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching Deribit DVOL {args.start} → {args.end} resolution={args.resolution}")
    resp = fetch(args.start, args.end, args.resolution)

    # Response: { result: { data: [[timestamp_ms, open, high, low, close], ...] } }
    data = resp.get("result", {}).get("data", [])
    records = [
        {
            "date": datetime.fromtimestamp(row[0] / 1000.0, tz=timezone.utc)
                            .strftime("%Y-%m-%d"),
            "timestamp_ms": row[0],
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
        }
        for row in data
    ]

    if not records:
        print("ERROR: no DVOL rows returned")
        print(json.dumps(resp)[:500])
        sys.exit(1)

    OUTPUT.write_text(json.dumps({
        "source": "Deribit public/get_volatility_index_data",
        "start": args.start,
        "end": args.end,
        "resolution": args.resolution,
        "n_rows": len(records),
        "data": records,
    }, indent=2))
    print(f"Saved: {OUTPUT} ({len(records)} rows)")

    # Summary
    closes = [r["close"] for r in records]
    print(f"\nDVOL summary (close values, 0-100 scale):")
    print(f"  min  = {min(closes):.2f}")
    print(f"  max  = {max(closes):.2f}")
    print(f"  mean = {sum(closes)/len(closes):.2f}")
    print(f"  last = {closes[-1]:.2f}")


if __name__ == "__main__":
    main()
