"""Pull Deribit BTC-PERPETUAL funding rate history (hourly).

Public API, no auth. Used to check whether mu=0 default in MC simulation
is justified by observed funding. High observed funding → large implied
drift → mu=0 may bias execution cost distributions.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "data" / "deribit_btc_funding_hourly.json"
BASE = "https://www.deribit.com/api/v2/public/get_funding_rate_history"


def _ms(iso: str) -> int:
    dt = datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch(start: str, end: str, instrument: str = "BTC-PERPETUAL") -> list:
    # Deribit caps at 744 rows per request (~31 days hourly). Loop.
    results = []
    start_ms = _ms(start)
    end_ms = _ms(end)
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + 744 * 3600 * 1000, end_ms)
        params = {
            "instrument_name": instrument,
            "start_timestamp": cursor,
            "end_timestamp": chunk_end,
        }
        url = f"{BASE}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=60) as r:
            resp = json.loads(r.read().decode("utf-8"))
        rows = resp.get("result", [])
        if not rows:
            break
        results.extend(rows)
        # Advance past latest returned
        max_ts = max(r["timestamp"] for r in rows)
        if max_ts <= cursor:
            break
        cursor = max_ts + 3600 * 1000  # next hour
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default="2026-04-08")
    args = p.parse_args()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching Deribit BTC-PERPETUAL funding {args.start} → {args.end}")
    rows = fetch(args.start, args.end)
    print(f"  Got {len(rows):,} rows")

    if not rows:
        print("ERROR: no funding rows — endpoint may have changed")
        return

    # Sort and clean — fields from Deribit: timestamp, interest_1h, interest_8h, prev_index_price, index_price
    rows.sort(key=lambda r: r["timestamp"])

    hourly_rates = [r.get("interest_1h", 0.0) for r in rows]
    import statistics
    mean_rate_1h = statistics.mean(hourly_rates)
    # Annualized: 24 * 365 hourly compoundings
    mean_annualized = (1 + mean_rate_1h) ** (24 * 365) - 1

    summary = {
        "source": "Deribit public/get_funding_rate_history",
        "instrument": "BTC-PERPETUAL",
        "start": args.start,
        "end": args.end,
        "n_rows": len(rows),
        "stats": {
            "mean_interest_1h": mean_rate_1h,
            "std_interest_1h": statistics.pstdev(hourly_rates),
            "min_interest_1h": min(hourly_rates),
            "max_interest_1h": max(hourly_rates),
            "mean_annualized_drift": mean_annualized,
            "median_interest_1h": statistics.median(hourly_rates),
        },
        "data": rows,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2))
    print(f"Saved: {OUTPUT}")
    print(f"\nFunding stats:")
    print(f"  mean hourly funding   = {mean_rate_1h:.6e}")
    print(f"  mean annualized drift = {mean_annualized:.4%}")
    print(f"  median hourly funding = {statistics.median(hourly_rates):.6e}")
    print(f"  range: [{min(hourly_rates):.6e}, {max(hourly_rates):.6e}]")

    # Narrative
    annual = mean_annualized
    if abs(annual) < 0.02:
        verdict = f"Funding implies drift ≈ {annual:.2%} — mu=0 in MC is a reasonable approximation."
    elif abs(annual) < 0.10:
        verdict = f"Funding implies drift ≈ {annual:.2%} — mu=0 introduces mild bias; mu=funding-implied would tighten MC results slightly."
    else:
        verdict = f"Funding implies drift ≈ {annual:.2%} — mu=0 is a material bias; consider updating MC default."
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
