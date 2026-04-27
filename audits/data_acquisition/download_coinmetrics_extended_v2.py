"""
CoinMetrics Community API — extended BTC metrics, multi-year pull.
Outputs: data/coinmetrics_btc_extended_metrics_v2.json

Free-tier quirks:
- Community endpoint: https://community-api.coinmetrics.io/v4/
- No API key required, but rate-limited (~10 req/s).
- Exchange-specific flows (BFX = Bitfinex) may return 401 on community tier.
- page_size max = 10000; use next_page_token for pagination.
- Some metrics have ~1-month data lag on community tier.
- FlowOutBFXNtv / FlowInBFXNtv are exchange-tagged and often paywalled.
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import argparse
import os
import time
from datetime import datetime


CANDIDATE_METRICS = [
    # On-chain activity
    "AdrActCnt",       # active addresses (daily)
    "TxCnt",           # transaction count
    "TxTfrCnt",        # transfer count
    "TxTfrValAdjUSD",  # adjusted transfer value USD
    # Fees
    "FeeMeanNtv",      # mean fee in BTC
    "FeeMeanUSD",      # mean fee USD
    # Network security
    "HashRate",        # hash rate (TH/s)
    "DiffMean",        # mining difficulty
    # Valuation ratios
    "NVTAdj",          # NVT ratio (adjusted)
    "CapMVRVCur",      # MVRV (current price / realized price)
    "CapRealUSD",      # realized cap USD
    # Supply dynamics
    "SplyAct1yr",      # supply active in past 1 year (HODL proxy)
    "SplyFF",          # free-float supply
    # Exchange flows (try; skip if 401/403)
    "FlowOutBFXNtv",   # Bitfinex outflow native — likely paywalled
    "FlowInBFXNtv",    # Bitfinex inflow native — likely paywalled
    "FlowOutExNtv",    # all-exchange outflow native
    "FlowInExNtv",     # all-exchange inflow native
    # Price reference
    "ReferenceRateUSD",
]

BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
PAGE_SIZE = 10000
RATE_LIMIT_SLEEP = 0.12  # ~8 req/s, safely under 10 req/s


def fetch_metric(metric: str, start: str, end: str) -> list[dict]:
    """Fetch all pages for one metric. Returns list of {time, <metric>} dicts."""
    results = []
    next_page_token = None

    while True:
        params = {
            "assets": "btc",
            "metrics": metric,
            "start_time": start,
            "end_time": end,
            "frequency": "1d",
            "page_size": str(PAGE_SIZE),
        }
        if next_page_token:
            params["next_page_token"] = next_page_token

        url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Python-urllib/3.x"})

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise e  # caller handles 401/403/etc.

        data = raw.get("data", [])
        results.extend(data)

        next_page_token = raw.get("next_page_token")
        if not next_page_token:
            break

        time.sleep(RATE_LIMIT_SLEEP)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download multi-year BTC metrics from CoinMetrics Community API (v2)"
    )
    parser.add_argument("--start", default="2023-05-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-04-22", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default="data/coinmetrics_btc_extended_metrics_v2.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Override metric list (default: all candidates)",
    )
    args = parser.parse_args()

    metrics_to_pull = args.metrics if args.metrics else CANDIDATE_METRICS

    combined_data: dict[str, dict] = {}  # date -> {metric: value}
    succeeded: dict[str, dict] = {}  # metric -> {count, first_date, last_date}
    failed: dict[str, str] = {}       # metric -> reason

    print(f"Date range: {args.start} → {args.end}")
    print(f"Metrics to attempt: {len(metrics_to_pull)}\n")

    for metric in metrics_to_pull:
        print(f"Fetching {metric} ...", end=" ", flush=True)
        time.sleep(RATE_LIMIT_SLEEP)

        try:
            rows = fetch_metric(metric, args.start, args.end)
        except urllib.error.HTTPError as e:
            reason = f"HTTP {e.code}"
            failed[metric] = reason
            print(f"SKIP ({reason})")
            continue
        except Exception as e:
            failed[metric] = str(e)
            print(f"SKIP ({e})")
            continue

        if not rows:
            failed[metric] = "empty response"
            print("SKIP (empty)")
            continue

        dates_seen = []
        count = 0
        for entry in rows:
            date_str = entry.get("time", "")[:10]
            val_raw = entry.get(metric)
            if date_str and val_raw is not None:
                try:
                    val = float(val_raw)
                except (ValueError, TypeError):
                    continue
                if date_str not in combined_data:
                    combined_data[date_str] = {}
                combined_data[date_str][metric] = val
                dates_seen.append(date_str)
                count += 1

        if count == 0:
            failed[metric] = "no parseable values"
            print("SKIP (no parseable values)")
            continue

        dates_seen.sort()
        succeeded[metric] = {
            "count": count,
            "first_date": dates_seen[0],
            "last_date": dates_seen[-1],
        }
        print(f"OK  ({count} pts, {dates_seen[0]} – {dates_seen[-1]})")

    # Build output
    sorted_dates = sorted(combined_data.keys())
    data_list = [{"date": d, **combined_data[d]} for d in sorted_dates]

    output_obj = {
        "start": args.start,
        "end": args.end,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics_succeeded": succeeded,
        "metrics_failed": failed,
        "data": data_list,
    }

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output_obj, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)
    for m, info in succeeded.items():
        print(f"  [OK]   {m:25s}  {info['count']:4d} pts  {info['first_date']} – {info['last_date']}")
    for m, reason in failed.items():
        print(f"  [FAIL] {m:25s}  {reason}")

    print(f"\nTotal rows saved: {len(data_list)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
