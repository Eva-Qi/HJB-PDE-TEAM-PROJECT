import argparse
import json
import sys
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from collections import defaultdict
import math

BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

def fetch_metric(metric, start, end):
    params = {
        "assets": "btc",
        "metrics": metric,
        "start_time": start,
        "end_time": end,
        "frequency": "1d",
        "page_size": 10000
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    
    req = urllib.request.Request(url, headers={"User-Agent": "CryptoResearch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", []), None
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return [], f"{e.code} {e.reason}"
    except Exception as e:
        return [], str(e)

def compute_stats(values):
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        std = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    return {
        "count": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4)
    }

def main():
    parser = argparse.ArgumentParser(description="Download BTC on-chain metrics from CoinMetrics Community API")
    parser.add_argument("--start", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-04-08", help="End date (YYYY-MM-DD)")
    parser.add_argument("--metrics", default="PriceUSD,FlowInExUSD,FlowOutExUSD,TxTfrValAdjUSD,AdrActCnt",
                        help="Comma-separated list of metrics")
    args = parser.parse_args()

    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics_retrieved = []
    metrics_failed = {}
    date_values = defaultdict(dict)
    all_dates = set()

    for metric in metrics_list:
        print(f"Fetching {metric} ...", end=" ", flush=True)
        data, error = fetch_metric(metric, args.start, args.end)
        if error:
            metrics_failed[metric] = error
            print(f"FAILED ({error})")
            continue
        if not data:
            metrics_failed[metric] = "empty response"
            print("FAILED (empty response)")
            continue
        
        count = 0
        for entry in data:
            date_str = entry.get("time", "")[:10]
            val_str = entry.get(metric)
            if date_str and val_str is not None:
                try:
                    val = float(val_str)
                    date_values[date_str][metric] = val
                    all_dates.add(date_str)
                    count += 1
                except (ValueError, TypeError):
                    pass
        
        if count == 0:
            metrics_failed[metric] = "no valid data points"
            print("FAILED (no valid data points)")
        else:
            metrics_retrieved.append(metric)
            print(f"OK ({count} rows)")

    sorted_dates = sorted(all_dates)
    output_data = []
    for d in sorted_dates:
        row = {"date": d}
        row.update(date_values[d])
        output_data.append(row)

    output = {
        "start": args.start,
        "end": args.end,
        "metrics_retrieved": metrics_retrieved,
        "metrics_failed": metrics_failed,
        "data": output_data
    }

    import os
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "coinmetrics_btc_onchain.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n===== SUMMARY =====")
    print(f"Row count       : {len(output_data)}")
    print(f"Date range      : {sorted_dates[0] if sorted_dates else 'N/A'} -> {sorted_dates[-1] if sorted_dates else 'N/A'}")
    print(f"Metrics OK      : {metrics_retrieved}")
    print(f"Metrics FAILED  : {list(metrics_failed.keys())}")

    for metric in metrics_retrieved:
        vals = [row[metric] for row in output_data if metric in row]
        stats = compute_stats(vals)
        if stats:
            print(f"\n  {metric}:")
            print(f"    count = {stats['count']}")
            print(f"    mean  = {stats['mean']}")
            print(f"    std   = {stats['std']}")
            print(f"    min   = {stats['min']}")
            print(f"    max   = {stats['max']}")

    flow_metrics = [m for m in metrics_retrieved if "Flow" in m]
    if flow_metrics:
        print(f"\n[OK] Exchange flow metrics available: {flow_metrics}")
    else:
        print("\n[WARNING] No exchange flow metrics were available (likely pro-tier only).")
        print("         Consider using PriceUSD + TxTfrValAdjUSD + AdrActCnt as emission features.")


if __name__ == "__main__":
    main()