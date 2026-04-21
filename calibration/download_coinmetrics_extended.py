import urllib.request
import urllib.parse
import json
import argparse
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Download extended BTC metrics from CoinMetrics Community API")
    parser.add_argument("--start", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-04-08", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/coinmetrics_btc_extended_metrics.json", help="Output file path")
    parser.add_argument("--metrics", nargs="*", default=None, help="Specific metrics to pull (default: all candidates)")
    args = parser.parse_args()

    base_url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    
    all_candidate_metrics = [
        "NVTAdj",
        "CapMVRVCur",
        "HashRate",
        "CapRealUSD",
        "ReferenceRateUSD"
    ]
    
    metrics_to_pull = args.metrics if args.metrics else all_candidate_metrics

    metrics_retrieved = []
    metrics_failed = {}
    combined_data = {}

    for metric in metrics_to_pull:
        print(f"Fetching {metric}...")
        params = {
            "assets": "btc",
            "metrics": metric,
            "start_time": args.start,
            "end_time": args.end,
            "frequency": "1d",
            "page_size": "1000"
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}?{query_string}"

        req = urllib.request.Request(url, headers={"User-Agent": "Python-urllib/3.x"})
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status != 200:
                    metrics_failed[metric] = f"HTTP {response.status}"
                    print(f"  Failed: HTTP {response.status}")
                    continue
                
                raw_data = json.loads(response.read().decode("utf-8"))
                response_data = raw_data.get("data", [])
                
                if not response_data:
                    metrics_failed[metric] = "Empty data"
                    print("  Failed: Empty data")
                    continue
                
                count = 0
                for entry in response_data:
                    date_str = entry.get("time", "")[:10]
                    val_str = entry.get(metric)
                    
                    if date_str and val_str is not None:
                        try:
                            val = float(val_str)
                            if date_str not in combined_data:
                                combined_data[date_str] = {}
                            combined_data[date_str][metric] = val
                            count += 1
                        except (ValueError, TypeError):
                            continue
                
                metrics_retrieved.append(metric)
                print(f"  Success: {count} data points")

        except urllib.error.HTTPError as e:
            metrics_failed[metric] = f"HTTP {e.code}"
            print(f"  Failed: HTTP {e.code}")
        except Exception as e:
            metrics_failed[metric] = str(e)
            print(f"  Failed: {e}")

    sorted_dates = sorted(combined_data.keys())
    data_list = [{"date": d, **combined_data[d]} for d in sorted_dates]

    output_obj = {
        "start": args.start,
        "end": args.end,
        "metrics_retrieved": metrics_retrieved,
        "metrics_failed": metrics_failed,
        "data": data_list
    }

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output_obj, f, indent=2)
    
    print(f"\nSaved to {args.output}")


    print("\n" + "=" * 60)
    print("FREE-TIER AVAILABLE METRICS")
    print("=" * 60)
    if metrics_retrieved:
        for m in metrics_retrieved:
            print(f"  [FREE] {m}")
    else:
        print("  None available")
    
    if metrics_failed:
        print("\nFAILED / PRO-TIER METRICS:")
        for m, reason in metrics_failed.items():
            print(f"  [FAIL] {m}: {reason}")

    print("\n" + "=" * 60)
    print("PER-METRIC SUMMARY")
    print("=" * 60)

    metric_values = {}
    for entry in data_list:
        for m in metrics_retrieved:
            if m in entry:
                if m not in metric_values:
                    metric_values[m] = []
                metric_values[m].append(entry[m])

    def compute_stats(vals):
        n = len(vals)
        if n == 0:
            return None
        mean_val = sum(vals) / n
        variance = sum((x - mean_val) ** 2 for x in vals) / n if n > 1 else 0.0
        std_val = variance ** 0.5
        min_val = min(vals)
        max_val = max(vals)
        return mean_val, std_val, min_val, max_val

    for m in metrics_retrieved:
        vals = metric_values.get(m, [])
        stats = compute_stats(vals)
        if stats:
            mean_val, std_val, min_val, max_val = stats
            print(f"\n  {m}:")
            print(f"    Count: {len(vals)}")
            print(f"    Mean:  {mean_val:.4f}")
            print(f"    Std:   {std_val:.4f}")
            print(f"    Min:   {min_val:.4f}")
            print(f"    Max:   {max_val:.4f}")

    print("\n" + "=" * 60)
    print("BIMODALITY / REGIME-INFORMATIVENESS CHECK")
    print("=" * 60)

    def hartigan_dip_test(vals):
        if len(vals) < 4:
            return 0.0
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        
        if n == 0:
            return 0.0

        ecdf = [i / n for i in range(n + 1)]
        sample_points = []
        for i in range(n):
            sample_points.append(sorted_vals[i])
        sample_points.append(sorted_vals[-1] + 1e-9)
        
        dip = 0.0
        for i in range(n):
            g_min = ecdf[i] - (i / n)
            g_max = ((i + 1) / n) - ecdf[i]
            candidate = max(abs(g_min), abs(g_max))
            if candidate > dip:
                dip = candidate

        return dip / (2.0 ** 0.5)

    def bimodality_coefficient(vals):
        n = len(vals)
        if n < 4:
            return 0.0
        mean_val = sum(vals) / n
        m2 = sum((x - mean_val) ** 2 for x in vals) / n
        m3 = sum((x - mean_val) ** 3 for x in vals) / n
        m4 = sum((x - mean_val) ** 4 for x in vals) / n
        
        if m2 == 0:
            return 0.0
        
        skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
        kurt = m4 / (m2 ** 2) if m2 > 0 else 0.0
        
        bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))) if n > 3 else 0.0
        return bc

    def histogram_mode_score(vals, bins=10):
        if len(vals) < 2:
            return 0.0
        min_v = min(vals)
        max_v = max(vals)
        if min_v == max_v:
            return 0.0
        
        counts = [0] * bins
        for v in vals:
            idx = int((v - min_v) / (max_v - min_v) * (bins - 0.0001))
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1
        
        max_count = max(counts)
        non_modal_count = sum(counts) - max_count
        non_modal_bins = sum(1 for c in counts if c > 0) - 1
        
        if non_modal_bins <= 0 or max_count == 0:
            return 0.0
        
        secondary_mode_count = max(c for i, c in enumerate(counts) if c != max_count) if any(c != max_count for c in counts) else 0
        
        bimodal_ratio = secondary_mode_count / max_count if max_count > 0 else 0.0
        return bimodal_ratio

    regime_scores = {}
    
    for m in metrics_retrieved:
        vals = metric_values.get(m, [])
        if len(vals) < 4:
            continue
        
        bc = bimodality_coefficient(vals)
        hms = histogram_mode_score(vals)
        
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        
        dip = hartigan_dip_test(vals)
        
        combined_score = bc * 0.4 + hms * 0.3 + dip * 0.3
        
        regime_scores[m] = {
            "bimodality_coefficient": bc,
            "histogram_bimodal_ratio": hms,
            "dip_estimate": dip,
            "combined_score": combined_score
        }
        
        print(f"\n  {m}:")
        print(f"    Bimodality Coefficient: {bc:.4f}")
        print(f"    Histogram Bimodal Ratio: {hms:.4f}")
        print(f"    Dip Estimate: {dip:.4f}")
        print(f"    Combined Score: {combined_score:.4f}")

    if regime_scores:
        best_metric = max(regime_scores, key=lambda k: regime_scores[k]["combined_score"])
        best_score = regime_scores[best_metric]["combined_score"]
        
        print(f"\n  MOST REGIME-INFORMATIVE (highest combined score):")
        print(f"    => {best_metric} (score: {best_score:.4f})")
        
        if regime_scores[best_metric]["bimodality_coefficient"] > 0.555:
            print(f"    Bimodality likely (BC > 0.555 threshold)")
        else:
            print(f"    Bimodality uncertain (BC <= 0.555)")
        
        ranked = sorted(regime_scores.items(), key=lambda x: x[1]["combined_score"], reverse=True)
        print(f"\n  Ranking by regime-informativeness:")
        for rank, (metric_name, scores) in enumerate(ranked, 1):
            print(f"    {rank}. {metric_name}: {scores['combined_score']:.4f}")
    else:
        print("\n  No metrics available for bimodality analysis.")


if __name__ == "__main__":
    main()