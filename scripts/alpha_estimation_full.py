"""Full alpha estimation with results saved to JSON.

Processes data day-by-day to avoid loading all trades into memory at once.
Runs aggregated flow regression at 1-min and 5-min windows across
all available aggTrades data, plus per-day stability analysis.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from calibration.data_loader import _load_single_csv


def aggregate_buckets(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate trades into time buckets."""
    df = trades.set_index("timestamp").sort_index()
    df["signed_flow"] = df["quantity"] * df["side"]

    buckets = df.resample(freq).agg(
        net_flow=("signed_flow", "sum"),
        total_volume=("quantity", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        n_trades=("price", "count"),
    )

    buckets["price_change"] = buckets["last_price"] - buckets["first_price"]
    buckets["return"] = buckets["price_change"] / buckets["first_price"]
    buckets["ofi"] = buckets["net_flow"] / buckets["total_volume"].replace(0, np.nan)

    buckets = buckets.dropna()
    buckets = buckets[buckets["n_trades"] > 0]
    return buckets.reset_index()


def run_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Run log-log regression: log(|return|) = alpha * log(|flow|) + c."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 20:
        return {"alpha": float("nan"), "r2": float("nan"),
                "se": float("nan"), "n_obs": int(len(x))}

    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    return {
        "alpha": round(float(slope), 6),
        "r2": round(float(r_value ** 2), 6),
        "se": round(float(std_err), 6),
        "n_obs": int(len(x)),
        "p_value": float(p_value),
        "intercept": round(float(intercept), 6),
    }


def estimate_alpha_for_freq_from_buckets(buckets: pd.DataFrame) -> dict:
    """Estimate alpha from pre-computed buckets."""
    abs_flow = buckets["net_flow"].abs().values
    abs_return = buckets["return"].abs().values
    return run_regression(abs_flow, abs_return)


def main():
    data_path = PROJECT_ROOT / "data"
    csv_files = sorted(data_path.glob("*-aggTrades-*.csv"))
    n_files = len(csv_files)

    if n_files == 0:
        print("ERROR: No aggTrades CSV files found in data/")
        sys.exit(1)

    print("=" * 70)
    print(f"FULL ALPHA ESTIMATION — {n_files}-day dataset")
    print("=" * 70)
    print(f"\nProcessing {n_files} CSV files day-by-day...")

    t_start = time.time()

    # Accumulate bucketed data across all days for pooled regression
    all_buckets_1min = []
    all_buckets_5min = []
    per_day = []
    total_trades = 0
    daily_volumes = []
    all_dates = []

    for i, csv_file in enumerate(csv_files):
        day_label = csv_file.stem.split("aggTrades-")[-1]  # e.g. "2026-01-01"
        print(f"  [{i+1:3d}/{n_files}] {day_label} ...", end="", flush=True)

        trades = _load_single_csv(csv_file)
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        n_day = len(trades)
        total_trades += n_day
        daily_volumes.append(trades["quantity"].sum())
        all_dates.append(day_label)

        # Compute buckets for this day
        buckets_1 = aggregate_buckets(trades, "1min")
        buckets_5 = aggregate_buckets(trades, "5min")

        # Save buckets for pooled regression (much smaller than raw trades)
        all_buckets_1min.append(buckets_1[["net_flow", "return"]].copy())
        all_buckets_5min.append(buckets_5[["net_flow", "return"]].copy())

        # Per-day alpha estimation
        r_1min = estimate_alpha_for_freq_from_buckets(buckets_1)
        r_5min = estimate_alpha_for_freq_from_buckets(buckets_5)

        per_day.append({
            "date": day_label,
            "n_trades": n_day,
            "alpha_1min": r_1min["alpha"],
            "r2_1min": r_1min["r2"],
            "n_obs_1min": r_1min["n_obs"],
            "alpha_5min": r_5min["alpha"],
            "r2_5min": r_5min["r2"],
            "n_obs_5min": r_5min["n_obs"],
        })

        print(f"  {n_day:>10,} trades  a1={r_1min['alpha']:.3f}  a5={r_5min['alpha']:.3f}")

        # Free memory
        del trades, buckets_1, buckets_5

    elapsed = time.time() - t_start
    print(f"\n  Total: {total_trades:,} trades loaded in {elapsed:.1f}s")

    date_range = f"{all_dates[0]} to {all_dates[-1]}"
    n_days = len(all_dates)
    adv = np.mean(daily_volumes)
    print(f"  Days: {n_days} ({date_range})")
    print(f"  Average daily volume: {adv:,.2f} BTC")

    # ---- Pooled regressions from accumulated buckets ----
    print(f"\n{'=' * 70}")
    print("AGGREGATED REGRESSIONS (all days pooled)")
    print(f"{'=' * 70}")

    pooled_1min = pd.concat(all_buckets_1min, ignore_index=True)
    pooled_5min = pd.concat(all_buckets_5min, ignore_index=True)

    freq_results = {}
    for freq, pooled in [("1min", pooled_1min), ("5min", pooled_5min)]:
        r = estimate_alpha_for_freq_from_buckets(pooled)
        freq_results[freq] = r
        sig = ""
        if r.get("p_value") is not None:
            if r["p_value"] < 0.001:
                sig = "***"
            elif r["p_value"] < 0.01:
                sig = "**"
            elif r["p_value"] < 0.05:
                sig = "*"
        print(f"\n  {freq}:")
        print(f"    alpha  = {r['alpha']:.4f} +/- {r['se']:.4f}  {sig}")
        print(f"    R^2    = {r['r2']:.4f}")
        print(f"    n_obs  = {r['n_obs']:,}")

    # ---- Per-day stability ----
    print(f"\n{'=' * 70}")
    print("PER-DAY STABILITY ANALYSIS")
    print(f"{'=' * 70}")

    # Table header
    print(f"\n  {'Date':<12} {'alpha_1m':>9} {'R2_1m':>7} {'n_1m':>6}  "
          f"{'alpha_5m':>9} {'R2_5m':>7} {'n_5m':>6}  {'trades':>10}")
    print(f"  {'-' * 12} {'-' * 9} {'-' * 7} {'-' * 6}  "
          f"{'-' * 9} {'-' * 7} {'-' * 6}  {'-' * 10}")

    alphas_1 = []
    alphas_5 = []
    for row in per_day:
        a1 = row["alpha_1min"]
        a5 = row["alpha_5min"]
        alphas_1.append(a1)
        alphas_5.append(a5)
        a1_s = f"{a1:.4f}" if not np.isnan(a1) else "  NaN"
        a5_s = f"{a5:.4f}" if not np.isnan(a5) else "  NaN"
        r2_1_s = f"{row['r2_1min']:.4f}" if not np.isnan(row['r2_1min']) else " NaN"
        r2_5_s = f"{row['r2_5min']:.4f}" if not np.isnan(row['r2_5min']) else " NaN"
        print(f"  {row['date']:<12} {a1_s:>9} {r2_1_s:>7} {row['n_obs_1min']:>6}  "
              f"{a5_s:>9} {r2_5_s:>7} {row['n_obs_5min']:>6}  {row['n_trades']:>10,}")

    # Summary stats
    alphas_1_clean = [a for a in alphas_1 if not np.isnan(a)]
    alphas_5_clean = [a for a in alphas_5 if not np.isnan(a)]

    print(f"\n  Stability Summary:")
    if alphas_1_clean:
        print(f"    alpha_1min: mean={np.mean(alphas_1_clean):.4f}  "
              f"std={np.std(alphas_1_clean):.4f}  "
              f"range=[{np.min(alphas_1_clean):.4f}, {np.max(alphas_1_clean):.4f}]")
    if alphas_5_clean:
        print(f"    alpha_5min: mean={np.mean(alphas_5_clean):.4f}  "
              f"std={np.std(alphas_5_clean):.4f}  "
              f"range=[{np.min(alphas_5_clean):.4f}, {np.max(alphas_5_clean):.4f}]")

    # ---- Build output JSON ----
    results_clean = {}
    for freq, r in freq_results.items():
        results_clean[freq] = {
            "alpha": r["alpha"],
            "r2": r["r2"],
            "se": r["se"],
            "n_obs": r["n_obs"],
        }

    output = {
        "date_range": date_range,
        "n_days": n_days,
        "n_trades": total_trades,
        "adv_btc": round(float(adv), 2),
        "results": results_clean,
        "per_day": per_day,
        "stability": {
            "alpha_1min_mean": round(float(np.mean(alphas_1_clean)), 6) if alphas_1_clean else None,
            "alpha_1min_std": round(float(np.std(alphas_1_clean)), 6) if alphas_1_clean else None,
            "alpha_5min_mean": round(float(np.mean(alphas_5_clean)), 6) if alphas_5_clean else None,
            "alpha_5min_std": round(float(np.std(alphas_5_clean)), 6) if alphas_5_clean else None,
        },
    }

    out_path = data_path / "alpha_estimation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 70}")
    print(f"DONE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
