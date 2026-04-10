"""Order book depth alpha estimation — gold standard.

Directly measures market impact by walking through L2 order book snapshots.
For each test quantity q, we compute the VWAP from consuming ask-side
liquidity and derive slippage in basis points relative to the mid price.

Then we fit the power law:
    slippage_bps = eta * q^alpha
via log-log regression:
    log(slippage_bps) = alpha * log(q) + log(eta)

This is the most direct way to measure the price impact exponent alpha,
free of the aggregation assumptions in trade-flow regressions.

Data: Tardis.dev book_snapshot_25 (top 25 levels of Binance BTCUSDT L2 book)
"""

import json
import sys
import time
from pathlib import Path

# sys.path hack — add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARDIS_DIR = Path("/Users/evanolott/Desktop/QC_Trade_Platform/data/raw/tardis")
GLOB_PATTERN = "binance_book_snapshot_25_*_BTCUSDT.csv.gz"

# Test quantities in BTC
TEST_QUANTITIES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Sample every N-th row to keep runtime manageable
# book_snapshot_25 updates ~100ms → ~36,000 per hour → ~864,000 per day
# Sampling every 600 rows ≈ once per 60 seconds
SAMPLE_EVERY = 600

N_LEVELS = 25  # top 25 ask/bid levels in the snapshot


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_snapshot_file(filepath: Path, sample_every: int = SAMPLE_EVERY) -> pd.DataFrame:
    """Load a single book_snapshot_25 CSV, subsampled."""
    df = pd.read_csv(filepath)

    # Subsample
    if sample_every > 1:
        df = df.iloc[::sample_every].reset_index(drop=True)

    return df


def compute_slippage_for_snapshot(row, quantities: list[float]) -> dict[float, float]:
    """Walk the ask side of one snapshot to compute slippage for each q.

    Returns dict: {q: slippage_bps} for each quantity.
    """
    # Extract mid price
    best_ask = row["asks[0].price"]
    best_bid = row["bids[0].price"]
    mid = (best_ask + best_bid) / 2.0

    if mid <= 0 or np.isnan(mid):
        return {q: np.nan for q in quantities}

    # Build ask ladder: list of (price, amount)
    ask_prices = []
    ask_amounts = []
    for i in range(N_LEVELS):
        p = row[f"asks[{i}].price"]
        a = row[f"asks[{i}].amount"]
        if np.isnan(p) or np.isnan(a) or a <= 0:
            break
        ask_prices.append(p)
        ask_amounts.append(a)

    if len(ask_prices) == 0:
        return {q: np.nan for q in quantities}

    ask_prices = np.array(ask_prices)
    ask_amounts = np.array(ask_amounts)

    # Cumulative available quantity at each level
    cum_amount = np.cumsum(ask_amounts)
    total_available = cum_amount[-1]

    results = {}
    for q in quantities:
        if q <= 0:
            results[q] = 0.0
            continue

        if q > total_available:
            # Cannot fill — not enough depth
            results[q] = np.nan
            continue

        # Walk the book
        remaining = q
        cost = 0.0
        for j in range(len(ask_prices)):
            fill = min(remaining, ask_amounts[j])
            cost += fill * ask_prices[j]
            remaining -= fill
            if remaining <= 1e-12:
                break

        vwap = cost / q
        slippage_bps = (vwap - mid) / mid * 10_000.0
        results[q] = slippage_bps

    return results


def process_all_files(
    data_dir: Path,
    pattern: str,
    quantities: list[float],
    sample_every: int,
) -> tuple[dict[float, list[float]], int, int, list[str]]:
    """Process all snapshot files and collect slippage samples.

    Returns:
        slippage_samples: {q: [slippage_bps, ...]} across all snapshots
        total_snapshots: number of snapshots processed
        total_unfillable: count of snapshots where largest q couldn't be filled
        file_dates: list of date strings processed
    """
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {data_dir}")

    slippage_samples: dict[float, list[float]] = {q: [] for q in quantities}
    total_snapshots = 0
    total_unfillable = 0
    file_dates = []

    for fpath in files:
        # Extract date from filename
        fname = fpath.stem  # e.g. binance_book_snapshot_25_2026-03-24_BTCUSDT.csv
        parts = fname.split("_")
        date_str = parts[4]  # "2026-03-24"
        file_dates.append(date_str)

        print(f"  Loading {fpath.name} ...", end="", flush=True)
        t0 = time.time()

        df = load_snapshot_file(fpath, sample_every=sample_every)
        n_rows = len(df)

        for idx in range(n_rows):
            row = df.iloc[idx]
            slip = compute_slippage_for_snapshot(row, quantities)

            any_valid = False
            for q in quantities:
                s = slip[q]
                if not np.isnan(s):
                    slippage_samples[q].append(s)
                    any_valid = True

            if any_valid:
                total_snapshots += 1

            # Track unfillable for largest q
            if np.isnan(slip[quantities[-1]]):
                total_unfillable += 1

        elapsed = time.time() - t0
        print(f"  {n_rows:,} snapshots in {elapsed:.1f}s")

    return slippage_samples, total_snapshots, total_unfillable, file_dates


def fit_power_law(
    quantities: list[float],
    mean_slippages: list[float],
) -> dict:
    """Fit log(slippage) = alpha * log(q) + log(eta) via OLS.

    Returns dict with alpha, eta_bps, r_squared, se_alpha, etc.
    """
    q_arr = np.array(quantities)
    s_arr = np.array(mean_slippages)

    # Filter out NaN/zero/negative
    mask = (q_arr > 0) & (s_arr > 0) & np.isfinite(s_arr)
    q_valid = q_arr[mask]
    s_valid = s_arr[mask]

    if len(q_valid) < 3:
        return {
            "alpha": float("nan"),
            "eta_bps": float("nan"),
            "r_squared": float("nan"),
            "error": f"Only {len(q_valid)} valid points for regression",
        }

    log_q = np.log(q_valid)
    log_s = np.log(s_valid)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_q, log_s)

    return {
        "alpha": round(float(slope), 6),
        "eta_bps": round(float(np.exp(intercept)), 6),
        "r_squared": round(float(r_value ** 2), 6),
        "se_alpha": round(float(std_err), 6),
        "p_value": float(p_value),
        "intercept": round(float(intercept), 6),
        "n_points": int(len(q_valid)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("ORDER BOOK DEPTH — ALPHA ESTIMATION (Gold Standard)")
    print("=" * 75)
    print(f"\nData source: {TARDIS_DIR}")
    print(f"Pattern:     {GLOB_PATTERN}")
    print(f"Sampling:    every {SAMPLE_EVERY} rows (~1 snapshot per minute)")
    print(f"Quantities:  {TEST_QUANTITIES} BTC")

    t_global = time.time()

    # ---- Process all files ----
    print(f"\n{'─' * 75}")
    print("LOADING & COMPUTING SLIPPAGE")
    print(f"{'─' * 75}\n")

    slippage_samples, total_snapshots, total_unfillable, file_dates = process_all_files(
        TARDIS_DIR, GLOB_PATTERN, TEST_QUANTITIES, SAMPLE_EVERY
    )

    print(f"\n  Total snapshots processed: {total_snapshots:,}")
    print(f"  Snapshots where q={TEST_QUANTITIES[-1]} BTC unfillable: {total_unfillable:,}")
    print(f"  Date range: {file_dates[0]} to {file_dates[-1]} ({len(file_dates)} days)")

    # ---- Compute summary statistics ----
    print(f"\n{'─' * 75}")
    print("SLIPPAGE TABLE (ask side, basis points)")
    print(f"{'─' * 75}\n")

    header = f"  {'q (BTC)':>10}  {'mean':>9}  {'median':>9}  {'std':>9}  {'p25':>9}  {'p75':>9}  {'n_obs':>8}"
    print(header)
    print(f"  {'─' * 10}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 8}")

    mean_slippages = []
    table_rows = []

    for q in TEST_QUANTITIES:
        samples = np.array(slippage_samples[q])
        n_obs = len(samples)

        if n_obs == 0:
            mean_slippages.append(np.nan)
            table_rows.append({
                "q_btc": q, "mean_bps": None, "median_bps": None,
                "std_bps": None, "p25_bps": None, "p75_bps": None,
                "n_obs": 0,
            })
            print(f"  {q:>10.3f}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {0:>8}")
            continue

        m = float(np.mean(samples))
        med = float(np.median(samples))
        sd = float(np.std(samples))
        p25 = float(np.percentile(samples, 25))
        p75 = float(np.percentile(samples, 75))

        mean_slippages.append(m)
        table_rows.append({
            "q_btc": q,
            "mean_bps": round(m, 4),
            "median_bps": round(med, 4),
            "std_bps": round(sd, 4),
            "p25_bps": round(p25, 4),
            "p75_bps": round(p75, 4),
            "n_obs": n_obs,
        })
        print(f"  {q:>10.3f}  {m:>9.4f}  {med:>9.4f}  {sd:>9.4f}  {p25:>9.4f}  {p75:>9.4f}  {n_obs:>8,}")

    # ---- Power law fit ----
    print(f"\n{'─' * 75}")
    print("POWER LAW FIT:  slippage_bps = eta * q^alpha")
    print(f"{'─' * 75}\n")

    fit = fit_power_law(TEST_QUANTITIES, mean_slippages)

    if "error" in fit:
        print(f"  ERROR: {fit['error']}")
    else:
        print(f"  alpha    = {fit['alpha']:.4f} +/- {fit['se_alpha']:.4f}")
        print(f"  eta      = {fit['eta_bps']:.4f} bps")
        print(f"  R^2      = {fit['r_squared']:.4f}")
        print(f"  p-value  = {fit['p_value']:.2e}")
        print(f"  n_points = {fit['n_points']}")

    # ---- Bid-side symmetry check (optional — quick) ----
    # We already computed ask-side. For completeness, note that bid side
    # should be symmetric for a well-functioning market.

    # ---- Comparison ----
    print(f"\n{'─' * 75}")
    print("COMPARISON WITH OTHER ESTIMATES")
    print(f"{'─' * 75}\n")

    agg_alpha = 0.34
    lit_low, lit_high = 0.5, 0.6
    ob_alpha = fit.get("alpha", float("nan"))

    print(f"  Order book depth (this script):  alpha = {ob_alpha:.4f}")
    print(f"  Aggregated flow (1-min buckets):  alpha ~ {agg_alpha:.2f}")
    print(f"  Literature (square-root law):     alpha ~ {lit_low:.1f} - {lit_high:.1f}")
    print()

    if not np.isnan(ob_alpha):
        if lit_low <= ob_alpha <= lit_high:
            print("  --> OB alpha is WITHIN the literature range [0.5, 0.6]")
        elif 0.3 <= ob_alpha < lit_low:
            print("  --> OB alpha is BELOW literature but above aggregated flow")
        elif ob_alpha > lit_high:
            print("  --> OB alpha is ABOVE the literature range")
        else:
            print(f"  --> OB alpha = {ob_alpha:.4f} is outside typical ranges")

        if ob_alpha > agg_alpha:
            ratio = ob_alpha / agg_alpha
            print(f"  --> OB alpha is {ratio:.2f}x the aggregated flow alpha")

    # ---- Save results ----
    elapsed_total = time.time() - t_global

    output = {
        "method": "order_book_depth_walking",
        "data_source": "tardis_book_snapshot_25_binance_BTCUSDT",
        "date_range": f"{file_dates[0]} to {file_dates[-1]}",
        "n_days": len(file_dates),
        "n_snapshots": total_snapshots,
        "sample_interval_approx": "~60 seconds",
        "n_levels": N_LEVELS,
        "fit": fit,
        "slippage_table": table_rows,
        "comparison": {
            "ob_alpha": fit.get("alpha"),
            "aggregated_flow_alpha_1min": agg_alpha,
            "literature_range": [lit_low, lit_high],
        },
        "runtime_seconds": round(elapsed_total, 1),
    }

    out_path = PROJECT_ROOT / "data" / "orderbook_alpha_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    print(f"\n{'=' * 75}")
    print(f"DONE ({elapsed_total:.1f}s)")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
