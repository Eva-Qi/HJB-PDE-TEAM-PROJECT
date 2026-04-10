"""Aggregated alpha estimation — v2 with improved metrics.

The v1 approach gives alpha ≈ 0.2-0.3, still below literature 0.5-0.6.
This version tries additional refinements:

1. Order flow imbalance (OFI): net_flow / total_volume, normalized
2. Volume-normalized impact: |dp/p| per unit of volume
3. Quantile trimming: remove extreme outliers
4. Sign-matched regression: only use buckets where sign(flow) == sign(dp)
5. ADV-normalized flow: flow / average_daily_volume
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from calibration.data_loader import load_trades


def aggregate_buckets(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate trades into time buckets with multiple flow metrics."""
    df = trades.set_index("timestamp").sort_index()
    df["signed_flow"] = df["quantity"] * df["side"]

    buckets = df.resample(freq).agg(
        net_flow=("signed_flow", "sum"),
        total_volume=("quantity", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        n_trades=("price", "count"),
        mean_price=("price", "mean"),
    )

    buckets["price_change"] = buckets["last_price"] - buckets["first_price"]
    buckets["return"] = buckets["price_change"] / buckets["first_price"]

    # Order flow imbalance = net_flow / total_volume ∈ [-1, 1]
    buckets["ofi"] = buckets["net_flow"] / buckets["total_volume"].replace(0, np.nan)

    buckets = buckets.dropna()
    buckets = buckets[buckets["n_trades"] > 0]
    return buckets.reset_index()


def run_regression(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Run log-log regression with diagnostics."""
    # Filter positive only
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 20:
        return {"alpha": np.nan, "r_squared": np.nan, "n": len(x), "label": label, "error": "too few obs"}

    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    return {
        "alpha": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "n": len(x),
        "std_err": std_err,
        "p_value": p_value,
        "label": label,
    }


def analyze_freq(trades: pd.DataFrame, freq: str, adv: float):
    """Run all analysis variants for a given frequency."""
    buckets = aggregate_buckets(trades, freq)
    n = len(buckets)

    print(f"\n{'=' * 70}")
    print(f"FREQUENCY: {freq}  ({n} buckets)")
    print(f"{'=' * 70}")

    # --- Approach 1: Raw net_flow vs |return| ---
    abs_flow = buckets["net_flow"].abs().values
    abs_return = buckets["return"].abs().values
    r1 = run_regression(abs_flow, abs_return, "net_flow vs |return|")
    print_result(r1)

    # --- Approach 2: OFI-based ---
    # Use |OFI| (normalized) as x, |return| as y
    abs_ofi = buckets["ofi"].abs().values
    r2 = run_regression(abs_ofi, abs_return, "|OFI| vs |return|")
    print_result(r2)

    # --- Approach 3: ADV-normalized flow ---
    # x = |net_flow| / ADV (makes flow dimensionless)
    norm_flow = abs_flow / adv
    r3 = run_regression(norm_flow, abs_return, "ADV-normalized flow vs |return|")
    print_result(r3)

    # --- Approach 4: Sign-matched only ---
    # Only keep buckets where flow direction matches price direction
    sign_match = np.sign(buckets["net_flow"].values) == np.sign(buckets["price_change"].values)
    matched = buckets[sign_match]
    r4 = run_regression(
        matched["net_flow"].abs().values,
        matched["return"].abs().values,
        f"Sign-matched flow vs |return| ({sign_match.sum()}/{n} matched)"
    )
    print_result(r4)

    # --- Approach 5: Trimmed (5th-95th percentile on flow) ---
    flow_q5 = np.percentile(abs_flow[abs_flow > 0], 5)
    flow_q95 = np.percentile(abs_flow[abs_flow > 0], 95)
    trim_mask = (abs_flow >= flow_q5) & (abs_flow <= flow_q95) & (abs_return > 0)
    r5 = run_regression(
        abs_flow[trim_mask],
        abs_return[trim_mask],
        f"Trimmed [5,95] percentile ({trim_mask.sum()} obs)"
    )
    print_result(r5)

    # --- Approach 6: Binned median regression ---
    df_valid = buckets[(buckets["net_flow"].abs() > 0) & (buckets["return"].abs() > 0)].copy()
    if len(df_valid) >= 50:
        n_bins = min(30, len(df_valid) // 5)
        df_valid["flow_bin"] = pd.qcut(df_valid["net_flow"].abs(), q=n_bins, duplicates="drop")
        binned = df_valid.groupby("flow_bin", observed=True).agg(
            median_flow=("net_flow", lambda x: x.abs().median()),
            median_return=("return", lambda x: x.abs().median()),
            count=("return", "size"),
        )
        binned = binned[binned["count"] >= 3]
        r6 = run_regression(
            binned["median_flow"].values,
            binned["median_return"].values,
            f"Binned median ({len(binned)} bins)"
        )
        print_result(r6)
    else:
        print(f"  [Binned median]: Skipped (too few obs)")

    return buckets


def print_result(r: dict):
    """Print a regression result."""
    label = r["label"]
    if "error" in r:
        print(f"  [{label}]:  {r['error']}")
        return
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
    quality = ""
    a = r["alpha"]
    if 0.4 <= a <= 0.7:
        quality = " <-- IN RANGE"
    elif 0.3 <= a <= 0.8:
        quality = " <-- close"
    print(f"  [{label}]:")
    print(f"    alpha = {a:.4f} ± {r['std_err']:.4f}  R² = {r['r_squared']:.4f}  "
          f"n = {r['n']}  {sig}{quality}")


def main():
    print("AGGREGATED ALPHA ESTIMATION — v2")
    print("Literature target: alpha ∈ [0.5, 0.6]")
    print()

    data_path = PROJECT_ROOT / "data"
    trades = load_trades(data_path)
    print(f"Loaded {len(trades):,} trades")
    print(f"Time: {trades['timestamp'].min()} → {trades['timestamp'].max()}")

    # Compute ADV
    trades_tmp = trades.copy()
    trades_tmp["date"] = trades_tmp["timestamp"].dt.date
    daily_vol = trades_tmp.groupby("date")["quantity"].sum()
    adv = daily_vol.mean()
    print(f"Average daily volume: {adv:,.2f} BTC")

    for freq in ["1min", "5min", "15min", "30min"]:
        analyze_freq(trades, freq, adv)

    # Additional: try sqrt(volume) as alternative x-variable
    # Theory: impact ~ flow^alpha, but some papers use sqrt(volume) directly
    print(f"\n{'=' * 70}")
    print("ALTERNATIVE: sqrt(total_volume) as regressor")
    print(f"{'=' * 70}")
    for freq in ["1min", "5min", "15min"]:
        buckets = aggregate_buckets(trades, freq)
        sqrt_vol = np.sqrt(buckets["total_volume"].values)
        abs_return = buckets["return"].abs().values
        r = run_regression(sqrt_vol, abs_return, f"{freq} sqrt(volume) vs |return|")
        print_result(r)

    # Correlation analysis: how much does net_flow explain?
    print(f"\n{'=' * 70}")
    print("CORRELATION ANALYSIS (5min buckets)")
    print(f"{'=' * 70}")
    buckets = aggregate_buckets(trades, "5min")
    valid = buckets[(buckets["net_flow"].abs() > 0) & (buckets["return"].abs() > 0)]

    flow = valid["net_flow"].values
    ret = valid["return"].values
    print(f"  Corr(net_flow, return):          {np.corrcoef(flow, ret)[0,1]:.4f}")
    print(f"  Corr(|net_flow|, |return|):      {np.corrcoef(np.abs(flow), np.abs(ret))[0,1]:.4f}")
    print(f"  Corr(sign(flow), sign(return)):  {np.corrcoef(np.sign(flow), np.sign(ret))[0,1]:.4f}")
    ofi = valid["ofi"].values
    print(f"  Corr(OFI, return):               {np.corrcoef(ofi, ret)[0,1]:.4f}")
    print(f"  Corr(|OFI|, |return|):           {np.corrcoef(np.abs(ofi), np.abs(ret))[0,1]:.4f}")

    # Fraction of buckets where sign matches
    sign_match_rate = (np.sign(flow) == np.sign(ret)).mean()
    print(f"  Sign agreement rate:             {sign_match_rate:.4f} ({sign_match_rate*100:.1f}%)")

    print(f"\n{'=' * 70}")
    print("DISCUSSION")
    print(f"{'=' * 70}")
    print("""
  The aggregated approach improves alpha from ~0.007 (single-trade) to ~0.2-0.3.
  This is still below literature's 0.5-0.6. Possible reasons:

  1. aggTrades data aggregates multiple order book events into one record,
     so individual aggTrade quantities don't directly map to "orders" in the
     Almgren-Chriss framework.

  2. The square-root law (alpha≈0.5) typically refers to the impact of
     *meta-orders* (large parent orders split across time), not individual
     aggTrades or short-window flows.

  3. For the AC model calibration, alpha ≈ 0.2-0.3 from aggregated flow
     is a more realistic estimate than the 0.007 from single trades or
     the literature default of 0.5-0.6 (which applies to a different scale).

  RECOMMENDATION: Use alpha ≈ 0.3 (from 1-min aggregated regression,
  highest R² among raw regressions) as the calibrated value, or keep
  the literature fallback of 0.5 if you want to match the meta-order
  impact assumption in AC.
""")


if __name__ == "__main__":
    main()
