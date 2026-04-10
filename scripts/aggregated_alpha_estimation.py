"""Aggregated trade flow alpha estimation.

Instead of looking at individual trades (too noisy → alpha ≈ 0),
we aggregate trades into time buckets and regress:
    log(|price_change|) = alpha * log(|net_flow|) + const

This improves signal-to-noise by averaging out microstructure noise.
Literature expects alpha ∈ [0.5, 0.6] (square-root law of market impact).

References:
    Almgren et al. (2005) — "Direct Estimation of Equity Market Impact"
    Bouchaud et al. (2009) — "How Markets Slowly Digest Changes in Supply and Demand"
    Tóth et al. (2011) — "Anomalous Price Impact and the Critical Nature of Liquidity"
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from calibration.data_loader import load_trades


def aggregate_into_buckets(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate trades into time buckets.

    For each bucket, compute:
        - net_flow: sum(quantity * side)  — net signed order flow
        - price_change: last_price - first_price
        - n_trades: number of trades in the bucket
        - volume: sum(quantity)

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with columns: timestamp, price, quantity, side.
    freq : str
        Pandas frequency string (e.g., '1min', '5min', '15min').

    Returns
    -------
    pd.DataFrame
        Aggregated bucket data.
    """
    df = trades.set_index("timestamp").sort_index()

    # Signed flow per trade
    df["signed_flow"] = df["quantity"] * df["side"]

    # Aggregate
    buckets = df.resample(freq).agg(
        net_flow=("signed_flow", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        n_trades=("price", "count"),
        volume=("quantity", "sum"),
        vwap=("price", "mean"),  # simple average as proxy
    )

    # Price change over the bucket
    buckets["price_change"] = buckets["last_price"] - buckets["first_price"]

    # Drop empty buckets
    buckets = buckets.dropna()
    buckets = buckets[buckets["n_trades"] > 0]

    return buckets.reset_index()


def estimate_alpha_from_buckets(
    buckets: pd.DataFrame,
    min_obs: int = 30,
    use_relative: bool = True,
) -> dict:
    """Run log-log regression on aggregated buckets.

    Regresses:
        log(|price_change|) = alpha * log(|net_flow|) + const

    If use_relative=True, uses relative price change (|dp/p|) instead
    of absolute, which is more robust to price level differences.

    Parameters
    ----------
    buckets : pd.DataFrame
        Output from aggregate_into_buckets().
    min_obs : int
        Minimum observations required.
    use_relative : bool
        If True, use |price_change / vwap| instead of |price_change|.

    Returns
    -------
    dict with keys: alpha, intercept, r_squared, n_obs, std_err, p_value
    """
    df = buckets.copy()

    # Take absolute values
    df["abs_net_flow"] = df["net_flow"].abs()
    if use_relative:
        df["abs_price_change"] = (df["price_change"] / df["vwap"]).abs()
    else:
        df["abs_price_change"] = df["price_change"].abs()

    # Filter: both must be positive for log-log
    mask = (df["abs_net_flow"] > 0) & (df["abs_price_change"] > 0)
    df = df[mask]

    if len(df) < min_obs:
        return {
            "alpha": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "n_obs": len(df),
            "std_err": np.nan,
            "p_value": np.nan,
            "error": f"Only {len(df)} valid obs (need >= {min_obs})",
        }

    log_flow = np.log(df["abs_net_flow"].values)
    log_impact = np.log(df["abs_price_change"].values)

    # OLS via scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_flow, log_impact)

    return {
        "alpha": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "n_obs": len(df),
        "std_err": std_err,
        "p_value": p_value,
    }


def estimate_alpha_binned(
    buckets: pd.DataFrame,
    n_bins: int = 50,
    use_relative: bool = True,
) -> dict:
    """Binned regression: group by net_flow quantile, average, then regress.

    This further reduces noise by averaging within flow-size bins.
    More robust to outliers than raw observation regression.
    """
    df = buckets.copy()
    df["abs_net_flow"] = df["net_flow"].abs()
    if use_relative:
        df["abs_price_change"] = (df["price_change"] / df["vwap"]).abs()
    else:
        df["abs_price_change"] = df["price_change"].abs()

    mask = (df["abs_net_flow"] > 0) & (df["abs_price_change"] > 0)
    df = df[mask]

    if len(df) < n_bins:
        return {
            "alpha": np.nan,
            "r_squared": np.nan,
            "n_bins_used": 0,
            "error": f"Only {len(df)} obs, need >= {n_bins}",
        }

    # Bin by net flow quantile
    df["flow_bin"] = pd.qcut(df["abs_net_flow"], q=n_bins, duplicates="drop")

    binned = df.groupby("flow_bin", observed=True).agg(
        avg_flow=("abs_net_flow", "mean"),
        avg_impact=("abs_price_change", "mean"),
        count=("abs_net_flow", "size"),
    )
    binned = binned[binned["count"] >= 3]  # at least 3 per bin

    if len(binned) < 5:
        return {
            "alpha": np.nan,
            "r_squared": np.nan,
            "n_bins_used": len(binned),
            "error": "Too few valid bins",
        }

    log_flow = np.log(binned["avg_flow"].values)
    log_impact = np.log(binned["avg_impact"].values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_flow, log_impact)

    return {
        "alpha": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "n_bins_used": len(binned),
        "std_err": std_err,
        "p_value": p_value,
    }


def run_current_method(trades: pd.DataFrame) -> dict:
    """Run the existing estimate_temporary_impact_from_trades for comparison."""
    from calibration.impact_estimator import estimate_temporary_impact_from_trades

    try:
        eta, alpha = estimate_temporary_impact_from_trades(trades, n_buckets=20)
        return {"alpha": alpha, "eta": eta, "method": "current (single-trade)"}
    except Exception as e:
        return {"alpha": np.nan, "eta": np.nan, "error": str(e), "method": "current (single-trade)"}


def main():
    print("=" * 80)
    print("AGGREGATED TRADE FLOW — ALPHA ESTIMATION")
    print("=" * 80)

    # Load data
    data_path = PROJECT_ROOT / "data"
    print(f"\nLoading trades from {data_path} ...")
    trades = load_trades(data_path)
    print(f"  Loaded {len(trades):,} trades")
    print(f"  Time range: {trades['timestamp'].min()} → {trades['timestamp'].max()}")
    print(f"  Price range: ${trades['price'].min():,.2f} — ${trades['price'].max():,.2f}")
    print(f"  Total volume: {trades['quantity'].sum():,.4f} BTC")

    # Current method baseline
    print("\n" + "-" * 80)
    print("BASELINE: Current single-trade method")
    print("-" * 80)
    baseline = run_current_method(trades)
    print(f"  alpha = {baseline['alpha']:.6f}")
    if "eta" in baseline:
        print(f"  eta   = {baseline.get('eta', 'N/A')}")
    if "error" in baseline:
        print(f"  ERROR: {baseline['error']}")

    # Aggregated method — multiple time windows
    freqs = ["1min", "5min", "15min", "30min", "60min"]

    print("\n" + "=" * 80)
    print("AGGREGATED METHOD: Time-bucketed net flow regression")
    print("=" * 80)

    for freq in freqs:
        print(f"\n{'─' * 60}")
        print(f"Time window: {freq}")
        print(f"{'─' * 60}")

        buckets = aggregate_into_buckets(trades, freq)
        total_buckets = len(buckets)
        nonzero_flow = (buckets["net_flow"].abs() > 0).sum()
        nonzero_dp = (buckets["price_change"].abs() > 0).sum()
        both = ((buckets["net_flow"].abs() > 0) & (buckets["price_change"].abs() > 0)).sum()

        print(f"  Total buckets:       {total_buckets:,}")
        print(f"  Nonzero net_flow:    {nonzero_flow:,}")
        print(f"  Nonzero price_chg:   {nonzero_dp:,}")
        print(f"  Both nonzero:        {both:,}")

        # Summary stats
        abs_flow = buckets["net_flow"].abs()
        abs_dp = buckets["price_change"].abs()
        print(f"  Median |net_flow|:   {abs_flow.median():.6f} BTC")
        print(f"  Median |price_chg|:  ${abs_dp.median():.2f}")

        # Method A: Raw observation regression
        result_raw = estimate_alpha_from_buckets(buckets, use_relative=True)
        print(f"\n  [A] Raw log-log regression (relative price change):")
        if "error" in result_raw:
            print(f"      ERROR: {result_raw['error']}")
        else:
            print(f"      alpha  = {result_raw['alpha']:.4f} ± {result_raw['std_err']:.4f}")
            print(f"      R²     = {result_raw['r_squared']:.4f}")
            print(f"      n_obs  = {result_raw['n_obs']:,}")
            print(f"      p-val  = {result_raw['p_value']:.2e}")

        # Method B: Binned regression
        n_bins = min(50, total_buckets // 10)
        if n_bins >= 10:
            result_binned = estimate_alpha_binned(buckets, n_bins=n_bins, use_relative=True)
            print(f"\n  [B] Binned log-log regression ({n_bins} bins, relative):")
            if "error" in result_binned:
                print(f"      ERROR: {result_binned['error']}")
            else:
                print(f"      alpha  = {result_binned['alpha']:.4f} ± {result_binned['std_err']:.4f}")
                print(f"      R²     = {result_binned['r_squared']:.4f}")
                print(f"      n_bins = {result_binned['n_bins_used']}")
        else:
            print(f"\n  [B] Binned regression: skipped (not enough buckets for {n_bins} bins)")

    # Per-day analysis for stability check
    print("\n" + "=" * 80)
    print("PER-DAY STABILITY CHECK (5min buckets)")
    print("=" * 80)

    trades["date"] = trades["timestamp"].dt.date
    for date, day_trades in trades.groupby("date"):
        day_trades = day_trades.drop(columns=["date"])
        buckets = aggregate_into_buckets(day_trades, "5min")
        result = estimate_alpha_from_buckets(buckets, use_relative=True)
        if "error" not in result:
            print(f"  {date}:  alpha = {result['alpha']:.4f} ± {result['std_err']:.4f},  "
                  f"R² = {result['r_squared']:.4f},  n = {result['n_obs']}")
        else:
            print(f"  {date}:  {result['error']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Literature reference:       alpha ∈ [0.5, 0.6] (square-root law)")
    print(f"  Current single-trade method: alpha = {baseline['alpha']:.4f} (too noisy)")
    print(f"\n  Aggregated method results (raw regression, relative price change):")
    for freq in freqs:
        buckets = aggregate_into_buckets(trades, freq)
        result = estimate_alpha_from_buckets(buckets, use_relative=True)
        if "error" not in result:
            quality = ""
            a = result["alpha"]
            if 0.4 <= a <= 0.7:
                quality = "  ✓ in literature range"
            elif 0.3 <= a <= 0.8:
                quality = "  ~ close to literature"
            else:
                quality = "  ✗ outside expected range"
            print(f"    {freq:>5s}:  alpha = {a:.4f},  R² = {result['r_squared']:.4f}{quality}")
        else:
            print(f"    {freq:>5s}:  {result.get('error', 'failed')}")


if __name__ == "__main__":
    main()
