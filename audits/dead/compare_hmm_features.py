"""Compare univariate (log_return only) vs bivariate (log_return + F&G) HMM.

Uses real Binance aggTrades for 5-min returns and alternative.me Fear & Greed
Index as the sentiment feature (forward-filled daily → 5-min grid).

Purpose: test the hypothesis that log-return-only HMM finds weak regimes
(~8% sigma multiplier spread, p=0.84 in paired test) because 2026-Q1 BTC
had little return-variance regime structure, and that adding sentiment
amplifies separation enough to be statistically meaningful.

Expected if hypothesis holds:
    Multi-feature HMM finds regimes with >20% sigma multiplier spread,
    cleanly tagging "extreme fear" days as risk_off.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from calibration.data_loader import load_trades, compute_ohlc
from extensions.regime import (
    fit_hmm,
    align_sentiment_to_returns_bar,
)


PROJECT = Path(__file__).resolve().parent.parent


def load_fear_greed() -> pd.DataFrame:
    """Load F&G JSON and return as DataFrame with 'date' and 'sentiment' cols."""
    with open(PROJECT / "data" / "fear_greed_btc.json") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"])  # tz-naive
    # Rename F&G 'value' to 'sentiment' so align_sentiment helper works out of box
    df = df.rename(columns={"value": "sentiment"})
    return df[["date", "sentiment"]]


def build_returns_df(start: str, end: str) -> pd.DataFrame:
    """Load aggTrades, compute 5-min OHLCV, return DataFrame with
    timestamp + log_return columns."""
    trades = load_trades("data/", start=start, end=end)
    ohlc = compute_ohlc(trades, freq="5min")
    # compute_ohlc yields columns including timestamp and close
    ohlc = ohlc.copy()
    ohlc["log_return"] = np.log(ohlc["close"] / ohlc["close"].shift(1))
    ohlc = ohlc.dropna(subset=["log_return"])
    return ohlc[["timestamp", "log_return"]]


def summarize_regimes(regimes, label: str) -> dict:
    """Extract the metrics we care about from fit_hmm output."""
    # regimes is list of RegimeParams, ordered with risk_on at index 0
    on, off = regimes[0], regimes[1]
    spread = abs(off.sigma - on.sigma) / max(on.sigma, 1e-12)
    return {
        "label": label,
        "sigma_on": float(on.sigma),
        "sigma_off": float(off.sigma),
        "sigma_spread_pct": 100.0 * float(spread),
        "prob_on": float(on.probability),
        "prob_off": float(off.probability),
        "state_vol_on": float(on.state_vol),
        "state_vol_off": float(off.state_vol),
    }


def main() -> None:
    print("=" * 78)
    print("Univariate vs Bivariate HMM comparison (log_return vs +F&G sentiment)")
    print("=" * 78)

    # 1. Load full 98 days of BTCUSDT returns for maximum regime variation
    print("\n[1/4] Loading Binance aggTrades → 5-min log returns...")
    rdf = build_returns_df(start="2026-01-01", end="2026-04-08")
    print(f"      {len(rdf):,} five-min bars, "
          f"{rdf['timestamp'].min()} → {rdf['timestamp'].max()}")

    # 2. Load F&G and align
    print("\n[2/4] Loading Fear & Greed Index and aligning to 5-min grid...")
    sdf = load_fear_greed()
    features_2d = align_sentiment_to_returns_bar(
        rdf, sdf, returns_col="log_return", sentiment_col="sentiment",
    )
    print(f"      Aligned feature matrix shape: {features_2d.shape}")
    print(f"      Col 0 (log_return): mean={features_2d[:,0].mean():.6f}, "
          f"std={features_2d[:,0].std():.6f}")
    print(f"      Col 1 (sentiment):  mean={features_2d[:,1].mean():.2f}, "
          f"std={features_2d[:,1].std():.2f}, "
          f"range=[{features_2d[:,1].min():.0f}, {features_2d[:,1].max():.0f}]")

    # 3. Fit univariate HMM (log_return only)
    print("\n[3/4] Fitting univariate HMM on log_return only...")
    regimes_uni, _ = fit_hmm(features_2d[:, 0])
    uni = summarize_regimes(regimes_uni, "univariate (log_return)")
    print(f"      sigma_on={uni['sigma_on']:.4f}, sigma_off={uni['sigma_off']:.4f}, "
          f"spread={uni['sigma_spread_pct']:.1f}%")
    print(f"      prob_on={uni['prob_on']:.3f}, prob_off={uni['prob_off']:.3f}")

    # 4. Fit bivariate HMM (log_return + sentiment)
    print("\n[4/4] Fitting bivariate HMM on (log_return, sentiment)...")
    regimes_bi, _ = fit_hmm(features=features_2d)
    bi = summarize_regimes(regimes_bi, "bivariate (+ sentiment)")
    print(f"      sigma_on={bi['sigma_on']:.4f}, sigma_off={bi['sigma_off']:.4f}, "
          f"spread={bi['sigma_spread_pct']:.1f}%")
    print(f"      prob_on={bi['prob_on']:.3f}, prob_off={bi['prob_off']:.3f}")
    print(f"      state_vol_on={bi['state_vol_on']:.6f}, "
          f"state_vol_off={bi['state_vol_off']:.6f}")

    # Comparison table
    print("\n" + "=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print(f"{'metric':<25} {'univariate':>15} {'bivariate':>15} {'improvement':>15}")
    print("-" * 78)
    print(f"{'sigma_on':<25} {uni['sigma_on']:>15.4f} {bi['sigma_on']:>15.4f}")
    print(f"{'sigma_off':<25} {uni['sigma_off']:>15.4f} {bi['sigma_off']:>15.4f}")
    spread_ratio = (
        bi['sigma_spread_pct'] / uni['sigma_spread_pct']
        if uni['sigma_spread_pct'] > 0 else float("inf")
    )
    print(f"{'sigma_spread_pct':<25} {uni['sigma_spread_pct']:>14.1f}% "
          f"{bi['sigma_spread_pct']:>14.1f}% {spread_ratio:>14.2f}x")
    print(f"{'prob_off (stationary)':<25} {uni['prob_off']:>15.3f} {bi['prob_off']:>15.3f}")
    print("-" * 78)

    # Interpretation
    print()
    if bi['sigma_spread_pct'] > 2.0 * uni['sigma_spread_pct']:
        print("✓ Bivariate HMM finds MUCH stronger regime separation — sentiment "
              "amplifies structure that log_return alone cannot detect. "
              "Re-running paired test is likely to reject H_0 now.")
    elif bi['sigma_spread_pct'] > 1.2 * uni['sigma_spread_pct']:
        print("~ Modest improvement — bivariate helps but sentiment alone doesn't "
              "fully resolve the regime-awareness narrative.")
    else:
        print("✗ No meaningful improvement — bivariate HMM does not find stronger "
              "regimes than univariate. The hypothesis that sentiment amplifies "
              "regime separation is NOT supported in this window.")

    # Save results
    out = PROJECT / "data" / "hmm_feature_comparison.json"
    out.write_text(json.dumps(
        {"univariate": uni, "bivariate": bi, "spread_ratio": spread_ratio},
        indent=2,
    ))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
