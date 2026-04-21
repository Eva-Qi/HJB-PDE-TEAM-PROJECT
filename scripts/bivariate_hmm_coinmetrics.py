"""Bivariate HMM experiment: log_return + CoinMetrics exchange FlowIn.

Hypothesis: Exchange inflow (FlowInExUSD) is more microstructure-relevant
than sentiment (Fear & Greed Index), so it might amplify regime separation
where F&G diluted it.

Earlier F&G result (commit 1c95911):
    univariate (log_return):      sigma_on=0.674  sigma_off=2.357  spread=250%
    bivariate (log_return + F&G): sigma_on=0.776  sigma_off=1.123  spread=45%
    Conclusion: F&G DILUTED the signal.

This script tests three variants:
    1. Univariate: log_return only          (baseline)
    2. Bivariate:  log_return + FlowInExUSD (raw USD, ~4 OOM range)
    3. Bivariate:  log_return + log(FlowInExUSD) (log-scale, more HMM-friendly)
    4. BONUS 3-feature: log_return + FlowInExUSD + FlowOutExUSD (net-flow proxy)

Interpretation rules (based on spread ratio vs univariate):
    spread_ratio > 1.2x → FlowIn AMPLIFIES regime separation
    spread_ratio > 0.9x → FlowIn doesn't dilute (valid alternative feature)
    spread_ratio < 0.9x → FlowIn dilutes same as F&G did
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

# ── Reference result from the F&G experiment ────────────────────────────────
FG_REFERENCE = {
    "univariate_spread_pct": 250.0,
    "bivariate_fg_spread_pct": 45.0,
}


def load_coinmetrics() -> pd.DataFrame:
    """Load CoinMetrics on-chain data; return DataFrame with date + all columns."""
    with open(PROJECT / "data" / "coinmetrics_btc_onchain.json") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_returns_df(start: str, end: str) -> pd.DataFrame:
    """Load aggTrades, compute 5-min OHLCV, return DataFrame with
    timestamp + log_return columns."""
    trades = load_trades("data/", start=start, end=end)
    ohlc = compute_ohlc(trades, freq="5min")
    ohlc = ohlc.copy()
    ohlc["log_return"] = np.log(ohlc["close"] / ohlc["close"].shift(1))
    ohlc = ohlc.dropna(subset=["log_return"])
    return ohlc[["timestamp", "log_return"]]


def summarize_regimes(regimes, label: str) -> dict:
    """Extract the metrics we care about from fit_hmm output."""
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


def interpret_result(spread_pct: float, uni_spread_pct: float, feature_name: str) -> str:
    """Programmatic interpretation of spread ratio."""
    ratio = spread_pct / uni_spread_pct if uni_spread_pct > 0 else float("inf")
    if ratio > 1.2:
        return (
            f"{feature_name} AMPLIFIES regime separation beyond log_return alone "
            f"(spread={spread_pct:.1f}% vs univariate {uni_spread_pct:.1f}%, ratio={ratio:.2f}x) "
            f"— valuable additional feature."
        )
    elif ratio > 0.9:
        return (
            f"{feature_name} doesn't dilute the regime signal "
            f"(spread={spread_pct:.1f}% vs univariate {uni_spread_pct:.1f}%, ratio={ratio:.2f}x) "
            f"— valid alternative feature."
        )
    else:
        return (
            f"{feature_name} DILUTES the regime signal same as F&G did "
            f"(spread={spread_pct:.1f}% vs univariate {uni_spread_pct:.1f}%, ratio={ratio:.2f}x) "
            f"— on-chain daily features don't align with 5-min vol regimes."
        )


def main() -> None:
    print("=" * 80)
    print("Bivariate HMM: log_return + CoinMetrics exchange FlowIn")
    print("=" * 80)

    # ── 1. Load returns ──────────────────────────────────────────────────────
    print("\n[1/5] Loading Binance aggTrades → 5-min log returns (98 days)...")
    rdf = build_returns_df(start="2026-01-01", end="2026-04-08")
    print(f"      {len(rdf):,} five-min bars, "
          f"{rdf['timestamp'].min()} → {rdf['timestamp'].max()}")

    # ── 2. Load CoinMetrics ──────────────────────────────────────────────────
    print("\n[2/5] Loading CoinMetrics on-chain data...")
    cm = load_coinmetrics()
    print(f"      {len(cm)} daily rows, columns: {list(cm.columns)}")
    print(f"      FlowInExUSD: mean={cm['FlowInExUSD'].mean():.2e}, "
          f"std={cm['FlowInExUSD'].std():.2e}, "
          f"range=[{cm['FlowInExUSD'].min():.2e}, {cm['FlowInExUSD'].max():.2e}]")

    # ── 3. Prepare sentiment DataFrames for align helper ─────────────────────
    print("\n[3/5] Aligning FlowIn features to 5-min bar grid (ffill daily)...")

    # Variant A: raw FlowInExUSD
    sdf_raw = cm[["date", "FlowInExUSD"]].copy()
    features_raw = align_sentiment_to_returns_bar(
        rdf, sdf_raw,
        returns_col="log_return",
        sentiment_col="FlowInExUSD",
    )
    print(f"      Raw FlowIn aligned — shape: {features_raw.shape}")
    print(f"      Col 1 (FlowInExUSD): mean={features_raw[:,1].mean():.2e}, "
          f"std={features_raw[:,1].std():.2e}")

    # Variant B: log(FlowInExUSD)
    cm["logFlowInExUSD"] = np.log(cm["FlowInExUSD"])
    sdf_log = cm[["date", "logFlowInExUSD"]].copy()
    features_log = align_sentiment_to_returns_bar(
        rdf, sdf_log,
        returns_col="log_return",
        sentiment_col="logFlowInExUSD",
    )
    print(f"      Log FlowIn aligned  — shape: {features_log.shape}")
    print(f"      Col 1 (log FlowIn): mean={features_log[:,1].mean():.4f}, "
          f"std={features_log[:,1].std():.4f}, "
          f"range=[{features_log[:,1].min():.4f}, {features_log[:,1].max():.4f}]")

    # Variant C (3-feature bonus): log_return + FlowInExUSD + FlowOutExUSD
    cm["logFlowOutExUSD"] = np.log(cm["FlowOutExUSD"])
    sdf_out = cm[["date", "logFlowOutExUSD"]].copy()
    features_out_aligned = align_sentiment_to_returns_bar(
        rdf, sdf_out,
        returns_col="log_return",
        sentiment_col="logFlowOutExUSD",
    )
    # Build 3-feature matrix: combine log_return, logFlowIn, logFlowOut
    # Both log-feature arrays share the same log_return col (col 0).
    # Use the log-FlowIn array as base, then append logFlowOut as col 2.
    # Must align lengths: take minimum to avoid shape mismatch from ffill edge effects.
    n3 = min(len(features_log), len(features_out_aligned))
    features_3d = np.column_stack([
        features_log[:n3, 0],   # log_return
        features_log[:n3, 1],   # logFlowIn
        features_out_aligned[:n3, 1],  # logFlowOut
    ])
    print(f"      3-feature (logReturn+logIn+logOut) — shape: {features_3d.shape}")

    # ── 4. Fit HMMs ──────────────────────────────────────────────────────────
    print("\n[4/5] Fitting HMMs...")

    # 4a. Univariate baseline
    print("  [4a] Univariate HMM (log_return only)...")
    regimes_uni, _ = fit_hmm(features_raw[:, 0])
    uni = summarize_regimes(regimes_uni, "univariate (log_return only)")
    print(f"       sigma_on={uni['sigma_on']:.4f}, sigma_off={uni['sigma_off']:.4f}, "
          f"spread={uni['sigma_spread_pct']:.1f}%, prob_off={uni['prob_off']:.3f}")

    # 4b. Bivariate: log_return + raw FlowInExUSD
    print("  [4b] Bivariate HMM (log_return + raw FlowInExUSD)...")
    regimes_raw, _ = fit_hmm(features=features_raw)
    bi_raw = summarize_regimes(regimes_raw, "bivariate (log_return + FlowInExUSD)")
    print(f"       sigma_on={bi_raw['sigma_on']:.4f}, sigma_off={bi_raw['sigma_off']:.4f}, "
          f"spread={bi_raw['sigma_spread_pct']:.1f}%, prob_off={bi_raw['prob_off']:.3f}")

    # 4c. Bivariate: log_return + log(FlowInExUSD)
    print("  [4c] Bivariate HMM (log_return + log FlowInExUSD)...")
    regimes_log, _ = fit_hmm(features=features_log)
    bi_log = summarize_regimes(regimes_log, "bivariate (log_return + log FlowInExUSD)")
    print(f"       sigma_on={bi_log['sigma_on']:.4f}, sigma_off={bi_log['sigma_off']:.4f}, "
          f"spread={bi_log['sigma_spread_pct']:.1f}%, prob_off={bi_log['prob_off']:.3f}")

    # 4d. 3-feature bonus: log_return + logFlowIn + logFlowOut
    print("  [4d] 3-feature HMM (log_return + logFlowIn + logFlowOut)...")
    tri_result = None
    try:
        regimes_3d, _ = fit_hmm(features=features_3d)
        tri = summarize_regimes(regimes_3d, "3-feature (log_return + logFlowIn + logFlowOut)")
        tri_result = tri
        print(f"       sigma_on={tri['sigma_on']:.4f}, sigma_off={tri['sigma_off']:.4f}, "
              f"spread={tri['sigma_spread_pct']:.1f}%, prob_off={tri['prob_off']:.3f}")
    except Exception as e:
        print(f"       3-feature HMM failed: {e}")

    # ── 5. Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    header = f"{'Features':<48} {'sigma_on':>9} {'sigma_off':>10} {'spread':>8} {'prob_off':>9}"
    print(header)
    print("-" * 80)

    def row(label, r):
        return (f"{label:<48} {r['sigma_on']:>9.4f} {r['sigma_off']:>10.4f} "
                f"{r['sigma_spread_pct']:>7.1f}% {r['prob_off']:>9.3f}")

    print(row("univariate (log_return only)", uni))
    print(row("bivariate (log_return + FlowInExUSD)", bi_raw))
    print(row("bivariate (log_return + log FlowInExUSD)", bi_log))
    if tri_result:
        print(row("3-feature (logReturn + logFlowIn + logFlowOut)", tri_result))

    print()
    print("Previous F&G experiment for reference:")
    print(f"  univariate (log_return):      spread≈250%")
    print(f"  bivariate (log_return + F&G): spread≈45%  (F&G DILUTED signal)")

    # ── Interpretations ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print(interpret_result(bi_raw['sigma_spread_pct'], uni['sigma_spread_pct'],
                           "FlowInExUSD (raw)"))
    print()
    print(interpret_result(bi_log['sigma_spread_pct'], uni['sigma_spread_pct'],
                           "log(FlowInExUSD)"))
    if tri_result:
        print()
        print(interpret_result(tri_result['sigma_spread_pct'], uni['sigma_spread_pct'],
                               "3-feature (logFlowIn + logFlowOut)"))

    # ── Save results ──────────────────────────────────────────────────────────
    spread_ratio_raw = (
        bi_raw['sigma_spread_pct'] / uni['sigma_spread_pct']
        if uni['sigma_spread_pct'] > 0 else float("inf")
    )
    spread_ratio_log = (
        bi_log['sigma_spread_pct'] / uni['sigma_spread_pct']
        if uni['sigma_spread_pct'] > 0 else float("inf")
    )

    results = {
        "experiment": "bivariate_hmm_coinmetrics_flowin",
        "date_range": "2026-01-01 to 2026-04-08",
        "n_5min_bars": int(features_raw.shape[0]),
        "univariate": uni,
        "bivariate_raw_flowin": bi_raw,
        "bivariate_log_flowin": bi_log,
        "spread_ratio_raw_vs_uni": float(spread_ratio_raw),
        "spread_ratio_log_vs_uni": float(spread_ratio_log),
        "fg_reference": FG_REFERENCE,
        "interpretation_raw": interpret_result(
            bi_raw['sigma_spread_pct'], uni['sigma_spread_pct'], "FlowInExUSD (raw)"),
        "interpretation_log": interpret_result(
            bi_log['sigma_spread_pct'], uni['sigma_spread_pct'], "log(FlowInExUSD)"),
    }

    if tri_result:
        spread_ratio_3d = (
            tri_result['sigma_spread_pct'] / uni['sigma_spread_pct']
            if uni['sigma_spread_pct'] > 0 else float("inf")
        )
        results["trivariate_logflowin_logflowout"] = tri_result
        results["spread_ratio_3d_vs_uni"] = float(spread_ratio_3d)
        results["interpretation_3d"] = interpret_result(
            tri_result['sigma_spread_pct'], uni['sigma_spread_pct'],
            "3-feature (logFlowIn + logFlowOut)")

    out = PROJECT / "data" / "hmm_bivariate_coinmetrics_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
