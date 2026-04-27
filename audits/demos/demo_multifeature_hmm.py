"""Demo: Multivariate (log_return + sentiment) HMM vs univariate HMM.

This script is gated on data/marketpsych_btc_sentiment.csv existing.
Run it once the WRDS MarketPsych worker has populated that file.

Expected output (approximate):
                sigma_on  sigma_off  spread  stationary_off
    univariate    0.958     1.039    8.4%       49.9%
    multivar        ?         ?       ?           ?

Usage:
    python scripts/demo_multifeature_hmm.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from extensions.regime import align_sentiment_to_returns_bar, fit_hmm

# ── Sentinel: check for sentiment file ───────────────────────────────────────
_SENTIMENT_PATH = os.path.join(_project_root, "data", "marketpsych_btc_sentiment.csv")
if not os.path.exists(_SENTIMENT_PATH):
    print(
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  data/marketpsych_btc_sentiment.csv not found.              ║\n"
        "║  This demo is a no-op until the WRDS MarketPsych worker     ║\n"
        "║  completes and populates that file.                         ║\n"
        "║  Re-run after the parallel worker finishes.                 ║\n"
        "╚══════════════════════════════════════════════════════════════╝"
    )
    sys.exit(0)

# ── Load sentiment data ───────────────────────────────────────────────────────
print(f"Loading sentiment from: {_SENTIMENT_PATH}")
sentiment_df = pd.read_csv(_SENTIMENT_PATH, parse_dates=["date"])

# Accept flexible column names: use the first non-date numeric column if
# 'sentiment' is not present.
if "sentiment" not in sentiment_df.columns:
    numeric_cols = sentiment_df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        raise ValueError(
            "marketpsych_btc_sentiment.csv has no numeric columns. "
            "Expected at least a 'sentiment' column."
        )
    sentiment_col = numeric_cols[0]
    print(
        f"  'sentiment' column not found; using first numeric column: '{sentiment_col}'"
    )
else:
    sentiment_col = "sentiment"

print(f"  Sentiment column: '{sentiment_col}', rows: {len(sentiment_df)}")
print(f"  Date range: {sentiment_df['date'].min()} → {sentiment_df['date'].max()}")

# ── Load Binance aggTrades (first 7 days) ─────────────────────────────────────
# Locate aggregated 5-min return files in data/.
_DATA_DIR = os.path.join(_project_root, "data")

# Prefer a pre-computed 5-min OHLCV / returns file if available.
_candidate_files = [
    os.path.join(_DATA_DIR, "btc_5min_returns.csv"),
    os.path.join(_DATA_DIR, "btc_5min_ohlcv.csv"),
    os.path.join(_DATA_DIR, "btcusdt_5min.csv"),
]
_returns_path = next((f for f in _candidate_files if os.path.exists(f)), None)

if _returns_path is None:
    # Fall back: look for any aggTrades parquet / CSV in data/
    from glob import glob
    candidates = (
        glob(os.path.join(_DATA_DIR, "*.parquet"))
        + glob(os.path.join(_DATA_DIR, "*agg*"))
        + glob(os.path.join(_DATA_DIR, "*btc*"))
    )
    candidates = [c for c in candidates if os.path.isfile(c) and "sentiment" not in c]
    if not candidates:
        print(
            "\nNo BTC price / returns data found in data/.\n"
            "Place a CSV with columns ['timestamp', 'log_return'] at:\n"
            f"  {os.path.join(_DATA_DIR, 'btc_5min_returns.csv')}\n"
            "and re-run."
        )
        sys.exit(0)
    _returns_path = sorted(candidates)[0]
    print(f"  Using returns file: {_returns_path}")
else:
    print(f"  Using returns file: {_returns_path}")

if _returns_path.endswith(".parquet"):
    returns_raw = pd.read_parquet(_returns_path)
else:
    returns_raw = pd.read_csv(_returns_path)

# Normalise timestamp column
if "timestamp" not in returns_raw.columns:
    # Try 'time', 'datetime', 'open_time'
    for alt in ("time", "datetime", "open_time", "date"):
        if alt in returns_raw.columns:
            returns_raw = returns_raw.rename(columns={alt: "timestamp"})
            break

if "timestamp" not in returns_raw.columns:
    raise ValueError(
        f"Cannot find a timestamp column in {_returns_path}. "
        f"Columns: {list(returns_raw.columns)}"
    )

returns_raw["timestamp"] = pd.to_datetime(returns_raw["timestamp"], utc=True)

# Compute log_return if not already present
if "log_return" not in returns_raw.columns:
    price_col = next(
        (c for c in ("close", "price", "last", "vwap") if c in returns_raw.columns),
        None,
    )
    if price_col is None:
        raise ValueError(
            f"Cannot find 'log_return' or a price column in {_returns_path}. "
            f"Columns: {list(returns_raw.columns)}"
        )
    returns_raw["log_return"] = np.log(
        returns_raw[price_col] / returns_raw[price_col].shift(1)
    )

# Keep first 7 days
first_ts = returns_raw["timestamp"].min()
cutoff = first_ts + pd.Timedelta(days=7)
returns_raw = returns_raw[returns_raw["timestamp"] < cutoff].copy()
returns_raw = returns_raw.dropna(subset=["log_return"])
print(f"\nReturns: {len(returns_raw)} bars over first 7 days")
print(f"  Range: {returns_raw['timestamp'].min()} → {returns_raw['timestamp'].max()}")

# ── Align sentiment to 5-min grid ────────────────────────────────────────────
print("\nAligning daily sentiment to 5-min grid (ffill)...")
feat_2d = align_sentiment_to_returns_bar(
    returns_raw,
    sentiment_df,
    returns_col="log_return",
    sentiment_col=sentiment_col,
    fill="ffill",
)
print(f"  Aligned feature matrix shape: {feat_2d.shape}  (dropped NaN rows)")

if feat_2d.shape[0] < 50:
    print(
        "\nToo few aligned observations after NaN removal. "
        "Check that sentiment dates overlap with returns dates."
    )
    sys.exit(1)

# ── Fit univariate HMM (log_return only) ─────────────────────────────────────
print("\nFitting univariate HMM on log_return only ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    uni_regimes, uni_states = fit_hmm(feat_2d[:, 0])

uni_on = next(r for r in uni_regimes if r.label == "risk_on")
uni_off = next(r for r in uni_regimes if r.label == "risk_off")
uni_spread_pct = (uni_off.sigma - uni_on.sigma) / max(uni_on.sigma, 1e-9) * 100

# ── Fit multivariate HMM (log_return + sentiment) ────────────────────────────
print(f"Fitting multivariate HMM on (log_return, {sentiment_col}) ...")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        multi_regimes, multi_states = fit_hmm(features=feat_2d)

    multi_on = next(r for r in multi_regimes if r.label == "risk_on")
    multi_off = next(r for r in multi_regimes if r.label == "risk_off")
    multi_spread_pct = (multi_off.sigma - multi_on.sigma) / max(multi_on.sigma, 1e-9) * 100
    multi_available = True
except NotImplementedError as e:
    print(f"\n  [!] Multivariate HMM unavailable: {e}")
    multi_available = False

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Regime comparison: univariate vs multivariate HMM")
print("=" * 60)
header = f"{'':>14}  {'sigma_on':>8}  {'sigma_off':>9}  {'spread':>8}  {'stationary_off':>14}"
print(header)
print("-" * 60)

uni_row = (
    f"{'univariate':>14}  "
    f"{uni_on.sigma:>8.3f}  "
    f"{uni_off.sigma:>9.3f}  "
    f"{uni_spread_pct:>7.1f}%  "
    f"{uni_off.probability:>14.1%}"
)
print(uni_row)

if multi_available:
    multi_row = (
        f"{'multivar':>14}  "
        f"{multi_on.sigma:>8.3f}  "
        f"{multi_off.sigma:>9.3f}  "
        f"{multi_spread_pct:>7.1f}%  "
        f"{multi_off.probability:>14.1%}"
    )
    print(multi_row)
    print("-" * 60)

    improvement = multi_spread_pct - uni_spread_pct
    print(f"\n  Sigma-spread improvement: {improvement:+.1f}pp")
    if improvement > 0:
        print("  Sentiment amplifies regime separation — proceed with bivariate fit.")
    else:
        print("  Sentiment did not amplify separation on this window — check data.")
else:
    print(f"{'multivar':>14}  {'(unavailable)':>35}")
    print("-" * 60)

print("=" * 60)
