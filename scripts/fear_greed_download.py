"""Download Bitcoin Fear & Greed Index from alternative.me — free public API.

Replaces the failed WRDS MarketPsych pull: BU's subscription only includes
mpsych_sample which ends 2024-10-20 and has all-NULL crypto columns.

Alternative.me F&G Index is a daily 0-100 composite of:
  - Volatility (25%)
  - Market momentum / volume (25%)
  - Social media (15%)
  - Surveys (15% — currently paused)
  - Bitcoin dominance (10%)
  - Google Trends (10%)

Output format:
  date (YYYY-MM-DD)    value (0-100, low=fear, high=greed)    classification (str)

For our HMM extension: F&G value is the sentiment feature aligned alongside
log returns. Semantically bimodal by design (low values cluster in bear
regimes, high values in bull regimes) — better regime separation than the
single log-return feature.

API: https://api.alternative.me/fng/?limit=0 (get everything; free, no key)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# JSON (not CSV) because data/.gitignore excludes *.csv — keeps this
# tiny 98-row file committable alongside other data/*_results.json files.
OUTPUT_JSON = PROJECT_ROOT / "data" / "fear_greed_btc.json"

# Match the Binance data window
START_DATE = "2026-01-01"
END_DATE = "2026-04-08"

# alternative.me returns newest-first; limit=0 means "all history" (~2018-present)
API_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def fetch_all_history() -> pd.DataFrame:
    """Pull complete F&G history and return tidy DataFrame."""
    with urllib.request.urlopen(API_URL, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if "data" not in data:
        raise RuntimeError(f"Unexpected API shape: keys={list(data.keys())}")

    rows = []
    for entry in data["data"]:
        # API returns unix timestamp as string
        ts = int(entry["timestamp"])
        rows.append({
            "date": pd.Timestamp(ts, unit="s", tz="UTC").normalize(),
            "value": int(entry["value"]),
            "classification": entry.get("value_classification", ""),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching full F&G history from {API_URL}...")
    df_all = fetch_all_history()
    print(f"  Got {len(df_all):,} daily rows ({df_all['date'].min().date()} → {df_all['date'].max().date()})")

    # Slice to our window
    start = pd.Timestamp(START_DATE, tz="UTC")
    end = pd.Timestamp(END_DATE, tz="UTC") + pd.Timedelta(days=1)  # inclusive
    df = df_all[(df_all["date"] >= start) & (df_all["date"] < end)].reset_index(drop=True)

    if df.empty:
        print(f"ERROR: no rows in window {START_DATE} to {END_DATE}")
        sys.exit(1)

    # Save as JSON (records orient) — small, committable, easy to reload
    # via pd.read_json(path, orient="records"). Dates serialized as ISO strings.
    records = [
        {
            "date": r["date"].date().isoformat(),
            "value": int(r["value"]),
            "classification": r["classification"],
        }
        for _, r in df.iterrows()
    ]
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "source": "alternative.me Fear & Greed Index",
            "api_url": API_URL,
            "start": START_DATE,
            "end": END_DATE,
            "n_rows": len(records),
            "data": records,
        }, f, indent=2)
    print(f"Saved: {OUTPUT_JSON} ({len(df)} rows)")

    # Quick distribution summary
    print()
    print("Distribution summary:")
    print(df["value"].describe())
    print()
    print("Classification counts:")
    print(df["classification"].value_counts())

    # Simple bimodality check: is min-max spread > 30? (fear to greed range)
    spread = int(df["value"].max()) - int(df["value"].min())
    print(f"\nValue spread: {spread} (min={df['value'].min()}, max={df['value'].max()})")
    if spread >= 30:
        print("Good: wide enough range to serve as a meaningful HMM feature.")
    else:
        print("Warning: narrow range — may not amplify regime separation as hoped.")


if __name__ == "__main__":
    main()
