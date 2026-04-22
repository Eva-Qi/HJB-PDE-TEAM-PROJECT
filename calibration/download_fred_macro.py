"""Pull daily macro series via yfinance (FRED CSV endpoint unreachable from local network).

Mappings (yfinance ticker → FRED-equivalent):
    ^VIX       — CBOE VIX (FRED: VIXCLS)            equity vol; cross-check vs Deribit DVOL
    ^TNX       — 10Y Treasury yield × 10 (DGS10 ÷ 10)
    ^IRX       — 13-week T-bill yield (~ DGS3MO)
    DX-Y.NYB   — DXY (broad dollar index, FRED: DTWEXBGS proxy)
    EURUSD=X   — USD/EUR spot (FRED: DEXUSEU; inverse — value 1/x)

Output: data/fred_macro_daily.json
    {
      "source": "yfinance (Yahoo Finance)",
      "fetched_at": ISO8601,
      "series": {
        "<ticker>": [{"date", "open", "high", "low", "close", "volume"}, ...]
      },
      "summary": { "<ticker>": {n_rows, start, end, last_close} }
    }
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "data" / "fred_macro_daily.json"

TICKERS = ["^VIX", "^TNX", "^IRX", "DX-Y.NYB", "EURUSD=X"]
START = "2023-01-01"
END = datetime.now(timezone.utc).date().isoformat()


def fetch(ticker: str) -> list[dict]:
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return []
    # yfinance occasionally returns MultiIndex columns when multiple tickers — collapse
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "date": r["Date"].strftime("%Y-%m-%d"),
            "open": float(r["Open"]) if pd.notna(r["Open"]) else None,
            "high": float(r["High"]) if pd.notna(r["High"]) else None,
            "low": float(r["Low"]) if pd.notna(r["Low"]) else None,
            "close": float(r["Close"]) if pd.notna(r["Close"]) else None,
            "volume": int(r["Volume"]) if pd.notna(r["Volume"]) else None,
        })
    return rows


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "source": "yfinance (Yahoo Finance)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "start": START,
        "end": END,
        "series": {},
        "summary": {},
        "yfinance_to_fred_map": {
            "^VIX": "VIXCLS",
            "^TNX": "DGS10 (×10 — TNX is yield × 10)",
            "^IRX": "DGS3MO (closest)",
            "DX-Y.NYB": "DTWEXBGS (proxy)",
            "EURUSD=X": "DEXUSEU (inverse — yfinance gives USD/EUR)",
        },
    }

    for ticker in TICKERS:
        print(f"[{ticker}] fetching {START} → {END}...", flush=True)
        try:
            rows = fetch(ticker)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            payload["series"][ticker] = {"error": f"{type(e).__name__}: {e}"}
            continue

        non_null = [r for r in rows if r["close"] is not None]
        payload["series"][ticker] = rows
        payload["summary"][ticker] = {
            "n_rows": len(rows),
            "n_non_null_close": len(non_null),
            "start": rows[0]["date"] if rows else None,
            "end": rows[-1]["date"] if rows else None,
            "first_close": non_null[0]["close"] if non_null else None,
            "last_close": non_null[-1]["close"] if non_null else None,
        }
        print(f"  {len(rows):,} rows  ({len(non_null):,} non-null close)  "
              f"{rows[0]['date']} → {rows[-1]['date']}  last_close={non_null[-1]['close']:.4f}")

    OUTPUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {OUTPUT}")
    print("\n=== Summary ===")
    for ticker, s in payload["summary"].items():
        if isinstance(s, dict):
            print(f"  {ticker:<10}  n={s['n_rows']:>5}  "
                  f"range={s['start']} → {s['end']}  last_close={s['last_close']}")


if __name__ == "__main__":
    sys.exit(main() or 0)
