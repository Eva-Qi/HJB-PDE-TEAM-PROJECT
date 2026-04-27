"""Pull extended Deribit public-API data for MF796 cross-validation.

Pulls three datasets:
  1. Historical realized volatility (BTC, daily)
  2. BTC-PERPETUAL OHLCV — daily (2023-05-01 → today) + hourly (2025-07-01 → today)
  3. BTC option OI/volume snapshot (all live options, single call)

Endpoint #4 (index price series) is merged into #2: BTC-PERPETUAL close ≈ BTC index
for cross-validation purposes; noted in output metadata.

All endpoints: Deribit public API v2, no auth required.
Uses only stdlib (urllib.request) — no third-party dependencies.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DERIBIT_BASE = "https://www.deribit.com/api/v2/public"

TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TODAY_MS = int(datetime.now(timezone.utc).timestamp() * 1000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(date_iso: str) -> int:
    dt = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch(endpoint: str, params: dict) -> dict:
    url = f"{DERIBIT_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"  ERROR fetching {url}: {exc}", file=sys.stderr)
        raise


def _nan_count(records: list[dict], key: str) -> int:
    count = 0
    for r in records:
        v = r.get(key)
        if v is None:
            count += 1
        elif isinstance(v, float) and (v != v):  # NaN check
            count += 1
    return count


def _print_file_summary(label: str, records: list[dict], ts_key: str = "timestamp_ms") -> None:
    if not records:
        print(f"  {label}: 0 rows — EMPTY")
        return
    # date range from first/last
    first = records[0]
    last = records[-1]
    n = len(records)
    numeric_keys = [k for k, v in first.items() if isinstance(v, (int, float)) and k != ts_key]
    nan_info = {k: _nan_count(records, k) for k in numeric_keys}
    nan_str = ", ".join(f"{k}={v}" for k, v in nan_info.items() if v > 0)
    print(f"\n  {label}:")
    print(f"    rows      = {n:,}")
    print(f"    first row = {first}")
    print(f"    last row  = {last}")
    if nan_str:
        print(f"    NaN counts: {nan_str}")
    else:
        print(f"    NaN counts: none")


# ---------------------------------------------------------------------------
# 1. Historical realized volatility
# ---------------------------------------------------------------------------

def pull_historical_vol() -> None:
    print("\n[1/3] Pulling BTC historical volatility ...")
    url = f"{DERIBIT_BASE}/get_historical_volatility?currency=BTC"
    with urllib.request.urlopen(url, timeout=60) as r:
        resp = json.loads(r.read().decode("utf-8"))

    raw = resp.get("result", [])
    if not raw:
        print(f"  ERROR: empty result. Full response: {json.dumps(resp)[:400]}")
        return

    # result is list of [timestamp_ms, hv_value]
    records = []
    for row in raw:
        ts_ms = int(row[0])
        hv = float(row[1])
        records.append({
            "date": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d"),
            "timestamp_ms": ts_ms,
            "hv": hv,
        })

    records.sort(key=lambda r: r["timestamp_ms"])

    out = {
        "source_url": url,
        "description": "Deribit own realized volatility for BTC, daily. "
                       "Cross-check vs GK vol in walk-forward results.",
        "n_rows": len(records),
        "start": records[0]["date"],
        "end": records[-1]["date"],
        "data": records,
    }
    outfile = DATA_DIR / "deribit_btc_historical_vol.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {outfile}")
    _print_file_summary("deribit_btc_historical_vol", records)


# ---------------------------------------------------------------------------
# 2. BTC-PERPETUAL OHLCV (daily + hourly)
# ---------------------------------------------------------------------------

def _fetch_ohlcv_chunked(
    instrument: str,
    resolution: str,
    start_iso: str,
    end_iso: str,
    chunk_days: int,
    sleep_s: float = 0.1,
) -> list[dict]:
    """Paginate get_tradingview_chart_data across time, max 1000 rows/request."""
    all_records: list[dict] = []
    seen_ts: set[int] = set()

    start_ms = _ms(start_iso)
    end_ms = _ms(end_iso)
    chunk_ms = chunk_days * 24 * 3600 * 1000
    cursor = start_ms

    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        params = {
            "instrument_name": instrument,
            "start_timestamp": cursor,
            "end_timestamp": chunk_end,
            "resolution": resolution,
        }
        resp = _fetch("get_tradingview_chart_data", params)
        result = resp.get("result", {})

        ticks = result.get("ticks", [])
        opens = result.get("open", [])
        highs = result.get("high", [])
        lows = result.get("low", [])
        closes = result.get("close", [])
        volumes = result.get("volume", [])
        costs = result.get("cost", [])

        if not ticks:
            # No data in window — advance cursor and continue
            cursor = chunk_end
            time.sleep(sleep_s)
            continue

        for i, ts in enumerate(ticks):
            if ts in seen_ts:
                continue
            seen_ts.add(ts)
            rec = {
                "timestamp_ms": ts,
                "datetime": _dt(ts),
                "open": opens[i] if i < len(opens) else None,
                "high": highs[i] if i < len(highs) else None,
                "low": lows[i] if i < len(lows) else None,
                "close": closes[i] if i < len(closes) else None,
                "volume": volumes[i] if i < len(volumes) else None,
                "cost": costs[i] if i < len(costs) else None,
            }
            all_records.append(rec)

        max_ts = max(ticks)
        cursor = max_ts + 1  # +1 ms to avoid re-fetching last row
        time.sleep(sleep_s)
        print(f"    ... fetched up to {_dt(max_ts)}, total rows so far: {len(all_records)}")

    all_records.sort(key=lambda r: r["timestamp_ms"])
    return all_records


def pull_perp_ohlcv_daily() -> None:
    print("\n[2a/3] Pulling BTC-PERPETUAL OHLCV daily (2023-05-01 → today) ...")
    records = _fetch_ohlcv_chunked(
        instrument="BTC-PERPETUAL",
        resolution="1D",
        start_iso="2023-05-01",
        end_iso=TODAY,
        chunk_days=365,  # 1D resolution: 365 rows / chunk well under 1000
        sleep_s=0.1,
    )
    if not records:
        print("  ERROR: no daily OHLCV rows returned")
        return

    out = {
        "source_url": f"{DERIBIT_BASE}/get_tradingview_chart_data",
        "params": {
            "instrument_name": "BTC-PERPETUAL",
            "resolution": "1D",
            "start": "2023-05-01",
            "end": TODAY,
        },
        "description": (
            "BTC-PERPETUAL daily OHLCV from Deribit. "
            "BTC-PERPETUAL close ≈ BTC spot/index for cross-validation vs Binance.US. "
            "Index endpoint (#4 in task spec) is subsumed here — perp tracks index within basis."
        ),
        "n_rows": len(records),
        "start": records[0]["datetime"],
        "end": records[-1]["datetime"],
        "data": records,
    }
    outfile = DATA_DIR / "deribit_btc_perp_ohlcv_daily.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {outfile}")
    _print_file_summary("deribit_btc_perp_ohlcv_daily", records)


def pull_perp_ohlcv_hourly() -> None:
    print("\n[2b/3] Pulling BTC-PERPETUAL OHLCV hourly (2025-07-01 → today) ...")
    records = _fetch_ohlcv_chunked(
        instrument="BTC-PERPETUAL",
        resolution="60",
        start_iso="2025-07-01",
        end_iso=TODAY,
        chunk_days=30,  # 60-min resolution: 30d = 720 rows/chunk, under 1000
        sleep_s=0.1,
    )
    if not records:
        print("  ERROR: no hourly OHLCV rows returned")
        return

    out = {
        "source_url": f"{DERIBIT_BASE}/get_tradingview_chart_data",
        "params": {
            "instrument_name": "BTC-PERPETUAL",
            "resolution": "60",
            "start": "2025-07-01",
            "end": TODAY,
        },
        "description": (
            "BTC-PERPETUAL hourly OHLCV from Deribit. "
            "Covers ~10 months to match aggTrades window. "
            "Cross-check vs Binance hourly klines."
        ),
        "n_rows": len(records),
        "start": records[0]["datetime"],
        "end": records[-1]["datetime"],
        "data": records,
    }
    outfile = DATA_DIR / "deribit_btc_perp_ohlcv_hourly.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {outfile}")
    _print_file_summary("deribit_btc_perp_ohlcv_hourly", records)


# ---------------------------------------------------------------------------
# 3. BTC option OI/volume snapshot
# ---------------------------------------------------------------------------

def pull_option_summary() -> None:
    print("\n[3/3] Pulling BTC option book summary snapshot ...")
    params = {"currency": "BTC", "kind": "option"}
    resp = _fetch("get_book_summary_by_currency", params)
    raw = resp.get("result", [])

    if not raw:
        print(f"  ERROR: empty result. Response: {json.dumps(resp)[:400]}")
        return

    # Normalize fields; keep all keys Deribit returns
    records = []
    for item in raw:
        rec = {
            "instrument_name": item.get("instrument_name"),
            "open_interest": item.get("open_interest"),
            "volume": item.get("volume"),
            "bid_price": item.get("bid_price"),
            "ask_price": item.get("ask_price"),
            "mark_iv": item.get("mark_iv"),
            "mid_price": item.get("mid_price"),
            "mark_price": item.get("mark_price"),
            "underlying_price": item.get("underlying_price"),
            "underlying_index": item.get("underlying_index"),
            "last": item.get("last"),
            "interest_rate": item.get("interest_rate"),
            "creation_timestamp": item.get("creation_timestamp"),
            "estimated_delivery_price": item.get("estimated_delivery_price"),
            "quote_currency": item.get("quote_currency"),
            "base_currency": item.get("base_currency"),
        }
        records.append(rec)

    # Sort by instrument name for readability
    records.sort(key=lambda r: r["instrument_name"] or "")

    snapshot_ts = int(time.time() * 1000)
    snapshot_dt = _dt(snapshot_ts)

    # Aggregate stats
    total_oi = sum(r["open_interest"] or 0 for r in records)
    total_vol = sum(r["volume"] or 0 for r in records)
    n_calls = sum(1 for r in records if r["instrument_name"] and r["instrument_name"].endswith("-C"))
    n_puts = sum(1 for r in records if r["instrument_name"] and r["instrument_name"].endswith("-P"))

    out = {
        "source_url": f"{DERIBIT_BASE}/get_book_summary_by_currency",
        "params": params,
        "description": (
            "Snapshot of ALL live BTC options on Deribit — OI, volume, bid/ask, mark_iv. "
            "Serves as partial replacement for missing Binance futures OI data."
        ),
        "snapshot_timestamp_ms": snapshot_ts,
        "snapshot_datetime": snapshot_dt,
        "n_rows": len(records),
        "stats": {
            "n_calls": n_calls,
            "n_puts": n_puts,
            "total_open_interest_btc": total_oi,
            "total_volume_btc": total_vol,
        },
        "data": records,
    }
    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    outfile = DATA_DIR / f"deribit_btc_option_summary_{date_tag}.json"
    outfile.write_text(json.dumps(out, indent=2))
    print(f"  Saved: {outfile}")
    _print_file_summary("deribit_btc_option_summary", records, ts_key="creation_timestamp")
    print(f"    calls={n_calls}, puts={n_puts}, total OI={total_oi:,.2f} BTC, vol={total_vol:,.4f} BTC")


# ---------------------------------------------------------------------------
# Cross-check: Deribit daily close vs Binance.US spot
# ---------------------------------------------------------------------------

def cross_check_vs_binance() -> None:
    print("\n[Cross-check] BTC-PERPETUAL daily close vs Binance.US spot ...")
    deribit_file = DATA_DIR / "deribit_btc_perp_ohlcv_daily.json"
    binance_file = DATA_DIR / "binance_btc_klines_1d.json"

    if not deribit_file.exists():
        print("  SKIP: deribit_btc_perp_ohlcv_daily.json not found")
        return
    if not binance_file.exists():
        print("  SKIP: binance_btc_klines_1d.json not found")
        return

    with open(deribit_file) as f:
        deribit_data = json.load(f)
    with open(binance_file) as f:
        binance_raw = json.load(f)

    # Build Deribit close map: date -> close
    deribit_map: dict[str, float] = {}
    for r in deribit_data.get("data", []):
        date_key = r.get("datetime", "")[:10]  # YYYY-MM-DD
        if r.get("close") is not None:
            deribit_map[date_key] = float(r["close"])

    # Build Binance close map — handle multiple possible schemas
    binance_map: dict[str, float] = {}
    raw_list = binance_raw if isinstance(binance_raw, list) else binance_raw.get("data", [])
    for row in raw_list:
        if isinstance(row, list):
            # Standard Binance kline array: [open_time_ms, open, high, low, close, ...]
            ts_ms = int(row[0])
            close = float(row[4])
            date_key = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            binance_map[date_key] = close
        elif isinstance(row, dict):
            # Dict format
            date_key = row.get("date") or row.get("datetime", "")[:10]
            close = row.get("close")
            if date_key and close is not None:
                binance_map[date_key] = float(close)

    # Find common dates and compute diff
    common_dates = sorted(set(deribit_map) & set(binance_map))
    if not common_dates:
        print("  No common dates found between Deribit and Binance files")
        return

    threshold_pct = 0.5
    flagged = []
    diffs = []
    for date in common_dates:
        d = deribit_map[date]
        b = binance_map[date]
        if b == 0:
            continue
        pct_diff = abs(d - b) / b * 100.0
        diffs.append(pct_diff)
        if pct_diff > threshold_pct:
            flagged.append((date, d, b, pct_diff))

    print(f"  Common dates: {len(common_dates)}")
    print(f"  Mean |diff|: {sum(diffs)/len(diffs):.4f}%")
    print(f"  Max |diff|:  {max(diffs):.4f}%")
    if flagged:
        print(f"\n  FLAG: {len(flagged)} dates where |Deribit - Binance| / Binance > {threshold_pct}%:")
        for date, d, b, pct in sorted(flagged, key=lambda x: -x[3])[:20]:
            print(f"    {date}: Deribit={d:,.2f}  Binance={b:,.2f}  diff={pct:.3f}%")
    else:
        print(f"  All {len(common_dates)} common dates within {threshold_pct}% — no cross-exchange anomaly flagged.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Run date     : {TODAY}")

    pull_historical_vol()
    pull_perp_ohlcv_daily()
    pull_perp_ohlcv_hourly()
    pull_option_summary()
    cross_check_vs_binance()

    print("\nDone.")


if __name__ == "__main__":
    main()
