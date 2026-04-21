"""Download Deribit BTC option chain snapshot for Q-measure Heston calibration.

Public REST endpoints only — no API key required.

Endpoints used:
    GET /api/v2/public/get_index_price?index_name=btc_usd
    GET /api/v2/public/get_instruments?currency=BTC&kind=option&expired=false
    GET /api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option

Output:
    data/deribit_btc_option_chain_YYYYMMDD.json  (skip-if-exists)

Usage:
    python calibration/download_deribit.py
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://www.deribit.com/api/v2/public"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _get(endpoint: str, params: dict) -> dict:
    """HTTP GET to Deribit public API, return parsed JSON result."""
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}/{endpoint}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "mf796-project/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    if "error" in data:
        raise RuntimeError(f"Deribit API error: {data['error']}")
    return data["result"]


def fetch_snapshot() -> dict:
    """Pull a single option chain snapshot from Deribit.

    Returns
    -------
    dict with keys:
        timestamp       : ISO UTC string of the pull time
        underlying_price: float, BTC/USD index price
        contracts       : list of dicts (one per valid option row)
    """
    # 1. Current index price
    idx_result = _get("get_index_price", {"index_name": "btc_usd"})
    underlying_price = float(idx_result["index_price"])
    pull_ts = datetime.now(tz=timezone.utc).isoformat()

    # 2. Instrument metadata (strikes, expiry, call/put)
    instruments_raw = _get("get_instruments", {
        "currency": "BTC",
        "kind": "option",
        "expired": "false",
    })
    instruments_by_name: dict[str, dict] = {
        inst["instrument_name"]: inst for inst in instruments_raw
    }

    # 3. Book summary (mark_price, mark_iv, open_interest)
    summaries_raw = _get("get_book_summary_by_currency", {
        "currency": "BTC",
        "kind": "option",
    })

    now_ms = time.time() * 1000  # current time in unix ms

    contracts = []
    for s in summaries_raw:
        name = s["instrument_name"]
        inst = instruments_by_name.get(name)
        if inst is None:
            continue

        mark_iv = s.get("mark_iv")
        if mark_iv is None or mark_iv <= 0:
            continue  # no usable IV

        expiry_ms = inst["expiration_timestamp"]  # unix ms
        T_seconds = (expiry_ms - now_ms) / 1000.0
        if T_seconds <= 0:
            continue  # already expired

        T_years = T_seconds / SECONDS_PER_YEAR
        expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=timezone.utc)

        contracts.append({
            "instrument_name": name,
            "kind": "C" if inst["option_type"] == "call" else "P",
            "strike": float(inst["strike"]),
            "expiry_date": expiry_dt.date().isoformat(),
            "T": round(T_years, 6),
            "mark_iv": float(mark_iv) / 100.0,   # convert % → decimal
            "bid_iv": None,                        # not provided by this endpoint
            "ask_iv": None,
            "mark_price": float(s.get("mark_price", 0.0)),   # in BTC
            "open_interest": float(s.get("open_interest", 0.0)),
            "underlying_price": float(s.get("underlying_price", underlying_price)),
        })

    return {
        "timestamp": pull_ts,
        "underlying_price": underlying_price,
        "contracts": contracts,
    }


def save_snapshot(snapshot: dict, date_str: str | None = None) -> Path:
    """Save snapshot to data/deribit_btc_option_chain_YYYYMMDD.json.

    Parameters
    ----------
    snapshot : dict
        Output of fetch_snapshot().
    date_str : str, optional
        Override date string (YYYYMMDD).  Defaults to today UTC.

    Returns
    -------
    Path to the written file (or existing file if skipped).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if date_str is None:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")

    out_path = DATA_DIR / f"deribit_btc_option_chain_{date_str}.json"

    if out_path.exists():
        print(f"[download_deribit] Skip — file already exists: {out_path.name}")
        return out_path

    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    n = len(snapshot["contracts"])
    print(
        f"[download_deribit] Saved {n} contracts → {out_path.name}  "
        f"(S={snapshot['underlying_price']:.0f})"
    )
    return out_path


def load_latest_snapshot() -> tuple[dict, Path]:
    """Load the most recently modified Deribit snapshot from data/.

    Returns
    -------
    (snapshot_dict, path)

    Raises
    ------
    FileNotFoundError if no deribit snapshot files exist.
    """
    files = sorted(DATA_DIR.glob("deribit_btc_option_chain_*.json"))
    if not files:
        raise FileNotFoundError(
            "No Deribit snapshot found in data/.  "
            "Run: python calibration/download_deribit.py"
        )
    latest = files[-1]  # lexicographic = chronological for YYYYMMDD names
    with open(latest) as f:
        snap = json.load(f)
    return snap, latest


def snapshot_to_dataframe(snapshot: dict):
    """Convert snapshot dict to a pandas DataFrame ready for calibration.

    Columns:
        instrument_name, kind, strike, expiry_date, T, mark_iv,
        bid_iv, ask_iv, mark_price, open_interest, underlying_price
    """
    import pandas as pd

    df = pd.DataFrame(snapshot["contracts"])
    if df.empty:
        return df

    # mark_iv is already in decimal (converted in fetch_snapshot)
    return df.reset_index(drop=True)


def fetch_volatility_index_data(
    start_timestamp: int,
    end_timestamp: int,
    resolution: str = "3600",
) -> list[dict]:
    """Fetch Deribit's BTC DVOL (implied volatility index) time series.

    Uses the public ``get_volatility_index_data`` endpoint which returns
    historical data up to 1 year back.  This is a *proxy* for the IV
    surface level over time (not full calibration data, but useful as a
    vol-regime indicator alongside repeated snapshot calibrations).

    Parameters
    ----------
    start_timestamp : int
        Unix timestamp in milliseconds.
    end_timestamp : int
        Unix timestamp in milliseconds.
    resolution : str
        Candle resolution in seconds.  Deribit accepts ``"3600"`` (1h),
        ``"1D"`` (daily), etc.

    Returns
    -------
    list of dicts with keys: timestamp_ms, open, high, low, close
    """
    result = _get("get_volatility_index_data", {
        "currency": "BTC",
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "resolution": resolution,
    })
    # API returns {"data": [[ts, o, h, l, c], ...]}
    rows = result.get("data", [])
    candles = []
    for row in rows:
        if len(row) >= 5:
            candles.append({
                "timestamp_ms": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
            })
    return candles


def download_deribit_snapshots_range(
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    """Infrastructure for a 7-day daily snapshot time series.

    IMPORTANT — Deribit's public REST API provides **current** snapshots only.
    Historical option chains (with per-instrument mark_iv, OI, etc.) are NOT
    available via the public API.  True historical IV surfaces would require
    Deribit's paid historical data product or a third-party data vendor.

    This function therefore:
        1. Fetches today's live snapshot (if not already saved).
        2. Attempts to fetch the DVOL index time series as a historical
           vol-level proxy (available via ``get_volatility_index_data``).
        3. Saves the DVOL series to ``data/deribit_dvol_history.json``.
        4. Returns the list of snapshot files present (typically just today's).

    To accumulate a multi-day time series, call this script once per day
    (e.g., via a cron job or manual run).  Each day's snapshot is saved
    with a YYYYMMDD suffix so repeated runs are idempotent.

    Parameters
    ----------
    start_date : str, optional
        Ignored (no historical snapshots available).  Kept for API symmetry.
    end_date : str, optional
        Ignored.  Kept for API symmetry.

    Returns
    -------
    list of Path
        Paths to snapshot JSON files found in data/.
    """
    saved_paths: list[Path] = []

    # 1. Today's snapshot
    print("[download_deribit_snapshots_range] Fetching today's snapshot ...")
    try:
        snap = fetch_snapshot()
        path = save_snapshot(snap)
        saved_paths.append(path)
    except Exception as exc:
        print(f"[download_deribit_snapshots_range] WARNING: snapshot fetch failed: {exc}")

    # 2. DVOL historical series as vol-level proxy (7-day window)
    try:
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        seven_days_ms = 7 * 24 * 3600 * 1000
        start_ms = now_ms - seven_days_ms
        candles = fetch_volatility_index_data(start_ms, now_ms, resolution="1D")
        if candles:
            dvol_path = DATA_DIR / "deribit_dvol_history.json"
            with open(dvol_path, "w") as f:
                json.dump({
                    "note": (
                        "Deribit DVOL index (daily candles, last 7 days). "
                        "This is a vol-level proxy, not a full calibrated surface."
                    ),
                    "candles": candles,
                }, f, indent=2)
            print(
                f"[download_deribit_snapshots_range] Saved {len(candles)} DVOL candles "
                f"→ {dvol_path.name}"
            )
        else:
            print("[download_deribit_snapshots_range] DVOL endpoint returned no data.")
    except Exception as exc:
        print(
            f"[download_deribit_snapshots_range] WARNING: DVOL fetch failed: {exc}\n"
            f"  Historical vol index is unavailable (may need different API params)."
        )

    # 3. Return all snapshot files present
    existing = sorted(DATA_DIR.glob("deribit_btc_option_chain_*.json"))
    print(
        f"[download_deribit_snapshots_range] Total snapshot files in data/: "
        f"{len(existing)}"
    )
    return existing


if __name__ == "__main__":
    print("[download_deribit] Fetching Deribit BTC option chain ...")
    snap = fetch_snapshot()
    n_total = len(snap["contracts"])
    print(
        f"[download_deribit] Raw pull: {n_total} contracts, "
        f"S={snap['underlying_price']:.0f} USD"
    )
    path = save_snapshot(snap)
    print(f"[download_deribit] Done → {path}")
