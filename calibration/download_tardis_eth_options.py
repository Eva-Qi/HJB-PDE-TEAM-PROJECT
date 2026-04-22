"""Download Deribit ETH option chain snapshots from Tardis.dev free tier.

Tardis.dev makes day-1-of-each-month data freely available (no auth required).
Each call pulls a short window (default 60s) of ETH option quote ticks,
keeps the LAST tick per instrument (most recent = freshest IV), and saves
a clean option chain JSON.

Output schema (data/tardis_deribit_options_ETH_YYYY-MM-01.json):
    {
        "date": "2025-05-01",
        "timestamp_iso": "...",
        "underlying_price": ...,
        "options": [
            {
                "instrument": "ETH-30MAY25-3000-C",
                "strike": 3000,
                "expiry": "2025-05-30",
                "T_years": 0.15,
                "kind": "C",
                "bid_price": ...,
                "ask_price": ...,
                "mark_iv": ...      # decimal (0.80 = 80% IV)
            },
            ...
        ]
    }

Usage
-----
    # Single date
    python calibration/download_tardis_eth_options.py --dates 2025-05-01

    # Range of months (inclusive)
    python calibration/download_tardis_eth_options.py --start 2025-05 --end 2026-04

    # Range with custom window
    python calibration/download_tardis_eth_options.py --start 2025-05 --end 2026-04 --duration-sec 90

Notes
-----
- Only hits /v1/data-feeds/deribit (public, no auth for day-1 dates).
- Accept-Encoding: gzip is required; Tardis returns raw gzip stream.
- ETH option instruments: regex ETH-\\d{1,2}[A-Z]{3}\\d{2}-\\d+-[CP]
- Each 60s window is ~25-40 MB compressed; we stream-decompress and discard
  non-option lines immediately to stay memory-efficient.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so we can import extensions.heston later
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARDIS_BASE = "https://api.tardis.dev/v1/data-feeds/deribit"
# ETH option instrument name, e.g. ETH-30MAY26-3000-C or ETH-3JAN25-1500-P
ETH_OPT_RE = re.compile(r"^ETH-\d{1,2}[A-Z]{3}\d{2}-\d+-[CP]$")
MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
SECONDS_PER_YEAR = 365.25 * 24 * 3600

DATA_DIR = Path(_PROJECT_ROOT) / "data"


# ---------------------------------------------------------------------------
# Instrument name parser
# ---------------------------------------------------------------------------

def _parse_instrument(name: str) -> dict | None:
    """Parse a Deribit ETH option instrument name into components.

    Returns dict with keys: strike (int), expiry (date), kind (C/P),
    or None if the name doesn't match the expected pattern.

    Example: "ETH-30MAY26-3000-C"
        strike=3000, expiry=date(2026,5,30), kind="C"
    """
    if not ETH_OPT_RE.match(name):
        return None
    parts = name.split("-")  # ["ETH", "30MAY26", "3000", "C"]
    if len(parts) != 4:
        return None
    try:
        day_part = parts[1]
        # day_part = "30MAY26"
        # Find where digits end and letters begin
        day_num = int(re.match(r"(\d+)", day_part).group(1))
        month_str = re.search(r"[A-Z]{3}", day_part).group(0)
        year_str = re.search(r"[A-Z]{3}(\d{2})$", day_part).group(1)
        year_full = 2000 + int(year_str)
        month_num = MONTH_ABBR[month_str]
        expiry_dt = date(year_full, month_num, day_num)
        strike = int(parts[2])
        kind = parts[3]  # "C" or "P"
    except (ValueError, AttributeError, KeyError):
        return None
    return {"strike": strike, "expiry": expiry_dt, "kind": kind}


# ---------------------------------------------------------------------------
# Tardis stream downloader
# ---------------------------------------------------------------------------

def download_day1_option_snapshot(
    date_str: str,
    duration_sec: int = 60,
    verbose: bool = True,
) -> Path | None:
    """Pull ETH option quote ticks from Tardis for the first N seconds of day-1.

    Parameters
    ----------
    date_str : str
        ISO date string in YYYY-MM-01 format (must be day 1 of a month).
    duration_sec : int
        How many seconds of data to pull (default 60). Tardis free tier
        allows day-1 access; keep <= 120s to limit data transfer.
    verbose : bool
        Print progress messages.

    Returns
    -------
    Path to the written JSON file, or None on failure.

    Output
    ------
    data/tardis_deribit_options_ETH_YYYY-MM-01.json  (skip-if-exists)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"tardis_deribit_options_ETH_{date_str}.json"

    if out_path.exists():
        if verbose:
            print(f"[tardis-eth] Skip {date_str} — file already exists.")
        return out_path

    # Parse date
    try:
        snap_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"[tardis-eth] ERROR: date_str must be YYYY-MM-DD, got {date_str!r}")
        return None

    if snap_date.day != 1:
        print(f"[tardis-eth] WARNING: {date_str} is not day-1. Tardis free tier only covers day-1 of each month.")

    # Build from/to timestamps
    from_ts = snap_date.strftime("%Y-%m-%dT00:00:00.000Z")
    # to = from + duration_sec seconds
    to_sec = duration_sec
    to_hour = to_sec // 3600
    to_sec_rem = to_sec % 3600
    to_min = to_sec_rem // 60
    to_sec_rem2 = to_sec_rem % 60
    to_ts = snap_date.strftime(f"%Y-%m-%dT{to_hour:02d}:{to_min:02d}:{to_sec_rem2:02d}.000Z")

    url = (
        f"{TARDIS_BASE}"
        f"?from={from_ts}"
        f"&to={to_ts}"
        f"&symbols=OPTIONS"
        f"&dataTypes=quote"
    )

    if verbose:
        print(f"[tardis-eth] {date_str}: fetching {duration_sec}s window ...")
        print(f"[tardis-eth]   URL: {url}")

    req = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "gzip",
            "User-Agent": "mf796-project/1.0",
        },
    )

    # Stream-decompress gzip response, parse NDJSON line by line
    # Keep last tick per ETH option instrument
    # Each line format: TIMESTAMP {jsonrpc json}\n
    last_tick: dict[str, dict] = {}  # instrument -> last quote data
    n_lines_total = 0
    n_option_lines = 0

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            # Tardis returns raw gzip stream; wrap in GzipFile for decompression
            raw_bytes = resp.read()

        if verbose:
            print(f"[tardis-eth] {date_str}: downloaded {len(raw_bytes)/1024/1024:.1f} MB compressed")

        # Decompress
        try:
            decompressed = gzip.decompress(raw_bytes)
        except Exception:
            # Try as raw stream (sometimes Tardis doesn't gzip if small)
            decompressed = raw_bytes

        text = decompressed.decode("utf-8", errors="replace")
        lines = text.splitlines()

        for line in lines:
            n_lines_total += 1
            if not line.strip():
                continue

            # Format (verified 2026-04-21): "TIMESTAMP {json}"
            # SPACE separator, not tab. First line begins with ISO-8601
            # date with letter 'T' (e.g., 2026-04-01T00:00:00.547...Z) so
            # split on the first space BEFORE the opening brace.
            brace_idx = line.find(" {")
            if brace_idx < 0:
                continue
            tab_idx = brace_idx  # reuse below for local_ts slice
            json_part = line[brace_idx + 1:]

            try:
                msg = json.loads(json_part)
            except json.JSONDecodeError:
                continue

            # Tardis replays raw Deribit WebSocket messages.
            # Actual format (verified empirically 2026-04-21):
            #   {"jsonrpc":"2.0","method":"subscription",
            #    "params":{"channel":"quote.ETH-30MAY26-3000-C","data":{...}}}
            if msg.get("method") != "subscription":
                continue

            params = msg.get("params", {})
            channel = params.get("channel", "")

            # Extract instrument name from channel string
            # channel examples: "quote.ETH-30MAY26-3000-C", "book.ETH-...", "ticker.ETH-..."
            # Accept quote OR ticker channels (both carry IV-adjacent data)
            if channel.startswith("quote."):
                instrument = channel[len("quote."):]
            elif channel.startswith("ticker."):
                # ticker channel may have suffix like ".raw" -> strip it
                rest = channel[len("ticker."):]
                instrument = rest.split(".")[0]
            else:
                continue

            if not ETH_OPT_RE.match(instrument):
                continue

            n_option_lines += 1
            data = params.get("data", {})
            # Keep the LAST tick for this instrument
            last_tick[instrument] = {
                "data": data,
                "local_ts": line[:tab_idx],  # the timestamp prefix before space
                "channel": channel,
            }

        if verbose:
            print(
                f"[tardis-eth] {date_str}: parsed {n_lines_total} lines, "
                f"{n_option_lines} ETH option ticks, "
                f"{len(last_tick)} unique instruments"
            )

    except urllib.error.HTTPError as exc:
        print(f"[tardis-eth] {date_str}: HTTP {exc.code} — {exc.reason}")
        if exc.code == 401:
            print("  NOTE: Tardis free tier only covers day-1 of each month. "
                  "Check that date_str ends in '-01'.")
        return None
    except Exception as exc:
        print(f"[tardis-eth] {date_str}: fetch error — {exc}")
        return None

    if not last_tick:
        print(f"[tardis-eth] {date_str}: no ETH option ticks found — snapshot NOT saved.")
        return None

    # ------------------------------------------------------------------ #
    # Build clean option chain from last ticks                           #
    # ------------------------------------------------------------------ #
    snap_date_obj = snap_date.date()

    # Estimate underlying price from index_price fields in data if available
    # or infer from mid-price and a rough delta assumption
    # We'll pick from any option's underlying_price field if present
    underlying_price: float | None = None

    options_list: list[dict] = []
    for instrument, tick_info in last_tick.items():
        parsed = _parse_instrument(instrument)
        if parsed is None:
            continue

        d = tick_info["data"]

        # Extract prices — Tardis quote fields vary; try common names
        bid_price = d.get("bestBidPrice") or d.get("bid_price") or d.get("bid") or None
        ask_price = d.get("bestAskPrice") or d.get("ask_price") or d.get("ask") or None
        mark_price = d.get("markPrice") or d.get("mark_price") or None
        mark_iv_raw = d.get("markIv") or d.get("mark_iv") or None
        und_price = d.get("underlyingPrice") or d.get("underlying_price") or None

        if und_price is not None and underlying_price is None:
            try:
                underlying_price = float(und_price)
            except (ValueError, TypeError):
                pass

        # Convert IV: Tardis sometimes gives percentage (e.g. 80.0 = 80% IV)
        mark_iv_decimal: float | None = None
        if mark_iv_raw is not None:
            try:
                iv_val = float(mark_iv_raw)
                # If > 5.0 assume it's in percentage form
                mark_iv_decimal = iv_val / 100.0 if iv_val > 5.0 else iv_val
            except (ValueError, TypeError):
                pass

        # T_years: time from snap_date to expiry
        expiry_dt = parsed["expiry"]
        days_to_expiry = (expiry_dt - snap_date_obj).days
        T_years = days_to_expiry / 365.25

        if T_years <= 0:
            continue  # already expired

        # Convert prices to float safely
        def _safe_float(v) -> float | None:
            if v is None:
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        options_list.append({
            "instrument": instrument,
            "strike": parsed["strike"],
            "expiry": expiry_dt.isoformat(),
            "T_years": round(T_years, 6),
            "kind": parsed["kind"],
            "bid_price": _safe_float(bid_price),
            "ask_price": _safe_float(ask_price),
            "mark_price": _safe_float(mark_price),
            "mark_iv": mark_iv_decimal,
        })

    # Sort by expiry then strike
    options_list.sort(key=lambda x: (x["expiry"], x["strike"]))

    if underlying_price is None:
        # Fallback: no underlying price found in ticks. Use None and warn.
        print(
            f"[tardis-eth] {date_str}: WARNING — underlying_price not found in ticks. "
            "Snapshot saved with underlying_price=null; you may need to supply it manually."
        )

    # Find a representative timestamp from the last tick
    # Use max localTimestamp across all instruments
    ts_list = [
        t["local_ts"] for t in last_tick.values() if t.get("local_ts")
    ]
    timestamp_iso = max(ts_list) if ts_list else from_ts

    snapshot = {
        "date": date_str,
        "timestamp_iso": timestamp_iso,
        "underlying_price": underlying_price,
        "options": options_list,
    }

    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    n_valid_iv = sum(1 for o in options_list if o["mark_iv"] is not None and o["mark_iv"] > 0)
    print(
        f"[tardis-eth] {date_str}: saved {len(options_list)} options "
        f"({n_valid_iv} with mark_iv) -> {out_path.name}  "
        f"S={underlying_price}"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _month_range_stdlib(start_ym: str, end_ym: str) -> list[str]:
    """Generate list of YYYY-MM-01 strings without dateutil dependency."""
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))

    dates = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        dates.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Tardis.dev Deribit ETH option snapshots (free day-1 tier)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dates",
        nargs="+",
        metavar="YYYY-MM-DD",
        help="Specific dates to download (must be day-1 of a month, e.g. 2025-05-01)",
    )
    group.add_argument(
        "--start",
        metavar="YYYY-MM",
        help="Start month for range (e.g. 2025-05)",
    )
    parser.add_argument(
        "--end",
        metavar="YYYY-MM",
        help="End month for range (e.g. 2026-04). Required with --start.",
    )
    parser.add_argument(
        "--duration-sec",
        type=int,
        default=60,
        help="Seconds of data to pull per snapshot (default 60, max 120)",
    )
    args = parser.parse_args()

    if args.duration_sec > 120:
        print("WARNING: --duration-sec capped at 120 to avoid excessive data transfer.")
        args.duration_sec = 120

    # Determine list of dates
    if args.dates:
        dates = args.dates
    elif args.start:
        if not args.end:
            parser.error("--end is required when --start is specified")
        dates = _month_range_stdlib(args.start, args.end)
    else:
        # Default: last 12 months ending this month
        today = date.today()
        end_ym = today.strftime("%Y-%m")
        ey, em = int(end_ym[:4]), int(end_ym[5:])
        sm = em - 11
        sy = ey
        if sm <= 0:
            sm += 12
            sy -= 1
        start_ym = f"{sy:04d}-{sm:02d}"
        dates = _month_range_stdlib(start_ym, end_ym)
        print(f"[tardis-eth] No dates specified; defaulting to {start_ym} -> {end_ym}")

    print(f"[tardis-eth] Will download {len(dates)} snapshots: {dates}")
    successes = 0
    failures = []

    for d in dates:
        result = download_day1_option_snapshot(d, duration_sec=args.duration_sec)
        if result is not None:
            successes += 1
        else:
            failures.append(d)

    print(f"\n[tardis-eth] Done. {successes}/{len(dates)} successful.")
    if failures:
        print(f"[tardis-eth] Failed dates: {failures}")


if __name__ == "__main__":
    main()
