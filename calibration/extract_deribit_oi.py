"""
Extract Open Interest summary from Deribit BTC option chain snapshots.
Produces data/deribit_btc_oi_summary.json.

Uses stdlib only: json, datetime, pathlib, math.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILES = {
    "2026-04-20": DATA_DIR / "deribit_btc_option_chain_20260420.json",
    "2026-04-21": DATA_DIR / "deribit_btc_option_chain_20260421.json",
}

OUTPUT_FILE = DATA_DIR / "deribit_btc_oi_summary.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def expiry_bucket(t_years: float) -> str:
    days = t_years * 365
    if days <= 7:
        return "0-7d"
    elif days <= 30:
        return "7-30d"
    elif days <= 90:
        return "30-90d"
    else:
        return "90d+"


def moneyness_bucket(strike: float, spot: float, kind: str) -> str:
    ratio = strike / spot
    if abs(ratio - 1) < 0.02:
        return "ATM"
    if kind == "C" and ratio > 1.02:
        return "OTM_call"
    if kind == "P" and ratio < 0.98:
        return "OTM_put"
    # ITM calls and puts get lumped into ATM for simplicity:
    # call below spot → ITM call (deep), put above spot → ITM put (deep)
    # Task only defines three buckets; assign ITM to ATM band
    return "ATM"


def nearest_expiry_within(contracts: list, max_days: float, spot: float) -> list:
    """Return contracts for the nearest expiry with T*365 <= max_days, sorted by |strike/spot - 1|."""
    candidates = [c for c in contracts if c["T"] * 365 <= max_days and c["T"] > 0]
    if not candidates:
        return []
    min_T = min(c["T"] for c in candidates)
    return [c for c in candidates if abs(c["T"] - min_T) < 1e-6]


def expiry_contracts_beyond(contracts: list, min_days: float) -> list:
    """Contracts with T*365 > min_days; nearest such expiry."""
    candidates = [c for c in contracts if c["T"] * 365 > min_days]
    if not candidates:
        return []
    min_T = min(c["T"] for c in candidates)
    return [c for c in candidates if abs(c["T"] - min_T) < 1e-6]


def atm_iv(expiry_contracts: list, spot: float) -> float | None:
    """Mean mark_iv of ATM contracts (|strike/spot - 1| < 0.02) in given expiry slice."""
    atm = [c for c in expiry_contracts if abs(c["strike"] / spot - 1) < 0.02 and c["mark_iv"] is not None]
    if not atm:
        return None
    return sum(c["mark_iv"] for c in atm) / len(atm)


def delta25_iv(expiry_contracts: list, spot: float, side: str) -> float | None:
    """
    Approximate 25-delta IV:
      put  → strike closest to spot * 0.93
      call → strike closest to spot * 1.07
    Returns mark_iv of that contract.
    """
    target = spot * (0.93 if side == "put" else 1.07)
    kind_filter = "P" if side == "put" else "C"
    candidates = [c for c in expiry_contracts if c["kind"] == kind_filter and c["mark_iv"] is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(c["strike"] - target))
    return best["mark_iv"]


def nearest_30d_expiry(contracts: list, spot: float) -> list:
    """
    Find expiry slice closest to 30 calendar days for skew computation.
    Preference: expiry where |T*365 - 30| is minimised among expiries with T > 0.
    """
    expiries = {}
    for c in contracts:
        if c["T"] > 0:
            expiries.setdefault(c["expiry_date"], []).append(c)
    if not expiries:
        return []
    best_date = min(expiries.keys(), key=lambda d: abs(expiries[d][0]["T"] * 365 - 30))
    return expiries[best_date]


# ---------------------------------------------------------------------------
# per-snapshot processing
# ---------------------------------------------------------------------------

def process_snapshot(date_str: str, filepath: Path) -> dict:
    raw = json.loads(filepath.read_text())

    spot = raw["underlying_price"]
    contracts = raw["contracts"]
    n = len(contracts)

    total_oi = sum(c["open_interest"] for c in contracts)
    total_oi_usd = total_oi * spot

    # by kind
    by_kind = {"call": {"n": 0, "oi": 0.0}, "put": {"n": 0, "oi": 0.0}}
    for c in contracts:
        key = "call" if c["kind"] == "C" else "put"
        by_kind[key]["n"] += 1
        by_kind[key]["oi"] += c["open_interest"]

    for key in by_kind:
        by_kind[key]["oi_share"] = (
            round(by_kind[key]["oi"] / total_oi, 6) if total_oi > 0 else 0.0
        )

    put_call_ratio = (
        by_kind["put"]["oi"] / by_kind["call"]["oi"]
        if by_kind["call"]["oi"] > 0
        else None
    )

    # by expiry bucket
    buckets = {"0-7d": {"n": 0, "oi": 0.0},
               "7-30d": {"n": 0, "oi": 0.0},
               "30-90d": {"n": 0, "oi": 0.0},
               "90d+": {"n": 0, "oi": 0.0}}
    for c in contracts:
        b = expiry_bucket(c["T"])
        buckets[b]["n"] += 1
        buckets[b]["oi"] += c["open_interest"]

    # by moneyness
    mono = {"OTM_call": {"n": 0, "oi": 0.0},
            "ATM": {"n": 0, "oi": 0.0},
            "OTM_put": {"n": 0, "oi": 0.0}}
    for c in contracts:
        b = moneyness_bucket(c["strike"], spot, c["kind"])
        mono[b]["n"] += 1
        mono[b]["oi"] += c["open_interest"]

    # ATM IV short (nearest expiry within 30d)
    short_slice = nearest_expiry_within(contracts, 30, spot)
    atm_iv_short = atm_iv(short_slice, spot)

    # ATM IV long (nearest expiry beyond 90d)
    long_slice = expiry_contracts_beyond(contracts, 90)
    atm_iv_long = atm_iv(long_slice, spot)

    # IV skew 25d — at expiry closest to 30d
    skew_slice = nearest_30d_expiry(contracts, spot)
    put_25d = delta25_iv(skew_slice, spot, "put")
    call_25d = delta25_iv(skew_slice, spot, "call")
    iv_skew_25d = (round(put_25d - call_25d, 6)
                   if put_25d is not None and call_25d is not None
                   else None)

    return {
        "date": date_str,
        "underlying_price": spot,
        "n_contracts": n,
        "total_oi_contracts": round(total_oi, 4),
        "total_oi_btc_notional": round(total_oi, 4),
        "total_oi_usd_notional": round(total_oi_usd, 2),
        "by_kind": {
            "call": {
                "n": by_kind["call"]["n"],
                "oi": round(by_kind["call"]["oi"], 4),
                "oi_share": by_kind["call"]["oi_share"],
            },
            "put": {
                "n": by_kind["put"]["n"],
                "oi": round(by_kind["put"]["oi"], 4),
                "oi_share": by_kind["put"]["oi_share"],
            },
        },
        "put_call_ratio": round(put_call_ratio, 6) if put_call_ratio is not None else None,
        "by_expiry_bucket": {k: {"n": v["n"], "oi": round(v["oi"], 4)} for k, v in buckets.items()},
        "by_moneyness": {k: {"n": v["n"], "oi": round(v["oi"], 4)} for k, v in mono.items()},
        "atm_iv_short": round(atm_iv_short, 6) if atm_iv_short is not None else None,
        "atm_iv_long": round(atm_iv_long, 6) if atm_iv_long is not None else None,
        "iv_skew_25d": iv_skew_25d,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    snapshots = []
    for date_str, fp in sorted(INPUT_FILES.items()):
        snap = process_snapshot(date_str, fp)
        snapshots.append(snap)
        print(f"\n=== {date_str} ===")
        print(f"  Underlying price  : ${snap['underlying_price']:,.2f}")
        print(f"  N contracts       : {snap['n_contracts']}")
        print(f"  Total OI (BTC)    : {snap['total_oi_btc_notional']:,.2f}")
        print(f"  Total OI (USD)    : ${snap['total_oi_usd_notional']:,.0f}")
        print(f"  Put/Call ratio    : {snap['put_call_ratio']}")
        print(f"  ATM IV short-dated: {snap['atm_iv_short']}")
        print(f"  ATM IV long-dated : {snap['atm_iv_long']}")
        print(f"  IV skew 25d (P-C) : {snap['iv_skew_25d']}")
        print(f"  By kind           : {snap['by_kind']}")
        print(f"  By expiry bucket  : {snap['by_expiry_bucket']}")

    s0, s1 = snapshots
    oi_0, oi_1 = s0["total_oi_btc_notional"], s1["total_oi_btc_notional"]
    oi_change_pct = (oi_1 - oi_0) / oi_0 if oi_0 != 0 else None
    pcr_change = (
        (s1["put_call_ratio"] - s0["put_call_ratio"])
        if s0["put_call_ratio"] is not None and s1["put_call_ratio"] is not None
        else None
    )
    iv_short_change = (
        (s1["atm_iv_short"] - s0["atm_iv_short"])
        if s0["atm_iv_short"] is not None and s1["atm_iv_short"] is not None
        else None
    )

    diff = {
        "oi_change_pct": round(oi_change_pct, 6) if oi_change_pct is not None else None,
        "put_call_ratio_change": round(pcr_change, 6) if pcr_change is not None else None,
        "atm_iv_short_change": round(iv_short_change, 6) if iv_short_change is not None else None,
    }

    print("\n=== 20→21 deltas ===")
    print(f"  OI change %       : {diff['oi_change_pct']}")
    print(f"  PCR change        : {diff['put_call_ratio_change']}")
    print(f"  ATM IV short chg  : {diff['atm_iv_short_change']}")

    # identical-snapshot check
    if oi_0 == oi_1 and s0["atm_iv_short"] == s1["atm_iv_short"]:
        print("\n⚠ WARNING: Both snapshots produce identical numbers — one file may be a copy!")
    else:
        print("\n✓ Snapshots look genuinely different.")

    result = {
        "source": "Aggregated from data/deribit_btc_option_chain_2026042{0,1}.json",
        "snapshots": snapshots,
        "diff_20_to_21": diff,
    }

    OUTPUT_FILE.write_text(json.dumps(result, indent=2))
    print(f"\nOutput written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
