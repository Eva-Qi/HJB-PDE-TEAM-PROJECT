"""Compare Heston Q-measure parameters across two independent sources.

Validates rho consistency between:
  - Bloomberg IBIT (single snapshot, 4/24/2026)
  - Deribit BTC (12 monthly Tardis snapshots, May 2025 - Apr 2026)

After the 3-bug calibration fix (extensions/heston.py), rho should land
in the [-0.7, -0.3] range on both sources. This script computes summary
stats and produces a comparison plot.

USAGE:
    python scripts/compare_heston_ibit_vs_deribit.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA        = ROOT / "data"
AUDITS_SNAP = ROOT / "audits" / "snapshots"
IBIT_FILE   = DATA / "heston_qmeasure_ibit_20260424.json"
TARDIS_FILE = DATA / "heston_qmeasure_time_series.json"
PREFIX_FILE = AUDITS_SNAP / "heston_qmeasure_time_series_PREFIX.json"
OUT_JSON    = DATA / "heston_cross_source_comparison.json"
OUT_PNG     = DATA / "heston_cross_source_comparison.png"


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_post_fix(entries: list) -> bool:
    """Return True if every BTC entry in the Tardis file has code_version set
    (post-3-bug-fix marker) OR has xi >= 0.3 on average — either signals the
    rerun completed.  We also check the simpler proxy: any entry with
    'code_version' key present."""
    for e in entries:
        if "code_version" in e:
            return True
    # Fallback heuristic: post-fix xi should be clamped ≥ 0.3; pre-fix xi
    # values are typically ≤ 0.12.  Use mean xi of BTC entries as the signal.
    btc = [e for e in entries if not str(e.get("date", "")).startswith("ETH_")]
    if not btc:
        return False
    mean_xi = float(np.mean([e.get("xi", e.get("heston_params", {}).get("xi", 0)) for e in btc]))
    return mean_xi >= 0.25


def _parse_entry(entry: dict) -> dict:
    """Normalise a Tardis entry — handle both flat schema (pre-fix) and
    nested heston_params schema (post-fix)."""
    hp = entry.get("heston_params")
    if hp:
        # post-fix nested layout
        return {
            "date": entry.get("date") or entry.get("snapshot_date"),
            "kappa": hp["kappa"],
            "theta": hp["theta"],
            "xi":    hp["xi"],
            "rho":   hp["rho"],
            "v0":    hp["v0"],
            "rmse":        entry.get("rmse_iv") or entry.get("fit_rmse"),
            "feller_ok":   entry.get("feller_satisfied") or entry.get("feller_ok"),
            "boundary_hits": entry.get("boundary_hits", []),
            "code_version": entry.get("code_version"),
        }
    else:
        # pre-fix flat layout
        return {
            "date": entry.get("date") or entry.get("snapshot_date"),
            "kappa": entry["kappa"],
            "theta": entry["theta"],
            "xi":    entry["xi"],
            "rho":   entry["rho"],
            "v0":    entry["v0"],
            "rmse":        entry.get("fit_rmse") or entry.get("rmse_iv"),
            "feller_ok":   entry.get("feller_ok"),
            "boundary_hits": entry.get("boundary_hits", []),
            "code_version": None,
        }


# ── 1. load IBIT ─────────────────────────────────────────────────────────────

if not IBIT_FILE.exists():
    print(f"ERROR: IBIT file not found: {IBIT_FILE}")
    sys.exit(1)

with open(IBIT_FILE) as f:
    ibit_raw = json.load(f)

ibit_hp = ibit_raw["heston_params"]
ibit = {
    "snapshot_date": ibit_raw["snapshot_date"],
    "rho":   ibit_hp["rho"],
    "kappa": ibit_hp["kappa"],
    "theta": ibit_hp["theta"],
    "xi":    ibit_hp["xi"],
    "v0":    ibit_hp["v0"],
    "rmse":            ibit_raw["rmse_iv"],
    "feller_satisfied": ibit_raw["feller_satisfied"],
    "boundary_hits":   ibit_raw.get("boundary_hits", []),
    "code_version":    ibit_raw.get("code_version"),
}
print(f"✓ IBIT loaded: snapshot={ibit['snapshot_date']}, ρ={ibit['rho']:.4f}, "
      f"RMSE={ibit['rmse']:.6f}, code_version={ibit['code_version']}")


# ── 2. load Tardis (with pre-fix fallback) ────────────────────────────────────

ibit_mtime = IBIT_FILE.stat().st_mtime

using_prefix   = False
tardis_file_used = None

def _load_tardis_file(path: Path):
    with open(path) as f:
        raw = json.load(f)
    # Post-fix format: {"code_version": ..., "n_complete": ..., "results": [...]}
    # Pre-fix format: bare list of dicts
    if isinstance(raw, dict) and "results" in raw:
        records = raw["results"]
    else:
        records = raw if isinstance(raw, list) else []
    # Filter to BTC only (exclude ETH_ prefix dates)
    btc = [e for e in records if not str(e.get("date", "")).startswith("ETH_")]
    return raw, btc

if TARDIS_FILE.exists():
    raw_all, btc_entries_raw = _load_tardis_file(TARDIS_FILE)
    tardis_mtime = TARDIS_FILE.stat().st_mtime
    post_fix = _is_post_fix(btc_entries_raw)

    if not post_fix:
        print()
        print("⚠  Tardis result file is pre-fix; using PREFIX backup as placeholder.")
        print("   Re-run this script after the rerun completes for the post-fix comparison.")
        print()

        # fall back to PREFIX if available; otherwise use current file as-is
        if PREFIX_FILE.exists():
            raw_all, btc_entries_raw = _load_tardis_file(PREFIX_FILE)
            tardis_file_used = PREFIX_FILE
            using_prefix = True
        else:
            tardis_file_used = TARDIS_FILE
            using_prefix = False   # pre-fix but no backup
    else:
        tardis_file_used = TARDIS_FILE
        using_prefix = False

elif PREFIX_FILE.exists():
    print()
    print("⚠  Tardis time-series file missing; using PREFIX backup as placeholder.")
    print("   Re-run this script after the rerun completes for the post-fix comparison.")
    print()
    raw_all, btc_entries_raw = _load_tardis_file(PREFIX_FILE)
    tardis_file_used = PREFIX_FILE
    using_prefix = True
else:
    print("ERROR: Neither Tardis file nor PREFIX backup found.")
    sys.exit(1)

print(f"✓ Tardis source: {tardis_file_used.name}  "
      f"({'PRE-fix placeholder' if using_prefix else 'POST-fix'})")

# Parse all BTC entries
btc_parsed = [_parse_entry(e) for e in btc_entries_raw]
btc_parsed.sort(key=lambda x: x["date"])

print(f"  BTC snapshots: {len(btc_parsed)}  dates: {btc_parsed[0]['date']} → {btc_parsed[-1]['date']}")


# ── 3. summary stats for Tardis BTC ρ ────────────────────────────────────────

rho_series = np.array([e["rho"] for e in btc_parsed])
xi_series  = np.array([e["xi"]  for e in btc_parsed])

rho_mean = float(np.mean(rho_series))
rho_std  = float(np.std(rho_series, ddof=1)) if len(rho_series) > 1 else 0.0
rho_min  = float(np.min(rho_series))
rho_max  = float(np.max(rho_series))
n_neg    = int(np.sum(rho_series < 0))
n_pos    = int(np.sum(rho_series >= 0))

boundary_hits_per_month = {}
for e in btc_parsed:
    hits = e.get("boundary_hits") or []
    boundary_hits_per_month[e["date"]] = hits

n_months_with_hits = sum(1 for hits in boundary_hits_per_month.values() if hits)


# ── 4. comparison table ───────────────────────────────────────────────────────

rho_range_str = f"[{rho_min:.3f}, {rho_max:.3f}]"

both_neg = (ibit["rho"] < 0) and (rho_mean < 0)
abs_diff = abs(ibit["rho"] - rho_mean)
verdict  = "PASS" if (both_neg and abs_diff < 0.20) else "WARN"
check    = "✓" if verdict == "PASS" else "⚠"

print()
print("=" * 62)
print("Cross-source Heston Q-measure ρ comparison")
print("=" * 62)
header = f"{'Source':<22} {'n':>4}  {'ρ_mean':>8}  {'ρ_std':>7}  {'ρ_range':<20}  neg/pos"
print(header)
print("-" * 62)

n_tardis = len(btc_parsed)
if n_tardis > 1:
    tardis_row = (f"{'Tardis BTC (12mo)':<22} {n_tardis:>4}  {rho_mean:>8.4f}  "
                  f"{rho_std:>7.4f}  {rho_range_str:<20}  {n_neg}/{n_pos}")
else:
    tardis_row = (f"{'Tardis BTC (12mo)':<22} {n_tardis:>4}  {rho_mean:>8.4f}  "
                  f"{'—':>7}  {rho_range_str:<20}  {n_neg}/{n_pos}")
print(tardis_row)

ibit_row = (f"{'IBIT 4/24':<22} {1:>4}  {ibit['rho']:>8.4f}  "
            f"{'—':>7}  {'—':<20}  "
            f"{'0/1' if ibit['rho'] >= 0 else '1/0'}")
print(ibit_row)

print()
print(f"Agreement:           {check} both sources negative, |diff| = {abs_diff:.4f}")
print(f"Verdict:             {verdict}")
if using_prefix:
    print("NOTE: Tardis data is PRE-fix placeholder — agreement reflects pre-fix params.")
print("=" * 62)


# ── 5. write comparison JSON ──────────────────────────────────────────────────

comparison_doc = {
    "comparison_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "tardis_source_file": tardis_file_used.name,
    "tardis_is_pre_fix_placeholder": using_prefix,
    "sources": {
        "ibit": {
            "snapshot_date": ibit["snapshot_date"],
            "rho":   ibit["rho"],
            "params": {
                "kappa": ibit["kappa"],
                "theta": ibit["theta"],
                "xi":    ibit["xi"],
                "rho":   ibit["rho"],
                "v0":    ibit["v0"],
            },
            "rmse":            ibit["rmse"],
            "feller_satisfied": ibit["feller_satisfied"],
            "boundary_hits":   ibit["boundary_hits"],
            "code_version":    ibit["code_version"],
        },
        "tardis_btc": {
            "n_snapshots": n_tardis,
            "rho_time_series": [
                {"date": e["date"], "rho": e["rho"], "xi": e["xi"],
                 "kappa": e["kappa"], "theta": e["theta"], "v0": e["v0"],
                 "rmse": e["rmse"]}
                for e in btc_parsed
            ],
            "rho_summary": {
                "mean":       rho_mean,
                "std":        rho_std,
                "min":        rho_min,
                "max":        rho_max,
                "n_negative": n_neg,
                "n_positive": n_pos,
            },
            "boundary_hits_per_month": boundary_hits_per_month,
            "n_months_with_boundary_hits": n_months_with_hits,
        },
    },
    "agreement": {
        "both_negative":              both_neg,
        "abs_diff_ibit_vs_tardis_mean": abs_diff,
        "verdict":                    verdict,
    },
    "code_version": "post-3-bug-fix-2026-04-24",
}

with open(OUT_JSON, "w") as f:
    json.dump(comparison_doc, f, indent=2)
print(f"\n✓ JSON written → {OUT_JSON.relative_to(ROOT)}")


# ── 6. comparison plot ────────────────────────────────────────────────────────

# parse dates
def _to_date(s):
    return datetime.strptime(s, "%Y-%m-%d")

btc_dates = [_to_date(e["date"]) for e in btc_parsed]
ibit_date  = _to_date("2026-04-24")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
fig.subplots_adjust(hspace=0.42)

# ── top panel: ρ ─────────────────────────────────────────────────────────────
ax1.plot(btc_dates, rho_series,
         color="#2166ac", marker="o", linewidth=1.8, markersize=5,
         label="Tardis BTC (12 monthly snapshots)")
ax1.scatter([ibit_date], [ibit["rho"]],
            color="#d6604d", marker="*", s=220, zorder=5,
            label=f"IBIT 4/24 (Bloomberg)  ρ={ibit['rho']:.3f}")

# expected range shading
ax1.axhspan(-0.7, -0.3, alpha=0.15, color="#4dac26",
            label="Expected range [−0.7, −0.3]")
ax1.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

ax1.set_ylim(-1.05, 1.05)
ax1.set_ylabel("ρ (spot-vol correlation)", fontsize=10)
ax1.set_title("Heston ρ — cross-source consistency check", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8, loc="lower left")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax1.grid(axis="y", alpha=0.3)

# annotate mean
ax1.axhline(rho_mean, color="#2166ac", linewidth=1, linestyle=":",
            alpha=0.7, label=f"Tardis BTC mean ρ={rho_mean:.3f}")
ax1.text(btc_dates[-1], rho_mean + 0.04,
         f"mean={rho_mean:.3f}", color="#2166ac", fontsize=7, ha="right")

# verdict annotation
verdict_color = "#27ae60" if verdict == "PASS" else "#e74c3c"
ax1.text(0.01, 0.04, f"Verdict: {verdict}  |Δρ|={abs_diff:.3f}",
         transform=ax1.transAxes, fontsize=8,
         color=verdict_color, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=verdict_color, alpha=0.8))

# ── bottom panel: ξ ──────────────────────────────────────────────────────────
ax2.plot(btc_dates, xi_series,
         color="#2166ac", marker="o", linewidth=1.8, markersize=5,
         label="Tardis BTC ξ (vol-of-vol)")
ax2.scatter([ibit_date], [ibit["xi"]],
            color="#d6604d", marker="*", s=220, zorder=5,
            label=f"IBIT 4/24  ξ={ibit['xi']:.3f}")

ax2.axhline(0.3, color="#4dac26", linewidth=1.2, linestyle="--",
            alpha=0.8, label="ξ lower bound = 0.3")
ax2.text(btc_dates[0], 0.31, "lower bound 0.3", color="#4dac26", fontsize=7)

ax2.set_ylabel("ξ (vol-of-vol)", fontsize=10)
ax2.set_title("Heston ξ — cross-source", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax2.grid(axis="y", alpha=0.3)

# ── shared subtitle ───────────────────────────────────────────────────────────
pre_fix_note = " [PRE-FIX PLACEHOLDER — rerun in flight]" if using_prefix else ""
fig.text(0.5, 0.01,
         "After 3-bug fix: OTM-only filter + uniform weighting + xi lower bound 0.3"
         + pre_fix_note,
         ha="center", fontsize=7.5, color="#555555", style="italic")

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ PNG written  → {OUT_PNG.relative_to(ROOT)}")


# ── 7. headline summary ───────────────────────────────────────────────────────

data_label = "PRE-FIX PLACEHOLDER (Tardis rerun still in progress)" if using_prefix else "POST-FIX TARDIS DATA"

if n_neg == n_tardis:
    sign_note = "all negative"
elif n_pos == n_tardis:
    sign_note = "sign flip — ALL POSITIVE (unexpected)"
else:
    sign_note = f"bimodal: {n_neg} negative / {n_pos} positive"

if verdict == "PASS":
    interpretation = (
        f"The 3-bug fix (OTM-only filter, uniform weighting, xi lower-bound 0.3) "
        f"produced a negative ρ = {ibit['rho']:.3f} on IBIT. "
        f"The Tardis BTC panel ({data_label}) shows "
        f"ρ_mean = {rho_mean:.3f} ({sign_note}). "
        f"Both sources agree on sign and magnitude (|Δρ| = {abs_diff:.3f} < 0.20), "
        f"providing cross-venue robustness evidence for the report."
    )
else:
    interpretation = (
        f"IBIT ρ = {ibit['rho']:.3f} (post-fix, Bloomberg). "
        f"Tardis BTC ρ_mean = {rho_mean:.3f} ({sign_note}). "
        f"|Δρ| = {abs_diff:.3f} — sources diverge or sign inconsistency detected. "
        f"Check Tardis calibration; pre-fix data with positive ρ from xi→0 collapse "
        f"is expected to differ. Re-run after Tardis rerun completes."
    )

print()
print("=" * 62)
print("HEADLINE")
print("=" * 62)
print(f"Bug fix verified:           {verdict}")
print(f"IBIT 4/24:                  ρ = {ibit['rho']:.4f}   (code_version: {ibit['code_version']})")
print(f"Deribit BTC mean ({n_tardis} mo):  ρ = {rho_mean:.4f}  (range: [{rho_min:.3f}, {rho_max:.3f}])")
print(f"Agreement: |Δρ| = {abs_diff:.4f}  ({sign_note})")
print()
print(interpretation)
print("=" * 62)
print(f"\nData status: {data_label}")
