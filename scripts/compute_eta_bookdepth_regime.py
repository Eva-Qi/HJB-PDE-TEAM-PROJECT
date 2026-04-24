"""Compute regime-conditional η from bookDepth and write eta_bookdepth_regime.json.

Pipeline
--------
1. Load 28 days of BTCUSDT-bookDepth CSVs (2026-03-23 → 2026-04-19).
2. Load the matching aggTrades window to fit a 2-state HMM on 5-min returns
   so every bookDepth snapshot can be labelled risk_on (0) or risk_off (1).
3. For each regime, estimate η by inverting the Almgren-Chriss power law
   ``temp_impact_USD = η · v^α`` with α fixed at the risk_on calibrated
   value (0.42) and v = trade_size_btc / dt_years.
4. Persist the two ηs, snapshot counts and schema notes to
   ``data/eta_bookdepth_regime.json``.

No existing files are modified.  The V5 driver opts in to these values by
reading the new JSON before building ``v5_params_riskoff``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from calibration.bookdepth_impact_estimator import (
    assign_regime_to_snapshots,
    estimate_eta_per_regime_from_bookdepth,
    load_bookdepth,
)
from calibration.data_loader import compute_mid_prices, load_trades
from extensions.regime import fit_hmm


# ─── Configuration (mirrors scripts/paired_test_regime_aware_v5.py) ──────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START = "2026-03-23"
END = "2026-04-19"

# Execution horizon / sizing (identical to V5 driver)
X0 = 10.0
T = 1.0/(365.25*24)    # 1-hour horizon in years
N_STEPS = 250
DT_YEARS = T / N_STEPS
Q_PER_STEP = X0 / N_STEPS  # 0.2 BTC

# α fixed at the risk_on calibrated value.  Both regimes share this α so the
# only bookDepth-derived degree of freedom is η — avoids re-inventing another
# fallback constant for risk_off α.
ALPHA_FIXED = 0.41968422109505343   # from data/regime_conditional_impact.json

# Subsample every Nth snapshot to keep the loop inside reason (~30s walltime).
# 28 days × ~2600 snap/day ≈ 73k snapshots; subsample=4 → ~18k snapshots
# still yields thousands per regime, well above the 50-snapshot floor.
SUBSAMPLE = 4


def _fit_hmm_on_window(agg_dir: Path, start: str, end: str):
    """Load aggTrades in window, build 5-min return series, fit 2-state HMM."""
    print(f"  loading aggTrades {start}..{end}")
    trades = load_trades(agg_dir, start=start, end=end)
    print(f"  loaded {len(trades):,} trades from {trades['timestamp'].iloc[0]} "
          f"to {trades['timestamp'].iloc[-1]}")

    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    bar_ts = pd.DatetimeIndex(mid.index)
    returns = mid["log_return"].to_numpy()
    returns = returns[np.isfinite(returns)]
    # mid.index may have lost a row to the dropna; align bar_ts to returns.
    bar_ts = pd.DatetimeIndex(mid.index[np.isfinite(mid["log_return"])])

    print(f"  fitting 2-state HMM on {len(returns):,} 5-min returns…")
    regimes, state_seq = fit_hmm(returns, n_regimes=2)
    # state_seq length == returns length == len(bar_ts)
    assert len(state_seq) == len(bar_ts), (len(state_seq), len(bar_ts))

    for r in regimes:
        print(f"  {r.label:9s} sigma_mult={r.sigma:.4f} prob={r.probability:.4f}")
    return bar_ts, state_seq, regimes


def main() -> None:
    print("\n" + "=" * 80)
    print("bookDepth regime-conditional η estimator")
    print("=" * 80)

    # ── 1. Load bookDepth window ─────────────────────────────────────────
    print(f"\n[1/4] Loading bookDepth CSVs {START}..{END}")
    depth = load_bookdepth(DATA_DIR, start=START, end=END)
    dates_loaded = depth.attrs.get("dates_loaded", [])
    unique_snapshots = depth["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    print(f"  rows                    : {len(depth):,}")
    print(f"  unique snapshots        : {len(unique_snapshots):,}")
    print(f"  distinct dates          : {len(dates_loaded)}")
    print(f"  time range              : {unique_snapshots.iloc[0]} → {unique_snapshots.iloc[-1]}")

    # Verify schema once
    buckets = sorted(depth["percentage"].unique().tolist())
    expected = [-5.0, -4.0, -3.0, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
    if buckets != expected:
        raise RuntimeError(
            f"Unexpected bucket set: {buckets} (expected {expected}). "
            "Refusing to proceed — schema mismatch would corrupt η."
        )
    print(f"  buckets verified        : {buckets}")

    # ── 2. Fit HMM on matching aggTrades window ──────────────────────────
    print(f"\n[2/4] Fitting HMM on aggTrades over the bookDepth window")
    bar_ts, state_seq, regimes = _fit_hmm_on_window(DATA_DIR, START, END)
    reg_labels = [r.label for r in regimes]
    print(f"  regime order            : {reg_labels} (0=risk_on, 1=risk_off)")

    # ── 3. Assign each snapshot a regime via searchsorted ────────────────
    print(f"\n[3/4] Labelling every bookDepth snapshot")
    snap_regime = assign_regime_to_snapshots(
        pd.DatetimeIndex(unique_snapshots.values),
        bar_ts,
        state_seq,
    )
    n_pre_first = int((snap_regime < 0).sum())
    n_on = int((snap_regime == 0).sum())
    n_off = int((snap_regime == 1).sum())
    print(f"  snapshots before first 5-min bar (dropped): {n_pre_first}")
    print(f"  risk_on snapshots  : {n_on:,}")
    print(f"  risk_off snapshots : {n_off:,}")
    if n_off < 50:
        raise RuntimeError(
            f"Only {n_off} risk_off snapshots over the bookDepth window — "
            "HMM may have collapsed; refusing to fabricate η."
        )

    # ── 4. Estimate η per regime ─────────────────────────────────────────
    print(f"\n[4/4] Estimating η per regime (trade_size={Q_PER_STEP} BTC, "
          f"dt={DT_YEARS:.4e} yr, α={ALPHA_FIXED:.4f}, subsample={SUBSAMPLE})")
    results = estimate_eta_per_regime_from_bookdepth(
        depth_df=depth,
        regime_labels=snap_regime,
        trade_size_btc=Q_PER_STEP,
        dt_years=DT_YEARS,
        alpha=ALPHA_FIXED,
        snapshot_subsample=SUBSAMPLE,
    )

    out: dict = {
        "trade_size_btc": Q_PER_STEP,
        "dt_years": DT_YEARS,
        "alpha_fixed": ALPHA_FIXED,
        "alpha_source": (
            "risk_on calibrated value from data/regime_conditional_impact.json "
            "(aggregated_1min). Held fixed across both regimes so the only "
            "bookDepth-derived degree of freedom is η."
        ),
        "snapshot_subsample": SUBSAMPLE,
        "date_range": [START, END],
        "dates_loaded": dates_loaded,
        "n_snapshots_risk_on_raw": n_on,
        "n_snapshots_risk_off_raw": n_off,
        "n_snapshots_pre_first_bar_dropped": n_pre_first,
        "bookdepth_schema_notes": (
            "Binance Vision BTCUSDT-bookDepth-YYYY-MM-DD.csv. Columns: "
            "timestamp, percentage, depth, notional. 12 buckets per snapshot "
            "at % offsets {-5,-4,-3,-2,-1,-0.2,+0.2,+1,+2,+3,+4,+5}. depth is "
            "CUMULATIVE BTC from mid out to that band; notional is CUMULATIVE "
            "USD. Verified via (notional[+1]-notional[+0.2]) / "
            "(depth[+1]-depth[+0.2]) ≈ mid·1.006. NOT level-by-level L2."
        ),
        "method_notes": (
            "For each snapshot, executed a ±0.2-BTC round-trip against the "
            "cumulative ask/bid curves (linear interpolation within buckets). "
            "Per-snapshot slippage_USD averaged across the two sides, then η "
            "recovered from slippage_USD = η · (trade_size/dt)^α. Regime η is "
            "the median across snapshots within that regime — robust to the "
            "occasional book-thin outlier."
        ),
    }

    mapping = {0: "risk_on", 1: "risk_off"}
    for r_idx, diag in sorted(results.items()):
        key = mapping.get(r_idx, f"regime_{r_idx}")
        out[f"eta_{key}"] = diag.eta
        out[f"n_obs_{key}"] = diag.n_snapshots
        out[f"median_slippage_usd_{key}"] = diag.median_slippage_usd
        out[f"mean_slippage_usd_{key}"] = diag.mean_slippage_usd
        out[f"mid_price_mean_{key}"] = diag.mid_price_mean
        print(
            f"  {key}  η={diag.eta:.4e}  "
            f"median_slippage_usd={diag.median_slippage_usd:.4f}  "
            f"n_snapshots={diag.n_snapshots:,}  "
            f"mid≈{diag.mid_price_mean:.0f}"
        )

    # ── Cross-check against aggTrades η ──────────────────────────────────
    agg_path = DATA_DIR / "regime_conditional_impact.json"
    if agg_path.exists():
        with open(agg_path) as f:
            agg_json = json.load(f)
        per_regime = {r["label"]: r for r in agg_json["2_state"]["per_regime"]}
        out["crosscheck_aggtrades"] = {}
        for label in ("risk_on", "risk_off"):
            eta_agg = per_regime[label]["true_eta"]
            eta_bk = out.get(f"eta_{label}")
            source = per_regime[label]["sources"]["eta"]
            ratio = (eta_bk / eta_agg) if (eta_bk and eta_agg) else None
            out["crosscheck_aggtrades"][label] = {
                "eta_aggtrades": eta_agg,
                "eta_aggtrades_source": source,
                "eta_bookdepth": eta_bk,
                "ratio_bk_over_agg": ratio,
            }
            if ratio is not None:
                note = ""
                if ratio > 10 or ratio < 0.1:
                    note = "  *** OUT OF ORDER-OF-MAGNITUDE AGREEMENT ***"
                print(
                    f"  cross-check {label:9s}: agg={eta_agg:.4e} "
                    f"({source})  bk={eta_bk:.4e}  ratio={ratio:.2f}x{note}"
                )

    out_path = DATA_DIR / "eta_bookdepth_regime.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
