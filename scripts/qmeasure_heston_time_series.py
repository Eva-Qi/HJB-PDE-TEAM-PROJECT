"""Q-measure Heston parameter time series from Tardis monthly snapshots.

For each data/tardis_deribit_options_YYYY-MM-01.json file:
    1. Load and convert to DataFrame compatible with calibrate_heston_from_options
    2. Call calibrate_heston_from_options(option_chain, underlying_price)
    3. Record (date, kappa, theta, xi, rho, v0, Feller_OK, fit_RMSE)

Output
------
data/heston_qmeasure_time_series.json
figures/heston_qmeasure_time_series.png

Report
------
For each param: mean, std, CV (std/|mean|), range (min, max)
Stability narrative: CV < 0.20 → stable, CV >= 0.20 → unstable

Usage
-----
    python scripts/qmeasure_heston_time_series.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from extensions.heston import (
    calibrate_heston_from_options,
    _heston_model_iv_batch,
)

PROJECT_ROOT = Path(_PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUT_PATH = DATA_DIR / "heston_qmeasure_time_series.json"
SECONDS_PER_YEAR = 365.25 * 24 * 3600


# ---------------------------------------------------------------------------
# Convert Tardis snapshot JSON → pandas DataFrame for calibrate_heston_from_options
# ---------------------------------------------------------------------------

def tardis_snapshot_to_dataframe(snapshot: dict) -> tuple[pd.DataFrame, float]:
    """Convert a Tardis option snapshot dict to a DataFrame.

    The calibrate_heston_from_options function expects a DataFrame with columns:
        kind (C/P), strike (float), T (years), mark_iv (decimal)
    Optional columns (used when available):
        bid_iv, ask_iv, open_interest

    Parameters
    ----------
    snapshot : dict
        Loaded from tardis_deribit_options_YYYY-MM-01.json

    Returns
    -------
    (df, S0) where df has the required columns and S0 is the underlying price.
    """
    S0 = float(snapshot.get("underlying_price") or 0.0)
    options = snapshot.get("options", [])

    if not options:
        return pd.DataFrame(), S0

    rows = []
    for opt in options:
        mark_iv = opt.get("mark_iv")
        if mark_iv is None or mark_iv <= 0:
            continue

        T_years = opt.get("T_years", 0.0)
        if T_years <= 0:
            continue

        strike = float(opt.get("strike", 0))
        if strike <= 0:
            continue

        kind = opt.get("kind", "C")

        # bid/ask in price units (BTC); we need IV
        # For Tardis quotes, bid_price/ask_price are in BTC (index units)
        # We don't have bid_iv/ask_iv directly — use mark_iv only
        # (calibrate_heston_from_options handles mark_iv fallback gracefully)
        rows.append({
            "instrument_name": opt.get("instrument", ""),
            "kind": kind,
            "strike": strike,
            "T": T_years,
            "mark_iv": float(mark_iv),
            # No open_interest in Tardis quote stream (use uniform weights)
        })

    df = pd.DataFrame(rows)
    return df, S0


# ---------------------------------------------------------------------------
# RMSE helper (mirrors deribit_qmeasure_time_series.py)
# ---------------------------------------------------------------------------

def _compute_heston_rmse(
    df: pd.DataFrame,
    S0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """Compute IV RMSE of calibrated Heston fit on the chain DataFrame."""
    df_c = df.copy()
    df_c["T_key"] = df_c["T"].round(4)

    # Resolve market_iv column
    if "market_iv" not in df_c.columns:
        df_c["market_iv"] = df_c["mark_iv"]

    all_diffs: list[float] = []
    for T_key, grp in df_c.groupby("T_key"):
        strikes = grp["strike"].values
        mkt_ivs = grp["market_iv"].values
        try:
            mdl_ivs = _heston_model_iv_batch(
                S0, strikes, T_key, r, q, kappa, theta, xi, rho, v0
            )
        except Exception:
            continue
        valid = ~np.isnan(mdl_ivs)
        if valid.sum() == 0:
            continue
        all_diffs.extend((mdl_ivs[valid] - mkt_ivs[valid]).tolist())

    if not all_diffs:
        return float("nan")
    return float(np.sqrt(np.mean(np.array(all_diffs) ** 2)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all Tardis monthly snapshot files
    snapshot_files = sorted(DATA_DIR.glob("tardis_deribit_options_*.json"))

    if not snapshot_files:
        print(
            "[qmeasure_ts] No Tardis snapshot files found in data/.\n"
            "Run: python calibration/download_tardis_options.py --start 2025-05 --end 2026-04"
        )
        return

    print(
        f"[qmeasure_ts] Found {len(snapshot_files)} Tardis snapshot(s):\n"
        + "\n".join(f"  {f.name}" for f in snapshot_files)
    )

    records: list[dict] = []

    for snap_path in snapshot_files:
        # Filename: tardis_deribit_options_YYYY-MM-DD.json
        date_str = snap_path.stem.replace("tardis_deribit_options_", "")

        print(f"\n--- Calibrating: {snap_path.name} ---")
        t0 = time.time()

        with open(snap_path) as f:
            snapshot = json.load(f)

        df, S0 = tardis_snapshot_to_dataframe(snapshot)

        if S0 <= 0:
            print(f"  SKIP: underlying_price missing or zero in {snap_path.name}")
            continue

        if df.empty:
            print(f"  SKIP: no valid options in {snap_path.name}")
            continue

        n_all = len(df)
        print(f"  S0={S0:.0f}, n_options={n_all}")

        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            try:
                params = calibrate_heston_from_options(
                    df,
                    underlying_price=S0,
                    r=0.0,
                    q=0.0,
                    n_starts=8,
                    delta_filter=(0.10, 0.90),
                    use_bid_ask=False,   # Tardis quotes don't include bid_iv/ask_iv in decimal
                    use_oi_weights=False,  # no OI in Tardis quote stream
                    seed=42,
                )
            except Exception as exc:
                print(f"  WARNING: calibration failed: {exc}")
                continue

        elapsed = time.time() - t0
        feller_lhs = 2.0 * params.kappa * params.theta
        feller_rhs = params.xi ** 2
        feller_ok = feller_lhs >= feller_rhs
        feller_margin = feller_lhs - feller_rhs

        # RMSE on the filtered chain
        df_filt = df[
            df["mark_iv"].notna()
            & (df["mark_iv"] > 0)
            & (df["kind"] == "C")
        ].copy()
        T_min = 7 / 365.25
        T_max = 180 / 365.25
        df_filt = df_filt[(df_filt["T"] >= T_min) & (df_filt["T"] <= T_max)].copy()
        df_filt["market_iv"] = df_filt["mark_iv"]

        rmse = _compute_heston_rmse(
            df_filt, S0,
            params.kappa, params.theta, params.xi, params.rho, params.v0,
        ) if not df_filt.empty else float("nan")

        record = {
            "date": date_str,
            "kappa": round(params.kappa, 6),
            "theta": round(params.theta, 6),
            "xi": round(params.xi, 6),
            "rho": round(params.rho, 6),
            "v0": round(params.v0, 6),
            "feller_ok": feller_ok,
            "feller_margin": round(feller_margin, 6),
            "fit_rmse": round(rmse, 6) if not np.isnan(rmse) else None,
            "n_options_used": n_all,
        }
        records.append(record)

        print(
            f"  kappa={params.kappa:.4f}  theta={params.theta:.4f}  "
            f"xi={params.xi:.4f}  rho={params.rho:.4f}  v0={params.v0:.4f}\n"
            f"  feller={'OK' if feller_ok else 'VIOLATED'}  "
            f"fit_rmse={('nan' if np.isnan(rmse) else f'{rmse:.4f}')}  "
            f"elapsed={elapsed:.1f}s"
        )
        for wr in w_list:
            print(f"  WARN: {wr.message}")

    # ------------------------------------------------------------------ #
    # Save JSON                                                           #
    # ------------------------------------------------------------------ #
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[qmeasure_ts] Saved {len(records)} records → {OUTPUT_PATH.name}")

    if not records:
        print("[qmeasure_ts] No records to analyse.")
        return

    ts_df = pd.DataFrame(records)
    n = len(ts_df)

    # ------------------------------------------------------------------ #
    # Stability report                                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("Q-MEASURE HESTON PARAMETER TIME SERIES (12-MONTH TARDIS DATA)")
    print("=" * 65)
    print(ts_df[["date", "kappa", "theta", "xi", "rho", "v0",
                  "feller_ok", "fit_rmse"]].to_string(index=False))

    if n > 1:
        print("\n=== Stability Metrics ===")
        print(f"{'Param':8s}  {'Mean':>10s}  {'Std':>10s}  {'CV':>8s}  {'Min':>10s}  {'Max':>10s}")
        print("-" * 65)

        stability_summary = {}
        for col in ["kappa", "theta", "xi", "rho", "v0"]:
            vals = ts_df[col].dropna().values
            if len(vals) == 0:
                continue
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            cv = sd / abs(mu) if abs(mu) > 1e-10 else float("nan")
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            stability_summary[col] = {"mean": mu, "std": sd, "cv": cv, "min": vmin, "max": vmax}
            print(
                f"{col:8s}  {mu:10.4f}  {sd:10.4f}  {cv:8.4f}  {vmin:10.4f}  {vmax:10.4f}"
            )

        # Biggest month-to-month swing
        print("\n=== Biggest Month-to-Month Swing ===")
        for col in ["kappa", "theta", "xi", "rho", "v0"]:
            vals = ts_df[col].dropna().values
            dates = ts_df.loc[ts_df[col].notna(), "date"].values
            if len(vals) < 2:
                continue
            diffs = np.abs(np.diff(vals))
            if len(diffs) == 0:
                continue
            idx = int(np.argmax(diffs))
            print(
                f"  {col:8s}: max |Δ| = {diffs[idx]:.4f}  "
                f"({dates[idx]} → {dates[idx+1]})"
            )

        # Feller compliance
        feller_ok_count = int(ts_df["feller_ok"].sum())
        print(
            f"\n=== Feller Compliance: {feller_ok_count}/{n} months satisfied (2κθ ≥ ξ²) ==="
        )

        # Narrative
        print("\n=== Part D Narrative ===")
        all_cvs = [stability_summary[c]["cv"] for c in ["kappa", "theta", "xi", "rho", "v0"]
                   if c in stability_summary and not np.isnan(stability_summary[c]["cv"])]
        if all_cvs:
            max_cv = max(all_cvs)
            if max_cv < 0.20:
                print(
                    "STABLE: All Q-measure Heston parameters have CV < 0.20 across 12 months.\n"
                    "Single snapshot calibration is reliable for Part D analysis.\n"
                    "The Q-measure parameter set is temporally stable; one calibration\n"
                    "suffices for the execution-cost framework."
                )
            elif max_cv < 0.40:
                print(
                    "MODERATELY UNSTABLE: At least one Q-measure parameter has CV in [0.20, 0.40).\n"
                    "Parameters show material drift over 12 months — periodic re-calibration\n"
                    "(e.g., monthly) is advisable for production execution."
                )
            else:
                print(
                    "UNSTABLE: At least one Q-measure parameter has CV >= 0.40.\n"
                    "Q-measure Heston params drift materially over 12 months.\n"
                    "Live re-calibration is needed for production execution; a single\n"
                    "snapshot captures only a regime-specific parameter set."
                )
    else:
        print("\n[qmeasure_ts] Only 1 record — need ≥2 months for stability analysis.")

    # ------------------------------------------------------------------ #
    # Plot                                                               #
    # ------------------------------------------------------------------ #
    if n > 1:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            # Parse dates — format is YYYY-MM-DD
            try:
                dates_dt = pd.to_datetime(ts_df["date"], format="%Y-%m-%d")
            except Exception:
                dates_dt = pd.to_datetime(ts_df["date"])

            params_to_plot = [
                ("kappa", "κ (mean reversion speed)", "steelblue"),
                ("rho",   "ρ (price–variance correlation)", "firebrick"),
                ("xi",    "ξ (vol-of-vol)", "darkorange"),
                ("v0",    "v₀ (initial variance)", "mediumseagreen"),
            ]

            fig, axes = plt.subplots(
                len(params_to_plot), 1,
                figsize=(11, 2.8 * len(params_to_plot)),
                sharex=True,
            )
            if len(params_to_plot) == 1:
                axes = [axes]

            for ax, (col, label, color) in zip(axes, params_to_plot):
                ax.plot(
                    dates_dt, ts_df[col],
                    marker="o", color=color, linewidth=2.0, markersize=6,
                )
                ax.set_ylabel(label, fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="x", rotation=30)

            axes[-1].set_xlabel("Date", fontsize=11)
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            axes[-1].xaxis.set_major_locator(mdates.MonthLocator())

            fig.suptitle(
                "Heston Q-measure Parameters Over Time\n"
                "(Deribit BTC Options, Tardis Free Tier, Monthly Snapshots)",
                fontsize=13,
                y=1.01,
            )
            fig.tight_layout()

            plot_path = FIGURES_DIR / "heston_qmeasure_time_series.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n[qmeasure_ts] Plot saved → {plot_path}")
        except ImportError:
            print("[qmeasure_ts] matplotlib not available; skipping plot.")
        except Exception as exc:
            print(f"[qmeasure_ts] Plot error: {exc}")


if __name__ == "__main__":
    main()
