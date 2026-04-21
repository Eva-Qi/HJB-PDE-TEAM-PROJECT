"""Q-measure Heston parameter time series from Deribit daily snapshots.

Loads every ``data/deribit_btc_option_chain_YYYYMMDD.json`` file present,
runs ``calibrate_heston_from_options`` on each, and outputs a time-series
DataFrame of calibrated parameters.

Output
------
``data/heston_qmeasure_time_series.json`` — list of dicts with columns:
    date, kappa, theta, xi, rho, v0, feller_margin, fit_rmse

Plots (if > 1 snapshot available):
    κ, ρ, ξ time series saved to ``figures/heston_qmeasure_timeseries.png``

Report (always printed):
    std(κ), std(ρ), std(ξ) across days  — stability metric

Usage
-----
    python scripts/deribit_qmeasure_time_series.py

Notes
-----
Deribit's public REST API provides only the CURRENT option chain snapshot;
no historical chains are accessible without a paid data licence.  To build
a genuine multi-day series, run ``python calibration/download_deribit.py``
once per calendar day.  This script processes whatever snapshots are already
present in ``data/``.
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

from calibration.download_deribit import snapshot_to_dataframe
from extensions.heston import calibrate_heston_from_options, _heston_model_iv_batch

PROJECT_ROOT = Path(_PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUT_PATH = DATA_DIR / "heston_qmeasure_time_series.json"


# ---------------------------------------------------------------------------
# Helper: compute RMSE of Heston fit on the filtered chain
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
    """Compute IV RMSE of calibrated Heston model on the chain DataFrame.

    DataFrame must have columns: strike, T, market_iv (resolved IV used in
    calibration).  Groups by expiry, runs one FFT per expiry.
    """
    df_c = df.copy()
    df_c["T_key"] = df_c["T"].round(4)

    all_diffs: list[float] = []
    for T_key, grp in df_c.groupby("T_key"):
        strikes = grp["strike"].values
        mkt_ivs = grp["market_iv"].values if "market_iv" in grp.columns else grp["mark_iv"].values
        try:
            mdl_ivs = _heston_model_iv_batch(S0, strikes, T_key, r, q, kappa, theta, xi, rho, v0)
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

    # Discover all daily snapshot files
    snapshot_files = sorted(DATA_DIR.glob("deribit_btc_option_chain_*.json"))
    if not snapshot_files:
        print(
            "[deribit_qmeasure_time_series] No snapshot files found in data/.\n"
            "Run: python calibration/download_deribit.py"
        )
        return

    print(
        f"[deribit_qmeasure_time_series] Found {len(snapshot_files)} snapshot(s): "
        + ", ".join(f.name for f in snapshot_files)
    )

    records: list[dict] = []

    for snap_path in snapshot_files:
        # Extract date string from filename (deribit_btc_option_chain_YYYYMMDD.json)
        stem = snap_path.stem  # e.g. "deribit_btc_option_chain_20260418"
        date_str = stem.split("_")[-1]  # "20260418"

        print(f"\n--- Calibrating snapshot: {snap_path.name} ---")
        t0 = time.time()

        with open(snap_path) as f:
            snap = json.load(f)

        df = snapshot_to_dataframe(snap)
        S0 = float(snap["underlying_price"])
        print(f"  S0={S0:.0f}, n_contracts={len(df)}")

        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            try:
                params = calibrate_heston_from_options(
                    df,
                    underlying_price=S0,
                    r=0.0,
                    q=0.0,
                    n_starts=5,
                    delta_filter=(0.10, 0.90),
                    use_bid_ask=True,
                    use_oi_weights=True,
                    seed=42,
                )
            except Exception as exc:
                print(f"  WARNING: calibration failed: {exc}")
                continue

        elapsed = time.time() - t0
        feller_margin = 2.0 * params.kappa * params.theta - params.xi**2

        # Build filtered df with market_iv for RMSE calculation
        # Re-run the same filtering logic to get aligned chain
        df_filt = df[df["mark_iv"].notna() & (df["mark_iv"] > 0) & (df["kind"] == "C")].copy()
        T_min = 7 / 365.25
        T_max = 180 / 365.25
        df_filt = df_filt[(df_filt["T"] >= T_min) & (df_filt["T"] <= T_max)].copy()

        # Resolve market_iv for RMSE (mid-IV if available, else mark_iv)
        if "bid_iv" in df_filt.columns and "ask_iv" in df_filt.columns:
            bid_ok = df_filt["bid_iv"].notna() & (df_filt["bid_iv"] > 0)
            ask_ok = df_filt["ask_iv"].notna() & (df_filt["ask_iv"] > 0)
            df_filt["market_iv"] = df_filt["mark_iv"].copy()
            df_filt.loc[bid_ok & ask_ok, "market_iv"] = (
                df_filt.loc[bid_ok & ask_ok, "bid_iv"]
                + df_filt.loc[bid_ok & ask_ok, "ask_iv"]
            ) / 2.0
        else:
            df_filt["market_iv"] = df_filt["mark_iv"]

        rmse = _compute_heston_rmse(
            df_filt, S0,
            params.kappa, params.theta, params.xi, params.rho, params.v0,
        )

        record = {
            "date": date_str,
            "kappa": round(params.kappa, 6),
            "theta": round(params.theta, 6),
            "xi": round(params.xi, 6),
            "rho": round(params.rho, 6),
            "v0": round(params.v0, 6),
            "feller_margin": round(feller_margin, 6),
            "fit_rmse": round(rmse, 6) if not np.isnan(rmse) else None,
        }
        records.append(record)

        print(
            f"  kappa={params.kappa:.4f}  theta={params.theta:.4f}  "
            f"xi={params.xi:.4f}  rho={params.rho:.4f}  v0={params.v0:.4f}\n"
            f"  feller_margin={feller_margin:.4f}  fit_rmse={rmse:.4f}  "
            f"elapsed={elapsed:.1f}s"
        )

        if w_list:
            for wr in w_list:
                print(f"  WARNING: {wr.message}")

    # ------------------------------------------------------------------ #
    # Save JSON                                                           #
    # ------------------------------------------------------------------ #
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[deribit_qmeasure_time_series] Saved {len(records)} records → {OUTPUT_PATH.name}")

    if not records:
        print("[deribit_qmeasure_time_series] No records to report.")
        return

    ts_df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Stability report                                                    #
    # ------------------------------------------------------------------ #
    n = len(ts_df)
    print("\n=== Q-measure parameter time series ===")
    print(ts_df[["date", "kappa", "theta", "xi", "rho", "v0", "feller_margin", "fit_rmse"]].to_string(index=False))
    print()

    if n > 1:
        print("=== Stability metrics (std across days) ===")
        for col in ["kappa", "theta", "xi", "rho", "v0"]:
            print(f"  std({col:5s}) = {ts_df[col].std():.4f}")
    else:
        print(
            f"=== Single snapshot (date={ts_df['date'].iloc[0]}) ===\n"
            f"  Only 1 snapshot available — stability metrics require multiple days.\n"
            f"  Run this script again after the next daily download to build time series."
        )

    # ------------------------------------------------------------------ #
    # Plot if > 1 snapshot                                               #
    # ------------------------------------------------------------------ #
    if n > 1:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            dates = pd.to_datetime(ts_df["date"], format="%Y%m%d")

            for ax, col, label, color in zip(
                axes,
                ["kappa", "rho", "xi"],
                ["κ (mean reversion)", "ρ (correlation)", "ξ (vol-of-vol)"],
                ["steelblue", "firebrick", "darkorange"],
            ):
                ax.plot(dates, ts_df[col], marker="o", color=color, linewidth=1.5)
                ax.set_ylabel(label, fontsize=11)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Date")
            fig.suptitle("Heston Q-measure parameter time series (Deribit BTC)", fontsize=13)
            fig.tight_layout()

            plot_path = FIGURES_DIR / "heston_qmeasure_timeseries.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"\n[deribit_qmeasure_time_series] Plot saved → {plot_path.name}")
        except ImportError:
            print("[deribit_qmeasure_time_series] matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
