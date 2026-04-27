"""Cross-asset comparison: ETH vs BTC Q-measure Heston parameters.

Loads both monthly Q-measure result files and produces a 4-panel comparison
figure showing kappa, rho, xi, v0 for BTC and ETH on the same axes over 12 months.

Inputs
------
data/heston_qmeasure_time_series.json       (BTC — from qmeasure_heston_time_series.py)
data/heston_qmeasure_eth_time_series.json   (ETH — from qmeasure_heston_eth_time_series.py)

Output
------
figures/heston_qmeasure_eth_vs_btc.png

Report (printed to stdout)
------
- ETH ρ vs BTC ρ: both negative -> leverage effect generalises across assets
- κ range comparison
- Feller violation counts for each asset
- Seasonality / cross-asset co-movement commentary

Usage
-----
    python scripts/compare_heston_eth_vs_btc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(_PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

BTC_PATH = DATA_DIR / "heston_qmeasure_time_series.json"
ETH_PATH = DATA_DIR / "heston_qmeasure_eth_time_series.json"
OUTPUT_PNG = FIGURES_DIR / "heston_qmeasure_eth_vs_btc.png"


def _load_results(path: Path, asset: str) -> pd.DataFrame | None:
    """Load monthly Heston JSON results and return a clean DataFrame."""
    if not path.exists():
        print(f"[compare] WARNING: {path.name} not found — {asset} will be skipped.")
        return None
    with open(path) as f:
        records = json.load(f)
    if not records:
        print(f"[compare] WARNING: {path.name} is empty — {asset} will be skipped.")
        return None

    df = pd.DataFrame(records)
    # Standardise date column to datetime — handle YYYYMMDD or YYYY-MM-DD
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"].astype(str), infer_datetime_format=True, errors="coerce")
    # Drop rows where calibration clearly failed (kappa/rho missing)
    df = df.dropna(subset=["kappa", "rho"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _summary_stats(df: pd.DataFrame, param: str) -> dict:
    vals = df[param].dropna().values
    if len(vals) == 0:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "n": len(vals),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    btc_df = _load_results(BTC_PATH, "BTC")
    eth_df = _load_results(ETH_PATH, "ETH")

    if btc_df is None and eth_df is None:
        print("[compare] No data available for either asset. Exiting.")
        return

    # ------------------------------------------------------------------ #
    # Cross-asset summary report                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("CROSS-ASSET Q-MEASURE HESTON COMPARISON: ETH vs BTC")
    print("=" * 70)

    params_of_interest = ["kappa", "theta", "xi", "rho", "v0"]

    for asset_label, df in [("BTC", btc_df), ("ETH", eth_df)]:
        if df is None:
            print(f"\n[{asset_label}] No data.")
            continue
        print(f"\n--- {asset_label} ({len(df)} monthly observations) ---")
        for p in params_of_interest:
            if p not in df.columns:
                continue
            s = _summary_stats(df, p)
            if not s:
                continue
            print(
                f"  {p:6s}: mean={s['mean']:8.4f}  std={s['std']:7.4f}  "
                f"[{s['min']:.4f}, {s['max']:.4f}]"
            )

        if "feller_ok" in df.columns:
            feller_count = int(df["feller_ok"].sum())
            print(f"  Feller satisfied: {feller_count}/{len(df)} months")

    # ------------------------------------------------------------------ #
    # Leverage effect comparison                                         #
    # ------------------------------------------------------------------ #
    print("\n=== Leverage Effect (rho) Comparison ===")
    for asset_label, df in [("BTC", btc_df), ("ETH", eth_df)]:
        if df is None or "rho" not in df.columns:
            continue
        rho_vals = df["rho"].dropna().values
        rho_mean = float(np.mean(rho_vals))
        rho_neg_pct = float(np.mean(rho_vals < 0)) * 100
        sign_desc = "negative" if rho_mean < 0 else "positive"
        print(
            f"  {asset_label}: rho mean = {rho_mean:.4f} ({sign_desc}), "
            f"{rho_neg_pct:.0f}% of months have rho < 0"
        )

    if btc_df is not None and eth_df is not None:
        btc_rho = float(np.mean(btc_df["rho"].dropna().values))
        eth_rho = float(np.mean(eth_df["rho"].dropna().values))
        if btc_rho < 0 and eth_rho < 0:
            print(
                "\n  RESULT: Both BTC and ETH show negative rho -> leverage effect"
                " generalises across crypto assets. Heston methodology is asset-agnostic."
            )
        elif btc_rho < 0 and eth_rho >= 0:
            print(
                "\n  RESULT: BTC shows negative rho but ETH does not -> leverage"
                " structure differs. Investigate ETH calibration quality."
            )
        else:
            print(
                f"\n  RESULT: BTC rho={btc_rho:.4f}, ETH rho={eth_rho:.4f} -> "
                "mixed leverage signals."
            )

    # ------------------------------------------------------------------ #
    # kappa comparison                                                   #
    # ------------------------------------------------------------------ #
    print("\n=== Mean Reversion Speed (kappa) Comparison ===")
    for asset_label, df in [("BTC", btc_df), ("ETH", eth_df)]:
        if df is None or "kappa" not in df.columns:
            continue
        s = _summary_stats(df, "kappa")
        print(f"  {asset_label}: kappa in [{s['min']:.2f}, {s['max']:.2f}], mean={s['mean']:.2f}")

    if btc_df is not None and eth_df is not None:
        btc_k = _summary_stats(btc_df, "kappa")
        eth_k = _summary_stats(eth_df, "kappa")
        if btc_k and eth_k:
            overlap_lo = max(btc_k["min"], eth_k["min"])
            overlap_hi = min(btc_k["max"], eth_k["max"])
            if overlap_lo <= overlap_hi:
                print(f"  Kappa ranges overlap in [{overlap_lo:.2f}, {overlap_hi:.2f}] -> similar mean reversion structure.")
            else:
                print(f"  Kappa ranges do NOT overlap -> different mean reversion regime between BTC and ETH.")

    # ------------------------------------------------------------------ #
    # Feller compliance                                                  #
    # ------------------------------------------------------------------ #
    print("\n=== Feller Compliance Summary ===")
    for asset_label, df in [("BTC", btc_df), ("ETH", eth_df)]:
        if df is None or "feller_ok" not in df.columns:
            continue
        feller_ok = int(df["feller_ok"].sum())
        total = len(df)
        print(f"  {asset_label}: {feller_ok}/{total} months satisfy 2*kappa*theta >= xi^2")

    # ------------------------------------------------------------------ #
    # Plot                                                               #
    # ------------------------------------------------------------------ #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        params_to_plot = [
            ("kappa", "kappa (mean reversion speed)"),
            ("rho",   "rho (price-variance correlation)"),
            ("xi",    "xi (vol-of-vol)"),
            ("v0",    "v0 (initial variance)"),
        ]

        BTC_COLOR = "steelblue"
        ETH_COLOR = "darkorange"

        fig, axes = plt.subplots(
            len(params_to_plot), 1,
            figsize=(12, 3.0 * len(params_to_plot)),
            sharex=False,
        )
        if len(params_to_plot) == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, params_to_plot):
            plotted_any = False

            if btc_df is not None and col in btc_df.columns:
                btc_valid = btc_df.dropna(subset=[col, "date"])
                if not btc_valid.empty:
                    ax.plot(
                        btc_valid["date"], btc_valid[col],
                        marker="o", color=BTC_COLOR, linewidth=2.0, markersize=6,
                        label="BTC",
                    )
                    plotted_any = True

            if eth_df is not None and col in eth_df.columns:
                eth_valid = eth_df.dropna(subset=[col, "date"])
                if not eth_valid.empty:
                    ax.plot(
                        eth_valid["date"], eth_valid[col],
                        marker="s", color=ETH_COLOR, linewidth=2.0, markersize=6,
                        linestyle="--", label="ETH",
                    )
                    plotted_any = True

            ax.set_ylabel(label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())

            # Add rho=0 reference line for the rho panel
            if col == "rho":
                ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
                ax.annotate(
                    "rho=0", xy=(0.01, 0.52), xycoords="axes fraction",
                    fontsize=8, color="black", alpha=0.7,
                )

            if plotted_any:
                ax.legend(fontsize=9, loc="upper right")

        axes[-1].set_xlabel("Date", fontsize=11)

        fig.suptitle(
            "Q-measure Heston Parameters: ETH vs BTC\n"
            "(Deribit Options, Tardis Free Tier, Monthly Snapshots 2025-05 to 2026-04)",
            fontsize=12,
            y=1.01,
        )
        fig.tight_layout()

        fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[compare] Figure saved -> {OUTPUT_PNG}")

    except ImportError:
        print("[compare] matplotlib not available; skipping plot.")
    except Exception as exc:
        print(f"[compare] Plot error: {exc}")

    print("\n[compare] Cross-asset comparison complete.")


if __name__ == "__main__":
    main()
