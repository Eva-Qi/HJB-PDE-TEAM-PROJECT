"""Compare Heston vs 3/2 model on Deribit BTC IV surface.

Calibrates both models on the latest Deribit snapshot and compares:
    - Calibrated parameters
    - IV surface fit RMSE (Heston vs 3/2)
    - Side-by-side parameter table

Outputs
-------
``data/heston_variants_comparison.json`` — comparison results
``figures/heston_variants_iv_fit.png``   — IV model vs market (optional)

Usage
-----
    python scripts/compare_heston_variants.py
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

from calibration.download_deribit import load_latest_snapshot, snapshot_to_dataframe
from extensions.heston import (
    HestonParams,
    ThreeHalfParams,
    _bs_iv,
    _heston_model_iv_batch,
    _threehalf_model_iv_batch,
    calibrate_32_from_options,
    calibrate_heston_from_options,
    mc_call_prices_32,
)

PROJECT_ROOT = Path(_PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUT_PATH = DATA_DIR / "heston_variants_comparison.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_rmse(
    df: pd.DataFrame,
    S0: float,
    model_iv_fn,   # callable(S0, strikes, T, r, q, *model_params) -> np.ndarray
    model_params: tuple,
    r: float = 0.0,
    q: float = 0.0,
) -> tuple[float, int]:
    """Compute IV RMSE and count of valid contracts for a model.

    Returns (rmse, n_valid).
    """
    df_c = df.copy()
    df_c["T_key"] = df_c["T"].round(4)

    all_diffs: list[float] = []
    for T_key, grp in df_c.groupby("T_key"):
        strikes = grp["strike"].values
        mkt_ivs = (
            grp["market_iv"].values if "market_iv" in grp.columns
            else grp["mark_iv"].values
        )
        try:
            mdl_ivs = model_iv_fn(S0, strikes, T_key, r, q, *model_params)
        except Exception:
            continue
        valid = ~np.isnan(mdl_ivs)
        if valid.sum() == 0:
            continue
        all_diffs.extend((mdl_ivs[valid] - mkt_ivs[valid]).tolist())

    if not all_diffs:
        return float("nan"), 0
    arr = np.array(all_diffs)
    return float(np.sqrt(np.mean(arr**2))), len(arr)


def _prepare_chain(df: pd.DataFrame, S0: float) -> pd.DataFrame:
    """Apply standard filter + resolve market_iv (mid or mark).

    Returns filtered DataFrame with ``market_iv`` column.
    """
    df = df[df["mark_iv"].notna() & (df["mark_iv"] > 0) & (df["kind"] == "C")].copy()
    T_min = 7 / 365.25
    T_max = 180 / 365.25
    df = df[(df["T"] >= T_min) & (df["T"] <= T_max)].copy()

    if "bid_iv" in df.columns and "ask_iv" in df.columns:
        bid_ok = df["bid_iv"].notna() & (df["bid_iv"] > 0)
        ask_ok = df["ask_iv"].notna() & (df["ask_iv"] > 0)
        df["market_iv"] = df["mark_iv"].copy()
        df.loc[bid_ok & ask_ok, "market_iv"] = (
            df.loc[bid_ok & ask_ok, "bid_iv"]
            + df.loc[bid_ok & ask_ok, "ask_iv"]
        ) / 2.0
    else:
        df["market_iv"] = df["mark_iv"]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main calibration + comparison
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load snapshot
    snap, snap_path = load_latest_snapshot()
    print(f"[compare_heston_variants] Snapshot: {snap_path.name}")
    df_raw = snapshot_to_dataframe(snap)
    S0 = float(snap["underlying_price"])
    snapshot_date = snap_path.stem.split("_")[-1]
    print(f"  S0={S0:.0f}, raw contracts={len(df_raw)}, date={snapshot_date}")

    # ------------------------------------------------------------------ #
    # Calibrate Heston (bid/ask + OI weighted)                          #
    # ------------------------------------------------------------------ #
    print("\n=== Heston model calibration ===")
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w_h:
        warnings.simplefilter("always")
        heston_params = calibrate_heston_from_options(
            df_raw,
            underlying_price=S0,
            r=0.0, q=0.0,
            n_starts=5,
            delta_filter=(0.10, 0.90),
            use_bid_ask=True,
            use_oi_weights=True,
            seed=42,
        )
    t_heston = time.time() - t0
    print(f"  Done in {t_heston:.1f}s")
    if w_h:
        for wr in w_h:
            print(f"  WARNING: {wr.message}")

    # Heston (mark_iv only, unweighted) — for comparison of Task 2 delta
    print("\n=== Heston model calibration (mark_iv, unweighted) ===")
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w_h2:
        warnings.simplefilter("always")
        heston_params_markiv = calibrate_heston_from_options(
            df_raw,
            underlying_price=S0,
            r=0.0, q=0.0,
            n_starts=5,
            delta_filter=(0.10, 0.90),
            use_bid_ask=False,
            use_oi_weights=False,
            seed=42,
        )
    t_heston_markiv = time.time() - t0
    print(f"  Done in {t_heston_markiv:.1f}s")

    # ------------------------------------------------------------------ #
    # Calibrate 3/2 model                                                #
    # ------------------------------------------------------------------ #
    print("\n=== 3/2 model calibration ===")
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w_32:
        warnings.simplefilter("always")
        params_32 = calibrate_32_from_options(
            df_raw,
            underlying_price=S0,
            r=0.0, q=0.0,
            n_starts=5,
            delta_filter=(0.10, 0.90),
            use_bid_ask=True,
            use_oi_weights=True,
            seed=42,
        )
    t_32 = time.time() - t0
    print(f"  Done in {t_32:.1f}s")
    if w_32:
        for wr in w_32:
            print(f"  WARNING: {wr.message}")

    # ------------------------------------------------------------------ #
    # Compute RMSEs on the filtered + resolved chain                     #
    # ------------------------------------------------------------------ #
    df_filt = _prepare_chain(df_raw, S0)

    def heston_iv_fn(S0, strikes, T, r, q, kappa, theta, xi, rho, v0):
        return _heston_model_iv_batch(S0, strikes, T, r, q, kappa, theta, xi, rho, v0)

    def threehalf_iv_fn(S0, strikes, T, r, q, p, q_par, xi, rho, v0):
        return _threehalf_model_iv_batch(S0, strikes, T, r, q, p, q_par, xi, rho, v0)

    rmse_heston, n_heston = _compute_rmse(
        df_filt, S0, heston_iv_fn,
        (heston_params.kappa, heston_params.theta, heston_params.xi,
         heston_params.rho, heston_params.v0),
    )
    rmse_heston_markiv, n_heston_mv = _compute_rmse(
        df_filt, S0, heston_iv_fn,
        (heston_params_markiv.kappa, heston_params_markiv.theta,
         heston_params_markiv.xi, heston_params_markiv.rho,
         heston_params_markiv.v0),
    )
    rmse_32, n_32 = _compute_rmse(
        df_filt, S0, threehalf_iv_fn,
        (params_32.p, params_32.q, params_32.xi, params_32.rho, params_32.v0),
    )

    # ------------------------------------------------------------------ #
    # Print comparison table                                             #
    # ------------------------------------------------------------------ #
    feller_h = 2.0 * heston_params.kappa * heston_params.theta - heston_params.xi**2
    feller_h2 = 2.0 * heston_params_markiv.kappa * heston_params_markiv.theta - heston_params_markiv.xi**2

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — Deribit BTC IV surface")
    print("=" * 70)
    print(f"Snapshot date : {snapshot_date}   S0 = {S0:.0f} USD")
    print(f"Filtered chain: {len(df_filt)} contracts (delta 10–90%, T 7–180 days)")
    print()
    print(f"{'':30s} {'Heston (mid+OI)':>18} {'Heston (mark_iv)':>18} {'3/2 model':>12}")
    print("-" * 80)
    print(f"  kappa / p          {'':2s} {heston_params.kappa:>18.4f} {heston_params_markiv.kappa:>18.4f} {params_32.p:>12.4f}")
    print(f"  theta / (p/q)      {'':2s} {heston_params.theta:>18.4f} {heston_params_markiv.theta:>18.4f} {params_32.p/params_32.q:>12.4f}")
    print(f"  xi                 {'':2s} {heston_params.xi:>18.4f} {heston_params_markiv.xi:>18.4f} {params_32.xi:>12.4f}")
    print(f"  rho                {'':2s} {heston_params.rho:>18.4f} {heston_params_markiv.rho:>18.4f} {params_32.rho:>12.4f}")
    print(f"  v0                 {'':2s} {heston_params.v0:>18.4f} {heston_params_markiv.v0:>18.4f} {params_32.v0:>12.4f}")
    print("-" * 80)
    print(f"  Feller margin      {'':2s} {feller_h:>18.4f} {feller_h2:>18.4f} {'N/A':>12}")
    print(f"  IV RMSE            {'':2s} {rmse_heston:>18.4f} {rmse_heston_markiv:>18.4f} {rmse_32:>12.4f}")
    print(f"  n_valid_contracts  {'':2s} {n_heston:>18d} {n_heston_mv:>18d} {n_32:>12d}")
    print("=" * 70)

    # Task 2: param delta (mid+OI vs mark_iv)
    print("\n--- Task 2: bid/ask+OI vs mark_iv parameter delta ---")
    for attr in ["kappa", "theta", "xi", "rho", "v0"]:
        v_new = getattr(heston_params, attr)
        v_old = getattr(heston_params_markiv, attr)
        delta = v_new - v_old
        print(f"  Δ{attr:5s} = {delta:+.4f}  ({v_old:.4f} → {v_new:.4f})")
    rmse_delta = rmse_heston - rmse_heston_markiv
    print(f"  Δrmse  = {rmse_delta:+.4f}  ({rmse_heston_markiv:.4f} → {rmse_heston:.4f})")

    # Task 3: Heston vs 3/2
    print("\n--- Task 3: 3/2 vs Heston fit quality ---")
    better = "3/2" if rmse_32 < rmse_heston else "Heston"
    pct_diff = 100.0 * abs(rmse_32 - rmse_heston) / max(rmse_heston, 1e-9)
    print(f"  Heston RMSE = {rmse_heston:.4f}")
    print(f"  3/2    RMSE = {rmse_32:.4f}")
    print(f"  Better fit : {better}  ({pct_diff:.1f}% improvement)")

    # ------------------------------------------------------------------ #
    # Save JSON                                                          #
    # ------------------------------------------------------------------ #
    output = {
        "snapshot_date": snapshot_date,
        "underlying_price": S0,
        "n_contracts_filtered": len(df_filt),
        "heston_bid_ask_oi": {
            "kappa": round(heston_params.kappa, 6),
            "theta": round(heston_params.theta, 6),
            "xi": round(heston_params.xi, 6),
            "rho": round(heston_params.rho, 6),
            "v0": round(heston_params.v0, 6),
            "feller_margin": round(feller_h, 6),
            "fit_rmse": round(rmse_heston, 6),
            "n_valid_contracts": n_heston,
        },
        "heston_mark_iv_unweighted": {
            "kappa": round(heston_params_markiv.kappa, 6),
            "theta": round(heston_params_markiv.theta, 6),
            "xi": round(heston_params_markiv.xi, 6),
            "rho": round(heston_params_markiv.rho, 6),
            "v0": round(heston_params_markiv.v0, 6),
            "feller_margin": round(feller_h2, 6),
            "fit_rmse": round(rmse_heston_markiv, 6),
            "n_valid_contracts": n_heston_mv,
        },
        "model_32": {
            "p": round(params_32.p, 6),
            "q": round(params_32.q, 6),
            "long_run_variance_pq": round(params_32.p / params_32.q, 6),
            "xi": round(params_32.xi, 6),
            "rho": round(params_32.rho, 6),
            "v0": round(params_32.v0, 6),
            "fit_rmse": round(rmse_32, 6),
            "n_valid_contracts": n_32,
        },
        "task2_param_delta": {
            attr: round(getattr(heston_params, attr) - getattr(heston_params_markiv, attr), 6)
            for attr in ["kappa", "theta", "xi", "rho", "v0"]
        },
        "task3_winner": better,
        "task3_rmse_improvement_pct": round(pct_diff, 3),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[compare_heston_variants] Results saved → {OUTPUT_PATH.name}")

    # ------------------------------------------------------------------ #
    # Optional: IV surface plot                                          #
    # ------------------------------------------------------------------ #
    try:
        import matplotlib.pyplot as plt

        # Pick a single near-ATM expiry to visualise
        T_targets = np.array([30, 60, 90]) / 365.25
        df_filt["T_key"] = df_filt["T"].round(4)
        available_T = df_filt["T_key"].unique()
        plot_Ts = []
        for tgt in T_targets:
            close = available_T[np.argmin(np.abs(available_T - tgt))]
            if abs(close - tgt) < 20 / 365.25:
                plot_Ts.append(close)
        plot_Ts = sorted(set(plot_Ts))[:3]

        if plot_Ts:
            fig, axes = plt.subplots(1, len(plot_Ts), figsize=(5 * len(plot_Ts), 5))
            if len(plot_Ts) == 1:
                axes = [axes]

            for ax, T_key in zip(axes, plot_Ts):
                grp = df_filt[df_filt["T_key"] == T_key].sort_values("strike")
                K = grp["strike"].values
                mkt = grp["market_iv"].values

                h_iv = _heston_model_iv_batch(
                    S0, K, T_key, 0.0, 0.0,
                    heston_params.kappa, heston_params.theta,
                    heston_params.xi, heston_params.rho, heston_params.v0,
                )
                m32_iv = _threehalf_model_iv_batch(
                    S0, K, T_key, 0.0, 0.0,
                    params_32.p, params_32.q, params_32.xi,
                    params_32.rho, params_32.v0,
                )

                days = round(T_key * 365.25)
                ax.scatter(K / S0, mkt, s=15, color="black", zorder=5, label="Market")
                ax.plot(K / S0, np.where(np.isnan(h_iv), np.nan, h_iv),
                        color="steelblue", linewidth=1.5, label=f"Heston (RMSE={rmse_heston:.3f})")
                ax.plot(K / S0, np.where(np.isnan(m32_iv), np.nan, m32_iv),
                        color="firebrick", linewidth=1.5, linestyle="--",
                        label=f"3/2 (RMSE={rmse_32:.3f})")
                ax.set_xlabel("K / S0")
                ax.set_ylabel("Implied vol")
                ax.set_title(f"T ≈ {days}d")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            fig.suptitle(f"Heston vs 3/2 — Deribit BTC ({snapshot_date})", fontsize=13)
            fig.tight_layout()
            plot_path = FIGURES_DIR / "heston_variants_iv_fit.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"[compare_heston_variants] IV fit plot saved → {plot_path.name}")

    except ImportError:
        print("[compare_heston_variants] matplotlib not available; skipping IV plot.")


if __name__ == "__main__":
    main()
