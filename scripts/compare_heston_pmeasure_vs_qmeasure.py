"""Compare Heston P-measure vs Q-measure calibration.

P-measure: calibrate_heston_from_spot on Jan–Apr 2026 Binance aggTrades OHLC
Q-measure: calibrate_heston_from_options on Deribit BTC option IV surface

Outputs:
    data/heston_pmeasure_vs_qmeasure.json

Usage:
    python scripts/compare_heston_pmeasure_vs_qmeasure.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

# Ensure project root is on sys.path so calibration/extensions are importable
_PROJECT_ROOT_STR = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
PROJECT_ROOT = Path(_PROJECT_ROOT_STR)
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "heston_pmeasure_vs_qmeasure.json"


def _run_q_measure() -> dict:
    """Calibrate Heston from Deribit IV surface (Q-measure)."""
    from calibration.download_deribit import load_latest_snapshot, snapshot_to_dataframe
    from extensions.heston import calibrate_heston_from_options

    snap, snap_path = load_latest_snapshot()
    print(f"[Q-measure] Loading snapshot: {snap_path.name}")
    df = snapshot_to_dataframe(snap)
    S0 = snap["underlying_price"]

    n_raw = len(df)
    print(f"[Q-measure] Raw contracts: {n_raw}, S0={S0:.0f}")

    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        params = calibrate_heston_from_options(
            df,
            underlying_price=S0,
            r=0.0,
            q=0.0,
            n_starts=8,
            delta_filter=(0.10, 0.90),
            seed=42,
        )
    elapsed = time.time() - t0

    feller_lhs = 2 * params.kappa * params.theta
    feller_rhs = params.xi**2
    feller_ok = feller_lhs >= feller_rhs
    warning_msgs = [str(x.message) for x in w]

    # Count filtered contracts used
    T_min, T_max = 7 / 365.25, 180 / 365.25
    df_filt = df[(df["kind"] == "C") & df["mark_iv"].notna() & (df["mark_iv"] > 0)
                 & (df["T"] >= T_min) & (df["T"] <= T_max)]
    n_filtered = len(df_filt)

    print(f"[Q-measure] Contracts used (filtered): {n_filtered}")
    print(f"[Q-measure] Runtime: {elapsed:.1f}s")
    print(f"[Q-measure] kappa={params.kappa:.4f}, theta={params.theta:.4f}, "
          f"xi={params.xi:.4f}, rho={params.rho:.4f}, v0={params.v0:.4f}")
    print(f"[Q-measure] Feller: 2κθ={feller_lhs:.4f}, ξ²={feller_rhs:.4f} "
          f"({'OK' if feller_ok else 'VIOLATED'})")

    return {
        "measure": "Q",
        "source": snap_path.name,
        "underlying_price": S0,
        "n_contracts_raw": n_raw,
        "n_contracts_filtered": n_filtered,
        "kappa": params.kappa,
        "theta": params.theta,
        "xi": params.xi,
        "rho": params.rho,
        "v0": params.v0,
        "feller_lhs": feller_lhs,
        "feller_rhs": feller_rhs,
        "feller_satisfied": feller_ok,
        "runtime_seconds": round(elapsed, 2),
        "warnings": warning_msgs,
    }


def _run_p_measure() -> dict:
    """Calibrate Heston from Binance spot OHLC (P-measure)."""
    import glob

    from calibration.data_loader import compute_ohlc, load_trades
    from extensions.heston import calibrate_heston_from_spot

    # Load all available aggTrades CSVs
    csv_pattern = str(DATA_DIR / "BTCUSDT-aggTrades-*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No aggTrades CSVs found at {csv_pattern}.  "
            f"Run calibration/download_binance.py first."
        )

    print(f"[P-measure] Loading {len(csv_files)} aggTrades CSVs ...")
    trades = load_trades(DATA_DIR)
    ohlc = compute_ohlc(trades, freq="5min")
    print(f"[P-measure] OHLC bars: {len(ohlc)}")

    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        params = calibrate_heston_from_spot(ohlc, freq_seconds=300.0, window_bars=24)
    elapsed = time.time() - t0

    feller_lhs = 2 * params.kappa * params.theta
    feller_rhs = params.xi**2
    feller_ok = feller_lhs >= feller_rhs
    warning_msgs = [str(x.message) for x in w]

    print(f"[P-measure] Runtime: {elapsed:.1f}s")
    print(f"[P-measure] kappa={params.kappa:.4f}, theta={params.theta:.4f}, "
          f"xi={params.xi:.4f}, rho={params.rho:.4f}, v0={params.v0:.4f}")
    print(f"[P-measure] Feller: 2κθ={feller_lhs:.4f}, ξ²={feller_rhs:.4f} "
          f"({'OK' if feller_ok else 'VIOLATED'})")

    return {
        "measure": "P",
        "source": f"{len(csv_files)} Binance aggTrades CSVs",
        "n_ohlc_bars": len(ohlc),
        "kappa": params.kappa,
        "theta": params.theta,
        "xi": params.xi,
        "rho": params.rho,
        "v0": params.v0,
        "feller_lhs": feller_lhs,
        "feller_rhs": feller_rhs,
        "feller_satisfied": feller_ok,
        "runtime_seconds": round(elapsed, 2),
        "warnings": warning_msgs,
    }


def _print_comparison_table(q_res: dict, p_res: dict) -> None:
    """Print a side-by-side table of P vs Q parameters."""
    params = ["kappa", "theta", "xi", "rho", "v0"]
    header = f"{'Parameter':<12} {'P-measure (spot)':<22} {'Q-measure (IV surf)':<22} {'Ratio Q/P':<12}"
    print("\n" + "=" * 70)
    print("  Heston Calibration: P-measure vs Q-measure Comparison")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for p in params:
        pv = p_res[p]
        qv = q_res[p]
        ratio = qv / pv if pv != 0 else float("nan")
        print(f"  {p:<10} {pv:>18.4f}   {qv:>18.4f}   {ratio:>10.3f}")
    print("-" * 70)
    print(f"  {'Feller P':<10} {'OK' if p_res['feller_satisfied'] else 'VIOLATED':>18}   "
          f"{'OK' if q_res['feller_satisfied'] else 'VIOLATED':>18}")
    print("=" * 70)
    print(f"\n  P-measure runtime: {p_res['runtime_seconds']:.1f}s")
    print(f"  Q-measure runtime: {q_res['runtime_seconds']:.1f}s")

    # Narrative
    kappa_ratio = q_res["kappa"] / p_res["kappa"] if p_res["kappa"] != 0 else float("nan")
    print("\n  NARRATIVE:")
    if abs(kappa_ratio - 1.0) > 0.50:
        print(f"  - κ differs by {abs(kappa_ratio-1)*100:.0f}%: P-measure mean-reversion "
              f"speed ({p_res['kappa']:.2f}) vs Q-measure ({q_res['kappa']:.2f}). "
              f"This is expected — κ is not Q-invariant.")
    if p_res["kappa"] >= 19.0:
        print(f"  - P-measure κ={p_res['kappa']:.1f} is clipped to ceiling (20.0) — "
              f"GK autocorrelation estimator is unreliable at this horizon.")
    if abs(q_res["rho"]) > 0.1:
        print(f"  - Q-measure ρ={q_res['rho']:.3f} captures the IV skew (leverage effect). "
              f"P-measure ρ={p_res['rho']:.3f} is less reliable (rolling window noise).")


def main() -> None:
    print("[compare_heston] Starting P-measure vs Q-measure calibration comparison")
    print("=" * 70)

    # Q-measure (Deribit IV surface)
    print("\n--- Q-measure (Deribit BTC option IV surface) ---")
    q_res = _run_q_measure()

    # P-measure (Binance spot OHLC)
    print("\n--- P-measure (Binance aggTrades OHLC moment-matching) ---")
    try:
        p_res = _run_p_measure()
    except Exception as e:
        print(f"[P-measure] FAILED: {e}")
        p_res = {
            "measure": "P",
            "source": "FAILED",
            "n_ohlc_bars": 0,
            "kappa": float("nan"),
            "theta": float("nan"),
            "xi": float("nan"),
            "rho": float("nan"),
            "v0": float("nan"),
            "feller_lhs": float("nan"),
            "feller_rhs": float("nan"),
            "feller_satisfied": False,
            "runtime_seconds": 0.0,
            "warnings": [str(e)],
        }

    # Print comparison
    _print_comparison_table(q_res, p_res)

    # Save
    output = {
        "q_measure": q_res,
        "p_measure": p_res,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[compare_heston] Saved comparison → {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
