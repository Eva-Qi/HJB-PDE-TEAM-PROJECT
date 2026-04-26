"""Heston Q-measure calibration on IBIT options snapshot (Bloomberg).

Single-snapshot calibration on 2026-04-24 IBIT options data exported from
Bloomberg OMON. Cross-validation companion to the Deribit Tardis monthly
calibration.

USAGE:
    python scripts/heston_calibrate_ibit.py
"""

from __future__ import annotations

import json
import re
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
    _bs_iv,
)

PROJECT_ROOT = Path(_PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
XLSX_PATH = DATA_DIR / "bloomberg" / "IBIT_options_20260424.xlsx"
JSON_OUT = DATA_DIR / "heston_qmeasure_ibit_20260424.json"
PNG_OUT = DATA_DIR / "heston_ibit_smile_20260424.png"

SPOT = 44.02
R = 0.0394
Q = 0.0
SNAP_DATE = "2026-04-24"

# ---------------------------------------------------------------------------
# Step 1 — Parse xlsx
# ---------------------------------------------------------------------------

# Regex for expiry header rows:
#   "15-May-26 (21d); CSize 100; R 3.94; IFwd 44.11"
_HEADER_RE = re.compile(
    r"(\d{1,2}-\w{3}-\d{2})\s+\((\d+)d\).*?R\s+([\d.]+).*?IFwd\s+([\d.]+)"
)


def parse_xlsx(path: Path) -> tuple[list[dict], list[dict]]:
    """Parse Bloomberg OMON xlsx export.

    Returns
    -------
    contracts : list[dict]
        One dict per valid contract (call or put) with fields:
        instrument, kind, strike, T, mark_iv, bid_price, ask_price,
        mid_price, open_interest, expiry_label, days, forward, r
    expiry_meta : list[dict]
        One dict per expiry: label, days, forward, r, T
    """
    import openpyxl

    wb = openpyxl.load_workbook(str(path))
    ws = wb.active

    contracts: list[dict] = []
    expiry_meta: list[dict] = []

    current_meta: dict | None = None

    for row in ws.iter_rows(values_only=True):
        cell0 = row[0]

        # Detect expiry header row
        if isinstance(cell0, str):
            m = _HEADER_RE.match(cell0)
            if m:
                exp_label, days_str, r_str, fwd_str = m.groups()
                days = int(days_str)
                fwd = float(fwd_str)
                r_hdr = float(r_str) / 100.0  # convert % → decimal
                T = days / 365.25
                current_meta = {
                    "label": exp_label,
                    "days": days,
                    "forward": fwd,
                    "r": r_hdr,
                    "T": T,
                }
                expiry_meta.append(current_meta)
                continue

            # Skip "Calls" header row or column-name row
            if cell0 in ("Calls", "Ticker"):
                continue

        if current_meta is None:
            continue  # haven't seen first expiry yet

        # Contract row: cols 0-6 = call, cols 7-13 = put
        # Schema: Ticker(0), Strike(1), Bid(2), Ask(3), Last(4), IVM(5), Volm(6)
        call_ticker = row[0]
        call_strike = row[1]
        call_bid = row[2]
        call_ask = row[3]
        call_ivm = row[5]

        put_ticker = row[7]
        put_strike = row[8]
        put_bid = row[9]
        put_ask = row[10]
        put_ivm = row[12]

        meta = current_meta

        # --- Process call ---
        if (
            call_ticker is not None
            and isinstance(call_ticker, str)
            and call_strike is not None
            and call_ivm is not None
            and float(call_ivm) > 0
            and call_bid is not None
            and float(call_bid) > 0
        ):
            strike = float(call_strike)
            mark_iv = float(call_ivm) / 100.0
            bid_p = float(call_bid)
            ask_p = float(call_ask) if call_ask is not None else bid_p
            mid_p = (bid_p + ask_p) / 2.0
            contracts.append({
                "instrument": call_ticker,
                "kind": "C",
                "strike": strike,
                "T": meta["T"],
                "mark_iv": mark_iv,
                "bid_price": bid_p,
                "ask_price": ask_p,
                "mid_price": mid_p,
                "open_interest": None,
                "expiry_label": meta["label"],
                "days": meta["days"],
                "forward": meta["forward"],
                "r": meta["r"],
            })

        # --- Process put ---
        if (
            put_ticker is not None
            and isinstance(put_ticker, str)
            and put_strike is not None
            and put_ivm is not None
            and float(put_ivm) > 0
            and put_bid is not None
            and float(put_bid) > 0
        ):
            strike = float(put_strike)
            mark_iv = float(put_ivm) / 100.0
            bid_p = float(put_bid)
            ask_p = float(put_ask) if put_ask is not None else bid_p
            mid_p = (bid_p + ask_p) / 2.0
            contracts.append({
                "instrument": put_ticker,
                "kind": "P",
                "strike": strike,
                "T": meta["T"],
                "mark_iv": mark_iv,
                "bid_price": bid_p,
                "ask_price": ask_p,
                "mid_price": mid_p,
                "open_interest": None,
                "expiry_label": meta["label"],
                "days": meta["days"],
                "forward": meta["forward"],
                "r": meta["r"],
            })

    return contracts, expiry_meta


# ---------------------------------------------------------------------------
# RMSE helper (same pattern as deribit_qmeasure_time_series.py)
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
    df_c = df.copy()
    df_c["T_key"] = df_c["T"].round(4)

    all_diffs: list[float] = []
    for T_key, grp in df_c.groupby("T_key"):
        strikes = grp["strike"].values
        mkt_ivs = grp["mark_iv"].values
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
    t0 = time.time()
    print(f"[heston_calibrate_ibit] Loading {XLSX_PATH.name} ...")
    contracts, expiry_meta = parse_xlsx(XLSX_PATH)

    n_input = len(contracts)
    print(f"[heston_calibrate_ibit] Parsed {n_input} valid contracts "
          f"from {len(expiry_meta)} expiries")

    # Step 2 — Build DataFrame
    df = pd.DataFrame(contracts)

    # Step 3 — Calibrate
    # Note: calibrate_heston_from_options applies OTM filter internally using
    # forward ≈ S0 (r=q=0 for crypto).  For IBIT we have non-zero r and forward
    # slightly above S0 due to the cost-of-carry.  The internal OTM filter uses
    # "forward = S0" as a simplification — acceptable given r=3.94% and T≤55d
    # (forward displacement < 0.6%).
    print(f"\n[heston_calibrate_ibit] Starting Heston calibration "
          f"(n_starts=8, uniform weighting) ...")

    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        params = calibrate_heston_from_options(
            df,
            underlying_price=SPOT,
            r=R,
            q=Q,
            n_starts=8,
            delta_filter=(0.10, 0.90),
            use_bid_ask=False,   # use mark_iv (IVM/100) directly
            weighting="uniform",
            seed=42,
        )

    elapsed = time.time() - t0

    # Count contracts that survive the OTM filter (mirror internal logic)
    df_filt = df[df["mark_iv"].notna() & (df["mark_iv"] > 0)].copy()
    is_otm_call = (df_filt["kind"] == "C") & (df_filt["strike"] >= SPOT)
    is_otm_put = (df_filt["kind"] == "P") & (df_filt["strike"] <= SPOT)
    df_otm = df_filt[is_otm_call | is_otm_put].copy()
    T_min, T_max = 7 / 365.25, 180 / 365.25
    df_otm = df_otm[(df_otm["T"] >= T_min) & (df_otm["T"] <= T_max)].copy()
    n_after_filter = len(df_otm)
    n_calls_filt = int((df_otm["kind"] == "C").sum())
    n_puts_filt = int((df_otm["kind"] == "P").sum())

    # RMSE on OTM-filtered chain
    rmse = _compute_heston_rmse(
        df_otm, SPOT,
        params.kappa, params.theta, params.xi, params.rho, params.v0,
        r=R, q=Q,
    )

    # Feller diagnostics
    feller_lhs = 2.0 * params.kappa * params.theta
    feller_rhs = params.xi ** 2
    feller_ok = feller_lhs >= feller_rhs

    # Boundary hits (within 5% of bound)
    bounds = {
        "kappa": (0.1, 10.0),
        "theta": (0.01, 2.0),
        "xi": (0.3, 3.0),
        "rho": (-0.99, 0.99),
        "v0": (0.01, 2.0),
    }
    param_vals = {
        "kappa": params.kappa,
        "theta": params.theta,
        "xi": params.xi,
        "rho": params.rho,
        "v0": params.v0,
    }
    boundary_hits = []
    for pname, (lo, hi) in bounds.items():
        val = param_vals[pname]
        span = hi - lo
        if abs(val - lo) / span < 0.05:
            boundary_hits.append(f"{pname}@lower({lo})")
        elif abs(val - hi) / span < 0.05:
            boundary_hits.append(f"{pname}@upper({hi})")

    sigma_lr_pct = float(np.sqrt(params.theta)) * 100.0
    sigma_0_pct = float(np.sqrt(params.v0)) * 100.0

    # Step 4 — Save JSON
    expiry_labels = [m["label"] for m in expiry_meta]
    days_list = [m["days"] for m in expiry_meta]
    result = {
        "snapshot_date": SNAP_DATE,
        "source": "Bloomberg OMON IBIT US Equity",
        "spot": SPOT,
        "risk_free_rate": R,
        "n_contracts_input": n_input,
        "n_contracts_after_otm_filter": n_after_filter,
        "expiries": expiry_labels,
        "T_range_days": [min(days_list), max(days_list)],
        "heston_params": {
            "kappa": round(params.kappa, 6),
            "theta": round(params.theta, 6),
            "xi": round(params.xi, 6),
            "rho": round(params.rho, 6),
            "v0": round(params.v0, 6),
        },
        "feller_lhs": round(feller_lhs, 6),
        "feller_rhs": round(feller_rhs, 6),
        "feller_satisfied": bool(feller_ok),
        "rmse_iv": round(rmse, 6) if not np.isnan(rmse) else None,
        "boundary_hits": boundary_hits,
        "elapsed_sec": round(elapsed, 2),
        "code_version": "post-3-bug-fix-2026-04-24",
        "caveats": ["american_exercise_unadjusted"],
    }
    with open(JSON_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[heston_calibrate_ibit] Saved results → {JSON_OUT.name}")

    # Step 5 — Smile plot
    _plot_smile(df, params, expiry_meta, PNG_OUT)

    # Step 6 — Print summary
    print()
    print("=" * 60)
    print(f"IBIT Heston calibration — {SNAP_DATE}")
    print("=" * 60)
    print(f"spot:        ${SPOT:.2f}")
    print(f"r:           {R*100:.2f}%")
    print(
        f"contracts:   {n_after_filter} / {n_input} OTM-filtered "
        f"({n_calls_filt} calls + {n_puts_filt} puts)"
    )
    print(f"expiries:    {len(expiry_meta)} ({min(days_list)}-{max(days_list)} days)")
    print()
    print(f"κ (kappa):   {params.kappa:.4f}")
    print(f"θ (theta):   {params.theta:.4f}  "
          f"(long-run vol = σ_LR = {sigma_lr_pct:.1f}%)")
    print(f"ξ (xi):      {params.xi:.4f}  (vol-of-vol)")
    print(f"ρ (rho):     {params.rho:.4f}  "
          f"← {'in expected [-0.7,-0.3] range' if -0.7 <= params.rho <= -0.3 else 'OUTSIDE expected [-0.7,-0.3]'}")
    print(f"v₀:          {params.v0:.4f}  "
          f"(initial vol = σ₀ = {sigma_0_pct:.1f}%)")
    print()
    feller_sym = ">=" if feller_ok else "<"
    feller_word = "SATISFIED" if feller_ok else "VIOLATED"
    print(f"Feller:      2κθ = {feller_lhs:.4f} {feller_sym} "
          f"ξ² = {feller_rhs:.4f} → {feller_word}")
    rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "n/a"
    print(f"RMSE IV:     {rmse_str} vol pts")
    print()
    hits_str = ", ".join(boundary_hits) if boundary_hits else "none"
    print(f"Boundary hits: {hits_str}")
    print("=" * 60)

    if w_list:
        print("\n[Warnings from calibrator]")
        for wr in w_list:
            print(f"  {wr.category.__name__}: {wr.message}")


# ---------------------------------------------------------------------------
# Smile plot
# ---------------------------------------------------------------------------

def _plot_smile(
    df: pd.DataFrame,
    params,
    expiry_meta: list[dict],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[heston_calibrate_ibit] matplotlib not available — skipping plot")
        return

    n_exp = len(expiry_meta)
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5), sharey=False)
    if n_exp == 1:
        axes = [axes]

    kappa, theta, xi, rho, v0 = (
        params.kappa, params.theta, params.xi, params.rho, params.v0
    )

    for ax, meta in zip(axes, expiry_meta):
        label = meta["label"]
        days = meta["days"]
        fwd = meta["forward"]
        T = meta["T"]

        sub = df[df["expiry_label"] == label].copy()
        calls = sub[sub["kind"] == "C"].copy()
        puts = sub[sub["kind"] == "P"].copy()

        # Data points: X = K/F
        if not puts.empty:
            ax.scatter(
                puts["strike"] / fwd,
                puts["mark_iv"] * 100,
                color="steelblue",
                s=30,
                alpha=0.8,
                label="Puts (market)",
                zorder=3,
            )
        if not calls.empty:
            ax.scatter(
                calls["strike"] / fwd,
                calls["mark_iv"] * 100,
                color="firebrick",
                s=30,
                alpha=0.8,
                label="Calls (market)",
                zorder=3,
            )

        # Model curve: fine grid over observed strike range
        all_strikes = sub["strike"].values
        if len(all_strikes) > 0:
            K_lo = all_strikes.min() * 0.98
            K_hi = all_strikes.max() * 1.02
            K_grid = np.linspace(K_lo, K_hi, 80)

            try:
                mdl_ivs = _heston_model_iv_batch(
                    SPOT, K_grid, T, R, Q,
                    kappa, theta, xi, rho, v0,
                )
                valid = ~np.isnan(mdl_ivs)
                if valid.sum() >= 2:
                    ax.plot(
                        K_grid[valid] / fwd,
                        mdl_ivs[valid] * 100,
                        color="black",
                        linewidth=1.8,
                        label="Heston model",
                        zorder=4,
                    )
            except Exception as exc:
                print(f"[heston_calibrate_ibit] Model curve failed for {label}: {exc}")

        ax.set_title(f"{label} — T={days}d, F=${fwd:.2f}", fontsize=10)
        ax.set_xlabel("Moneyness K/F", fontsize=9)
        ax.set_ylabel("Implied Vol (%)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    sup = (
        f"IBIT Q-measure Heston smile fit (4/24/2026, "
        f"ρ={rho:.3f}, ξ={xi:.3f})"
    )
    fig.suptitle(sup, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[heston_calibrate_ibit] Smile plot saved → {out_path.name}")


if __name__ == "__main__":
    main()
