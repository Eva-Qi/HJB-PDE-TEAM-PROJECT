"""Compare 2-state vs 3-state HMM on a rolling-volatility feature.

Motivation
----------
Raw 5-min log-returns are heavy-tailed and zero-mean — Gaussian HMM has to
fit regimes via emission *variance* alone (means are all ~0).  This pushes
3-state models toward duplicate-σ collapse on bimodal-vol data.

Volatility itself, however, is monotonic by regime: low-vol bars have low
realized σ, high-vol bars have high realized σ.  A Gaussian HMM on
``log(rolling_std)`` separates regimes mainly via emission MEAN (not σ),
so 3-state models can be non-degenerate even when raw-return 3-state can't.

Note on filtering
-----------------
The σ-ratio duplicate-state filter (used in compare_2state_vs_3state_hmm)
is feature-specific.  It is correct for raw returns (regimes separated by
σ).  On log_vol, regimes are separated by μ — different regimes can have
similar σ but well-separated mean.  We therefore disable the σ-ratio
filter here (threshold=0.0) and rely only on the occupancy filter.

Output: ``data/hmm_vol_feature_comparison.json``

Usage
-----
    python scripts/compare_hmm_vol_feature.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration.data_loader import load_trades, compute_mid_prices
from extensions.regime import _rolling_vol_feature
from scripts.compare_2state_vs_3state_hmm import _fit_and_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = sorted(DATA_DIR.glob("BTCUSDT-aggTrades-2026-*.csv"))

VOL_WINDOW = 6  # 6 × 5min = 30min rolling-std window


def main() -> None:
    print("\n" + "=" * 72)
    print("HMM 2-state vs 3-state on rolling-vol feature (BTCUSDT 5-min)")
    print("=" * 72)

    if not DATA_FILES:
        raise FileNotFoundError(f"No aggTrades CSVs in {DATA_DIR}")

    print(f"\nLoading {len(DATA_FILES)} CSV files (full window)...")
    trades = load_trades(DATA_DIR, start="2026-01-01", end="2026-04-08")
    print(f"  {len(trades):,} trades loaded")

    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    returns = mid["log_return"].to_numpy()
    returns = returns[np.isfinite(returns)]
    T_ret = len(returns)
    print(f"  {T_ret:,} 5-min log-returns")

    log_vol = _rolling_vol_feature(returns, window=VOL_WINDOW)
    log_vol = log_vol[np.isfinite(log_vol)]
    T_vol = len(log_vol)
    print(f"  {T_vol:,} log-rolling-σ observations (window={VOL_WINDOW} bars = {VOL_WINDOW * 5}min)")
    print(f"  log_vol range: [{log_vol.min():.3f}, {log_vol.max():.3f}], mean={log_vol.mean():.3f}, std={log_vol.std():.3f}")

    # Fit on log-vol feature.  σ-ratio filter disabled because regime
    # separation on log_vol is via μ (mean), not σ — see module docstring.
    print(f"\nFitting 2-state HMM on log_vol (15 restarts, σ-filter off)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_2_vol = _fit_and_score(
            log_vol, n_regimes=2, duplicate_sigma_ratio_threshold=0.0,
        )

    print(f"Fitting 3-state HMM on log_vol (15 restarts, σ-filter off)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_3_vol = _fit_and_score(
            log_vol, n_regimes=3, duplicate_sigma_ratio_threshold=0.0,
        )

    # Print comparison
    print("\n" + "=" * 72)
    print("BIC Comparison — log_vol feature")
    print("=" * 72)
    print(f"{'Model':<12} {'n_params':>10} {'log-lik':>14} {'BIC':>14}")
    print("-" * 52)
    for res in (result_2_vol, result_3_vol):
        ll_str = f"{res['log_likelihood']:>14.1f}" if res['log_likelihood'] is not None else f"{'REJECTED':>14}"
        bic_str = f"{res['bic']:>14.1f}" if res['bic'] is not None else f"{'REJECTED':>14}"
        print(f"  {res['n_regimes']}-state    {res['n_params']:>10d} {ll_str} {bic_str}")

    print()
    for res in (result_2_vol, result_3_vol):
        if res['min_sigma_ratio_observed'] is None:
            audit = f"  {res['n_regimes']}-state restart audit: 0 converged"
        else:
            audit = (
                f"  {res['n_regimes']}-state restart audit: "
                f"converged={res['n_converged']}/{res['n_init']}, "
                f"rejected_occupancy={res['n_rejected_occupancy']}, "
                f"rejected_duplicate_state={res['n_rejected_duplicate']}, "
                f"min_σ_ratio_observed={res['min_sigma_ratio_observed']:.3f}"
            )
        print(audit)

    # Recommendation
    print()
    if result_3_vol["rejected_no_valid_fit"]:
        print(
            "  🔴 3-state log_vol REJECTED — even on vol feature, all 3-state\n"
            "     restarts collapsed (occupancy or duplicate-σ).\n"
            "     Conclusion: BTCUSDT volatility is genuinely bimodal."
        )
        delta_bic_vol = None
        verdict = "bimodal_confirmed"
    elif result_2_vol["bic"] is not None and result_3_vol["bic"] is not None:
        delta_bic_vol = result_3_vol["bic"] - result_2_vol["bic"]
        print(f"  ΔBIC (3-state − 2-state) on log_vol = {delta_bic_vol:+.1f}")
        if delta_bic_vol < 0:
            print(
                "  ✅ 3-state on log_vol is non-degenerate AND lower BIC.\n"
                "     Vol feature unlocks a third regime (low / medium / high vol).\n"
                "     Recommendation: switch to log_vol feature for 3-state HMM."
            )
            verdict = "trimodal_on_vol_feature"
        else:
            print(
                "  2-state on log_vol still wins BIC.\n"
                "     3-state runs without collapse but adds parameters w/o gain."
            )
            verdict = "bimodal_log_vol"
    else:
        delta_bic_vol = None
        verdict = "indeterminate"

    # Compare with raw-return baseline if available
    raw_path = DATA_DIR / "hmm_2state_vs_3state.json"
    raw_summary = None
    if raw_path.exists():
        try:
            raw_summary = json.loads(raw_path.read_text())
        except json.JSONDecodeError:
            raw_summary = None

    output = {
        "code_version": "vol-feature-2026-04-26",
        "vol_window_bars": VOL_WINDOW,
        "vol_window_minutes": VOL_WINDOW * 5,
        "n_returns": T_ret,
        "n_log_vol_obs": T_vol,
        "log_vol_stats": {
            "min": float(log_vol.min()),
            "max": float(log_vol.max()),
            "mean": float(log_vol.mean()),
            "std": float(log_vol.std()),
        },
        "two_state_log_vol": result_2_vol,
        "three_state_log_vol": result_3_vol,
        "delta_bic_3_minus_2_log_vol": float(delta_bic_vol) if delta_bic_vol is not None else None,
        "verdict": verdict,
        "raw_return_baseline": (
            {
                "preferred_model": raw_summary.get("preferred_model"),
                "delta_bic_3_minus_2": raw_summary.get("delta_bic_3_minus_2"),
                "three_state_rejected": (
                    raw_summary.get("three_state", {}).get("rejected_no_valid_fit")
                ),
            }
            if raw_summary is not None else None
        ),
    }

    out_path = DATA_DIR / "hmm_vol_feature_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
