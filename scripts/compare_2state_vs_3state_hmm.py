"""Compare 2-state vs 3-state HMM on 98-day BTCUSDT 5-min returns.

Computes BIC for each model:
    BIC = -2 * log_likelihood + k * ln(T)
where k = number of free parameters and T = number of observations.

For a K-state Gaussian HMM on 1-D returns:
    k = K*(K-1)  [transition matrix rows minus one constraint each]
      + K        [means]
      + K        [variances]
      + (K-1)    [initial probs minus constraint]
    = K^2 + 2K - 1   (simplified)

Prints a comparison table and saves results to
    data/hmm_2state_vs_3state.json

Usage:
    python scripts/compare_2state_vs_3state_hmm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration.data_loader import load_trades, compute_mid_prices
from extensions.regime import fit_hmm, _HAS_HMMLEARN

# ── Try hmmlearn for log-likelihood extraction ─────────────────────────────
if _HAS_HMMLEARN:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILES = sorted(DATA_DIR.glob("BTCUSDT-aggTrades-2026-*.csv"))


def _fit_and_score(
    returns: np.ndarray,
    n_regimes: int,
    n_init: int = 15,
    duplicate_sigma_ratio_threshold: float = 1.15,
) -> dict:
    """Fit an n-state HMM and return diagnostics including BIC.

    Two filters reject degenerate fits during the multi-restart search:
      1. Occupancy collapse: any state with <1% of observations.
      2. Duplicate-state collapse: any two states whose σ ratio < threshold
         (default 1.15) — i.e., two states are statistically indistinguishable.

    BIC is computed on the BEST log-likelihood model that passes both filters.
    If no fit passes for a given K, we record bic=None and
    rejected_no_valid_fit=True — strong evidence the data does not support
    K distinct regimes.

    Parameters
    ----------
    returns : np.ndarray
        1-D feature series (raw log returns or a derived feature like log-vol).
    n_regimes : int
        Number of HMM states.
    n_init : int
        Number of random restarts.
    duplicate_sigma_ratio_threshold : float
        Reject fits where any two states have σ ratio below this threshold.
        1.15 means: states must differ in σ by at least 15% to count as distinct.
    """
    import warnings

    T = len(returns)
    K = n_regimes
    n_params = K * (K - 1) + K + K + (K - 1)

    log_likelihood = None
    bic = None
    occupancy_collapsed = False
    duplicate_collapsed = False
    rejected_no_valid_fit = False
    n_converged = 0
    n_rejected_occupancy = 0
    n_rejected_duplicate = 0
    min_sigma_ratio_observed = float("inf")
    best_model_min_state_frac = None
    best_model_min_sigma_ratio = None

    if _HAS_HMMLEARN:
        best_score = -np.inf
        best_model = None
        X = returns.reshape(-1, 1)

        for seed in range(n_init):
            model = _GaussianHMM(
                n_components=K,
                covariance_type="full",
                n_iter=500,
                tol=1e-6,
                random_state=seed,
                min_covar=1e-12,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X)
                score = model.score(X)
            except Exception:
                continue
            n_converged += 1

            # Track diagnostics across all converged fits
            counts = np.bincount(model.predict(X), minlength=K)
            min_frac = counts.min() / T
            state_sigmas = np.sqrt(model.covars_.reshape(K, -1)[:, 0])
            sorted_sigmas = np.sort(state_sigmas)
            if K > 1:
                ratios = sorted_sigmas[1:] / np.maximum(sorted_sigmas[:-1], 1e-12)
                min_ratio = float(ratios.min())
                if min_ratio < min_sigma_ratio_observed:
                    min_sigma_ratio_observed = min_ratio
            else:
                min_ratio = float("inf")

            # Filter 1: occupancy collapse (<1% in any state)
            if min_frac < 0.01:
                n_rejected_occupancy += 1
                continue

            # Filter 2: duplicate-state collapse (σ ratio < threshold)
            if K > 1 and min_ratio < duplicate_sigma_ratio_threshold:
                n_rejected_duplicate += 1
                continue

            if score > best_score:
                best_score = score
                best_model = model
                best_model_min_state_frac = float(min_frac)
                best_model_min_sigma_ratio = float(min_ratio)

        if best_model is not None:
            # FIX 2026-04-26: hmmlearn's score(X) already returns total log-lik
            # of the entire sequence (forward algorithm). Previous code multiplied
            # by T which inflated by ~28000x — gave BIC values ~10^9 when true
            # values are ~10^5.
            log_likelihood = float(best_model.score(X))
            bic = -2.0 * log_likelihood + n_params * np.log(T)
            # Diagnostic flags (descriptive — best_model already passed both filters)
            occupancy_collapsed = bool(best_model_min_state_frac < 0.05)
            duplicate_collapsed = False
        else:
            rejected_no_valid_fit = True

    # Fallback: approximate via per-state Gaussian emission (only if hmmlearn unavailable)
    if log_likelihood is None and not _HAS_HMMLEARN:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regimes_fb, state_seq_fb = fit_hmm(returns, n_regimes=n_regimes)
        log_likelihood = 0.0
        label_to_idx = {"risk_on": 0, "neutral": 1, "risk_off": 2} if K == 3 else {"risk_on": 0, "risk_off": 1}
        for r in regimes_fb:
            cidx = label_to_idx[r.label]
            sub = returns[state_seq_fb == cidx]
            if len(sub) > 1:
                mu = np.mean(sub)
                sigma = max(np.std(sub, ddof=1), 1e-12)
                log_likelihood += float(np.sum(
                    -0.5 * np.log(2 * np.pi) - np.log(sigma)
                    - 0.5 * ((sub - mu) / sigma) ** 2
                ))
        bic = -2.0 * log_likelihood + n_params * np.log(T)

    # Call fit_hmm for regime characterisation (separate from BIC model;
    # uses production filter which may give different model than BIC fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            regimes, state_seq = fit_hmm(returns, n_regimes=n_regimes)
            regimes_info = [{
                "label": r.label,
                "sigma_multiplier": float(r.sigma),
                "state_vol": float(r.state_vol),
                "probability": float(r.probability),
            } for r in regimes]
        except Exception as e:
            regimes_info = []
            print(f"  WARNING: fit_hmm raised {type(e).__name__}: {e}")

    return {
        "n_regimes": n_regimes,
        "n_observations": T,
        "n_params": n_params,
        "log_likelihood": float(log_likelihood) if log_likelihood is not None else None,
        "bic": float(bic) if bic is not None else None,
        "rejected_no_valid_fit": rejected_no_valid_fit,
        "n_init": n_init,
        "n_converged": n_converged,
        "n_rejected_occupancy": n_rejected_occupancy,
        "n_rejected_duplicate": n_rejected_duplicate,
        "min_sigma_ratio_observed": (
            float(min_sigma_ratio_observed)
            if min_sigma_ratio_observed != float("inf") else None
        ),
        "duplicate_sigma_ratio_threshold": duplicate_sigma_ratio_threshold,
        "best_model_min_state_frac": best_model_min_state_frac,
        "best_model_min_sigma_ratio": best_model_min_sigma_ratio,
        "occupancy_collapsed": occupancy_collapsed,
        # legacy field name kept for backward compat with downstream consumers
        "collapsed": rejected_no_valid_fit or occupancy_collapsed,
        "regimes": regimes_info,
    }


def main() -> None:
    print("\n" + "=" * 72)
    print("Compare 2-state vs 3-state Gaussian HMM on BTCUSDT 5-min returns")
    print("=" * 72)

    if not DATA_FILES:
        raise FileNotFoundError(
            f"No BTCUSDT-aggTrades-2026-*.csv files found in {DATA_DIR}."
        )

    print(f"\nLoading {len(DATA_FILES)} CSV files (full 98-day window)...")
    trades = load_trades(
        DATA_DIR,
        start="2026-01-01",
        end="2026-04-08",
    )
    print(f"  {len(trades):,} trades loaded")

    # ── Compute 5-min log returns ─────────────────────────────────────────
    mid = compute_mid_prices(trades, freq="5min").copy()
    mid["log_return"] = np.log(mid["mid_price"]).diff()
    mid = mid.dropna()
    returns = mid["log_return"].to_numpy()
    returns = returns[np.isfinite(returns)]
    T = len(returns)
    print(f"  {T:,} 5-min return observations")

    # ── Fit 2-state and 3-state HMMs ─────────────────────────────────────
    print("\nFitting 2-state HMM (15 random restarts; reject occupancy <1% & σ ratio <1.15)...")
    result_2 = _fit_and_score(returns, n_regimes=2)

    print("Fitting 3-state HMM (15 random restarts; reject occupancy <1% & σ ratio <1.15)...")
    result_3 = _fit_and_score(returns, n_regimes=3)

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BIC Comparison Table")
    print("=" * 72)
    print(f"{'Model':<12} {'n_params':>10} {'log-lik':>14} {'BIC':>14}")
    print("-" * 52)
    for res in (result_2, result_3):
        ll_str = f"{res['log_likelihood']:>14.1f}" if res['log_likelihood'] is not None else f"{'REJECTED':>14}"
        bic_str = f"{res['bic']:>14.1f}" if res['bic'] is not None else f"{'REJECTED':>14}"
        print(
            f"  {res['n_regimes']}-state    "
            f"{res['n_params']:>10d} "
            f"{ll_str} {bic_str}"
        )

    # Diagnostic: restart accounting + min sigma ratio
    print()
    for res in (result_2, result_3):
        print(
            f"  {res['n_regimes']}-state restart audit: "
            f"converged={res['n_converged']}/{res['n_init']}, "
            f"rejected_occupancy={res['n_rejected_occupancy']}, "
            f"rejected_duplicate_state={res['n_rejected_duplicate']}, "
            f"min_σ_ratio_observed={res['min_sigma_ratio_observed']:.3f}"
            if res['min_sigma_ratio_observed'] is not None
            else f"  {res['n_regimes']}-state restart audit: 0 converged"
        )

    bic_2 = result_2["bic"]
    bic_3 = result_3["bic"]

    print()
    if result_3["rejected_no_valid_fit"]:
        delta_bic = None
        preferred_model = "2-state"
        preference_reason = (
            "3-state has NO valid fit: every restart either has a state with "
            "<1% occupancy or two states with σ ratio <1.15 (duplicate-state "
            "collapse). BTCUSDT 5-min returns are bimodal, not trimodal."
        )
        print(
            "  🔴 3-state REJECTED — no restart produced non-degenerate fit.\n"
            "  All 3-state fits collapsed to either phantom-occupancy or duplicate-σ.\n"
            "  Recommendation: keep 2-state. Data is bimodal."
        )
    elif bic_3 is None or bic_2 is None:
        delta_bic = None
        preferred_model = "2-state" if bic_2 is not None else "neither"
        preference_reason = "one model has no valid BIC"
        print("  Cannot compare BIC — one model produced no valid fit.")
    else:
        delta_bic = bic_3 - bic_2
        print(f"  ΔBIC (3-state − 2-state) = {delta_bic:+.1f}")
        if delta_bic < 0:
            preferred_model = "3-state"
            preference_reason = (
                "3-state has lower BIC even after rejecting occupancy-collapsed "
                "and duplicate-state fits"
            )
            print(
                "  3-state BIC is LOWER → data supports a third regime.\n"
                "  Recommendation: use 3-state going forward."
            )
        else:
            preferred_model = "2-state"
            preference_reason = "2-state has lower BIC; complexity penalty exceeds log-likelihood gain"
            print(
                f"  2-state BIC is LOWER by {abs(delta_bic):.1f} → simpler model preferred.\n"
                "  Recommendation: keep 2-state; additional regime is not justified\n"
                "  by data (complexity penalty exceeds log-likelihood gain)."
            )

    # ── Detailed regime table ─────────────────────────────────────────────
    for res in (result_2, result_3):
        print(f"\n  {res['n_regimes']}-state regimes (from production fit_hmm path):")
        if not res['regimes']:
            print("    (no regimes — fit_hmm raised an exception)")
            continue
        print(f"  {'Label':<12} {'σ_mult':>8} {'state_vol':>12} {'prob':>8}")
        print("  " + "-" * 44)
        for r in res["regimes"]:
            print(
                f"  {r['label']:<12} "
                f"{r['sigma_multiplier']:>8.4f} "
                f"{r['state_vol']:>12.6f} "
                f"{r['probability']:>8.4f}"
            )

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "code_version": "post-bic-bug-fix-2026-04-26",
        "fix_notes": (
            "Fixed log-likelihood scaling bug (score(X) is already total, no *T). "
            "Added duplicate-state σ-ratio filter (threshold 1.15) in addition "
            "to occupancy filter."
        ),
        "two_state": result_2,
        "three_state": result_3,
        "delta_bic_3_minus_2": float(delta_bic) if delta_bic is not None else None,
        "preferred_model": preferred_model,
        "preference_reason": preference_reason,
    }

    out_path = DATA_DIR / "hmm_2state_vs_3state.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
