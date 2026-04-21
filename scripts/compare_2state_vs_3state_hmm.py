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


def _fit_and_score(returns: np.ndarray, n_regimes: int) -> dict:
    """Fit an n-state HMM and return diagnostics including BIC.

    For BIC computation we use hmmlearn's total log-likelihood directly.
    We take the BEST-SCORING converged model across 15 random restarts
    (not filtered by state occupancy) so that the likelihood comparison
    is apples-to-apples.  Occupancy collapse is itself a finding — if the
    best 3-state model still puts <1% of data in a third state, that is
    evidence the data is really bimodal.

    Parameters
    ----------
    returns : np.ndarray
        1-D log-return series.
    n_regimes : int
        Number of HMM states.

    Returns
    -------
    dict with keys: n_regimes, log_likelihood, bic, n_params, regimes_info,
        collapsed (bool — True if best model has a phantom state < 1% of T)
    """
    import warnings

    T = len(returns)

    # Free parameters for a K-state univariate Gaussian HMM:
    #   transitions: K*(K-1)  (each row sums to 1, so K-1 free per row)
    #   means:       K
    #   variances:   K
    #   initial:     K-1   (sums to 1)
    K = n_regimes
    n_params = K * (K - 1) + K + K + (K - 1)

    log_likelihood = None
    collapsed = False
    n_init = 15  # enough restarts to sample the likelihood landscape

    # Use hmmlearn for log-likelihood (best-scoring convergent model, no occupancy filter)
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
            if score > best_score:
                best_score = score
                best_model = model

        if best_model is not None:
            log_likelihood = float(best_model.score(X)) * T  # total log-lik
            counts = np.bincount(best_model.predict(X), minlength=K)
            min_frac = counts.min() / T
            # Flag if the best model has a "phantom" state (< 1% of data)
            collapsed = bool(min_frac < 0.01)

    # Fallback: approximate via per-state Gaussian emission (fit_hmm path)
    if log_likelihood is None:
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

    # Call fit_hmm for regime characterisation (separate from BIC model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regimes, state_seq = fit_hmm(returns, n_regimes=n_regimes)

    regimes_info = []
    for r in regimes:
        regimes_info.append({
            "label": r.label,
            "sigma_multiplier": float(r.sigma),
            "state_vol": float(r.state_vol),
            "probability": float(r.probability),
        })

    return {
        "n_regimes": n_regimes,
        "n_observations": T,
        "n_params": n_params,
        "log_likelihood": float(log_likelihood),
        "bic": float(bic),
        "collapsed": collapsed,
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
    print("\nFitting 2-state HMM (10 random restarts)...")
    result_2 = _fit_and_score(returns, n_regimes=2)

    print("Fitting 3-state HMM (10 random restarts)...")
    result_3 = _fit_and_score(returns, n_regimes=3)

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BIC Comparison Table")
    print("=" * 72)
    print(f"{'Model':<12} {'n_params':>10} {'log-lik':>14} {'BIC':>14}")
    print("-" * 52)
    for res in (result_2, result_3):
        print(
            f"  {res['n_regimes']}-state    "
            f"{res['n_params']:>10d} "
            f"{res['log_likelihood']:>14.1f} "
            f"{res['bic']:>14.1f}"
        )

    bic_2 = result_2["bic"]
    bic_3 = result_3["bic"]
    delta_bic = bic_3 - bic_2

    if result_3["collapsed"]:
        print(
            "\n  NOTE: Best 3-state hmmlearn model has a phantom state with <1%\n"
            "  of observations — the 3-state solution is unstable on this data.\n"
            "  This is a data finding: BTCUSDT 5-min returns are bimodal, not\n"
            "  trimodal.  BIC still computed for completeness."
        )

    print()
    print(f"  ΔBIC (3-state − 2-state) = {delta_bic:+.1f}")
    if delta_bic < 0:
        print(
            "  3-state BIC is LOWER → data supports a third regime.\n"
            "  Recommendation: use 3-state going forward."
        )
    else:
        print(
            f"  2-state BIC is LOWER by {abs(delta_bic):.1f} → simpler model preferred.\n"
            "  Recommendation: keep 2-state; additional regime is not justified\n"
            "  by data (complexity penalty exceeds log-likelihood gain)."
        )

    # ── Detailed regime table ─────────────────────────────────────────────
    for res in (result_2, result_3):
        print(f"\n  {res['n_regimes']}-state regimes:")
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
        "two_state": result_2,
        "three_state": result_3,
        "delta_bic_3_minus_2": float(delta_bic),
        "preferred_model": "3-state" if delta_bic < 0 else "2-state",
        "preference_reason": (
            "lower BIC (better fit adjusted for complexity)"
            if delta_bic < 0
            else (
                "lower BIC; 3-state model collapses to phantom third state "
                "(data is bimodal, not trimodal)"
                if result_3["collapsed"]
                else "lower BIC (simpler model, data does not support 3 states)"
            )
        ),
        "three_state_collapse_note": (
            "Best 3-state hmmlearn solution has a phantom state with <1% occupancy. "
            "3-state HMM code is functional (confirmed on synthetic 3-regime data) "
            "but this BTCUSDT dataset does not exhibit a separable third regime."
        ) if result_3["collapsed"] else None,
    }

    out_path = DATA_DIR / "hmm_2state_vs_3state.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
