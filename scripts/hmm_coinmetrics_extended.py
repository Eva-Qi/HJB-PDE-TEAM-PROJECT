"""HMM regime experiment: screen CoinMetrics "extended" metrics as HMM features.

Context
-------
We have ``data/coinmetrics_btc_extended_metrics.json`` that was pulled but never
loaded into any model. The task brief hypothesised exchange inflow/outflow,
but those fields are NOT in this file — they live in
``coinmetrics_btc_onchain.json`` and were already wired into
``scripts/bivariate_hmm_coinmetrics.py``. The extended file actually contains:

    ['AdrActCnt', 'TxCnt', 'CapMrktCurUSD', 'CapMVRVCur', 'HashRate',
     'ReferenceRateUSD']

plus 5 fields that came back HTTP 403 (``TxTfrValAdjUSD``, ``VtyDayRet30d``,
``FeeMeanUSD``, ``CapRealUSD``, ``NVTAdj``). ``ReferenceRateUSD`` has only
7/296 non-null rows → excluded. That leaves 5 live candidates.

Procedure
---------
1. Load CoinMetrics extended JSON; drop ``ReferenceRateUSD`` (coverage too low).
2. Build a daily realized-vol series from the same Binance 5-min bars used in
   the baseline bivariate script (window: 2026-01-01 → 2026-04-08, matching
   ``data/hmm_bivariate_coinmetrics_results.json``). RV = sqrt of the sum of
   squared 5-min log returns within each UTC day.
3. Screen each candidate on two criteria:
     - Sarle's bimodality coefficient: BC = (skew**2 + 1) / (kurt + 3)
       where kurt is the excess kurtosis. BC > 5/9 ≈ 0.555 hints at bimodality.
     - |Pearson correlation with daily realized vol|. Low is better — a
       feature highly correlated with vol is redundant.
4. Pick the winner (highest BC with a sanity-low |corr|) and fit a 2-state
   Gaussian HMM on (5-min log_return, ffilled winner feature).
5. For apples-to-apples AIC/BIC, also refit the existing bivariate baseline
   (log_return + log(FlowInExUSD), same window) with the SAME hmmlearn call.
   (The existing results JSON never recorded AIC/BIC, so there is nothing to
   directly read off of disk.)
6. Save everything to ``data/hmm_coinmetrics_extended_results.json``.

Hard rules
----------
- No modification of existing scripts / existing results.
- No imputation: rows with NaN in the candidate feature are dropped.
- If a field is missing/empty we say so and skip it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from calibration.data_loader import load_trades, compute_ohlc  # noqa: E402
from extensions.regime import align_sentiment_to_returns_bar  # noqa: E402


# ── Config (mirrors the baseline bivariate script window) ──────────────────
WINDOW_START = "2026-01-01"
WINDOW_END = "2026-04-08"
N_REGIMES = 2
N_INIT = 5                       # same as extensions.regime._fit_hmm_hmmlearn_multifeature
SEED = 0

# ReferenceRateUSD has only 7/296 non-null values → drop it outright
LOW_COVERAGE = {"ReferenceRateUSD"}

# Metrics that came back HTTP 403 from CoinMetrics community tier
FAILED_METRICS_NOTE = [
    "TxTfrValAdjUSD", "VtyDayRet30d", "FeeMeanUSD", "CapRealUSD", "NVTAdj",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient.

        BC = (skew**2 + 1) / (kurt + 3)

    ``skew`` is the sample skew and ``kurt`` is the EXCESS kurtosis
    (normal distribution = 0). BC ranges (0, 1]; BC > 5/9 ≈ 0.555 is the
    classic threshold indicating possible bimodality.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    sk = stats.skew(x, bias=False)
    kt = stats.kurtosis(x, bias=False, fisher=True)  # excess kurtosis
    denom = kt + 3.0
    if denom <= 0:
        return float("nan")
    return float((sk ** 2 + 1.0) / denom)


def fit_gaussian_hmm_full(
    X: np.ndarray, n_states: int = 2, n_init: int = 5, seed: int = 0,
) -> tuple[GaussianHMM, float, int, int]:
    """Multi-restart Gaussian HMM fit with full covariance.

    Returns (best_model, log_likelihood, n_params, T).

    n_params for full-cov Gaussian HMM (d features, k states):
        means:        k * d
        covariances:  k * d * (d+1) / 2      (symmetric)
        transmat:     k * (k - 1)            (rows sum to 1)
        startprob:    k - 1                  (sums to 1)
    """
    T, d = X.shape
    best_score = -np.inf
    best_model = None
    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            tol=1e-6,
            random_state=seed + i,
            min_covar=1e-12,
        )
        try:
            model.fit(X)
            score = model.score(X)
            counts = np.bincount(model.predict(X), minlength=n_states)
            if counts.min() < 0.05 * T:  # same degeneracy guard
                continue
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError(
            "All HMM restarts degenerated to a single state — check features."
        )

    k = n_states
    n_params = (
        k * d                          # means
        + k * d * (d + 1) // 2         # covariances
        + k * (k - 1)                  # transmat
        + (k - 1)                      # startprob
    )
    return best_model, float(best_score), int(n_params), int(T)


def aic_bic(log_lik: float, n_params: int, T: int) -> tuple[float, float]:
    aic = 2 * n_params - 2 * log_lik
    bic = n_params * np.log(T) - 2 * log_lik
    return float(aic), float(bic)


def summarize_hmm(
    model: GaussianHMM, X: np.ndarray, log_lik: float, n_params: int,
    feature_names: list[str],
) -> dict:
    """Canonical-order summary (state 0 = low return-vol = risk_on).

    Sorts states by the per-state std of column 0 (log_return).
    Returns means/variances for every feature, transition matrix, AIC/BIC.
    """
    states = model.predict(X)
    T, d = X.shape
    # Per-state std of column 0 (log_return) is the canonical vol axis
    stds_col0 = []
    for s in range(model.n_components):
        mask = states == s
        stds_col0.append(float(np.std(X[mask, 0], ddof=1)) if mask.sum() > 1 else np.inf)
    order = np.argsort(stds_col0)  # ascending → [risk_on, risk_off]

    perm = {raw: canon for canon, raw in enumerate(order)}
    canon_states = np.array([perm[s] for s in states], dtype=int)

    means = model.means_[order]                      # (k, d)
    covars = model.covars_[order]                    # (k, d, d)
    variances = np.array([np.diag(C) for C in covars])  # (k, d)

    transmat = model.transmat_[order][:, order]
    startprob = model.startprob_[order]

    aic, bic = aic_bic(log_lik, n_params, T)

    regime_blobs = []
    labels = ["risk_on", "risk_off"] if model.n_components == 2 else [
        f"state_{i}" for i in range(model.n_components)
    ]
    for canon_idx in range(model.n_components):
        mask = canon_states == canon_idx
        regime_blobs.append({
            "label": labels[canon_idx],
            "probability": float(mask.mean()),
            "stationary_startprob": float(startprob[canon_idx]),
            "means": {
                name: float(means[canon_idx, j])
                for j, name in enumerate(feature_names)
            },
            "variances": {
                name: float(variances[canon_idx, j])
                for j, name in enumerate(feature_names)
            },
            "log_return_std": float(np.sqrt(variances[canon_idx, 0])),
        })

    return {
        "feature_names": feature_names,
        "n_observations": int(T),
        "n_features": int(d),
        "n_states": int(model.n_components),
        "log_likelihood": float(log_lik),
        "n_params": int(n_params),
        "AIC": aic,
        "BIC": bic,
        "regimes": regime_blobs,
        "transition_matrix": transmat.tolist(),
        "transition_matrix_rows": labels,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 80)
    print("HMM feature screening — CoinMetrics extended metrics")
    print("=" * 80)

    # ── 1. Load extended metrics ─────────────────────────────────────────
    ext_path = PROJECT / "data" / "coinmetrics_btc_extended_metrics.json"
    with open(ext_path) as f:
        payload = json.load(f)
    print(f"\n[1/6] Loaded {ext_path.name}")
    print(f"      covers {payload['start']} → {payload['end']}")
    print(f"      retrieved: {payload['metrics_retrieved']}")
    print(f"      failed (HTTP 403, unavailable on community tier): "
          f"{list(payload['metrics_failed'].keys())}")

    ext = pd.DataFrame(payload["data"])
    ext["date"] = pd.to_datetime(ext["date"])
    ext = ext.sort_values("date").reset_index(drop=True)

    candidate_cols = [
        c for c in payload["metrics_retrieved"]
        if c not in LOW_COVERAGE and c in ext.columns
    ]
    dropped_low_cov = [
        c for c in payload["metrics_retrieved"]
        if c in LOW_COVERAGE
    ]
    print(f"      dropped (low coverage, <10% non-null): {dropped_low_cov}")
    print(f"      candidates to screen: {candidate_cols}")

    # ── 2. Build realized vol series from the SAME window the baseline used ──
    print(f"\n[2/6] Loading Binance aggTrades → 5-min log returns "
          f"[{WINDOW_START}, {WINDOW_END}]")
    trades = load_trades(PROJECT / "data", start=WINDOW_START, end=WINDOW_END)
    ohlc = compute_ohlc(trades, freq="5min")
    ohlc["log_return"] = np.log(ohlc["close"] / ohlc["close"].shift(1))
    ohlc = ohlc.dropna(subset=["log_return"]).reset_index(drop=True)
    print(f"      {len(ohlc):,} 5-min bars")

    # Daily realized vol (UTC) from 5-min log returns
    r = ohlc.copy()
    r["date"] = r["timestamp"].dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    rv_daily = (
        r.groupby("date")["log_return"]
         .apply(lambda s: float(np.sqrt(np.sum(s.values ** 2))))
         .reset_index()
         .rename(columns={"log_return": "realized_vol"})
    )
    print(f"      {len(rv_daily)} daily RV rows, "
          f"{rv_daily['date'].min().date()} → {rv_daily['date'].max().date()}")

    # ── 3. Screen candidates on bimodality + |corr w/ RV| ───────────────
    print(f"\n[3/6] Screening {len(candidate_cols)} candidate features")
    # Intersect CoinMetrics dates with the RV window
    ext_win = ext[(ext["date"] >= rv_daily["date"].min()) &
                  (ext["date"] <= rv_daily["date"].max())].copy()
    merged = pd.merge(ext_win, rv_daily, on="date", how="inner")
    print(f"      merged daily panel: {len(merged)} rows "
          f"({merged['date'].min().date()} → {merged['date'].max().date()})")

    screen_rows = []
    for col in candidate_cols:
        if col not in merged.columns:
            screen_rows.append({"feature": col, "note": "missing from frame",
                                "n_non_null": 0})
            continue
        raw = merged[col].dropna().values
        n_nn = int(len(raw))
        if n_nn < 20:
            screen_rows.append({
                "feature": col, "n_non_null": n_nn,
                "note": "too few non-null obs — skipped",
            })
            continue
        # Level + log-level variants (many on-chain metrics are skewed)
        bc_lvl = bimodality_coefficient(raw)
        # Robust |corr| with RV on the joint non-null sample
        sub = merged[["date", col, "realized_vol"]].dropna()
        rv_corr = float(sub[col].corr(sub["realized_vol"]))

        # Log variant, only if strictly positive
        if np.all(raw > 0):
            log_raw = np.log(raw)
            bc_log = bimodality_coefficient(log_raw)
            log_corr = float(np.log(sub[col]).corr(sub["realized_vol"]))
        else:
            bc_log = float("nan")
            log_corr = float("nan")

        # Also try first-difference (change-based signal)
        diff = np.diff(raw)
        bc_diff = bimodality_coefficient(diff)
        sub_diff = sub.copy()
        sub_diff["d" + col] = sub_diff[col].diff()
        diff_corr = float(sub_diff["d" + col].dropna().corr(
            sub_diff["realized_vol"].iloc[1:]
        ))

        screen_rows.append({
            "feature": col,
            "n_non_null": n_nn,
            "mean": float(np.mean(raw)),
            "std": float(np.std(raw, ddof=1)),
            "BC_level": float(bc_lvl),
            "BC_log": float(bc_log),
            "BC_diff": float(bc_diff),
            "abs_corr_RV_level": float(abs(rv_corr)),
            "abs_corr_RV_log": float(abs(log_corr)) if np.isfinite(log_corr) else float("nan"),
            "abs_corr_RV_diff": float(abs(diff_corr)) if np.isfinite(diff_corr) else float("nan"),
        })

    screen_df = pd.DataFrame(screen_rows)
    print("\n  Screening table (BC > 0.555 hints at bimodality; low |corr| is better):")
    print(screen_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 4. Pick winner ───────────────────────────────────────────────────
    # Rank by (best available BC across level/log/diff) minus a corr penalty.
    # Prefer the variant with highest BC among {level, log, diff} for each feature,
    # and subtract |corr_RV| of that variant as a dilution penalty.
    def best_variant(row):
        variants = []
        if np.isfinite(row.get("BC_level", np.nan)):
            variants.append(("level", row["BC_level"], row["abs_corr_RV_level"]))
        if np.isfinite(row.get("BC_log", np.nan)):
            variants.append(("log", row["BC_log"], row["abs_corr_RV_log"]))
        if np.isfinite(row.get("BC_diff", np.nan)):
            variants.append(("diff", row["BC_diff"], row["abs_corr_RV_diff"]))
        if not variants:
            return (None, np.nan, np.nan, np.nan)
        # score = BC − |corr|    (both in [0, 1])
        best = max(variants, key=lambda t: t[1] - (t[2] if np.isfinite(t[2]) else 0.0))
        return (best[0], best[1], best[2], best[1] - (best[2] if np.isfinite(best[2]) else 0.0))

    ranked = []
    for _, row in screen_df.iterrows():
        if "note" in row and isinstance(row.get("note"), str):
            continue
        variant, bc, cr, score = best_variant(row)
        if variant is None:
            continue
        ranked.append({
            "feature": row["feature"],
            "variant": variant,
            "BC": float(bc),
            "abs_corr_RV": float(cr),
            "score": float(score),
        })
    ranked_df = pd.DataFrame(ranked).sort_values(
        "score", ascending=False,
    ).reset_index(drop=True)
    print("\n  Ranking (score = BC − |corr_RV|):")
    print(ranked_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if ranked_df.empty:
        raise RuntimeError("No viable features after screening — abort.")
    winner = ranked_df.iloc[0].to_dict()
    winner_col = winner["feature"]
    winner_variant = winner["variant"]
    print(f"\n  WINNER: {winner_col} ({winner_variant}) "
          f"BC={winner['BC']:.4f}, |corr_RV|={winner['abs_corr_RV']:.4f}")

    # ── 5. Fit HMM on (log_return, winner_feature) ──────────────────────
    print(f"\n[4/6] Fitting HMM with feature = {winner_col} [{winner_variant}]")

    # Build the daily winner series
    daily = ext_win[["date", winner_col]].dropna().copy()
    if winner_variant == "log":
        daily[winner_col] = np.log(daily[winner_col])
        feature_label = f"log_{winner_col}"
    elif winner_variant == "diff":
        daily[winner_col] = daily[winner_col].diff()
        daily = daily.dropna()
        feature_label = f"d_{winner_col}"
    else:
        feature_label = winner_col
    daily = daily.rename(columns={winner_col: feature_label})
    print(f"      daily feature rows: {len(daily)}")

    returns_df = ohlc[["timestamp", "log_return"]].copy()
    X_winner = align_sentiment_to_returns_bar(
        returns_df, daily.rename(columns={"date": "date"}),
        returns_col="log_return", sentiment_col=feature_label,
    )
    print(f"      aligned feature matrix: {X_winner.shape}")

    model_w, ll_w, np_w, T_w = fit_gaussian_hmm_full(
        X_winner, n_states=N_REGIMES, n_init=N_INIT, seed=SEED,
    )
    summary_w = summarize_hmm(
        model_w, X_winner, ll_w, np_w,
        feature_names=["log_return", feature_label],
    )
    print(f"      log-lik={ll_w:.2f}  n_params={np_w}  T={T_w}")
    print(f"      AIC={summary_w['AIC']:.2f}  BIC={summary_w['BIC']:.2f}")

    # ── 6. Refit baseline (log_return + log(FlowInExUSD)) for AIC/BIC ────
    print("\n[5/6] Refitting baseline (log_return + log(FlowInExUSD)) for "
          "apples-to-apples AIC/BIC")
    onchain_path = PROJECT / "data" / "coinmetrics_btc_onchain.json"
    with open(onchain_path) as f:
        oc_payload = json.load(f)
    oc = pd.DataFrame(oc_payload["data"])
    oc["date"] = pd.to_datetime(oc["date"])
    oc = oc[["date", "FlowInExUSD"]].dropna().copy()
    oc["logFlowInExUSD"] = np.log(oc["FlowInExUSD"])
    oc = oc[["date", "logFlowInExUSD"]]
    X_base = align_sentiment_to_returns_bar(
        returns_df, oc, returns_col="log_return", sentiment_col="logFlowInExUSD",
    )
    print(f"      baseline feature matrix: {X_base.shape}")
    model_b, ll_b, np_b, T_b = fit_gaussian_hmm_full(
        X_base, n_states=N_REGIMES, n_init=N_INIT, seed=SEED,
    )
    summary_b = summarize_hmm(
        model_b, X_base, ll_b, np_b,
        feature_names=["log_return", "logFlowInExUSD"],
    )
    print(f"      log-lik={ll_b:.2f}  n_params={np_b}  T={T_b}")
    print(f"      AIC={summary_b['AIC']:.2f}  BIC={summary_b['BIC']:.2f}")

    # Sanity: is the WINNER HMM's risk_off period also a high-RV period?
    states_w = model_w.predict(X_winner)
    # Reorder to canonical
    stds0 = []
    for s in range(model_w.n_components):
        stds0.append(float(np.std(X_winner[states_w == s, 0], ddof=1))
                     if (states_w == s).sum() > 1 else np.inf)
    order_w = np.argsort(stds0)
    canon_w = np.array([int(np.where(order_w == s)[0][0]) for s in states_w])
    per_bar_df = pd.DataFrame({
        "timestamp": returns_df["timestamp"].iloc[-len(canon_w):].values,
        "canonical_state": canon_w,
    })
    per_bar_df["date"] = pd.to_datetime(per_bar_df["timestamp"]).dt.tz_convert(
        "UTC"
    ).dt.floor("D").dt.tz_localize(None)
    daily_state = per_bar_df.groupby("date")["canonical_state"].agg(
        lambda s: int(pd.Series(s).mode().iloc[0])
    ).reset_index()
    sanity = pd.merge(daily_state, rv_daily, on="date", how="inner")
    rv_on = sanity.loc[sanity["canonical_state"] == 0, "realized_vol"].mean()
    rv_off = sanity.loc[sanity["canonical_state"] == 1, "realized_vol"].mean()
    regime_alignment = {
        "mean_rv_risk_on_days": float(rv_on) if np.isfinite(rv_on) else None,
        "mean_rv_risk_off_days": float(rv_off) if np.isfinite(rv_off) else None,
        "rv_spread_pct": (
            float((rv_off - rv_on) / rv_on * 100) if rv_on and np.isfinite(rv_on)
            else None
        ),
        "note": (
            "risk_off days should have higher realized vol than risk_on days "
            "if the HMM's regime split is consistent with vol-based risk-on/off."
        ),
    }
    print(f"\n[6/6] Regime alignment sanity check")
    print(f"      RV on  risk_on days  = {rv_on:.5f}")
    print(f"      RV on  risk_off days = {rv_off:.5f}")
    if rv_on and np.isfinite(rv_on):
        print(f"      spread            = {regime_alignment['rv_spread_pct']:.1f}%")

    # ── Assemble result payload ──────────────────────────────────────────
    results = {
        "experiment": "hmm_coinmetrics_extended_feature_screening",
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "extended_metrics_file": str(ext_path.name),
        "metrics_unavailable_http_403": FAILED_METRICS_NOTE,
        "metrics_dropped_low_coverage": sorted(dropped_low_cov),
        "candidate_features": candidate_cols,
        "screening_table": screen_df.to_dict(orient="records"),
        "ranking": ranked_df.to_dict(orient="records"),
        "winner": {
            "feature": winner_col,
            "variant": winner_variant,
            "BC": winner["BC"],
            "abs_corr_RV": winner["abs_corr_RV"],
            "score": winner["score"],
        },
        "winner_hmm": summary_w,
        "baseline_hmm_logflowin": summary_b,
        "comparison_vs_baseline": {
            "winner_feature": f"log_return + {feature_label}",
            "baseline_feature": "log_return + logFlowInExUSD",
            "winner_AIC": summary_w["AIC"],
            "baseline_AIC": summary_b["AIC"],
            "AIC_delta_winner_minus_baseline": (
                summary_w["AIC"] - summary_b["AIC"]
            ),
            "winner_BIC": summary_w["BIC"],
            "baseline_BIC": summary_b["BIC"],
            "BIC_delta_winner_minus_baseline": (
                summary_w["BIC"] - summary_b["BIC"]
            ),
            "note": (
                "Negative delta = winner beats baseline. "
                "The baseline was refit here (not read from disk) because the "
                "existing results JSON did not record AIC/BIC."
            ),
        },
        "regime_alignment_sanity": regime_alignment,
        "caveats": [
            "Task brief mentioned exchange flows, but the extended metrics "
            "file does not contain FlowInExUSD/FlowOutExUSD — those are in "
            "coinmetrics_btc_onchain.json and were already wired by "
            "scripts/bivariate_hmm_coinmetrics.py.",
            "5 requested fields (TxTfrValAdjUSD, VtyDayRet30d, FeeMeanUSD, "
            "CapRealUSD, NVTAdj) returned HTTP 403 and are unavailable on the "
            "community-tier CoinMetrics endpoint — so MVRV is the only classic "
            "valuation regime signal we can actually test.",
            "Daily on-chain features are forward-filled to 5-min bars. Prior "
            "work (F&G, FlowIn) shows this mismatch compresses HMM regime "
            "spread; the same caveat applies here.",
        ],
    }

    out = PROJECT / "data" / "hmm_coinmetrics_extended_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
