"""HMM with Deribit DVOL as a second feature + ρ vs DVOL regime test.

Two independent experiments wired in one script:

Experiment 1 — ρ (Heston) vs DVOL regime
    Pair each of the 12 monthly Heston-Q calibrations (BTC only) with the
    mean Deribit DVOL during that month. Test whether high-DVOL months
    correspond to Heston ρ < 0 (fear regime) and low-DVOL months to ρ > 0
    (euphoria regime). Saves scatter plot to plots/dvol_vs_rho_scatter.png.

Experiment 2 — Bivariate HMM with DVOL
    Fit a 2-state Gaussian HMM on the 2026-01-01 → 2026-04-08 window:
        baseline:  log_return only               (5-min bars)
        bivariate: (log_return, DVOL)            (5-min bars, DVOL ffilled)
    Compare AIC / BIC / log-likelihood + regime means / variances /
    transition matrix. Results written to data/hmm_dvol_results.json.

Hard rules:
    • NEW files only. Does not touch existing HMM scripts.
    • If DVOL date coverage doesn't overlap Heston months, report and skip
      rather than interpolate.
    • No Binance or DVOL data is re-downloaded; both read from disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.data_loader import load_trades, compute_ohlc


PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
PLOTS = PROJECT / "plots"
PLOTS.mkdir(exist_ok=True)


# ────────────────────────────────────────────────────────────────────────
# Data loaders
# ────────────────────────────────────────────────────────────────────────

def load_dvol() -> pd.DataFrame:
    with open(DATA / "deribit_dvol_btc_daily.json") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)


def load_heston_btc() -> pd.DataFrame:
    """Load Heston monthly snapshots, filter to BTC only (exclude ETH_*)."""
    with open(DATA / "heston_qmeasure_time_series.json") as f:
        payload = json.load(f)
    rows = [r for r in payload if not r["date"].startswith("ETH_")]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────
# Experiment 1 — ρ vs DVOL
# ────────────────────────────────────────────────────────────────────────

def pair_rho_with_monthly_dvol(heston: pd.DataFrame, dvol: pd.DataFrame) -> pd.DataFrame:
    """For each Heston month (date = 1st), compute mean DVOL over the
    calendar month starting on that date."""
    out = []
    for _, row in heston.iterrows():
        m_start = row["date"]
        m_end = (m_start + pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        mask = (dvol["date"] >= m_start) & (dvol["date"] <= m_end)
        sub = dvol.loc[mask]
        if len(sub) == 0:
            mean_dvol = np.nan
            n_days = 0
        else:
            mean_dvol = float(sub["close"].mean())
            n_days = int(len(sub))
        out.append({
            "month": m_start.strftime("%Y-%m"),
            "month_start": m_start,
            "month_end": m_end,
            "rho": float(row["rho"]),
            "mean_dvol": mean_dvol,
            "dvol_n_days": n_days,
        })
    return pd.DataFrame(out)


def run_rho_vs_dvol(heston: pd.DataFrame, dvol: pd.DataFrame) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("EXPERIMENT 1 — ρ (Heston) vs DVOL regime")
    print("=" * 78)

    paired = pair_rho_with_monthly_dvol(heston, dvol)
    # Flag any month lacking DVOL coverage (do NOT interpolate)
    missing = paired[paired["mean_dvol"].isna()]
    if len(missing) > 0:
        print(f"[WARN] {len(missing)} Heston months have ZERO DVOL rows:")
        for _, r in missing.iterrows():
            print(f"        {r['month']} — no DVOL data, skipping")

    paired_ok = paired.dropna(subset=["mean_dvol"]).reset_index(drop=True)
    print(f"\nUsable pairs: {len(paired_ok)} / {len(paired)}")

    # Print table
    print(f"\n{'month':<10} {'rho':>9} {'sign':>5} {'mean_DVOL':>10} {'n_days':>7}")
    print("-" * 50)
    for _, r in paired_ok.iterrows():
        sign = "neg" if r["rho"] < 0 else "pos"
        print(f"{r['month']:<10} {r['rho']:>9.4f} {sign:>5} "
              f"{r['mean_dvol']:>10.3f} {r['dvol_n_days']:>7d}")

    # Correlation (Pearson)
    rhos = paired_ok["rho"].values
    dvols = paired_ok["mean_dvol"].values
    corr_p = float(np.corrcoef(rhos, dvols)[0, 1])

    # Spearman (rank correlation) — robust to bimodal ρ
    from scipy.stats import spearmanr, pointbiserialr
    corr_s, pval_s = spearmanr(rhos, dvols)

    # High-DVOL → ρ<0 hypothesis check via point-biserial:
    # Binary: ρ<0 (1) vs ρ>0 (0) against continuous DVOL
    rho_neg = (rhos < 0).astype(int)
    n_neg = int(rho_neg.sum())
    n_pos = int(len(rho_neg) - n_neg)
    if n_neg >= 2 and n_pos >= 2:
        pb_corr, pb_pval = pointbiserialr(rho_neg, dvols)
        dvol_when_neg = float(dvols[rho_neg == 1].mean())
        dvol_when_pos = float(dvols[rho_neg == 0].mean())
    else:
        pb_corr, pb_pval = np.nan, np.nan
        dvol_when_neg = float(dvols[rho_neg == 1].mean()) if n_neg > 0 else np.nan
        dvol_when_pos = float(dvols[rho_neg == 0].mean()) if n_pos > 0 else np.nan

    print()
    print(f"Pearson r(ρ, meanDVOL)  = {corr_p:+.4f}")
    print(f"Spearman r(ρ, meanDVOL) = {corr_s:+.4f}  (p={pval_s:.4f})")
    print(f"Mean DVOL when ρ<0 ({n_neg} months) = {dvol_when_neg:.3f}")
    print(f"Mean DVOL when ρ>0 ({n_pos} months) = {dvol_when_pos:.3f}")
    if not np.isnan(pb_corr):
        print(f"Point-biserial r (ρ<0 vs DVOL) = {pb_corr:+.4f}  (p={pb_pval:.4f})")

    high_dvol_implies_neg_rho = (
        dvol_when_neg > dvol_when_pos
        and corr_p < 0
    )
    print(f"\nHigh-DVOL ⇒ ρ<0 hypothesis: "
          f"{'SUPPORTED' if high_dvol_implies_neg_rho else 'NOT SUPPORTED'}")

    # ── Scatter plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if r < 0 else "#2ca02c" for r in rhos]
    ax.scatter(dvols, rhos, c=colors, s=110, edgecolors="black", linewidths=0.6, zorder=3)
    for _, r in paired_ok.iterrows():
        ax.annotate(r["month"], (r["mean_dvol"], r["rho"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlabel("Mean DVOL during month (Deribit BTC implied vol index, %)")
    ax.set_ylabel("Heston ρ (monthly Q-measure calibration)")
    ax.set_title(f"Monthly ρ vs mean DVOL (BTC, {len(paired_ok)} months)\n"
                 f"Pearson r = {corr_p:+.3f}  |  "
                 f"Spearman = {corr_s:+.3f}  |  "
                 f"ρ<0 mean DVOL = {dvol_when_neg:.2f} vs ρ>0 = {dvol_when_pos:.2f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = PLOTS / "dvol_vs_rho_scatter.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\nSaved scatter → {out_png}")

    return {
        "n_months_total": int(len(paired)),
        "n_months_usable": int(len(paired_ok)),
        "n_months_missing_dvol": int(len(missing)),
        "pearson_r": corr_p,
        "spearman_r": float(corr_s),
        "spearman_p": float(pval_s),
        "point_biserial_r": None if np.isnan(pb_corr) else float(pb_corr),
        "point_biserial_p": None if np.isnan(pb_pval) else float(pb_pval),
        "mean_dvol_when_rho_negative": dvol_when_neg,
        "mean_dvol_when_rho_positive": dvol_when_pos,
        "n_rho_negative": n_neg,
        "n_rho_positive": n_pos,
        "high_dvol_implies_neg_rho": bool(high_dvol_implies_neg_rho),
        "monthly_table": paired_ok.assign(
            month_start=paired_ok["month_start"].dt.strftime("%Y-%m-%d"),
            month_end=paired_ok["month_end"].dt.strftime("%Y-%m-%d"),
        ).to_dict(orient="records"),
    }


# ────────────────────────────────────────────────────────────────────────
# Experiment 2 — Bivariate HMM baseline vs +DVOL
# ────────────────────────────────────────────────────────────────────────

def build_returns_df(start: str, end: str) -> pd.DataFrame:
    trades = load_trades("data/", start=start, end=end)
    ohlc = compute_ohlc(trades, freq="5min").copy()
    ohlc["log_return"] = np.log(ohlc["close"] / ohlc["close"].shift(1))
    ohlc = ohlc.dropna(subset=["log_return"])
    return ohlc[["timestamp", "log_return"]].reset_index(drop=True)


def align_dvol_to_bars(returns_df: pd.DataFrame, dvol: pd.DataFrame) -> np.ndarray:
    """Return (T, 2) array: col0 = log_return, col1 = DVOL (daily ffill → 5-min).

    DVOL values are in % (e.g. 45.0 = 45% annualised vol). We keep in %
    rather than /100 so the HMM feature is on a similar order-of-magnitude
    to typical scaled log_returns after hmmlearn's internal standardisation.
    """
    ts = pd.to_datetime(returns_df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    dv = dvol[["date", "close"]].copy()
    dv["date"] = pd.to_datetime(dv["date"]).dt.tz_localize("UTC")
    dv = dv.set_index("date")["close"]

    # Merge: reindex onto union, ffill daily → 5min
    aligned = dv.reindex(dv.index.union(ts)).sort_index().ffill().reindex(ts).values

    feat = np.column_stack([returns_df["log_return"].values, aligned])
    # Drop any rows with NaN (beginning-of-window DVOL missing)
    mask = np.isfinite(feat).all(axis=1)
    return feat[mask]


def fit_gaussian_hmm_metrics(X: np.ndarray, n_states: int = 2, n_init: int = 5,
                              seed: int = 42) -> dict[str, Any]:
    """Fit GaussianHMM with multi-restart; return LL / AIC / BIC / params.

    X: shape (T,) or (T, d). Fits full covariance if d>=2, diag otherwise.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T, d = X.shape
    cov_type = "full" if d > 1 else "diag"

    best_ll = -np.inf
    best_model = None
    rng = np.random.default_rng(seed)
    for i in range(n_init):
        try:
            m = GaussianHMM(
                n_components=n_states,
                covariance_type=cov_type,
                n_iter=200,
                tol=1e-4,
                random_state=int(rng.integers(0, 10**6)),
            )
            m.fit(X)
            ll = m.score(X)
            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_model = m
        except Exception as e:
            print(f"    restart {i} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All HMM restarts failed.")

    # Parameter count k for AIC/BIC
    # - startprob: n-1 free
    # - transmat:  n*(n-1) free
    # - means:     n*d
    # - covars:
    #     full:  n * d*(d+1)/2
    #     diag:  n * d
    if cov_type == "full":
        cov_params = n_states * d * (d + 1) // 2
    else:
        cov_params = n_states * d
    k = (n_states - 1) + n_states * (n_states - 1) + n_states * d + cov_params

    aic = 2 * k - 2 * best_ll
    bic = np.log(T) * k - 2 * best_ll

    # Canonicalise state order by log_return variance (risk_on = lower col0 var)
    # col0 is log_return in both baseline and bivariate
    cv = best_model.covars_
    if cv.ndim == 3:
        col0_vars = cv[:, 0, 0]
    elif cv.ndim == 2:
        col0_vars = cv[:, 0]
    else:
        col0_vars = np.asarray(cv).ravel()
    col0_vars = np.asarray(col0_vars).ravel()
    order = np.asarray(np.argsort(col0_vars)).ravel()  # ascending → risk_on first

    means = best_model.means_[order].tolist()
    if cov_type == "full":
        covars = best_model.covars_[order].tolist()
    else:
        covars = best_model.covars_[order].tolist()
    transmat = best_model.transmat_[np.ix_(order, order)].tolist()
    startprob = best_model.startprob_[order].tolist()

    # Stationary distribution
    try:
        eigvals, eigvecs = np.linalg.eig(best_model.transmat_.T)
        stat = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))])
        stat = stat / stat.sum()
        stat = stat[order].tolist()
    except Exception:
        stat = None

    return {
        "n_states": n_states,
        "n_features": d,
        "cov_type": cov_type,
        "T": int(T),
        "k_params": int(k),
        "log_likelihood": float(best_ll),
        "aic": float(aic),
        "bic": float(bic),
        "means": means,
        "covariances": covars,
        "transition_matrix": transmat,
        "startprob": startprob,
        "stationary": stat,
        "n_init": int(n_init),
    }


def run_hmm_compare(window_start: str, window_end: str, dvol: pd.DataFrame) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"EXPERIMENT 2 — HMM baseline vs +DVOL ({window_start} → {window_end})")
    print("=" * 78)

    print(f"\n[1/3] Loading Binance aggTrades → 5-min log returns...")
    rdf = build_returns_df(start=window_start, end=window_end)
    print(f"      {len(rdf):,} five-min bars  "
          f"({rdf['timestamp'].min()} → {rdf['timestamp'].max()})")

    print(f"\n[2/3] Aligning DVOL (daily ffill → 5-min grid)...")
    feat_bi = align_dvol_to_bars(rdf, dvol)
    print(f"      bivariate shape: {feat_bi.shape}")
    print(f"      DVOL col: mean={feat_bi[:,1].mean():.3f}  "
          f"std={feat_bi[:,1].std():.3f}  "
          f"range=[{feat_bi[:,1].min():.3f}, {feat_bi[:,1].max():.3f}]")

    # Baseline uses ONLY the rows that survived bivariate alignment,
    # so T is identical → AIC/BIC comparable.
    feat_uni = feat_bi[:, 0:1]

    print(f"\n[3/3] Fitting 2-state HMMs...")
    print("  [a] baseline (log_return only, diag cov)...")
    base = fit_gaussian_hmm_metrics(feat_uni, n_states=2, n_init=5, seed=42)
    print(f"       LL={base['log_likelihood']:.2f}  "
          f"AIC={base['aic']:.2f}  BIC={base['bic']:.2f}  k={base['k_params']}")
    print(f"       means (col0=logret): {[f'{float(np.asarray(m).ravel()[0]):+.2e}' for m in base['means']]}")
    print(f"       var   (col0=logret): {[f'{float(np.asarray(c).ravel()[0]):.2e}' for c in base['covariances']]}")

    print("  [b] bivariate (log_return + DVOL, full cov)...")
    bi = fit_gaussian_hmm_metrics(feat_bi, n_states=2, n_init=5, seed=42)
    print(f"       LL={bi['log_likelihood']:.2f}  "
          f"AIC={bi['aic']:.2f}  BIC={bi['bic']:.2f}  k={bi['k_params']}")
    print(f"       means:  state0={bi['means'][0]}  state1={bi['means'][1]}")
    print(f"       (note: state0 = lower log_return variance = risk_on)")

    # ΔAIC / ΔBIC — lower is better
    d_aic = bi["aic"] - base["aic"]
    d_bic = bi["bic"] - base["bic"]
    d_ll  = bi["log_likelihood"] - base["log_likelihood"]
    print(f"\n  ΔLL   (bi - base) = {d_ll:+.2f}   (positive = +DVOL fits better per-row)")
    print(f"  ΔAIC  (bi - base) = {d_aic:+.2f}  (negative = +DVOL preferred)")
    print(f"  ΔBIC  (bi - base) = {d_bic:+.2f}  (negative = +DVOL preferred)")

    verdict = ("DVOL improves model (lower AIC AND lower BIC)"
               if (d_aic < 0 and d_bic < 0)
               else "DVOL improves AIC only (BIC penalises extra params)"
               if d_aic < 0
               else "DVOL does NOT improve model under AIC/BIC")
    print(f"  Verdict: {verdict}")

    return {
        "window_start": window_start,
        "window_end": window_end,
        "n_bars": int(feat_bi.shape[0]),
        "baseline_log_return_only": base,
        "bivariate_log_return_plus_dvol": bi,
        "delta_log_likelihood": float(d_ll),
        "delta_aic": float(d_aic),
        "delta_bic": float(d_bic),
        "verdict": verdict,
    }


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 78)
    print("HMM with DVOL — Heston-ρ regime test + bivariate HMM comparison")
    print("=" * 78)

    dvol = load_dvol()
    heston = load_heston_btc()
    print(f"\nDVOL coverage:   {dvol['date'].min().date()} → {dvol['date'].max().date()}  "
          f"({len(dvol)} rows, units = %)")
    print(f"Heston months:   {len(heston)} BTC snapshots "
          f"({heston['date'].min().date()} → {heston['date'].max().date()})")

    # ── Experiment 1 ──────────────────────────────────────────────────
    exp1 = run_rho_vs_dvol(heston, dvol)

    # ── Experiment 2 ──────────────────────────────────────────────────
    exp2 = run_hmm_compare(
        window_start="2026-01-01",
        window_end="2026-04-08",
        dvol=dvol,
    )

    # ── Save ──────────────────────────────────────────────────────────
    out = {
        "experiment_id": "hmm_with_dvol",
        "dvol_source": "data/deribit_dvol_btc_daily.json",
        "heston_source": "data/heston_qmeasure_time_series.json",
        "dvol_coverage": {
            "start": str(dvol["date"].min().date()),
            "end": str(dvol["date"].max().date()),
            "n_days": int(len(dvol)),
            "units": "percent (annualised implied vol)",
        },
        "experiment_1_rho_vs_dvol": exp1,
        "experiment_2_hmm_baseline_vs_dvol": exp2,
    }
    out_path = DATA / "hmm_dvol_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n\nSaved results → {out_path}")
    print(f"Saved scatter → {PLOTS / 'dvol_vs_rho_scatter.png'}")


if __name__ == "__main__":
    main()
