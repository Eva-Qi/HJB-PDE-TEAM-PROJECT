"""Generate the HMM regime-detection figure for slide use.

Fits a 2-state Gaussian HMM on cached 5-minute Binance BTC log returns,
runs Viterbi to decode the most-likely state sequence, and plots:

    Top    — |log return| time-series colored by Viterbi state
             (visualizes risk-off "bursts" against the risk-on baseline)
    Bottom — log-return histogram split by state
             (visualizes σ_risk-off ≈ 2.36 × σ_risk-on, the slide claim)

Output: figures/hmm_regime_overlay.png

Numbers are taken verbatim from data/hmm_2state_vs_3state.json so the
figure annotations and the slide bullet points stay in sync.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETURNS_NPY = PROJECT_ROOT / "data" / "btc_5min_log_returns_2026-01-01_to_2026-04-08.npy"
META_JSON = PROJECT_ROOT / "data" / "hmm_2state_vs_3state.json"
OUT_PATH = PROJECT_ROOT / "figures" / "hmm_regime_overlay.png"

SEED = 42
N_INIT = 15  # match the canonical fit


def main() -> None:
    returns = np.load(RETURNS_NPY)
    meta = json.load(open(META_JSON))
    canonical = meta["two_state"]["regimes"]
    risk_on_meta = next(r for r in canonical if r["label"] == "risk_on")
    risk_off_meta = next(r for r in canonical if r["label"] == "risk_off")

    print(f"Loaded {len(returns):,} 5-min returns; canonical 2-state JSON loaded")
    print(f"  Canonical risk-on prob = {risk_on_meta['probability']:.4f}, "
          f"σ_mult = {risk_on_meta['sigma_multiplier']:.3f}")
    print(f"  Canonical risk-off prob = {risk_off_meta['probability']:.4f}, "
          f"σ_mult = {risk_off_meta['sigma_multiplier']:.3f}")

    # ─── Re-fit canonical 2-state HMM (best-of-N_INIT inits) ─────────
    X = returns.reshape(-1, 1)
    best_ll = -np.inf
    best_model = None
    for i in range(N_INIT):
        m = GaussianHMM(n_components=2, covariance_type="full",
                        n_iter=200, random_state=SEED + i)
        try:
            m.fit(X)
            if m.score(X) > best_ll:
                best_ll = m.score(X)
                best_model = m
        except Exception:
            continue
    assert best_model is not None, "All 15 inits failed"

    states = best_model.predict(X)
    state_vols = np.array([np.sqrt(best_model.covars_[k][0, 0]) for k in range(2)])
    # Label so risk-on = lower vol, risk-off = higher vol
    risk_on_idx = int(np.argmin(state_vols))
    risk_off_idx = 1 - risk_on_idx
    is_risk_off = (states == risk_off_idx)
    risk_off_frac = is_risk_off.mean()
    sigma_ratio = state_vols[risk_off_idx] / state_vols[risk_on_idx]

    print(f"  Refit risk-off prob = {risk_off_frac:.4f}, σ-ratio = {sigma_ratio:.2f}×")

    # JSON-canonical multipliers (used in slide annotations to stay in sync
    # with whatever the teammate cites in his Regime-Aware slide)
    on_mult = risk_on_meta["sigma_multiplier"]
    off_mult = risk_off_meta["sigma_multiplier"]

    # ─── Plot ────────────────────────────────────────────────────────
    abs_ret_pct = np.abs(returns) * 100  # in % for readability
    n = len(returns)
    t_idx = np.arange(n)
    days = n / 288  # 288 5-min bars per day

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[3, 1.2],
                          hspace=0.32, wspace=0.18)

    # ── Top: |return| time-series colored by state ──
    ax_top = fig.add_subplot(gs[0, :])
    on_mask = ~is_risk_off
    ax_top.scatter(t_idx[on_mask], abs_ret_pct[on_mask], s=1.4,
                   c="#2E5BFF", alpha=0.50, label="Risk-on", rasterized=True)
    ax_top.scatter(t_idx[is_risk_off], abs_ret_pct[is_risk_off], s=2.0,
                   c="#E63946", alpha=0.75, label="Risk-off", rasterized=True)
    ax_top.set_xlabel(f"5-minute bar index (≈ {days:.0f} days, {n:,} bars)",
                      fontsize=10)
    ax_top.set_ylabel("|log return| (%)", fontsize=10)
    ax_top.set_title(
        f"Viterbi-decoded HMM regimes on Binance BTCUSDT 5-min returns  "
        f"(risk-off bars cluster into volatility bursts; "
        rf"risk-off vol $\approx$ {off_mult:.2f}$\times$ baseline)",
        fontsize=11, pad=8,
    )
    ax_top.set_ylim(bottom=0)
    ax_top.legend(loc="upper right", fontsize=9, markerscale=4)
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # ── Bottom-left: split histogram (log-y) ──
    ax_hist = fig.add_subplot(gs[1, 0])
    ret_pct = returns * 100
    bins = np.linspace(-3.0, 3.0, 121)
    ax_hist.hist(ret_pct[on_mask], bins=bins, density=True,
                 color="#2E5BFF", alpha=0.55,
                 label=f"Risk-on  (σ={state_vols[risk_on_idx]*100:.3f}%, "
                       f"{(~is_risk_off).mean():.1%})")
    ax_hist.hist(ret_pct[is_risk_off], bins=bins, density=True,
                 color="#E63946", alpha=0.55,
                 label=f"Risk-off (σ={state_vols[risk_off_idx]*100:.3f}%, "
                       f"{is_risk_off.mean():.1%})")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("5-min log return (%)", fontsize=10)
    ax_hist.set_ylabel("density (log scale)", fontsize=10)
    ax_hist.set_title("Per-regime return distribution", fontsize=10, pad=6)
    ax_hist.legend(loc="upper right", fontsize=8.5)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.35)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    # ── Bottom-right: σ-multiplier callout box ──
    ax_box = fig.add_subplot(gs[1, 1])
    ax_box.axis("off")
    summary = (
        r"$\bf{Headline\ numbers}$"
        "\n"
        f"Risk-on occupancy:  {risk_on_meta['probability']:.1%}\n"
        f"Risk-off occupancy: {risk_off_meta['probability']:.1%}\n"
        "\n"
        rf"$\sigma$ multipliers (vs baseline)"
        "\n"
        rf"  risk-on:  {on_mult:.2f}$\times$"
        "\n"
        rf"  risk-off: {off_mult:.2f}$\times$"
        f"  ← slide cite\n"
        "\n"
        rf"$\bf{{Model\ selection\ (BIC)}}$"
        "\n"
        f"2-state BIC:  {meta['two_state']['bic']:.0f}\n"
        f"3-state BIC:  {meta['three_state']['bic']:.0f}\n"
        rf"$\Delta$BIC$_{{3-2}}$ = +{meta['delta_bic_3_minus_2']:.1f}"
        "\n  (2-state preferred)\n"
        "\n"
        rf"$\bf{{Code\ note}}$"
        "\n"
        "BIC fix 2026-04-26 reversed\n"
        "earlier 3-state preference;\n"
        "see research/HMM.md"
    )
    ax_box.text(
        0.02, 0.98, summary, transform=ax_box.transAxes,
        fontsize=8.5, va="top", ha="left", family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="#888", alpha=0.95),
    )

    fig.suptitle(
        "Hidden Markov regime detection — 2-state GaussianHMM, Viterbi-decoded",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
