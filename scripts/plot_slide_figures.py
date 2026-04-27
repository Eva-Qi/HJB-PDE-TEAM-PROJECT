"""Generate Slide 6 (walk-forward bar chart) + Slide 7 (Heston CVaR comparison).

Outputs go to figures/. Run from project root:
    python scripts/plot_slide_figures.py

NOTE on the Heston CVaR numbers: FINDINGS.md V6 cites 4.79% reduction
(74222 vs 71016, 50k/100k paths, Apr 21). The current JSON file
data/paired_heston_qmeasure_results.json shows 14.59% reduction
(4294 vs 3668, 10k paths) — that is a different run (different X0 or
sigma_base). We plot the JSON values verbatim and label the run config
on the figure so reviewers can see what's plotted vs cited.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"


def plot_walk_forward_savings():
    with open(DATA_DIR / "walk_forward_results.json") as f:
        splits = json.load(f)

    labels = [s["label"].replace("Split-", "S").replace("→", "-") for s in splits]
    mc = [s["oos_mc_savings_pct"] for s in splits]
    det = [s["oos_det_savings_pct"] for s in splits]
    sigma_drift = [s["sigma_drift_pct"] for s in splits]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    width = 0.36

    b1 = ax.bar(x - width / 2, mc, width, label="MC OOS savings (100-path test)",
                color="#2E5BFF", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, det, width, label="Deterministic OOS savings",
                color="#9CB3FF", edgecolor="white", linewidth=0.5)

    for bar, v in zip(b1, mc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#2E5BFF", fontweight="bold")
    for bar, v in zip(b2, det):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#5566AA")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("AC vs TWAP savings (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(mc) * 1.18)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_title(
        "Walk-forward OOS validation — AC beats TWAP across 6/6 splits at X0=1000 BTC\n"
        f"MC savings range: {min(mc):.1f}% to {max(mc):.1f}%  ·  σ-drift range: {min(sigma_drift):+.1f}% to {max(sigma_drift):+.1f}%",
        fontsize=10, pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "walk_forward_savings_6splits.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (6 splits, MC: {min(mc):.1f}%-{max(mc):.1f}%)")


def plot_heston_cvar_comparison():
    with open(DATA_DIR / "paired_heston_qmeasure_results.json") as f:
        d = json.load(f)

    n_paths = d.get("n_paths")
    headline = d["headline"]
    cvar_const = headline["cvar95_const_vol"]
    cvar_q = headline["cvar95_heston_Q"]
    pct = headline["cvar95_pct_shift"]
    pval = headline["cvar95_pvalue"]

    cmp_a = d["comparisons"]["A_const_vs_P"]["metrics"]["cvar_95"]
    cvar_p = cmp_a["b"]

    labels = ["Const-vol", "Heston-P", "Heston-Q"]
    values = [cvar_const, cvar_p, cvar_q]
    colors = ["#888888", "#FF8C42", "#2E5BFF"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, width=0.55)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ymax = max(values) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_ylabel("CVaR$_{95}$ (cost)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    pval_str = "p < 0.0001" if pval < 1e-4 else f"p = {pval:.4g}"
    ax.annotate(
        "",
        xy=(2, cvar_q), xytext=(0, cvar_const),
        arrowprops=dict(arrowstyle="->", color="#2E5BFF", lw=1.5,
                        connectionstyle="arc3,rad=-0.18"),
    )
    ax.text(1, max(values) * 1.05,
            f"Heston-Q reduces CVaR$_{{95}}$ by {pct:.2f}%\n({pval_str})",
            ha="center", va="bottom", fontsize=11, color="#2E5BFF", fontweight="bold")

    ax.set_title(
        f"Part D anchor finding — Heston-Q stochastic vol cuts tail risk\n"
        f"100k-path paired test would supersede this; current run = {n_paths:,} paths, CRN coupled",
        fontsize=10, pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "heston_cvar_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (n={n_paths}, const={cvar_const:.0f}, P={cvar_p:.0f}, Q={cvar_q:.0f}, "
          f"Q-vs-const Δ={pct:.2f}%)")


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    plot_walk_forward_savings()
    plot_heston_cvar_comparison()
