import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from extensions.heston import HestonParams
from shared.params import ACParams
from montecarlo.sde_engine import simulate_execution, simulate_heston_execution
from montecarlo.strategies import twap_trajectory

# --- Paths ---
PARAMS_PATH = Path(__file__).resolve().parent.parent / "data" / "heston_pmeasure_vs_qmeasure.json"
FIGURE_PATH = Path(__file__).resolve().parent.parent / "figures" / "tail_qq_heston_vs_const.png"


def main():
    # Load Q-measure Heston params
    with open(PARAMS_PATH, "r") as f:
        params_data = json.load(f)

    q_params = params_data["q_measure"]
    heston_params = HestonParams(
        kappa=q_params["kappa"],
        theta=q_params["theta"],
        xi=q_params["xi"],
        rho=q_params["rho"],
        v0=q_params["v0"],
    )

    # AC params
    theta_Q = q_params["theta"]
    sigma_const = np.sqrt(theta_Q)  # constant vol = sqrt(long-run Q-measure variance

    ac_params = ACParams(
        S0=69000.0,
        sigma=sigma_const,
        mu=0.0,
        X0=10.0,
        T=1.0 / 24.0,
        N=50,
        gamma=1.48,
        eta=1.58e-4,
        alpha=1.0,
        lam=1e-6,
        fee_bps=0,
    )

    n_paths = 50000
    seed = 42

    # TWAP trajectory (same for both)
    trajectory = twap_trajectory(ac_params)

    print(f"Running {n_paths} const-vol simulations...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cost_const = simulate_execution(
            ac_params, trajectory,
            n_paths=n_paths, seed=seed,
            antithetic=False, scheme="exact",
        )

    print(f"Running {n_paths} Heston-Q simulations...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, cost_heston = simulate_heston_execution(
            ac_params, heston_params, trajectory,
            n_paths=n_paths, seed=seed,
        )

    # Sort for QQ plot
    cost_const_sorted = np.sort(cost_const)
    cost_heston_sorted = np.sort(cost_heston)

    # CVaR 95 (expected shortfall beyond 95th percentile)
    q95_const = np.percentile(cost_const, 95)
    q95_heston = np.percentile(cost_heston, 95)
    cvar95_const = np.mean(cost_const[cost_const >= q95_const])
    cvar95_heston = np.mean(cost_heston[cost_heston >= q95_heston])

    # Percentiles
    p90_const = np.percentile(cost_const, 90)
    p95_const_val = q95_const
    p99_const = np.percentile(cost_const, 99)
    p90_heston = np.percentile(cost_heston, 90)
    p95_heston_val = q95_heston
    p99_heston = np.percentile(cost_heston, 99)

    print(f"Const-vol CVaR_95: {cvar95_const:.2f}")
    print(f"Heston-Q CVaR_95: {cvar95_heston:.2f}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left subplot: QQ plot
    n_pts = min(len(cost_const_sorted), len(cost_heston_sorted))
    # Subsample for plotting speed if needed
    step = max(1, n_pts // 5000)
    qq_x = cost_const_sorted[::step]
    qq_y = cost_heston_sorted[::step]

    ax1.scatter(qq_x, qq_y, s=1, alpha=0.3, color="steelblue", rasterized=True)

    # 45-degree reference line
    all_vals = np.concatenate([qq_x, qq_y])
    lo, hi = np.percentile(all_vals, [1, 99])
    ax1.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="45° reference")

    ax1.set_xlabel("Cost — Constant Vol (BPS)")
    ax1.set_ylabel("Cost — Heston-Q (BPS)")
    ax1.set_title("QQ Plot: Heston-Q vs Constant-Vol Costs")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Annotate CVaR on QQ plot
    ax1.annotate(
        f"CVaR₉₅ const: {cvar95_const:.1f}\nCVaR₉₅ Heston: {cvar95_heston:.1f}",
        xy=(0.60, 0.15), xycoords="axes fraction",
        fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # Right subplot: Upper-tail focus
    # Focus on the tail region: above the 85th percentile of the wider distribution
    p85_combined = min(np.percentile(cost_const, 85), np.percentile(cost_heston, 85))
    tail_max = max(np.percentile(cost_const, 99.9), np.percentile(cost_heston, 99.9))

    bins = np.linspace(p85_combined, tail_max, 80)

    ax2.hist(cost_const, bins=bins, alpha=0.5, density=True, color="steelblue",
             label="Constant Vol")
    ax2.hist(cost_heston, bins=bins, alpha=0.5, density=True, color="firebrick",
             label="Heston-Q")

    # KDE for tails
    tail_const = cost_const[cost_const >= p85_combined]
    tail_heston = cost_heston[cost_heston >= p85_combined]

    if len(tail_const) > 10:
        try:
            kde_const = gaussian_kde(tail_const, bw_method="scott")
            x_grid = np.linspace(p85_combined, tail_max, 300)
            ax2.plot(x_grid, kde_const(x_grid), color="steelblue", linewidth=1.5, linestyle="-")
        except Exception:
            pass

    if len(tail_heston) > 10:
        try:
            kde_heston = gaussian_kde(tail_heston, bw_method="scott")
            x_grid = np.linspace(p85_combined, tail_max, 300)
            ax2.plot(x_grid, kde_heston(x_grid), color="firebrick", linewidth=1.5, linestyle="-")
        except Exception:
            pass

    # Percentile reference lines
    for pval, col, ls in [(p90_const, "steelblue", ":"), (p95_const_val, "steelblue", "--"),
                           (p99_const, "steelblue", "-."),
                           (p90_heston, "firebrick", ":"), (p95_heston_val, "firebrick", "--"),
                           (p99_heston, "firebrick", "-.")]:
        ax2.axvline(pval, color=col, linestyle=ls, linewidth=1.0, alpha=0.7)

    ax2.set_xlabel("Execution Cost (BPS)")
    ax2.set_ylabel("Density")
    ax2.set_title("Upper Tail Distribution (≥ 85th pctile)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Annotation box for percentiles + CVaR
    annot_text = (
        f"Percentiles (const / Heston-Q):\n"
        f"  90th: {p90_const:.1f} / {p90_heston:.1f}\n"
        f"  95th: {p95_const_val:.1f} / {p95_heston_val:.1f}\n"
        f"  99th: {p99_const:.1f} / {p99_heston:.1f}\n"
        f"CVaR₉₅: {cvar95_const:.1f} / {cvar95_heston:.1f}"
    )
    ax2.annotate(
        annot_text,
        xy=(0.40, 0.60), xycoords="axes fraction",
        fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    fig.suptitle(
        f"Execution Cost Tail: Heston-Q vs Constant-Vol (n={n_paths:,})\n"
        f"σ_const={sigma_const:.4f}  κ={heston_params.kappa:.2f}  "
        f"θ={heston_params.theta:.4f}  ξ={heston_params.xi:.2f}  "
        f"ρ={heston_params.rho:.3f}",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()