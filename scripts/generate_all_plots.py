"""Generate all comparison plots for the MF796 project.

Produces six publication-quality figures:
  1. Alpha estimation comparison (3 methods)
  2. Optimal vs TWAP trajectory + cost distribution
  3. Heston stochastic vol vs constant vol
  4. HMM regime detection on real BTC data
  5. PDE vs MC cross-validation
  6. SDE discretization scheme comparison

Run from project root:
    python scripts/generate_all_plots.py
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── sys.path hack ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.params import ACParams, DEFAULT_PARAMS, almgren_chriss_closed_form
from shared.cost_model import execution_cost
from montecarlo.sde_engine import (
    simulate_execution,
    simulate_heston_execution,
    simulate_pde_optimal_execution,
)
from montecarlo.strategies import twap_trajectory, optimal_trajectory
from montecarlo.cost_analysis import compute_metrics
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory
from extensions.heston import HestonParams
from extensions.regime import fit_hmm
from calibration.data_loader import load_trades, compute_ohlc, compute_mid_prices

# ── Style setup ────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # fall back to default

# Consistent color palette
C_BLUE = "#2171B5"
C_ORANGE = "#D94801"
C_GREEN = "#238B45"
C_PURPLE = "#6A51A3"
C_RED = "#CB181D"
C_GRAY = "#636363"

SAVE_DIR = PROJECT_ROOT  # plots saved to project root
N_PATHS = 10_000
SEED = 42
DPI = 150


# ===================================================================
# Plot 1: Alpha Estimation Comparison
# ===================================================================
def plot_alpha_comparison():
    print("[1/6] Generating plot_alpha_comparison.png ...")

    # Load result files
    with open(PROJECT_ROOT / "data" / "alpha_estimation_results.json") as f:
        agg_results = json.load(f)
    with open(PROJECT_ROOT / "data" / "orderbook_alpha_results.json") as f:
        ob_results = json.load(f)

    # Three methods
    methods = [
        "Aggregated Flow\n(1-min)",
        "Order Book\nDepth",
        "Literature\nBenchmark",
    ]
    alphas = [
        agg_results["results"]["1min"]["alpha"],
        ob_results["fit"]["alpha"],
        0.5,
    ]
    errors = [
        agg_results["results"]["1min"]["se"],
        ob_results["fit"]["se_alpha"],
        0.05,  # literature range ~0.45-0.55
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    x_pos = np.arange(len(methods))
    colors = [C_BLUE, C_ORANGE, C_GREEN]
    bars = ax.bar(
        x_pos, alphas, yerr=errors,
        color=colors, alpha=0.85, width=0.55,
        edgecolor="white", linewidth=1.5,
        capsize=8, error_kw={"linewidth": 2, "capthick": 2},
    )

    # Add value labels on bars
    for bar, alpha_val, err in zip(bars, alphas, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.01,
            f"{alpha_val:.3f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(r"Impact Exponent $\alpha$", fontsize=12)
    ax.set_title("Impact Exponent Estimation: Three Methods", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 0.7)
    ax.axhline(y=0.5, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.5, label=r"$\alpha=0.5$ (square-root law)")
    ax.legend(fontsize=10, loc="upper left")

    # Annotation
    r2_agg = agg_results["results"]["1min"]["r2"]
    r2_ob = ob_results["fit"]["r_squared"]
    ax.text(
        0.98, 0.02,
        f"Agg. Flow: R$^2$={r2_agg:.3f}, n={agg_results['results']['1min']['n_obs']:,}\n"
        f"Order Book: R$^2$={r2_ob:.3f}, n={ob_results['n_snapshots']:,}\n"
        f"Data: {agg_results['n_days']} days, {agg_results['n_trades']:,} trades",
        transform=ax.transAxes, fontsize=9,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(SAVE_DIR / "plot_alpha_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("    -> Saved plot_alpha_comparison.png")


# ===================================================================
# Plot 2: TWAP vs Optimal Trajectory + Cost Distribution
# ===================================================================
def plot_strategy_comparison():
    print("[2/6] Generating plot_strategy_comparison.png ...")

    params = replace(DEFAULT_PARAMS, alpha=1.0)  # ensure alpha=1 for closed-form
    x_twap = twap_trajectory(params)
    x_opt = optimal_trajectory(params)
    t_grid = np.linspace(0, params.T, params.N + 1)

    # Closed-form expected cost
    _, _, cf_cost = almgren_chriss_closed_form(params)

    # MC simulation
    _, costs_twap = simulate_execution(
        params, x_twap, n_paths=N_PATHS, seed=SEED, antithetic=True, scheme="exact",
    )
    _, costs_opt = simulate_execution(
        params, x_opt, n_paths=N_PATHS, seed=SEED, antithetic=True, scheme="exact",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Inventory trajectories
    ax1.plot(t_grid * 252, x_twap, linewidth=2.5, color=C_BLUE, label="TWAP")
    ax1.plot(t_grid * 252, x_opt, linewidth=2.5, color=C_ORANGE, label="Optimal (A&C)")
    ax1.fill_between(t_grid * 252, x_twap, x_opt, alpha=0.1, color=C_ORANGE)
    ax1.set_xlabel("Trading Days", fontsize=12)
    ax1.set_ylabel("Remaining Inventory (shares)", fontsize=12)
    ax1.set_title("Execution Trajectories", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    # Annotation for kappa
    kappa_T = params.kappa * params.T
    ax1.text(
        0.02, 0.02,
        f"$\\kappa T$ = {kappa_T:.2f}\nFront-loading ratio",
        transform=ax1.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Right: Cost distributions
    bins = np.linspace(
        min(costs_twap.min(), costs_opt.min()),
        max(costs_twap.max(), costs_opt.max()),
        80,
    )
    ax2.hist(costs_twap, bins=bins, alpha=0.55, color=C_BLUE, label="TWAP", density=True)
    ax2.hist(costs_opt, bins=bins, alpha=0.55, color=C_ORANGE, label="Optimal", density=True)

    # Mean lines
    mean_twap = np.mean(costs_twap)
    mean_opt = np.mean(costs_opt)
    ax2.axvline(mean_twap, color=C_BLUE, linestyle="--", linewidth=2,
                label=f"TWAP mean: {mean_twap:,.0f}")
    ax2.axvline(mean_opt, color=C_ORANGE, linestyle="--", linewidth=2,
                label=f"Optimal mean: {mean_opt:,.0f}")
    ax2.axvline(cf_cost, color=C_RED, linestyle=":", linewidth=2,
                label=f"Closed-form: {cf_cost:,.0f}")

    savings_pct = (mean_twap - mean_opt) / mean_twap * 100
    ax2.set_xlabel("Execution Cost ($)", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Cost Distribution (10K MC Paths)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax2.text(
        0.02, 0.95,
        f"Cost savings: {savings_pct:.1f}%",
        transform=ax2.transAxes, fontsize=11, fontweight="bold",
        va="top", color=C_GREEN,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.suptitle("Optimal vs TWAP: Trajectory and Cost Distribution",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(SAVE_DIR / "plot_strategy_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("    -> Saved plot_strategy_comparison.png")


# ===================================================================
# Plot 3: Heston Stochastic Vol vs Constant Vol
# ===================================================================
def plot_heston_vs_constant():
    print("[3/6] Generating plot_heston_vs_constant.png ...")

    params = replace(DEFAULT_PARAMS, alpha=1.0)
    heston = HestonParams(kappa=2.0, theta=0.09, xi=0.5, rho=-0.3, v0=0.09)

    x_twap = twap_trajectory(params)

    # Constant vol MC
    _, costs_const = simulate_execution(
        params, x_twap, n_paths=N_PATHS, seed=SEED, antithetic=True, scheme="exact",
    )

    # Heston MC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress Feller condition warning
        _, _, costs_heston = simulate_heston_execution(
            params, heston, x_twap, n_paths=N_PATHS, seed=SEED,
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(
        min(costs_const.min(), costs_heston.min()),
        max(costs_const.max(), costs_heston.max()),
        80,
    )

    ax.hist(costs_const, bins=bins, alpha=0.55, color=C_BLUE,
            label="Constant Vol (GBM)", density=True)
    ax.hist(costs_heston, bins=bins, alpha=0.55, color=C_ORANGE,
            label="Stochastic Vol (Heston)", density=True)

    # Mean lines
    m_const = np.mean(costs_const)
    m_heston = np.mean(costs_heston)
    ax.axvline(m_const, color=C_BLUE, linestyle="--", linewidth=2,
               label=f"GBM mean: {m_const:,.0f}")
    ax.axvline(m_heston, color=C_ORANGE, linestyle="--", linewidth=2,
               label=f"Heston mean: {m_heston:,.0f}")

    # VaR95 lines
    var95_const = np.percentile(costs_const, 95)
    var95_heston = np.percentile(costs_heston, 95)
    ax.axvline(var95_const, color=C_BLUE, linestyle=":", linewidth=1.5,
               label=f"GBM VaR95: {var95_const:,.0f}")
    ax.axvline(var95_heston, color=C_ORANGE, linestyle=":", linewidth=1.5,
               label=f"Heston VaR95: {var95_heston:,.0f}")

    ax.set_xlabel("Execution Cost ($)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Stochastic Vol Impact on Execution Cost", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Heston params annotation
    ax.text(
        0.02, 0.95,
        f"Heston: $\\kappa$={heston.kappa}, $\\theta$={heston.theta}, "
        f"$\\xi$={heston.xi}, $\\rho$={heston.rho}, $v_0$={heston.v0}\n"
        f"Std ratio: {np.std(costs_heston)/np.std(costs_const):.2f}x",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(SAVE_DIR / "plot_heston_vs_constant.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("    -> Saved plot_heston_vs_constant.png")


# ===================================================================
# Plot 4: HMM Regime Detection on Real BTC Data
# ===================================================================
def plot_regime_detection():
    print("[4/6] Generating plot_regime_detection.png ...")

    data_dir = PROJECT_ROOT / "data"

    # Load a subset of real data (2 weeks for readability)
    trades = load_trades(data_dir, start="2026-03-01", end="2026-03-14")
    ohlc = compute_ohlc(trades, freq="5min")
    mid_df = compute_mid_prices(trades, freq="5min")

    prices = mid_df["mid_price"].values
    log_returns = np.log(prices[1:] / prices[:-1])

    # Fit HMM
    regimes, states = fit_hmm(log_returns, n_regimes=2)
    # states has length = len(log_returns) = len(prices) - 1
    # Align: states[i] corresponds to the return from price[i] to price[i+1]
    # We plot prices[1:] colored by state (dropping first price point)
    plot_prices = prices[1:]
    timestamps = mid_df["timestamp"].values[1:]

    # Map regime labels
    label_map = {r.label: r for r in regimes}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1],
                                    sharex=True)

    # Top: Price series colored by regime
    ax1.plot(timestamps, plot_prices, color=C_GRAY, linewidth=0.3, alpha=0.5, zorder=1)

    # Overlay colored segments
    risk_on_mask = states == 0  # risk_on after sorting
    risk_off_mask = states == 1

    ax1.scatter(timestamps[risk_on_mask], plot_prices[risk_on_mask],
                c=C_GREEN, s=1, alpha=0.6, label="Risk-On", zorder=2)
    ax1.scatter(timestamps[risk_off_mask], plot_prices[risk_off_mask],
                c=C_RED, s=1, alpha=0.6, label="Risk-Off", zorder=2)

    ax1.set_ylabel("BTC Mid Price (USDT)", fontsize=12)
    ax1.set_title("HMM Regime Detection on BTCUSDT", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left", markerscale=10)
    ax1.grid(True, alpha=0.3)

    # Regime params annotation
    ro = label_map["risk_on"]
    rf = label_map["risk_off"]
    ax1.text(
        0.98, 0.95,
        f"Risk-On: $\\sigma_{{ann}}$={ro.sigma:.1%}, P={ro.probability:.1%}\n"
        f"Risk-Off: $\\sigma_{{ann}}$={rf.sigma:.1%}, P={rf.probability:.1%}",
        transform=ax1.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    # Bottom: Regime state bar
    colors_arr = np.where(states == 0, C_GREEN, C_RED)
    ax2.bar(timestamps, np.ones_like(states), color=colors_arr, width=0.005, alpha=0.8)
    ax2.set_ylabel("Regime", fontsize=12)
    ax2.set_yticks([])
    ax2.set_xlabel("Date", fontsize=12)

    # Add text labels
    ax2.text(0.02, 0.7, "Green = Risk-On", transform=ax2.transAxes,
             fontsize=10, color=C_GREEN, fontweight="bold")
    ax2.text(0.02, 0.3, "Red = Risk-Off", transform=ax2.transAxes,
             fontsize=10, color=C_RED, fontweight="bold")

    plt.tight_layout()
    fig.savefig(SAVE_DIR / "plot_regime_detection.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("    -> Saved plot_regime_detection.png")


# ===================================================================
# Plot 5: PDE-MC Cross-Validation
# ===================================================================
def plot_pde_mc_crossval():
    print("[5/6] Generating plot_pde_mc_crossval.png ...")

    results = {}

    for alpha_val, label in [(1.0, r"$\alpha=1.0$ (Linear)"), (0.5, r"$\alpha=0.5$ (Square-root)")]:
        params = replace(DEFAULT_PARAMS, alpha=alpha_val)

        # PDE solve
        grid, V, v_star = solve_hjb(params, M=200)
        x_pde = extract_optimal_trajectory(grid, v_star, params)
        pde_cost = execution_cost(x_pde, params)

        # MC on PDE trajectory
        _, mc_costs = simulate_execution(
            params, x_pde, n_paths=N_PATHS, seed=SEED,
            antithetic=True, scheme="exact",
        )
        mc_mean = np.mean(mc_costs)
        mc_std = np.std(mc_costs)
        mc_se = mc_std / np.sqrt(N_PATHS)

        results[alpha_val] = {
            "label": label,
            "pde_cost": pde_cost,
            "mc_mean": mc_mean,
            "mc_se": mc_se,
            "mc_std": mc_std,
        }

    # Closed-form for alpha=1
    params_lin = replace(DEFAULT_PARAMS, alpha=1.0)
    _, _, cf_cost = almgren_chriss_closed_form(params_lin)

    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.array([0, 1, 2.5, 3.5])
    labels_x = [
        "PDE\n(Linear)",
        "MC Mean\n(Linear)",
        "PDE\n(Square-root)",
        "MC Mean\n(Square-root)",
    ]
    values = [
        results[1.0]["pde_cost"],
        results[1.0]["mc_mean"],
        results[0.5]["pde_cost"],
        results[0.5]["mc_mean"],
    ]
    errors_bar = [
        0,
        results[1.0]["mc_se"] * 1.96,
        0,
        results[0.5]["mc_se"] * 1.96,
    ]
    colors = [C_BLUE, C_BLUE, C_ORANGE, C_ORANGE]
    hatches = ["", "///", "", "///"]

    bars = ax.bar(
        x_pos, values, yerr=errors_bar,
        color=colors, alpha=0.7, width=0.7,
        edgecolor="white", linewidth=1.5,
        capsize=8, error_kw={"linewidth": 2, "capthick": 2},
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Closed-form reference line for alpha=1
    ax.axhline(y=cf_cost, color=C_RED, linestyle="--", linewidth=2,
               label=f"Closed-form (A&C): {cf_cost:,.0f}")

    # Value labels
    for xp, val, err in zip(x_pos, values, errors_bar):
        ax.text(xp, val + err + max(values) * 0.01, f"{val:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_x, fontsize=10)
    ax.set_ylabel("Expected Execution Cost ($)", fontsize=12)
    ax.set_title("PDE-MC Cross-Validation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Relative error annotations
    for alpha_val, x_offset in [(1.0, 0.5), (0.5, 3.0)]:
        r = results[alpha_val]
        rel_err = abs(r["mc_mean"] - r["pde_cost"]) / r["pde_cost"] * 100
        ax.text(
            x_offset, max(values) * 0.05,
            f"Rel. error: {rel_err:.2f}%",
            ha="center", fontsize=9, color=C_GRAY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

    plt.tight_layout()
    fig.savefig(SAVE_DIR / "plot_pde_mc_crossval.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("    -> Saved plot_pde_mc_crossval.png")


# ===================================================================
# Plot 6: SDE Discretization Scheme Comparison
# ===================================================================
def plot_scheme_convergence():
    print("[6/6] Generating plot_scheme_convergence.png ...")

    params = replace(DEFAULT_PARAMS, alpha=1.0)
    x_twap = twap_trajectory(params)
    true_cost = execution_cost(x_twap, params)

    schemes = ["exact", "euler", "milstein"]
    scheme_labels = {"exact": "Exact (Log-normal)", "euler": "Euler-Maruyama", "milstein": "Milstein"}
    scheme_colors = {"exact": C_BLUE, "euler": C_ORANGE, "milstein": C_GREEN}

    cost_arrays = {}
    for scheme in schemes:
        _, costs = simulate_execution(
            params, x_twap, n_paths=N_PATHS, seed=SEED,
            antithetic=True, scheme=scheme,
        )
        cost_arrays[scheme] = costs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Overlaid histograms
    all_costs = np.concatenate(list(cost_arrays.values()))
    bins = np.linspace(np.percentile(all_costs, 1), np.percentile(all_costs, 99), 80)

    for scheme in schemes:
        ax1.hist(
            cost_arrays[scheme], bins=bins, alpha=0.45,
            color=scheme_colors[scheme], label=scheme_labels[scheme],
            density=True,
        )

    ax1.axvline(true_cost, color=C_RED, linestyle="--", linewidth=2,
                label=f"Deterministic cost: {true_cost:,.0f}")

    ax1.set_xlabel("Execution Cost ($)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Cost Distribution by Scheme", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Summary statistics comparison
    stat_labels = ["Mean", "Std", "VaR95", "CVaR95"]
    x_stat = np.arange(len(stat_labels))
    width = 0.22

    for i, scheme in enumerate(schemes):
        m = compute_metrics(cost_arrays[scheme])
        vals = [m.mean, m.std, m.var_95, m.cvar_95]
        offset = (i - 1) * width
        bars = ax2.bar(
            x_stat + offset, vals, width,
            color=scheme_colors[scheme], alpha=0.8,
            label=scheme_labels[scheme],
            edgecolor="white", linewidth=0.8,
        )
        # Value labels for mean only
        ax2.text(
            x_stat[0] + offset, vals[0] + max(vals) * 0.01,
            f"{vals[0]:,.0f}", ha="center", va="bottom", fontsize=8,
        )

    # True cost reference on mean group
    ax2.axhline(y=true_cost, color=C_RED, linestyle="--", linewidth=1.5,
                alpha=0.7, label=f"True cost: {true_cost:,.0f}")

    ax2.set_xticks(x_stat)
    ax2.set_xticklabels(stat_labels, fontsize=11)
    ax2.set_ylabel("Cost ($)", fontsize=12)
    ax2.set_title("Scheme Statistics Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("SDE Discretization Scheme Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(SAVE_DIR / "plot_scheme_convergence.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("    -> Saved plot_scheme_convergence.png")


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("MF796 Project — Generating All Comparison Plots")
    print(f"Output directory: {SAVE_DIR}")
    print(f"MC paths: {N_PATHS}, Seed: {SEED}, DPI: {DPI}")
    print("=" * 60)

    plot_alpha_comparison()
    plot_strategy_comparison()
    plot_heston_vs_constant()
    plot_regime_detection()
    plot_pde_mc_crossval()
    plot_scheme_convergence()

    print("\n" + "=" * 60)
    print("All 6 plots generated successfully!")
    print("Files:")
    for name in [
        "plot_alpha_comparison.png",
        "plot_strategy_comparison.png",
        "plot_heston_vs_constant.png",
        "plot_regime_detection.png",
        "plot_pde_mc_crossval.png",
        "plot_scheme_convergence.png",
    ]:
        print(f"  {SAVE_DIR / name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
