"""Multi-X0 comparison: how order size affects optimal execution strategy.

Runs the calibrated pipeline at three order sizes (X0 = 10, 100, 1000 BTC)
to demonstrate that larger orders benefit more from optimal execution vs TWAP.

Uses alpha=1 (linear impact) for the primary analysis, where closed-form
Almgren-Chriss is available and PDE can cross-validate.  Also runs alpha=0.47
(from order book estimation) via PDE for comparison.

Outputs:
    - plot_multi_x0_trajectories.png  (3-panel trajectory comparison)
    - plot_multi_x0_costs.png         (cost distribution histograms)
    - plot_multi_x0_savings.png       (savings bar chart)
    - Printed comparison table
"""

import sys
from pathlib import Path

# Allow running from scripts/ without package install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.impact_estimator import calibrated_params
from shared.params import ACParams, almgren_chriss_closed_form
from shared.cost_model import execution_cost, execution_risk, objective
from montecarlo.strategies import twap_trajectory, optimal_trajectory
from montecarlo.sde_engine import simulate_execution
from montecarlo.cost_analysis import compute_metrics
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "data")
X0_VALUES = [10, 100, 1000]     # BTC
ALPHA_OB = 0.47                 # from order book data (used for supplementary PDE run)
N_STEPS = 50
T_HOURS = 1.0                   # execution horizon in hours
T_YEARS = T_HOURS / (365.25 * 24)
N_PATHS = 5000
SEED = 42
PDE_M = 200

# kappa*T target for meaningful front-loading
# kappa = sqrt(lam * S0^2 * sigma^2 / eta)
# kappa*T = target => lam = (target/T)^2 * eta / (S0^2 * sigma^2)
KAPPA_T_TARGET = 1.5


def find_lam_for_kappa_T(S0, sigma, eta, T, target_kT=1.5):
    """Compute lambda that gives kappa*T = target (linear impact formula)."""
    kappa_needed = target_kT / T
    lam = kappa_needed**2 * eta / (S0**2 * sigma**2)
    return lam


def run_single_x0(X0, S0, sigma, gamma, eta, alpha, lam, label_prefix=""):
    """Run full pipeline for one (X0, alpha) configuration.

    Returns dict with all results.
    """
    params = ACParams(
        S0=S0, sigma=sigma, mu=0.0,
        X0=float(X0), T=T_YEARS, N=N_STEPS,
        gamma=gamma, eta=eta, alpha=alpha, lam=lam,
    )

    kappa = params.kappa
    kT = kappa * T_YEARS
    print(f"  {label_prefix}lam = {lam:.6e}")
    print(f"  {label_prefix}kappa = {kappa:.4f},  kappa*T = {kT:.4f}")

    # TWAP trajectory
    x_twap = twap_trajectory(params)

    # Optimal trajectory
    is_linear = abs(alpha - 1.0) < 1e-10
    if is_linear:
        # Closed-form (exact)
        x_opt = optimal_trajectory(params)
        opt_label = "Optimal (CF)"
        # Also PDE for cross-validation
        print(f"  {label_prefix}Solving HJB PDE (M={PDE_M}, alpha=1)...")
        grid, V, v_star = solve_hjb(params, M=PDE_M)
        x_opt_pde = extract_optimal_trajectory(grid, v_star, params)
        pde_cf_err = np.max(np.abs(x_opt - x_opt_pde))
        print(f"  {label_prefix}CF vs PDE max trajectory diff: {pde_cf_err:.6e}")
    else:
        # PDE only
        print(f"  {label_prefix}Solving HJB PDE (M={PDE_M}, alpha={alpha:.2f})...")
        grid, V, v_star = solve_hjb(params, M=PDE_M)
        x_opt = extract_optimal_trajectory(grid, v_star, params)
        x_opt_pde = x_opt
        opt_label = "Optimal (PDE)"

    # Deterministic costs
    cost_twap = execution_cost(x_twap, params)
    cost_opt = execution_cost(x_opt, params)
    risk_twap = execution_risk(x_twap, params)
    risk_opt = execution_risk(x_opt, params)
    obj_twap = objective(x_twap, params)
    obj_opt = objective(x_opt, params)

    print(f"  {label_prefix}Det cost:     TWAP={cost_twap:.4f}  Opt={cost_opt:.4f}")
    print(f"  {label_prefix}Risk:         TWAP={risk_twap:.4f}  Opt={risk_opt:.4f}")
    print(f"  {label_prefix}Objective:    TWAP={obj_twap:.4f}  Opt={obj_opt:.4f}")

    # Monte Carlo
    print(f"  {label_prefix}Running MC ({N_PATHS} paths, exact)...")
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((N_PATHS, N_STEPS))

    _, costs_twap_mc = simulate_execution(
        params, x_twap, n_paths=N_PATHS,
        antithetic=False, Z_extern=Z, scheme="exact",
    )
    _, costs_opt_mc = simulate_execution(
        params, x_opt, n_paths=N_PATHS,
        antithetic=False, Z_extern=Z, scheme="exact",
    )

    m_twap = compute_metrics(costs_twap_mc)
    m_opt = compute_metrics(costs_opt_mc)

    savings_obj = (obj_twap - obj_opt) / abs(obj_twap) * 100 if abs(obj_twap) > 0 else 0.0
    savings_det = (cost_twap - cost_opt) / abs(cost_twap) * 100 if abs(cost_twap) > 1e-12 else 0.0
    savings_mc = (m_twap.mean - m_opt.mean) / abs(m_twap.mean) * 100 if abs(m_twap.mean) > 1e-12 else 0.0

    print(f"  {label_prefix}MC mean:  TWAP={m_twap.mean:.4f}  Opt={m_opt.mean:.4f}")
    print(f"  {label_prefix}MC std:   TWAP={m_twap.std:.4f}  Opt={m_opt.std:.4f}")
    print(f"  {label_prefix}MC VaR95: TWAP={m_twap.var_95:.4f}  Opt={m_opt.var_95:.4f}")
    print(f"  {label_prefix}Obj savings: {savings_obj:.2f}%")

    return {
        "params": params,
        "x_twap": x_twap,
        "x_opt": x_opt,
        "opt_label": opt_label,
        "costs_twap": costs_twap_mc,
        "costs_opt": costs_opt_mc,
        "m_twap": m_twap,
        "m_opt": m_opt,
        "kappa_T": kT,
        "savings_obj": savings_obj,
        "savings_det": savings_det,
        "savings_mc": savings_mc,
        "cost_twap": cost_twap,
        "cost_opt": cost_opt,
        "obj_twap": obj_twap,
        "obj_opt": obj_opt,
        "lam": lam,
    }


def main():
    print("\n" + "=" * 75)
    print("  MULTI-X0 COMPARISON: Optimal Execution vs TWAP")
    print(f"  T = {T_HOURS:.0f} hour, N = {N_STEPS}, target kappa*T = {KAPPA_T_TARGET}")
    print("=" * 75)

    # ------------------------------------------------------------------
    # Step 1: Calibrate base parameters from Binance data
    # ------------------------------------------------------------------
    print("\n  Loading Binance data and calibrating base parameters...")
    cal = calibrated_params(
        trades_path=DATA_DIR,
        X0=X0_VALUES[0],
        T=T_YEARS,
        N=N_STEPS,
        lam=1e-6,
    )
    base = cal.params
    S0 = base.S0
    sigma = base.sigma
    gamma = base.gamma
    eta = base.eta

    print(f"  S0 = ${S0:,.2f}")
    print(f"  sigma = {sigma:.4f} (annualized)")
    print(f"  gamma = {gamma:.6e}")
    print(f"  eta = {eta:.6e}")
    if cal.warnings:
        for w in cal.warnings:
            print(f"    Warning: {w}")

    # ------------------------------------------------------------------
    # Step 2: Run pipeline for each X0 with alpha=1.0 (primary)
    # ------------------------------------------------------------------
    results_linear = {}
    alpha_lin = 1.0

    for X0 in X0_VALUES:
        print(f"\n{'='*75}")
        print(f"  X0 = {X0} BTC  |  alpha = {alpha_lin}")
        print(f"{'='*75}")
        lam = find_lam_for_kappa_T(S0, sigma, eta, T_YEARS, KAPPA_T_TARGET)
        results_linear[X0] = run_single_x0(
            X0, S0, sigma, gamma, eta, alpha_lin, lam,
        )

    # ------------------------------------------------------------------
    # Step 3: Run pipeline for each X0 with alpha=0.47 (supplementary)
    # ------------------------------------------------------------------
    results_nonlin = {}

    for X0 in X0_VALUES:
        print(f"\n{'='*75}")
        print(f"  X0 = {X0} BTC  |  alpha = {ALPHA_OB}")
        print(f"{'='*75}")
        lam = find_lam_for_kappa_T(S0, sigma, eta, T_YEARS, KAPPA_T_TARGET)
        results_nonlin[X0] = run_single_x0(
            X0, S0, sigma, gamma, eta, ALPHA_OB, lam,
            label_prefix="[a=0.47] ",
        )

    # ------------------------------------------------------------------
    # Step 4: Summary tables
    # ------------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print("  SUMMARY TABLE: alpha=1.0 (linear impact, closed-form optimal)")
    print(f"{'='*90}")
    print(f"  {'X0 (BTC)':<12} {'kappa*T':>10} {'Obj TWAP':>14} "
          f"{'Obj Opt':>14} {'Obj Sav %':>12} {'MC Sav %':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
    for X0 in X0_VALUES:
        r = results_linear[X0]
        print(f"  {X0:<12} {r['kappa_T']:>10.4f} {r['obj_twap']:>14.2f} "
              f"{r['obj_opt']:>14.2f} {r['savings_obj']:>11.2f}% "
              f"{r['savings_mc']:>11.2f}%")

    print(f"\n{'='*90}")
    print(f"  SUMMARY TABLE: alpha={ALPHA_OB} (nonlinear impact, PDE optimal)")
    print(f"{'='*90}")
    print(f"  {'X0 (BTC)':<12} {'kappa*T':>10} {'Obj TWAP':>14} "
          f"{'Obj Opt':>14} {'Obj Sav %':>12} {'Note':>20}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*14} {'-'*12} {'-'*20}")
    for X0 in X0_VALUES:
        r = results_nonlin[X0]
        note = "OK" if r["savings_obj"] > 0 else "PDE suboptimal"
        print(f"  {X0:<12} {r['kappa_T']:>10.4f} {r['obj_twap']:>14.2f} "
              f"{r['obj_opt']:>14.2f} {r['savings_obj']:>11.2f}% "
              f"{note:>20}")
    print()

    # ------------------------------------------------------------------
    # Step 5: Generate plots (using alpha=1 primary results)
    # ------------------------------------------------------------------
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    COLOR_TWAP = "#378ADD"
    COLOR_OPT = "#1D9E75"
    COLOR_OPT_NL = "#D85A30"

    # --- Plot 1: Multi-X0 trajectories (3-panel, alpha=1) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, X0 in enumerate(X0_VALUES):
        ax = axes[idx]
        r = results_linear[X0]
        p = r["params"]
        t_frac = np.linspace(0, 1, p.N + 1)

        ax.plot(t_frac, r["x_twap"] / p.X0, linewidth=2,
                label="TWAP", color=COLOR_TWAP)
        ax.plot(t_frac, r["x_opt"] / p.X0, linewidth=2,
                label=r["opt_label"], color=COLOR_OPT)

        # Also overlay alpha=0.47 PDE trajectory
        r_nl = results_nonlin[X0]
        ax.plot(t_frac, r_nl["x_opt"] / p.X0, linewidth=2, linestyle="--",
                label=f"PDE (alpha={ALPHA_OB})", color=COLOR_OPT_NL)

        ax.set_xlabel("Time (fraction of T)")
        ax.set_title(f"X0 = {X0} BTC\n"
                     r"$\kappa T$ = " + f"{r['kappa_T']:.2f},  "
                     f"savings = {r['savings_obj']:.1f}%")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Remaining Inventory (fraction of X0)")
    fig.suptitle(f"Optimal Execution Trajectories (T=1h, calibrated from Binance)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path_traj = str(PROJECT_ROOT / "plot_multi_x0_trajectories.png")
    fig.savefig(path_traj, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_traj}")
    plt.close(fig)

    # --- Plot 2: Cost distributions (3 rows, alpha=1) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    for idx, X0 in enumerate(X0_VALUES):
        ax = axes[idx]
        r = results_linear[X0]

        all_c = np.concatenate([r["costs_twap"], r["costs_opt"]])
        lo, hi = np.percentile(all_c, [1, 99])
        bins = np.linspace(lo, hi, 80)

        ax.hist(r["costs_twap"], bins=bins, alpha=0.5, label="TWAP",
                color=COLOR_TWAP, density=True)
        ax.hist(r["costs_opt"], bins=bins, alpha=0.5, label=r["opt_label"],
                color=COLOR_OPT, density=True)

        ax.axvline(r["m_twap"].mean, color=COLOR_TWAP,
                   linestyle="--", linewidth=1.5, label="TWAP mean")
        ax.axvline(r["m_opt"].mean, color=COLOR_OPT,
                   linestyle="--", linewidth=1.5, label="Opt mean")

        ax.set_ylabel("Density")
        ax.set_title(f"X0 = {X0} BTC  |  "
                     f"TWAP mean={r['m_twap'].mean:.2f}  "
                     f"Opt mean={r['m_opt'].mean:.2f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Implementation Shortfall (USD)")
    fig.suptitle(f"Cost Distributions by Order Size (alpha=1, T=1h, {N_PATHS} paths)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path_costs = str(PROJECT_ROOT / "plot_multi_x0_costs.png")
    fig.savefig(path_costs, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_costs}")
    plt.close(fig)

    # --- Plot 3: Savings bar chart (alpha=1) ---
    # Show both absolute savings (USD) and percentage savings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x_labels = [f"{X0} BTC" for X0 in X0_VALUES]

    # Left: absolute objective savings (USD)
    abs_savings = [results_linear[X0]["obj_twap"] - results_linear[X0]["obj_opt"]
                   for X0 in X0_VALUES]
    bars1 = ax1.bar(x_labels, abs_savings, color=COLOR_OPT, alpha=0.8, width=0.5)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Order Size (X0)")
    ax1.set_ylabel("Absolute Objective Savings (USD)")
    ax1.set_title("Absolute Savings (scales quadratically with X0)")
    for bar, s in zip(bars1, abs_savings):
        y_pos = bar.get_height()
        va = "bottom" if y_pos >= 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"${s:,.0f}", ha="center", va=va, fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: percentage savings
    pct_savings = [results_linear[X0]["savings_obj"] for X0 in X0_VALUES]
    bars2 = ax2.bar(x_labels, pct_savings, color=COLOR_OPT, alpha=0.8, width=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Order Size (X0)")
    ax2.set_ylabel("Objective Savings vs TWAP (%)")
    ax2.set_title(f"Percentage Savings (constant at fixed " + r"$\kappa T$" + f"={KAPPA_T_TARGET})")
    for bar, s in zip(bars2, pct_savings):
        y_pos = bar.get_height()
        va = "bottom" if y_pos >= 0 else "top"
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{s:.2f}%", ha="center", va=va, fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Optimization Benefit by Order Size (alpha=1, T=1h)", fontsize=13)
    plt.tight_layout()
    path_savings = str(PROJECT_ROOT / "plot_multi_x0_savings.png")
    fig.savefig(path_savings, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_savings}")
    plt.close(fig)

    print("\n  Done. All plots saved to project root.\n")


if __name__ == "__main__":
    main()
