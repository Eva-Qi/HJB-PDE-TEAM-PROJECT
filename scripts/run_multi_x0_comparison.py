"""Multi-X0 comparison: how order size affects optimal execution strategy.

Runs the calibrated pipeline at three order sizes (X0 = 10, 100, 1000 BTC)
to demonstrate that larger orders benefit more from optimal execution vs TWAP.

For alpha != 1 (nonlinear impact), uses the PDE solver for optimal trajectory
(closed-form Almgren-Chriss requires alpha=1).

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
from shared.params import ACParams
from shared.cost_model import execution_cost, execution_risk, objective
from montecarlo.strategies import twap_trajectory
from montecarlo.sde_engine import simulate_execution, generate_normal_increments
from montecarlo.cost_analysis import compute_metrics
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "data")
X0_VALUES = [10, 100, 1000]     # BTC
ALPHA_OB = 0.47                 # from order book data
N_STEPS = 50
T_HOURS = 1.0                   # execution horizon in hours
T_YEARS = T_HOURS / (365.25 * 24)
N_PATHS = 5000
SEED = 42
PDE_M = 200

# Lambda sweep: for each X0, find lam that gives kappa*T ~ 1.5
# kappa = sqrt(lam * S0^2 * sigma^2 / eta)
# kappa * T = target => lam = (target / T)^2 * eta / (S0^2 * sigma^2)
KAPPA_T_TARGET = 1.5


def find_lam_for_kappa_T(S0, sigma, eta, T, target_kT=1.5):
    """Compute lambda that gives kappa*T = target.

    kappa = sqrt(lam * S0^2 * sigma^2 / eta)
    kappa*T = target  =>  lam = (target/T)^2 * eta / (S0^2 * sigma^2)
    """
    kappa_needed = target_kT / T
    lam = kappa_needed**2 * eta / (S0**2 * sigma**2)
    return lam


def main():
    print("\n" + "=" * 75)
    print("  MULTI-X0 COMPARISON: Optimal Execution vs TWAP")
    print("  alpha = {:.2f} (from order book), T = {:.0f} hour(s), N = {}".format(
        ALPHA_OB, T_HOURS, N_STEPS))
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
        lam=1e-6,  # placeholder, will override per X0
    )
    base_params = cal.params
    S0 = base_params.S0
    sigma = base_params.sigma
    gamma = base_params.gamma
    # Override alpha with the OB-estimated value
    alpha = ALPHA_OB
    # Use eta from calibration
    eta = base_params.eta

    print(f"  S0 = ${S0:,.2f}")
    print(f"  sigma = {sigma:.4f} (annualized)")
    print(f"  gamma = {gamma:.6e}")
    print(f"  eta = {eta:.6e}")
    print(f"  alpha = {alpha:.2f} (override from OB data)")
    if cal.warnings:
        for w in cal.warnings:
            print(f"    Warning: {w}")

    # ------------------------------------------------------------------
    # Step 2: Run pipeline for each X0
    # ------------------------------------------------------------------
    results = {}  # X0 -> dict of metrics

    for X0 in X0_VALUES:
        print(f"\n{'='*75}")
        print(f"  X0 = {X0} BTC")
        print(f"{'='*75}")

        # Find lambda for target kappa*T
        lam = find_lam_for_kappa_T(S0, sigma, eta, T_YEARS, KAPPA_T_TARGET)

        params = ACParams(
            S0=S0, sigma=sigma, mu=0.0,
            X0=float(X0), T=T_YEARS, N=N_STEPS,
            gamma=gamma, eta=eta, alpha=alpha, lam=lam,
        )

        kappa = params.kappa
        kT = kappa * T_YEARS
        print(f"  lam = {lam:.6e}")
        print(f"  kappa = {kappa:.4f},  kappa*T = {kT:.4f}")

        # --- TWAP trajectory ---
        x_twap = twap_trajectory(params)

        # --- PDE optimal trajectory (works for any alpha) ---
        print(f"  Solving HJB PDE (M={PDE_M}, alpha={alpha:.2f})...")
        grid, V, v_star = solve_hjb(params, M=PDE_M)
        x_opt = extract_optimal_trajectory(grid, v_star, params)

        # --- Deterministic costs ---
        cost_twap_det = execution_cost(x_twap, params)
        cost_opt_det = execution_cost(x_opt, params)
        risk_twap = execution_risk(x_twap, params)
        risk_opt = execution_risk(x_opt, params)
        obj_twap = objective(x_twap, params)
        obj_opt = objective(x_opt, params)

        print(f"  Deterministic cost:  TWAP={cost_twap_det:.4f}  Optimal={cost_opt_det:.4f}")
        print(f"  Execution risk:      TWAP={risk_twap:.4f}  Optimal={risk_opt:.4f}")
        print(f"  Objective (E+lam*V): TWAP={obj_twap:.4f}  Optimal={obj_opt:.4f}")

        # --- Monte Carlo simulation ---
        print(f"  Running Monte Carlo ({N_PATHS} paths, exact scheme)...")
        rng = np.random.default_rng(SEED)
        Z = rng.standard_normal((N_PATHS, N_STEPS))

        _, costs_twap = simulate_execution(
            params, x_twap, n_paths=N_PATHS,
            antithetic=False, Z_extern=Z, scheme="exact",
        )
        _, costs_opt = simulate_execution(
            params, x_opt, n_paths=N_PATHS,
            antithetic=False, Z_extern=Z, scheme="exact",
        )

        metrics_twap = compute_metrics(costs_twap)
        metrics_opt = compute_metrics(costs_opt)

        savings_det = (cost_twap_det - cost_opt_det) / abs(cost_twap_det) * 100
        savings_mc = (metrics_twap.mean - metrics_opt.mean) / abs(metrics_twap.mean) * 100

        print(f"\n  MC Results:")
        print(f"    TWAP:    mean={metrics_twap.mean:.4f}  std={metrics_twap.std:.4f}  "
              f"VaR95={metrics_twap.var_95:.4f}")
        print(f"    Optimal: mean={metrics_opt.mean:.4f}  std={metrics_opt.std:.4f}  "
              f"VaR95={metrics_opt.var_95:.4f}")
        print(f"    Savings vs TWAP (det): {savings_det:.2f}%")
        print(f"    Savings vs TWAP (MC):  {savings_mc:.2f}%")

        results[X0] = {
            "params": params,
            "x_twap": x_twap,
            "x_opt": x_opt,
            "costs_twap": costs_twap,
            "costs_opt": costs_opt,
            "metrics_twap": metrics_twap,
            "metrics_opt": metrics_opt,
            "kappa_T": kT,
            "savings_det": savings_det,
            "savings_mc": savings_mc,
            "cost_twap_det": cost_twap_det,
            "cost_opt_det": cost_opt_det,
            "lam": lam,
        }

    # ------------------------------------------------------------------
    # Step 3: Summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print("  SUMMARY TABLE: Multi-X0 Comparison")
    print(f"{'='*90}")
    print(f"  {'X0 (BTC)':<12} {'kappa*T':>10} {'lam':>14} "
          f"{'TWAP Mean':>14} {'Opt Mean':>14} {'Savings %':>12} "
          f"{'Opt Std':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
    for X0 in X0_VALUES:
        r = results[X0]
        print(f"  {X0:<12} {r['kappa_T']:>10.4f} {r['lam']:>14.6e} "
              f"{r['metrics_twap'].mean:>14.4f} {r['metrics_opt'].mean:>14.4f} "
              f"{r['savings_mc']:>11.2f}% {r['metrics_opt'].std:>12.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Generate plots
    # ------------------------------------------------------------------
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    COLOR_TWAP = "#378ADD"
    COLOR_OPT = "#D85A30"

    # --- Plot 1: Multi-X0 trajectories (3-panel) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for idx, X0 in enumerate(X0_VALUES):
        ax = axes[idx]
        r = results[X0]
        p = r["params"]
        t_frac = np.linspace(0, 1, p.N + 1)  # normalized time [0, 1]

        ax.plot(t_frac, r["x_twap"] / p.X0, linewidth=2,
                label="TWAP", color=COLOR_TWAP)
        ax.plot(t_frac, r["x_opt"] / p.X0, linewidth=2,
                label="Optimal (PDE)", color=COLOR_OPT)

        ax.set_xlabel("Time (fraction of T)")
        ax.set_title(f"X0 = {X0} BTC\n"
                     r"$\kappa T$ = " + f"{r['kappa_T']:.2f},  "
                     f"savings = {r['savings_mc']:.1f}%")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Remaining Inventory (fraction of X0)")
    fig.suptitle(f"Optimal Execution Trajectories (alpha={ALPHA_OB}, T=1h)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path_traj = str(PROJECT_ROOT / "plot_multi_x0_trajectories.png")
    fig.savefig(path_traj, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_traj}")
    plt.close(fig)

    # --- Plot 2: Cost distributions (3 rows x 2 cols, overlaid) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    for idx, X0 in enumerate(X0_VALUES):
        ax = axes[idx]
        r = results[X0]

        # Determine common range for bins
        all_costs = np.concatenate([r["costs_twap"], r["costs_opt"]])
        lo, hi = np.percentile(all_costs, [1, 99])
        bins = np.linspace(lo, hi, 80)

        ax.hist(r["costs_twap"], bins=bins, alpha=0.5, label="TWAP",
                color=COLOR_TWAP, density=True)
        ax.hist(r["costs_opt"], bins=bins, alpha=0.5, label="Optimal (PDE)",
                color=COLOR_OPT, density=True)

        # Mean lines
        ax.axvline(r["metrics_twap"].mean, color=COLOR_TWAP,
                   linestyle="--", linewidth=1.5, label=f"TWAP mean")
        ax.axvline(r["metrics_opt"].mean, color=COLOR_OPT,
                   linestyle="--", linewidth=1.5, label=f"Opt mean")

        ax.set_ylabel("Density")
        ax.set_title(f"X0 = {X0} BTC  |  "
                     f"TWAP mean={r['metrics_twap'].mean:.4f}  "
                     f"Opt mean={r['metrics_opt'].mean:.4f}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Implementation Shortfall (USD)")
    fig.suptitle(f"Cost Distributions by Order Size (alpha={ALPHA_OB}, T=1h, {N_PATHS} paths)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path_costs = str(PROJECT_ROOT / "plot_multi_x0_costs.png")
    fig.savefig(path_costs, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_costs}")
    plt.close(fig)

    # --- Plot 3: Savings bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [f"{X0} BTC" for X0 in X0_VALUES]
    savings_vals = [results[X0]["savings_mc"] for X0 in X0_VALUES]

    bars = ax.bar(x_labels, savings_vals, color=COLOR_OPT, alpha=0.8, width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Order Size (X0)")
    ax.set_ylabel("Cost Savings vs TWAP (%)")
    ax.set_title(f"Optimization Benefit by Order Size\n"
                 f"(alpha={ALPHA_OB}, T=1h, "
                 r"$\kappa T$" + f"={KAPPA_T_TARGET})")

    for bar, s in zip(bars, savings_vals):
        y_pos = bar.get_height()
        va = "bottom" if y_pos >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{s:.2f}%", ha="center", va=va, fontsize=12, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path_savings = str(PROJECT_ROOT / "plot_multi_x0_savings.png")
    fig.savefig(path_savings, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path_savings}")
    plt.close(fig)

    print("\n  Done. All plots saved to project root.\n")


if __name__ == "__main__":
    main()
