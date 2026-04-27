"""End-to-end calibrated execution pipeline: real Binance data → calibration → PDE → MC.

Bridges the architectural gap between P1 (calibration) and P2/P3 (PDE + MC):

    1. Load Binance aggTrades data and calibrate market impact parameters
    2. Generate TWAP + optimal trajectories using calibrated ACParams
    3. Solve HJB PDE → extract PDE-optimal trajectory
    4. Run Monte Carlo simulation (exact scheme, Sobol QMC)
    5. Compute cost metrics with bootstrap confidence intervals
    6. Print presentation-ready comparison table and save plot

This replaces all DEFAULT_PARAMS usage with real, data-driven parameters
for the April 28 presentation.
"""

import copy
import sys
from pathlib import Path

# Allow running from scripts/ without package install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from calibration.impact_estimator import calibrated_params, CalibrationResult
from calibration.data_loader import load_trades, compute_ohlc
from shared.cost_model import execution_cost, execution_risk, objective
from montecarlo.sde_engine import (
    generate_normal_increments,
    simulate_execution,
    simulate_heston_execution,
)
from montecarlo.strategies import twap_trajectory, optimal_trajectory
from montecarlo.cost_analysis import (
    compute_metrics_with_ci,
    CostMetricsWithCI,
)
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory
from extensions.heston import calibrate_heston_from_spot, HestonParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
N_PATHS = 8192          # power of 2 for Sobol
N_BOOTSTRAP = 5000
SEED = 42
SCHEME = "exact"
RNG_METHOD = "sobol"
PDE_M = 200             # inventory grid resolution for HJB solver


def print_calibration_summary(cal: CalibrationResult) -> None:
    """Print calibration results with source annotations."""
    p = cal.params
    print("=" * 70)
    print("  CALIBRATION RESULTS (Binance BTCUSDT aggTrades)")
    print("=" * 70)
    print(f"  {'Parameter':<12} {'Value':>14}  {'Source':<12}")
    print(f"  {'-'*12} {'-'*14}  {'-'*12}")

    rows = [
        ("S0",    f"${p.S0:,.2f}",         cal.sources.get("S0",    "market data")),
        ("sigma", f"{p.sigma:.4f}",         cal.sources.get("sigma", "—")),
        ("gamma", f"{p.gamma:.6e}",         cal.sources.get("gamma", "—")),
        ("eta",   f"{p.eta:.6e}",           cal.sources.get("eta",   "—")),
        ("alpha", f"{p.alpha:.4f}",         cal.sources.get("alpha", "—")),
        ("X0",    f"{p.X0:.2f} BTC",        "input"),
        ("T",     f"{p.T*365.25*24:.1f} h", "input"),
        ("N",     f"{p.N}",                 "input"),
        ("lam",   f"{p.lam:.2e}",           "input"),
    ]
    for name, val, src in rows:
        print(f"  {name:<12} {val:>14}  {src:<12}")

    # Rogers-Satchell cross-check
    if cal.sigma_rs is not None:
        drift_gap = abs(p.sigma - cal.sigma_rs) / p.sigma
        print(f"\n  Volatility cross-check: GK={p.sigma:.4f}  RS={cal.sigma_rs:.4f}  "
              f"gap={drift_gap:.1%}")

    # Derived quantities
    kappa = p.kappa
    kappa_T = kappa * p.T
    print(f"\n  Derived:  kappa = {kappa:.4f}   kappa*T = {kappa_T:.4f}")
    if kappa_T < 0.3:
        print("  → kappa*T is small: optimal trajectory is close to TWAP")
    elif kappa_T > 2.0:
        print("  → kappa*T is large: optimal trajectory is strongly front-loaded")
    else:
        print("  → kappa*T is moderate: noticeable front-loading vs TWAP")

    if cal.warnings:
        print(f"\n  Warnings ({len(cal.warnings)}):")
        for w in cal.warnings:
            print(f"    ⚠ {w}")
    print()


def print_deterministic_benchmarks(strategies: dict, params) -> None:
    """Print deterministic cost / risk / objective for each strategy."""
    print("=" * 70)
    print("  DETERMINISTIC BENCHMARKS (calibrated params)")
    print("=" * 70)
    print(f"  {'Strategy':<20} {'E[Cost]':>14} {'Var[Cost]':>14} {'Objective':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*14}")
    for name, x in strategies.items():
        cost = execution_cost(x, params)
        risk = execution_risk(x, params)
        obj = objective(x, params)
        print(f"  {name:<20} {cost:>14.4f} {risk:>14.4f} {obj:>14.4f}")
    print()


def print_mc_results(results: dict[str, CostMetricsWithCI]) -> None:
    """Print Monte Carlo cost metrics with bootstrap CIs."""
    print("=" * 70)
    print(f"  MONTE CARLO RESULTS ({N_PATHS} paths, {SCHEME} + {RNG_METHOD})")
    print("=" * 70)
    print(f"  {'Strategy':<20} {'Metric':<8} {'Estimate':>12} "
          f"{'95% CI Lower':>14} {'95% CI Upper':>14}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*14} {'-'*14}")

    for name, m in results.items():
        for stat_name, ci in [("Mean", m.mean), ("VaR95", m.var_95), ("CVaR95", m.cvar_95)]:
            print(f"  {name:<20} {stat_name:<8} {ci.estimate:>12.4f} "
                  f"{ci.ci_lower:>14.4f} {ci.ci_upper:>14.4f}")
        print(f"  {name:<20} {'Std':<8} {m.std:>12.4f}")
        print()


def print_heston_calibration(hp: HestonParams) -> None:
    """Print Heston stochastic volatility calibration results."""
    feller_lhs = 2.0 * hp.kappa * hp.theta
    feller_rhs = hp.xi ** 2
    feller_ok = feller_lhs >= feller_rhs

    print("=" * 70)
    print("  HESTON STOCHASTIC VOLATILITY CALIBRATION")
    print("=" * 70)
    print(f"  {'Parameter':<12} {'Value':>14}  {'Description':<30}")
    print(f"  {'-'*12} {'-'*14}  {'-'*30}")
    rows = [
        ("kappa", f"{hp.kappa:.4f}",     "mean reversion speed"),
        ("theta", f"{hp.theta:.6f}",     "long-run variance"),
        ("sqrt(theta)", f"{np.sqrt(hp.theta):.4f}", "long-run vol (annualized)"),
        ("xi",    f"{hp.xi:.4f}",        "vol-of-vol"),
        ("rho",   f"{hp.rho:.4f}",       "price-variance correlation"),
        ("v0",    f"{hp.v0:.6f}",        "initial variance"),
        ("sqrt(v0)", f"{np.sqrt(max(hp.v0, 0)):.4f}", "initial vol (annualized)"),
    ]
    for name, val, desc in rows:
        print(f"  {name:<12} {val:>14}  {desc:<30}")

    print(f"\n  Feller condition: 2*kappa*theta = {feller_lhs:.4f}  vs  "
          f"xi^2 = {feller_rhs:.4f}  →  {'SATISFIED' if feller_ok else 'VIOLATED'}")
    if not feller_ok:
        print("    (variance can reach zero; simulation uses full truncation)")
    print()


def print_heston_comparison(
    cv_metrics: dict[str, CostMetricsWithCI],
    heston_metrics: CostMetricsWithCI,
    heston_label: str,
) -> None:
    """Print constant-vol vs Heston cost distribution comparison."""
    print("=" * 70)
    print("  CONSTANT-VOL vs HESTON COMPARISON (TWAP trajectory, alpha=1)")
    print("=" * 70)
    print(f"  {'Model':<24} {'Mean':>12} {'Std':>12} {'VaR95':>12}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*12}")

    # Show TWAP constant-vol baseline
    if "TWAP" in cv_metrics:
        m = cv_metrics["TWAP"]
        print(f"  {'TWAP (const-vol)':<24} {m.mean.estimate:>12.4f} "
              f"{m.std:>12.4f} {m.var_95.estimate:>12.4f}")

    # Show Heston
    print(f"  {heston_label:<24} {heston_metrics.mean.estimate:>12.4f} "
          f"{heston_metrics.std:>12.4f} {heston_metrics.var_95.estimate:>12.4f}")

    # Difference
    if "TWAP" in cv_metrics:
        cv_mean = cv_metrics["TWAP"].mean.estimate
        h_mean = heston_metrics.mean.estimate
        diff_pct = (h_mean - cv_mean) / abs(cv_mean) * 100 if cv_mean != 0 else 0
        cv_std = cv_metrics["TWAP"].std
        h_std = heston_metrics.std
        std_ratio = h_std / cv_std if cv_std > 0 else float("inf")
        print(f"\n  Mean cost difference: {diff_pct:+.2f}% "
              f"(Heston {'higher' if diff_pct > 0 else 'lower'})")
        print(f"  Std ratio (Heston / const-vol): {std_ratio:.3f}")
        if std_ratio > 1.05:
            print("  → Stochastic vol WIDENS the cost distribution (fatter tails)")
        elif std_ratio < 0.95:
            print("  → Stochastic vol NARROWS the cost distribution")
        else:
            print("  → Cost distribution width roughly unchanged")
    print()


def make_plot(
    strategies: dict,
    all_costs: dict,
    params,
    output_path: str,
) -> None:
    """Generate a 3-panel figure: trajectories, cost distributions, risk profile."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        "TWAP":              "#378ADD",
        "Optimal (CF)":      "#1D9E75",
        "Optimal (PDE)":     "#D85A30",
        "TWAP (Heston)":     "#8B5CF6",
        "TWAP (Heston, α=1)":"#8B5CF6",
    }
    # Fallback color for any strategy name not pre-defined
    fallback_colors = ["#F59E0B", "#EC4899", "#06B6D4"]
    color_idx = 0
    for name in strategies:
        if name not in colors:
            colors[name] = fallback_colors[color_idx % len(fallback_colors)]
            color_idx += 1

    # --- Panel 1: Inventory trajectories ---
    ax = axes[0]
    t_hours = np.linspace(0, params.T * 365.25 * 24, params.N + 1)
    for name, x in strategies.items():
        ax.plot(t_hours, x, linewidth=2, label=name, color=colors.get(name, "#333"))
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Remaining Inventory (BTC)")
    ax.set_title("Execution Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Cost distributions ---
    ax = axes[1]
    for name, costs in all_costs.items():
        ax.hist(costs, bins=80, alpha=0.5, label=name,
                color=colors.get(name, "#333"), density=True)
        ax.axvline(np.mean(costs), color=colors.get(name, "#333"),
                   linestyle="--", linewidth=1.5)
    ax.set_xlabel("Implementation Shortfall (USD)")
    ax.set_ylabel("Density")
    ax.set_title(f"Cost Distributions ({N_PATHS} paths, {SCHEME} + {RNG_METHOD})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Cost savings vs TWAP ---
    ax = axes[2]
    if "TWAP" in all_costs:
        twap_costs = all_costs["TWAP"]
        twap_mean = np.mean(twap_costs)
        names_sorted = [n for n in all_costs if n != "TWAP"]
        savings = []
        labels = []
        bar_colors = []
        for name in names_sorted:
            pct = (twap_mean - np.mean(all_costs[name])) / abs(twap_mean) * 100
            savings.append(pct)
            labels.append(name)
            bar_colors.append(colors.get(name, "#333"))
        if savings:
            bars = ax.bar(labels, savings, color=bar_colors, alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel("Cost Savings vs TWAP (%)")
            ax.set_title("Optimal Strategy Improvement")
            for bar, s in zip(bars, savings):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{s:.2f}%", ha="center", va="bottom", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No comparison strategies", ha="center",
                    va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "TWAP baseline not available", ha="center",
                va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {output_path}")


def main():
    """Run the full calibrated pipeline."""

    # ------------------------------------------------------------------
    # Step 1: Calibration
    # ------------------------------------------------------------------
    print("\n  Loading Binance data and calibrating parameters...")
    cal = calibrated_params(trades_path=DATA_DIR)
    params = cal.params
    print_calibration_summary(cal)

    # ------------------------------------------------------------------
    # Step 1b: Heston stochastic volatility calibration
    # ------------------------------------------------------------------
    print("  Calibrating Heston stochastic volatility from OHLC data...")
    trades = load_trades(DATA_DIR)
    ohlc_df = compute_ohlc(trades, freq="5min")
    heston_params = calibrate_heston_from_spot(ohlc_df, freq_seconds=300.0)
    print_heston_calibration(heston_params)

    # ------------------------------------------------------------------
    # Step 2: Generate trajectories
    # ------------------------------------------------------------------
    strategies = {}

    # TWAP — always available
    x_twap = twap_trajectory(params)
    strategies["TWAP"] = x_twap

    # Closed-form optimal — only valid for alpha == 1
    is_linear = abs(params.alpha - 1.0) < 1e-10
    if is_linear:
        x_opt_cf = optimal_trajectory(params)
        strategies["Optimal (CF)"] = x_opt_cf

    # PDE optimal — works for any alpha
    print("  Solving HJB PDE...")
    grid, V, v_star = solve_hjb(params, M=PDE_M)
    x_opt_pde = extract_optimal_trajectory(grid, v_star, params)
    strategies["Optimal (PDE)"] = x_opt_pde
    print(f"  PDE solved (M={PDE_M}, N={params.N})")

    if is_linear:
        # Cross-validate: CF vs PDE trajectory should agree
        pde_cf_diff = np.max(np.abs(x_opt_cf - x_opt_pde))
        print(f"  CF vs PDE max trajectory diff: {pde_cf_diff:.6e}")

    # ------------------------------------------------------------------
    # Step 3: Deterministic benchmarks
    # ------------------------------------------------------------------
    print()
    print_deterministic_benchmarks(strategies, params)

    # ------------------------------------------------------------------
    # Step 4: Monte Carlo simulation
    # ------------------------------------------------------------------
    print(f"  Running Monte Carlo ({N_PATHS} paths, {SCHEME} scheme, {RNG_METHOD} RNG)...")

    Z = generate_normal_increments(N_PATHS, params.N, method=RNG_METHOD, seed=SEED)

    all_costs = {}
    mc_metrics = {}

    for name, x in strategies.items():
        _, costs = simulate_execution(
            params, x, n_paths=N_PATHS,
            antithetic=False, Z_extern=Z, scheme=SCHEME,
        )
        all_costs[name] = costs
        mc_metrics[name] = compute_metrics_with_ci(
            costs, n_bootstrap=N_BOOTSTRAP, seed=SEED,
        )

    # ------------------------------------------------------------------
    # Step 4b: Heston MC simulation (TWAP trajectory, alpha=1)
    # ------------------------------------------------------------------
    # Heston extension requires alpha=1 (separable ansatz limitation).
    # If calibrated alpha != 1, create a modified copy for Heston comparison.
    if abs(params.alpha - 1.0) < 1e-10:
        params_heston = params
        heston_label = "TWAP (Heston)"
        heston_note = None
    else:
        params_heston = copy.copy(params)
        params_heston.alpha = 1.0
        heston_label = "TWAP (Heston, α=1)"
        heston_note = (
            f"  NOTE: Calibrated alpha={params.alpha:.4f} != 1. "
            f"Heston comparison uses alpha=1.0 (separable ansatz limitation)."
        )

    # Regenerate TWAP trajectory for the (possibly modified) alpha=1 params.
    # TWAP is trajectory-invariant to alpha, but we use params_heston for
    # consistency in the cost computation.
    x_twap_heston = twap_trajectory(params_heston)

    print(f"  Running Heston MC ({N_PATHS} paths, TWAP trajectory)...")
    if heston_note:
        print(heston_note)
    _, _, heston_costs = simulate_heston_execution(
        params_heston, heston_params, x_twap_heston,
        n_paths=N_PATHS, seed=SEED,
    )
    all_costs[heston_label] = heston_costs
    mc_metrics[heston_label] = compute_metrics_with_ci(
        heston_costs, n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )

    # ------------------------------------------------------------------
    # Step 5: Print MC results
    # ------------------------------------------------------------------
    print()
    print_mc_results(mc_metrics)

    # ------------------------------------------------------------------
    # Step 5b: Constant-vol vs Heston comparison
    # ------------------------------------------------------------------
    print_heston_comparison(mc_metrics, mc_metrics[heston_label], heston_label)

    # ------------------------------------------------------------------
    # Step 6: Summary table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  SUMMARY (presentation-ready)")
    print("=" * 70)
    print(f"  {'Strategy':<20} {'Determ. Cost':>14} {'MC Mean':>14} "
          f"{'MC Std':>14} {'Savings vs TWAP':>16}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*14} {'-'*16}")

    twap_det = execution_cost(x_twap, params)
    twap_mc_mean = mc_metrics["TWAP"].mean.estimate

    for name, x in strategies.items():
        det_cost = execution_cost(x, params)
        mc_mean = mc_metrics[name].mean.estimate
        mc_std = mc_metrics[name].std
        if name == "TWAP":
            savings_str = "— (baseline)"
        else:
            savings_pct = (twap_mc_mean - mc_mean) / abs(twap_mc_mean) * 100
            savings_str = f"{savings_pct:+.2f}%"
        print(f"  {name:<20} {det_cost:>14.4f} {mc_mean:>14.4f} "
              f"{mc_std:>14.4f} {savings_str:>16}")

    # Heston row (no deterministic benchmark — stochastic vol has no closed form)
    h_mc_mean = mc_metrics[heston_label].mean.estimate
    h_mc_std = mc_metrics[heston_label].std
    h_savings_pct = (twap_mc_mean - h_mc_mean) / abs(twap_mc_mean) * 100
    print(f"  {heston_label:<20} {'N/A':>14} {h_mc_mean:>14.4f} "
          f"{h_mc_std:>14.4f} {h_savings_pct:+.2f}%")
    print()

    # ------------------------------------------------------------------
    # Step 7: Plot
    # ------------------------------------------------------------------
    output_plot = str(Path(__file__).resolve().parent.parent / "calibrated_pipeline_results.png")
    make_plot(strategies, all_costs, params, output_plot)
    print()


if __name__ == "__main__":
    main()
