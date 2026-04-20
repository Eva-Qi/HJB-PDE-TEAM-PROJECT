"""Sensitivity analysis: how X0, T, lambda, and alpha affect optimal execution.

Sweeps each parameter independently while holding the others at their
calibrated baseline. For each configuration, computes:
    - TWAP expected cost
    - AC-optimal expected cost (closed-form for alpha=1, PDE for alpha!=1)
    - Percentage savings from optimization
    - kappa*T (urgency parameter)

Outputs six publication-quality figures:
    1. Cost vs X0 (log-log, showing quadratic scaling)
    2. Savings (%) vs T (showing diminishing returns at long horizons)
    3. Cost-risk Pareto frontier vs lambda
    4. Optimal trajectory family vs lambda (the classic A&C fan plot)
    5. Cost vs alpha (linear vs nonlinear impact)
    6. Optimal trajectory family vs alpha

Usage:
    cd HJB-PDE-TEAM-PROJECT
    python scripts/sensitivity_sweeps.py
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.params import ACParams, almgren_chriss_closed_form
from shared.cost_model import execution_cost, execution_risk, objective
from montecarlo.strategies import twap_trajectory
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory


# ── Output directory ──────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ── Calibrated baseline (from Binance BTCUSDT, report Table 1) ────
#    eta is a literature fallback; alpha from order book depth walking.
T_1HR = 1.0 / (365.25 * 24)  # 1 hour in years

def find_lam_for_kappa_T(S0, sigma, eta, T, target_kT=1.5):
    """Compute lambda that gives kappa*T = target (linear impact)."""
    kappa_needed = target_kT / T
    return kappa_needed**2 * eta / (S0**2 * sigma**2)

BASE_S0    = 68_918.0
BASE_SIGMA = 0.4214
BASE_GAMMA = 1.48
BASE_ETA   = 1.58e-4
BASE_ALPHA = 1.0       # linear for closed-form; nonlinear sweeps use PDE
BASE_X0    = 10.0
BASE_T     = T_1HR
BASE_N     = 50
BASE_LAM   = find_lam_for_kappa_T(BASE_S0, BASE_SIGMA, BASE_ETA, BASE_T, 1.5)

BASELINE = ACParams(
    S0=BASE_S0, sigma=BASE_SIGMA, mu=0.0,
    X0=BASE_X0, T=BASE_T, N=BASE_N,
    gamma=BASE_GAMMA, eta=BASE_ETA, alpha=BASE_ALPHA, lam=BASE_LAM,
)

# Plotting style
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def get_optimal_trajectory(params: ACParams) -> np.ndarray:
    """Return optimal trajectory: closed-form if alpha=1, PDE otherwise."""
    if abs(params.alpha - 1.0) < 1e-10:
        _, x_opt, _ = almgren_chriss_closed_form(params)
        return x_opt
    else:
        grid, _, v_star = solve_hjb(params, M=200)
        return extract_optimal_trajectory(grid, v_star, params)


def compute_costs(params: ACParams):
    """Return cost/risk/objective for TWAP and optimal, plus savings on objective.

    The meaningful comparison is on the OBJECTIVE (E[cost] + lambda * Var[cost]),
    not on expected cost alone. The optimal trajectory deliberately pays more
    temporary impact to reduce risk — so it always has higher E[cost] than TWAP
    when lambda > 0, but lower objective.
    """
    x_twap = twap_trajectory(params)
    x_opt = get_optimal_trajectory(params)

    cost_twap = execution_cost(x_twap, params)
    cost_opt = execution_cost(x_opt, params)
    risk_twap = execution_risk(x_twap, params)
    risk_opt = execution_risk(x_opt, params)
    obj_twap = cost_twap + params.lam * risk_twap
    obj_opt = cost_opt + params.lam * risk_opt

    # Savings on objective (the metric the optimizer actually minimizes)
    obj_savings_pct = (obj_twap - obj_opt) / abs(obj_twap) * 100 if obj_twap != 0 else 0.0

    kappa_T = params.kappa * params.T
    return {
        "cost_twap": cost_twap, "cost_opt": cost_opt,
        "risk_twap": risk_twap, "risk_opt": risk_opt,
        "obj_twap": obj_twap, "obj_opt": obj_opt,
        "obj_savings_pct": obj_savings_pct, "kappa_T": kappa_T,
        "x_twap": x_twap, "x_opt": x_opt,
    }


# ═══════════════════════════════════════════════════════════════════
# Sweep 1: X0 (order size)
# ═══════════════════════════════════════════════════════════════════

def sweep_x0():
    """Task 3: How does order size affect execution cost and savings?

    Key insight: Under linear impact (alpha=1), both TWAP and optimal costs
    scale as X0^2 (the quadratic cost structure of Almgren-Chriss). The
    percentage savings on the OBJECTIVE stays CONSTANT at fixed kappa*T
    because the model is scale-invariant.

    Under nonlinear impact (alpha<1, e.g. square-root), the scaling is
    sub-quadratic and the savings percentage changes with X0.
    """
    x0_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    # Linear impact sweep
    results_lin = [compute_costs(replace(BASELINE, X0=float(x0))) for x0 in x0_values]

    # Nonlinear impact sweep (alpha=0.47 from order book)
    results_nl = [compute_costs(replace(BASELINE, X0=float(x0), alpha=0.47)) for x0 in x0_values]

    # ── Figure 1: Objective vs X0 + savings ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.loglog(x0_values, [r["obj_twap"] for r in results_lin], "s-",
               label=r"TWAP ($\alpha=1$)", color="tab:blue")
    ax1.loglog(x0_values, [r["obj_opt"] for r in results_lin], "o-",
               label=r"Optimal ($\alpha=1$)", color="tab:orange")
    ax1.loglog(x0_values, [r["obj_twap"] for r in results_nl], "s--",
               label=r"TWAP ($\alpha=0.47$)", color="tab:blue", alpha=0.5)
    ax1.loglog(x0_values, [r["obj_opt"] for r in results_nl], "o--",
               label=r"Optimal ($\alpha=0.47$)", color="tab:orange", alpha=0.5)
    ax1.set_xlabel("Order Size $X_0$ (BTC)")
    ax1.set_ylabel(r"Objective $E[C] + \lambda \cdot Var[C]$ (\$)")
    ax1.set_title("Objective vs Order Size")
    ax1.legend(fontsize=9)

    ax2.semilogx(x0_values, [r["obj_savings_pct"] for r in results_lin],
                 "o-", label=r"$\alpha=1.0$ (linear)", color="tab:green")
    ax2.semilogx(x0_values, [r["obj_savings_pct"] for r in results_nl],
                 "s--", label=r"$\alpha=0.47$ (square-root)", color="tab:purple")
    ax2.set_xlabel("Order Size $X_0$ (BTC)")
    ax2.set_ylabel("Objective Savings (%)")
    ax2.set_title("Optimization Benefit vs Order Size")
    ax2.legend()

    fig.suptitle(f"Sensitivity to Order Size  (T=1hr, $\\kappa T$={BASELINE.kappa*BASELINE.T:.2f})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_x0.png", bbox_inches="tight")
    plt.close(fig)

    # Print table
    print("\n" + "=" * 80)
    print("  SWEEP 1: ORDER SIZE (X0)")
    print("=" * 80)
    print(f"  {'X0':>8}  {'Obj(TWAP)':>14}  {'Obj(Opt)':>14}  "
          f"{'Savings':>10}  {'E[C] TWAP':>12}  {'E[C] Opt':>12}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*12}  {'-'*12}")
    for i, x0 in enumerate(x0_values):
        r = results_lin[i]
        print(f"  {x0:>8}  ${r['obj_twap']:>13,.2f}  ${r['obj_opt']:>13,.2f}  "
              f"{r['obj_savings_pct']:>9.2f}%  ${r['cost_twap']:>11,.2f}  ${r['cost_opt']:>11,.2f}")


# ═══════════════════════════════════════════════════════════════════
# Sweep 2: T (execution horizon)
# ═══════════════════════════════════════════════════════════════════

def sweep_T():
    """Task 4: How does execution horizon affect costs?

    Key insight: Longer horizons reduce temporary impact (slower trading)
    but increase risk exposure. At fixed lambda, kappa*T grows with T
    (kappa is T-independent), so longer horizons make the optimal more
    aggressive. The cost-risk Pareto curve shows the efficient frontier.
    """
    t_hours = [0.25, 0.5, 1, 2, 4, 8, 12, 24]
    t_years = [h / (365.25 * 24) for h in t_hours]

    results = [compute_costs(replace(BASELINE, T=T)) for T in t_years]

    # ── Figure 2: Cost, savings, Pareto vs T ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(t_hours, [r["obj_twap"] for r in results], "s-", label="TWAP", color="tab:blue")
    ax1.plot(t_hours, [r["obj_opt"] for r in results], "o-", label="Optimal", color="tab:orange")
    ax1.set_xlabel("Execution Horizon (hours)")
    ax1.set_ylabel(r"Objective $E[C] + \lambda \cdot Var[C]$")
    ax1.set_title("Objective vs Horizon")
    ax1.legend()

    ax2.plot(t_hours, [r["obj_savings_pct"] for r in results], "D-", color="tab:green")
    ax2.set_xlabel("Execution Horizon (hours)")
    ax2.set_ylabel("Objective Savings (%)")
    ax2.set_title("Optimization Benefit vs Horizon")
    ax2b = ax2.twinx()
    ax2b.plot(t_hours, [r["kappa_T"] for r in results], "--", color="tab:red", alpha=0.6, label=r"$\kappa T$")
    ax2b.set_ylabel(r"$\kappa T$", color="tab:red")
    ax2b.tick_params(axis="y", labelcolor="tab:red")
    ax2b.legend(loc="center right")

    ax3.plot([r["risk_twap"] for r in results], [r["cost_twap"] for r in results],
             "s", color="tab:blue", markersize=8, label="TWAP")
    ax3.plot([r["risk_opt"] for r in results], [r["cost_opt"] for r in results],
             "o", color="tab:orange", markersize=8, label="Optimal")
    for i, h in enumerate(t_hours):
        ax3.annotate(f"{h}h", (results[i]["risk_opt"], results[i]["cost_opt"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax3.set_xlabel("Execution Risk (Variance)")
    ax3.set_ylabel("Expected Cost ($)")
    ax3.set_title("Cost-Risk Tradeoff")
    ax3.legend()

    fig.suptitle(f"Sensitivity to Execution Horizon  ($X_0$={BASE_X0} BTC)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_T.png", bbox_inches="tight")
    plt.close(fig)

    # Print table
    print("\n" + "=" * 80)
    print("  SWEEP 2: EXECUTION HORIZON (T)")
    print("=" * 80)
    print(f"  {'T(hrs)':>8}  {'Obj(TWAP)':>14}  {'Obj(Opt)':>14}  "
          f"{'Savings':>10}  {'κT':>8}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*8}")
    for i, h in enumerate(t_hours):
        r = results[i]
        print(f"  {h:>8.2f}  ${r['obj_twap']:>13,.2f}  ${r['obj_opt']:>13,.2f}  "
              f"{r['obj_savings_pct']:>9.2f}%  {r['kappa_T']:>7.3f}")


# ═══════════════════════════════════════════════════════════════════
# Sweep 3: Lambda (risk aversion)
# ═══════════════════════════════════════════════════════════════════

def sweep_lambda():
    """Task 5: How does risk aversion reshape the optimal trajectory?

    Key insight: Lambda controls the cost-risk tradeoff. At lambda→0,
    the objective is pure cost minimization → TWAP. As lambda increases,
    the risk penalty dominates → front-load aggressively. kappa grows as
    sqrt(lambda), so kappa*T sweeps from ~0 (patient) to large (urgent).

    The Pareto frontier shows the efficient set. The trajectory fan plot
    shows the classic Almgren-Chriss picture of how inventory paths
    change from TWAP-like to front-loaded as risk aversion increases.
    """
    kappa_T_targets = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    results = []
    trajectories = []
    t_grid = np.linspace(0, BASELINE.T, BASELINE.N + 1)
    t_hours_grid = t_grid * 365.25 * 24

    for kt in kappa_T_targets:
        lam = find_lam_for_kappa_T(BASE_S0, BASE_SIGMA, BASE_ETA, BASE_T, kt)
        params = replace(BASELINE, lam=lam)
        r = compute_costs(params)
        r["lam"] = lam
        results.append(r)
        trajectories.append(r["x_opt"])

    x_twap = twap_trajectory(BASELINE)

    # ── Figure 3: Pareto frontier + trajectory fan ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot([r["risk_opt"] for r in results], [r["cost_opt"] for r in results],
             "o-", color="tab:purple", markersize=7, zorder=3)
    ax1.plot(results[0]["risk_twap"], results[0]["cost_twap"], "s",
             color="tab:blue", markersize=10, label="TWAP", zorder=4)
    for i, kt in enumerate(kappa_T_targets):
        ax1.annotate(f"κT={kt}", (results[i]["risk_opt"], results[i]["cost_opt"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    ax1.set_xlabel("Execution Risk (Variance)")
    ax1.set_ylabel("Expected Cost ($)")
    ax1.set_title("Cost-Risk Pareto Frontier")
    ax1.legend()

    cmap = plt.cm.coolwarm
    for i, kt in enumerate(kappa_T_targets):
        color = cmap(i / (len(kappa_T_targets) - 1))
        ax2.plot(t_hours_grid, trajectories[i] / BASE_X0, "-",
                 color=color, label=f"κT={kt:.1f}", linewidth=1.5)
    ax2.plot(t_hours_grid, x_twap / BASE_X0, "k--", label="TWAP", linewidth=2, alpha=0.6)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Remaining Inventory $x(t)/X_0$")
    ax2.set_title("Optimal Trajectories vs Risk Aversion")
    ax2.legend(fontsize=7.5, ncol=2)

    fig.suptitle(f"Sensitivity to Risk Aversion $\\lambda$  ($X_0$={BASE_X0} BTC, T=1hr)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_lambda.png", bbox_inches="tight")
    plt.close(fig)

    # Print table
    print("\n" + "=" * 80)
    print("  SWEEP 3: RISK AVERSION (LAMBDA)")
    print("=" * 80)
    print(f"  {'κT':>6}  {'λ':>14}  {'E[Cost]':>12}  {'Risk':>14}  "
          f"{'Objective':>14}  {'Obj Savings':>12}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*12}")
    for i, kt in enumerate(kappa_T_targets):
        r = results[i]
        print(f"  {kt:>6.2f}  {r['lam']:>14.4e}  ${r['cost_opt']:>11,.2f}  "
              f"{r['risk_opt']:>14.2f}  ${r['obj_opt']:>13,.2f}  {r['obj_savings_pct']:>11.2f}%")
    r0 = results[0]
    print(f"\n  TWAP:  {'':>14}  ${r0['cost_twap']:>11,.2f}  "
          f"{r0['risk_twap']:>14.2f}  (varies with λ)")


# ═══════════════════════════════════════════════════════════════════
# Sweep 4: Alpha (impact exponent)
# ═══════════════════════════════════════════════════════════════════

def sweep_alpha():
    """Task 2 + Task 7: How does the impact exponent affect optimal execution?

    Key insight: Alpha controls the curvature of temporary impact.
    - alpha=1 (linear): h(v) = eta*v → cost is quadratic in trade rate
    - alpha=0.5 (square-root): h(v) = eta*sqrt(v) → concave impact
      penalizes large trades less → optimal front-loads even more
      aggressively (relative to linear) because the impact cost of
      front-loading grows more slowly

    The literature benchmark is Almgren et al. (2005), "Direct Estimation
    of Equity Market Impact", which found alpha ≈ 0.5-0.6 for US equities.
    Our order book estimate gives alpha=0.47 for BTC.
    """
    alpha_values = [0.3, 0.4, 0.47, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []
    trajectories = []
    t_grid = np.linspace(0, BASELINE.T, BASELINE.N + 1)
    t_hours_grid = t_grid * 365.25 * 24

    for alpha in alpha_values:
        params = replace(BASELINE, alpha=alpha)
        r = compute_costs(params)
        results.append(r)
        trajectories.append(r["x_opt"])

    # ── Figure 4: Objective + trajectories vs alpha ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(alpha_values, [r["obj_twap"] for r in results], "s-",
             label="TWAP", color="tab:blue")
    ax1.plot(alpha_values, [r["obj_opt"] for r in results], "o-",
             label="Optimal", color="tab:orange")
    ax1.axvline(x=0.47, color="tab:green", linestyle=":", alpha=0.6, label="Our estimate (0.47)")
    ax1.axvspan(0.5, 0.6, alpha=0.1, color="tab:purple", label="Almgren et al. (2005)")
    ax1.set_xlabel(r"Impact Exponent $\alpha$")
    ax1.set_ylabel(r"Objective $E[C] + \lambda \cdot Var[C]$")
    ax1.set_title(r"Objective vs Impact Exponent $\alpha$")
    ax1.legend(fontsize=9)

    cmap = plt.cm.viridis
    x_twap = twap_trajectory(BASELINE)
    for i, alpha in enumerate(alpha_values):
        color = cmap(i / (len(alpha_values) - 1))
        lw = 2.5 if abs(alpha - 0.47) < 0.01 or abs(alpha - 1.0) < 0.01 else 1.2
        ax2.plot(t_hours_grid, trajectories[i] / BASE_X0, "-",
                 color=color, label=f"α={alpha}", linewidth=lw)
    ax2.plot(t_hours_grid, x_twap / BASE_X0, "k--", label="TWAP", linewidth=2, alpha=0.6)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Remaining Inventory $x(t)/X_0$")
    ax2.set_title(r"Optimal Trajectory vs Impact Exponent")
    ax2.legend(fontsize=7.5, ncol=2)

    fig.suptitle(f"Sensitivity to Impact Exponent  ($X_0$={BASE_X0} BTC, T=1hr, κT={BASELINE.kappa*BASELINE.T:.2f})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_alpha.png", bbox_inches="tight")
    plt.close(fig)

    # Print table
    print("\n" + "=" * 80)
    print("  SWEEP 4: IMPACT EXPONENT (ALPHA)")
    print("=" * 80)
    print(f"  {'Alpha':>8}  {'Obj(TWAP)':>14}  {'Obj(Opt)':>14}  "
          f"{'Obj Savings':>12}  {'Note':>22}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*22}")
    for i, alpha in enumerate(alpha_values):
        r = results[i]
        note = ""
        if abs(alpha - 0.47) < 0.01:
            note = "← our OB estimate"
        elif abs(alpha - 0.5) < 0.01:
            note = "← square-root law"
        elif abs(alpha - 1.0) < 0.01:
            note = "← linear (closed-form)"
        print(f"  {alpha:>8.2f}  ${r['obj_twap']:>13,.2f}  ${r['obj_opt']:>13,.2f}  "
              f"{r['obj_savings_pct']:>11.2f}%  {note:>22}")

    print("\n  Literature benchmark: Almgren et al. (2005) found α ≈ 0.5–0.6")
    print("  for US equities (NYSE). Our BTC estimate (α = 0.47 from order book)")
    print("  is slightly lower, consistent with crypto's higher liquidity provision")
    print("  from 24/7 automated market makers and tighter maker-taker spreads.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("  ALMGREN-CHRISS SENSITIVITY ANALYSIS")
    print(f"  Baseline: S0=${BASE_S0:,.0f}  σ={BASE_SIGMA:.1%}  "
          f"γ={BASE_GAMMA:.2e}  η={BASE_ETA:.1e}")
    print(f"           X0={BASE_X0} BTC  T=1hr  N={BASE_N}  "
          f"λ={BASE_LAM:.4e}  κT={BASELINE.kappa*BASELINE.T:.3f}")
    print("=" * 75)

    sweep_x0()
    sweep_T()
    sweep_lambda()
    sweep_alpha()

    print(f"\n  Figures saved to: {OUT_DIR}/")
    print(f"    sensitivity_x0.png")
    print(f"    sensitivity_T.png")
    print(f"    sensitivity_lambda.png")
    print(f"    sensitivity_alpha.png")
    print()


if __name__ == "__main__":
    main()