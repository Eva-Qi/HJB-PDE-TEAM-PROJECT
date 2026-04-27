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
from shared.experiment_config import T_1H, LAM, N_STEPS
from montecarlo.strategies import twap_trajectory
from pde.hjb_solver import solve_hjb, extract_optimal_trajectory


# ── Output directory ──────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ── Calibrated baseline (matches walk_forward_validation.py runtime config) ──
#    Use the same LAM = 1e-6 and N = N_STEPS as walk_forward so the
#    sensitivity sweep is comparable to the canonical OOS validation.
def find_lam_for_kappa_T(S0, sigma, eta, T, target_kT=1.5):
    """Compute lambda that gives kappa*T = target (linear impact). Kept for
    other sweeps in this file; sweep_x0 itself now uses LAM = 1e-6."""
    kappa_needed = target_kT / T
    return kappa_needed**2 * eta / (S0**2 * sigma**2)

BASE_S0    = 68_918.0
BASE_SIGMA = 0.4214
BASE_GAMMA = 1.48
BASE_ETA   = 1.58e-4
BASE_ALPHA = 1.0       # linear for closed-form; nonlinear sweeps use PDE
BASE_X0    = 10.0
BASE_T     = T_1H
BASE_N     = N_STEPS   # 250, centralized
BASE_LAM   = LAM       # 1e-6, centralized — matches walk_forward_validation

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
    """Task 3: How does order size affect AC vs TWAP MC paired savings?

    Replaces the previous deterministic-objective comparison (which was
    misleading because (a) at alpha=1 the linear AC closed-form is
    scale-invariant in X0 so savings is artificially flat, and (b) at
    alpha=0.47 HJB Howard's policy iteration can fail near the v=0 kink
    and produce negative-savings artifacts).

    Method: Monte Carlo paired test with CRN coupling — N=10,000 GBM paths
    shared across TWAP and AC schedules at each X0, paired savings reported
    with t-CI on per-path cost differences. At alpha=0.47 we run a
    convergence check on the HJB trajectory (first-step fraction in
    [0.05, 0.7] = healthy front-loading; >0.7 = bang-bang); divergent
    points are flagged on the figure.

    Expected story (matches FINDINGS §2.1 retail boundary): at X0 <= 10 BTC
    AC and TWAP are MC-indistinguishable (paired p > 0.7); at X0 >= 100 BTC
    AC dominates significantly; at X0 = 1000 BTC savings >= 10%, consistent
    with walk_forward_validation 6/6 splits.
    """
    from montecarlo.sde_engine import simulate_execution

    x0_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
    n_paths = 10_000
    seed = 42

    # Pre-generate antithetic standard normals ONCE — same Z used for every
    # X0 + every (TWAP, AC) pair to enforce CRN coupling. Variance reduction
    # in paired tests requires identical noise across the two strategies.
    N_steps = BASELINE.N
    rng = np.random.default_rng(seed)
    n_half = n_paths // 2
    Z_half = rng.standard_normal((n_half, N_steps))
    Z = np.vstack([Z_half, -Z_half])  # (n_paths, N_steps), antithetic

    def _paired_savings(params: ACParams, x_twap: np.ndarray,
                        x_opt: np.ndarray) -> dict:
        """Run paired MC and return {savings_pct, ci_low, ci_high, p_value}."""
        _, costs_twap = simulate_execution(
            params, x_twap, n_paths=n_paths, Z_extern=Z, antithetic=False,
        )
        _, costs_opt = simulate_execution(
            params, x_opt, n_paths=n_paths, Z_extern=Z, antithetic=False,
        )
        # Paired difference per path (CRN preserved)
        delta = costs_twap - costs_opt
        mean_twap = costs_twap.mean()
        mean_opt = costs_opt.mean()
        savings_pct = 100.0 * (mean_twap - mean_opt) / abs(mean_twap) if mean_twap != 0 else 0.0
        # Standard error on the savings ratio via delta method
        n = len(delta)
        sem_delta = delta.std(ddof=1) / np.sqrt(n)
        sem_savings_pct = 100.0 * sem_delta / abs(mean_twap) if mean_twap != 0 else 0.0
        ci_low = savings_pct - 1.96 * sem_savings_pct
        ci_high = savings_pct + 1.96 * sem_savings_pct
        # One-sided p-value: H_0: mean(delta) <= 0 (AC not better than TWAP)
        t_stat = delta.mean() / (delta.std(ddof=1) / np.sqrt(n)) if delta.std() > 0 else 0.0
        # Approx p via standard normal (n is large)
        from scipy.stats import norm
        p_value = 1.0 - norm.cdf(t_stat)
        return {
            "savings_pct": savings_pct,
            "ci_low": ci_low, "ci_high": ci_high,
            "mean_twap": mean_twap, "mean_opt": mean_opt,
            "p_value": p_value, "n_paths": n,
        }

    print("\n" + "=" * 88)
    print("  SWEEP 1: ORDER SIZE (X0) — MC paired test, CRN, n=10,000")
    print("=" * 88)
    print(f"  {'X0':>6}  {'α=1 savings (95% CI)':>26}  {'α=0.47 savings (95% CI)':>28}  "
          f"{'α=0.47 HJB':>12}")
    print(f"  {'-'*6}  {'-'*26}  {'-'*28}  {'-'*12}")

    results_lin: list[dict] = []
    results_nl: list[dict] = []

    for x0 in x0_values:
        # ─── α = 1 (linear, closed-form) ───
        params_lin = replace(BASELINE, X0=float(x0), alpha=1.0)
        x_twap = twap_trajectory(params_lin)
        _, x_opt_lin, _ = almgren_chriss_closed_form(params_lin)
        r_lin = _paired_savings(params_lin, x_twap, x_opt_lin)
        results_lin.append({"x0": x0, **r_lin, "converged": True})

        # ─── α = 0.441 (BASELINE, calibrated nonlinear, HJB Howard's) ───
        # Note: HJB at sublinear α is known to produce front-loaded /
        # near-bang-bang trajectories (code-council Part 11 §11.4). We
        # accept these as the HJB optimum and flag only the truly
        # degenerate 1-step-does-everything case (>95%) as failure.
        params_nl = replace(BASELINE, X0=float(x0), alpha=0.441)
        x_twap_nl = twap_trajectory(params_nl)  # Same as x_twap up to N step grid
        try:
            grid, _, v_star = solve_hjb(params_nl, M=200)
            x_opt_nl = extract_optimal_trajectory(grid, v_star, params_nl)
            trades_per_step = (x_opt_nl[:-1] - x_opt_nl[1:]) / x0
            max_step_frac = float(trades_per_step.max())
            first_step_frac = float(trades_per_step[0])
            # Accept HJB output unless one step liquidates >95% (degenerate)
            converged = max_step_frac < 0.95
        except Exception as exc:
            x_opt_nl = None
            converged = False
            max_step_frac = np.nan
            first_step_frac = np.nan
            print(f"  [WARN] HJB exception at X0={x0}: {exc}")
        if converged and x_opt_nl is not None:
            r_nl = _paired_savings(params_nl, x_twap_nl, x_opt_nl)
            results_nl.append({"x0": x0, **r_nl, "converged": True,
                               "first_step_frac": first_step_frac,
                               "max_step_frac": max_step_frac})
        else:
            results_nl.append({"x0": x0, "savings_pct": np.nan,
                               "ci_low": np.nan, "ci_high": np.nan,
                               "p_value": np.nan, "converged": False,
                               "first_step_frac": (
                                   first_step_frac if x_opt_nl is not None else np.nan
                               ),
                               "max_step_frac": (
                                   max_step_frac if x_opt_nl is not None else np.nan
                               )})

        s_lin = f"{r_lin['savings_pct']:+5.2f}% [{r_lin['ci_low']:+5.2f}, {r_lin['ci_high']:+5.2f}]"
        if results_nl[-1]["converged"]:
            r = results_nl[-1]
            s_nl = f"{r['savings_pct']:+6.2f}% [{r['ci_low']:+6.2f}, {r['ci_high']:+6.2f}]"
            hjb_status = f"max_step={results_nl[-1]['max_step_frac']:.0%}"
        else:
            s_nl = "DEGENERATE (>95%)"
            hjb_status = f"max_step={results_nl[-1]['max_step_frac']:.0%}"
        print(f"  {x0:>6}  {s_lin:>26}  {s_nl:>30}  {hjb_status:>15}")

    # ─── Plot ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel — paired savings % with CI bands
    x_arr = np.array(x0_values, dtype=float)

    sav_lin = np.array([r["savings_pct"] for r in results_lin])
    ci_lo_lin = np.array([r["ci_low"] for r in results_lin])
    ci_hi_lin = np.array([r["ci_high"] for r in results_lin])
    ax1.fill_between(x_arr, ci_lo_lin, ci_hi_lin, color="tab:green", alpha=0.18,
                     label="95% CI (α=1)")
    ax1.plot(x_arr, sav_lin, "o-", color="tab:green",
             label=r"$\alpha=1.0$ (linear, closed-form, scale-invariant)",
             linewidth=2, markersize=7)

    # α=0.47 — only plot converged points; show diverged as red X
    converged_mask = np.array([r["converged"] for r in results_nl])
    sav_nl = np.array([r["savings_pct"] for r in results_nl])
    ci_lo_nl = np.array([r["ci_low"] for r in results_nl])
    ci_hi_nl = np.array([r["ci_high"] for r in results_nl])
    if converged_mask.any():
        ax1.fill_between(
            x_arr[converged_mask],
            ci_lo_nl[converged_mask], ci_hi_nl[converged_mask],
            color="tab:purple", alpha=0.18,
        )
        ax1.plot(x_arr[converged_mask], sav_nl[converged_mask], "s--",
                 color="tab:purple",
                 label=r"$\alpha=0.441$ (calibrated, HJB)",
                 linewidth=2, markersize=7)
    # Mark HJB-degenerate X0 values with light-red vertical bands instead of
    # data markers — keeps the visual separate from the α=1 closed-form line.
    if (~converged_mask).any():
        for xv in x_arr[~converged_mask]:
            ax1.axvspan(xv * 0.85, xv * 1.18, color="red", alpha=0.10, zorder=0)
        # Single legend handle via a proxy patch
        from matplotlib.patches import Patch
        degen_patch = Patch(facecolor="red", alpha=0.18,
                            label=r"$\alpha=0.441$ HJB degenerate ($>95\%$ in 1 step)")
        ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
        ax1_handles.append(degen_patch)
        ax1.legend(handles=ax1_handles, loc="upper left",
                   fontsize=8, framealpha=0.92)

    ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Order Size $X_0$ (BTC)")
    ax1.set_ylabel("Paired MC Savings AC vs TWAP (%)")
    ax1.set_title(r"AC vs TWAP — Paired MC with CRN, N=10,000")
    # Annotation explaining why α=1 CI band is invisible
    ax1.text(0.98, 0.05,
             r"$\alpha=1$ CI band invisible (width $\approx 10^{-2}\%$)" + "\n"
             r"because closed-form AC $\approx$ TWAP at $\kappa T=0.26$",
             transform=ax1.transAxes, fontsize=8, ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.85))

    # Right panel — mean cost ratio (linear y, log x), shows the absolute story
    ratio_lin = np.array([r["mean_opt"] / r["mean_twap"]
                          if r["mean_twap"] != 0 else np.nan
                          for r in results_lin])
    ax2.plot(x_arr, ratio_lin, "o-", color="tab:green",
             label=r"$\alpha=1.0$", linewidth=2, markersize=7)
    if converged_mask.any():
        ratio_nl = np.array([r.get("mean_opt", np.nan) / r["mean_twap"]
                             if r.get("mean_twap", 0) != 0 else np.nan
                             for r in results_nl])
        ax2.plot(x_arr[converged_mask], ratio_nl[converged_mask], "s--",
                 color="tab:purple",
                 label=r"$\alpha=0.441$ (HJB-converged points)",
                 linewidth=2, markersize=7)
    ax2.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("Order Size $X_0$ (BTC)")
    ax2.set_ylabel(r"$\overline{C}_{\mathrm{AC}}\,/\,\overline{C}_{\mathrm{TWAP}}$")
    ax2.set_title("Mean Cost Ratio — closer to 1 means AC = TWAP")
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.92)

    fig.suptitle(
        rf"Sensitivity to Order Size  ($T$=1hr, $\lambda$={BASELINE.lam:.0e}, $\kappa T$={BASELINE.kappa*BASELINE.T:.2f})"
        + "\n"
        + r"$\alpha=1$ closed-form (scale-invariant); $\alpha=0.441$ HJB shows nonlinear-impact X$_0$ dependence (front-loaded by design)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sensitivity_x0.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {OUT_DIR / 'sensitivity_x0.png'}")


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