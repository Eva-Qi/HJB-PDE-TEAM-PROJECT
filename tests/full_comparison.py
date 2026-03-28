"""Full strategy comparison for presentation.

Runs TWAP / VWAP / Optimal across:
- Schemes: exact, euler, milstein
- RNG: pseudo, sobol, antithetic
- Bootstrap (simple + block)

Produces the final comparison table and distribution plots.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from shared.params import DEFAULT_PARAMS, almgren_chriss_closed_form
from shared.cost_model import execution_cost, execution_risk, objective
from montecarlo.sde_engine import generate_normal_increments, simulate_execution
from montecarlo.strategies import twap_trajectory, vwap_trajectory, optimal_trajectory
from montecarlo.cost_analysis import compute_metrics, print_comparison
from montecarlo.bootstrap import (
    generate_synthetic_returns,
    bootstrap_paths_simple,
    bootstrap_paths_block,
    bootstrap_execution_cost,
)


def main():
    params = DEFAULT_PARAMS
    N = params.N
    n_paths = 8192  # power of 2 for Sobol

    # ---- Generate trajectories ----
    x_twap = twap_trajectory(params)
    x_vwap = vwap_trajectory(params)
    x_opt = optimal_trajectory(params)

    strategies = {"TWAP": x_twap, "VWAP": x_vwap, "Optimal": x_opt}

    # ---- Deterministic benchmarks ----
    print("=" * 70)
    print("DETERMINISTIC BENCHMARKS")
    print("=" * 70)
    for name, x in strategies.items():
        cost = execution_cost(x, params)
        risk = execution_risk(x, params)
        obj = objective(x, params)
        print(f"  {name:<10}  E[C]={cost:>14.2f}  Var[C]={risk:>14.2f}  Obj={obj:>14.2f}")

    # ---- Parametric MC: scheme x RNG comparison ----
    print("\n" + "=" * 70)
    print("PARAMETRIC MC: SCHEME x RNG METHOD")
    print("=" * 70)

    schemes = ["exact", "euler", "milstein"]
    rng_methods = ["pseudo", "sobol", "antithetic"]

    # Use TWAP for the comparison (cleanest benchmark)
    true_cost = execution_cost(x_twap, params)

    print(f"\nTWAP true cost: {true_cost:.2f}")
    print(f"{'Scheme':<10} {'RNG':<12} {'Mean Cost':>12} {'Std':>12} "
          f"{'Rel Err':>10} {'VaR95':>12} {'CVaR95':>12}")
    print("-" * 80)

    for scheme in schemes:
        for rng_method in rng_methods:
            Z = generate_normal_increments(n_paths, N, method=rng_method, seed=42)
            _, costs = simulate_execution(
                params, x_twap, n_paths=n_paths,
                antithetic=False, Z_extern=Z, scheme=scheme,
            )
            m = compute_metrics(costs)
            rel_err = abs(m.mean - true_cost) / true_cost
            print(f"{scheme:<10} {rng_method:<12} {m.mean:>12.2f} {m.std:>12.2f} "
                  f"{rel_err:>9.4%} {m.var_95:>12.2f} {m.cvar_95:>12.2f}")

    # ---- Full strategy comparison (best config: exact + sobol) ----
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON (Exact + Sobol, 8192 paths)")
    print("=" * 70)

    Z = generate_normal_increments(n_paths, N, method="sobol", seed=42)
    all_costs = {}

    for name, x in strategies.items():
        _, costs = simulate_execution(
            params, x, n_paths=n_paths,
            antithetic=False, Z_extern=Z, scheme="exact",
        )
        all_costs[name] = costs

    metrics = {name: compute_metrics(costs) for name, costs in all_costs.items()}
    print()
    print_comparison(metrics)

    # ---- Bootstrap comparison ----
    print("\n" + "=" * 70)
    print("BOOTSTRAP COMPARISON (model-free, synthetic returns)")
    print("=" * 70)

    returns = generate_synthetic_returns(params, n_obs=5000, seed=0)

    bootstrap_costs = {}
    for bs_method, bs_func in [("Simple", bootstrap_paths_simple), ("Block", bootstrap_paths_block)]:
        if bs_method == "Simple":
            S_bs = bs_func(returns, S0=params.S0, n_steps=N, n_paths=n_paths)
        else:
            S_bs = bs_func(returns, S0=params.S0, n_steps=N, n_paths=n_paths, block_size=10)

        print(f"\n  {bs_method} bootstrap:")
        for name, x in strategies.items():
            costs = bootstrap_execution_cost(S_bs, x, params)
            m = compute_metrics(costs)
            bootstrap_costs[f"{name} ({bs_method})"] = costs
            print(f"    {name:<10}  Mean={m.mean:>12.2f}  Std={m.std:>12.2f}  "
                  f"VaR95={m.var_95:>12.2f}  CVaR95={m.cvar_95:>12.2f}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"TWAP": "#378ADD", "VWAP": "#D85A30", "Optimal": "#1D9E75"}

    # Panel 1: Parametric cost distributions
    ax = axes[0]
    for name, costs in all_costs.items():
        ax.hist(costs, bins=80, alpha=0.5, label=name, color=colors[name], density=True)
        ax.axvline(np.mean(costs), color=colors[name], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Execution Cost")
    ax.set_ylabel("Density")
    ax.set_title("Parametric MC (Exact + Sobol)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Bootstrap vs Parametric for TWAP
    ax = axes[1]
    ax.hist(all_costs["TWAP"], bins=60, alpha=0.5, label="Parametric", color="#378ADD", density=True)
    bs_simple = bootstrap_execution_cost(
        bootstrap_paths_simple(returns, S0=params.S0, n_steps=N, n_paths=n_paths),
        x_twap, params
    )
    ax.hist(bs_simple, bins=60, alpha=0.5, label="Bootstrap (simple)", color="#D85A30", density=True)
    ax.set_xlabel("Execution Cost")
    ax.set_ylabel("Density")
    ax.set_title("TWAP: Parametric vs Bootstrap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Trajectories
    ax = axes[2]
    t_grid = np.linspace(0, params.T, N + 1) * 252  # convert to trading days
    for name, x in strategies.items():
        ax.plot(t_grid, x, linewidth=2, label=name, color=colors[name])
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Remaining Inventory")
    ax.set_title("Execution Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("full_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to full_comparison.png")


if __name__ == "__main__":
    main()
