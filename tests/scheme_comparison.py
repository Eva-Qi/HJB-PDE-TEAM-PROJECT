"""Scheme comparison: Exact log-normal vs Euler-Maruyama vs Milstein.

For GBM with market impact, compares:
1. Mean cost accuracy (all should agree with deterministic cost)
2. Price path behavior (Euler can go negative; exact/Milstein shouldn't for small dt)
3. Convergence with dt refinement
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from shared.params import ACParams, almgren_chriss_closed_form
from shared.cost_model import execution_cost
from montecarlo.sde_engine import simulate_execution
from montecarlo.strategies import twap_trajectory


def scheme_comparison():
    """Compare the three schemes at fixed dt."""
    from shared.params import DEFAULT_PARAMS

    x_twap = twap_trajectory(DEFAULT_PARAMS)
    true_cost = execution_cost(x_twap, DEFAULT_PARAMS)

    schemes = ["exact", "euler", "milstein"]
    n_paths = 10000

    print(f"True deterministic cost: {true_cost:.2f}\n")
    print(f"{'Scheme':<12} {'Mean Cost':>12} {'Std':>12} {'Rel Error':>12} {'Min Price':>12}")
    print("-" * 64)

    cost_arrays = {}
    for scheme in schemes:
        paths, costs = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=n_paths, seed=42,
            antithetic=True, scheme=scheme,
        )
        cost_arrays[scheme] = costs
        rel_err = abs(np.mean(costs) - true_cost) / true_cost
        min_price = paths.min()
        print(f"{scheme:<12} {np.mean(costs):>12.2f} {np.std(costs):>12.2f} "
              f"{rel_err:>11.4%} {min_price:>12.2f}")

    # Plot cost distributions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"exact": "#378ADD", "euler": "#D85A30", "milstein": "#1D9E75"}

    # Panel 1: overlaid cost distributions
    ax = axes[0]
    for scheme in schemes:
        ax.hist(cost_arrays[scheme], bins=80, alpha=0.5,
                label=scheme.capitalize(), color=colors[scheme], density=True)
    ax.axvline(true_cost, color="black", linestyle="--", linewidth=1, label="True cost")
    ax.set_xlabel("Execution Cost")
    ax.set_ylabel("Density")
    ax.set_title("Cost distribution by scheme")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: convergence with dt refinement
    ax = axes[1]
    N_values = [10, 20, 50, 100, 200, 500]
    for scheme in schemes:
        errors = []
        for N_val in N_values:
            params_fine = ACParams(
                S0=DEFAULT_PARAMS.S0, sigma=DEFAULT_PARAMS.sigma,
                mu=DEFAULT_PARAMS.mu, X0=DEFAULT_PARAMS.X0,
                T=DEFAULT_PARAMS.T, N=N_val,
                gamma=DEFAULT_PARAMS.gamma, eta=DEFAULT_PARAMS.eta,
                alpha=DEFAULT_PARAMS.alpha, lam=DEFAULT_PARAMS.lam,
            )
            x_twap_fine = params_fine.X0 * np.linspace(1, 0, N_val + 1)
            true_cost_fine = execution_cost(x_twap_fine, params_fine)

            _, costs = simulate_execution(
                params_fine, x_twap_fine, n_paths=5000, seed=42,
                antithetic=True, scheme=scheme,
            )
            rel_err = abs(np.mean(costs) - true_cost_fine) / true_cost_fine
            errors.append(rel_err)

        dt_values = [DEFAULT_PARAMS.T / N_val for N_val in N_values]
        ax.loglog(dt_values, errors, "o-", color=colors[scheme],
                  label=scheme.capitalize(), linewidth=2)

    ax.set_xlabel("dt (time step size)")
    ax.set_ylabel("Relative error of mean cost")
    ax.set_title("Convergence with dt refinement")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: sample price paths (first 20 paths)
    ax = axes[2]
    for scheme in schemes:
        paths, _ = simulate_execution(
            DEFAULT_PARAMS, x_twap, n_paths=20, seed=42,
            antithetic=False, scheme=scheme,
        )
        for i in range(5):
            ax.plot(paths[i, :], alpha=0.4, color=colors[scheme],
                    label=scheme.capitalize() if i == 0 else None)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Price")
    ax.set_title("Sample price paths (5 per scheme)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scheme_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to scheme_comparison.png")


if __name__ == "__main__":
    scheme_comparison()
