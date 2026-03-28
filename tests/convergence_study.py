"""Convergence study: Pseudo-random vs Sobol vs Antithetic.

Shows that Sobol converges faster to the true mean cost.
The "true" cost is the deterministic execution_cost (no stochastic component).
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from shared.params import DEFAULT_PARAMS, almgren_chriss_closed_form
from shared.cost_model import execution_cost
from montecarlo.sde_engine import generate_normal_increments, simulate_execution
from montecarlo.strategies import twap_trajectory


def convergence_study():
    x_twap = twap_trajectory(DEFAULT_PARAMS)
    true_cost = execution_cost(x_twap, DEFAULT_PARAMS)
    N = DEFAULT_PARAMS.N

    # Powers of 2 for Sobol compatibility
    path_counts = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    methods = ["pseudo", "sobol", "antithetic"]
    n_trials = 20  # repeat each to get standard error of the mean estimate

    results = {m: {"means": [], "std_errs": []} for m in methods}

    for n_paths in path_counts:
        print(f"n_paths = {n_paths}")
        for method in methods:
            trial_means = []
            for trial in range(n_trials):
                Z = generate_normal_increments(n_paths, N, method=method, seed=trial)
                _, costs = simulate_execution(
                    DEFAULT_PARAMS, x_twap, n_paths=n_paths,
                    antithetic=False, Z_extern=Z,
                )
                trial_means.append(np.mean(costs))

            # How much does the mean estimate jump around across trials?
            results[method]["means"].append(np.mean(trial_means))
            results[method]["std_errs"].append(np.std(trial_means))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"pseudo": "#378ADD", "sobol": "#1D9E75", "antithetic": "#D85A30"}
    labels = {"pseudo": "Pseudo-random", "sobol": "Sobol QMC", "antithetic": "Antithetic"}

    # Left: standard error vs n_paths (log-log)
    for method in methods:
        ax1.loglog(path_counts, results[method]["std_errs"],
                   "o-", color=colors[method], label=labels[method], linewidth=2)

    # Reference lines for convergence rates
    x_ref = np.array(path_counts, dtype=float)
    ax1.loglog(x_ref, results["pseudo"]["std_errs"][0] * np.sqrt(path_counts[0] / x_ref),
               "--", color="gray", alpha=0.5, label="O(1/√N) reference")

    ax1.set_xlabel("Number of paths")
    ax1.set_ylabel("Std error of mean cost estimate")
    ax1.set_title("Convergence rate comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: mean cost estimate vs n_paths
    for method in methods:
        ax2.semilogx(path_counts, results[method]["means"],
                     "o-", color=colors[method], label=labels[method], linewidth=2)
    ax2.axhline(true_cost, color="black", linestyle="--", linewidth=1, label=f"True cost = {true_cost:.2f}")

    ax2.set_xlabel("Number of paths")
    ax2.set_ylabel("Mean cost estimate")
    ax2.set_title("Convergence to true cost")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nTrue deterministic cost: {true_cost:.2f}")
    print("Plot saved to convergence_comparison.png")
    plt.show()


if __name__ == "__main__":
    convergence_study()
