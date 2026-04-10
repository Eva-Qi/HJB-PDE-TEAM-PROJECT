"""Diagnose the PDE-MC cross-validation discrepancy at alpha != 1.

Runs the PDE-derived trajectory through MC at various path counts
and schemes to distinguish MC noise from a real mismatch.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from shared.params import DEFAULT_PARAMS, ACParams
from shared.cost_model import execution_cost
from montecarlo.sde_engine import simulate_execution, simulate_pde_optimal_execution


def diagnose(params: ACParams, label: str):
    print(f"\n{'=' * 60}")
    print(f"  {label}  (alpha={params.alpha})")
    print('=' * 60)

    # Get the PDE trajectory once
    trajectory, _, _ = simulate_pde_optimal_execution(params, n_paths=1000)

    # Trajectory sanity checks
    print(f"\nTrajectory sanity:")
    print(f"  Start: {trajectory[0]:.2f}  (want {params.X0:.2f})")
    print(f"  End:   {trajectory[-1]:.6f}  (want ~0)")
    print(f"  Monotone decreasing: {np.all(np.diff(trajectory) <= 1e-6)}")
    print(f"  Min value: {trajectory.min():.6f}")

    # Deterministic cost on this trajectory
    det_cost = execution_cost(trajectory, params)
    print(f"\nDeterministic cost (formula): {det_cost:.2f}")

    # MC at increasing path counts, check convergence
    print(f"\nMC convergence (scheme=exact):")
    print(f"  {'n_paths':>10}  {'MC mean':>15}  {'Std err':>15}  {'Rel err vs det':>15}")

    for n_paths in [2000, 8000, 32000, 128000]:
        _, costs = simulate_execution(
            params, trajectory, n_paths=n_paths, seed=42,
            antithetic=True, scheme="exact",
        )
        mc_mean = costs.mean()
        std_err = costs.std() / np.sqrt(n_paths)
        rel_err = abs(mc_mean - det_cost) / abs(det_cost) * 100
        print(f"  {n_paths:>10}  {mc_mean:>15.2f}  {std_err:>15.2f}  {rel_err:>14.2f}%")

    # Try different schemes at fixed large n_paths
    print(f"\nScheme comparison (n_paths=32000):")
    print(f"  {'scheme':<10}  {'MC mean':>15}  {'Rel err vs det':>15}")
    for scheme in ["exact", "euler", "milstein"]:
        _, costs = simulate_execution(
            params, trajectory, n_paths=32000, seed=42,
            antithetic=True, scheme=scheme,
        )
        rel_err = abs(costs.mean() - det_cost) / abs(det_cost) * 100
        print(f"  {scheme:<10}  {costs.mean():>15.2f}  {rel_err:>14.2f}%")


def main():
    # Linear baseline — should match within MC noise
    diagnose(DEFAULT_PARAMS, "LINEAR IMPACT (baseline)")

    # Nonlinear — the suspicious case
    nonlinear = ACParams(
        S0=DEFAULT_PARAMS.S0, sigma=DEFAULT_PARAMS.sigma, mu=DEFAULT_PARAMS.mu,
        X0=DEFAULT_PARAMS.X0, T=DEFAULT_PARAMS.T, N=DEFAULT_PARAMS.N,
        gamma=DEFAULT_PARAMS.gamma, eta=DEFAULT_PARAMS.eta,
        alpha=0.6, lam=DEFAULT_PARAMS.lam,
    )
    diagnose(nonlinear, "NONLINEAR IMPACT (alpha=0.6)")


if __name__ == "__main__":
    main()