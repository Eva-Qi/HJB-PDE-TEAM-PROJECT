"""Cost distribution analysis: VaR, CVaR, and strategy comparison.

After MC simulation produces cost arrays for each strategy,
this module computes risk metrics and statistical comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostMetrics:
    """Summary statistics for an execution cost distribution."""

    mean: float
    std: float
    var_95: float      # Value-at-Risk at 95%
    cvar_95: float     # Conditional VaR (Expected Shortfall) at 95%
    median: float
    min: float
    max: float
    n_paths: int


def compute_metrics(costs: np.ndarray) -> CostMetrics:
    """Compute cost distribution summary statistics.

    Parameters
    ----------
    costs : np.ndarray, shape (n_paths,)
        Execution costs from MC simulation.

    Returns
    -------
    CostMetrics
    """
    var_95 = float(np.percentile(costs, 95))
    # CVaR = average of costs above VaR
    tail = costs[costs >= var_95]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    return CostMetrics(
        mean=float(np.mean(costs)),
        std=float(np.std(costs)),
        var_95=var_95,
        cvar_95=cvar_95,
        median=float(np.median(costs)),
        min=float(np.min(costs)),
        max=float(np.max(costs)),
        n_paths=len(costs),
    )


def compare_strategies(
    strategy_costs: dict[str, np.ndarray],
) -> dict[str, CostMetrics]:
    """Compare cost distributions across execution strategies.

    Parameters
    ----------
    strategy_costs : dict
        {strategy_name: cost_array} from MC simulation.

    Returns
    -------
    dict
        {strategy_name: CostMetrics}
    """
    return {name: compute_metrics(costs) for name, costs in strategy_costs.items()}


def print_comparison(metrics: dict[str, CostMetrics]) -> None:
    """Pretty-print strategy comparison table."""
    header = f"{'Strategy':<15} {'Mean':>12} {'Std':>12} {'VaR95':>12} {'CVaR95':>12}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(
            f"{name:<15} {m.mean:>12.2f} {m.std:>12.2f} "
            f"{m.var_95:>12.2f} {m.cvar_95:>12.2f}"
        )
