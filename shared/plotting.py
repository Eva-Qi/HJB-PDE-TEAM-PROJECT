"""Shared plotting utilities for execution trajectory visualization."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from shared.params import ACParams


def plot_trajectories(
    t_grid: np.ndarray,
    trajectories: dict[str, np.ndarray],
    params: ACParams,
    title: str = "Execution Trajectories",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot multiple inventory trajectories on the same axes.

    Parameters
    ----------
    t_grid : np.ndarray
        Time grid (shared across all trajectories).
    trajectories : dict
        {label: x_trajectory} pairs.
    params : ACParams
        For axis labels and annotations.
    title : str
    save_path : str or None
        If provided, saves figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: inventory path
    for label, x in trajectories.items():
        ax1.plot(t_grid * 252, x, linewidth=2, label=label)  # convert to trading days
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Remaining Inventory")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: trading rate
    dt = params.dt
    for label, x in trajectories.items():
        n_k = x[:-1] - x[1:]
        v_k = n_k / dt
        ax2.step(t_grid[:-1] * 252, v_k, where="post", linewidth=1.5, label=label)
    ax2.set_xlabel("Trading Days")
    ax2.set_ylabel("Trading Rate (shares/year)")
    ax2.set_title("Execution Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cost_distribution(
    costs: dict[str, np.ndarray],
    title: str = "Execution Cost Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot cost distributions from Monte Carlo simulation.

    Parameters
    ----------
    costs : dict
        {strategy_label: cost_array} from MC paths.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(costs)))

    for (label, c), color in zip(costs.items(), colors):
        ax1.hist(c, bins=80, alpha=0.6, label=label, color=color, density=True)
        # Mark mean and VaR95
        mean_c = np.mean(c)
        var95 = np.percentile(c, 95)
        ax1.axvline(mean_c, color=color, linestyle="--", linewidth=1.5)
        ax1.axvline(var95, color=color, linestyle=":", linewidth=1.5)

    ax1.set_xlabel("Execution Cost")
    ax1.set_ylabel("Density")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: box plot comparison
    labels = list(costs.keys())
    data = [costs[l] for l in labels]
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Execution Cost")
    ax2.set_title("Strategy Comparison")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
