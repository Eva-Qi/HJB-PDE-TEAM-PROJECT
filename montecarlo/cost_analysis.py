"""Cost distribution analysis: VaR, CVaR, and strategy comparison.

After MC simulation produces cost arrays for each strategy,
this module computes risk metrics, bootstrap confidence intervals,
and statistical comparisons.

Bootstrap CI method:
    Resample the MC cost array B times with replacement. Compute the
    statistic (mean, VaR, CVaR) on each resample. The 2.5th and 97.5th
    percentiles of the bootstrap distribution give the 95% CI.

References:
    Efron & Tibshirani (1993), Ch. 12-14 (bootstrap CIs)
    Glasserman (2004), Ch. 9.3 (bootstrap in MC simulation)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import Literal, Optional

from scipy import stats

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


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a single statistic."""

    estimate: float    # point estimate from original sample
    ci_lower: float    # lower bound of confidence interval
    ci_upper: float    # upper bound of confidence interval
    ci_level: float    # confidence level (e.g., 0.95)
    n_bootstrap: int   # number of bootstrap resamples


@dataclass
class CostMetricsWithCI:
    """Cost metrics with bootstrap confidence intervals."""

    mean: BootstrapCI
    var_95: BootstrapCI
    cvar_95: BootstrapCI
    std: float
    median: float
    n_paths: int

@dataclass
class PairedTestResult:
    """Result of a paired comparison between two execution strategies.

    Sign convention: positive `mean_diff` means strategy A costs MORE than
    strategy B on average (i.e. B is the better strategy). Flip the argument
    order in `paired_strategy_test` to reverse the reading.

    Fields populated by `paired_strategy_test` depend on the `test` argument:
    fields for tests that were not requested are left as None.

    Attributes
    ----------
    label_a, label_b : str
        Human-readable strategy names used for printing and plotting.
    n_paths : int
        Number of paired observations (paths) used.
    mean_diff : float
        Sample mean of D_i = C^A_i - C^B_i. Point estimate of E[D].
    t_statistic : float or None
        Paired t statistic. None if the t-test was not requested.
    t_pvalue : float or None
        Two-sided p-value from the paired t-test. None if not requested.
    bootstrap_pvalue : float or None
        Two-sided p-value from the paired bootstrap test. None if not requested.
    """

    label_a: str
    label_b: str
    n_paths: int
    mean_diff: float
    t_statistic: Optional[float] = None
    t_pvalue: Optional[float] = None
    bootstrap_pvalue: Optional[float] = None

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


def bootstrap_confidence_interval(
    costs: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a cost statistic.

    Resamples the cost array B times with replacement, computes the
    statistic on each resample, and returns percentile-based CI.

    Parameters
    ----------
    costs : np.ndarray, shape (n_paths,)
        Execution costs from MC simulation.
    statistic : "mean" | "var_95" | "cvar_95"
        Which statistic to compute CI for.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95 → 95% CI).
    seed : int

    Returns
    -------
    BootstrapCI
    """
    rng = np.random.default_rng(seed)
    n = len(costs)

    # Point estimate from original sample
    point_est = _compute_statistic(costs, statistic)

    # Bootstrap resampling
    boot_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        resample = costs[rng.integers(0, n, size=n)]
        boot_stats[b] = _compute_statistic(resample, statistic)

    # Percentile CI
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        estimate=point_est,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def compute_metrics_with_ci(
    costs: np.ndarray,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> CostMetricsWithCI:
    """Compute cost metrics with bootstrap CIs for mean, VaR95, CVaR95.

    Parameters
    ----------
    costs : np.ndarray, shape (n_paths,)
    n_bootstrap : int
    ci_level : float
    seed : int

    Returns
    -------
    CostMetricsWithCI
    """
    return CostMetricsWithCI(
        mean=bootstrap_confidence_interval(
            costs, "mean", n_bootstrap, ci_level, seed
        ),
        var_95=bootstrap_confidence_interval(
            costs, "var_95", n_bootstrap, ci_level, seed + 1
        ),
        cvar_95=bootstrap_confidence_interval(
            costs, "cvar_95", n_bootstrap, ci_level, seed + 2
        ),
        std=float(np.std(costs)),
        median=float(np.median(costs)),
        n_paths=len(costs),
    )


def _compute_statistic(costs: np.ndarray, statistic: str) -> float:
    """Compute a named statistic on a cost array."""
    if statistic == "mean":
        return float(np.mean(costs))
    elif statistic == "var_95":
        return float(np.percentile(costs, 95))
    elif statistic == "cvar_95":
        var_95 = np.percentile(costs, 95)
        tail = costs[costs >= var_95]
        return float(np.mean(tail)) if len(tail) > 0 else float(var_95)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")


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


def print_comparison_with_ci(
    metrics: dict[str, CostMetricsWithCI],
) -> None:
    """Pretty-print strategy comparison with bootstrap CIs."""
    print(f"{'Strategy':<12} {'Statistic':<8} {'Estimate':>12} "
          f"{'95% CI Lower':>14} {'95% CI Upper':>14}")
    print("-" * 64)
    for name, m in metrics.items():
        for stat_name, ci in [("Mean", m.mean), ("VaR95", m.var_95), ("CVaR95", m.cvar_95)]:
            print(f"{name:<12} {stat_name:<8} {ci.estimate:>12.2f} "
                  f"{ci.ci_lower:>14.2f} {ci.ci_upper:>14.2f}")
        print()

def paired_t_test(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
) -> tuple[float, float]:
    """Paired t-test comparing two strategies' per-path execution costs.

    Tests H_0: E[C_a - C_b] = 0 against a two-sided alternative. Assumes
    costs_a[i] and costs_b[i] are realized on the SAME simulated price path
    (i.e. caller ran both simulations with a shared Z_extern). Pairing is
    what gives this test its variance-reduction power over an unpaired
    two-sample t-test on the same data.

    Validity at large n_paths is justified by the CLT acting on the sample
    mean of D_i = costs_a[i] - costs_b[i], not by D_i itself being normal.

    Parameters
    ----------
    costs_a : np.ndarray, shape (n_paths,)
        Realized execution costs for strategy A.
    costs_b : np.ndarray, shape (n_paths,)
        Realized execution costs for strategy B, paired by index with costs_a.

    Returns
    -------
    t_stat : float
        Paired t statistic.
    p_value : float
        Two-sided p-value under H_0: E[C_a - C_b] = 0.

    Raises
    ------
    ValueError
        If costs_a and costs_b have different shapes.
    """
    if costs_a.shape != costs_b.shape:
        raise ValueError(
            f"shape mismatch: {costs_a.shape} vs {costs_b.shape}"
        )

    diff = costs_a - costs_b
    result = stats.ttest_1samp(diff, popmean=0.0)
    return float(result.statistic), float(result.pvalue)

def paired_bootstrap_test(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap test comparing two strategies' per-path costs.

    Tests H_0: E[C_a - C_b] = 0 against a two-sided alternative, without
    the CLT-based normality assumption used by paired_t_test. Appropriate
    when n_paths is small or when the distribution of D_i = costs_a - costs_b
    is heavy-tailed enough that CLT convergence is suspect.

    Method: shift D so its mean is zero (making it obey H_0 by construction),
    then resample the shifted array with replacement n_bootstrap times and
    compute the mean on each resample. The empirical distribution of these
    null bootstrap means is the null distribution. The p-value is the
    fraction of null bootstrap means at least as extreme (in absolute value)
    as the observed mean_diff.

    Parameters
    ----------
    costs_a : np.ndarray, shape (n_paths,)
        Realized execution costs for strategy A.
    costs_b : np.ndarray, shape (n_paths,)
        Paired with costs_a by index.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Seed for the bootstrap resampler.

    Returns
    -------
    mean_diff : float
        Observed sample mean of D_i = costs_a[i] - costs_b[i].
    p_value : float
        Two-sided bootstrap p-value under H_0: E[D] = 0.

    Raises
    ------
    ValueError
        If costs_a and costs_b have different shapes.
    """
    if costs_a.shape != costs_b.shape:
        raise ValueError(
            f"shape mismatch: {costs_a.shape} vs {costs_b.shape}"
        )

    diff = costs_a - costs_b
    mean_diff = float(np.mean(diff))
    shifted = diff - mean_diff  # now sample-mean is exactly 0 (null world)

    rng = np.random.default_rng(seed)
    n = len(diff)

    # Vectorized resampling: one 2D index array, one fancy-index lookup.
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    null_boot_means = shifted[idx].mean(axis=1)

    p_value = float(np.mean(np.abs(null_boot_means) >= abs(mean_diff)))

    return mean_diff, p_value

def paired_strategy_test(
    costs_a: np.ndarray,
    costs_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    test: Literal["t", "bootstrap", "both"] = "both",
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> PairedTestResult:
    """Paired comparison of two strategies' per-path costs.

    Dispatches to `paired_t_test` and/or `paired_bootstrap_test` depending
    on the `test` argument and packages the results into a single
    `PairedTestResult`. Both tests share the same null hypothesis
    H_0: E[C_a - C_b] = 0 (two-sided).

    The recommended default is `test="both"`: the t-test provides the
    classical benchmark (valid by CLT at large n_paths) and the bootstrap
    provides a distribution-free check robust to the skew in execution
    cost distributions. Agreement between the two makes the finding robust.

    Parameters
    ----------
    costs_a, costs_b : np.ndarray, shape (n_paths,)
        Realized execution costs, paired by index (same simulated paths).
    label_a, label_b : str
        Names for the two strategies in the returned result.
    test : {"t", "bootstrap", "both"}
        Which test(s) to run.
    n_bootstrap : int
        Number of bootstrap resamples (ignored if test == "t").
    seed : int
        Seed for the bootstrap resampler (ignored if test == "t").

    Returns
    -------
    PairedTestResult

    Raises
    ------
    ValueError
        If `test` is not one of the allowed values, or if the two cost
        arrays have different shapes (propagated from the primitives).
    """
    if test not in ("t", "bootstrap", "both"):
        raise ValueError(
            f"unknown test: {test!r}. Use 't', 'bootstrap', or 'both'."
        )

    n_paths = len(costs_a)
    mean_diff = float(np.mean(costs_a - costs_b))

    t_statistic: Optional[float] = None
    t_pvalue: Optional[float] = None
    bootstrap_pvalue: Optional[float] = None

    if test in ("t", "both"):
        t_statistic, t_pvalue = paired_t_test(costs_a, costs_b)

    if test in ("bootstrap", "both"):
        _, bootstrap_pvalue = paired_bootstrap_test(
            costs_a, costs_b, n_bootstrap=n_bootstrap, seed=seed
        )

    return PairedTestResult(
        label_a=label_a,
        label_b=label_b,
        n_paths=n_paths,
        mean_diff=mean_diff,
        t_statistic=t_statistic,
        t_pvalue=t_pvalue,
        bootstrap_pvalue=bootstrap_pvalue,
    )