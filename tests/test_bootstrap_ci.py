"""Tests for bootstrap confidence intervals and CostMetricsWithCI.

REWRITE HISTORY 2026-04-19: audit deleted 7 theater/tautology tests and
FIXED a wrong-direction assertion bug in test_ci_level:
  DELETED:
    test_ci_contains_estimate    — identity of percentile construction
    test_ci_finite               — covered by percentile nature
    test_cvar95_ci               — CVaR>=VaR covered in test_cost_model
    test_returns_correct_type    — isinstance of dataclass field
    test_n_paths_correct         — literal field assignment
    test_n_bootstrap_recorded    — literal field assignment
  BUG FIX:
    test_ci_level had `width_99 >= width_95 * 0.9` which ALLOWED the 99%
    CI to be NARROWER than the 95% CI. Fixed to `width_99 > width_95` —
    the actual invariant that wider confidence = wider interval.
"""

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS
from montecarlo.sde_engine import simulate_execution
from montecarlo.strategies import twap_trajectory
from montecarlo.cost_analysis import (
    bootstrap_confidence_interval,
    compute_metrics_with_ci,
    BootstrapCI,
    CostMetricsWithCI,
)


@pytest.fixture
def mc_costs():
    """Generate MC costs for testing."""
    x_twap = twap_trajectory(DEFAULT_PARAMS)
    _, costs = simulate_execution(DEFAULT_PARAMS, x_twap, n_paths=5000, seed=42)
    return costs


class TestBootstrapCI:

    def test_ci_narrower_with_more_data(self):
        """More MC paths should give narrower CI (CLT scaling)."""
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        _, costs_small = simulate_execution(DEFAULT_PARAMS, x_twap, n_paths=500, seed=42)
        _, costs_large = simulate_execution(DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42)
        ci_small = bootstrap_confidence_interval(costs_small, "mean", n_bootstrap=1000, seed=0)
        ci_large = bootstrap_confidence_interval(costs_large, "mean", n_bootstrap=1000, seed=0)
        width_small = ci_small.ci_upper - ci_small.ci_lower
        width_large = ci_large.ci_upper - ci_large.ci_lower
        assert width_large < width_small

    def test_var95_ci(self, mc_costs):
        """var_95 bootstrap CI should contain the point estimate."""
        ci = bootstrap_confidence_interval(mc_costs, "var_95", n_bootstrap=1000)
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper

    def test_invalid_statistic(self, mc_costs):
        """Unknown statistic name must raise."""
        with pytest.raises(ValueError):
            bootstrap_confidence_interval(mc_costs, "invalid_stat")

    def test_ci_level_higher_confidence_gives_wider_interval(self, mc_costs):
        """99% CI MUST be wider than 95% CI — this is the defining invariant
        of a confidence level. Audit fixed a prior wrong-direction bug where
        `width_99 >= width_95 * 0.9` allowed a NARROWER 99% CI to pass."""
        ci_95 = bootstrap_confidence_interval(mc_costs, "mean", ci_level=0.95)
        ci_99 = bootstrap_confidence_interval(mc_costs, "mean", ci_level=0.99)
        width_95 = ci_95.ci_upper - ci_95.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_95, (
            f"99% CI width ({width_99:.4f}) must exceed 95% CI width "
            f"({width_95:.4f}) — higher confidence REQUIRES wider interval. "
            f"If reversed, bootstrap quantile mapping is inverted."
        )


class TestCostMetricsWithCI:

    def test_all_fields_finite(self, mc_costs):
        """All CI fields and point estimates must be finite numbers."""
        m = compute_metrics_with_ci(mc_costs, n_bootstrap=500)
        for ci in [m.mean, m.var_95, m.cvar_95]:
            assert np.isfinite(ci.estimate)
            assert np.isfinite(ci.ci_lower)
            assert np.isfinite(ci.ci_upper)
        assert np.isfinite(m.std)
        assert np.isfinite(m.median)
