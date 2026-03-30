"""Tests for bootstrap confidence intervals and CostMetricsWithCI."""

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

    def test_ci_contains_estimate(self, mc_costs):
        ci = bootstrap_confidence_interval(mc_costs, "mean", n_bootstrap=2000)
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper

    def test_ci_finite(self, mc_costs):
        ci = bootstrap_confidence_interval(mc_costs, "mean", n_bootstrap=1000)
        assert np.isfinite(ci.ci_lower)
        assert np.isfinite(ci.ci_upper)
        assert np.isfinite(ci.estimate)

    def test_ci_narrower_with_more_data(self):
        """More MC paths should give narrower CI."""
        x_twap = twap_trajectory(DEFAULT_PARAMS)

        _, costs_small = simulate_execution(DEFAULT_PARAMS, x_twap, n_paths=500, seed=42)
        _, costs_large = simulate_execution(DEFAULT_PARAMS, x_twap, n_paths=10000, seed=42)

        ci_small = bootstrap_confidence_interval(costs_small, "mean", n_bootstrap=1000, seed=0)
        ci_large = bootstrap_confidence_interval(costs_large, "mean", n_bootstrap=1000, seed=0)

        width_small = ci_small.ci_upper - ci_small.ci_lower
        width_large = ci_large.ci_upper - ci_large.ci_lower
        assert width_large < width_small

    def test_var95_ci(self, mc_costs):
        ci = bootstrap_confidence_interval(mc_costs, "var_95", n_bootstrap=1000)
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper

    def test_cvar95_ci(self, mc_costs):
        ci = bootstrap_confidence_interval(mc_costs, "cvar_95", n_bootstrap=1000)
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper
        # CVaR >= VaR by definition
        ci_var = bootstrap_confidence_interval(mc_costs, "var_95", n_bootstrap=1000)
        assert ci.estimate >= ci_var.estimate - 1e-6

    def test_invalid_statistic(self, mc_costs):
        with pytest.raises(ValueError):
            bootstrap_confidence_interval(mc_costs, "invalid_stat")

    def test_ci_level(self, mc_costs):
        ci_95 = bootstrap_confidence_interval(mc_costs, "mean", ci_level=0.95)
        ci_99 = bootstrap_confidence_interval(mc_costs, "mean", ci_level=0.99)
        # 99% CI should be wider than 95%
        assert (ci_99.ci_upper - ci_99.ci_lower) >= (ci_95.ci_upper - ci_95.ci_lower) * 0.9


class TestCostMetricsWithCI:

    def test_returns_correct_type(self, mc_costs):
        m = compute_metrics_with_ci(mc_costs, n_bootstrap=500)
        assert isinstance(m, CostMetricsWithCI)
        assert isinstance(m.mean, BootstrapCI)
        assert isinstance(m.var_95, BootstrapCI)
        assert isinstance(m.cvar_95, BootstrapCI)

    def test_all_fields_finite(self, mc_costs):
        m = compute_metrics_with_ci(mc_costs, n_bootstrap=500)
        for ci in [m.mean, m.var_95, m.cvar_95]:
            assert np.isfinite(ci.estimate)
            assert np.isfinite(ci.ci_lower)
            assert np.isfinite(ci.ci_upper)
        assert np.isfinite(m.std)
        assert np.isfinite(m.median)

    def test_n_paths_correct(self, mc_costs):
        m = compute_metrics_with_ci(mc_costs, n_bootstrap=500)
        assert m.n_paths == len(mc_costs)

    def test_n_bootstrap_recorded(self, mc_costs):
        m = compute_metrics_with_ci(mc_costs, n_bootstrap=777)
        assert m.mean.n_bootstrap == 777
