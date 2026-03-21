"""Tests for cost model functions and cost analysis utilities."""

import numpy as np
import pytest

from shared.params import ACParams, DEFAULT_PARAMS
from shared.cost_model import permanent_impact, temporary_impact, execution_cost
from montecarlo.cost_analysis import compute_metrics, compare_strategies


class TestImpactFunctions:
    """Direct tests for impact functions (previously untested)."""

    def test_permanent_impact_linear(self):
        v = np.array([100, 200, 300])
        gamma = 1e-4
        result = permanent_impact(v, gamma)
        assert np.allclose(result, gamma * v)

    def test_permanent_impact_scalar(self):
        assert permanent_impact(500.0, 2e-5) == pytest.approx(0.01)

    def test_temporary_impact_linear(self):
        """alpha=1: h(v) = eta * v."""
        v = np.array([100, -50, 0])
        eta = 0.01
        result = temporary_impact(v, eta, alpha=1.0)
        assert np.allclose(result, eta * v)

    def test_temporary_impact_nonlinear(self):
        """alpha=0.5: h(v) = eta * |v|^0.5 * sign(v)."""
        eta = 0.01
        assert temporary_impact(100.0, eta, alpha=0.5) == pytest.approx(eta * 10.0)
        assert temporary_impact(0.0, eta, alpha=0.5) == pytest.approx(0.0)

    def test_temporary_impact_negative_v(self):
        """Negative v (buying) should give negative impact."""
        eta = 0.01
        result = temporary_impact(-100.0, eta, alpha=0.5)
        assert result < 0


class TestCostAnalysis:
    """Tests for compute_metrics and compare_strategies (previously untested)."""

    def test_constant_costs(self):
        """All-same costs: mean=VaR=CVaR."""
        costs = np.full(100, 5000.0)
        m = compute_metrics(costs)
        assert m.mean == pytest.approx(5000.0)
        assert m.std == pytest.approx(0.0, abs=1e-10)
        assert m.var_95 == pytest.approx(5000.0)
        assert m.cvar_95 == pytest.approx(5000.0)
        assert m.n_paths == 100

    def test_cvar_geq_var(self):
        """CVaR >= VaR always (Expected Shortfall >= Value-at-Risk)."""
        rng = np.random.default_rng(42)
        costs = rng.normal(1000, 200, 10000)
        m = compute_metrics(costs)
        assert m.cvar_95 >= m.var_95

    def test_known_distribution(self):
        """For uniform [0, 1000]: mean≈500, VaR95≈950."""
        rng = np.random.default_rng(42)
        costs = rng.uniform(0, 1000, 100000)
        m = compute_metrics(costs)
        assert abs(m.mean - 500) < 10
        assert abs(m.var_95 - 950) < 10

    def test_compare_strategies(self):
        rng = np.random.default_rng(42)
        strategy_costs = {
            "TWAP": rng.normal(1000, 100, 1000),
            "Optimal": rng.normal(900, 150, 1000),
        }
        result = compare_strategies(strategy_costs)
        assert "TWAP" in result
        assert "Optimal" in result
        assert result["TWAP"].n_paths == 1000
