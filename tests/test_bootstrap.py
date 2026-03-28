"""Tests for statistical bootstrapping module."""

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS
from shared.cost_model import execution_cost
from montecarlo.bootstrap import (
    compute_log_returns,
    bootstrap_paths_simple,
    bootstrap_paths_block,
    bootstrap_execution_cost,
    generate_synthetic_returns,
)
from montecarlo.strategies import twap_trajectory


class TestComputeLogReturns:

    def test_shape(self):
        prices = np.array([100, 101, 99, 102, 98])
        lr = compute_log_returns(prices)
        assert lr.shape == (4,)

    def test_values(self):
        prices = np.array([100.0, 110.0])
        lr = compute_log_returns(prices)
        assert np.isclose(lr[0], np.log(1.1))


class TestBootstrapPaths:

    def setup_method(self):
        self.returns = generate_synthetic_returns(DEFAULT_PARAMS, n_obs=5000, seed=0)

    def test_simple_shape(self):
        S = bootstrap_paths_simple(self.returns, S0=50.0, n_steps=50, n_paths=1000)
        assert S.shape == (1000, 51)

    def test_simple_starts_at_S0(self):
        S = bootstrap_paths_simple(self.returns, S0=50.0, n_steps=50, n_paths=100)
        assert np.allclose(S[:, 0], 50.0)

    def test_simple_positive_prices(self):
        S = bootstrap_paths_simple(self.returns, S0=50.0, n_steps=50, n_paths=1000)
        assert np.all(S > 0), "Bootstrapped paths should be positive (log-normal construction)"

    def test_block_shape(self):
        S = bootstrap_paths_block(self.returns, S0=50.0, n_steps=50, n_paths=1000, block_size=10)
        assert S.shape == (1000, 51)

    def test_block_starts_at_S0(self):
        S = bootstrap_paths_block(self.returns, S0=50.0, n_steps=50, n_paths=100, block_size=10)
        assert np.allclose(S[:, 0], 50.0)

    def test_block_positive_prices(self):
        S = bootstrap_paths_block(self.returns, S0=50.0, n_steps=50, n_paths=1000, block_size=10)
        assert np.all(S > 0)

    def test_reproducibility(self):
        S1 = bootstrap_paths_simple(self.returns, S0=50.0, n_steps=50, n_paths=100, seed=42)
        S2 = bootstrap_paths_simple(self.returns, S0=50.0, n_steps=50, n_paths=100, seed=42)
        assert np.allclose(S1, S2)


class TestBootstrapExecutionCost:

    def test_cost_has_variance(self):
        """Bootstrap costs should be stochastic (nonzero variance)."""
        returns = generate_synthetic_returns(DEFAULT_PARAMS, n_obs=5000, seed=0)
        S = bootstrap_paths_simple(returns, S0=DEFAULT_PARAMS.S0,
                                   n_steps=DEFAULT_PARAMS.N, n_paths=5000)
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        costs = bootstrap_execution_cost(S, x_twap, DEFAULT_PARAMS)

        assert np.std(costs) > 0, "Bootstrap costs should have positive variance"

    def test_mean_cost_reasonable(self):
        """Bootstrap mean cost should be in the same ballpark as parametric."""
        returns = generate_synthetic_returns(DEFAULT_PARAMS, n_obs=5000, seed=0)
        S = bootstrap_paths_simple(returns, S0=DEFAULT_PARAMS.S0,
                                   n_steps=DEFAULT_PARAMS.N, n_paths=10000)
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        costs = bootstrap_execution_cost(S, x_twap, DEFAULT_PARAMS)

        det_cost = execution_cost(x_twap, DEFAULT_PARAMS)

        # Bootstrap with synthetic returns should be in same order of magnitude
        # (not exact match — different methodology)
        ratio = np.mean(costs) / det_cost
        assert 0.5 < ratio < 2.0, (
            f"Bootstrap mean ({np.mean(costs):.2f}) too far from "
            f"deterministic ({det_cost:.2f}), ratio={ratio:.2f}"
        )
