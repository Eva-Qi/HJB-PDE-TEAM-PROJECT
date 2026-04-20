"""Tests for statistical bootstrapping module.

REWRITE 2026-04-19 audit deletions:
  TestComputeLogReturns (both):           tests `np.log` and Python arithmetic
  TestBootstrapPaths::
    test_simple_shape / test_block_shape:       numpy allocation
    test_simple_starts_at_S0 / test_block_*:    literal S[:,0]=S0 assignment
    test_block_positive_prices:                 redundant with simple version
    test_reproducibility:                       tests numpy RNG, not bootstrap
"""

import numpy as np
import pytest

from shared.params import DEFAULT_PARAMS
from shared.cost_model import execution_cost
from montecarlo.bootstrap import (
    bootstrap_paths_simple,
    bootstrap_execution_cost,
    generate_synthetic_returns,
)
from montecarlo.strategies import twap_trajectory


class TestBootstrapPaths:
    """Bootstrap path construction — tests the few genuine invariants."""

    def setup_method(self):
        self.returns = generate_synthetic_returns(DEFAULT_PARAMS, n_obs=5000, seed=0)

    def test_simple_positive_prices(self):
        """Bootstrapped paths via exp(log-returns) must be strictly positive.
        If a bug lets paths go negative, downstream cost calc breaks."""
        S = bootstrap_paths_simple(
            self.returns, S0=50.0, n_steps=50, n_paths=1000,
        )
        assert np.all(S > 0), "Bootstrap paths should be positive (log-normal construction)"


class TestBootstrapExecutionCost:

    def test_cost_has_variance(self):
        """Bootstrap costs should be stochastic (nonzero variance)."""
        returns = generate_synthetic_returns(DEFAULT_PARAMS, n_obs=5000, seed=0)
        S = bootstrap_paths_simple(
            returns, S0=DEFAULT_PARAMS.S0,
            n_steps=DEFAULT_PARAMS.N, n_paths=5000,
        )
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        costs = bootstrap_execution_cost(S, x_twap, DEFAULT_PARAMS)
        assert np.std(costs) > 0, "Bootstrap costs should have positive variance"

    def test_mean_cost_reasonable(self):
        """Bootstrap mean cost should be in the same order of magnitude as
        parametric MC. Ratio bound is intentionally wide (0.5-2.0) because
        bootstrap uses different noise from parametric GBM; this is a
        methodology sanity check, not a precise match assertion."""
        returns = generate_synthetic_returns(DEFAULT_PARAMS, n_obs=5000, seed=0)
        S = bootstrap_paths_simple(
            returns, S0=DEFAULT_PARAMS.S0,
            n_steps=DEFAULT_PARAMS.N, n_paths=10000,
        )
        x_twap = twap_trajectory(DEFAULT_PARAMS)
        costs = bootstrap_execution_cost(S, x_twap, DEFAULT_PARAMS)
        det_cost = execution_cost(x_twap, DEFAULT_PARAMS)
        ratio = np.mean(costs) / det_cost
        assert 0.5 < ratio < 2.0, (
            f"Bootstrap mean ({np.mean(costs):.2f}) too far from "
            f"deterministic ({det_cost:.2f}), ratio={ratio:.2f}"
        )
