"""Integration test for simulate_regime_execution() in montecarlo/sde_engine.py.

Tests Yuhao's regime-conditional MC engine with both control_mode="rule"
and control_mode="pde". Uses synthetic RegimeParams with dimensionless
multipliers as provided by the new extensions/regime.py contract.

Run:
    pytest tests/test_regime_conditional_mc.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.params import DEFAULT_PARAMS
from extensions.regime import regime_aware_params, RegimeParams
from montecarlo.sde_engine import simulate_regime_execution


def _make_regime_params():
    """Build synthetic RegimeParams using dimensionless multipliers."""
    risk_on_regime = RegimeParams(
        label="risk_on",
        sigma=0.8,
        gamma=0.8,
        eta=0.8,
        probability=0.5,
        state_vol=0.0,
        state_abs_ret=0.0,
        state_mean_ret=0.0,
    )
    risk_off_regime = RegimeParams(
        label="risk_off",
        sigma=1.2,
        gamma=1.2,
        eta=1.2,
        probability=0.5,
        state_vol=0.0,
        state_abs_ret=0.0,
        state_mean_ret=0.0,
    )
    return risk_on_regime, risk_off_regime


class TestRegimeConditionalMC:
    """Integration tests for simulate_regime_execution."""

    def _build_inputs(self):
        base_params = DEFAULT_PARAMS
        risk_on_regime, risk_off_regime = _make_regime_params()
        risk_on_params = regime_aware_params(base_params, risk_on_regime)
        risk_off_params = regime_aware_params(base_params, risk_off_regime)

        # First half risk_on (0), second half risk_off (1)
        regime_path = np.zeros(base_params.N, dtype=int)
        regime_path[base_params.N // 2:] = 1

        return base_params, risk_on_params, risk_off_params, regime_path

    def test_rule_control_runs_and_returns_dict(self):
        """simulate_regime_execution with control_mode='rule' should run
        without errors and return a dict with the expected keys."""
        base_params, risk_on_params, risk_off_params, regime_path = self._build_inputs()

        result = simulate_regime_execution(
            regime_path=regime_path,
            base_params=base_params,
            risk_on_params=risk_on_params,
            risk_off_params=risk_off_params,
            random_state=42,
            control_mode="rule",
        )

        assert isinstance(result, dict)
        for key in ("t", "regime_path", "inventory", "price", "cost_path",
                    "trade_rate", "total_cost"):
            assert key in result, f"Missing key '{key}' in result"

    def test_rule_inventory_shape_and_bounds(self):
        """Inventory should start at X0 and be non-negative throughout."""
        base_params, risk_on_params, risk_off_params, regime_path = self._build_inputs()

        result = simulate_regime_execution(
            regime_path=regime_path,
            base_params=base_params,
            risk_on_params=risk_on_params,
            risk_off_params=risk_off_params,
            random_state=42,
            control_mode="rule",
        )

        inv = result["inventory"]
        assert len(inv) == base_params.N + 1, (
            f"inventory length {len(inv)} != N+1={base_params.N + 1}"
        )
        assert inv[0] == pytest.approx(base_params.X0, rel=1e-6), (
            f"inventory[0]={inv[0]} != X0={base_params.X0}"
        )
        assert np.all(inv >= -1e-9), (
            f"inventory has negative values: min={inv.min()}"
        )

    def test_rule_total_cost_is_finite_and_positive(self):
        """Total execution cost under rule control should be finite and positive."""
        base_params, risk_on_params, risk_off_params, regime_path = self._build_inputs()

        result = simulate_regime_execution(
            regime_path=regime_path,
            base_params=base_params,
            risk_on_params=risk_on_params,
            risk_off_params=risk_off_params,
            random_state=42,
            control_mode="rule",
        )

        cost = result["total_cost"]
        assert np.isfinite(cost), f"total_cost={cost} is not finite"
        assert cost > 0, f"total_cost={cost} is not positive"

    def test_regime_path_preserved_in_output(self):
        """Output regime_path should equal input regime_path."""
        base_params, risk_on_params, risk_off_params, regime_path = self._build_inputs()

        result = simulate_regime_execution(
            regime_path=regime_path,
            base_params=base_params,
            risk_on_params=risk_on_params,
            risk_off_params=risk_off_params,
            random_state=42,
            control_mode="rule",
        )

        np.testing.assert_array_equal(result["regime_path"], regime_path)

    def test_wrong_regime_path_length_raises(self):
        """Passing regime_path with wrong length should raise ValueError."""
        base_params, risk_on_params, risk_off_params, _ = self._build_inputs()

        bad_path = np.zeros(base_params.N + 5, dtype=int)
        with pytest.raises(ValueError, match="regime_path length"):
            simulate_regime_execution(
                regime_path=bad_path,
                base_params=base_params,
                risk_on_params=risk_on_params,
                risk_off_params=risk_off_params,
                random_state=42,
                control_mode="rule",
            )

    def test_pde_control_runs_and_returns_dict(self):
        """simulate_regime_execution with control_mode='pde' should run
        without errors using pde_M=100 for speed."""
        base_params, risk_on_params, risk_off_params, regime_path = self._build_inputs()

        result = simulate_regime_execution(
            regime_path=regime_path,
            base_params=base_params,
            risk_on_params=risk_on_params,
            risk_off_params=risk_off_params,
            random_state=42,
            control_mode="pde",
            pde_M=100,
        )

        assert isinstance(result, dict)
        assert np.isfinite(result["total_cost"]), (
            f"PDE total_cost={result['total_cost']} is not finite"
        )
        assert result["total_cost"] > 0, (
            f"PDE total_cost={result['total_cost']} is not positive"
        )

    def test_rule_and_pde_costs_differ(self):
        """Rule and PDE costs should differ — they use different control strategies."""
        base_params, risk_on_params, risk_off_params, regime_path = self._build_inputs()

        result_rule = simulate_regime_execution(
            regime_path=regime_path,
            base_params=base_params,
            risk_on_params=risk_on_params,
            risk_off_params=risk_off_params,
            random_state=42,
            control_mode="rule",
        )

        result_pde = simulate_regime_execution(
            regime_path=regime_path,
            base_params=base_params,
            risk_on_params=risk_on_params,
            risk_off_params=risk_off_params,
            random_state=42,
            control_mode="pde",
            pde_M=100,
        )

        rule_cost = result_rule["total_cost"]
        pde_cost = result_pde["total_cost"]

        # Both should be valid
        assert np.isfinite(rule_cost) and np.isfinite(pde_cost)
        # They should not be identical (different control logic)
        assert rule_cost != pytest.approx(pde_cost, rel=1e-6), (
            f"rule_cost={rule_cost:.4f} equals pde_cost={pde_cost:.4f} — "
            "expected different costs from different control strategies"
        )

    def test_regime_aware_params_o1_multipliers(self):
        """regime_aware_params() should produce O(1) differences between regimes,
        not 8 orders of magnitude. This is the core audit fix."""
        risk_on_regime, risk_off_regime = _make_regime_params()

        p1 = regime_aware_params(DEFAULT_PARAMS, risk_on_regime)
        p2 = regime_aware_params(DEFAULT_PARAMS, risk_off_regime)

        ratio_sigma = p2.sigma / p1.sigma
        ratio_gamma = p2.gamma / p1.gamma
        ratio_eta = p2.eta / p1.eta

        for name, ratio in [("sigma", ratio_sigma), ("gamma", ratio_gamma),
                             ("eta", ratio_eta)]:
            assert 0.3 < ratio < 3.0, (
                f"{name} ratio={ratio:.3f} is not O(1). "
                "regime params should be dimensionless multipliers, not "
                "sigma*1e-8 magic constants."
            )
