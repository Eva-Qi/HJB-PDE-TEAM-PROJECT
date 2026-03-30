"""Tests for Garman-Klass and Rogers-Satchell volatility estimators."""

import numpy as np
import pandas as pd
import pytest

from calibration.impact_estimator import (
    estimate_realized_vol_gk,
    estimate_realized_vol_rs,
)


def _make_ohlc(n_bars: int = 500, sigma: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC bars from GBM for testing."""
    rng = np.random.default_rng(seed)
    dt = 300.0 / (365.25 * 24 * 3600)  # 5 min in years

    prices_per_bar = 60  # simulate 60 ticks per 5-min bar
    dt_tick = dt / prices_per_bar

    rows = []
    S = 100.0
    for _ in range(n_bars):
        ticks = [S]
        for _ in range(prices_per_bar):
            S = S * np.exp(-0.5 * sigma**2 * dt_tick + sigma * np.sqrt(dt_tick) * rng.standard_normal())
            ticks.append(S)
        rows.append({
            "open": ticks[0],
            "high": max(ticks),
            "low": min(ticks),
            "close": ticks[-1],
        })
    return pd.DataFrame(rows)


class TestGarmanKlass:

    def test_returns_positive_float(self):
        ohlc = _make_ohlc()
        vol = estimate_realized_vol_gk(ohlc)
        assert isinstance(vol, float)
        assert vol > 0

    def test_close_to_true_vol(self):
        """GK should estimate σ within 15% of the true value."""
        true_sigma = 0.3
        ohlc = _make_ohlc(n_bars=2000, sigma=true_sigma, seed=123)
        vol = estimate_realized_vol_gk(ohlc, freq_seconds=300.0, annualize=True)
        rel_err = abs(vol - true_sigma) / true_sigma
        assert rel_err < 0.15, f"GK vol={vol:.4f}, true={true_sigma}, err={rel_err:.2%}"

    def test_higher_vol_gives_higher_estimate(self):
        vol_low = estimate_realized_vol_gk(_make_ohlc(sigma=0.2, seed=0))
        vol_high = estimate_realized_vol_gk(_make_ohlc(sigma=0.5, seed=0))
        assert vol_high > vol_low

    def test_unannualized(self):
        ohlc = _make_ohlc()
        vol_ann = estimate_realized_vol_gk(ohlc, annualize=True)
        vol_raw = estimate_realized_vol_gk(ohlc, annualize=False)
        assert vol_ann > vol_raw  # annualization multiplies by large factor

    def test_min_bars(self):
        ohlc = _make_ohlc(n_bars=1)
        with pytest.raises(ValueError):
            estimate_realized_vol_gk(ohlc)


class TestRogersSatchell:

    def test_returns_positive_float(self):
        ohlc = _make_ohlc()
        vol = estimate_realized_vol_rs(ohlc)
        assert isinstance(vol, float)
        assert vol > 0

    def test_close_to_true_vol(self):
        true_sigma = 0.3
        ohlc = _make_ohlc(n_bars=2000, sigma=true_sigma, seed=123)
        vol = estimate_realized_vol_rs(ohlc, freq_seconds=300.0, annualize=True)
        rel_err = abs(vol - true_sigma) / true_sigma
        assert rel_err < 0.15, f"RS vol={vol:.4f}, true={true_sigma}, err={rel_err:.2%}"

    def test_rs_robust_to_drift(self):
        """RS should handle trending markets better than GK.

        We add drift and check RS doesn't blow up relative to no-drift case.
        """
        rng = np.random.default_rng(42)
        sigma = 0.3
        dt = 300.0 / (365.25 * 24 * 3600)
        n_bars = 1000
        ticks_per_bar = 60
        dt_tick = dt / ticks_per_bar

        # With strong upward drift
        S = 100.0
        rows = []
        drift = 0.5  # 50% annual drift
        for _ in range(n_bars):
            ticks = [S]
            for _ in range(ticks_per_bar):
                S = S * np.exp((drift - 0.5 * sigma**2) * dt_tick
                               + sigma * np.sqrt(dt_tick) * rng.standard_normal())
                ticks.append(S)
            rows.append({"open": ticks[0], "high": max(ticks),
                          "low": min(ticks), "close": ticks[-1]})

        ohlc_drift = pd.DataFrame(rows)
        vol_rs = estimate_realized_vol_rs(ohlc_drift)
        # RS should still be in the ballpark (not blown up by drift)
        assert 0.1 < vol_rs < 1.0, f"RS with drift={drift}: vol={vol_rs:.4f}"

    def test_min_bars(self):
        ohlc = _make_ohlc(n_bars=1)
        with pytest.raises(ValueError):
            estimate_realized_vol_rs(ohlc)


class TestGKvsRS:

    def test_agree_for_zero_drift(self):
        """GK and RS should give similar results when drift ≈ 0."""
        ohlc = _make_ohlc(n_bars=2000, sigma=0.3, seed=99)
        vol_gk = estimate_realized_vol_gk(ohlc)
        vol_rs = estimate_realized_vol_rs(ohlc)
        rel_diff = abs(vol_gk - vol_rs) / vol_gk
        assert rel_diff < 0.15, f"GK={vol_gk:.4f}, RS={vol_rs:.4f}, diff={rel_diff:.2%}"
