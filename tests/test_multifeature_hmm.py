"""Tests for multi-feature (bivariate) HMM extension and sentiment alignment helper.

All tests use synthetic data — no dependency on data/marketpsych_btc_sentiment.csv.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import extensions.regime as regime_mod
from extensions.regime import (
    RegimeParams,
    align_sentiment_to_returns_bar,
    fit_hmm,
)


# ═══════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════

def _make_1d_returns(n: int = 500, seed: int = 0) -> np.ndarray:
    """Simple 1-D log-return series with mild regime structure."""
    rng = np.random.default_rng(seed)
    # Regime 0: low vol; Regime 1: high vol
    states = rng.integers(0, 2, size=n)
    sigmas = np.where(states == 0, 0.001, 0.003)
    returns = rng.normal(0, sigmas)
    return returns


def _make_2d_features_strong_separation(n: int = 600, seed: int = 42) -> np.ndarray:
    """2-D (log_return, sentiment) with strong bimodal separation in both dims.

    Regime 0: low vol returns + negative sentiment (~-1)
    Regime 1: high vol returns + positive sentiment (~+1)

    With a clean 50/50 split and 3x vol difference, the HMM should reliably
    find two distinct regimes.
    """
    rng = np.random.default_rng(seed)
    half = n // 2

    # Block structure: first half = regime 0, second half = regime 1
    ret_0 = rng.normal(0.0, 0.001, half)
    ret_1 = rng.normal(0.0, 0.003, n - half)
    sent_0 = rng.normal(-1.0, 0.1, half)
    sent_1 = rng.normal(+1.0, 0.1, n - half)

    returns = np.concatenate([ret_0, ret_1])
    sentiment = np.concatenate([sent_0, sent_1])
    return np.column_stack([returns, sentiment])


def _make_2d_features_sentiment_drives_separation(
    n: int = 800, seed: int = 7
) -> np.ndarray:
    """Sentiment strongly bimodal; returns nearly iid across regimes.

    Used to verify that multivariate HMM picks up regime separation
    that univariate (returns-only) misses.
    """
    rng = np.random.default_rng(seed)
    half = n // 2

    # Returns nearly identical in both regimes
    ret_0 = rng.normal(0.0, 0.001, half)
    ret_1 = rng.normal(0.0, 0.0012, n - half)

    # Sentiment very different
    sent_0 = rng.normal(-2.0, 0.15, half)
    sent_1 = rng.normal(+2.0, 0.15, n - half)

    returns = np.concatenate([ret_0, ret_1])
    sentiment = np.concatenate([sent_0, sent_1])
    return np.column_stack([returns, sentiment])


# ═══════════════════════════════════════════════════════════════
# Test 1: 1-D backward compatibility
# ═══════════════════════════════════════════════════════════════

def test_fit_hmm_1d_returns_unchanged_behavior():
    """1-D returns path: output shape and RegimeParams fields unchanged."""
    returns = _make_1d_returns(n=300)
    regimes, states = fit_hmm(returns)

    assert len(regimes) == 2
    assert states.shape == (300,)
    assert set(np.unique(states)).issubset({0, 1})

    labels = {r.label for r in regimes}
    assert "risk_on" in labels
    assert "risk_off" in labels

    for r in regimes:
        assert isinstance(r, RegimeParams)
        assert r.sigma > 0
        assert r.gamma > 0
        assert r.eta > 0
        assert 0.0 <= r.probability <= 1.0
        assert r.state_vol > 0
        assert r.state_abs_ret > 0

    # risk_off should have higher sigma multiplier than risk_on
    risk_on = next(r for r in regimes if r.label == "risk_on")
    risk_off = next(r for r in regimes if r.label == "risk_off")
    assert risk_off.sigma > risk_on.sigma


# ═══════════════════════════════════════════════════════════════
# Test 2: 2-D features — valid regime fit
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not regime_mod._HAS_HMMLEARN,
    reason="hmmlearn not installed",
)
def test_fit_hmm_2d_features_returns_valid_regimes():
    """2-D features with strong separation: two regimes with sensible multipliers."""
    feat = _make_2d_features_strong_separation(n=600)
    regimes, states = fit_hmm(features=feat)

    assert len(regimes) == 2
    assert states.shape == (600,)
    assert set(np.unique(states)).issubset({0, 1})

    # Both regimes must be populated
    counts = np.bincount(states, minlength=2)
    assert counts[0] > 0 and counts[1] > 0

    risk_on = next(r for r in regimes if r.label == "risk_on")
    risk_off = next(r for r in regimes if r.label == "risk_off")

    # risk_off sigma should be > 1 (above average vol)
    assert risk_off.sigma > 1.0, f"risk_off.sigma={risk_off.sigma:.3f}, expected >1.0"
    # risk_on sigma should be < 1
    assert risk_on.sigma < 1.0, f"risk_on.sigma={risk_on.sigma:.3f}, expected <1.0"

    # Spread should be detectable (at least 5% apart in sigma)
    spread = (risk_off.sigma - risk_on.sigma) / max(risk_on.sigma, 1e-9)
    assert spread > 0.05, f"sigma spread={spread:.3f}, expected >5%"


# ═══════════════════════════════════════════════════════════════
# Test 3: 2-D without hmmlearn → NotImplementedError
# ═══════════════════════════════════════════════════════════════

def test_fit_hmm_2d_rejects_without_hmmlearn():
    """Multi-feature path raises NotImplementedError when hmmlearn is absent."""
    feat = _make_2d_features_strong_separation(n=200)

    with patch.object(regime_mod, "_HAS_HMMLEARN", False):
        with pytest.raises(NotImplementedError) as exc_info:
            fit_hmm(features=feat)

    msg = str(exc_info.value)
    assert "hmmlearn" in msg.lower()
    assert "pip install hmmlearn" in msg


# ═══════════════════════════════════════════════════════════════
# Test 4: Multivariate beats univariate when sentiment drives separation
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not regime_mod._HAS_HMMLEARN,
    reason="hmmlearn not installed",
)
def test_fit_hmm_2d_both_features_matter():
    """Sentiment-driven separation: multivariate HMM finds wider sigma spread.

    Synthetic data where returns are nearly iid but sentiment is strongly
    bimodal. The univariate HMM (returns only) should find weak regimes
    (small sigma-multiplier spread); the bivariate HMM should find stronger
    regimes because sentiment provides the discriminating signal.
    """
    feat = _make_2d_features_sentiment_drives_separation(n=800)
    returns_only = feat[:, 0]

    # Univariate fit
    uni_regimes, _ = fit_hmm(returns_only)
    uni_on = next(r for r in uni_regimes if r.label == "risk_on")
    uni_off = next(r for r in uni_regimes if r.label == "risk_off")
    uni_spread = uni_off.sigma - uni_on.sigma

    # Multivariate fit (returns + sentiment)
    multi_regimes, _ = fit_hmm(features=feat)
    multi_on = next(r for r in multi_regimes if r.label == "risk_on")
    multi_off = next(r for r in multi_regimes if r.label == "risk_off")
    multi_spread = multi_off.sigma - multi_on.sigma

    # Bivariate should produce at least as large a spread; sentiment helps
    assert multi_spread >= uni_spread, (
        f"Expected multivariate spread ({multi_spread:.4f}) >= "
        f"univariate spread ({uni_spread:.4f}). "
        "Bivariate HMM should leverage sentiment to amplify regime separation."
    )


# ═══════════════════════════════════════════════════════════════
# Test 5: Alignment — basic forward-fill
# ═══════════════════════════════════════════════════════════════

def _make_returns_df_and_sentiment_df():
    """5-min returns over 2 days, daily sentiment."""
    # 2 days × 288 bars/day (5-min in 24h) = 576 rows
    start = pd.Timestamp("2025-01-10 00:00:00", tz="UTC")
    timestamps = pd.date_range(start, periods=576, freq="5min")

    rng = np.random.default_rng(99)
    returns_df = pd.DataFrame({
        "timestamp": timestamps,
        "log_return": rng.normal(0, 0.001, 576),
    })

    # Daily sentiment: one value per calendar day
    sentiment_df = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-10", "2025-01-11"]),
        "sentiment": [0.3, 0.7],
    })
    return returns_df, sentiment_df


def test_align_sentiment_forward_fill():
    """Output is (T, 2); column 0 matches returns; column 1 is ffilled sentiment."""
    returns_df, sentiment_df = _make_returns_df_and_sentiment_df()

    out = align_sentiment_to_returns_bar(
        returns_df, sentiment_df,
        returns_col="log_return",
        sentiment_col="sentiment",
        fill="ffill",
    )

    assert out.ndim == 2
    assert out.shape[1] == 2
    assert out.shape[0] == 576  # all rows finite

    # Column 0 should match original returns exactly
    np.testing.assert_array_almost_equal(
        out[:, 0], returns_df["log_return"].values, decimal=12
    )

    # Day 1 bars (first 288): sentiment should be 0.3
    assert np.allclose(out[:288, 1], 0.3), (
        f"Expected sentiment=0.3 for day 1, got min={out[:288,1].min():.3f}"
    )
    # Day 2 bars (next 288): sentiment should be 0.7
    assert np.allclose(out[288:, 1], 0.7), (
        f"Expected sentiment=0.7 for day 2, got min={out[288:,1].min():.3f}"
    )


# ═══════════════════════════════════════════════════════════════
# Test 6: Alignment — tz-aware returns with tz-naive sentiment
# ═══════════════════════════════════════════════════════════════

def test_align_sentiment_handles_tz_aware_returns():
    """UTC-aware returns + tz-naive sentiment dates: alignment must not error."""
    start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start, periods=200, freq="5min")

    rng = np.random.default_rng(11)
    returns_df = pd.DataFrame({
        "timestamp": timestamps,
        "log_return": rng.normal(0, 0.001, 200),
    })

    # tz-naive daily sentiment
    sentiment_df = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-01", "2025-03-02"]),
        "sentiment": [-0.5, 0.5],
    })

    out = align_sentiment_to_returns_bar(returns_df, sentiment_df)

    assert out.ndim == 2
    assert out.shape[1] == 2
    assert np.all(np.isfinite(out))


# ═══════════════════════════════════════════════════════════════
# Test 7: Alignment — NaN returns are dropped
# ═══════════════════════════════════════════════════════════════

def test_align_sentiment_drops_rows_with_nan():
    """Rows with NaN in either column are removed; output is fully finite."""
    start = pd.Timestamp("2025-02-01 00:00:00", tz="UTC")
    n = 300
    timestamps = pd.date_range(start, periods=n, freq="5min")

    rng = np.random.default_rng(77)
    log_returns = rng.normal(0, 0.001, n).tolist()
    # Introduce 20 NaN values at known positions
    nan_positions = list(range(0, 20))
    for i in nan_positions:
        log_returns[i] = float("nan")

    returns_df = pd.DataFrame({
        "timestamp": timestamps,
        "log_return": log_returns,
    })

    sentiment_df = pd.DataFrame({
        "date": pd.to_datetime(["2025-02-01", "2025-02-02"]),
        "sentiment": [0.2, 0.8],
    })

    out = align_sentiment_to_returns_bar(returns_df, sentiment_df)

    # Output should have fewer rows than input
    assert out.shape[0] < n
    assert out.shape[0] == n - len(nan_positions)

    # All remaining rows must be finite
    assert np.all(np.isfinite(out)), "Output contains non-finite values after alignment."
