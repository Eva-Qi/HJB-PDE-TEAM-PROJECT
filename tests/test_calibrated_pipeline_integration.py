"""Integration test for calibrated_params() end-to-end pipeline.

Uses the real Binance aggTrades CSV for 2026-01-01 (single day, ~407K trades).
The CalibrationResult is computed once at module scope and shared across all tests.

Run:
    pytest tests/test_calibrated_pipeline_integration.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "BTCUSDT-aggTrades-2026-01-01.csv"
FIXTURE_FILE = PROJECT_ROOT / "tests" / "fixtures" / "BTCUSDT-aggTrades-2026-01-01.csv"


def _data_available() -> bool:
    """Return True if the real trade data file is present."""
    return DATA_FILE.exists()


# ---------------------------------------------------------------------------
# Module-level fixture — runs calibrated_params() once for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibration_result():
    """Run calibrated_params() on one full day of BTCUSDT aggTrades.

    Skips the entire module if the data file is not present.
    """
    if not _data_available():
        pytest.skip(
            f"Real trade data not found at {DATA_FILE}. "
            "Cannot run integration tests without live data."
        )

    from calibration.impact_estimator import calibrated_params

    return calibrated_params(trades_path=str(DATA_FILE))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalibratedParamsPipeline:
    """End-to-end integration tests for calibrated_params()."""

    def test_calibrated_params_returns_valid_structure(self, calibration_result):
        """CalibrationResult has all expected fields; sources dict has all keys."""
        result = calibration_result

        # Top-level dataclass fields must all be present and non-None
        assert result.params is not None, "result.params must not be None"
        assert result.sources is not None, "result.sources must not be None"
        assert result.warnings is not None, "result.warnings must be a list (may be empty)"

        # sources must contain all four parameter keys
        required_source_keys = {"sigma", "gamma", "eta", "alpha"}
        missing = required_source_keys - set(result.sources.keys())
        assert not missing, f"sources dict missing keys: {missing}"

        # ACParams must have the critical fields populated and be finite
        p = result.params
        for attr in ("S0", "sigma", "gamma", "eta", "alpha", "lam", "fee_bps"):
            val = getattr(p, attr)
            assert math.isfinite(val), f"params.{attr} = {val} is not finite"

    def test_calibrated_params_sigma_plausible(self, calibration_result):
        """Annualized volatility must be in the crypto-realistic range [0.1, 2.0]
        and must not be NaN."""
        sigma = calibration_result.params.sigma
        assert not math.isnan(sigma), "sigma is NaN"
        assert 0.1 <= sigma <= 2.0, (
            f"sigma={sigma:.4f} outside crypto-plausible range [0.1, 2.0]. "
            "Likely a unit error (per-bar vs annualised)."
        )

    def test_calibrated_params_alpha_in_literature_range(self, calibration_result):
        """Power-law exponent alpha must land in the Almgren-Chriss literature
        range [0.2, 1.5] when fitted via aggregated 1-min/5-min regression."""
        alpha = calibration_result.params.alpha
        assert 0.2 <= alpha <= 1.5, (
            f"alpha={alpha:.4f} outside literature range [0.2, 1.5]. "
            "Aggregated regression may have collapsed or the fallback was used "
            "when it should not have been."
        )

    def test_calibrated_params_eta_not_fallback(self, calibration_result):
        """sources['eta'] must not be 'fallback' for a full day of real data.

        With 407K trades there are >1200 one-minute buckets; the aggregated
        regression should always succeed.  A 'fallback' label here means the
        P0-2 fix regressed.
        """
        eta_source = calibration_result.sources.get("eta")
        assert eta_source != "fallback", (
            f"sources['eta'] == 'fallback' — the aggregated estimator failed "
            f"on real data.  sources={calibration_result.sources}  "
            f"warnings={calibration_result.warnings}"
        )

    def test_calibrated_params_gamma_positive(self, calibration_result):
        """Kyle's lambda (or its fallback) must produce a positive gamma.

        The fallback value is 1e-4 > 0.  If Kyle's lambda is estimated from
        data and is positive, that is even better.  A negative gamma would
        produce negative permanent impact — economically impossible.
        """
        gamma = calibration_result.params.gamma
        assert gamma > 0, (
            f"gamma={gamma} <= 0. Kyle's lambda returned a negative value and "
            "the fallback guard did not fire, or the fallback itself is wrong."
        )

    def test_calibrated_params_includes_fee_bps(self, calibration_result):
        """fee_bps must equal 7.5 (Binance BTCUSDT spot taker rate)."""
        fee_bps = calibration_result.params.fee_bps
        assert fee_bps == pytest.approx(7.5), (
            f"fee_bps={fee_bps}, expected 7.5 (Binance taker).  "
            "Check ACParams constructor in calibrated_params()."
        )


# ---------------------------------------------------------------------------
# Fixture-based smoke test (fast path — uses small fixture CSV)
# ---------------------------------------------------------------------------

class TestCalibratedParamsStructureSmoke:
    """Fast structural smoke test using the small fixture file (5000 trades).

    Numerical assertions are NOT made here because 5000 trades cover only
    ~12 minutes — too few for aggregated regression.  This test verifies
    that the function returns a valid structure even when falling back.
    """

    @pytest.fixture(scope="class")
    def smoke_result(self):
        if not FIXTURE_FILE.exists():
            pytest.skip(f"Fixture not found: {FIXTURE_FILE}")
        from calibration.impact_estimator import calibrated_params
        return calibrated_params(trades_path=str(FIXTURE_FILE))

    def test_smoke_returns_calibration_result(self, smoke_result):
        """calibrated_params() returns CalibrationResult even on tiny dataset."""
        from calibration.impact_estimator import CalibrationResult
        assert isinstance(smoke_result, CalibrationResult)

    def test_smoke_params_has_s0(self, smoke_result):
        """S0 must be a positive price (BTC price > 1000 in 2026-01-01 data)."""
        s0 = smoke_result.params.S0
        assert s0 > 0, f"S0={s0} is not positive"
