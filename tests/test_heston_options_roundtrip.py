"""Round-trip test for Q-measure Heston calibration from synthetic IV surface.

Strategy:
    1. Start with known Heston params (κ=2, θ=0.08, ξ=0.5, ρ=-0.6, v0=0.08).
    2. Use fft_call_price() to generate call prices for 10 strikes × 3 expiries.
    3. Invert each price to BS implied volatility via _bs_iv().
    4. Build a synthetic option_chain DataFrame.
    5. Run calibrate_heston_from_options on the synthetic chain.
    6. Assert recovered params are within 20% of true values.

Why this matters vs spot-based round-trip:
    The spot-based calibrator (calibrate_heston_from_spot) cannot reliably
    recover κ or ρ: κ clips to ceiling (20.0 when true=2.0) and ρ returns
    noise.  IV-surface calibration is a mathematically well-posed inversion
    — for clean synthetic data it should achieve tight recovery.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from extensions.heston import (
    HestonParams,
    _bs_iv,
    calibrate_heston_from_options,
    fft_call_price,
)

# ---------------------------------------------------------------------------
# True parameters for the round-trip test
# ---------------------------------------------------------------------------

TRUE_KAPPA = 2.0
TRUE_THETA = 0.08
TRUE_XI = 0.5
TRUE_RHO = -0.6
TRUE_V0 = 0.08

TRUE_PARAMS = HestonParams(
    kappa=TRUE_KAPPA,
    theta=TRUE_THETA,
    xi=TRUE_XI,
    rho=TRUE_RHO,
    v0=TRUE_V0,
)

# Feller: 2*2.0*0.08 = 0.32 > 0.25 = 0.5² — satisfied.

S0 = 76000.0     # approximate BTC spot
R = 0.0
Q = 0.0

# Expiries: 30, 60, 90 calendar days
EXPIRIES_DAYS = [30, 60, 90]
EXPIRIES_YEARS = [d / 365.25 for d in EXPIRIES_DAYS]

# Strikes: 10 strikes spanning ±30% of spot (covers delta 0.10 – 0.90)
STRIKES = np.array([
    S0 * m for m in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
])

TOLERANCE = 0.20   # 20% relative error


# ---------------------------------------------------------------------------
# Build synthetic option chain once (module-level fixture)
# ---------------------------------------------------------------------------

def _build_synthetic_chain() -> pd.DataFrame:
    """Generate synthetic IV surface using fft_call_price with TRUE_PARAMS."""
    rows = []
    for T in EXPIRIES_YEARS:
        _, fft_prices, _ = fft_call_price(
            S0, S0, R, Q, T,
            alpha=1.0, N=2**12, B=1000.0,
            kappa=TRUE_KAPPA, theta=TRUE_THETA, xi=TRUE_XI,
            rho=TRUE_RHO, v0=TRUE_V0,
        )
        fft_strikes = None  # we re-call below to get the strike grid
        # Re-call to get actual strike array
        fft_strikes_arr, fft_prices_arr, _ = fft_call_price(
            S0, S0, R, Q, T,
            alpha=1.0, N=2**12, B=1000.0,
            kappa=TRUE_KAPPA, theta=TRUE_THETA, xi=TRUE_XI,
            rho=TRUE_RHO, v0=TRUE_V0,
        )

        log_grid = np.log(fft_strikes_arr)

        for K in STRIKES:
            # Interpolate model call price at this strike
            price = float(np.interp(np.log(K), log_grid, fft_prices_arr))

            # Invert to IV
            iv = _bs_iv(price, S0, K, T, R, Q, is_call=True)
            if iv is None:
                continue  # skip deep OTM / ITM that can't be inverted

            rows.append({
                "instrument_name": f"SYNTH-{int(K)}-C",
                "kind": "C",
                "strike": K,
                "expiry_date": f"T+{int(T*365.25)}d",
                "T": T,
                "mark_iv": iv,
                "bid_iv": None,
                "ask_iv": None,
                "mark_price": price / S0,   # in "BTC" units (price / spot)
                "open_interest": 100.0,
                "underlying_price": S0,
            })

    return pd.DataFrame(rows)


_SYNTHETIC_CHAIN = _build_synthetic_chain()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHestonOptionsRoundTrip:
    """Synthetic IV surface → calibrate → check recovery within 20%.

    All assertions use recovered / true ratio to measure relative error.
    The 20% tolerance is MUCH tighter than the spot-based round-trip which
    cannot recover κ at all (clips to 20.0) or ρ (returns sign-noise).
    """

    @pytest.fixture(scope="class", autouse=True)
    def calibrate(self, request):
        """Run calibration once for the whole class."""
        chain = _SYNTHETIC_CHAIN.copy()
        assert len(chain) >= 10, (
            f"Synthetic chain has only {len(chain)} rows — IV inversion "
            f"failed for too many strikes."
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = calibrate_heston_from_options(
                chain,
                underlying_price=S0,
                r=R,
                q=Q,
                n_starts=5,
                delta_filter=(0.05, 0.95),   # wide filter for synthetic chain
                seed=42,
            )

        request.cls.recovered = params
        request.cls.warnings_issued = [str(x.message) for x in w]

    # ------------------------------------------------------------------ #
    # Individual parameter recovery assertions                           #
    # ------------------------------------------------------------------ #

    def test_recovers_kappa_within_20pct(self):
        """True κ=2.0 → recovered κ within 20%.

        This is the PRIMARY regression test for Q-measure calibration.
        The spot-based calibrator clips κ to ceiling (20.0); this test
        verifies the IV-surface calibrator recovers it tightly.
        """
        rec = self.recovered.kappa
        rel_err = abs(rec - TRUE_KAPPA) / TRUE_KAPPA
        assert rel_err < TOLERANCE, (
            f"kappa: true={TRUE_KAPPA:.2f}, recovered={rec:.4f}, "
            f"rel_err={rel_err*100:.1f}% > 20% tolerance. "
            f"Q-measure calibration failed to recover mean-reversion speed."
        )

    def test_recovers_theta_within_20pct(self):
        """True θ=0.08 → recovered θ within 20%."""
        rec = self.recovered.theta
        rel_err = abs(rec - TRUE_THETA) / TRUE_THETA
        assert rel_err < TOLERANCE, (
            f"theta: true={TRUE_THETA:.4f}, recovered={rec:.4f}, "
            f"rel_err={rel_err*100:.1f}% > 20%"
        )

    def test_recovers_xi_within_20pct(self):
        """True ξ=0.5 → recovered ξ within 20%."""
        rec = self.recovered.xi
        rel_err = abs(rec - TRUE_XI) / TRUE_XI
        assert rel_err < TOLERANCE, (
            f"xi: true={TRUE_XI:.3f}, recovered={rec:.4f}, "
            f"rel_err={rel_err*100:.1f}% > 20%"
        )

    def test_recovers_rho_within_20pct(self):
        """True ρ=-0.6 → recovered ρ within 20%.

        This is the SECONDARY regression test.  The spot-based calibrator
        returns ρ as noise (no reliable sign recovery).  The IV-surface
        skew directly encodes ρ — recovery should be tight.
        """
        rec = self.recovered.rho
        rel_err = abs(rec - TRUE_RHO) / abs(TRUE_RHO)
        assert rel_err < TOLERANCE, (
            f"rho: true={TRUE_RHO:.3f}, recovered={rec:.4f}, "
            f"rel_err={rel_err*100:.1f}% > 20%"
        )

    def test_recovers_v0_within_20pct(self):
        """True v0=0.08 → recovered v0 within 20%."""
        rec = self.recovered.v0
        rel_err = abs(rec - TRUE_V0) / TRUE_V0
        assert rel_err < TOLERANCE, (
            f"v0: true={TRUE_V0:.4f}, recovered={rec:.4f}, "
            f"rel_err={rel_err*100:.1f}% > 20%"
        )

    def test_rho_correct_sign(self):
        """Recovered ρ must have the correct (negative) sign.

        A wrong-sign ρ would mean the model has the wrong leverage direction —
        catastrophic for Part D narrative about vol-of-vol skew.
        """
        assert self.recovered.rho < 0, (
            f"rho sign wrong: true={TRUE_RHO:.3f}, recovered={self.recovered.rho:.4f}. "
            f"Q-measure calibration failed on leverage effect direction."
        )

    def test_feller_condition_satisfied(self):
        """Recovered params should satisfy (or be close to) Feller condition.

        The true params satisfy Feller (2*2*0.08=0.32 > 0.25=0.5²).
        Recovery should produce params near the Feller boundary.
        """
        p = self.recovered
        feller_lhs = 2.0 * p.kappa * p.theta
        feller_rhs = p.xi**2
        # Soft check: allow up to 20% violation (optimization may land near boundary)
        assert feller_lhs >= feller_rhs * 0.80, (
            f"Feller badly violated: 2κθ={feller_lhs:.4f}, ξ²={feller_rhs:.4f}. "
            f"Recovered params may be unphysical."
        )

    def test_synthetic_chain_has_sufficient_contracts(self):
        """Synthetic chain must have enough contracts for reliable calibration."""
        n = len(_SYNTHETIC_CHAIN)
        assert n >= 20, (
            f"Synthetic chain only has {n} rows — too few for reliable calibration. "
            f"Check fft_call_price output and _bs_iv inversion."
        )
