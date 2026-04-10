"""Heston stochastic volatility model — characteristic function and calibration.

Directly extracted from MF796 HW2 (verified implementation).
Used in Part D to extend the constant-vol Almgren-Chriss framework.

The Heston model replaces constant sigma with stochastic variance:
    dS = mu*S*dt + sqrt(v)*S*dW_S
    dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
    corr(dW_S, dW_v) = rho
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HestonParams:
    """Heston stochastic volatility parameters.

    Attributes
    ----------
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-run variance level.
    xi : float
        Vol-of-vol (volatility of the variance process).
    rho : float
        Correlation between price and variance Brownian motions.
        Typically negative (leverage effect).
    v0 : float
        Initial variance.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float


# --- Characteristic function (from HW2 cell-3, verified) ---

def heston_cf(
    u: np.ndarray | complex,
    S0: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
) -> np.ndarray | complex:
    """Heston model characteristic function Phi(u) of ln(S_T).

    Phi(u) = omega(u) * exp(-(u^2 + iu)*v0 / (lam*coth(lam*T/2) + kappa - i*rho*xi*u))

    where:
        lam = sqrt(xi^2*(u^2+iu) + (kappa - i*rho*xi*u)^2)
        omega involves cosh/sinh terms and model parameters

    Parameters
    ----------
    u : complex or array of complex
        Transform variable.
    S0, r, q, T : float
        Spot price, risk-free rate, dividend yield, time to maturity.
    kappa, theta, xi, rho, v0 : float
        Heston parameters.

    Returns
    -------
    complex or array of complex
        Characteristic function values.
    """
    i = 1j
    lam = np.sqrt(xi**2 * (u**2 + i * u) + (kappa - i * rho * xi * u) ** 2)

    cosh_half = np.cosh(lam * T / 2)
    sinh_half = np.sinh(lam * T / 2)

    # omega(u)
    numer_exp = (
        i * u * np.log(S0)
        + i * u * (r - q) * T
        + kappa * theta * T * (kappa - i * rho * xi * u) / xi**2
    )
    denom_base = cosh_half + (kappa - i * rho * xi * u) / lam * sinh_half
    w = np.exp(numer_exp) / (denom_base ** (2 * kappa * theta / xi**2))

    # Exponential part
    exp_part = (
        -(u**2 + i * u)
        * v0
        / (lam / np.tanh(lam * T / 2) + kappa - i * rho * xi * u)
    )

    return w * np.exp(exp_part)


def fft_call_price(
    S0: float,
    K_target: float,
    r: float,
    q: float,
    T: float,
    alpha: float,
    N: int,
    B: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Price European calls via FFT under Heston model (Carr-Madan method).

    From HW2 cell-7 (verified). Used for Heston parameter calibration
    in Part D.

    Parameters
    ----------
    S0 : float
        Spot price.
    K_target : float
        Strike price to price.
    r, q : float
        Risk-free rate and dividend yield.
    T : float
        Time to maturity.
    alpha : float
        FFT damping factor (recommend 0.75-1.5).
    N : int
        FFT points (must be power of 2, recommend 2**12).
    B : float
        Upper integration bound in frequency space (recommend 1000).
    kappa, theta, xi, rho, v0 : float
        Heston parameters.

    Returns
    -------
    strikes : np.ndarray
        Strike grid.
    call_prices : np.ndarray
        Call prices on the strike grid.
    price_at_target : float
        Interpolated price at K_target.
    """
    from scipy.fft import fft as scipy_fft

    i_c = 1j
    d_nu = B / N
    d_k = 2 * np.pi / (N * d_nu)
    beta = np.log(S0) - d_k * N / 2

    j_vals = np.arange(1, N + 1)
    nu = (j_vals - 1) * d_nu
    m_vals = np.arange(1, N + 1)
    k = beta + (m_vals - 1) * d_k
    strikes = np.exp(k)

    DF = np.exp(-r * T)
    u_shifted = nu - (alpha + 1) * 1j
    phi_vals = heston_cf(u_shifted, S0, r, q, T, kappa, theta, xi, rho, v0)
    psi = DF * phi_vals / ((alpha + i_c * nu) * (alpha + i_c * nu + 1))

    w = np.full(N, d_nu)
    w[0] = d_nu / 2

    x = np.exp(-i_c * nu * beta) * psi * w
    y = scipy_fft(x)
    call_prices = (np.exp(-alpha * k) / np.pi) * np.real(y)

    price_at_target = float(np.interp(np.log(K_target), k, call_prices))
    return strikes, call_prices, price_at_target


def calibrate_heston(
    market_strikes: np.ndarray,
    market_prices: np.ndarray,
    S0: float,
    r: float,
    T: float,
) -> HestonParams:
    """HW2 reference — options-based calibration.

    Not used in this project; see calibrate_heston_from_spot() instead.

    Minimizes sum of squared pricing errors using scipy.optimize.

    Parameters
    ----------
    market_strikes : np.ndarray
        Observed option strikes.
    market_prices : np.ndarray
        Observed option prices.
    S0, r, T : float
        Market conditions.

    Returns
    -------
    HestonParams
        Calibrated parameters.
    """
    raise NotImplementedError(
        "Part D: implement Heston calibration via scipy.optimize.minimize.\n"
        "Objective: min sum((model_price - market_price)^2)\n"
        "Use fft_call_price() for model prices.\n"
        "Bounds: kappa>0, theta>0, xi>0, -1<rho<1, v0>0\n"
        "Feller condition: 2*kappa*theta > xi^2"
    )


# --- Spot-based calibration (moment-matching on realized variance) ---

def calibrate_heston_from_spot(
    ohlc_df: pd.DataFrame,
    freq_seconds: float = 300.0,
    window_bars: int = 24,
) -> HestonParams:
    """Calibrate Heston parameters from spot OHLC data via moment-matching.

    Uses realized variance computed from Garman-Klass estimator in rolling
    windows, then matches the mean, autocorrelation, and variability of
    the realized variance series to the Heston model's CIR variance process.

    This avoids the need for options data (no Bloomberg access). The approach
    is based on the fact that under Heston, variance follows:
        dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v

    so the time series of realized variance inherits the CIR dynamics.

    Parameters
    ----------
    ohlc_df : pd.DataFrame
        OHLC bars with columns: open, high, low, close.
        Generated by calibration.data_loader.compute_ohlc().
    freq_seconds : float
        Bar frequency in seconds. Default 300 (5-minute bars).
    window_bars : int
        Number of bars per rolling window for realized variance.
        Default 24 (= 2 hours of 5-minute bars).

    Returns
    -------
    HestonParams
        Calibrated Heston parameters.

    Notes
    -----
    Literature fallbacks are applied when estimates are unreasonable:
        - kappa: clipped to [0.5, 20.0], fallback 5.0
        - xi: clipped to [0.1, 3.0], fallback 0.8
        - rho: clipped to [-0.95, 0.95], fallback 0.0 (crypto shows weak leverage)
    """
    o = ohlc_df["open"].values.astype(np.float64)
    h = ohlc_df["high"].values.astype(np.float64)
    l = ohlc_df["low"].values.astype(np.float64)
    c = ohlc_df["close"].values.astype(np.float64)

    # Filter bars where OHLC are all positive
    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o, h, l, c = o[valid], h[valid], l[valid], c[valid]

    n_bars = len(o)
    if n_bars < 2 * window_bars:
        raise ValueError(
            f"Need at least {2 * window_bars} valid OHLC bars, got {n_bars}. "
            f"Reduce window_bars or supply more data."
        )

    # --- Step 1: Per-bar Garman-Klass variance ---
    u = np.log(h / o)  # ln(High/Open)
    d = np.log(l / o)  # ln(Low/Open)
    cc = np.log(c / o)  # ln(Close/Open)
    # Garman-Klass (1980) Eq. 12: 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2
    # Note: u - d = ln(H/O) - ln(L/O) = ln(H/L)
    gk_var_per_bar = 0.5 * (u - d) ** 2 - (2 * np.log(2) - 1) * cc ** 2

    # --- Step 2: Rolling realized variance (annualized) ---
    seconds_per_year = 365.25 * 24 * 3600
    bars_per_year = seconds_per_year / freq_seconds

    # Rolling mean of per-bar variance, then annualize
    gk_series = pd.Series(gk_var_per_bar)
    rv_per_bar = gk_series.rolling(window=window_bars, min_periods=window_bars).mean()
    rv_annualized = (rv_per_bar * bars_per_year).dropna().values

    if len(rv_annualized) < 10:
        raise ValueError(
            f"Only {len(rv_annualized)} realized variance observations after "
            f"rolling window — need at least 10."
        )

    # --- Step 3: Log returns for rho estimation ---
    log_returns = np.log(c[1:] / c[:-1])
    # Align log returns with the rv_annualized series.
    # rv_annualized starts at index (window_bars - 1) in the original bar series,
    # log_returns starts at index 0 (covering bar 0→1).
    # After the rolling window drops NaNs, rv_annualized[i] corresponds to
    # original bar index (window_bars - 1 + i).
    rv_start = window_bars - 1  # first valid rv index in original bars
    # log_returns[j] = log(close[j+1]/close[j]), so it maps to bar index j+1.
    # We need log_returns aligned to the same bar indices as rv_annualized.
    # rv_annualized[i] → original bar (rv_start + i)
    # log_returns[j] → original bar (j + 1), so j = original_bar - 1
    n_rv = len(rv_annualized)
    lr_aligned = log_returns[rv_start: rv_start + n_rv]

    # Trim to matching length (in case of edge effects)
    n_common = min(len(lr_aligned), n_rv)
    lr_aligned = lr_aligned[:n_common]
    rv_aligned = rv_annualized[:n_common]

    # --- Step 4: Estimate theta (long-run variance) ---
    theta = float(np.mean(rv_annualized))
    if theta <= 0:
        warnings.warn(
            "calibrate_heston_from_spot: mean realized variance <= 0, "
            "using fallback theta = 0.04 (20% annualized vol squared)."
        )
        theta = 0.04

    # --- Step 5: Estimate kappa (mean reversion speed) ---
    # From CIR process, the lag-1 autocorrelation of v is:
    #   autocorr(1) ≈ exp(-kappa * dt)
    # where dt is the time step between consecutive rv observations (= 1 bar).
    # dt between consecutive rv observations: each rolling window advances
    # by 1 bar, but the rolling mean smooths over window_bars, so the
    # effective time step for the autocorrelation is 1 bar, not 1 window.
    # However, overlapping windows inflate autocorrelation. To correct,
    # use dt = window_bars * bar_dt (the non-overlapping window span).
    dt_rv = window_bars * freq_seconds / seconds_per_year

    rv_centered = rv_annualized - np.mean(rv_annualized)
    n_rv_pts = len(rv_centered)
    # Use consistent denominator (n) for both autocovariances
    autocov_0 = np.dot(rv_centered, rv_centered) / n_rv_pts
    autocov_1 = np.dot(rv_centered[:-1], rv_centered[1:]) / n_rv_pts

    if autocov_0 > 0 and autocov_1 > 0:
        rho_1 = autocov_1 / autocov_0
        # rho_1 = exp(-kappa * dt_rv) → kappa = -ln(rho_1) / dt_rv
        kappa = -np.log(rho_1) / dt_rv
    else:
        kappa = np.nan

    # Apply bounds and fallback
    if np.isnan(kappa) or kappa <= 0:
        warnings.warn(
            "calibrate_heston_from_spot: kappa estimation failed "
            "(non-positive autocorrelation), using fallback kappa = 5.0."
        )
        kappa = 5.0
    else:
        kappa = float(np.clip(kappa, 0.5, 20.0))

    # --- Step 6: Estimate xi (vol-of-vol) ---
    # For CIR process: Var(v_t) = theta * xi^2 / (2*kappa) in stationary state.
    # So xi = sqrt(2 * kappa * Var(v) / theta)
    var_rv = float(np.var(rv_annualized, ddof=1))

    if var_rv > 0 and theta > 0:
        xi = np.sqrt(2.0 * kappa * var_rv / theta)
    else:
        xi = np.nan

    if np.isnan(xi) or xi <= 0:
        warnings.warn(
            "calibrate_heston_from_spot: xi estimation failed, "
            "using fallback xi = 0.8."
        )
        xi = 0.8
    else:
        xi = float(np.clip(xi, 0.1, 3.0))

    # --- Step 7: Estimate rho (price-variance correlation) ---
    # Correlation between log returns and changes in realized variance
    if n_common >= 3:
        delta_rv = np.diff(rv_aligned)
        lr_for_rho = lr_aligned[1:]  # align with delta_rv
        n_rho = min(len(delta_rv), len(lr_for_rho))
        delta_rv = delta_rv[:n_rho]
        lr_for_rho = lr_for_rho[:n_rho]

        std_lr = np.std(lr_for_rho)
        std_drv = np.std(delta_rv)

        if std_lr > 0 and std_drv > 0:
            rho = float(np.corrcoef(lr_for_rho, delta_rv)[0, 1])
        else:
            rho = np.nan
    else:
        rho = np.nan

    if np.isnan(rho):
        warnings.warn(
            "calibrate_heston_from_spot: rho estimation failed, "
            "using fallback rho = 0.0 (typical for crypto)."
        )
        rho = 0.0
    else:
        rho = float(np.clip(rho, -0.95, 0.95))

    # --- Step 8: v0 (current variance) ---
    v0 = float(rv_annualized[-1])
    if v0 <= 0:
        v0 = theta  # fallback to long-run level

    # --- Step 9: Feller condition check ---
    feller_lhs = 2.0 * kappa * theta
    feller_rhs = xi ** 2
    if feller_lhs < feller_rhs:
        warnings.warn(
            f"calibrate_heston_from_spot: Feller condition violated — "
            f"2*kappa*theta = {feller_lhs:.4f} < xi^2 = {feller_rhs:.4f}. "
            f"Variance process can reach zero; simulation may need reflection "
            f"or truncation at v=0."
        )

    return HestonParams(
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        v0=v0,
    )
