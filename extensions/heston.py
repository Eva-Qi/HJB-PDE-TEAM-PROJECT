"""Heston stochastic volatility model — characteristic function and calibration.

Directly extracted from MF796 HW2 (verified implementation).
Used in Part D to extend the constant-vol Almgren-Chriss framework.

The Heston model replaces constant sigma with stochastic variance:
    dS = mu*S*dt + sqrt(v)*S*dW_S
    dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_v
    corr(dW_S, dW_v) = rho
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    """Calibrate Heston parameters to market option prices.

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
