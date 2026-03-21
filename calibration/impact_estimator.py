"""Estimate market impact parameters from real data.

Three parameters to calibrate:
    gamma (permanent impact): via Kyle's lambda regression
    eta   (temporary impact coefficient): via walk-the-book slippage regression
    alpha (temporary impact exponent): via power-law fit

References:
    Kyle (1985) — lambda = Cov(ΔP, flow) / Var(flow)
    Almgren et al. (2005) — power-law temporary impact
    Cont, Kukanov, Stoikov (2014) — order book price impact
"""

from __future__ import annotations

import numpy as np
from shared.params import ACParams


def estimate_kyle_lambda(
    delta_prices: np.ndarray,
    signed_flows: np.ndarray,
    min_obs: int = 10,
) -> float | None:
    """Estimate Kyle's lambda = Cov(ΔP, signed_flow) / Var(signed_flow).

    This is the permanent impact parameter gamma.
    Uses Welford's online algorithm for numerical stability.

    Algorithm (from QC_Trade_Platform/src/analysis/queue_model.py):
        For each (delta_p, flow) pair:
            n += 1
            delta_dp = dp - mean_dp
            delta_flow = flow - mean_flow
            mean_dp += delta_dp / n
            mean_flow += delta_flow / n
            C += delta_dp * (flow - mean_flow)     # co-moment
            M2 += delta_flow * (flow - mean_flow)  # variance accumulator
        lambda = C / M2  (n cancels)

    Parameters
    ----------
    delta_prices : np.ndarray
        Mid price changes between consecutive trades.
    signed_flows : np.ndarray
        Signed trade quantities (+qty for buys, -qty for sells).
    min_obs : int
        Minimum observations required.

    Returns
    -------
    float or None
        Kyle's lambda estimate, or None if insufficient data.
    """
    if len(delta_prices) < min_obs or len(delta_prices) != len(signed_flows):
        return None

    n = 0
    mean_dp = 0.0
    mean_flow = 0.0
    C = 0.0
    M2_flow = 0.0

    for dp, flow in zip(delta_prices, signed_flows):
        n += 1
        delta_dp = dp - mean_dp
        delta_flow = flow - mean_flow
        mean_dp += delta_dp / n
        mean_flow += delta_flow / n
        # co-moment: use NEW mean_dp but compute NEW delta_flow for M2
        delta_flow2 = flow - mean_flow
        C += delta_dp * delta_flow2
        M2_flow += delta_flow * delta_flow2

    if M2_flow == 0:
        return None

    return C / M2_flow


def estimate_temporary_impact(
    quantities: np.ndarray,
    slippages_bps: np.ndarray,
) -> tuple[float, float]:
    """Fit temporary impact power law: slippage = eta * quantity^alpha.

    Takes log-log regression:
        log(slippage) = log(eta) + alpha * log(quantity)

    Parameters
    ----------
    quantities : np.ndarray
        Order sizes (positive).
    slippages_bps : np.ndarray
        Observed slippage in bps for each order size.

    Returns
    -------
    (eta, alpha) : tuple[float, float]
        Temporary impact coefficient and exponent.
    """
    raise NotImplementedError(
        "P1: implement power-law fit for temporary impact. "
        "Use walk_the_book_slippage() output as input data. "
        "Log-log OLS: np.polyfit(log(qty), log(slippage), 1) → [alpha, log(eta)]"
    )


def estimate_realized_vol(
    prices: np.ndarray,
    freq_seconds: float = 300.0,
    annualize: bool = True,
) -> float:
    """Estimate realized volatility from price series.

    Parameters
    ----------
    prices : np.ndarray
        Price series (e.g., mid prices sampled at regular intervals).
    freq_seconds : float
        Sampling frequency in seconds (default: 300 = 5 minutes).
    annualize : bool
        If True, annualize by sqrt(seconds_per_year / freq_seconds).

    Returns
    -------
    float
        Realized volatility (annualized if requested).
    """
    raise NotImplementedError(
        "P1: implement realized vol estimation. "
        "log_returns = np.diff(np.log(prices)); "
        "vol = std(log_returns) * sqrt(annualization_factor)"
    )


def calibrated_params(
    trades_path: str | None = None,
    ob_path: str | None = None,
) -> ACParams:
    """Build ACParams from calibrated real data.

    This is the FINAL interface P2 and P3 call once P1 is done.
    Until then, use DEFAULT_PARAMS.

    Parameters
    ----------
    trades_path : str, optional
        Path to trade data.
    ob_path : str, optional
        Path to order book snapshots.

    Returns
    -------
    ACParams
        Fully calibrated parameter set.
    """
    raise NotImplementedError(
        "P1: implement full calibration pipeline. "
        "1. Load data via data_loader "
        "2. estimate_kyle_lambda → gamma "
        "3. estimate_temporary_impact → (eta, alpha) "
        "4. estimate_realized_vol → sigma "
        "5. Return ACParams with real values"
    )
