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

from dataclasses import dataclass

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
    quantities = np.asarray(quantities, dtype=np.float64)
    slippages_bps = np.asarray(slippages_bps, dtype=np.float64)

    # Filter to positive values only (log-log requires > 0)
    mask = (quantities > 0) & (slippages_bps > 0)
    q = quantities[mask]
    s = slippages_bps[mask]

    if len(q) < 3:
        raise ValueError(f"Need at least 3 valid (qty, slippage) pairs, got {len(q)}")

    # Log-log OLS: log(slippage) = alpha * log(qty) + log(eta)
    coeffs = np.polyfit(np.log(q), np.log(s), 1)
    alpha = float(coeffs[0])
    eta = float(np.exp(coeffs[1]))

    return eta, alpha


def estimate_temporary_impact_from_trades(
    trades_df,
    n_buckets: int = 20,
    min_trades_per_bucket: int = 100,
) -> tuple[float, float]:
    """Estimate temporary impact from trade-level data (no order book needed).

    Groups trades into quantity buckets and measures the average absolute
    price change per bucket. Then fits the power law.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade data with columns: price, quantity, side.
    n_buckets : int
        Number of quantity percentile buckets.
    min_trades_per_bucket : int
        Minimum trades per bucket to include in regression.

    Returns
    -------
    (eta, alpha) : tuple[float, float]
    """
    import pandas as pd

    df = trades_df.copy()
    df["abs_price_change"] = df["price"].diff().abs()
    df = df.dropna(subset=["abs_price_change"])
    df = df[df["abs_price_change"] > 0]

    # Bucket by quantity percentile
    df["qty_bucket"] = pd.qcut(df["quantity"], q=n_buckets, duplicates="drop")

    grouped = df.groupby("qty_bucket", observed=True).agg(
        avg_qty=("quantity", "mean"),
        avg_impact=("abs_price_change", "mean"),
        count=("quantity", "size"),
    )

    # Filter buckets with enough trades
    grouped = grouped[grouped["count"] >= min_trades_per_bucket]

    if len(grouped) < 3:
        raise ValueError(f"Only {len(grouped)} valid buckets (need >= 3)")

    quantities = grouped["avg_qty"].values
    # Convert price impact to bps: impact / avg_price * 10000
    avg_price = df["price"].mean()
    slippages_bps = grouped["avg_impact"].values / avg_price * 10000

    return estimate_temporary_impact(quantities, slippages_bps)


def estimate_realized_vol_gk(
    ohlc_df,
    freq_seconds: float = 300.0,
    annualize: bool = True,
) -> float:
    """Garman-Klass realized volatility estimator using OHLC bars.

    ~7.4x more efficient than close-to-close: uses Open, High, Low, Close
    from each bar instead of just the closing price. Same data, much
    tighter estimate.

    GK variance per bar:
        σ²_GK = 0.5*(u-d)² - (2ln2-1)*c²
    where:
        u = ln(High/Open), d = ln(Low/Open), c = ln(Close/Open)

    Reference: Garman & Klass (1980), "On the Estimation of Security
    Price Volatilities from Historical Data"

    Parameters
    ----------
    ohlc_df : pd.DataFrame
        OHLC bars with columns: open, high, low, close.
        Use calibration.data_loader.compute_ohlc() to generate.
    freq_seconds : float
        Bar frequency in seconds (default: 300 = 5 minutes).
    annualize : bool
        If True, annualize for crypto 24/7 calendar.

    Returns
    -------
    float
        Garman-Klass realized volatility (annualized if requested).
    """
    o = ohlc_df["open"].values.astype(np.float64)
    h = ohlc_df["high"].values.astype(np.float64)
    l = ohlc_df["low"].values.astype(np.float64)
    c = ohlc_df["close"].values.astype(np.float64)

    # Filter bars where OHLC are all positive
    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o, h, l, c = o[valid], h[valid], l[valid], c[valid]

    if len(o) < 2:
        raise ValueError("Need at least 2 valid OHLC bars")

    u = np.log(h / o)  # ln(High/Open)
    d = np.log(l / o)  # ln(Low/Open)
    cc = np.log(c / o)  # ln(Close/Open)

    # Garman-Klass per-bar variance
    gk_var_per_bar = 0.5 * (u - d)**2 - (2 * np.log(2) - 1) * cc**2

    # Average variance per bar
    vol_per_bar = np.sqrt(np.mean(gk_var_per_bar))

    if annualize:
        seconds_per_year = 365.25 * 24 * 3600
        bars_per_year = seconds_per_year / freq_seconds
        vol_per_bar *= np.sqrt(bars_per_year)

    return float(vol_per_bar)


def estimate_realized_vol_rs(
    ohlc_df,
    freq_seconds: float = 300.0,
    annualize: bool = True,
) -> float:
    """Rogers-Satchell realized volatility estimator.

    Unlike Garman-Klass, RS does NOT assume zero drift — it handles
    trending markets correctly. No overnight gap term, so it's naturally
    suited for 24/7 crypto and intraday bars.

    RS variance per bar:
        σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)

    Use alongside GK as a robustness check. If RS and GK differ by >10%,
    the market is likely trending and GK is overestimating.

    Reference: Rogers & Satchell (1991), "Estimating Variance From
    High, Low and Closing Prices"

    Parameters
    ----------
    ohlc_df : pd.DataFrame
        OHLC bars with columns: open, high, low, close.
    freq_seconds : float
        Bar frequency in seconds (default: 300 = 5 minutes).
    annualize : bool
        If True, annualize for crypto 24/7 calendar.

    Returns
    -------
    float
        Rogers-Satchell realized volatility (annualized if requested).
    """
    o = ohlc_df["open"].values.astype(np.float64)
    h = ohlc_df["high"].values.astype(np.float64)
    l = ohlc_df["low"].values.astype(np.float64)
    c = ohlc_df["close"].values.astype(np.float64)

    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o, h, l, c = o[valid], h[valid], l[valid], c[valid]

    if len(o) < 2:
        raise ValueError("Need at least 2 valid OHLC bars")

    # Rogers-Satchell per-bar variance (drift-robust)
    rs_var_per_bar = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    # Clip negative values (can occur from microstructure noise)
    rs_var_per_bar = np.clip(rs_var_per_bar, 0, None)

    vol_per_bar = np.sqrt(np.mean(rs_var_per_bar))

    if annualize:
        seconds_per_year = 365.25 * 24 * 3600
        bars_per_year = seconds_per_year / freq_seconds
        vol_per_bar *= np.sqrt(bars_per_year)

    return float(vol_per_bar)


@dataclass
class CalibrationResult:
    """Calibrated parameters with metadata on estimation quality.

    Attributes
    ----------
    params : ACParams
        Calibrated parameter set.
    sources : dict[str, str]
        Maps parameter name to its source:
        "estimated" = from real data, "fallback" = literature default.
    warnings : list[str]
        Any warnings generated during calibration.
    sigma_rs : float | None
        Rogers-Satchell vol estimate for robustness comparison.
    """

    params: ACParams
    sources: dict  # e.g. {"sigma": "estimated", "gamma": "estimated", "eta": "fallback", "alpha": "fallback"}
    warnings: list
    sigma_rs: float | None = None


def calibrated_params(
    trades_path: str = "data/",
    X0: float = 10.0,
    T: float = 1.0 / (365.25 * 24),
    N: int = 50,
    lam: float = 1e-6,
) -> CalibrationResult:
    """Build ACParams from calibrated real Binance data.

    This is the FINAL interface P2 and P3 call once P1 is done.
    Returns CalibrationResult with metadata on which parameters were
    estimated from data vs literature fallback.

    Parameters
    ----------
    trades_path : str
        Path to trade data directory or CSV file.
    X0 : float
        Inventory to liquidate (in BTC). Default 10 BTC.
    T : float
        Execution horizon in years. Default ~1 hour.
    N : int
        Number of time steps.
    lam : float
        Risk aversion parameter.

    Returns
    -------
    CalibrationResult
        Calibrated parameters + metadata. Access params via result.params.
    """
    from calibration.data_loader import load_trades, compute_ohlc

    sources = {}
    warnings = []

    # 1. Load data
    trades = load_trades(trades_path)

    # 2. Realized volatility — Garman-Klass (7.4x more efficient than close-to-close)
    ohlc = compute_ohlc(trades, freq="5min")
    sigma = estimate_realized_vol_gk(ohlc, freq_seconds=300.0, annualize=True)
    sources["sigma"] = "estimated"

    # 2b. Rogers-Satchell for robustness comparison
    sigma_rs = None
    try:
        sigma_rs = estimate_realized_vol_rs(ohlc, freq_seconds=300.0, annualize=True)
        drift_gap = abs(sigma - sigma_rs) / sigma
        if drift_gap > 0.10:
            msg = (f"GK ({sigma:.4f}) and RS ({sigma_rs:.4f}) differ by "
                   f"{drift_gap:.1%} — market may be trending, GK could overestimate")
            warnings.append(msg)
    except ValueError:
        pass

    # 3. Kyle's lambda → gamma (permanent impact)
    trades_sorted = trades.sort_values("timestamp")
    delta_prices = trades_sorted["price"].diff().dropna().values
    signed_flows = (trades_sorted["quantity"] * trades_sorted["side"]).values[1:]
    gamma = estimate_kyle_lambda(delta_prices, signed_flows)
    if gamma is None or gamma <= 0:
        msg = f"Kyle's lambda estimation failed (gamma={gamma}), using fallback 1e-4"
        warnings.append(msg)
        gamma = 1e-4
        sources["gamma"] = "fallback"
    else:
        sources["gamma"] = "estimated"

    # 4. Temporary impact → (eta, alpha)
    try:
        eta, alpha = estimate_temporary_impact_from_trades(trades, n_buckets=20)
        # Sanity check: alpha should be in [0.3, 1.5]
        if alpha < 0.3 or alpha > 1.5:
            msg = f"estimated alpha={alpha:.3f} out of range [0.3, 1.5], using literature fallback"
            warnings.append(msg)
            eta, alpha = 1e-3, 0.6
            sources["eta"] = "fallback"
            sources["alpha"] = "fallback"
        else:
            sources["eta"] = "estimated"
            sources["alpha"] = "estimated"
    except ValueError as e:
        msg = f"temporary impact estimation failed ({e}), using literature fallback"
        warnings.append(msg)
        alpha = 0.5
        eta = sigma * 1e-3
        sources["eta"] = "fallback"
        sources["alpha"] = "fallback"

    # 5. S0 from most recent price
    S0 = float(trades["price"].iloc[-1])

    params = ACParams(
        S0=S0,
        sigma=sigma,
        mu=0.0,
        X0=X0,
        T=T,
        N=N,
        gamma=gamma,
        eta=eta,
        alpha=alpha,
        lam=lam,
    )

    return CalibrationResult(
        params=params,
        sources=sources,
        warnings=warnings,
        sigma_rs=sigma_rs,
    )
