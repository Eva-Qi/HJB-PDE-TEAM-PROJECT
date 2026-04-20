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


def estimate_kyle_lambda_aggregated(
    trades_df,
    freq: str = "1min",
) -> tuple[float, dict]:
    """Estimate Kyle's lambda (permanent impact γ) via time-bucket aggregation.

    Why this replaces tick-level: tick-by-tick `price.diff()` is dominated
    by bid-ask bounce. When a large aggressive buy hits the ask, the NEXT
    trade often hits the bid (mean reversion of best quote), producing
    NEGATIVE empirical Cov(dp, flow) — i.e., a NEGATIVE Kyle lambda,
    which is economically absurd ("buys push price down"). The tick-level
    estimator has been empirically observed to return γ ≈ -0.01 on real
    BTCUSDT data; `calibrated_params()` then silently falls back to the
    literature constant γ=1e-4.

    Aggregating to 1-minute buckets and regressing net_flow vs
    price_change recovers economically-sensible positive γ (~2.5 per BTC
    on BTCUSDT), matching the bar-level figures in PROJECT_AUDIT_REPORT.

    Formula: γ = Cov(Δp, net_flow) / Var(net_flow) where
        Δp         = last_price − first_price (within bucket)
        net_flow   = Σ (quantity × side) within bucket
        side       = +1 if taker buy, -1 if taker sell

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade data with columns: timestamp, price, quantity, side.
    freq : str
        Pandas resample frequency (e.g. "1min", "5min"). Default "1min".

    Returns
    -------
    (gamma, diagnostics) : tuple[float, dict]
        diagnostics contains: n_buckets, r_squared, sign_correct.
    """
    import pandas as pd

    df = trades_df.set_index("timestamp").sort_index()
    df["signed_flow"] = df["quantity"] * df["side"]

    buckets = df.resample(freq).agg(
        net_flow=("signed_flow", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        n_trades=("price", "count"),
    ).dropna()
    buckets = buckets[buckets["n_trades"] > 0]

    dp = (buckets["last_price"] - buckets["first_price"]).values
    flow = buckets["net_flow"].values
    mask = np.isfinite(dp) & np.isfinite(flow)
    dp, flow = dp[mask], flow[mask]

    if len(dp) < 20:
        raise ValueError(
            f"Only {len(dp)} valid buckets after filtering — need ≥ 20."
        )

    var_flow = float(np.var(flow))
    if var_flow == 0.0:
        raise ValueError("Zero variance in net_flow — cannot estimate lambda.")

    cov_dp_flow = float(np.cov(dp, flow, ddof=1)[0, 1])
    gamma = cov_dp_flow / var_flow

    # R² from the regression dp = γ × flow + ε
    corr = float(np.corrcoef(dp, flow)[0, 1])
    r_squared = corr ** 2

    diag = {
        "n_buckets": int(len(dp)),
        "r_squared": r_squared,
        "sign_correct": gamma > 0,  # economically, buys should push price up
        "freq": freq,
    }
    return gamma, diag


def estimate_temporary_impact_aggregated(
    trades_df,
    freq: str = "1min",
) -> tuple[float, float, dict]:
    """Estimate temporary impact via time-bucket aggregation (preferred method).

    Why this replaces trade-level: trade-by-trade `abs_price_change` is
    dominated by bid-ask bounce, not impact from quantity. The log-log
    regression of quantity vs price change then recovers alpha ≈ 0
    (see `estimate_temporary_impact_from_trades` pitfall).

    Aggregating to 1-minute bars and regressing |net_signed_flow| against
    |return| recovers literature-range alpha (0.3-1.0) because:
        1. Bid-ask noise averages out within each bucket
        2. Net flow (not gross quantity) reflects directional impact
        3. Return (not absolute price) is dimensionless

    This matches the approach in `scripts/aggregated_alpha_v2.py` which
    the PROJECT_AUDIT_REPORT cites as giving alpha=0.441 (R²=0.147) on
    1-minute aggregation across 98 days.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade data with columns: timestamp, price, quantity, side.
    freq : str
        Pandas resample frequency (e.g. "1min", "5min"). Default "1min".

    Returns
    -------
    (eta, alpha, diagnostics) : tuple[float, float, dict]
        diagnostics contains: n_buckets, r_squared, p_value, std_err.
    """
    import pandas as pd
    from scipy import stats

    df = trades_df.set_index("timestamp").sort_index()
    df["signed_flow"] = df["quantity"] * df["side"]

    buckets = df.resample(freq).agg(
        net_flow=("signed_flow", "sum"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        n_trades=("price", "count"),
    ).dropna()
    buckets = buckets[buckets["n_trades"] > 0]
    buckets["return"] = (buckets["last_price"] - buckets["first_price"]) / buckets["first_price"]

    abs_flow = buckets["net_flow"].abs().values
    abs_return = buckets["return"].abs().values
    mask = (abs_flow > 0) & (abs_return > 0) & np.isfinite(abs_flow) & np.isfinite(abs_return)
    abs_flow = abs_flow[mask]
    abs_return = abs_return[mask]

    if len(abs_flow) < 20:
        raise ValueError(
            f"Only {len(abs_flow)} valid buckets after filtering — "
            f"need ≥ 20 for a stable regression."
        )

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.log(abs_flow), np.log(abs_return),
    )

    alpha = float(slope)
    eta = float(np.exp(intercept))
    diag = {
        "n_buckets": len(abs_flow),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "freq": freq,
    }
    return eta, alpha, diag


def estimate_temporary_impact_from_trades(
    trades_df,
    n_buckets: int = 20,
    min_trades_per_bucket: int = 100,
) -> tuple[float, float]:
    """Estimate temporary impact from trade-level data (no order book needed).

    WARNING: this trade-level estimator is known to produce alpha ≈ 0
    because bid-ask bounce dominates the per-trade price change.
    `calibrated_params()` uses `estimate_temporary_impact_aggregated()`
    first (1-min aggregation) and only falls back to this function if
    aggregated fails.

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
    # Cascade: aggregated_1min → aggregated_5min → tick-level → fallback
    # Why: tick-level price.diff() is dominated by bid-ask bounce and
    # empirically produces NEGATIVE gamma on real BTCUSDT data (e.g.,
    # 2026-01 → gamma ≈ -0.0113, economically absurd). Bar-level
    # aggregation recovers the positive γ ≈ 2.5 the audit report cites.
    gamma = None
    _gamma_method = None

    # --- 3a. Try aggregated 1-min ---
    try:
        g1, g1_diag = estimate_kyle_lambda_aggregated(trades, freq="1min")
        if g1 > 0 and g1_diag["r_squared"] >= 0.01:
            gamma = g1
            _gamma_method = "aggregated_1min"
            sources["gamma"] = "aggregated_1min"
            warnings.append(
                f"[gamma aggregated_1min] γ={g1:.4e} r²={g1_diag['r_squared']:.3f} "
                f"n_buckets={g1_diag['n_buckets']}"
            )
        else:
            warnings.append(
                f"[gamma aggregated_1min] γ={g1:.4e} r²={g1_diag['r_squared']:.3f} "
                f"rejected (negative γ or low R²) — trying 5min"
            )
    except (ValueError, Exception) as e:
        warnings.append(f"[gamma aggregated_1min] failed ({e}) — trying 5min")

    # --- 3b. Try aggregated 5-min ---
    if gamma is None:
        try:
            g5, g5_diag = estimate_kyle_lambda_aggregated(trades, freq="5min")
            if g5 > 0 and g5_diag["r_squared"] >= 0.01:
                gamma = g5
                _gamma_method = "aggregated_5min"
                sources["gamma"] = "aggregated_5min"
                warnings.append(
                    f"[gamma aggregated_5min] γ={g5:.4e} r²={g5_diag['r_squared']:.3f}"
                )
        except (ValueError, Exception) as e:
            warnings.append(f"[gamma aggregated_5min] failed ({e})")

    # --- 3c. Last resort: tick-level (known to give wrong sign on BTCUSDT) ---
    if gamma is None:
        trades_sorted = trades.sort_values("timestamp")
        delta_prices = trades_sorted["price"].diff().dropna().values
        signed_flows = (trades_sorted["quantity"] * trades_sorted["side"]).values[1:]
        g_tick = estimate_kyle_lambda(delta_prices, signed_flows)
        if g_tick is not None and g_tick > 0:
            gamma = g_tick
            sources["gamma"] = "tick_level"
            warnings.append(
                f"[gamma tick_level] γ={g_tick:.4e} — both aggregated "
                f"frequencies failed; tick-level is known to be "
                f"bid-ask-bounce-dominated"
            )
        else:
            msg = (f"ALL gamma methods failed (tick={g_tick}); using "
                   f"literature fallback γ=1e-4 — downstream AC "
                   f"trajectory will be dominated by this magic constant")
            warnings.append(msg)
            gamma = 1e-4
            sources["gamma"] = "fallback"

    # 4. Temporary impact → (eta, alpha)
    # Cascade: aggregated_1min → aggregated_5min → trade_level → fallback
    eta = alpha = None
    _impact_method = None

    # --- 4a. Try aggregated 1-min ---
    try:
        eta_1, alpha_1, diag_1 = estimate_temporary_impact_aggregated(trades, freq="1min")
        if 0.3 <= alpha_1 <= 1.5 and diag_1["r_squared"] >= 0.05:
            eta, alpha = eta_1, alpha_1
            _impact_method = "aggregated_1min"
            warnings.append(
                f"[aggregated_1min] alpha={alpha_1:.3f} r²={diag_1['r_squared']:.3f} "
                f"n_buckets={diag_1['n_buckets']} p={diag_1['p_value']:.4f}"
            )
        else:
            warnings.append(
                f"[aggregated_1min] alpha={alpha_1:.3f} r²={diag_1['r_squared']:.3f} "
                f"out of acceptance window — trying 5min"
            )
    except (ValueError, Exception) as e:
        warnings.append(f"[aggregated_1min] failed ({e}) — trying 5min")

    # --- 4b. Try aggregated 5-min ---
    if eta is None:
        try:
            eta_5, alpha_5, diag_5 = estimate_temporary_impact_aggregated(trades, freq="5min")
            if 0.3 <= alpha_5 <= 1.5 and diag_5["r_squared"] >= 0.05:
                eta, alpha = eta_5, alpha_5
                _impact_method = "aggregated_5min"
                warnings.append(
                    f"[aggregated_5min] alpha={alpha_5:.3f} r²={diag_5['r_squared']:.3f} "
                    f"n_buckets={diag_5['n_buckets']} p={diag_5['p_value']:.4f}"
                )
            else:
                warnings.append(
                    f"[aggregated_5min] alpha={alpha_5:.3f} r²={diag_5['r_squared']:.3f} "
                    f"out of acceptance window — trying trade-level"
                )
        except (ValueError, Exception) as e:
            warnings.append(f"[aggregated_5min] failed ({e}) — trying trade-level")

    # --- 4c. Fall back to trade-level ---
    if eta is None:
        try:
            eta_tl, alpha_tl = estimate_temporary_impact_from_trades(trades, n_buckets=20)
            if 0.3 <= alpha_tl <= 1.5:
                eta, alpha = eta_tl, alpha_tl
                _impact_method = "trade_level"
                warnings.append(f"[trade_level] alpha={alpha_tl:.3f} accepted")
            else:
                warnings.append(
                    f"[trade_level] alpha={alpha_tl:.3f} out of range [0.3, 1.5] — using literature fallback"
                )
        except (ValueError, Exception) as e:
            warnings.append(f"[trade_level] failed ({e}) — using literature fallback")

    # --- 4d. Literature fallback ---
    if eta is None:
        eta, alpha = 1e-3, 0.6
        _impact_method = "fallback"
        warnings.append("temporary impact: all methods failed, using literature fallback eta=1e-3 alpha=0.6")

    sources["eta"] = _impact_method
    sources["alpha"] = _impact_method

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
        fee_bps=7.5,  # Binance BTCUSDT spot taker fee
    )

    return CalibrationResult(
        params=params,
        sources=sources,
        warnings=warnings,
        sigma_rs=sigma_rs,
    )
