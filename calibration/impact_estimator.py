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


# ── Fallback constants (centralized 2026-04-26) ──────────────────────
# Used when estimation fails (insufficient data, exception, OLS r² out
# of acceptable window). These MUST match
# scripts/walk_forward_validation.py — historically gamma was 1e-4 here
# but 2.5 there, a 25,000× unit mismatch that silently dragged AC
# trajectory toward TWAP whenever per-regime fallback fired.
#
# gamma = 2.5  $/BTC   (Kyle's-λ bar-level ballpark on BTCUSDT)
# eta   = 1e-3 $/BTC^α (Almgren et al. (2005) crypto-adapted)
# alpha = 0.6          (Almgren et al. (2005) crypto-adapted)
# sigma_floor = 0.01   (1% annualised — prevents sinh(0)/sinh(0)=NaN
#                       in AC closed-form when fallback fires)
FALLBACK_GAMMA = 2.5
FALLBACK_ETA = 1e-3
FALLBACK_ALPHA = 0.6
FALLBACK_SIGMA_FLOOR = 0.01


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
    literature constant γ=FALLBACK_GAMMA (2.5 $/BTC).

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
    n_trades : int | None
        Number of trades used for this calibration (per-regime sub-sample
        size, or None for pooled calibration).  Populated by
        ``calibrated_params_per_regime`` so downstream consumers can tell
        whether the result is data-driven or fell back to literature
        constants because the sub-sample was too small.
    """

    params: ACParams
    sources: dict  # e.g. {"sigma": "estimated", "gamma": "aggregated_1min",
    #                       "eta": "fallback", "alpha": "fallback"}.
    # Possible values per key:
    #   sigma : "estimated", "insufficient_data" (floored), "error"
    #   gamma : "aggregated_1min", "aggregated_5min", "tick_level",
    #           "fallback" (FALLBACK_GAMMA), "insufficient_data", "error"
    #   eta   : "aggregated_1min", "aggregated_5min", "trade_level",
    #           "fallback" (FALLBACK_ETA), "insufficient_data", "error"
    #   alpha : same source as eta (set together by impact regression)
    warnings: list
    sigma_rs: float | None = None
    n_trades: int | None = None


# ── Calibration cascade helpers ──────────────────────────────────────
# Extracted 2026-04-26 to remove the duplicate gamma/eta cascade logic
# that lived inline in both ``calibrated_params`` and
# ``calibrated_params_per_regime``.  Historically the duplicated blocks
# silently drifted apart on the FALLBACK_GAMMA constant — a 25,000×
# unit-mismatch bug (audit P0-2).  Routing both call sites through these
# helpers eliminates the underlying smell so future fallback / r²-window
# changes happen in one place.
#
# Behaviour preservation: the helpers are byte-equivalent to the inline
# code they replaced.  Two warning dialects are supported because the
# original cascades emitted different strings:
#
#   * pooled mode (regime_label is None) — used by ``calibrated_params``
#     emits diagnostic accept warnings AND step-transition reject
#     warnings ("trying 5min", "trying trade-level", ...) and the
#     verbose final-fallback message.
#
#   * per-regime mode (regime_label is not None) — used by
#     ``calibrated_params_per_regime`` emits only reject/exception
#     warnings (no accept diagnostics) with the "for regime {label}"
#     suffix and a short "Regime {label}: all gamma methods failed —
#     using literature fallback ..." final message.
#
# Adding a new method to an existing site means appending it to the
# ``methods`` list and (if it is a brand-new method type) extending the
# dispatch table at the top of each helper.


def _estimate_gamma_with_cascade(
    trades_df,
    methods: "list[str]",
    regime_label: "int | None" = None,
) -> "tuple[float, str, list[str]]":
    """Run the gamma (Kyle's lambda) calibration cascade.

    Tries each method in ``methods`` in order, accepting the first that
    passes the per-method acceptance test (γ > 0 and r² ≥ 0.01 for the
    aggregated methods; γ > 0 only for tick-level).  If every method
    fails, returns ``(FALLBACK_GAMMA, "fallback", warnings)``.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade data with ``timestamp``, ``price``, ``quantity``, ``side``.
    methods : list[str]
        Ordered list of method names.  Recognised values:
        ``"aggregated_1min"``, ``"aggregated_5min"``, ``"tick_level"``.
    regime_label : int | None
        When given, the helper uses the per-regime warning style
        (``"for regime {label}"`` suffix, no accept diagnostics).
        When ``None`` (pooled mode), the helper emits the diagnostic
        accept warnings and step-transition reject warnings used by
        ``calibrated_params``.

    Returns
    -------
    (gamma, source_label, warnings) : tuple[float, str, list[str]]
        ``source_label`` is one of the entries in ``methods`` (when a
        method succeeded) or ``"fallback"``.
    """
    warnings: "list[str]" = []
    pooled = regime_label is None

    # Pooled-mode tick-level inputs are computed lazily once, the first
    # time the tick step is reached, so per-regime callers (which never
    # list tick_level) pay nothing for the diff/sort.
    _tick_inputs: "tuple | None" = None

    def _tick_arrays():
        nonlocal _tick_inputs
        if _tick_inputs is None:
            ts = trades_df.sort_values("timestamp")
            dp = ts["price"].diff().dropna().values
            sf = (ts["quantity"] * ts["side"]).values[1:]
            _tick_inputs = (dp, sf)
        return _tick_inputs

    # Step-transition suffix for pooled-mode reject/exception warnings —
    # mirrors the inline strings ("— trying 5min", "— trying trade-level",
    # final → "" because the fallback message is emitted separately).
    pooled_next_suffix = {
        "aggregated_1min": " — trying 5min",
        "aggregated_5min": "",      # inline 5min reject branch is silent
        "tick_level": "",           # tick reject path emits the fallback msg
    }

    for idx, method in enumerate(methods):
        if method in ("aggregated_1min", "aggregated_5min"):
            freq = "1min" if method == "aggregated_1min" else "5min"
            try:
                g, g_diag = estimate_kyle_lambda_aggregated(trades_df, freq=freq)
                if g > 0 and g_diag["r_squared"] >= 0.01:
                    if pooled:
                        if method == "aggregated_1min":
                            warnings.append(
                                f"[gamma {method}] γ={g:.4e} "
                                f"r²={g_diag['r_squared']:.3f} "
                                f"n_buckets={g_diag['n_buckets']}"
                            )
                        else:  # aggregated_5min — no n_buckets, matches inline
                            warnings.append(
                                f"[gamma {method}] γ={g:.4e} "
                                f"r²={g_diag['r_squared']:.3f}"
                            )
                    return g, method, warnings
                else:
                    if pooled:
                        if method == "aggregated_1min":
                            warnings.append(
                                f"[gamma {method}] γ={g:.4e} "
                                f"r²={g_diag['r_squared']:.3f} "
                                f"rejected (negative γ or low R²)"
                                f"{pooled_next_suffix[method]}"
                            )
                        # aggregated_5min reject branch is silent in pooled
                        # mode (matches the original inline behaviour).
                    else:
                        warnings.append(
                            f"[gamma {freq}] γ={g:.4e} "
                            f"r²={g_diag['r_squared']:.3f} "
                            f"rejected for regime {regime_label}"
                        )
            except Exception as exc:
                if pooled:
                    warnings.append(
                        f"[gamma {method}] failed ({exc})"
                        f"{pooled_next_suffix[method]}"
                    )
                else:
                    warnings.append(
                        f"[gamma {freq}] failed ({exc}) for regime {regime_label}"
                    )

        elif method == "tick_level":
            # Tick-level only used by pooled cascade (regime_label is None).
            dp, sf = _tick_arrays()
            g_tick = estimate_kyle_lambda(dp, sf)
            if g_tick is not None and g_tick > 0:
                warnings.append(
                    f"[gamma tick_level] γ={g_tick:.4e} — both aggregated "
                    f"frequencies failed; tick-level is known to be "
                    f"bid-ask-bounce-dominated"
                )
                return g_tick, "tick_level", warnings
            else:
                # Inline behaviour: the tick-level rejection IS the
                # final-fallback warning in pooled mode.  Emit it here
                # and return immediately.
                warnings.append(
                    f"ALL gamma methods failed (tick={g_tick}); using "
                    f"literature fallback γ={FALLBACK_GAMMA} — downstream AC "
                    f"trajectory will be dominated by this magic constant"
                )
                return FALLBACK_GAMMA, "fallback", warnings

        else:
            raise ValueError(
                f"_estimate_gamma_with_cascade: unknown method {method!r}"
            )

    # All listed methods failed.  Per-regime mode emits a short final
    # message; pooled mode without a tick step would have emitted nothing
    # so we mirror the per-regime style only when regime_label is set
    # (pooled callers always include tick_level which handles its own
    # final message above).
    if not pooled:
        warnings.append(
            f"Regime {regime_label}: all gamma methods failed — "
            f"using literature fallback γ={FALLBACK_GAMMA}"
        )
    return FALLBACK_GAMMA, "fallback", warnings


def _estimate_eta_alpha_with_cascade(
    trades_df,
    methods: "list[str]",
    regime_label: "int | None" = None,
) -> "tuple[float, float, str, list[str]]":
    """Run the temporary-impact (eta, alpha) calibration cascade.

    Mirrors :func:`_estimate_gamma_with_cascade` but for the power-law
    temporary-impact regression.  Acceptance test for the aggregated
    methods is ``0.3 ≤ alpha ≤ 1.5`` and ``r² ≥ 0.05``; for trade-level
    the test is just ``0.3 ≤ alpha ≤ 1.5`` (no r² gate, matching the
    inline code in ``calibrated_params``).

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trade data with ``timestamp``, ``price``, ``quantity``, ``side``.
    methods : list[str]
        Ordered list of method names.  Recognised values:
        ``"aggregated_1min"``, ``"aggregated_5min"``, ``"trade_level"``.
    regime_label : int | None
        When given, per-regime warning style; when ``None``, pooled
        warning style.

    Returns
    -------
    (eta, alpha, source_label, warnings) : tuple[float, float, str, list[str]]
        ``source_label`` is one of the entries in ``methods`` (when a
        method succeeded) or ``"fallback"``.
    """
    warnings: "list[str]" = []
    pooled = regime_label is None

    pooled_next_suffix = {
        "aggregated_1min": " — trying 5min",
        "aggregated_5min": " — trying trade-level",
        "trade_level": " — using literature fallback",
    }

    for method in methods:
        if method in ("aggregated_1min", "aggregated_5min"):
            freq = "1min" if method == "aggregated_1min" else "5min"
            try:
                e, a, d = estimate_temporary_impact_aggregated(trades_df, freq=freq)
                if 0.3 <= a <= 1.5 and d["r_squared"] >= 0.05:
                    if pooled:
                        warnings.append(
                            f"[{method}] alpha={a:.3f} "
                            f"r²={d['r_squared']:.3f} "
                            f"n_buckets={d['n_buckets']} "
                            f"p={d['p_value']:.4f}"
                        )
                    return e, a, method, warnings
                else:
                    if pooled:
                        warnings.append(
                            f"[{method}] alpha={a:.3f} "
                            f"r²={d['r_squared']:.3f} "
                            f"out of acceptance window"
                            f"{pooled_next_suffix[method]}"
                        )
                    else:
                        warnings.append(
                            f"[impact {freq}] alpha={a:.3f} "
                            f"r²={d['r_squared']:.3f} "
                            f"out of window for regime {regime_label}"
                        )
            except Exception as exc:
                if pooled:
                    warnings.append(
                        f"[{method}] failed ({exc})"
                        f"{pooled_next_suffix[method]}"
                    )
                else:
                    warnings.append(
                        f"[impact {freq}] failed ({exc}) for regime {regime_label}"
                    )

        elif method == "trade_level":
            # Trade-level only used by pooled cascade.
            try:
                eta_tl, alpha_tl = estimate_temporary_impact_from_trades(
                    trades_df, n_buckets=20
                )
                if 0.3 <= alpha_tl <= 1.5:
                    warnings.append(f"[trade_level] alpha={alpha_tl:.3f} accepted")
                    return eta_tl, alpha_tl, "trade_level", warnings
                else:
                    warnings.append(
                        f"[trade_level] alpha={alpha_tl:.3f} out of range "
                        f"[0.3, 1.5] — using literature fallback"
                    )
            except Exception as exc:
                warnings.append(
                    f"[trade_level] failed ({exc}) — using literature fallback"
                )

        else:
            raise ValueError(
                f"_estimate_eta_alpha_with_cascade: unknown method {method!r}"
            )

    # All listed methods failed → literature fallback.
    if pooled:
        warnings.append(
            "temporary impact: all methods failed, using literature "
            "fallback eta=1e-3 alpha=0.6"
        )
    else:
        warnings.append(
            f"Regime {regime_label}: all impact methods failed — "
            f"using literature fallback eta={FALLBACK_ETA} "
            f"alpha={FALLBACK_ALPHA}"
        )
    return FALLBACK_ETA, FALLBACK_ALPHA, "fallback", warnings


def calibrated_params_per_regime(
    trades_df,
    state_sequence: "np.ndarray",
    bar_timestamps: "pd.DatetimeIndex",
    X0: float = 10.0,
    T: float = 1.0 / (365.25 * 24),
    N: int = 50,
    lam: float = 1e-6,
    _min_trades_per_regime: int = 1000,
) -> "dict[int, CalibrationResult]":
    """Calibrate impact parameters independently for each HMM regime.

    This replaces Yuhao's sigma-multiplier scaling with true regime-
    conditional impact calibration.  The HMM state_sequence is indexed
    to 5-min return bars; trades are tick-level.  For each trade we look
    up which 5-min bar it falls into and assign that bar's regime label.

    Alignment strategy (CRITICAL-1 fix, 2026-04-21):
        Previously this used ``pd.merge_asof`` with tz-aware timestamps
        on both sides.  The DATA_INTEGRITY_AUDIT.md flagged this as
        silently failing because the caller was stripping tz via
        ``pd.DatetimeIndex(series.values)``.  In practice merge_asof
        works once both sides are re-localized inside this function,
        but we now prefer a simpler direct ``np.searchsorted`` lookup:
        each trade is assigned the regime of the most recent bar whose
        start ≤ trade_timestamp.  This is more robust (no tz silent
        coercion, no sort re-key cost) and exposes a true ``n_trades``
        count per regime.  A hard-fail assertion fires if any regime
        receives < ``min_trades_per_regime`` trades (default 1000) so
        downstream consumers cannot silently read fallback constants
        for a sub-sample that was never actually calibrated.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Full trade data with columns: timestamp, price, quantity, side.
        ``timestamp`` must be UTC-aware.
    state_sequence : np.ndarray, shape (T_bars,)
        Integer regime labels (0, 1, …) from fit_hmm Viterbi output,
        aligned to ``bar_timestamps``.
    bar_timestamps : pd.DatetimeIndex
        UTC timestamps of the 5-min return bars, length must equal
        len(state_sequence).  Typically the index of the 5-min OHLC /
        mid-price DataFrame after dropna().
    X0, T, N, lam : float / int
        Passed through to calibrated_params for each sub-sample.
    min_trades_per_regime : int
        Hard floor for per-regime trade count.  If any regime has fewer
        than this many trades, a ValueError is raised rather than
        silently falling back.  Defaults to 1000, which is comfortably
        above the ~200 floor used inside the sub-sample calibrator.

    Returns
    -------
    dict[int, CalibrationResult]
        Keys are unique regime labels from state_sequence.
        Values are CalibrationResult objects; ``n_trades`` records the
        true sub-sample size used for that regime's calibration.
    """
    import pandas as pd

    min_trades_per_regime = int(_min_trades_per_regime)

    state_sequence = np.asarray(state_sequence, dtype=int)
    bar_ts = pd.DatetimeIndex(bar_timestamps)
    if bar_ts.tz is None:
        bar_ts = bar_ts.tz_localize("UTC")
    else:
        bar_ts = bar_ts.tz_convert("UTC")

    if len(bar_ts) != len(state_sequence):
        raise ValueError(
            f"bar_timestamps length ({len(bar_ts)}) != "
            f"state_sequence length ({len(state_sequence)})."
        )

    # Sort bars (they should already be sorted, but be defensive).
    bar_order = np.argsort(bar_ts.view("int64"))
    bar_ts_sorted = bar_ts[bar_order]
    state_sorted = state_sequence[bar_order]

    # Normalize trade timestamps to UTC (handle both tz-aware and tz-naive
    # inputs so the function tolerates callers that strip tz via .values).
    trades_sorted = trades_df.sort_values("timestamp").reset_index(drop=True)
    trade_ts = trades_sorted["timestamp"].copy()
    if trade_ts.dt.tz is None:
        trade_ts = trade_ts.dt.tz_localize("UTC")
    else:
        trade_ts = trade_ts.dt.tz_convert("UTC")
    trades_sorted = trades_sorted.copy()
    trades_sorted["timestamp"] = trade_ts

    # Direct bar assignment via searchsorted on int64 nanoseconds — no
    # merge_asof, no tz surprises.  For each trade, find the largest bar
    # index whose timestamp is ≤ the trade's timestamp.
    bar_ns = bar_ts_sorted.view("int64")
    trade_ns = trade_ts.to_numpy().astype("datetime64[ns]").view("int64")
    # side="right" gives position ABOVE the match; subtract 1 for the last
    # bar_ts ≤ trade_ts.  Trades before bar 0 get idx = -1 → drop later.
    bar_idx = np.searchsorted(bar_ns, trade_ns, side="right") - 1

    # Drop trades that landed before the first bar (no regime defined).
    valid_mask = bar_idx >= 0
    if not valid_mask.all():
        n_dropped = int((~valid_mask).sum())
        # Emit a single informational note (caller's script can inspect it
        # by reading the first regime's warnings list below).
        pre_first_bar_dropped = n_dropped
    else:
        pre_first_bar_dropped = 0

    trades_valid = trades_sorted.loc[valid_mask].copy()
    trades_valid["regime"] = state_sorted[bar_idx[valid_mask]].astype(int)
    merged = trades_valid

    results: dict[int, CalibrationResult] = {}
    unique_regimes = sorted(merged["regime"].unique().tolist())

    # Hard-fail guard: refuse to silently return fallback constants for
    # a regime that never had enough trades to calibrate.  This is the
    # CRITICAL-1 assertion requested by the audit.
    trade_counts = {r: int((merged["regime"] == r).sum()) for r in unique_regimes}
    undersized = {r: n for r, n in trade_counts.items() if n < min_trades_per_regime}
    if undersized:
        raise ValueError(
            "calibrated_params_per_regime: regime(s) with fewer than "
            f"{min_trades_per_regime} trades: {undersized}.  "
            "This indicates genuine data sparsity — widen the calibration "
            "window (e.g. 190-day data) or drop per-regime sub-sampling "
            "in favour of joint calibration.  Fallback constants must not "
            "be silently returned for under-sampled regimes."
        )

    for regime_label in unique_regimes:
        sub = merged[merged["regime"] == regime_label].reset_index(drop=True)

        # Need enough data for calibration to be meaningful
        if len(sub) < 200:
            # Return a fallback CalibrationResult with a documenting warning.
            # NB: sigma must NOT be 0 — AC kappa, sinh(kappa·T) and HJB
            # PDE all blow up at sigma→0.
            fallback_params = ACParams(
                S0=float(trades_df["price"].iloc[-1]),
                sigma=FALLBACK_SIGMA_FLOOR,
                mu=0.0,
                X0=X0,
                T=T,
                N=N,
                gamma=FALLBACK_GAMMA,
                eta=FALLBACK_ETA,
                alpha=FALLBACK_ALPHA,
                lam=lam,
                fee_bps=7.5,
            )
            results[regime_label] = CalibrationResult(
                params=fallback_params,
                sources={
                    "sigma": "insufficient_data_floor",
                    "gamma": "insufficient_data",
                    "eta": "insufficient_data",
                    "alpha": "insufficient_data",
                },
                warnings=[
                    f"Regime {regime_label}: only {len(sub)} trades — "
                    "too few for stable calibration (need ≥ 200). "
                    f"Returned fallback (γ={FALLBACK_GAMMA}, η={FALLBACK_ETA}, "
                    f"α={FALLBACK_ALPHA}, σ_floor={FALLBACK_SIGMA_FLOOR})."
                ],
                sigma_rs=None,
                n_trades=len(sub),
            )
            continue

        # Write sub-sample to a temporary in-memory structure so we can
        # reuse calibrated_params() logic.  We call each estimator directly
        # rather than going through the file-based interface.
        try:
            from calibration.data_loader import compute_ohlc

            ohlc_sub = compute_ohlc(sub, freq="5min")
            sigma_gk = estimate_realized_vol_gk(
                ohlc_sub, freq_seconds=300.0, annualize=True
            )

            sigma_rs_sub = None
            sub_warnings = []
            try:
                sigma_rs_sub = estimate_realized_vol_rs(
                    ohlc_sub, freq_seconds=300.0, annualize=True
                )
            except ValueError:
                pass

            # Kyle's lambda — try 1-min, then 5-min (per-regime cascade).
            gamma_sub, gamma_source, gamma_warns = _estimate_gamma_with_cascade(
                sub,
                methods=["aggregated_1min", "aggregated_5min"],
                regime_label=regime_label,
            )
            sub_warnings.extend(gamma_warns)

            # Temporary impact (per-regime cascade).
            eta_sub, alpha_sub, impact_source, impact_warns = (
                _estimate_eta_alpha_with_cascade(
                    sub,
                    methods=["aggregated_1min", "aggregated_5min"],
                    regime_label=regime_label,
                )
            )
            sub_warnings.extend(impact_warns)

            S0 = float(sub["price"].iloc[-1])
            params_sub = ACParams(
                S0=S0,
                sigma=sigma_gk,
                mu=0.0,
                X0=X0,
                T=T,
                N=N,
                gamma=gamma_sub,
                eta=eta_sub,
                alpha=alpha_sub,
                lam=lam,
                fee_bps=7.5,
            )
            if pre_first_bar_dropped and regime_label == unique_regimes[0]:
                sub_warnings.insert(
                    0,
                    f"{pre_first_bar_dropped} trades dropped: timestamp "
                    "before first 5-min bar (no regime label).",
                )
            results[regime_label] = CalibrationResult(
                params=params_sub,
                sources={
                    "sigma": "estimated",
                    "gamma": gamma_source,
                    "eta": impact_source,
                    "alpha": impact_source,
                },
                warnings=sub_warnings,
                sigma_rs=sigma_rs_sub,
                n_trades=len(sub),
            )

        except Exception as exc:
            fallback_params = ACParams(
                S0=float(trades_df["price"].iloc[-1]),
                sigma=FALLBACK_SIGMA_FLOOR,
                mu=0.0,
                X0=X0,
                T=T,
                N=N,
                gamma=FALLBACK_GAMMA,
                eta=FALLBACK_ETA,
                alpha=FALLBACK_ALPHA,
                lam=lam,
                fee_bps=7.5,
            )
            results[regime_label] = CalibrationResult(
                params=fallback_params,
                sources={k: "error" for k in ("sigma", "gamma", "eta", "alpha")},
                warnings=[
                    f"Regime {regime_label}: calibration raised exception: {exc!r}"
                ],
                sigma_rs=None,
                n_trades=len(sub),
            )

    return results


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
    gamma, gamma_source, gamma_warns = _estimate_gamma_with_cascade(
        trades,
        methods=["aggregated_1min", "aggregated_5min", "tick_level"],
        regime_label=None,
    )
    sources["gamma"] = gamma_source
    warnings.extend(gamma_warns)

    # 4. Temporary impact → (eta, alpha)
    # Cascade: aggregated_1min → aggregated_5min → trade_level → fallback
    eta, alpha, _impact_method, impact_warns = _estimate_eta_alpha_with_cascade(
        trades,
        methods=["aggregated_1min", "aggregated_5min", "trade_level"],
        regime_label=None,
    )
    warnings.extend(impact_warns)
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
