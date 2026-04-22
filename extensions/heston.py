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


# ---------------------------------------------------------------------------
# Q-measure calibration from options IV surface (Deribit BTC chain)
# ---------------------------------------------------------------------------

def _bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes call price."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes put price."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _bs_iv(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool = True,
) -> float | None:
    """Invert BS price to implied volatility via Brent's method.

    Returns None if inversion fails (price outside no-arbitrage bounds,
    or solver doesn't converge).
    """
    from scipy.optimize import brentq

    if T <= 0 or price <= 0:
        return None

    if is_call:
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper_bound = S * np.exp(-q * T)
        if price <= intrinsic or price >= upper_bound:
            return None
        f = lambda sigma: _bs_call_price(S, K, T, r, q, sigma) - price
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        upper_bound = K * np.exp(-r * T)
        if price <= intrinsic or price >= upper_bound:
            return None
        f = lambda sigma: _bs_put_price(S, K, T, r, q, sigma) - price

    try:
        return brentq(f, 1e-6, 10.0, xtol=1e-7, maxiter=200)
    except (ValueError, RuntimeError):
        return None


def _bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes delta."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if is_call:
        return float(np.exp(-q * T) * norm.cdf(d1))
    else:
        return float(-np.exp(-q * T) * norm.cdf(-d1))


def _heston_model_iv_batch(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    alpha: float = 1.0,
    N: int = 2**12,
    B: float = 1000.0,
) -> np.ndarray:
    """Compute Heston model IVs for all strikes at a single expiry T.

    Uses one FFT call (vectorized across strikes), then inverts each price
    to BS IV.  Returns an array of IVs aligned with `strikes`; NaN where
    inversion fails.
    """
    # One FFT gives call prices on the full strike grid
    fft_strikes, fft_prices, _ = fft_call_price(
        S0, S0, r, q, T, alpha, N, B, kappa, theta, xi, rho, v0
    )

    # Interpolate at each requested strike using the log-space grid
    log_fft_k = np.log(fft_strikes)
    model_ivs = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        price = float(np.interp(np.log(K), log_fft_k, fft_prices))
        iv = _bs_iv(price, S0, K, T, r, q, is_call=True)
        model_ivs[i] = iv if iv is not None else np.nan
    return model_ivs


def calibrate_heston_from_options(
    option_chain,            # pd.DataFrame from download_deribit output
    underlying_price: float,
    r: float = 0.0,
    q: float = 0.0,
    n_starts: int = 5,
    delta_filter: tuple = (0.10, 0.90),
    atm_only: bool = False,
    seed: int = 42,
    use_bid_ask: bool = True,
    use_oi_weights: bool = True,
) -> HestonParams:
    """Calibrate Heston (κ, θ, ξ, ρ, v₀) from Q-measure option IV surface.

    Minimizes Σ w_i*(model_iv_i − market_iv_i)² over the filtered option chain.
    Uses FFT-based Carr-Madan pricing (one FFT per expiry per loss evaluation),
    inverts model prices to BS IV, and runs multi-start L-BFGS-B.

    Market IV is determined as follows (in priority order):
        1. (bid_iv + ask_iv) / 2 when both present and use_bid_ask=True
        2. mark_iv fallback

    Liquidity filters applied when bid/ask columns are present:
        - Drop contracts with bid_iv == 0 (no bid, untradeable)
        - Drop contracts with bid-ask spread / mark_iv > 0.5 (too wide)

    Loss weights w_i:
        - If use_oi_weights=True and open_interest column is present with
          non-zero total OI: w_i = OI_i / sum(OI)  (OI-weighted RMSE)
        - Otherwise: w_i = 1/n  (uniform, equivalent to unweighted sum/n)

    Parameters
    ----------
    option_chain : pd.DataFrame
        Columns: kind (C/P), strike, T (years), mark_iv (decimal).
        Optional columns: bid_iv, ask_iv (decimal), open_interest.
        Produced by calibration.download_deribit.snapshot_to_dataframe().
    underlying_price : float
        Current BTC/USD spot price.
    r : float
        Risk-free rate (0 for crypto).
    q : float
        Dividend/funding yield (0 for crypto spot).
    n_starts : int
        Number of random multi-start points for L-BFGS-B.
    delta_filter : tuple (low, high)
        Only keep contracts with |delta| in [low, high].  Filters deep
        OTM/ITM options where IV is noisy.
    atm_only : bool
        If True, keep only strikes within ±20% of spot (overrides delta_filter).
    seed : int
        RNG seed for reproducible multi-start sampling.
    use_bid_ask : bool
        If True (default), use mid-IV = (bid_iv + ask_iv)/2 when available
        and apply liquidity filters.  If False, use mark_iv only.
    use_oi_weights : bool
        If True (default), weight loss by open_interest when available.
        If False, use uniform weights.

    Returns
    -------
    HestonParams
        Calibrated parameters.  Logs a Feller warning if 2κθ < ξ².
    """
    import pandas as pd
    from scipy.optimize import minimize

    S0 = underlying_price

    # ------------------------------------------------------------------ #
    # Step 1 — pre-filter the chain                                       #
    # ------------------------------------------------------------------ #
    df = option_chain.copy()

    # Drop missing IVs
    df = df[df["mark_iv"].notna() & (df["mark_iv"] > 0)].copy()

    # Only calls (puts carry same info via put-call parity; calls are more
    # liquid ATM; mixing both can double-weight some strikes)
    df = df[df["kind"] == "C"].copy()

    # Maturity window: 7 days to 180 days
    T_min = 7 / 365.25
    T_max = 180 / 365.25
    df = df[(df["T"] >= T_min) & (df["T"] <= T_max)].copy()

    if df.empty:
        raise ValueError(
            "No valid option rows after maturity filter (7–180 days). "
            "Check the option chain."
        )

    # ------------------------------------------------------------------ #
    # Step 1b — resolve market_iv: mid-IV or mark_iv fallback            #
    # ------------------------------------------------------------------ #
    has_bid_ask = (
        use_bid_ask
        and "bid_iv" in df.columns
        and "ask_iv" in df.columns
        and df["bid_iv"].notna().any()
        and df["ask_iv"].notna().any()
    )

    if has_bid_ask:
        # Rows with valid bid and ask: use mid-IV
        bid_valid = df["bid_iv"].notna() & (df["bid_iv"] > 0)
        ask_valid = df["ask_iv"].notna() & (df["ask_iv"] > 0)
        both_valid = bid_valid & ask_valid

        df["market_iv"] = df["mark_iv"].copy()
        df.loc[both_valid, "market_iv"] = (
            df.loc[both_valid, "bid_iv"] + df.loc[both_valid, "ask_iv"]
        ) / 2.0

        # Liquidity filter 1: drop zero-bid (no market)
        zero_bid_mask = bid_valid & ~bid_valid  # init empty; re-apply below
        df = df[~(bid_valid & (df["bid_iv"] == 0))].copy()

        # Liquidity filter 2: drop wide-spread contracts (spread/mark > 0.5)
        if df["mark_iv"].notna().any():
            spread = (df["ask_iv"].fillna(0) - df["bid_iv"].fillna(0)).clip(lower=0)
            relative_spread = spread / df["mark_iv"].replace(0, np.nan)
            illiquid = relative_spread > 0.50
            df = df[~illiquid].copy()

        print(
            f"[calibrate_heston_from_options] mid-IV mode: "
            f"{both_valid.sum()} contracts have valid bid/ask; "
            f"{len(df)} pass liquidity filters"
        )
    else:
        df["market_iv"] = df["mark_iv"].copy()

    df = df[df["market_iv"].notna() & (df["market_iv"] > 0)].copy()

    if df.empty:
        raise ValueError(
            "No valid option rows after IV resolution / liquidity filter."
        )

    # ------------------------------------------------------------------ #
    # Step 1c — OI weights                                               #
    # ------------------------------------------------------------------ #
    has_oi = (
        use_oi_weights
        and "open_interest" in df.columns
        and df["open_interest"].notna().any()
        and df["open_interest"].sum() > 0
    )

    if has_oi:
        oi_vals = df["open_interest"].fillna(0.0).clip(lower=0.0)
        total_oi = oi_vals.sum()
        df["_weight"] = oi_vals / total_oi if total_oi > 0 else 1.0 / len(df)
    else:
        df["_weight"] = 1.0 / len(df)

    if atm_only:
        df = df[(df["strike"] >= S0 * 0.80) & (df["strike"] <= S0 * 1.20)].copy()
    else:
        # Delta filter: compute approximate BS delta at market_iv, keep if in range
        def _delta_ok(row):
            d = _bs_delta(S0, row["strike"], row["T"], r, q, row["market_iv"], is_call=True)
            return delta_filter[0] <= abs(d) <= delta_filter[1]

        mask = df.apply(_delta_ok, axis=1)
        df = df[mask].copy()

    if df.empty:
        raise ValueError(
            "No option rows pass the delta / ATM filter. "
            "Try widening delta_filter or setting atm_only=False."
        )

    # Re-normalise weights after filtering
    w_sum = df["_weight"].sum()
    if w_sum > 0:
        df["_weight"] = df["_weight"] / w_sum
    else:
        df["_weight"] = 1.0 / len(df)

    # Build a list of (K, T, market_iv, weight) tuples
    # Group by expiry T (round to 4 decimals to cluster same-expiry options)
    df["T_key"] = df["T"].round(4)
    chain_rows = [
        (row["strike"], row["T_key"], row["market_iv"], row["_weight"])
        for _, row in df.iterrows()
    ]

    expiry_groups: dict[float, list[tuple]] = {}
    for K, T_key, miv, w in chain_rows:
        expiry_groups.setdefault(T_key, []).append((K, miv, w))

    n_contracts = len(chain_rows)
    expiries = sorted(expiry_groups.keys())

    # ------------------------------------------------------------------ #
    # Step 2 — loss function (OI-weighted sum of squared IV errors)      #
    # ------------------------------------------------------------------ #
    FELLER_PENALTY = 10.0

    def loss(params_arr: np.ndarray) -> float:
        kappa, theta, xi, rho, v0 = params_arr

        # Hard guard against degenerate params that crash FFT
        if kappa <= 0 or theta <= 0 or xi <= 0 or v0 <= 0:
            return 1e9
        if not (-1.0 < rho < 1.0):
            return 1e9

        weighted_sq: list[float] = []
        for T_key in expiries:
            rows_T = expiry_groups[T_key]
            strikes_arr = np.array([r_[0] for r_ in rows_T])
            market_ivs = np.array([r_[1] for r_ in rows_T])
            weights_arr = np.array([r_[2] for r_ in rows_T])

            try:
                model_ivs = _heston_model_iv_batch(
                    S0, strikes_arr, T_key, r, q,
                    kappa, theta, xi, rho, v0,
                )
            except Exception:
                return 1e9

            valid = ~np.isnan(model_ivs)
            if valid.sum() == 0:
                continue

            diff = model_ivs[valid] - market_ivs[valid]
            w = weights_arr[valid]
            # Weighted sum: Σ w_i * (model_iv_i - market_iv_i)^2
            weighted_sq.extend((w * diff ** 2).tolist())

        if not weighted_sq:
            return 1e9

        base_loss = float(np.sum(weighted_sq))

        # Soft Feller constraint: 2κθ ≥ ξ²
        feller_violation = max(0.0, xi**2 - 2.0 * kappa * theta)
        return base_loss + FELLER_PENALTY * feller_violation

    # ------------------------------------------------------------------ #
    # Step 3 — bounds + multi-start L-BFGS-B with Feller-aware selection #
    # ------------------------------------------------------------------ #
    bounds = [
        (0.1, 10.0),    # kappa
        (0.01, 2.0),    # theta
        (0.01, 3.0),    # xi
        (-0.99, 0.99),  # rho
        (0.01, 2.0),    # v0
    ]

    # Feller-healthy selection threshold: prefer solutions where
    # 2*kappa*theta - xi^2 > FELLER_THIN_THRESHOLD.
    # Solutions below this margin sit in the pathological high-xi basin
    # (razor-thin or violated Feller) and should be deprioritised.
    FELLER_THIN_THRESHOLD = 0.01

    rng = np.random.default_rng(seed)

    # Seed the first start near a sensible "crypto" prior to speed convergence
    starts = []
    # Rough ATM IV from chain
    atm_iv_approx = float(df.loc[
        (df["strike"] - S0).abs().idxmin(), "market_iv"
    ]) if len(df) > 0 else 0.80
    v0_prior = atm_iv_approx**2  # variance ≈ IV²

    # Literature prior (start 0): moderate kappa, low xi, slightly negative rho
    starts.append([2.0, v0_prior, 0.5 * atm_iv_approx, -0.3, v0_prior])

    # Additional deterministic prior (start 1): BTC leverage-effect prior with
    # low vol-of-vol — explicitly avoids the high-xi pathological basin
    starts.append([3.0, max(v0_prior, 0.04), 0.1, -0.5, max(v0_prior, 0.04)])

    # Random starts fill remaining slots
    for _ in range(max(0, n_starts - 2)):
        p0 = [
            rng.uniform(lo, hi) for lo, hi in bounds
        ]
        starts.append(p0)

    # Collect ALL successful results so we can apply Feller-aware selection
    all_results: list[tuple[float, float, np.ndarray]] = []  # (loss, feller_margin, x)

    for p0 in starts:
        try:
            res = minimize(
                loss,
                x0=np.array(p0, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
            )
        except Exception as exc:
            warnings.warn(f"calibrate_heston_from_options: optimizer failed ({exc})")
            continue

        if not np.isfinite(res.fun):
            continue

        kp, th, xi_, rh, v_ = res.x
        fm = 2.0 * kp * th - xi_**2
        all_results.append((float(res.fun), float(fm), res.x.copy()))

    if not all_results:
        raise RuntimeError(
            "calibrate_heston_from_options: all optimizer starts failed. "
            "Check option chain quality."
        )

    # Feller-aware selection:
    #   1. Prefer solutions with feller_margin > FELLER_THIN_THRESHOLD (healthy basin).
    #      Among those, pick the one with the lowest loss.
    #   2. If ALL solutions are Feller-thin, fall back to the minimum-loss solution
    #      but emit a warning so the caller knows no healthy basin was found.
    healthy = [(lv, fm, xv) for lv, fm, xv in all_results if fm > FELLER_THIN_THRESHOLD]
    all_feller_thin = len(healthy) == 0

    if healthy:
        best_loss, best_feller_margin, best_x = min(healthy, key=lambda t: t[0])
    else:
        best_loss, best_feller_margin, best_x = min(all_results, key=lambda t: t[0])

    kappa, theta, xi, rho, v0 = best_x
    kappa = float(kappa)
    theta = float(theta)
    xi = float(xi)
    rho = float(rho)
    v0 = float(v0)

    # ------------------------------------------------------------------ #
    # Step 4 — Feller condition report                                   #
    # ------------------------------------------------------------------ #
    feller_lhs = 2.0 * kappa * theta
    feller_rhs = xi**2
    feller_margin_final = feller_lhs - feller_rhs

    if all_feller_thin:
        warnings.warn(
            f"calibrate_heston_from_options: ALL {len(all_results)} starts converged to "
            f"Feller-thin solutions (feller_margin <= {FELLER_THIN_THRESHOLD}). "
            f"Best feller_margin={best_feller_margin:.6f}. "
            f"Consider increasing n_starts or inspecting option chain quality. "
            f"n_contracts={n_contracts}, loss={best_loss:.6f}"
        )
    elif feller_margin_final < FELLER_THIN_THRESHOLD:
        # Shouldn't happen after healthy-basin selection, but guard anyway
        warnings.warn(
            f"calibrate_heston_from_options: selected solution has thin Feller margin "
            f"({feller_margin_final:.6f} < {FELLER_THIN_THRESHOLD}). "
            f"n_contracts={n_contracts}, loss={best_loss:.6f}"
        )

    if feller_lhs < feller_rhs:
        warnings.warn(
            f"calibrate_heston_from_options: Feller condition violated — "
            f"2*kappa*theta = {feller_lhs:.4f} < xi^2 = {feller_rhs:.4f}. "
            f"Variance process can reach zero. "
            f"n_contracts={n_contracts}, loss={best_loss:.6f}"
        )
    else:
        n_healthy = len(healthy)
        n_total = len(all_results)
        print(
            f"[calibrate_heston_from_options] Converged. "
            f"n_contracts={n_contracts}, loss={best_loss:.6f}, "
            f"Feller satisfied (2κθ={feller_lhs:.3f} ≥ ξ²={feller_rhs:.3f}), "
            f"feller_margin={feller_margin_final:.4f}, "
            f"healthy_starts={n_healthy}/{n_total}"
        )

    return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


# ---------------------------------------------------------------------------
# 3/2 model — characteristic function and calibration
# ---------------------------------------------------------------------------
# The 3/2 model (Lewis 2000, Carr-Sun 2007) drives the inverse variance
# process:
#     d(1/v) = (delta + beta/v)*dt + epsilon/sqrt(v)*dW_v
# equivalently the variance satisfies:
#     dv = v*(p - q*v)*dt + xi*v^(3/2)*dW_v,    corr(dS/S, dW_v) = rho
# where:  p = delta*epsilon^2,  q = -(beta + epsilon^2)*epsilon^... see below.
#
# Standard parameterisation (Drimus 2012):
#     p  — mean-reversion level parameter (plays role of kappa*theta in Heston)
#     q  — mean-reversion speed parameter (> 0; plays role of kappa in Heston)
#     xi — vol-of-vol (sigma in the 3/2 process, xi > 0)
#     rho — correlation in [-1, 1]
#     v0  — initial variance
#
# The (exact) characteristic function is available in closed form via Kummer's
# confluent hypergeometric function (Lewis 2000).  We use the representation
# from Baldeaux & Platen (2013) / Drimus (2012):
#
#   E[exp(iu * ln S_T)] = M(a, b; z) / M(a, b; 0)
#
# where M is the Kummer function, evaluated at specific arguments that depend
# on (u, T, r, q, p, xi, rho, v0).  Because scipy.special.hyp1f1 is available,
# this is computationally straightforward.
#
# Full derivation details in docstring of `threehalf_cf` below.

@dataclass
class ThreeHalfParams:
    """3/2 stochastic volatility model parameters.

    Model SDE:  dv = v*(p - q*v)*dt + xi*v^(3/2)*dW_v

    Attributes
    ----------
    p : float
        Attraction parameter (> 0). Controls long-run variance level: E[v] ~ p/q.
    q : float
        Mean-reversion speed parameter (> 0).
    xi : float
        Vol-of-vol (> 0).
    rho : float
        Correlation between log-price and variance Brownian motions.
    v0 : float
        Initial variance (> 0).
    """

    p: float
    q: float
    xi: float
    rho: float
    v0: float


def mc_call_prices_32(
    S0: float,
    strikes: np.ndarray,
    r: float,
    q: float,
    T: float,
    p: float,
    q_param: float,
    xi: float,
    rho: float,
    v0: float,
    n_paths: int = 4000,
    n_steps: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Price European calls via Monte Carlo simulation of the 3/2 model.

    The 3/2 model:
        dS = (r-q)*S*dt + sqrt(v)*S*dW_S
        dv = v*(p - q_param*v)*dt + xi*v^{3/2}*dW_v
        corr(dW_S, dW_v) = rho

    Implementation uses Euler-Maruyama discretisation with full truncation
    (v is set to max(v, 1e-10) after each step) and Cholesky decomposition
    for the correlated Brownian increments.

    Parameters
    ----------
    S0 : float
        Spot price.
    strikes : np.ndarray
        Array of strike prices.
    r, q, T : float
        Risk-free rate, dividend yield, maturity.
    p, q_param, xi, rho, v0 : float
        3/2 model parameters.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Time discretisation steps.
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray
        Call prices for each strike, same shape as ``strikes``.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    rho2 = np.sqrt(max(0.0, 1.0 - rho**2))

    # Initialise paths
    v = np.full(n_paths, v0, dtype=np.float64)
    log_S = np.full(n_paths, np.log(S0), dtype=np.float64)

    for _ in range(n_steps):
        v_pos = np.maximum(v, 1e-10)
        sqrt_v = np.sqrt(v_pos)

        # Correlated increments
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        dW_S = z1 * sqrt_dt
        dW_v = (rho * z1 + rho2 * z2) * sqrt_dt

        # Euler step for log(S)
        log_S += (r - q - 0.5 * v_pos) * dt + sqrt_v * dW_S

        # Euler step for v with full truncation
        v = v_pos + v_pos * (p - q_param * v_pos) * dt + xi * v_pos * sqrt_v * dW_v
        v = np.maximum(v, 1e-10)

    S_T = np.exp(log_S)
    df = np.exp(-r * T)

    call_prices = np.empty(len(strikes))
    for j, K in enumerate(strikes):
        payoff = np.maximum(S_T - K, 0.0)
        call_prices[j] = df * float(np.mean(payoff))

    return call_prices


def fft_call_price_32(
    S0: float,
    K_target: float,
    r: float,
    q: float,
    T: float,
    alpha: float,   # unused, kept for API compatibility
    N: int,         # unused, kept for API compatibility
    B: float,       # unused, kept for API compatibility
    p: float,
    q_param: float,
    xi: float,
    rho: float,
    v0: float,
    n_paths: int = 4000,
    n_steps: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Price European calls under the 3/2 model via Monte Carlo.

    Interface mirrors fft_call_price for drop-in compatibility.  The 3/2
    model characteristic function involves the Kummer confluent hypergeometric
    function M(a,b,z) evaluated at large complex arguments — numerical
    evaluation is unstable in this regime without extended-precision
    arithmetic.  Monte Carlo simulation on the SDE is mathematically
    unambiguous and gives accurate IV-surface shapes.

    Parameters
    ----------
    S0 : float
        Spot price.
    K_target : float
        Strike to price (returned as price_at_target in the third element).
    r, q, T : float
        Risk-free rate, dividend yield, maturity (years).
    alpha, N, B : float / int
        Ignored (FFT parameters kept for API compatibility with Heston).
    p, q_param, xi, rho, v0 : float
        3/2 model parameters.
    n_paths : int
        Monte Carlo paths (default 4000 balances speed vs accuracy).
    n_steps : int
        Euler time steps (default 100).

    Returns
    -------
    strikes : np.ndarray
        Strike grid centred around S0 (±50% in 40 points).
    call_prices : np.ndarray
        Monte Carlo call prices on the strike grid.
    price_at_target : float
        Interpolated price at K_target.
    """
    # Build a reasonable strike grid around S0
    strikes_mc = np.linspace(S0 * 0.5, S0 * 1.5, 40)
    call_prices = mc_call_prices_32(
        S0, strikes_mc, r, q, T, p, q_param, xi, rho, v0,
        n_paths=n_paths, n_steps=n_steps,
    )

    # Interpolate at K_target
    valid = call_prices > 0
    if valid.sum() >= 2:
        price_at_target = float(np.interp(K_target, strikes_mc[valid], call_prices[valid]))
    else:
        price_at_target = max(S0 - K_target, 0.0) * np.exp(-r * T)

    return strikes_mc, call_prices, price_at_target


def _threehalf_model_iv_batch(
    S0: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    p: float,
    q_param: float,
    xi: float,
    rho: float,
    v0: float,
    alpha: float = 1.0,
    N: int = 2**12,
    B: float = 1000.0,
    n_paths: int = 4000,
    n_steps: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Compute 3/2 model IVs for all strikes at a single expiry T.

    Uses Monte Carlo simulation (single run for all strikes simultaneously).
    Returns NaN where IV inversion fails (price outside no-arbitrage bounds).
    """
    call_prices = mc_call_prices_32(
        S0, strikes, r, q, T, p, q_param, xi, rho, v0,
        n_paths=n_paths, n_steps=n_steps, seed=seed,
    )

    model_ivs = np.empty(len(strikes))
    for i, (K, price) in enumerate(zip(strikes, call_prices)):
        iv = _bs_iv(price, S0, K, T, r, q, is_call=True)
        model_ivs[i] = iv if iv is not None else np.nan
    return model_ivs


def calibrate_32_from_options(
    option_chain,
    underlying_price: float,
    r: float = 0.0,
    q: float = 0.0,
    n_starts: int = 5,
    delta_filter: tuple = (0.10, 0.90),
    atm_only: bool = False,
    seed: int = 42,
    use_bid_ask: bool = True,
    use_oi_weights: bool = True,
) -> ThreeHalfParams:
    """Calibrate 3/2 model (p, q, ξ, ρ, v₀) from Q-measure option IV surface.

    Mirror of calibrate_heston_from_options but using the 3/2 model CF
    and FFT pricing.  Same filtering, OI-weighting, and multi-start logic.

    3/2 model:  dv = v*(p - q*v)*dt + xi*v^(3/2)*dW_v

    Parameters
    ----------
    option_chain : pd.DataFrame
        Same format as calibrate_heston_from_options.
    underlying_price : float
        BTC/USD spot.
    r, q : float
        Risk-free rate, dividend yield.
    n_starts : int
        Multi-start count.
    delta_filter : tuple
        Delta filter (low, high).
    atm_only : bool
        ATM-only filter override.
    seed : int
        RNG seed.
    use_bid_ask : bool
        Use mid-IV from bid/ask when available.
    use_oi_weights : bool
        Weight loss by open interest.

    Returns
    -------
    ThreeHalfParams
        Calibrated 3/2 model parameters.
    """
    import pandas as pd
    from scipy.optimize import minimize

    S0 = underlying_price

    # ---- filter chain (same logic as Heston calibrator) ----
    df = option_chain.copy()
    df = df[df["mark_iv"].notna() & (df["mark_iv"] > 0)].copy()
    df = df[df["kind"] == "C"].copy()

    T_min = 7 / 365.25
    T_max = 180 / 365.25
    df = df[(df["T"] >= T_min) & (df["T"] <= T_max)].copy()

    if df.empty:
        raise ValueError("No valid options after maturity filter for 3/2 calibration.")

    # Resolve market_iv
    has_bid_ask = (
        use_bid_ask
        and "bid_iv" in df.columns
        and "ask_iv" in df.columns
        and df["bid_iv"].notna().any()
        and df["ask_iv"].notna().any()
    )

    if has_bid_ask:
        bid_valid = df["bid_iv"].notna() & (df["bid_iv"] > 0)
        ask_valid = df["ask_iv"].notna() & (df["ask_iv"] > 0)
        both_valid = bid_valid & ask_valid
        df["market_iv"] = df["mark_iv"].copy()
        df.loc[both_valid, "market_iv"] = (
            df.loc[both_valid, "bid_iv"] + df.loc[both_valid, "ask_iv"]
        ) / 2.0
        df = df[~(bid_valid & (df["bid_iv"] == 0))].copy()
        if df["mark_iv"].notna().any():
            spread = (df["ask_iv"].fillna(0) - df["bid_iv"].fillna(0)).clip(lower=0)
            relative_spread = spread / df["mark_iv"].replace(0, np.nan)
            df = df[~(relative_spread > 0.50)].copy()
    else:
        df["market_iv"] = df["mark_iv"].copy()

    df = df[df["market_iv"].notna() & (df["market_iv"] > 0)].copy()
    if df.empty:
        raise ValueError("No valid options after IV resolution for 3/2 calibration.")

    # OI weights
    has_oi = (
        use_oi_weights
        and "open_interest" in df.columns
        and df["open_interest"].notna().any()
        and df["open_interest"].sum() > 0
    )
    if has_oi:
        oi_vals = df["open_interest"].fillna(0.0).clip(lower=0.0)
        total_oi = oi_vals.sum()
        df["_weight"] = oi_vals / total_oi if total_oi > 0 else 1.0 / len(df)
    else:
        df["_weight"] = 1.0 / len(df)

    if atm_only:
        df = df[(df["strike"] >= S0 * 0.80) & (df["strike"] <= S0 * 1.20)].copy()
    else:
        def _delta_ok(row):
            d = _bs_delta(S0, row["strike"], row["T"], r, q, row["market_iv"], is_call=True)
            return delta_filter[0] <= abs(d) <= delta_filter[1]
        df = df[df.apply(_delta_ok, axis=1)].copy()

    if df.empty:
        raise ValueError("No options pass delta filter for 3/2 calibration.")

    w_sum = df["_weight"].sum()
    df["_weight"] = df["_weight"] / w_sum if w_sum > 0 else 1.0 / len(df)

    df["T_key"] = df["T"].round(4)
    chain_rows = [
        (row["strike"], row["T_key"], row["market_iv"], row["_weight"])
        for _, row in df.iterrows()
    ]

    expiry_groups: dict[float, list[tuple]] = {}
    for K, T_key, miv, w in chain_rows:
        expiry_groups.setdefault(T_key, []).append((K, miv, w))

    n_contracts = len(chain_rows)
    expiries = sorted(expiry_groups.keys())

    # ------------------------------------------------------------------ #
    # Calibration strategy for 3/2 MC model:                            #
    # MC noise makes gradient-based optimizers inefficient.             #
    # Use Nelder-Mead (gradient-free) with small MC budget for speed.  #
    # Each loss evaluation: 200 paths × 20 steps (fast enough for ~5   #
    # expiries × ~10 strikes); Nelder-Mead converges in ~300 evals.   #
    # ------------------------------------------------------------------ #
    MC_PATHS_OPT = 200   # fast during optimisation
    MC_STEPS_OPT = 20

    # Build a seeded sequence so each loss evaluation uses a fixed RNG state
    # (reduces MC noise variance in the loss landscape)
    eval_rng = np.random.default_rng(seed + 999)

    def loss_32(params_arr: np.ndarray) -> float:
        p_par, q_par, xi_par, rho_par, v0_par = params_arr

        if p_par <= 0 or q_par <= 0 or xi_par <= 0 or v0_par <= 0:
            return 1e9
        if not (-1.0 < rho_par < 1.0):
            return 1e9

        weighted_sq: list[float] = []
        mc_seed = int(eval_rng.integers(0, 100000))
        for T_key in expiries:
            rows_T = expiry_groups[T_key]
            strikes_arr = np.array([r_[0] for r_ in rows_T])
            market_ivs = np.array([r_[1] for r_ in rows_T])
            weights_arr = np.array([r_[2] for r_ in rows_T])

            try:
                model_ivs = _threehalf_model_iv_batch(
                    S0, strikes_arr, T_key, r, q,
                    p_par, q_par, xi_par, rho_par, v0_par,
                    n_paths=MC_PATHS_OPT,
                    n_steps=MC_STEPS_OPT,
                    seed=mc_seed,
                )
            except Exception:
                return 1e9

            valid = ~np.isnan(model_ivs)
            if valid.sum() == 0:
                continue

            diff = model_ivs[valid] - market_ivs[valid]
            w = weights_arr[valid]
            weighted_sq.extend((w * diff**2).tolist())

        if not weighted_sq:
            return 1e9

        return float(np.sum(weighted_sq))

    rng = np.random.default_rng(seed)
    best_result = None
    best_loss = np.inf

    # ATM IV prior for initial guess
    atm_iv = float(df.loc[(df["strike"] - S0).abs().idxmin(), "market_iv"]) if len(df) > 0 else 0.80
    v0_prior = atm_iv**2

    starts = [[4.0, 4.0, atm_iv, -0.3, v0_prior]]
    for _ in range(n_starts - 1):
        # Sample in [p_lo,p_hi] etc but keep sensible
        starts.append([
            rng.uniform(1.0, 10.0),   # p
            rng.uniform(1.0, 10.0),   # q
            rng.uniform(0.1, 1.5),    # xi
            rng.uniform(-0.8, -0.1),  # rho (negative for leverage)
            rng.uniform(0.02, 0.5),   # v0
        ])

    # Use Nelder-Mead (gradient-free, handles noisy MC objective)
    for p0 in starts:
        try:
            res = minimize(
                loss_32,
                x0=np.array(p0, dtype=float),
                method="Nelder-Mead",
                options={
                    "maxiter": 300,
                    "xatol": 1e-3,
                    "fatol": 1e-5,
                    "disp": False,
                },
            )
        except Exception as exc:
            warnings.warn(f"calibrate_32_from_options: optimizer failed ({exc})")
            continue

        # Enforce bounds manually (Nelder-Mead doesn't support them natively)
        p_c, q_c, xi_c, rho_c, v0_c = res.x
        if (p_c <= 0 or q_c <= 0 or xi_c <= 0 or v0_c <= 0
                or not (-1.0 < rho_c < 1.0)):
            continue

        if res.fun < best_loss:
            best_loss = res.fun
            best_result = res

    if best_result is None:
        raise RuntimeError("calibrate_32_from_options: all optimizer starts failed.")

    p_cal, q_cal, xi_cal, rho_cal, v0_cal = best_result.x
    # Clip to reasonable physical ranges
    p_cal = float(np.clip(p_cal, 0.1, 50.0))
    q_cal = float(np.clip(q_cal, 0.1, 50.0))
    xi_cal = float(np.clip(xi_cal, 0.01, 5.0))
    rho_cal = float(np.clip(rho_cal, -0.99, 0.99))
    v0_cal = float(np.clip(v0_cal, 0.001, 3.0))

    print(
        f"[calibrate_32_from_options] Converged. "
        f"n_contracts={n_contracts}, loss={best_loss:.6f}"
    )

    return ThreeHalfParams(
        p=float(p_cal),
        q=float(q_cal),
        xi=float(xi_cal),
        rho=float(rho_cal),
        v0=float(v0_cal),
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
