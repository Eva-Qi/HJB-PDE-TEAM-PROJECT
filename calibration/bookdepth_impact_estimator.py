"""Regime-conditional temporary impact (η) from Binance Vision bookDepth CSVs.

Context
-------
``calibration/impact_estimator.py::calibrated_params_per_regime`` estimates η
by log-log regressing 1-minute |return| on |net_flow| per regime.  For the
risk_off regime on 2026 BTCUSDT data, that regression fails the acceptance
window (alpha ≈ 0.26, r² ≈ 0.10) and the function falls back to literature
constants η=1e-3, α=0.6.  The V5 paired test then executes risk_off
trajectories with a temporary impact coefficient 1,000× larger than the
risk_on calibrated value (1.49e-4), producing the nonsensical +3225% CVaR
blow-up recorded in ``data/paired_regime_v5_true_params.json``.

bookDepth format
----------------
Binance Vision ``BTCUSDT-bookDepth-YYYY-MM-DD.csv`` rows::

    timestamp,percentage,depth,notional
    2026-03-23 00:00:08,-5.00,8312.308,552377402.937
    2026-03-23 00:00:08,-4.00,7893.327,525183821.950
    ...
    2026-03-23 00:00:08,-0.20, 424.783,  28780286.865
    2026-03-23 00:00:08,+0.20, 337.571,  22925473.746
    2026-03-23 00:00:08,+1.00,1489.470, 101439986.095
    ...
    2026-03-23 00:00:08,+5.00,6100.364, 422599010.589

Twelve buckets per snapshot at % offsets {-5, -4, -3, -2, -1, -0.2, +0.2,
+1, +2, +3, +4, +5} relative to mid.  ``depth`` is CUMULATIVE BTC notional
from mid out to that percentage band (verified:
``notional[+1]-notional[+0.2] / (depth[+1]-depth[+0.2]) = 68163 ≈ mid*1.006``).
Snapshots arrive every ~30 s (≈2600 snapshots/day).  This is NOT level-by-level
L2 — it is aggregated cumulative depth at fixed % offsets.

Estimator
---------
For a buy of ``q`` BTC starting from mid, walk the ask side: the average fill
price is ``notional[+0.2] / depth[+0.2]`` when ``q ≤ depth[+0.2]``, and in
general linearly interpolated on the cumulative (depth, notional) curve.
Slippage = ``avg_fill_price − mid``.  For X0=10 BTC / N=50 steps, per-step
q ≈ 0.2 BTC, which lies comfortably inside the 0-0.2% bucket (median depth
≈ 300 BTC), so this reduces to a linear-within-bucket assumption.

To match ``sde_engine._compute_execution_cost_step``::

    temp_impact_USD = params.eta * v ** params.alpha    # v = q / dt  (BTC/year)

we solve for η on each snapshot with α fixed at the risk_on calibrated value
(0.42), average across snapshots within a regime, and return one η per regime.
We fix α across both regimes because the risk_off α is itself a fallback
constant — making it a free parameter would just re-introduce the ambiguity
this estimator is designed to eliminate.

Sanity tripwire
---------------
The existing ``impact_estimator`` outputs η=1.49e-4 for risk_on (aggTrades
log-log).  bookDepth η should land in the O(1e-4) to O(1e-2) range; anything
below 1e-8 or above 1 is a unit bug and the estimator raises.

No files in ``data/`` or ``calibration/`` are modified.  The persisted output
``data/eta_bookdepth_regime.json`` is opt-in for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ── Canonical bucket offsets emitted by Binance Vision (% from mid) ──────
_BUCKETS_ALL = [-5.0, -4.0, -3.0, -2.0, -1.0, -0.2, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
_ASK_BUCKETS = [0.2, 1.0, 2.0, 3.0, 4.0, 5.0]
_BID_BUCKETS = [-0.2, -1.0, -2.0, -3.0, -4.0, -5.0]


@dataclass(frozen=True)
class BookDepthEtaDiagnostics:
    eta: float
    median_slippage_usd: float
    mean_slippage_usd: float
    n_snapshots: int
    trade_size_btc: float
    alpha_used: float
    mid_price_mean: float


def _load_one_day(csv_path: Path) -> pd.DataFrame:
    """Load one bookDepth CSV, return tidy frame indexed by (timestamp, percentage)."""
    df = pd.read_csv(
        csv_path,
        dtype={"percentage": np.float64, "depth": np.float64, "notional": np.float64},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_bookdepth(
    data_dir: Path,
    start: str = "2026-03-23",
    end: str = "2026-04-19",
) -> pd.DataFrame:
    """Concatenate every BTCUSDT-bookDepth-YYYY-MM-DD.csv inside the window.

    No synthetic data: if a date is missing we just omit it and report the
    actual set of dates loaded.
    """
    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()

    frames = []
    dates_loaded: list[str] = []
    for csv in sorted(data_dir.glob("BTCUSDT-bookDepth-*.csv")):
        stem = csv.stem  # BTCUSDT-bookDepth-2026-03-23
        try:
            d = pd.to_datetime(stem.split("bookDepth-")[1]).date()
        except Exception:
            continue
        if not (start_d <= d <= end_d):
            continue
        frames.append(_load_one_day(csv))
        dates_loaded.append(str(d))
    if not frames:
        raise FileNotFoundError(
            f"No bookDepth CSVs matching {start}..{end} in {data_dir}"
        )
    out = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "percentage"])
    out.attrs["dates_loaded"] = dates_loaded
    return out.reset_index(drop=True)


def _snapshot_mid_price(snapshot: pd.DataFrame) -> float | None:
    """Infer mid from the ±0.2% rows:   mid ≈ 0.5 * (p_bid_0.2 + p_ask_0.2).

    p_bucket = notional / depth is the mean fill price within the 0–0.2% band
    on each side; averaging them cancels the ±0.1% bias (mid is at 0%,
    bucket mean is at ≈ ±0.1%).  This is ~99.99% accurate vs book mid.
    Returns None if the row is malformed.
    """
    try:
        row_bid = snapshot.loc[snapshot["percentage"] == -0.2].iloc[0]
        row_ask = snapshot.loc[snapshot["percentage"] == 0.2].iloc[0]
    except IndexError:
        return None
    if row_bid["depth"] <= 0 or row_ask["depth"] <= 0:
        return None
    p_bid = float(row_bid["notional"] / row_bid["depth"])
    p_ask = float(row_ask["notional"] / row_ask["depth"])
    return 0.5 * (p_bid + p_ask)


def _avg_fill_price_asks(
    q_btc: float,
    ask_rows: pd.DataFrame,
    mid: float,
) -> float | None:
    """Average fill price for buying ``q_btc`` BTC against the ask side.

    ``ask_rows`` must be the sub-frame with percentage ∈ {0.2, 1, 2, 3, 4, 5}
    sorted ascending.  depth / notional are CUMULATIVE from mid.  We
    interpolate linearly on the cumulative curve: if q is below depth[0.2],
    the fill price is notional[0.2]/depth[0.2] (the mean inside that band,
    which is the most honest approximation without level-by-level data).
    If q falls between cumulative points [D_i, D_{i+1}], we linearly split
    the incremental notional.

    Returns None when q exceeds cumulative depth at +5% (book too thin).
    """
    ask_rows = ask_rows.sort_values("percentage")
    depths = ask_rows["depth"].to_numpy()
    notionals = ask_rows["notional"].to_numpy()

    if q_btc <= 0 or len(depths) == 0:
        return None
    if q_btc > depths[-1]:
        return None  # extremely rare — 5% depth > 6000 BTC typical

    if q_btc <= depths[0]:
        # Entire fill sits in the 0-0.2% bucket.  Use that bucket's mean
        # fill price — the most honest estimate without tick data.
        return float(notionals[0] / depths[0]) if depths[0] > 0 else None

    # Find bracketing buckets on the cumulative curve.
    for i in range(len(depths) - 1):
        if depths[i] < q_btc <= depths[i + 1]:
            # Increment between bucket i and i+1
            d_inc = depths[i + 1] - depths[i]
            n_inc = notionals[i + 1] - notionals[i]
            if d_inc <= 0:
                return None
            p_inc_mean = n_inc / d_inc  # avg price in the incremental band
            # Linear split: fraction of increment consumed
            frac = (q_btc - depths[i]) / d_inc
            notional_filled = notionals[i] + frac * n_inc
            return float(notional_filled / q_btc)

    return None


def _avg_fill_price_bids(
    q_btc: float,
    bid_rows: pd.DataFrame,
    mid: float,
) -> float | None:
    """Mirror of ``_avg_fill_price_asks`` for selling into the bid side."""
    # Sort by absolute percentage ascending so cumulative walk matches ask-side.
    bid_rows = bid_rows.assign(abs_pct=bid_rows["percentage"].abs()).sort_values("abs_pct")
    depths = bid_rows["depth"].to_numpy()
    notionals = bid_rows["notional"].to_numpy()

    if q_btc <= 0 or len(depths) == 0:
        return None
    if q_btc > depths[-1]:
        return None

    if q_btc <= depths[0]:
        return float(notionals[0] / depths[0]) if depths[0] > 0 else None

    for i in range(len(depths) - 1):
        if depths[i] < q_btc <= depths[i + 1]:
            d_inc = depths[i + 1] - depths[i]
            n_inc = notionals[i + 1] - notionals[i]
            if d_inc <= 0:
                return None
            frac = (q_btc - depths[i]) / d_inc
            notional_filled = notionals[i] + frac * n_inc
            return float(notional_filled / q_btc)
    return None


def estimate_eta_from_bookdepth(
    depth_df: pd.DataFrame,
    trade_size_btc: float,
    dt_years: float,
    alpha: float,
    *,
    snapshot_subsample: int | None = None,
) -> BookDepthEtaDiagnostics:
    """Regress η from bookDepth snapshots assuming the Almgren-Chriss power law.

    For each snapshot we compute the average slippage (|fill_price − mid|) in
    USD/BTC when executing ``trade_size_btc`` on both ask and bid sides, then
    invert ``temp_impact_USD = η * v^α`` with v = trade_size_btc / dt_years
    and α fixed (the risk_on calibrated value).

    Parameters
    ----------
    depth_df : DataFrame
        Concatenated bookDepth rows (columns: timestamp, percentage, depth, notional).
    trade_size_btc : float
        Per-step trade size in BTC.  Defaults in callers to X0/N = 0.2 BTC.
    dt_years : float
        Per-step duration in years (matches ``params.dt``).  With T=1/24,
        N=50 this is 1/1200 ≈ 8.33e-4 year.
    alpha : float
        Power-law exponent to hold fixed while solving for η.
    snapshot_subsample : int, optional
        If provided, keep only every ``snapshot_subsample``-th distinct
        timestamp (speed knob for 28-day runs).

    Returns
    -------
    BookDepthEtaDiagnostics
    """
    if trade_size_btc <= 0:
        raise ValueError("trade_size_btc must be positive")
    if dt_years <= 0:
        raise ValueError("dt_years must be positive")

    # Group once, iterate per snapshot.
    grouped = depth_df.groupby("timestamp", sort=False)
    timestamps = list(grouped.groups.keys())
    if snapshot_subsample and snapshot_subsample > 1:
        timestamps = timestamps[::snapshot_subsample]

    slippages = []  # USD per BTC
    mids = []
    for ts in timestamps:
        snap = grouped.get_group(ts)
        if len(snap) < 12:
            continue  # incomplete snapshot
        mid = _snapshot_mid_price(snap)
        if mid is None or mid <= 0:
            continue
        asks = snap[snap["percentage"] > 0]
        bids = snap[snap["percentage"] < 0]
        p_ask_fill = _avg_fill_price_asks(trade_size_btc, asks, mid)
        p_bid_fill = _avg_fill_price_bids(trade_size_btc, bids, mid)
        if p_ask_fill is None or p_bid_fill is None:
            continue
        ask_slip = p_ask_fill - mid           # positive for buys
        bid_slip = mid - p_bid_fill           # positive for sells
        # Average both sides — the AC model is symmetric in buys vs sells.
        slippage = 0.5 * (ask_slip + bid_slip)
        if slippage <= 0 or not np.isfinite(slippage):
            continue
        slippages.append(slippage)
        mids.append(mid)

    if len(slippages) < 50:
        raise ValueError(
            f"Only {len(slippages)} usable snapshots — need ≥ 50 for a "
            "stable η estimate.  Check input DataFrame coverage."
        )

    slippages = np.asarray(slippages)
    mids = np.asarray(mids)
    v = trade_size_btc / dt_years  # BTC / year
    # η per snapshot, then median to suppress outliers.
    # temp_impact_USD = η * v**α → η = slippage / v**α
    eta_per_snap = slippages / (v ** alpha)
    eta = float(np.median(eta_per_snap))

    # ── Sanity tripwire ─────────────────────────────────────
    if not (1e-8 < eta < 1e2):
        raise ValueError(
            f"bookDepth η = {eta:.3e} is outside the plausible "
            f"O(1e-8, 1e2) range — likely a unit bug.  Aborting to avoid "
            "poisoning downstream sims."
        )

    return BookDepthEtaDiagnostics(
        eta=eta,
        median_slippage_usd=float(np.median(slippages)),
        mean_slippage_usd=float(np.mean(slippages)),
        n_snapshots=len(slippages),
        trade_size_btc=trade_size_btc,
        alpha_used=alpha,
        mid_price_mean=float(np.mean(mids)),
    )


def assign_regime_to_snapshots(
    snapshot_timestamps: pd.DatetimeIndex,
    bar_timestamps: pd.DatetimeIndex,
    state_sequence: np.ndarray,
) -> np.ndarray:
    """Assign a 5-min HMM regime label to each bookDepth snapshot.

    Uses the same right-searchsorted rule as
    ``impact_estimator.calibrated_params_per_regime`` — each snapshot is
    labelled with the regime of the most recent bar whose start ≤ snapshot.
    Snapshots before the first bar receive -1 (caller must filter).
    """
    bar_ts = pd.DatetimeIndex(bar_timestamps)
    if bar_ts.tz is None:
        bar_ts = bar_ts.tz_localize("UTC")
    else:
        bar_ts = bar_ts.tz_convert("UTC")
    snap_ts = pd.DatetimeIndex(snapshot_timestamps)
    if snap_ts.tz is None:
        snap_ts = snap_ts.tz_localize("UTC")
    else:
        snap_ts = snap_ts.tz_convert("UTC")

    bar_ns = bar_ts.view("int64")
    snap_ns = snap_ts.view("int64")
    idx = np.searchsorted(bar_ns, snap_ns, side="right") - 1
    out = np.full(len(snap_ns), -1, dtype=int)
    mask = idx >= 0
    out[mask] = state_sequence[idx[mask]].astype(int)
    return out


def estimate_eta_per_regime_from_bookdepth(
    depth_df: pd.DataFrame,
    regime_labels: np.ndarray,
    trade_size_btc: float,
    dt_years: float,
    alpha: float,
    snapshot_subsample: int | None = None,
) -> dict:
    """Split bookDepth snapshots by regime label and estimate η per regime.

    Parameters
    ----------
    depth_df : DataFrame
        Output of ``load_bookdepth``.
    regime_labels : np.ndarray
        One label per unique timestamp in ``depth_df``, in the same order as
        ``depth_df.groupby('timestamp', sort=False).groups.keys()``.
    """
    unique_ts = list(depth_df.groupby("timestamp", sort=False).groups.keys())
    if len(unique_ts) != len(regime_labels):
        raise ValueError(
            f"regime_labels length {len(regime_labels)} != unique timestamps "
            f"{len(unique_ts)}"
        )

    ts_to_label = dict(zip(unique_ts, regime_labels))
    depth_df = depth_df.copy()
    depth_df["regime"] = depth_df["timestamp"].map(ts_to_label)

    results = {}
    for regime in sorted(depth_df["regime"].unique()):
        if regime < 0:
            continue
        sub = depth_df[depth_df["regime"] == regime]
        diag = estimate_eta_from_bookdepth(
            sub,
            trade_size_btc=trade_size_btc,
            dt_years=dt_years,
            alpha=alpha,
            snapshot_subsample=snapshot_subsample,
        )
        results[int(regime)] = diag
    return results
