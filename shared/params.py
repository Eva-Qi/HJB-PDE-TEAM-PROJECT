"""Almgren-Chriss parameter definitions and closed-form solution.

This module is the CONTRACT for the entire project. All modules (PDE, MC,
calibration) read parameters from ACParams. P2 and P3 start with
DEFAULT_PARAMS; P1 replaces them with calibrated_params() once real data
is processed.

References:
    Almgren & Chriss (2001), "Optimal Execution of Portfolio Transactions"
    Cartea, Jaimungal & Penalva (2015), Ch. 6-7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# BTC 24h UTC hourly volume profile (from Binance historical data)
HOURLY_VOLUME_PROFILE: dict[int, float] = {
    0: 0.85, 1: 0.75, 2: 0.70, 3: 0.65,
    4: 0.70, 5: 0.80, 6: 0.90, 7: 1.00,
    8: 1.10, 9: 1.20, 10: 1.25, 11: 1.20,
    12: 1.15, 13: 1.10, 14: 1.15, 15: 1.20,
    16: 1.30, 17: 1.35, 18: 1.30, 19: 1.20,
    20: 1.10, 21: 1.00, 22: 0.95, 23: 0.90,
}


@dataclass
class ACParams:
    """Almgren-Chriss optimal execution parameters.

    Attributes
    ----------
    S0 : float
        Initial asset price.
    sigma : float
        Annualized volatility.
    mu : float
        Drift (typically 0 for execution problems — short horizon).
    X0 : float
        Initial inventory (shares/units to liquidate).
    T : float
        Execution horizon in years (e.g., 1/252 for one trading day).
    N : int
        Number of discrete time steps.
    gamma : float
        Permanent impact coefficient. Price shift per unit traded:
        ΔS_perm = gamma * n_k
    eta : float
        Temporary impact coefficient. Cost per unit:
        h(v) = eta * |v|^alpha * sign(v)
    alpha : float
        Temporary impact exponent (1.0 = linear, default).
    lam : float
        Risk aversion parameter (lambda). Higher = more risk-averse
        (trades faster to reduce variance).
    fee_bps : float
        Exchange fee in basis points per notional traded. Default 0.0
        (fees excluded from cost). Binance BTCUSDT spot taker = 7.5 bps.
        Applied as fee_bps/1e4 * S0 * |n_k| per trade.
    volume_profile : np.ndarray or None
        Normalized intraday volume weights for VWAP. Length N.
        If None, uniform (TWAP).
    """

    S0: float
    sigma: float
    mu: float
    X0: float
    T: float
    N: int
    gamma: float
    eta: float
    alpha: float
    lam: float
    fee_bps: float = 0.0
    volume_profile: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def dt(self) -> float:
        """Time step size."""
        return self.T / self.N

    @property
    def tau(self) -> float:
        """Alias for dt (Almgren-Chriss notation)."""
        return self.dt

    @property
    def kappa(self) -> float:
        """Urgency parameter: kappa = sqrt(lambda * S0^2 * sigma^2 / eta).

        Uses dollar volatility (S0 * sigma_pct) so that risk is in dollar units.
        Large kappa → trade aggressively early (front-loaded).
        Small kappa → close to TWAP (uniform).
        """
        return np.sqrt(self.lam * self.S0**2 * self.sigma**2 / self.eta)


# Synthetic parameters giving kappa*T ≈ 1.5 — clearly front-loaded
# optimal trajectory vs TWAP. Magnitudes from Almgren-Chriss (2001).
# P1 will replace these with real Binance-calibrated values.
DEFAULT_PARAMS = ACParams(
    S0=50.0,            # price per share
    sigma=0.3,          # 30% annualized vol (crypto-like)
    mu=0.0,             # zero drift over execution horizon
    X0=1_000_000,       # 1M shares to liquidate
    T=0.25,             # ~63 trading days (~3 months)
    N=50,               # time steps
    gamma=2.5e-7,       # permanent impact
    eta=2.5e-6,         # temporary impact
    alpha=1.0,          # linear temporary impact
    lam=4e-7,           # risk aversion (adjusted for S0² in kappa)
    # kappa = sqrt(lam * S0^2 * sigma^2 / eta) = sqrt(4e-7 * 2500 * 0.09 / 2.5e-6) = 6.0
    # kappa * T = 6.0 * 0.25 = 1.5 → meaningful front-loading
)


def almgren_chriss_closed_form(
    params: ACParams,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form optimal trajectory and expected cost (linear impact only).

    Valid only when alpha = 1.0 (linear temporary impact).

    The optimal inventory trajectory is:
        x*(t_k) = X0 * sinh(kappa * (T - t_k)) / sinh(kappa * T)

    The optimal trade list (shares per step) is:
        n_k = x*(t_{k-1}) - x*(t_k)

    Expected cost is computed via direct summation of the discrete
    permanent and temporary impact costs along the optimal trajectory.

    Parameters
    ----------
    params : ACParams
        Model parameters. Must have alpha == 1.0.

    Returns
    -------
    t_grid : np.ndarray, shape (N+1,)
        Time grid [0, dt, 2*dt, ..., T].
    x_trajectory : np.ndarray, shape (N+1,)
        Optimal inventory path. x[0] = X0, x[N] ≈ 0.
    expected_cost : float
        E[cost] under the optimal trajectory.
    """
    if abs(params.alpha - 1.0) > 1e-10:
        raise ValueError(
            f"Closed-form requires linear impact (alpha=1.0), got {params.alpha}"
        )

    kappa = params.kappa
    T = params.T
    X0 = params.X0
    N = params.N
    dt = params.dt

    t_grid = np.linspace(0, T, N + 1)

    # Optimal inventory trajectory
    sinh_kT = np.sinh(kappa * T)
    x_trajectory = X0 * np.sinh(kappa * (T - t_grid)) / sinh_kT

    # Trade list
    n_k = x_trajectory[:-1] - x_trajectory[1:]  # shares traded each step
    v_k = n_k / dt  # trading rate

    # Expected cost via direct summation (more transparent):
    # Permanent: sum_k gamma * n_k * S (≈ 0.5 * gamma * X0^2 for small impact)
    # Temporary: sum_k eta * (n_k/tau)^alpha * n_k = eta/tau * sum_k n_k^2
    perm_cost = params.gamma * np.sum(n_k * np.cumsum(n_k))
    temp_cost = params.eta / dt * np.sum(n_k**2)
    expected_cost = perm_cost + temp_cost

    return t_grid, x_trajectory, expected_cost
