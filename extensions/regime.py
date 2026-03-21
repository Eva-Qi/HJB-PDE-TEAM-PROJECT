"""Part E: Regime-aware execution via Hidden Markov Model.

Classifies market conditions into regimes (e.g., risk-on vs risk-off)
and provides regime-specific parameters for the execution problem.

In risk-off: higher sigma, wider spreads, larger impact → trade faster.
In risk-on:  lower sigma, tighter spreads → more patient execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.params import ACParams


@dataclass
class RegimeParams:
    """Per-regime execution parameters."""

    label: str          # e.g., "risk_on", "risk_off"
    sigma: float        # regime-specific volatility
    gamma: float        # regime-specific permanent impact
    eta: float          # regime-specific temporary impact
    probability: float  # steady-state probability of this regime


def fit_hmm(
    returns: np.ndarray,
    n_regimes: int = 2,
) -> tuple[list[RegimeParams], np.ndarray]:
    """Fit a Hidden Markov Model to classify market regimes.

    Parameters
    ----------
    returns : np.ndarray
        Log returns series.
    n_regimes : int
        Number of regimes (default 2: risk-on, risk-off).

    Returns
    -------
    regimes : list[RegimeParams]
        Per-regime parameter estimates.
    state_sequence : np.ndarray
        Most likely regime at each time step (Viterbi path).
    """
    raise NotImplementedError(
        "Part E: implement HMM regime classification.\n"
        "Option 1: hmmlearn library (GaussianHMM)\n"
        "Option 2: manual Baum-Welch EM algorithm\n"
        "Output: per-regime (mu, sigma) + transition matrix"
    )


def regime_aware_params(
    base_params: ACParams,
    regime: RegimeParams,
) -> ACParams:
    """Create regime-specific ACParams by overriding volatility and impact.

    Parameters
    ----------
    base_params : ACParams
        Base parameter set.
    regime : RegimeParams
        Regime-specific overrides.

    Returns
    -------
    ACParams
        Modified parameters for this regime.
    """
    raise NotImplementedError(
        "Part E: create ACParams with regime-specific sigma, gamma, eta.\n"
        "Use dataclasses.replace(base_params, sigma=regime.sigma, ...)"
    )
