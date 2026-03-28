"""Statistical bootstrapping for model-free execution cost estimation.

Instead of assuming GBM (parametric), this module constructs synthetic price
paths by resampling historical returns. This provides a model-free benchmark
to compare against the parametric MC results.

Two bootstrap methods:
    - Simple (i.i.d.): resample individual returns with replacement.
      Destroys autocorrelation — appropriate if returns are truly i.i.d.
    - Block: resample contiguous blocks of returns, preserving short-range
      dependence (volatility clustering). More appropriate for BTC.

References:
    Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
    Politis & Romano (1994), "The Stationary Bootstrap"
"""

from __future__ import annotations

import numpy as np

from shared.params import ACParams
from shared.cost_model import temporary_impact


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series.

    Parameters
    ----------
    prices : np.ndarray, shape (M,)
        Historical price series (e.g., 5-min BTC prices from Binance).

    Returns
    -------
    np.ndarray, shape (M-1,)
        Log returns: ln(S_{i+1} / S_i).
    """
    return np.log(prices[1:] / prices[:-1])


def bootstrap_paths_simple(
    log_returns: np.ndarray,
    S0: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic price paths via simple i.i.d. bootstrap.

    Each time step draws a single historical return at random (with
    replacement). Destroys any autocorrelation in the original series.

    Parameters
    ----------
    log_returns : np.ndarray, shape (M,)
        Historical log returns to resample from.
    S0 : float
        Starting price for synthetic paths.
    n_steps : int
        Number of time steps per path (should match params.N).
    n_paths : int
        Number of synthetic paths to generate.
    seed : int

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps+1)
        Synthetic price paths. paths[:, 0] = S0.
    """
    rng = np.random.default_rng(seed)

    # Draw random return indices: (n_paths, n_steps)
    indices = rng.integers(0, len(log_returns), size=(n_paths, n_steps))
    sampled_returns = log_returns[indices]

    # Build price paths from cumulative returns
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(np.cumsum(sampled_returns, axis=1))

    return S


def bootstrap_paths_block(
    log_returns: np.ndarray,
    S0: float,
    n_steps: int,
    n_paths: int,
    block_size: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic price paths via block bootstrap.

    Resamples contiguous blocks of returns, preserving short-range
    dependence (e.g., volatility clustering in BTC).

    Parameters
    ----------
    log_returns : np.ndarray, shape (M,)
        Historical log returns.
    S0 : float
        Starting price.
    n_steps : int
        Number of time steps per path.
    n_paths : int
        Number of synthetic paths.
    block_size : int
        Length of each contiguous block. Larger = more autocorrelation
        preserved, but fewer unique blocks available.
    seed : int

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps+1)
        Synthetic price paths. paths[:, 0] = S0.
    """
    rng = np.random.default_rng(seed)
    M = len(log_returns)

    # Number of blocks needed to fill n_steps
    n_blocks = int(np.ceil(n_steps / block_size))

    # Maximum valid starting index for a block
    max_start = M - block_size

    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    sampled_returns = np.zeros((n_paths, n_steps))

    for i in range(n_paths):
        # Draw random block starting positions
        starts = rng.integers(0, max_start, size=n_blocks)

        # Concatenate blocks
        blocks = np.concatenate([
            log_returns[s : s + block_size] for s in starts
        ])

        # Trim to exactly n_steps
        sampled_returns[i, :] = blocks[:n_steps]

    # Build price paths
    S[:, 1:] = S0 * np.exp(np.cumsum(sampled_returns, axis=1))

    return S


def bootstrap_execution_cost(
    price_paths: np.ndarray,
    trajectory_x: np.ndarray,
    params: ACParams,
) -> np.ndarray:
    """Compute execution cost on bootstrapped price paths.

    Same cost computation as simulate_execution, but uses pre-built
    price paths (from bootstrap) instead of SDE simulation.

    Parameters
    ----------
    price_paths : np.ndarray, shape (n_paths, N+1)
        Bootstrapped price paths.
    trajectory_x : np.ndarray, shape (N+1,)
        Inventory trajectory (TWAP, VWAP, or optimal).
    params : ACParams
        For temporary impact parameters.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        Execution cost per path (implementation shortfall).
    """
    dt = params.dt
    n_k = trajectory_x[:-1] - trajectory_x[1:]
    v_k = n_k / dt
    h_k = temporary_impact(v_k, params.eta, params.alpha)

    n_paths = price_paths.shape[0]
    N = len(n_k)
    costs = np.zeros(n_paths)

    for k in range(N):
        costs += n_k[k] * (params.S0 - price_paths[:, k]) + h_k[k] * n_k[k]

    return costs


def generate_synthetic_returns(params: ACParams, n_obs: int = 5000, seed: int = 0) -> np.ndarray:
    """Generate synthetic historical returns for testing when real data is unavailable.

    Simulates a GBM price series and extracts log returns.
    This lets P3 develop and test the bootstrap pipeline before
    P1 delivers real Binance data.

    Parameters
    ----------
    params : ACParams
        Uses S0, mu, sigma to generate realistic returns.
    n_obs : int
        Number of return observations to generate.
    seed : int

    Returns
    -------
    np.ndarray, shape (n_obs,)
        Synthetic log returns calibrated to params.
    """
    rng = np.random.default_rng(seed)

    # Daily-scale returns (assuming 252 trading days/year)
    dt_daily = 1.0 / 252
    returns = (params.mu - 0.5 * params.sigma**2) * dt_daily + \
              params.sigma * np.sqrt(dt_daily) * rng.standard_normal(n_obs)

    return returns
