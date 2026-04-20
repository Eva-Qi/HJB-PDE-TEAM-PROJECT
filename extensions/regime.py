"""Part E: Regime-aware execution via Hidden Markov Model.

Classifies market conditions into regimes (e.g., risk-on vs risk-off)
and provides regime-specific parameters for the execution problem.

In risk-off: higher sigma, wider spreads, larger impact → trade faster.
In risk-on:  lower sigma, tighter spreads → more patient execution.

Implementation strategy:
    - Primary: hmmlearn.GaussianHMM (if installed)
    - Fallback: manual 2-state Gaussian HMM via Baum-Welch EM
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp

from shared.params import ACParams

# Minimum observations required for a reliable 2-state fit.
_MIN_OBS = 50

# ── Try importing hmmlearn; flag availability ────────────────
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM

    _HAS_HMMLEARN = True
except ImportError:  # pragma: no cover
    _HAS_HMMLEARN = False


@dataclass
class RegimeParams:
    """Per-regime execution parameters."""

    label: str          # e.g., "risk_on", "risk_off"
    sigma: float        # regime-specific volatility
    gamma: float        # regime-specific permanent impact
    eta: float          # regime-specific temporary impact
    probability: float  # steady-state probability of this regime


# ═══════════════════════════════════════════════════════════════
# Manual 2-state Gaussian HMM (Baum-Welch EM + Viterbi)
# ═══════════════════════════════════════════════════════════════

def _log_normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Log pdf of a univariate Gaussian, broadcast over *x*."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def _forward(log_emission: np.ndarray, log_A: np.ndarray,
             log_pi: np.ndarray) -> tuple[np.ndarray, float]:
    """Forward algorithm in log space.

    Parameters
    ----------
    log_emission : (T, K) log p(x_t | state_k)
    log_A : (K, K) log transition matrix
    log_pi : (K,) log initial state probabilities

    Returns
    -------
    log_alpha : (T, K)
    log_likelihood : float
    """
    T, K = log_emission.shape
    log_alpha = np.full((T, K), -np.inf)
    log_alpha[0] = log_pi + log_emission[0]

    for t in range(1, T):
        for j in range(K):
            log_alpha[t, j] = (
                logsumexp(log_alpha[t - 1] + log_A[:, j])
                + log_emission[t, j]
            )
    log_likelihood = float(logsumexp(log_alpha[-1]))
    return log_alpha, log_likelihood


def _backward(log_emission: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    """Backward algorithm in log space."""
    T, K = log_emission.shape
    log_beta = np.zeros((T, K))  # log(1) = 0 at t = T-1

    for t in range(T - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = logsumexp(
                log_A[i, :] + log_emission[t + 1] + log_beta[t + 1]
            )
    return log_beta


def _viterbi(log_emission: np.ndarray, log_A: np.ndarray,
             log_pi: np.ndarray) -> np.ndarray:
    """Viterbi decoding — most likely state sequence."""
    T, K = log_emission.shape
    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    delta[0] = log_pi + log_emission[0]

    for t in range(1, T):
        for j in range(K):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = int(np.argmax(scores))
            delta[t, j] = scores[psi[t, j]] + log_emission[t, j]

    # Back-trace
    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _baum_welch(
    x: np.ndarray,
    n_states: int = 2,
    n_iter: int = 200,
    tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Baum-Welch EM for a univariate Gaussian HMM.

    Returns
    -------
    means : (K,)
    stds : (K,)
    A : (K, K)  transition matrix
    pi : (K,)   initial state distribution
    log_likelihood : float
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = len(x)
    K = n_states

    # ── Initialise with K-means-style heuristic ──────────────
    sorted_x = np.sort(x)
    chunk = T // K
    means = np.array([sorted_x[i * chunk:(i + 1) * chunk].mean() for i in range(K)])
    stds = np.array([max(sorted_x[i * chunk:(i + 1) * chunk].std(), 1e-8) for i in range(K)])

    # Slight random perturbation to break symmetry
    means += rng.normal(0, stds * 0.1, size=K)
    stds = np.abs(stds) + 1e-8

    # Uniform initial / transition
    A = np.full((K, K), 1.0 / K)
    # Add some persistence bias
    A = 0.7 * np.eye(K) + 0.3 / K
    A /= A.sum(axis=1, keepdims=True)
    pi = np.ones(K) / K

    prev_ll = -np.inf

    for iteration in range(n_iter):
        # ── E step ───────────────────────────────────────────
        log_emission = np.column_stack(
            [_log_normal_pdf(x, means[k], stds[k]) for k in range(K)]
        )
        log_A = np.log(A + 1e-300)
        log_pi = np.log(pi + 1e-300)

        log_alpha, ll = _forward(log_emission, log_A, log_pi)
        log_beta = _backward(log_emission, log_A)

        # Convergence check
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # Posterior gamma(t, k) = P(state_t = k | X)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Xi(t, i, j) = P(state_t=i, state_{t+1}=j | X)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (
                        log_alpha[t, i]
                        + log_A[i, j]
                        + log_emission[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            xi[t] -= logsumexp(xi[t].ravel())
        xi = np.exp(xi)

        # ── M step ───────────────────────────────────────────
        pi = gamma[0] / gamma[0].sum()

        for k in range(K):
            gamma_k_sum = gamma[:, k].sum()
            if gamma_k_sum < 1e-10:
                continue
            means[k] = (gamma[:, k] * x).sum() / gamma_k_sum
            diff = x - means[k]
            stds[k] = np.sqrt((gamma[:, k] * diff ** 2).sum() / gamma_k_sum)
            stds[k] = max(stds[k], 1e-8)  # floor

        for i in range(K):
            xi_i_sum = xi[:, i, :].sum(axis=0)
            denom = xi_i_sum.sum()
            if denom > 1e-10:
                A[i] = xi_i_sum / denom

    return means, stds, A, pi, prev_ll


def _fit_hmm_manual(
    returns: np.ndarray,
    n_regimes: int = 2,
    n_init: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a Gaussian HMM using multiple random restarts of Baum-Welch.

    Returns (means, stds, A, pi, state_sequence) for the best run.
    """
    best_ll = -np.inf
    best_result = None

    for seed in range(n_init):
        rng = np.random.default_rng(seed)
        try:
            means, stds, A, pi, ll = _baum_welch(
                returns, n_states=n_regimes, n_iter=200, rng=rng,
            )
        except Exception:
            continue
        if ll > best_ll:
            best_ll = ll
            best_result = (means, stds, A, pi)

    if best_result is None:
        raise RuntimeError("All Baum-Welch initialisations failed to converge.")

    means, stds, A, pi = best_result

    # Viterbi decode with best parameters
    log_emission = np.column_stack(
        [_log_normal_pdf(returns, means[k], stds[k]) for k in range(n_regimes)]
    )
    states = _viterbi(log_emission, np.log(A + 1e-300), np.log(pi + 1e-300))

    return means, stds, A, pi, states


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def fit_hmm(
    returns: np.ndarray,
    n_regimes: int = 2,
) -> tuple[list[RegimeParams], np.ndarray]:
    """Fit a Hidden Markov Model to classify market regimes.

    Uses hmmlearn.GaussianHMM if available, otherwise falls back to a
    manual Baum-Welch EM implementation.

    Parameters
    ----------
    returns : np.ndarray
        Log returns series (1-D array).
    n_regimes : int
        Number of regimes (default 2: risk-on, risk-off).

    Returns
    -------
    regimes : list[RegimeParams]
        Per-regime parameter estimates.  Convention: regime with
        *higher* std is labelled ``"risk_off"``; lower std is
        ``"risk_on"``.
    state_sequence : np.ndarray
        Most likely regime at each time step (Viterbi path).
    """
    returns = np.asarray(returns, dtype=float).ravel()

    if len(returns) < _MIN_OBS:
        raise ValueError(
            f"Too few observations ({len(returns)}); need >= {_MIN_OBS} "
            "for a reliable HMM fit."
        )

    # NaN/Inf validation — cryptic "collapsed to single state" errors
    # were traced back to NaN contamination in input returns. Fail fast
    # with a clear message instead.
    n_bad = int(np.sum(~np.isfinite(returns)))
    if n_bad > 0:
        raise ValueError(
            f"Input returns contains {n_bad} non-finite values (NaN/Inf) "
            f"out of {len(returns)}. Clean with dropna()/isfinite() before "
            f"calling fit_hmm. NaN in emission PDFs causes Baum-Welch to "
            f"silently degenerate to a 'collapsed state' error."
        )

    if _HAS_HMMLEARN:
        means, stds, transmat, stationary, states = _fit_hmm_hmmlearn(
            returns, n_regimes,
        )
    else:
        warnings.warn(
            "hmmlearn not installed — using manual Baum-Welch EM fallback. "
            "Install hmmlearn for better performance: pip install hmmlearn",
            stacklevel=2,
        )
        means, stds, transmat, stationary, states = _fit_hmm_manual(
            returns, n_regimes,
        )

    # ── Build RegimeParams list ──────────────────────────────
    # Annualise the per-bar sigma.  Assume 5-min bars: 288 bars/day,
    # 365 days (crypto) → ~105_120 bars/year.
    BARS_PER_YEAR = 365 * 24 * 12  # 105_120
    annualised_stds = stds * np.sqrt(BARS_PER_YEAR)

    # Compute steady-state probabilities from transition matrix
    if transmat is not None:
        try:
            eigenvalues, eigenvectors = np.linalg.eig(transmat.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary_dist = np.real(eigenvectors[:, idx])
            stationary_dist = stationary_dist / stationary_dist.sum()
        except Exception:
            stationary_dist = np.ones(n_regimes) / n_regimes
    else:
        stationary_dist = np.ones(n_regimes) / n_regimes

    # Order regimes: risk_off = higher sigma, risk_on = lower sigma
    order = np.argsort(annualised_stds)  # ascending: [low_vol, high_vol]
    labels = ["risk_on", "risk_off"] if n_regimes == 2 else [
        f"regime_{i}" for i in range(n_regimes)
    ]

    regimes: list[RegimeParams] = []
    label_map = {}  # old_state_idx -> new_label_idx for remapping states

    for new_idx, old_idx in enumerate(order):
        label = labels[new_idx] if new_idx < len(labels) else f"regime_{new_idx}"
        sigma_ann = float(annualised_stds[old_idx])
        prob = float(stationary_dist[old_idx])

        # Scale gamma and eta proportionally to sigma.
        # Higher vol regime → larger impact coefficients (wider spreads,
        # thinner books during volatile periods).
        sigma_ratio = sigma_ann  # absolute annualised vol → used as-is

        regimes.append(RegimeParams(
            label=label,
            sigma=sigma_ann,
            gamma=sigma_ratio * 1e-8,   # rough BTC-scale permanent impact
            eta=sigma_ratio * 1e-6,     # rough BTC-scale temporary impact
            probability=prob,
        ))
        label_map[old_idx] = new_idx

    # Remap state sequence so state indices match sorted order
    remapped_states = np.array([label_map[s] for s in states], dtype=int)

    return regimes, remapped_states


def _fit_hmm_hmmlearn(
    returns: np.ndarray,
    n_regimes: int = 2,
    n_init: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit using hmmlearn with multiple random restarts."""
    X = returns.reshape(-1, 1)
    best_score = -np.inf
    best_model = None

    for seed in range(n_init):
        model = _GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=500,
            tol=1e-6,
            random_state=seed,
            min_covar=1e-12,
        )
        try:
            model.fit(X)
            score = model.score(X)
            # Reject degenerate solutions where one state has <5% of data
            counts = np.bincount(model.predict(X), minlength=n_regimes)
            if counts.min() < 0.05 * len(returns):
                continue
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError(
            "All hmmlearn GaussianHMM initialisations collapsed to a "
            "single state. The data may not have distinct regimes, or "
            "try increasing n_init."
        )

    means = best_model.means_.ravel()
    # covars_ shape for "full": (K, n_features, n_features)
    stds = np.sqrt(best_model.covars_.reshape(n_regimes, -1)[:, 0])
    transmat = best_model.transmat_
    stationary = best_model.startprob_
    states = best_model.predict(X)

    return means, stds, transmat, stationary, states


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
        Modified parameters with regime-specific sigma, gamma, and eta.
    """
    return replace(
        base_params,
        sigma=regime.sigma,
        gamma=regime.gamma,
        eta=regime.eta,
    )
