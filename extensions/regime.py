"""Part E: Regime-aware execution via Hidden Markov Model.

Classifies market conditions into regimes (e.g., risk-on vs risk-off)
and provides regime-specific parameters for the execution problem.

In risk-off: higher sigma, wider spreads, larger impact → trade faster.
In risk-on:  lower sigma, tighter spreads → more patient execution.

Implementation strategy:
    - Primary: hmmlearn.GaussianHMM (if installed) — multi-restart Baum-Welch
    - Fallback: Yuhao's rolling-vol-feature EM (pure numpy)

Scaling convention (Yuhao's fix, 2026-04-19):
    sigma, gamma, eta in RegimeParams are dimensionless multipliers (~0.8–1.2x).
    regime_aware_params() multiplies base_params by these ratios.
    This replaces the previous sigma*1e-8 / sigma*1e-6 magic constants
    which were 8 orders of magnitude below calibrated values (audit finding).
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
    """Per-regime execution parameter multipliers plus regime diagnostics.

    sigma, gamma, eta are dimensionless multipliers applied to base ACParams
    (e.g., sigma=0.8 means 80% of baseline volatility for this regime).
    state_vol, state_abs_ret, state_mean_ret are raw return statistics
    for the observations assigned to this regime.
    """

    label: str          # e.g., "risk_on", "risk_off"
    sigma: float        # dimensionless vol multiplier (ratio vs overall vol)
    gamma: float        # dimensionless permanent impact multiplier
    eta: float          # dimensionless temporary impact multiplier
    probability: float  # steady-state probability of this regime
    state_vol: float    # per-bar volatility of returns in this regime
    state_abs_ret: float  # mean |return| in this regime
    state_mean_ret: float  # mean return in this regime


# ═══════════════════════════════════════════════════════════════
# Input validation
# ═══════════════════════════════════════════════════════════════

def _validate_returns(returns: np.ndarray) -> np.ndarray:
    """Convert returns to a clean 1D float array.

    Raises ValueError if any non-finite values are present (NaN/Inf).
    Consistent with main's hard-error NaN validation contract.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)

    if r.size < _MIN_OBS:
        raise ValueError(
            f"Too few observations ({r.size}); need >= {_MIN_OBS} "
            "for a reliable HMM fit."
        )

    n_bad = int(np.sum(~np.isfinite(r)))
    if n_bad > 0:
        raise ValueError(
            f"Input returns contains {n_bad} non-finite values (NaN/Inf) "
            f"out of {len(r)}. Clean with dropna()/isfinite() before "
            f"calling fit_hmm. NaN in emission PDFs causes Baum-Welch to "
            f"silently degenerate to a 'collapsed state' error."
        )

    if np.allclose(r.std(ddof=1), 0.0):
        raise ValueError("Returns have near-zero variance; cannot fit regimes.")

    return r


# ═══════════════════════════════════════════════════════════════
# Yuhao's rolling-vol-feature EM (pure numpy fallback)
# ═══════════════════════════════════════════════════════════════

def _rolling_vol_feature(returns: np.ndarray, window: int) -> np.ndarray:
    """Build a 1D log-volatility feature from returns."""
    n = len(returns)
    if window >= n:
        window = max(5, n // 3)

    vols = np.empty(n - window + 1)
    for i in range(window - 1, n):
        sample = returns[i - window + 1:i + 1]
        vols[i - window + 1] = np.std(sample, ddof=1)

    vols = np.maximum(vols, 1e-12)
    return np.log(vols)


def _normal_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Gaussian log-density for each state."""
    x_col = x[:, None]
    return -0.5 * (
        np.log(2.0 * np.pi * var[None, :]) + (x_col - mean[None, :]) ** 2 / var[None, :]
    )


def _forward_backward(
    log_emission: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward in log space."""
    T, K = log_emission.shape

    log_pi = np.log(np.maximum(pi, 1e-300))
    log_A = np.log(np.maximum(A, 1e-300))

    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_emission[0]

    for t in range(1, T):
        log_alpha[t] = log_emission[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_A,
            axis=0,
        )

    loglik = logsumexp(log_alpha[-1])

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A + log_emission[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1,
        )

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi = np.empty((T - 1, K, K))
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None]
            + log_A
            + log_emission[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - loglik
        )
        xi[t] = np.exp(log_xi_t)

    return gamma, xi, float(loglik)


def _viterbi_numpy(
    log_emission: np.ndarray,
    pi: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Most likely hidden-state sequence (Viterbi, pure numpy)."""
    T, K = log_emission.shape

    log_pi = np.log(np.maximum(pi, 1e-300))
    log_A = np.log(np.maximum(A, 1e-300))

    delta = np.empty((T, K))
    psi = np.empty((T, K), dtype=int)

    delta[0] = log_pi + log_emission[0]
    psi[0] = 0

    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_emission[t]

    states = np.empty(T, dtype=int)
    states[-1] = np.argmax(delta[-1])

    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def _fit_hmm_rolling_vol(
    returns: np.ndarray,
    n_regimes: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit HMM using Yuhao's rolling-vol-feature EM approach.

    Returns (means, stds, A, pi, state_sequence) — where state_sequence
    is aligned back to the original returns length.
    """
    window = min(20, max(10, len(returns) // 20))
    x = _rolling_vol_feature(returns, window=window)

    q_low, q_high = np.quantile(x, [0.3, 0.7])
    means = np.array([q_low, q_high], dtype=float)

    common_var = np.var(x, ddof=1)
    var = np.array([common_var, common_var], dtype=float)
    var = np.maximum(var, 1e-6)

    pi = np.array([0.5, 0.5], dtype=float)
    A = np.array([[0.95, 0.05],
                  [0.05, 0.95]], dtype=float)

    prev_loglik = -np.inf

    for _ in range(max_iter):
        log_emission = _normal_logpdf(x, means, var)
        gamma, xi, loglik = _forward_backward(log_emission, pi, A)

        pi = gamma[0]
        A = xi.sum(axis=0)
        A /= A.sum(axis=1, keepdims=True)

        weights = gamma.sum(axis=0)
        means = (gamma * x[:, None]).sum(axis=0) / np.maximum(weights, 1e-12)

        centered_sq = (x[:, None] - means[None, :]) ** 2
        var = (gamma * centered_sq).sum(axis=0) / np.maximum(weights, 1e-12)
        var = np.maximum(var, 1e-6)

        if abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik

    log_emission = _normal_logpdf(x, means, var)
    states_feature = _viterbi_numpy(log_emission, pi, A)

    # Align feature states back to original returns length
    state_sequence = np.empty(len(returns), dtype=int)
    state_sequence[:window - 1] = states_feature[0]
    state_sequence[window - 1:] = states_feature

    # stds from the feature space; compute actual per-state return stds below
    stds = np.sqrt(var)

    return means, stds, A, pi, state_sequence


# ═══════════════════════════════════════════════════════════════
# Original manual Baum-Welch EM (kept for hmmlearn fallback chain)
# ═══════════════════════════════════════════════════════════════

def _log_normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Log pdf of a univariate Gaussian, broadcast over *x*."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def _forward(log_emission: np.ndarray, log_A: np.ndarray,
             log_pi: np.ndarray) -> tuple[np.ndarray, float]:
    """Forward algorithm in log space."""
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
    log_beta = np.zeros((T, K))

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
    """Baum-Welch EM for a univariate Gaussian HMM."""
    if rng is None:
        rng = np.random.default_rng(42)

    T = len(x)
    K = n_states

    sorted_x = np.sort(x)
    chunk = T // K
    means = np.array([sorted_x[i * chunk:(i + 1) * chunk].mean() for i in range(K)])
    stds = np.array([max(sorted_x[i * chunk:(i + 1) * chunk].std(), 1e-8) for i in range(K)])

    means += rng.normal(0, stds * 0.1, size=K)
    stds = np.abs(stds) + 1e-8

    A = 0.7 * np.eye(K) + 0.3 / K
    A /= A.sum(axis=1, keepdims=True)
    pi = np.ones(K) / K

    prev_ll = -np.inf

    for iteration in range(n_iter):
        log_emission = np.column_stack(
            [_log_normal_pdf(x, means[k], stds[k]) for k in range(K)]
        )
        log_A = np.log(A + 1e-300)
        log_pi = np.log(pi + 1e-300)

        log_alpha, ll = _forward(log_emission, log_A, log_pi)
        log_beta = _backward(log_emission, log_A)

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

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

        pi = gamma[0] / gamma[0].sum()

        for k in range(K):
            gamma_k_sum = gamma[:, k].sum()
            if gamma_k_sum < 1e-10:
                continue
            means[k] = (gamma[:, k] * x).sum() / gamma_k_sum
            diff = x - means[k]
            stds[k] = np.sqrt((gamma[:, k] * diff ** 2).sum() / gamma_k_sum)
            stds[k] = max(stds[k], 1e-8)

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
    """Fit a Gaussian HMM using multiple random restarts of Baum-Welch."""
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

    log_emission = np.column_stack(
        [_log_normal_pdf(returns, means[k], stds[k]) for k in range(n_regimes)]
    )
    states = _viterbi(log_emission, np.log(A + 1e-300), np.log(pi + 1e-300))

    return means, stds, A, pi, states


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
    stds = np.sqrt(best_model.covars_.reshape(n_regimes, -1)[:, 0])
    transmat = best_model.transmat_
    stationary = best_model.startprob_
    states = best_model.predict(X)

    return means, stds, transmat, stationary, states


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def fit_hmm(
    returns: np.ndarray,
    n_regimes: int = 2,
    use_abs_return_for_impact: bool = True,
) -> tuple[list[RegimeParams], np.ndarray]:
    """Fit a Hidden Markov Model to classify market regimes.

    Uses hmmlearn.GaussianHMM if available (multi-restart Baum-Welch,
    n_init=5), otherwise falls back to Yuhao's rolling-vol-feature EM
    (pure numpy). Both paths produce dimensionless multiplier scaling.

    Parameters
    ----------
    returns : np.ndarray
        Log returns series (1-D array).
    n_regimes : int
        Number of regimes (default 2: risk-on, risk-off).
    use_abs_return_for_impact : bool
        If True, use mean |return| ratio for gamma/eta scaling.
        If False, use vol ratio for all three.

    Returns
    -------
    regimes : list[RegimeParams]
        Per-regime parameter multipliers.  Convention: regime with
        *higher* vol is labelled ``"risk_off"``; lower vol is
        ``"risk_on"``.  sigma/gamma/eta are dimensionless multipliers
        (~0.8–1.2x) applied to base ACParams.
    state_sequence : np.ndarray
        Most likely regime at each time step (Viterbi path).
    """
    if n_regimes != 2:
        raise ValueError(
            "This implementation is designed for 2 regimes "
            "(risk_on / risk_off)."
        )

    r = _validate_returns(returns)

    if _HAS_HMMLEARN:
        means_raw, stds_raw, transmat, stationary, states_raw = _fit_hmm_hmmlearn(
            r, n_regimes,
        )
        # hmmlearn state indices may be in any order; we remap below
        # states_raw are on the full returns array r
        state_sequence_raw = np.asarray(states_raw, dtype=int)
    else:
        warnings.warn(
            "hmmlearn not installed — using rolling-vol-feature EM fallback. "
            "Install hmmlearn for better performance: pip install hmmlearn",
            stacklevel=2,
        )
        _means_f, _stds_f, _A_f, _pi_f, state_sequence_raw = _fit_hmm_rolling_vol(
            r, n_regimes,
        )

    # ── Build RegimeParams with dimensionless multipliers ────────────
    overall_vol = np.std(r, ddof=1)
    overall_vol = max(overall_vol, 1e-12)

    overall_abs_ret = np.mean(np.abs(r))
    overall_abs_ret = max(overall_abs_ret, 1e-12)

    overall_mean_ret = np.mean(r)

    # Identify which raw state index corresponds to risk_off (higher vol)
    # by comparing per-state return volatilities.
    vol_per_state = []
    for s in range(n_regimes):
        mask = state_sequence_raw == s
        if mask.sum() < 2:
            vol_per_state.append(overall_vol)
        else:
            vol_per_state.append(float(np.std(r[mask], ddof=1)))

    risk_off_raw = int(np.argmax(vol_per_state))
    risk_on_raw = 1 - risk_off_raw

    # Build label map: raw state index -> 0=risk_on, 1=risk_off
    label_map = {risk_on_raw: 0, risk_off_raw: 1}

    # Remap state sequence to canonical ordering (0=risk_on, 1=risk_off)
    remapped_states = np.array([label_map[s] for s in state_sequence_raw], dtype=int)

    regimes: list[RegimeParams] = []

    for canonical_idx, (raw_state, label) in enumerate(
        [(risk_on_raw, "risk_on"), (risk_off_raw, "risk_off")]
    ):
        mask = state_sequence_raw == raw_state

        if mask.sum() < 2:
            state_vol = overall_vol
            state_abs_ret = overall_abs_ret
            state_mean_ret = overall_mean_ret
        else:
            state_r = r[mask]
            state_vol = float(np.std(state_r, ddof=1))
            state_vol = max(state_vol, 1e-12)

            state_abs_ret = float(np.mean(np.abs(state_r)))
            state_abs_ret = max(state_abs_ret, 1e-12)

            state_mean_ret = float(np.mean(state_r))

        sigma_scale = state_vol / overall_vol

        if use_abs_return_for_impact:
            impact_scale = state_abs_ret / overall_abs_ret
        else:
            impact_scale = sigma_scale

        gamma_scale = impact_scale
        eta_scale = impact_scale * sigma_scale

        probability = float(mask.mean())

        regimes.append(
            RegimeParams(
                label=label,
                sigma=float(sigma_scale),
                gamma=float(gamma_scale),
                eta=float(eta_scale),
                probability=probability,
                state_vol=float(state_vol),
                state_abs_ret=float(state_abs_ret),
                state_mean_ret=float(state_mean_ret),
            )
        )

    return regimes, remapped_states


def regime_aware_params(
    base_params: ACParams,
    regime: RegimeParams,
) -> ACParams:
    """Create regime-specific ACParams by multiplicative scaling.

    regime.sigma / gamma / eta are dimensionless multipliers (~0.8–1.2x).
    The result is base_params with each field scaled by the corresponding
    regime multiplier. This produces O(1) differences between regimes,
    unlike the previous sigma*1e-8 constants which were cosmetically
    ineffective.

    Parameters
    ----------
    base_params : ACParams
        Base parameter set.
    regime : RegimeParams
        Regime-specific multipliers.

    Returns
    -------
    ACParams
        Modified parameters with regime-scaled sigma, gamma, and eta.
    """
    return replace(
        base_params,
        sigma=base_params.sigma * regime.sigma,
        gamma=base_params.gamma * regime.gamma,
        eta=base_params.eta * regime.eta,
    )


def align_states_to_execution_grid(
    state_sequence: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Downsample HMM state sequence to the execution time grid.

    The HMM state sequence has length T_returns (e.g., 8000+ 5-min bars),
    while the execution grid has length n_steps (base_params.N, e.g., 50-200).
    This function maps the longer sequence to the shorter grid by taking the
    mode (rounded mean) of each block.

    Parameters
    ----------
    state_sequence : np.ndarray
        Integer state sequence from fit_hmm(), length = len(returns).
    n_steps : int
        Target execution grid length = base_params.N.

    Returns
    -------
    aligned : np.ndarray, shape (n_steps,), dtype=int
        Regime label for each execution time step.
    """
    block_size = max(len(state_sequence) // n_steps, 1)
    aligned = np.array([
        int(np.round(np.mean(state_sequence[k * block_size:(k + 1) * block_size])))
        for k in range(n_steps)
    ], dtype=int)
    return aligned
