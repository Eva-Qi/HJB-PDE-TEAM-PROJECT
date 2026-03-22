# Hidden Markov Model Regime Detection for Optimal Execution

**Course:** MF796 — Computational Methods in Finance
**Topic:** Part E — Regime-Aware Almgren-Chriss Execution
**Date:** 2026-03-22

---

## 1. Why Regime Detection Matters for Execution

The Almgren-Chriss (2001) framework assumes market parameters are *stationary*: volatility `sigma`, permanent impact `gamma`, and temporary impact `eta` are fixed over the execution horizon. In practice, financial markets switch between qualitatively different environments.

### The Risk-On / Risk-Off Intuition

| Regime | Volatility | Bid-Ask Spread | Market Depth | Impact | Optimal Response |
|--------|-----------|---------------|-------------|--------|-----------------|
| **Risk-Off** (stressed) | High | Wide | Shallow | Large | **Trade faster** — reducing time-in-market cuts variance exposure |
| **Risk-On** (calm) | Low | Tight | Deep | Small | **Be patient** — impact costs dominate, spread risk over more intervals |

### Why This Matters Mathematically

Recall the Almgren-Chriss urgency parameter:

```
kappa = sqrt(lambda * sigma^2 / eta)
```

- In **risk-off**: `sigma` rises, `eta` rises (illiquidity). The net effect on `kappa` depends on which increases more.
  - If `sigma^2` grows faster than `eta`: `kappa` increases → front-load more aggressively.
  - If `eta` grows proportionally: `kappa` stays similar but the *absolute cost* is higher, favoring faster exit.
- In **risk-on**: both `sigma` and `eta` compress. `kappa` falls → trajectory flattens toward TWAP.

**Key takeaway:** A single static `kappa` computed from time-averaged parameters gives a trajectory that is systematically wrong — too slow during stress, too aggressive during calm periods.

### Empirical Evidence

Studies of cryptocurrency markets (Makarov & Schoar 2020; Cont & Kukanov 2014) show that:
- BTC/USDT 1-hour realized vol oscillates between roughly 30% annualized (calm) and 150%+ (crisis).
- Bid-ask spreads on Binance expand 3–8× during market stress events.
- Kyle's lambda roughly doubles during high-volatility regimes.

These magnitudes are large enough to meaningfully change the optimal trajectory shape.

---

## 2. HMM vs PELT vs Other Methods

### Method Comparison

| Method | Type | Latency | Online? | Probabilistic? | Regime Count |
|--------|------|---------|---------|---------------|-------------|
| **GaussianHMM** | Probabilistic latent-state | Low (Viterbi is O(T·K²)) | Yes (filtering) | Yes | Fixed K |
| **PELT** (Pruned Exact Linear Time) | Changepoint detection | Moderate O(T log T) | No (retrospective) | No | Inferred |
| **Markov-Switching ARMA** (Hamilton 1989) | Regime-switching time series | Moderate | Yes | Yes | Fixed K |
| **GARCH-based regimes** | Volatility clustering | Low | Yes | No | Usually 2 |
| **k-means on vol features** | Clustering | Fast | No | No | Fixed K |
| **Threshold autoregression (TAR)** | Nonlinear AR | Fast | Yes | No | Fixed K |

### HMM vs PELT in Detail

**PELT (Pruned Exact Linear Time changepoint):**
- Finds *breakpoints* where the distribution of observations changes.
- Globally optimal (minimizes BIC/AIC penalized cost).
- **Cons for execution:**
  - Retrospective only — requires seeing future data to locate a past changepoint.
  - Does not provide a probability of being in a regime *right now*.
  - Does not model regime *persistence* or *transition probabilities*.
  - Cannot filter online: on day t, you don't know if a changepoint occurred at t−5.

**GaussianHMM:**
- The market is in one of K *hidden* states at each time step.
- Within each state, observations follow a Gaussian (or Gaussian mixture) distribution.
- The state transitions follow a Markov chain with transition matrix A.
- **Pros for execution:**
  - Online regime inference via the forward algorithm (O(K²) per new observation).
  - Transition matrix encodes how sticky regimes are (e.g., risk-off persists for days).
  - Produces a *posterior probability* P(state=k | observations), enabling soft regime blending.
  - Baum-Welch (EM) gives maximum-likelihood parameter estimates in closed form for Gaussians.
- **Cons:**
  - K must be pre-specified (use BIC/AIC to select).
  - Assumes Markov property — cannot handle long-memory effects directly.
  - Gaussian emission is misspecified for fat-tailed returns (can mitigate with more components or Student-t HMM).

### Verdict

For **online execution**, GaussianHMM wins. PELT is appropriate for *post-hoc* regime labeling of historical data (e.g., to build training sets or to validate HMM results).

---

## 3. Recommended: 2-State GaussianHMM

### Why 2 States

- **Parsimony:** With limited BTC data (1-hour bars), estimating 3+ regimes leads to unstable Baum-Welch convergence and regime label-switching.
- **Interpretability:** The two-state interpretation maps directly to the risk-on/risk-off dichotomy used in the Almgren-Chriss regime extension.
- **Empirical support:** Studies of equity and crypto vol (Ang & Bekaert 2002; Liu 2015) consistently find two dominant regimes sufficient to capture the bulk of vol variation.
- **BIC criterion:** In practice, fit K=2,3,4 and select the K that minimizes BIC on held-out data. For daily/hourly crypto data K=2 typically wins.

### Model Specification

Let `r_t` be the observation at time t (e.g., log return, or a feature vector).

```
Hidden state: S_t ∈ {0 (risk-on), 1 (risk-off)}

Transition matrix A:
    A[i,j] = P(S_t = j | S_{t-1} = i)
    Typically: A[0,0] ≈ 0.95, A[1,1] ≈ 0.90  (regimes are sticky)

Emission:
    r_t | S_t = k ~ N(mu_k, sigma_k^2)

Or multivariate:
    x_t | S_t = k ~ N(mu_k, Sigma_k)    (x_t is a feature vector)
```

### Regime Identification

After fitting, identify which state is "risk-off" by:
```python
# State with higher emission variance = risk-off
risk_off_state = np.argmax(model.covars_.flatten())
```

For multivariate: use `np.trace(Sigma_k)` or the diagonal element corresponding to the volatility feature.

---

## 4. Feature Selection

What to feed into the HMM emission distribution:

### Option A: Univariate (log returns)

```python
features = log_returns.reshape(-1, 1)
```

- Simple, robust, sufficient for identifying high/low-vol regimes.
- Drawback: mean return differences between regimes are small; signal mainly in variance.

### Option B: Realized Volatility (recommended for execution)

```python
# Rolling 1-hour realized vol using 5-minute bars
rv = rolling_std(log_returns, window=12)  # 12 × 5min = 1hr
features = rv.reshape(-1, 1)
```

- Directly targets the volatility signal relevant to `sigma` in Almgren-Chriss.
- Smoother than raw returns; better separation between regimes.

### Option C: Multi-Feature Vector (most informative)

```python
features = np.column_stack([
    realized_vol,       # sigma proxy → maps to AC sigma
    log_spread_bps,     # bid-ask spread → maps to eta
    order_flow_imb,     # OFI = (buy_volume - sell_volume) / total_volume
    log_volume,         # trading volume
])
```

**Feature justification:**
- `realized_vol`: primary driver of execution risk term `lambda * sigma^2 * sum(x_k^2 * tau)`.
- `log_spread_bps`: proxy for `eta` (temporary impact). Wide spread → higher cost per unit traded.
- `order_flow_imbalance (OFI)`: Cont, Kukanov & Stoikov (2014) show OFI predicts short-term price impact; high OFI indicates directional pressure that increases adverse selection cost.
- `log_volume`: low-volume periods have worse market depth → higher impact per trade.

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
```

Critical: standardize features so that the HMM covariance matrix is well-conditioned and Baum-Welch converges smoothly.

### Feature Correlation Warning

OFI and log_volume are correlated. If using multivariate HMM, check condition number of per-state covariance matrices and consider PCA pre-processing:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X_scaled)
```

---

## 5. Implementation with hmmlearn

### Installation

```bash
pip install hmmlearn>=0.3.0
```

### Full Code Sketch

```python
"""
HMM regime detection for Almgren-Chriss execution.
Fits a 2-state GaussianHMM on realized volatility features
and extracts per-regime sigma, gamma, eta estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from dataclasses import replace

from shared.params import ACParams


# ─────────────────────────────────────────────────────────────
# Step 1: Feature engineering
# ─────────────────────────────────────────────────────────────

def build_features(
    prices: np.ndarray,
    spreads_bps: np.ndarray | None = None,
    ofi: np.ndarray | None = None,
    vol_window: int = 12,
) -> np.ndarray:
    """
    Build HMM feature matrix from price series and optional microstructure data.

    Parameters
    ----------
    prices : np.ndarray
        Mid-price time series (e.g., 5-minute bars).
    spreads_bps : np.ndarray, optional
        Bid-ask spread in basis points, same length as prices.
    ofi : np.ndarray, optional
        Order flow imbalance ∈ [-1, 1], same length as prices.
    vol_window : int
        Rolling window for realized vol (default 12 = 1 hour at 5-min bars).

    Returns
    -------
    np.ndarray, shape (T - vol_window, n_features)
        Standardized feature matrix ready for hmmlearn.
    """
    log_returns = np.diff(np.log(prices))

    # Rolling realized volatility (annualized)
    T = len(log_returns)
    rv = np.array([
        log_returns[max(0, t - vol_window):t].std()
        for t in range(vol_window, T + 1)
    ])
    # Annualize: 5-min bars → 288 per day → 252 trading days
    rv_ann = rv * np.sqrt(288 * 252)

    feature_list = [rv_ann]

    if spreads_bps is not None:
        # Align: drop first vol_window rows
        spread_aligned = spreads_bps[vol_window:]
        feature_list.append(np.log1p(spread_aligned))

    if ofi is not None:
        ofi_aligned = ofi[vol_window:]
        feature_list.append(ofi_aligned)

    X = np.column_stack(feature_list)

    # Standardize
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler, rv_ann


# ─────────────────────────────────────────────────────────────
# Step 2: Fit GaussianHMM
# ─────────────────────────────────────────────────────────────

def fit_gaussian_hmm(
    X: np.ndarray,
    n_components: int = 2,
    n_iter: int = 200,
    random_state: int = 42,
    covariance_type: str = "full",
) -> hmm.GaussianHMM:
    """
    Fit a GaussianHMM using Baum-Welch (EM).

    Parameters
    ----------
    X : np.ndarray, shape (T, n_features)
        Standardized feature matrix.
    n_components : int
        Number of hidden states (default 2).
    n_iter : int
        Maximum EM iterations.
    covariance_type : str
        One of "full", "diag", "tied", "spherical".
        Use "diag" for small datasets; "full" when T >> n_features^2.

    Returns
    -------
    model : GaussianHMM
        Fitted model with .means_, .covars_, .transmat_, .startprob_.
    """
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
        verbose=False,
    )
    model.fit(X)
    return model


# ─────────────────────────────────────────────────────────────
# Step 3: Decode regime sequence
# ─────────────────────────────────────────────────────────────

def decode_regimes(
    model: hmm.GaussianHMM,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode most likely state sequence (Viterbi) and posterior probabilities.

    Returns
    -------
    state_sequence : np.ndarray, shape (T,)
        Most likely hidden state at each timestep (argmax of Viterbi path).
    posterior_probs : np.ndarray, shape (T, n_components)
        Smoothed posterior P(S_t = k | X_{1:T}) from forward-backward.
    """
    state_sequence = model.predict(X)
    posterior_probs = model.predict_proba(X)
    return state_sequence, posterior_probs


def identify_risk_off_state(
    model: hmm.GaussianHMM,
    vol_feature_idx: int = 0,
) -> int:
    """
    Identify which HMM state corresponds to 'risk-off' (high-vol regime).

    Uses the mean of the volatility feature in each state's emission Gaussian.
    The state with the highest mean on the vol feature = risk-off.
    """
    vol_means = model.means_[:, vol_feature_idx]
    return int(np.argmax(vol_means))


# ─────────────────────────────────────────────────────────────
# Step 4: BIC-based model selection
# ─────────────────────────────────────────────────────────────

def select_n_components(
    X: np.ndarray,
    max_k: int = 5,
    n_iter: int = 200,
) -> int:
    """
    Select number of HMM states by BIC.

    BIC = -2 * log_likelihood + n_params * log(T)
    """
    T, n_features = X.shape
    bic_scores = []

    for k in range(2, max_k + 1):
        model = hmm.GaussianHMM(
            n_components=k,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=42,
        )
        model.fit(X)
        log_lik = model.score(X)

        # Number of free parameters:
        # startprob: k-1
        # transmat: k*(k-1)
        # means: k * n_features
        # covars (diag): k * n_features
        n_params = (k - 1) + k * (k - 1) + k * n_features + k * n_features
        bic = -2 * log_lik * T + n_params * np.log(T)
        bic_scores.append((k, bic, model))
        print(f"  k={k}: log_lik={log_lik:.2f}, BIC={bic:.1f}")

    best_k, best_bic, _ = min(bic_scores, key=lambda x: x[1])
    print(f"  Best k={best_k} (BIC={best_bic:.1f})")
    return best_k
```

### Online Filtering (for live execution)

```python
def online_regime_filter(
    model: hmm.GaussianHMM,
    new_observation: np.ndarray,
    prev_alpha: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Single-step forward update for online regime inference.

    Uses the HMM forward algorithm: given previous filtered probabilities
    alpha_{t-1} and a new observation x_t, compute alpha_t.

    Parameters
    ----------
    model : GaussianHMM
        Fitted model.
    new_observation : np.ndarray, shape (1, n_features)
        New standardized feature vector.
    prev_alpha : np.ndarray, shape (n_components,)
        Previous step's filtered probability vector.

    Returns
    -------
    alpha_t : np.ndarray
        Updated filtered probabilities P(S_t | x_{1:t}).
    current_regime : int
        Most likely regime (argmax of alpha_t).
    """
    from scipy.stats import multivariate_normal

    A = model.transmat_          # (K, K) transition matrix
    K = model.n_components

    # Emission probabilities b_k(x_t)
    b = np.array([
        multivariate_normal.pdf(
            new_observation.flatten(),
            mean=model.means_[k],
            cov=model.covars_[k],
        )
        for k in range(K)
    ])

    # Forward update: alpha_t ∝ b(x_t) * (A^T @ alpha_{t-1})
    alpha_t = b * (A.T @ prev_alpha)
    alpha_t /= alpha_t.sum()  # normalize

    return alpha_t, int(np.argmax(alpha_t))
```

---

## 6. Per-Regime Parameter Estimation

Once the Viterbi state sequence `state_sequence` is available, estimate regime-specific Almgren-Chriss parameters by subsetting the historical data.

### Regime-Specific Volatility

```python
def estimate_regime_vol(
    log_returns: np.ndarray,
    state_sequence: np.ndarray,
    regime: int,
    annualize_factor: float = np.sqrt(288 * 252),  # 5-min bars
) -> float:
    """Realized vol for observations belonging to a given regime."""
    mask = (state_sequence == regime)
    # Align: state_sequence has length T - vol_window; returns has length T-1
    # Must align indices carefully
    regime_returns = log_returns[mask]
    if len(regime_returns) < 30:
        raise ValueError(f"Too few observations ({len(regime_returns)}) for regime {regime}")
    return float(regime_returns.std() * annualize_factor)
```

### Regime-Specific Impact Parameters

Estimate `gamma` and `eta` by running Kyle's lambda and temporary impact regressions on regime-filtered trade data:

```python
def estimate_regime_impacts(
    delta_prices: np.ndarray,
    signed_flows: np.ndarray,
    spreads_bps: np.ndarray,
    state_sequence: np.ndarray,
    regime: int,
) -> tuple[float, float]:
    """
    Estimate gamma (permanent) and eta (temporary) for a specific regime.

    gamma: Kyle's lambda on regime-filtered trades.
    eta:   Mean spread / 2 in basis points, converted to price units.
           (Simplified proxy; production code should use walk-the-book.)

    Returns
    -------
    (gamma, eta) : tuple[float, float]
    """
    mask = (state_sequence == regime)
    dp_regime = delta_prices[mask]
    flow_regime = signed_flows[mask]
    spread_regime = spreads_bps[mask]

    # Kyle's lambda → gamma
    from calibration.impact_estimator import estimate_kyle_lambda
    gamma = estimate_kyle_lambda(dp_regime, flow_regime)
    if gamma is None:
        raise ValueError(f"Kyle lambda estimation failed for regime {regime}")

    # Temporary impact: half-spread as proxy for eta
    # In AC framework: h(v) = eta * v, eta has units of $/share per (share/time)
    # Rough proxy: eta ≈ (mean_spread / 2) in price units / (ADV / T)
    eta = float(np.mean(spread_regime) * 1e-4 / 2)  # bps → fraction, half-spread

    return gamma, eta
```

### Transition Matrix → Regime Duration

```python
def regime_statistics(model: hmm.GaussianHMM, risk_off_state: int) -> dict:
    """
    Extract regime persistence and steady-state probability.

    Expected duration of regime k:
        E[T_k] = 1 / (1 - A[k,k])

    Steady-state probability (stationary distribution pi):
        pi = left eigenvector of A corresponding to eigenvalue 1
    """
    A = model.transmat_
    K = model.n_components

    # Expected durations (in time steps)
    durations = {k: 1.0 / (1.0 - A[k, k]) for k in range(K)}

    # Stationary distribution via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = eigenvectors[:, stationary_idx].real
    pi /= pi.sum()

    return {
        "transition_matrix": A,
        "expected_duration_steps": durations,
        "stationary_probs": {k: pi[k] for k in range(K)},
        "risk_off_persistence": A[risk_off_state, risk_off_state],
    }
```

---

## 7. Regime-Conditional Execution

### Connecting HMM to Almgren-Chriss

The key interface is `extensions/regime.py`. After fitting the HMM:

```python
from dataclasses import replace
from extensions.regime import RegimeParams, regime_aware_params
from shared.params import ACParams, DEFAULT_PARAMS
from shared.cost_model import objective
import numpy as np


def build_regime_params(
    base_params: ACParams,
    sigma_risk_on: float,
    sigma_risk_off: float,
    gamma_risk_on: float,
    gamma_risk_off: float,
    eta_risk_on: float,
    eta_risk_off: float,
    pi_risk_on: float,
    pi_risk_off: float,
) -> tuple[RegimeParams, RegimeParams]:
    """Construct RegimeParams objects from estimated per-regime values."""
    risk_on = RegimeParams(
        label="risk_on",
        sigma=sigma_risk_on,
        gamma=gamma_risk_on,
        eta=eta_risk_on,
        probability=pi_risk_on,
    )
    risk_off = RegimeParams(
        label="risk_off",
        sigma=sigma_risk_off,
        gamma=gamma_risk_off,
        eta=eta_risk_off,
        probability=pi_risk_off,
    )
    return risk_on, risk_off


def regime_aware_params(base_params: ACParams, regime: RegimeParams) -> ACParams:
    """Override base AC params with regime-specific values."""
    return replace(
        base_params,
        sigma=regime.sigma,
        gamma=regime.gamma,
        eta=regime.eta,
    )


# ──────────────────────────────────────────────────────────────
# Example: regime-switching execution at decision time
# ──────────────────────────────────────────────────────────────

def get_current_regime_params(
    base_params: ACParams,
    risk_on: RegimeParams,
    risk_off: RegimeParams,
    current_regime_label: str,
) -> ACParams:
    """
    Return ACParams for the currently detected regime.

    In production: call with the output of online_regime_filter().
    In research/backtesting: call with Viterbi-decoded state label.
    """
    if current_regime_label == "risk_on":
        return regime_aware_params(base_params, risk_on)
    elif current_regime_label == "risk_off":
        return regime_aware_params(base_params, risk_off)
    else:
        raise ValueError(f"Unknown regime: {current_regime_label}")


def compare_regime_trajectories(
    base_params: ACParams,
    risk_on: RegimeParams,
    risk_off: RegimeParams,
) -> dict:
    """
    Compare kappa and optimal cost across regimes.

    Shows quantitatively why regime-awareness changes execution speed.
    """
    from shared.params import almgren_chriss_closed_form

    results = {}
    for regime in [risk_on, risk_off]:
        p = regime_aware_params(base_params, regime)
        kappa = p.kappa
        t_grid, x_traj, cost = almgren_chriss_closed_form(p)
        results[regime.label] = {
            "kappa": kappa,
            "kappa_T": kappa * p.T,
            "expected_cost": cost,
            "trajectory_shape": "front-loaded" if kappa * p.T > 1.0 else "TWAP-like",
        }
    return results
```

### Soft Regime Blending (Advanced)

Instead of hard switching, use posterior probabilities to interpolate parameters:

```python
def soft_blend_params(
    base_params: ACParams,
    risk_on: RegimeParams,
    risk_off: RegimeParams,
    alpha_risk_off: float,  # posterior P(state=risk_off | observations)
) -> ACParams:
    """
    Compute a convex blend of per-regime parameters weighted by
    current regime posterior probability.

    This avoids abrupt trajectory jumps at regime boundaries.

    Parameters
    ----------
    alpha_risk_off : float
        P(state = risk_off | data so far), output of online forward filter.
    """
    alpha_on = 1.0 - alpha_risk_off

    blended_sigma = alpha_on * risk_on.sigma + alpha_risk_off * risk_off.sigma
    blended_gamma = alpha_on * risk_on.gamma + alpha_risk_off * risk_off.gamma
    blended_eta   = alpha_on * risk_on.eta   + alpha_risk_off * risk_off.eta

    return replace(
        base_params,
        sigma=blended_sigma,
        gamma=blended_gamma,
        eta=blended_eta,
    )
```

### Integration Diagram

```
Raw market data (prices, spreads, OFI, volume)
        │
        ▼
Feature engineering (realized_vol, log_spread, OFI)
        │
        ▼
Standardize → GaussianHMM.fit(X_train)
        │
        ├─── Viterbi decode → historical regime labels
        │         └── Per-regime parameter estimation (sigma_k, gamma_k, eta_k)
        │
        └─── Online forward filter → P(S_t | X_{1:t})
                  │
                  ▼
         regime_aware_params(base_params, regime_k)   OR
         soft_blend_params(base_params, risk_on, risk_off, alpha_t)
                  │
                  ▼
         ACParams with regime-adjusted (sigma, gamma, eta)
                  │
                  ▼
         almgren_chriss_closed_form(params)  → x*(t)
         or PDE / Monte Carlo solver         → x*(t)
                  │
                  ▼
         Execute at regime-optimal trading speed
```

---

## 8. Key References

### Foundational

1. **Almgren, R. & Chriss, N. (2001).** Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.
   - The baseline framework. `sigma`, `gamma`, `eta`, `lambda` → optimal `kappa`.

2. **Hamilton, J. D. (1989).** A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
   - Original Markov-switching regime model. Theoretical foundation for HMM applied to macro and financial series.

3. **Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970).** A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. *The Annals of Mathematical Statistics*, 41(1), 164–171.
   - The Baum-Welch EM algorithm for HMM parameter estimation.

### Market Microstructure & Regime

4. **Kyle, A. S. (1985).** Continuous auctions and insider trading. *Econometrica*, 53(6), 1315–1335.
   - Kyle's lambda as the permanent impact parameter `gamma`.

5. **Cont, R., Kukanov, A., & Stoikov, S. (2014).** The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47–88.
   - OFI as an execution cost predictor; regime-conditional impact estimates.

6. **Ang, A. & Bekaert, G. (2002).** Regime switches in interest rates. *Journal of Business and Economic Statistics*, 20(2), 163–182.
   - Evidence for 2-state regime models in financial markets; BIC selection.

### Crypto & HMM

7. **Liu, Y. (2015).** Stock market volatility and regime change. *Journal of Economic Dynamics and Control*, 53, 149–163.
   - 2-state GaussianHMM applied to equity vol; validates risk-on/risk-off dichotomy.

8. **Makarov, I. & Schoar, A. (2020).** Trading and arbitrage in cryptocurrency markets. *Journal of Financial Economics*, 135(2), 293–319.
   - BTC microstructure; vol regimes and spread behavior.

### Software & Methods

9. **Weiss, R. et al. (2023).** hmmlearn: Hidden Markov Models in Python. *Journal of Open Source Software*.
   - `pip install hmmlearn`. GaussianHMM, GMMHMM, MultinomialHMM.
   - Docs: https://hmmlearn.readthedocs.io/

10. **Killick, R., Fearnhead, P., & Eckley, I. A. (2012).** Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590–1598.
    - The PELT algorithm. Python implementation: `ruptures` library.

11. **Rabiner, L. R. (1989).** A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
    - Comprehensive HMM tutorial: Baum-Welch, Viterbi, forward-backward algorithms. Classic reference.

---

## Appendix: Baum-Welch Algorithm (Summary)

The Baum-Welch algorithm is an **Expectation-Maximization (EM) procedure** for fitting HMM parameters `theta = (pi, A, B)` to observed data `X = (x_1, ..., x_T)`.

### E-Step: Forward-Backward

**Forward variable** `alpha_t(k)` = P(x_1, ..., x_t, S_t = k | theta):

```
alpha_1(k) = pi_k * b_k(x_1)
alpha_t(k) = b_k(x_t) * sum_j [ alpha_{t-1}(j) * A[j,k] ]
```

**Backward variable** `beta_t(k)` = P(x_{t+1}, ..., x_T | S_t = k, theta):

```
beta_T(k) = 1
beta_t(k) = sum_j [ A[k,j] * b_j(x_{t+1}) * beta_{t+1}(j) ]
```

**Posterior**:
```
gamma_t(k) = P(S_t = k | X, theta) = alpha_t(k) * beta_t(k) / sum_k [alpha_t(k) * beta_t(k)]
xi_t(j,k)  = P(S_t=j, S_{t+1}=k | X, theta)  [transition posterior]
```

### M-Step: Parameter Update (Gaussian emissions)

```
pi_k   = gamma_1(k)
A[j,k] = sum_t xi_t(j,k) / sum_t gamma_t(j)
mu_k   = sum_t gamma_t(k) * x_t / sum_t gamma_t(k)
Sigma_k = sum_t gamma_t(k) * (x_t - mu_k)(x_t - mu_k)^T / sum_t gamma_t(k)
```

Iterate E and M steps until log-likelihood converges. hmmlearn handles this automatically via `model.fit(X)`.

### Viterbi Decoding

Finds the single most probable state sequence (not the marginals):

```
delta_t(k) = max_{s_1,...,s_{t-1}} P(x_1,...,x_t, s_1,...,s_{t-1}, S_t=k | theta)
delta_t(k) = b_k(x_t) * max_j [ delta_{t-1}(j) * A[j,k] ]
```

Backtrack from `argmax_k delta_T(k)` to recover full state path. This is the `model.predict(X)` output in hmmlearn.

---

*Generated for MF796 Course Project — Part E: Regime-Aware Execution*
