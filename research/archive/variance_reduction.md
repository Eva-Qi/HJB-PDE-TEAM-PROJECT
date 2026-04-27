# Advanced Variance Reduction for Monte Carlo Execution Cost Estimation

**MF796 Course Project — Research Note**
Date: 2026-03-22

---

## 1. Current State — What Is Already Implemented

The project's `montecarlo/sde_engine.py` already implements two standard variance reduction techniques:

### Antithetic Variates

```python
if antithetic:
    n_half = n_paths // 2
    Z = rng.standard_normal((n_half, N))
    Z = np.vstack([Z, -Z])   # mirror paths
```

For any path driven by Brownian increments `Z`, the antithetic path uses `-Z`. Because execution cost is approximately monotone in the price path, `Cost(Z)` and `Cost(-Z)` are negatively correlated. The estimator:

```
C_av = (C(Z) + C(-Z)) / 2
```

has variance `Var(C_av) = (Var(C) + Cov(C(Z), C(-Z))) / 2`. Since `Cov < 0`, this is strictly less than `Var(C) / 2` — better than doubling paths naively.

**Typical variance reduction factor:** 2x-5x for smooth cost functions. Less effective when cost is non-monotone in the price path, which can occur under strategies with stopping rules or adaptive tactics.

### Control Variate (TWAP as Control)

```python
# simulate_execution_with_control_variate in sde_engine.py
C_cv = C_strategy - beta * (C_twap - E[C_twap])
```

where `beta = Cov(C_strategy, C_twap) / Var(C_twap)` is the optimal regression coefficient, and `E[C_twap]` is the known expected cost of TWAP (computed analytically via `execution_cost(twap_x, params)`).

**Key insight:** Both the AC-optimal strategy and TWAP execute in the same price environment, so their costs are highly correlated. The control variate exploits this correlation to cancel common noise.

**Typical variance reduction factor:** 5x-20x when the strategy is close to TWAP (small kappa). Degrades as the optimal strategy diverges from TWAP (large kappa, aggressive front-loading).

### Current Limitation

Both techniques are first-order — they work well for the variance of the mean cost estimator, but do not help much with:
- Estimation of tail quantiles (VaR, CVaR of execution cost)
- High-dimensional problems (many assets, many time steps)
- Problems where the rare events (large adverse price moves during execution) dominate the variance

The sections below assess three advanced techniques.

---

## 2. Importance Sampling — How It Applies to Execution Cost Tails

### The Core Idea

Importance sampling (IS) changes the sampling distribution from the nominal P to an alternative measure Q that puts more probability mass on the rare events of interest. The estimator is reweighted by the likelihood ratio (Radon-Nikodym derivative):

```
E_P[f(X)] = E_Q[f(X) * dP/dQ(X)]
```

If Q is chosen well, `f(X) * dP/dQ(X)` has much lower variance under Q than `f(X)` alone under P.

### Application to Execution Cost Tails

For the Almgren-Chriss setting, the cost of an execution strategy depends on the price path `S_0, S_1, ..., S_N`. The realized cost is:

```
C = sum_k [n_k * (S0 - S_k) + h(v_k) * n_k]
```

The high-cost scenarios occur when prices move adversely (downward for a seller) throughout the execution window. Under GBM, this corresponds to paths where the Brownian increments `Z_k` are systematically negative.

**Optimal IS distribution for CVaR estimation:**

The goal is to estimate `E[C * 1_{C > VaR_alpha}]` (expected shortfall). The optimal importance sampling measure shifts the Brownian drift to make adverse paths more likely. For GBM:

```
dS/S = (mu - theta * sigma) dt + sigma dW_Q
```

where `theta > 0` is the drift shift and `dP/dQ = exp(-theta * W_T - theta^2 * T / 2)` is the likelihood ratio.

**Cross-entropy / exponential tilting approach:**
The optimal IS measure for estimating `E[C | C > c]` is the exponential tilt:

```
dQ*/dP (path) = exp(s * C(path)) / E_P[exp(s * C)]
```

where `s > 0` is chosen to make the expected cost under Q* equal to the target level c. This is the minimum-variance IS distribution for the tail event.

**Practical algorithm:**
1. Run a pilot MC simulation (moderate n_paths) to estimate the cost distribution
2. Find the optimal drift shift theta by solving: `E_Q[C] = target_quantile`
3. Run IS simulation with the tilted measure, reweight by likelihood ratio
4. Estimate tail risk (CVaR) with much smaller variance

### Variance Reduction Potential for This Project

For CVaR estimation at 95th-99th percentile levels:
- Standard MC: variance scales as O(1 / (n * p^2)) where p is the tail probability
- IS with optimal tilting: variance scales as O(1 / n) — eliminates the tail probability penalty

**Expected variance reduction:** 50x-1000x for deep tail quantiles (99th percentile). For the mean cost estimator (not tail), IS is less beneficial — antithetic + control variate are already near-optimal.

### Implementation Sketch

```python
def simulate_execution_IS(params, trajectory_x, theta, n_paths, seed):
    """Importance sampling with drift shift theta for tail estimation."""
    rng = np.random.default_rng(seed)
    N, dt = params.N, params.dt

    Z = rng.standard_normal((n_paths, N))

    # Shifted increments: W_Q = W_P + theta * sqrt(dt)
    Z_shifted = Z + theta * np.sqrt(dt)

    # Likelihood ratio per path: dP/dQ
    log_lr = -theta * np.sum(Z * np.sqrt(dt), axis=1) - 0.5 * theta**2 * params.T
    likelihood_ratio = np.exp(log_lr)

    # Simulate costs under shifted measure (same cost function, different paths)
    _, costs = simulate_execution_from_increments(params, trajectory_x, Z_shifted)

    # IS-corrected mean
    costs_IS = costs * likelihood_ratio
    return costs_IS
```

The challenge is choosing theta optimally. A practical approach: use cross-entropy minimization on a pilot sample.

---

## 3. Quasi-Monte Carlo / Sobol Sequences — Benefits for High-Dimensional Problems

### Standard MC vs QMC

Standard Monte Carlo uses **pseudo-random** points that have statistical properties of randomness but can cluster and leave gaps. The convergence rate is O(1/sqrt(n)) regardless of dimension.

Quasi-Monte Carlo (QMC) uses **low-discrepancy sequences** — deterministic point sets that cover the unit hypercube more uniformly than random points. The convergence rate is O((log n)^d / n) in d dimensions, which beats MC for moderate d.

**Sobol sequences** (Sobol 1967, Joe & Kuo 2010) are the most widely used QMC sequence in finance. They are constructed using binary arithmetic to ensure that every 2^k point block is uniformly distributed in each projection.

### Relevance to Execution Cost Simulation

For the Almgren-Chriss Monte Carlo:
- Each path requires N random normal draws (N = number of time steps)
- With N = 50 time steps and n_paths paths, the total dimensionality is 50
- The cost integrand is smooth (no discontinuities for standard AC strategies)

QMC is most effective when:
1. The integrand is smooth (no kinks, no indicator functions)
2. Dimensionality d is moderate (d <= 50 or so)
3. The effective dimensionality (dimensions that actually matter) is lower than d

For execution cost estimation with fixed trajectories, all N dimensions contribute but the cost is predominantly driven by the aggregate price path level — the effective dimensionality is lower than 50. This is favorable for QMC.

### Implementation with SciPy Sobol Engine

```python
from scipy.stats.qmc import Sobol
from scipy.special import ndtri  # inverse normal CDF

def simulate_execution_qmc(params, trajectory_x, n_paths, seed=42):
    """QMC execution cost estimation using Sobol sequences."""
    N = params.N

    # Generate Sobol points in [0,1]^N
    sampler = Sobol(d=N, scramble=True, seed=seed)
    u = sampler.random(n_paths)     # shape: (n_paths, N), values in (0,1)

    # Transform to standard normals via inverse CDF
    Z = ndtri(u)  # shape: (n_paths, N)

    # Clip to avoid inf at exactly 0 or 1
    Z = np.clip(Z, -6, 6)

    # Simulate paths with QMC increments (same logic as simulate_execution)
    # ... (call simulate_execution_from_increments(params, trajectory_x, Z))

    return costs
```

**Scrambled Sobol** (Owen scrambling) adds a random permutation that preserves low-discrepancy while enabling valid confidence intervals. Always use scrambled Sobol in practice.

### Expected Performance Gains

For d = 50 (N = 50 time steps) and smooth integrand:
- Standard MC: std error ~ 1/sqrt(n)
- QMC (Sobol): effective std error ~ (log n)^50 / n, but in practice for financial applications the effective rate is closer to n^{-0.8} to n^{-1.0}

**Benchmark results from literature (Glasserman 2004, Appendix):**
- For European option pricing (d=1): QMC gives ~10x variance reduction vs MC
- For Asian options (d=252 daily steps): QMC gives ~5x-50x, depending on dimension reduction technique
- For execution cost estimation: expected 5x-20x variance reduction at moderate path counts

**Key caveat:** For n_paths < 1000, Sobol sequences may not have filled in enough of the hypercube to show benefits. The advantage grows with n_paths.

### Combining QMC with Other Techniques

QMC and antithetic variates can be combined: use the n/2 Sobol points AND their negatives in the normal-quantile space. This is "QMC + antithetic" and can double the effective sample size.

QMC and control variates are also compatible: the same Sobol sequence is used for both the strategy and TWAP, maintaining the correlation structure that the control variate exploits.

---

## 4. Milstein Scheme — When It Helps vs Euler-Maruyama

### Euler-Maruyama (Current Implementation)

The current `sde_engine.py` uses exact log-normal simulation for GBM:

```python
# Exact discretization of GBM (no discretization error)
drift = (params.mu - 0.5 * params.sigma**2) * dt
S[:, k+1] = S[:, k] * np.exp(drift + params.sigma * sqrt_dt * Z[:, k])
```

This is actually the **exact** solution for GBM (not an approximation), so there is zero strong discretization error for the price process itself. Euler-Maruyama refers to the first-order scheme for general SDEs that do have discretization error.

### When Milstein Becomes Relevant

The Milstein scheme becomes important when simulating SDEs where the diffusion coefficient is state-dependent (i.e., sigma depends on S or other state variables). It adds a correction term to Euler-Maruyama:

**Euler-Maruyama:**
```
X_{k+1} = X_k + a(X_k) * dt + b(X_k) * sqrt(dt) * Z_k
```

**Milstein:**
```
X_{k+1} = X_k + a(X_k) * dt + b(X_k) * sqrt(dt) * Z_k
                              + 0.5 * b(X_k) * b'(X_k) * (Z_k^2 - 1) * dt
```

The Milstein correction `0.5 * b * b' * (Z^2 - 1) * dt` has strong order 1.0 vs Euler-Maruyama's strong order 0.5. For the same step size dt, Milstein paths are much closer to the true SDE paths.

### Relevance to This Project

**For standard GBM (log-normal price dynamics):** The exact log-normal step already used is superior to both Euler-Maruyama AND Milstein — zero discretization error. Milstein is not needed.

**For extensions where Milstein matters:**

1. **Heston stochastic volatility model** (`extensions/heston.py`):
   The variance process in Heston:
   ```
   dV = kappa*(theta - V)*dt + xi * sqrt(V) * dW_2
   ```
   has `b(V) = xi * sqrt(V)`, so `b'(V) = xi / (2*sqrt(V))`. The Milstein correction is:
   ```
   V_{k+1} = V_k + kappa*(theta-V_k)*dt + xi*sqrt(V_k)*sqrt(dt)*Z
                  + 0.25 * xi^2 * (Z^2 - 1) * dt
   ```
   This reduces the strong error from O(dt^0.5) to O(dt), meaning you can use ~4x fewer time steps for the same path accuracy.

2. **Regime-switching volatility** (`extensions/regime.py`):
   If regime transitions affect the diffusion coefficient, Milstein can improve path accuracy for the switching SDE.

3. **Power-law impact SDEs:**
   If the SDE has state-dependent impact (e.g., impact depends on current price level), Milstein improves accuracy.

### Strong vs Weak Convergence

An important distinction:
- **Strong convergence** (Milstein's advantage): how close individual simulated paths are to the true SDE paths. Relevant when you need path accuracy — e.g., for computing path-dependent quantities like execution cost with adaptive strategies.
- **Weak convergence** (Euler-Maruyama is often sufficient): how close the distribution of outcomes is to the true distribution. For estimating E[cost] with a fixed strategy, weak convergence suffices and Euler-Maruyama with small dt is acceptable.

For the AC model with **deterministic** strategy trajectories (the strategy does not react to the price path), only weak convergence matters for cost estimation. The exact log-normal step already achieves this perfectly.

For **adaptive strategies** that adjust trading in response to observed prices (e.g., liquidation with stopping rules, VWAP adapting to volume), strong convergence matters and Milstein provides benefit.

### Summary for This Project

| Scenario | Current Scheme | Milstein Benefit? |
|---|---|---|
| GBM + deterministic trajectory | Exact (zero error) | No |
| Heston vol + fixed trajectory | Euler on V process | Yes: 4x fewer steps |
| Adaptive strategy (price-sensitive) | Exact GBM paths | No (GBM still exact) |
| Heston + adaptive strategy | Euler on V | Yes: strong order improvement |

**Recommendation:** For the base project (GBM + AC trajectory), Milstein is unnecessary. For the Heston extension, add Milstein to the variance process simulation.

---

## 5. ROI Assessment — Best Variance Reduction per Unit of Effort

### Framework for Comparison

Define "ROI" as: (Variance reduction factor) / (Implementation effort + Runtime overhead).

We measure variance reduction factor as: `Var(standard MC) / Var(improved estimator)` with the same computational budget (wall-clock time, not path count).

### Technique-by-Technique Assessment

#### Control Variate (TWAP) — Already Implemented

- **Variance reduction:** 5x-20x for mean cost estimation (high when strategy close to TWAP)
- **Runtime overhead:** Near zero (one extra simulation with same seed)
- **Implementation effort:** Already done
- **Limitations:** Less effective for tail estimation; degrades as strategy diverges from TWAP
- **ROI: Excellent. Already captured.**

#### Antithetic Variates — Already Implemented

- **Variance reduction:** 2x-5x
- **Runtime overhead:** None (paths generated symmetrically)
- **Limitations:** Only effective for monotone functionals; does not help with variance estimation itself
- **ROI: Excellent. Already captured.**

#### Quasi-Monte Carlo (Sobol) — Not Yet Implemented

- **Variance reduction:** 5x-50x for mean cost (dimension-dependent)
- **Runtime overhead:** ~same as standard MC; Sobol generation is fast
- **Implementation effort:** Low — SciPy provides `scipy.stats.qmc.Sobol`; requires refactoring `simulate_execution` to accept pre-generated increments
- **Compatibility:** Fully compatible with existing control variate
- **Limitations:** Requires smooth integrand; effectiveness degrades for >100 dimensions
- **For this project (N=50):** Expected 10x-30x improvement
- **ROI: High. Best bang-for-buck among the three advanced techniques.**

Concrete code change needed: extract the `Z` generation from `simulate_execution` into a separate function, then inject Sobol-generated Z instead of random Z. Approximately 20 lines of code.

#### Importance Sampling — Not Yet Implemented

- **Variance reduction:** 50x-1000x for tail quantiles (CVaR)
- **Runtime overhead:** Moderate (pilot simulation + optimization to find theta)
- **Implementation effort:** Medium — requires exponential tilting and likelihood ratio computation; about 50-80 lines of new code
- **Limitations:** Primarily useful for tail estimation, not mean cost; requires tuning theta
- **For mean cost estimation:** Limited benefit over existing techniques
- **ROI: High IF the project includes CVaR or tail risk analysis. Low for mean cost alone.**

If the project adds tail risk metrics (e.g., 99th percentile execution cost, or CVaR for risk-adjusted objective), importance sampling is essential.

#### Milstein Scheme — Not Yet Implemented

- **Variance reduction:** None for GBM base model (already exact)
- **Runtime overhead:** ~10% extra FLOPs per step
- **For Heston extension:** Meaningful path accuracy improvement, allows ~4x coarser time grid
- **ROI for base project: Zero. Skip.**
- **ROI for Heston extension: Moderate. Worth adding if Heston paths are needed.**

#### Stratified Sampling — Not Yet Implemented

Stratified sampling divides the sample space into strata and samples a fixed number from each. For a one-dimensional problem (e.g., estimating cost as a function of a single random variable), it achieves O(1/n) convergence rather than O(1/sqrt(n)).

For N=50 time steps, the multidimensional extension is "Latin hypercube sampling" (LHS): each dimension is stratified independently, then combined. SciPy provides `scipy.stats.qmc.LatinHypercube`.

- **Variance reduction:** 2x-5x typically; less than Sobol for smooth integrands
- **Implementation effort:** Same as Sobol (just different sampler)
- **ROI: Moderate. Sobol is superior for smooth integrands; LHS is more robust for rough ones.**

### Priority Ranking for This Project

| Technique | Var Reduction | Effort | ROI | Priority |
|---|---|---|---|---|
| Control variate (TWAP) | 5x-20x | Done | Excellent | Done |
| Antithetic variates | 2x-5x | Done | Excellent | Done |
| QMC / Sobol | 10x-30x | Low | High | **Implement next** |
| Importance sampling | 50x-1000x (tails) | Medium | High (if CVaR needed) | Add for tail analysis |
| Stratified / LHS | 2x-5x | Low | Moderate | Optional |
| Milstein | 0x (base) | Low | Zero (base) | Skip for GBM |

### Concrete Recommendation

**Step 1 (highest ROI):** Refactor `simulate_execution` to accept an optional pre-generated `Z` array. Then add a `simulate_execution_qmc` wrapper that injects Sobol-generated Z. This is the cheapest change with the largest impact on mean cost variance.

**Step 2 (if tail risk is in scope):** Add an importance sampling module for CVaR estimation. Implement exponential tilting with a pilot-based theta search.

**Step 3 (for Heston extension only):** Add Milstein correction to the variance process SDE.

---

## 6. Key References

1. **Glasserman, P. (2004).** *Monte Carlo Methods in Financial Engineering*. Springer, New York.
   - Comprehensive reference. Chapter 4: variance reduction. Chapter 5: importance sampling. Chapter 9: QMC methods. The standard graduate textbook.

2. **Joe, S. & Kuo, F.Y. (2010).** "Constructing Sobol Sequences with Better Two-Dimensional Projections." *SIAM Journal on Scientific Computing*, 30(5), 2635-2654.
   - Modern Sobol construction; implementation in SciPy's `qmc.Sobol` is based on this.

3. **L'Ecuyer, P. & Lemieux, C. (2002).** "Recent Advances in Randomized Quasi-Monte Carlo Methods." In: *Modeling Uncertainty: An Examination of Stochastic Theory, Methods, and Applications*, Kluwer Academic Publishers.
   - Theoretical convergence rates; scrambling methods; practical guidance on when QMC beats MC.

4. **Owen, A.B. (1995).** "Randomly Permuted (t,m,s)-Nets and (t,s)-Sequences." In: *Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing*, Springer.
   - Owen scrambling: the theoretical basis for randomized QMC with valid error bars.

5. **Kloeden, P.E. & Platen, E. (1992).** *Numerical Solution of Stochastic Differential Equations*. Springer.
   - Standard reference for SDE numerics. Chapter 10: Milstein scheme. Strong vs weak convergence theory.

6. **Rubinstein, R.Y. & Kroese, D.P. (2004).** *The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning*. Springer.
   - Cross-entropy method for finding optimal IS distributions; adaptive IS without gradient information.

7. **Broadie, M. & Glasserman, P. (1996).** "Estimating Security Price Derivatives Using Simulation." *Management Science*, 42(2), 269-285.
   - Pathwise estimators and likelihood ratio methods in finance; foundational IS paper.

8. **Paskov, S.H. & Traub, J.F. (1995).** "Faster Valuation of Financial Derivatives." *Journal of Portfolio Management*, 22(1), 113-120.
   - First systematic demonstration that QMC outperforms MC for financial derivatives; inspired finance QMC literature.

9. **Milstein, G.N. (1975).** "Approximate Integration of Stochastic Differential Equations." *Theory of Probability and its Applications*, 19(3), 557-562.
   - Original Milstein scheme paper.

10. **SciPy Documentation — `scipy.stats.qmc`.**
    - `scipy.stats.qmc.Sobol`: scrambled Sobol sequences, `d` dimensions, `n=2^m` points recommended.
    - `scipy.stats.qmc.LatinHypercube`: LHS sampling.
    - Both return uniform `[0,1]^d` samples; transform to normal via `scipy.special.ndtri`.

---

## Appendix: Implementation Checklist

For integrating QMC into the existing codebase:

```python
# In montecarlo/sde_engine.py — add this helper

from scipy.stats.qmc import Sobol
from scipy.special import ndtri

def generate_normal_increments(
    n_paths: int,
    n_steps: int,
    method: str = "pseudo",   # "pseudo", "sobol", "antithetic"
    seed: int = 42,
) -> np.ndarray:
    """Generate standard normal increments for Monte Carlo simulation.

    Parameters
    ----------
    method : "pseudo" | "sobol" | "antithetic"
        "pseudo"    : standard numpy random (current behavior)
        "sobol"     : scrambled Sobol QMC (low discrepancy)
        "antithetic": pseudo-random with antithetic pairing

    Returns
    -------
    Z : np.ndarray, shape (n_paths, n_steps)
        Standard normal increments.
    """
    if method == "pseudo":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_paths, n_steps))

    elif method == "antithetic":
        rng = np.random.default_rng(seed)
        n_half = n_paths // 2
        Z_half = rng.standard_normal((n_half, n_steps))
        return np.vstack([Z_half, -Z_half])

    elif method == "sobol":
        # n_paths must be a power of 2 for optimal Sobol properties
        sampler = Sobol(d=n_steps, scramble=True, seed=seed)
        u = sampler.random(n_paths)           # uniform [0,1]^(n_paths x n_steps)
        Z = ndtri(np.clip(u, 1e-10, 1-1e-10))  # inverse normal CDF
        return Z

    else:
        raise ValueError(f"Unknown method: {method}")
```

Then refactor `simulate_execution` to call `generate_normal_increments` internally with a `method` parameter. This is backward-compatible and adds QMC in ~25 lines.

---

*This note is part of the MF796 course project on Almgren-Chriss optimal execution. It directly informs potential improvements to `montecarlo/sde_engine.py`, which currently implements antithetic variates and control variates.*
