# MF796 Project — Research Q&A: Part C (Monte Carlo) & Part D (Heston Extension)

**Date:** 2026-03-26
**Scope:** Questions 7–12, covering MC permanent impact convention, QMC in high dimensions, CVaR bootstrap, separable ansatz for nonlinear impact, Feller boundary treatment, and Heston calibration from time series.

---

## Q7 — Permanent Impact Self-Inclusion: Standard Convention in Almgren-Chriss

### The Discrepancy

Our MC computes IS = Σ n_k (S₀ − S_k) + h_k·n_k, where h_k is the temporary impact slippage applied at the time of trade k. The deterministic `cost_model` uses γ·Σ(n_k · cumsum_k), which includes the self-impact of trade k on its own fill. The two differ by γΣn_k² (~0.05%).

### What the Original Paper Says

The Almgren and Chriss (2001) paper "Optimal Execution of Portfolio Transactions" defines the discrete price path as:

```
S_k = S_{k-1} + σ·√τ·ε_k − τ·g(n_k/τ)
```

where g(v) = γ·v is the permanent impact function, v = n_k/τ is the trading rate, and the price S_k is updated **after** trade k executes. The cost of each execution is then computed against the **post-impact price** S_k, meaning trade k does affect its own fill price in the standard formulation.

The expected implementation shortfall in Almgren-Chriss (2001), in the discrete-time formulation, reduces to:

```
E[IS] = (1/2)·γ·X² + η_tilde·Σ(n_k²/τ)  +  (drift adjustment)
```

The `(1/2)·γ·X²` permanent impact term is **trajectory-independent** and comes from telescoping the sum Σ n_k·(γ·cumsum of prior trades), but in the strict formulation the price impact of trade k is applied **before** reporting the execution price. This means the standard convention **includes self-impact** in the cumulative permanent impact sum.

### The Arbitrage-Free Convention

A key theoretical result from Almgren-Chriss is that permanent impact must be **linear** (α = 1) to prevent quasi-arbitrage ("price manipulation"). Under this linearity assumption, the entire permanent impact cost collapses to a single trajectory-independent constant (1/2)·γ·X², regardless of whether self-impact is included or excluded — the difference between the two formulations is exactly γΣn_k², which is of order O(γ·(X/N)²·N) = O(γ·X²/N), vanishing as N→∞.

**Conclusion for your code:** The ~0.05% discrepancy you observe is the finite-N discretization artifact of this exact simplification. In the continuous-time limit both conventions converge. The Almgren-Chriss literature does **not** prescribe one finite-step convention over the other as a matter of principle; the difference is immaterial for the optimization since it shifts E[IS] by a constant (γΣn_k²) that is invariant to the trajectory. Your MC convention (excluding self-impact by evaluating cost against the pre-impact price S_0) is consistent with the "arrival price" interpretation of IS and is the convention commonly used in simulation-based implementations. The deterministic formula's self-inclusion is the direct summation form. Neither is "wrong" — document which you use and note the discrepancy is O(γ·X²/N).

### Key Reference

- Almgren, R. and Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), 5–39. [PDF via smallake.kr](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- Anboto Labs (2024). "Deep Dive into IS: The Almgren-Chriss Framework." [Medium](https://medium.com/@anboto_labs/deep-dive-into-is-the-almgren-chriss-framework-be45a1bde831)
- Arthur Bagourd (2022). "A Tale of Two Models: Implementing the Almgren-Chriss framework through nonlinear and dynamic programming." [PDF](https://www.arthur.bagourd.com/wp-content/uploads/2022/08/A_Tale_of_Two_Models__Implementing_the_Almgren_Chriss_framework_through_nonlinear_and_dynamic_programming.pdf)

---

## Q8 — Sobol Sequences in 50 Dimensions: Does QMC Still Beat Pseudo-Random MC?

### Short Answer

**Yes, but with important caveats.** At 50 nominal dimensions, raw (unscrambled) Sobol sequences degrade substantially relative to pseudo-random MC. However, **randomized/scrambled Sobol sequences** with a Brownian Bridge or PCA dimension reduction of the path typically outperform pseudo-random MC by 20–50× in convergence rate for standard finance path integrals.

### The Effective Dimension Problem

The core insight is the concept of **effective dimension**, introduced by Caflisch, Morokoff, and Owen. The theoretical curse of dimensionality for QMC is:

- Koksma-Hlawka bound: error ≤ V(f) · D_N* where D_N* ∝ (log N)^d / N for d-dimensional Sobol.
- At d = 50, the (log N)^50 / N term crosses 1 for practically feasible N values, making the bound vacuous.

**However**, for path simulation of mean-reverting processes (relevant for Almgren-Chriss), the integrand is typically of low **effective truncation dimension**. The first few principal components of the Brownian path carry most of the variance.

### Recommended Remedies for d=50

1. **Brownian Bridge construction**: Reorder the dimensions so the first Sobol dimension controls the endpoint, the second controls the midpoint, etc. This concentrates variance in low-index dimensions where Sobol has the best uniformity.

2. **PCA/KL construction**: Apply PCA to the covariance matrix of the path and order dimensions by eigenvalue. Shown by Acworth, Broadie, and Glasserman (1998) to dramatically reduce effective dimension for Asian options.

3. **Owen scrambling (randomized Sobol)**: Maintains the equidistribution properties of Sobol while restoring statistical validity and improving high-dimension performance. Shown to outperform both crude QMC and standard MC for Asian options at d = 50, 250, 365 dimensions.

### Quantitative Evidence

- Caflisch et al. (1997) showed Asian option pricing (d ~ 360) with Sobol + Brownian Bridge achieves 50–100× speedup over MC.
- Gonçalves, M.V. (SciELO, 2005): "Crude quasi-Monte Carlo errors are too big for d=50 without modifications. Owen's scrambling maintains main characteristics and improves accuracy."
- Lemieux (2009): QMC is "20 to 50 times faster than MC with moderate sample sizes" for finance path problems, but this assumes appropriate dimension ordering.
- Kucherenko et al.: "Effective dimensions much lower than nominal dimensions" is common in finance; global sensitivity analysis (Sobol indices) can identify which dimensions carry most variance.

### Practical Recommendation for Your N=50 Project

For your MC with N=50 time steps:

1. Use `scipy.stats.qmc.Sobol(d=50, scramble=True)` (Owen scrambling is default in scipy 1.7+).
2. Apply Brownian Bridge path construction instead of sequential increments.
3. Validate by comparing convergence rate with pseudo-random; if effective dimension is truly 50, benefit will be modest but still present.
4. If the path integrand is "smooth" (no digital payoffs, no early exercise), QMC gains are near-certain.

### Key References

- Quasi-Monte Carlo methods in finance — [Wikipedia](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_methods_in_finance)
- Quasi-Monte Carlo method — [Wikipedia](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method)
- Gonçalves (2005). "Quasi-Monte Carlo in finance: extending for problems of high effective dimension." *SciELO*. [Link](https://www.scielo.br/j/ecoa/a/7KpwrrdgYxqsG3GRYxjwNRm/?lang=en)
- Caflisch, Morokoff, Owen (1997). [arXiv:1504.02896 related discussion](https://arxiv.org/pdf/1504.02896)
- Kucherenko et al. "The identification of model effective dimensions." [Link](http://www.andreasaltelli.eu/file/repository/ApplicationOfGSI_Kucherenko_RESS.pdf)
- "On the Use of Sobol' Sequence for High Dimensional Simulation." [ResearchGate](https://www.researchgate.net/publication/361466996_On_the_Use_of_Sobol'_Sequence_for_High_Dimensional_Simulation)

---

## Q9 — Bootstrap for CVaR₉₅ Confidence Intervals

### The Statistical Setting

CVaR₉₅ = E[cost | cost ≥ VaR₉₅] is a tail expectation. Its sample estimator is:

```
CVaR_hat = (1 / |{i : c_i >= q_0.95}|) · Σ_{c_i >= q_0.95} c_i
```

This estimator involves both a quantile (VaR) and a conditional mean, so its asymptotic variance is non-trivial.

### Standard Nonparametric Bootstrap vs. Subsampling

**Standard nonparametric bootstrap (iid with replacement)** is the baseline recommendation when:
- The simulation paths are iid (they are, in your MC setup).
- The tail is not extremely heavy (no α < 2 stable distributions).
- N is reasonably large (>1000 paths).

The procedure:
1. Draw B bootstrap resamples of size n with replacement.
2. Compute CVaR₉₅ on each resample.
3. Use percentile bootstrap CI: [CVaR*_(α/2), CVaR*_(1-α/2)] from the empirical distribution of bootstrap CVaR estimates.
4. B = 2000–5000 is standard.

**Subsampling** is preferred when:
- Underlying distribution has heavy tails (α-stable with α < 2, relevant for crypto).
- Bootstrap consistency cannot be verified (non-standard asymptotics).
- Root statistic's limiting distribution is non-Gaussian.

Key theoretical point: For CVaR with very heavy tails, the standard bootstrap can fail because CVaR is sensitive to extreme order statistics and the bootstrap may not resample extreme values correctly with small samples. In this case, subsampling (resample without replacement of size b < n, where b = o(n)) is consistent under weaker conditions.

### Academic Recommendation

Brazauskas, Jones, Zitikis (2009) and Chen (2008) established that:
- For iid data, the **standard percentile bootstrap** for CVaR has O(n^{-1/2}) convergence under finite variance assumptions.
- For heavy-tailed distributions, **subsampling** or **bootstrap with EVT tail correction** (fitting a GPD to exceedances) gives more reliable CIs.

A recent advance (Cambridge Core, 2023): simultaneous confidence bands for CVaR using location-scale models + extreme value theory + novel bootstrap procedure that "circumvents the slow convergence rates of the SCBs."

### For Your Project (iid MC Paths)

Since your MC paths are iid simulated:
- **Use standard nonparametric bootstrap** (B=2000+, percentile method).
- For crypto/high-volatility scenarios where tail behavior is heavy, add a sensitivity check: compare bootstrap CI width with a GPD-based (Pickands-Balkema-de Haan) parametric tail CI.
- The bias-corrected and accelerated (BCa) bootstrap is more accurate than percentile bootstrap for skewed distributions — consider `scipy.stats.bootstrap` with `method='BCa'`.

### Key References

- Acerbi, C. and Tasche, D. (2002). "Expected Shortfall: a natural coherent alternative to Value at Risk." [arXiv](https://arxiv.org/abs/cond-mat/0105191) | [PDF](https://faculty.washington.edu/ezivot/econ589/acertasc.pdf)
- "Simultaneous Confidence Bands for Conditional Value-at-Risk and Expected Shortfall." *Econometric Theory* (2023), 39(5). [Cambridge Core](https://www.cambridge.org/core/journals/econometric-theory/article/simultaneous-confidence-bands-for-conditional-valueatrisk-and-expected-shortfall/36337B7E7E1CDC09EEC7045AD07965C7)
- "The Automated Bias-Corrected and Accelerated Bootstrap Confidence Intervals for Risk Measures." *NAAJ* (2022). [Tandfonline](https://www.tandfonline.com/doi/full/10.1080/10920277.2022.2141781)
- "Cheap Subsampling bootstrap confidence intervals for fast and robust inference." (2025). [arXiv](https://arxiv.org/html/2501.10289)

---

## Q10 — CRITICAL: Does V(x,v,t) = A(v,t)·x² Hold When α≠1?

### Why This Question is Architecturally Critical

The separable ansatz V(x, v, t) = A(v, t)·x² is the cornerstone of the Heston-extension closed-form solution for Part D. If it breaks for α ≠ 1, the entire analytical approach collapses and numerical PDE methods are required.

### The Quadratic Framework Requires α = 1

The separable ansatz V(x, v, t) = A(v, t)·x² works **if and only if** the running cost is quadratic in the control u (trading rate). Here is why:

**Standard Almgren-Chriss Heston HJB equation** (linear temporary impact, α = 1):

Running cost: h(u) = η·u²  (temporary impact cost per unit time)

The HJB equation for value function V(x, v, t) = expected remaining cost:

```
0 = V_t + min_u [ η·u² + u·V_x ] + (1/2)·σ²·v·x²·V_{xx}  (price vol term)
  + κ(θ-v)·V_v + (1/2)·ξ²·v·V_{vv} + ρ·ξ·√v·...·V_{xv}
```

Minimizing over u: u* = -V_x / (2η), substituting back gives:

```
0 = V_t - V_x² / (4η) + (1/2)·σ²·v·x²·V_{xx} + κ(θ-v)·V_v + (1/2)·ξ²·v·V_{vv}
```

With ansatz V = A(v,t)·x² + B(v,t):
- V_x = 2A·x, V_xx = 2A, V_t = A_t·x² + B_t
- The equation separates cleanly into an ODE for A(v,t) and an ODE for B(v,t).
- This yields a **Riccati ODE** for A(v,t) and a **linear PDE** for B(v,t).
- With Heston's affine structure, A(v,t) = a₁(t)·v + a₂(t), giving a system of scalar ODEs — fully solvable.

**When α ≠ 1 (nonlinear temporary impact):**

Running cost: h(u) = η·|u|^(α+1)  (where α is the power law exponent)

The HJB minimization over u gives:

```
u* = -sign(V_x) · |V_x / (η·(α+1))|^(1/α)
```

Substituting back, the optimal cost term becomes:

```
-const · |V_x|^{(α+1)/α}
```

With V = A(v,t)·x²:  V_x = 2A·x, so |V_x|^{(α+1)/α} = |2A·x|^{(α+1)/α} = const · A^{(α+1)/α} · |x|^{(α+1)/α}

This is **NOT quadratic in x** unless α = 1. Therefore the ansatz V = A(v,t)·x² fails for α ≠ 1 — the x-dependence cannot be separated from the v,t dependence in the PDE.

### What the Literature Says

**Almgren (2003)** — "Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk," *Applied Mathematical Finance*, 10(1), 1–18:
- Solves the nonlinear case for **deterministic volatility** (no stochastic vol). The optimal strategy is still deterministic (bang-bang or power law decay in time), but requires solving a nonlinear ODE rather than a linear one.
- The paper does **NOT** extend to stochastic volatility; the closed-form solution is specific to the deterministic setup.
- Reference: [EconPapers](https://ideas.repec.org/a/taf/apmtfi/v10y2003i1p1-18.html) | [arXiv:1111.6826 — Alternative solution method](https://arxiv.org/abs/1111.6826)

**FlowOE (2025)** — "Imitation Learning with Flow Policy from Ensemble RL Experts for Optimal Execution under Heston Volatility and Concave Market Impacts," arXiv:2506.05755:
- Specifically addresses **Heston + concave/nonlinear impact** (α < 1, concave temporary impact).
- Key finding: No closed-form solution exists; the paper uses **reinforcement learning (ensemble RL experts + flow matching imitation)** precisely because the HJB has no analytical tractable form.
- This is the most directly relevant recent paper for your Q10. [arXiv](https://arxiv.org/abs/2506.05755)

**Guéant & Cartea framework (Cartea et al., 2015)**: For market-making/execution with stochastic vol, the quadratic inventory cost structure allows the exponential/quadratic ansatz. For nonlinear impact, numerical methods are standard.

**Portfolio optimization literature (Zariphopoulou 2001, Kraft 2005)**: The separable ansatz for Heston + power utility works because **wealth dynamics are multiplicative** (exponential), not because of impact cost structure. This analogy does not transfer to execution with power-law impact.

### Definitive Answer

**V(x, v, t) = A(v, t)·x² holds if and only if α = 1 (linear temporary impact).**

For α ≠ 1:
- The HJB equation contains a |V_x|^{(α+1)/α} term.
- Since (α+1)/α ≠ 2 for α ≠ 1, the x-dependence is no longer quadratic.
- The ansatz fails; V cannot be written as A(v,t)·x².
- No closed-form solution is known for Heston + power-law impact.

### Architectural Implications for Part D

| Scenario | Approach |
|---|---|
| α = 1 (linear temp impact) | V = A(v,t)·x² separates; Riccati ODE for A(v,t); closed-form possible |
| α ≠ 1 (power law) | No separation; must use numerical PDE (finite differences on x-v grid) or RL |
| α ≠ 1, small deviation from 1 | Perturbation expansion around α=1: V ≈ A₀(v,t)·x² + ε·A₁(v,t)·f(x) |

**Recommendation:** If Part D requires α ≠ 1, either (a) restrict to α = 1 and note the limitation, (b) use a 2D PDE solver on (x, v) grid with Crank-Nicolson or ADI scheme, or (c) use the RL approach from FlowOE as a reference.

### Key References

- Almgren, R. (2003). "Optimal Execution with Nonlinear Impact Functions." *Applied Mathematical Finance*. [EconPapers](https://ideas.repec.org/a/taf/apmtfi/v10y2003i1p1-18.html)
- Li, Y. and Chen, Z. (2025). "FlowOE." [arXiv:2506.05755](https://arxiv.org/abs/2506.05755)
- Optimal Execution: A Review. [KCL](https://kclpure.kcl.ac.uk/portal/files/196524329/Optimal_Execution_Review.pdf)
- Gathering, J. (2010). Optimal Execution lecture notes. [SNS pdf](http://mathfinance.sns.it/wp-content/uploads/2010/12/Gatheral_Optim_Exec.pdf)
- Brokmann et al. (2024). "Tackling nonlinear price impact with linear strategies." [LSE](https://eprints.lse.ac.uk/125888/1/Mathematical_Finance_-_2024_-_Brokmann_-_Tackling_nonlinear_price_impact_with_linear_strategies.pdf)

---

## Q11 — Feller Condition Violation: Boundary Treatment at v=0

### The Feller Condition

For the CIR-type variance process in Heston:

```
dv = κ(θ − v)dt + ξ√v dW_v
```

The **Feller condition** is:

```
2κθ ≥ ξ²
```

When satisfied: variance stays strictly positive; v = 0 is inaccessible.

When violated (2κθ < ξ²): v = 0 is accessible and **strongly reflecting** — the process touches zero recurrently but returns to the interior immediately (in continuous time).

In crypto markets, calibrated Heston parameters frequently violate Feller: large ξ (high vol-of-vol), modest κθ.

### Mathematical Classification via Fichera Theory

The Heston PDE is a **degenerate parabolic** PDE — the diffusion matrix is positive semi-definite but not strictly positive definite at v = 0. Classical elliptic/parabolic theory (requiring uniform ellipticity) does not apply.

**Fichera's theory** (Fichera boundary condition, 1956) classifies whether a boundary condition must be imposed at a degenerate boundary based on the sign of the "Fichera function" b_i - (1/2)Σ_j ∂a_{ij}/∂x_j:

- At v = 0 in the Heston model, the Fichera function analysis gives:
  - If 2κθ ≥ ξ² (Feller satisfied): Fichera function is **non-negative** at v=0 → **no boundary condition is needed or appropriate** at v=0.
  - If 2κθ < ξ² (Feller violated): Fichera function is **negative** at v=0 → **a boundary condition must be imposed**.

This is documented in:
- Canale, Mininni, Rhandi (2017). "Analytic approach to solve a degenerate parabolic PDE for the Heston model." *Mathematical Methods in the Applied Sciences*. [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/mma.4363)
- Feehan & Pop (CPAA 2023). "C^{1,α} regularity for degenerate parabolic equations arising from the Heston model." [AIMS](https://www.aimsciences.org//article/doi/10.3934/cpaa.2023122)

### Treatment Options at v=0 When Feller is Violated

#### Option 1: Neumann (Natural) Boundary — Most Common in Finance

At v = 0, impose ∂V/∂v = 0 (zero flux / reflecting). This is consistent with the strongly reflecting nature of the SDE process:

```
V_v(x, 0, t) = 0
```

This is used by most finite-difference Heston PDE solvers (e.g., In 't Hout & Foulon 2010).

#### Option 2: Analytical Boundary Condition (Dirichlet)

When Feller is violated, the PDE at v=0 degenerates to an ODE in x and t (since the ξ²v term vanishes). Solve this ODE explicitly and use it as a Dirichlet condition:

```
V_t(x, 0, t) + κθ·V_v(x, 0, t) = [reduced equation without v-diffusion]
```

Some solvers implement this as a special row in the finite-difference stencil.

#### Option 3: Fokker-Planck / Reflecting Boundary in Monte Carlo

For MC simulation of the variance path when Feller is violated:

- **Full truncation scheme** (Lord et al., 2010): set v = max(v_sim, 0) at each step. The variance is reflected at 0.
- **Absorption**: set v = 0 permanently if it hits 0 (rarely used; not consistent with CIR dynamics).
- **QE scheme** (Andersen 2008, [SSRN 946405](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405)): quadratic-exponential approximation that handles the Feller violation gracefully without discretization bias.

The HPC-QuantLib blog documents the Fokker-Planck perspective: "If the Feller constraint is violated, the stationary solution of the Fokker-Planck equation becomes hard to track numerically, and the transformed Fokker-Planck equation clearly outperforms the original solution." [HPC-QuantLib](https://hpcquantlib.wordpress.com/2013/05/04/fokker-planck-equation-feller-constraint-and-boundary-conditions/)

#### Option 4: Viscosity Solutions (No Explicit Boundary Condition)

Recent theoretical work avoids imposing any boundary condition by working in the viscosity solutions framework. Feehan & Pop (2012–2023) prove existence and uniqueness of viscosity solutions to the Heston PDE **without** the Feller condition, showing the boundary behavior is encoded in the PDE operator itself. Practically, this means well-posed finite-difference schemes near v=0 without explicit boundary conditions can still be consistent if properly formulated.

- Feehan & Pop (2023). [Springer](https://link.springer.com/article/10.1007/s13398-022-01374-7)
- Non-uniqueness without proper formulation: [arXiv:2511.11288](https://arxiv.org/html/2511.11288)

### Numerical Recommendation for Part D

For your Heston PDE solver in the optimal execution context:

1. **Check Feller condition** for your calibrated parameters.
2. If violated: use the **full truncation + Neumann (∂V/∂v = 0) boundary** at v = 0. This is the de facto standard in computational finance.
3. For MC variance path simulation: use **QE scheme** (Andersen 2008) or **full truncation** — both handle Feller violation reliably.
4. Add a test: run with slightly perturbed parameters that satisfy Feller and verify solutions converge to the same limit.

### Key References

- Heston PDE boundary study (2023). [IMACM Wuppertal preprint](https://www.imacm.uni-wuppertal.de/fileadmin/imacm/preprints/2023/imacm_23_11.pdf)
- Canale et al. (2017). [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/mma.4363)
- HPC-QuantLib blog: "Fokker-Planck Equation, Feller Constraint and Boundary Conditions." [Link](https://hpcquantlib.wordpress.com/2013/05/04/fokker-planck-equation-feller-constraint-and-boundary-conditions/)
- Feehan & Pop regularity results. [AIMS](https://www.aimsciences.org//article/doi/10.3934/cpaa.2023122)
- Andersen (2008). QE scheme for Heston simulation. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405)
- Non-uniqueness in Heston PDE. [arXiv:2511.11288](https://arxiv.org/html/2511.11288)

---

## Q12 — Heston Calibration Without Options Data

### The Core Question

Can we calibrate Heston (κ, θ, ξ, ρ, v₀) from realized volatility time series alone (moment matching on vol-of-vol and mean reversion), without access to Deribit implied vol surface?

### Short Answer

**Yes — it is mathematically feasible and well-studied, but with important limitations.** The key limitation is that time-series calibration identifies the **physical (P) measure** parameters while options prices reflect the **risk-neutral (Q) measure** parameters. For execution purposes (simulating realized price paths), P-measure is actually what you want.

### Available Methods

#### Method 1: GMM on Realized Variance Moments

The Generalized Method of Moments (GMM) matches sample moments of realized variance (and its lags) to theoretical moments implied by the Heston model under the physical measure.

Moment conditions used:
- E[v_t]  (identifies θ)
- Var(v_t)  (identifies ξ²/(2κ) in steady state)
- Cov(v_t, v_{t+h})  (identifies κ via exp(-κh) autocorrelation)
- Cov(r_t², r_{t+h}²)  (identifies ρ and ξ jointly)

Krishnaswamy et al. (2018), "Parameter estimates of Heston stochastic volatility model": Lagged realized volatility satisfies IV conditions; GMM achieves √n consistency. [Preprint](http://scis.scichina.com/en/2018/042202.pdf)

#### Method 2: MLE via Realized Volatility Azencott et al.

Azencott, Ren, Timofeyev (2017). "Realized volatility and parametric estimation of Heston SDEs." [arXiv:1706.04566](https://arxiv.org/pdf/1706.04566):

- Constructs estimators from empirical moments of **realized volatilities** computed over sliding windows.
- Proves convergence with explicit Lq bounds.
- Does NOT require option prices.
- Parameters estimated: κ, θ, ξ.
- ρ can be estimated from cross-variation of log-price and variance increments.
- v₀ is observed (realized var at t=0).

This is the most rigorous time-series-only approach identified in the literature.

#### Method 3: Kalman / Particle Filter (State-Space)

Treat v_t as a latent state. Use:
- **Extended Kalman Filter (EKF)** for fast approximate inference.
- **Unscented Kalman Filter (UKF)** for better nonlinear approximation.
- **Particle Filter (PF)** for exact (up to Monte Carlo error) inference; most expensive but most accurate.

Clayton (2020). "Time-Series Heston Model Calibration Using a Trinomial Tree." [SSRN:3718697](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3718697):

- Demonstrates full calibration from historical price series using a trinomial tree filtering approach.
- Recovers all 5 Heston parameters without options data.

Springer (2025). "Observations concerning the estimation of Heston's stochastic volatility model using HF data." [Link](https://link.springer.com/article/10.1007/s00362-025-01710-0)

#### Method 4: Polynomial Filtering

Springer (2019). "On parameter estimation of Heston's stochastic volatility model: a polynomial filtering method." [Link](https://link.springer.com/article/10.1007/s10203-019-00251-0): Uses polynomial approximations to the nonlinear filter, providing a tractable alternative to particle filters.

### What You Can and Cannot Get from Time-Series Alone

| Parameter | Identifiable from returns alone? | Notes |
|---|---|---|
| κ (mean reversion speed) | Yes | From vol autocorrelation |
| θ (long-run variance) | Yes | From long-run variance mean |
| ξ (vol-of-vol) | Yes | From variance-of-variance |
| ρ (leverage correlation) | Yes (approximately) | From price–variance cross-correlation |
| v₀ (initial variance) | Yes | Directly observable as realized var |

**Key limitation**: Parameters estimated under P-measure ≠ Q-measure parameters. For execution simulation (realized cost), P-measure is correct. For derivative pricing (Greeks hedging), Q-measure (from options) is needed.

### Comparison with Deribit Implied Vol Surface Calibration

| Dimension | Time-Series (P-measure) | Deribit IV Surface (Q-measure) |
|---|---|---|
| Data required | Historical prices / realized vol | Real-time options chain |
| Identifies | Physical dynamics | Risk-neutral pricing measure |
| Appropriate for | Execution simulation, risk | Options pricing, hedging |
| Precision | Moderate (noisy estimators) | High (many option strikes/maturities) |
| Availability | Always available | Requires options market liquidity |

For crypto execution projects:
- Deribit provides BTC/ETH options data with good liquidity across strikes/maturities — use it if available.
- If not available, GMM or Azencott et al. method on realized variance is a valid substitute for simulation purposes.
- Note that Q-measure parameters can have different κ, θ, ξ than P-measure (risk premium adjustment); your execution simulation will be slightly conservative/aggressive depending on the risk premium sign.

### Practical Recommendation

For Part D (Heston extension for optimal execution):

1. **Primary approach**: Use Deribit IV surface if the project targets crypto (BTC/ETH). Calibrate via least-squares on characteristic function prices (standard method, implemented in `calibration/` module).

2. **Fallback / cross-validation**: Calibrate κ and θ from realized variance autocorrelation (GMM). Use ξ from realized variance-of-variance. Set ρ from price-variance correlation. Compare with IV-surface calibration.

3. **Disclosure**: Report which measure you calibrate to and note the P/Q distinction in limitations.

### Key References

- Azencott, Ren, Timofeyev (2017). "Realized volatility and parametric estimation of Heston SDEs." [arXiv:1706.04566](https://arxiv.org/pdf/1706.04566)
- Clayton (2020). "Time-Series Heston Calibration Using Trinomial Tree." [SSRN:3718697](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3718697)
- Germano et al. (2017). "Full and fast calibration of the Heston stochastic volatility model." *EJOR*. [LSE](https://eprints.lse.ac.uk/83754/1/Germano_Full%20and%20fast%20calibration_2017.pdf)
- Springer (2019). Polynomial filtering method. [Springer](https://link.springer.com/article/10.1007/s10203-019-00251-0)
- Observations on HF estimation (2025). [Springer](https://link.springer.com/article/10.1007/s00362-025-01710-0)
- Heston model Wikipedia. [Link](https://en.wikipedia.org/wiki/Heston_model)

---

## Summary Table

| Q# | Question | Key Finding |
|---|---|---|
| Q7 | Self-impact convention | Both conventions differ by γΣn_k² = O(γX²/N), vanishing as N→∞. Neither is "wrong." Document which is used. |
| Q8 | QMC Sobol d=50 | Scrambled Sobol + Brownian Bridge/PCA still beats pseudo-random MC. Raw Sobol degrades; scrambling + dimension ordering is essential. |
| Q9 | Bootstrap for CVaR₉₅ | Standard nonparametric percentile bootstrap (B=2000+) for iid MC paths. Use BCa for skewed distributions. Subsampling for heavy-tailed crypto scenarios. |
| Q10 | Separable ansatz α≠1 | **CRITICAL: V = A(v,t)·x² fails for α≠1.** Separation requires quadratic running cost (η·u²). For power-law impact, no closed form; use 2D PDE solver or RL. |
| Q11 | Feller violation at v=0 | Fichera theory: boundary condition required if Feller violated. Use Neumann (∂V/∂v=0) for PDE; full truncation or QE scheme for MC. |
| Q12 | Heston calibration w/o options | Feasible via GMM on realized variance moments (Azencott 2017) or particle filter. Identifies P-measure params. Deribit IV gives Q-measure (better for pricing; either ok for execution simulation). |

---

*Research compiled 2026-03-26 for MF796 Course Project — Optimal Execution under Heston Stochastic Volatility.*
