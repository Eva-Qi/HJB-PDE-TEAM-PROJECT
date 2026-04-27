# Nonlinear Market Impact: Theory, Evidence, and Implications for Optimal Execution

**MF796 Course Project — Research Note**
Date: 2026-03-22

---

## 1. Linear vs Nonlinear Impact — What Changes When alpha != 1

### The Standard Almgren-Chriss Setup (alpha = 1)

The baseline Almgren-Chriss (2001) model parameterizes temporary market impact as a linear function of trading rate:

```
h(v) = eta * v
```

where `v = n_k / tau` is the trading rate (shares per unit time) and `eta` is the temporary impact coefficient. This linear form yields a clean closed-form optimal trajectory — the hyperbolic sine solution `x*(t) = X0 * sinh(kappa*(T-t)) / sinh(kappa*T)` — because the Hamilton-Jacobi-Bellman equation reduces to a linear ODE in this case.

### Generalization to Power-Law Impact

The generalized model (Almgren 2003) replaces linear temporary impact with a power law:

```
h(v) = eta * |v|^alpha * sign(v)
```

where `alpha in (0, 2]` is the impact exponent. The project's `cost_model.py` already implements this form:

```python
def temporary_impact(v, eta, alpha=1.0):
    if abs(alpha - 1.0) < 1e-10:
        return eta * v
    return eta * np.abs(v) ** alpha * np.sign(v)
```

### What Changes Structurally When alpha != 1

**Objective function.** The expected cost becomes:
```
E[C] = gamma * X0^2 / 2  +  eta * sum_k |v_k|^alpha * n_k
     = gamma * X0^2 / 2  +  eta * tau^(1-alpha) * sum_k |n_k|^(1+alpha)
```

The second term is now a sum of power-law terms. For alpha < 1, the exponent `1 + alpha < 2`, meaning the cost grows sublinearly with trade size — spreading a trade out saves less than in the linear case. For alpha > 1, the reverse: large trades are penalized superlinearly, creating stronger incentive to spread.

**Optimality conditions.** The Euler-Lagrange condition (first-order necessary condition) for the continuous-time limit becomes:

```
(1 + alpha) * eta * |v|^(alpha-1) * dv/dt = lambda * sigma^2 * x
```

For alpha = 1 this simplifies to `eta * dv/dt = lambda * sigma^2 * x`, which has the sinh solution. For alpha != 1 it is a nonlinear ODE with no closed form in general.

**No closed-form solution.** The critical practical consequence: the project's `almgren_chriss_closed_form()` function explicitly raises `ValueError` when `alpha != 1.0`. For nonlinear impact, optimal trajectories must be computed numerically — either via the HJB PDE solver (`pde/hjb_solver.py`) or by direct optimization.

**Trajectory shape.** The optimal trajectory changes qualitatively:
- alpha < 1 (concave impact): more uniform trading; the marginal cost of trading faster is lower at high rates, so the optimizer front-loads less aggressively
- alpha > 1 (convex impact): strong front-loading; high-rate trades are penalized heavily, so the optimizer tries to avoid them by trading early at moderate rates

---

## 2. Concave Impact (alpha < 1) — Why It Is More Realistic, How Trajectory Changes

### The Economic Argument for Concavity

A concave impact function `h(v) ~ v^alpha` with `alpha < 1` means that doubling the trading rate less than doubles the per-share market impact. This is economically sensible because:

1. **Order book depth is heterogeneous.** The bid-ask spread is thin near the mid-price (resting limit orders close to mid), but thinner still for small trades. A large block trade sweeps through multiple price levels, encountering progressively worse liquidity at each level. However, the *marginal* cost of the last share in the block is higher than the average — this sounds like convex impact at the order level.

2. **At the execution strategy level (coarser time scale)**, what matters is how realized market impact scales with the order flow rate over intervals of minutes or hours. Empirically, this scaling is concave (see Section 3). The intuition: market makers partially accommodate increased flow by widening spreads modestly and increasing depth, so proportional impact decreases with scale.

3. **Square-root law.** The most widely cited empirical regularity is the square-root impact law:
   ```
   Delta S  ~  sigma * sqrt(Q / V_daily)
   ```
   where `Q` is order size and `V_daily` is daily volume. This corresponds to `alpha = 0.5` in the Almgren framework when expressed in trading rate terms.

### Trajectory Changes Under Concave Impact

For alpha < 1, the marginal cost of trading at rate v is:
```
d/dv [h(v)] = alpha * eta * v^(alpha-1)
```

This is *decreasing* in v (for alpha < 1, `alpha - 1 < 0`). The implication: at high trading rates, the marginal impact is *lower*, not higher. The optimizer faces a qualitatively different tradeoff:

- Under linear impact (alpha = 1): doubling trading rate doubles instantaneous cost — strong incentive to smooth trading.
- Under concave impact (alpha < 1): doubling trading rate less-than-doubles cost — weaker incentive to smooth. The optimizer is willing to trade faster during periods of high remaining inventory, but because marginal impact is decreasing, the optimal policy tends toward **more uniform trading** compared to the hyperbolic front-loading of the linear case.

**Quantitative example.** For alpha = 0.5 (square root), Almgren (2003) shows that the optimal trajectory satisfies:
```
d/dt [|v|^(-0.5) * v] = 2 * lambda * sigma^2 * x / eta
```
The solution is approximately parabolic rather than hyperbolic, with less initial front-loading and a longer tail toward the deadline.

**Risk-urgency tradeoff under concave impact.** In the linear model, increasing risk aversion lambda monotonically increases front-loading (larger kappa). Under concave impact, this relationship is preserved qualitatively but the sensitivity is reduced — the optimizer has less scope to reduce variance by concentrating early trades because early large trades are only marginally cheaper, not proportionally cheaper.

---

## 3. Empirical Evidence — What Alpha Values Are Observed

### Key Empirical Studies

**Almgren, Thum, Hauptmann & Li (2005) — "Direct Estimation of Equity Market Impact"**
- Dataset: ~1300 institutional equity trades, Citigroup equity trading desk
- Finding: alpha approximately 0.6 for the temporary impact exponent
- Methodology: regressed log(impact) on log(participation rate), controlling for volatility, spread, and ADV (average daily volume)
- Notable: permanent impact also found to be concave, with exponent ~0.5

**Bouchaud, Gefen, Potters & Wyart (2004) — "Fluctuations and Response in Financial Markets"**
- Asset class: equity, Paris Bourse (now Euronext)
- Finding: price impact scales as `~Q^0.5` at the individual trade level
- Interpretation: consistent with a square-root law; alpha ~0.5

**Gomes & Waelbroeck (2015) — "Is Market Impact a Measure of the Information Content of Orders?"**
- Dataset: large institutional orders, multiple asset classes
- Finding: alpha approximately 0.5-0.6 for temporary impact
- Key insight: distinguished between informed and uninformed order flow; uninformed flow shows more concave impact (alpha closer to 0.5)

**Torre & Ferrari (1999) — Barra Market Impact Model**
- Widely used practitioner model: impact ~ sqrt(ADV participation rate)
- Implied alpha = 0.5

**Crypto markets (Binance BTC/USDT, relevant to this project)**
- Estimated alpha from order book data: approximately 0.4-0.6
- Crypto markets have thinner books and higher volatility, making impact estimation noisier
- Studies by Cont & Cucuringu (2021) suggest alpha ~0.5 holds in major crypto venues

### Permanent Impact Exponent

The permanent (informational) impact also follows a power law empirically. Hasbrouck (1991) and subsequent work found that permanent impact scales approximately linearly with trade sign (not size), suggesting that for institutional execution the permanent component is approximately linear in shares (alpha_perm ~1), while temporary impact is concave.

### Summary Table

| Study | Asset Class | alpha (temporary) | Method |
|---|---|---|---|
| Almgren et al. (2005) | US equities | ~0.6 | Regression on institutional trades |
| Bouchaud et al. (2004) | French equities | ~0.5 | TAQ microstructure data |
| Torre & Ferrari (1999) | US equities | ~0.5 | Barra model calibration |
| Gomes & Waelbroeck (2015) | Multi-asset | 0.5-0.6 | Institutional order data |
| Crypto (est.) | BTC/ETH | 0.4-0.6 | Order book reconstruction |

**Consensus:** alpha is significantly less than 1 in real markets. The linear assumption (`alpha = 1`) that gives the closed-form solution is a mathematical convenience, not an empirical truth.

---

## 4. Gatheral's No-Dynamic-Arbitrage Constraint — What Impact Functions Are Allowed

### The Problem: Not All Impact Functions Are Arbitrage-Free

Gatheral (2010) — "No-Dynamic-Arbitrage and Market Impact" — identified a crucial constraint: not every plausible-looking market impact function is consistent with the absence of price manipulation. If the impact function is chosen carelessly, there exist round-trip trading strategies with zero net inventory change but positive expected profit.

### The Formal Constraint

Define the **price impact kernel** G(t, t') as the price move at time t caused by a trade at time t'. The no-dynamic-arbitrage (NDA) condition requires that the matrix with entries G(t_i, t_j) — when integrated over trading rates — must be **positive semi-definite** in an appropriate sense.

For a transient impact model where the price impact of a trade at time s decays according to a kernel K(t-s):
```
S(t) = S(0) - integral_0^t K(t-s) * f(v(s)) ds
```

The NDA condition is satisfied if and only if K is a **completely monotone** function (i.e., K(t) = integral_0^inf e^{-st} mu(ds) for some positive measure mu). This is equivalent to K being the Laplace transform of a non-negative measure.

### Implications for Common Kernels

**Power-law decay:** K(t) = t^{-gamma} is completely monotone for gamma in (0,1), hence NDA-compatible. This is the canonical transient impact model (Bouchaud et al.).

**Exponential decay:** K(t) = e^{-rho*t} is completely monotone (it is the Laplace transform of a point mass at rho). NDA-compatible. This is the Obizhaeva-Wang model.

**Linear (permanent) impact:** K(t) = constant is the degenerate case — completely monotone. The Almgren-Chriss permanent impact term falls here.

**Oscillatory kernels:** Any kernel with oscillations or negative values can violate NDA and admit price manipulation.

### Implications for the Impact Exponent f(v)

Gatheral also shows that the functional form `f(v) = v^alpha` must satisfy `alpha >= 1/2` to avoid round-trip arbitrage under some model specifications. This is important:

- `alpha = 0.5` (square root) sits on the boundary — it is NDA-compatible but just barely
- `alpha < 0.5` raises arbitrage concerns
- `alpha in [0.5, 1]` is the empirically consistent AND theoretically safe zone

**For this project:** The default `alpha = 1.0` is safely NDA-compatible. Extensions to `alpha = 0.5` (or alpha = 0.6) are also valid. Values below 0.5 should be used with caution and may require explicit verification.

### Practical Rule for Model Selection

When choosing an impact model:
1. Verify that the decay kernel is completely monotone
2. Verify that `alpha >= 0.5` for the power-law form
3. Ensure that permanent impact is non-negative (no price reversal that creates profit from round-trips)

---

## 5. Transient vs Permanent Impact — Bouchaud's Model

### The Three-Component Decomposition

In the modern microstructure literature (following Bouchaud, Farmer, Lillo 2009), price impact decomposes into three components:

```
Price_impact = Immediate_response + Transient_relaxation + Permanent_drift
```

**Permanent impact** (informational): The component that persists indefinitely. Caused by the information content of the trade — the market infers that a large buyer likely has positive information, so the price permanently adjusts upward. In Almgren-Chriss: `gamma * v * dt`.

**Transient (temporary) impact**: The component that decays after trading stops. Caused by liquidity demand — the market maker widens the spread temporarily to offset inventory risk, then mean-reverts when the trade is absorbed. In Almgren-Chriss: `h(v) = eta * |v|^alpha`.

**Resilience**: The speed at which the transient component decays. Fast resilience (liquid markets): the temporary impact disappears within seconds to minutes. Slow resilience: impact persists for hours.

### Bouchaud's Propagator Model

Bouchaud, Gefen, Potters & Wyart (2004) propose a propagator model:

```
S(t) = S(0) + sum_{t'<t} epsilon(t') * G(t - t') + noise
```

where:
- `epsilon(t')` = trade sign (+1 sell, -1 buy) at time t'
- `G(tau)` = impact propagator (decay kernel), G(0) = initial impact, G(infinity) = permanent impact
- The decay G(tau) ~ tau^{-beta} empirically, with beta ~0.5 for equities

Key findings:
- If trades were independent, persistent impact would cause arbitrage. But empirically, trade signs are negatively autocorrelated (Lillo & Farmer 2004): large institutional orders split into many small trades create positive autocorrelation, which cancels with the decaying impact to produce a near-martingale price.
- This "conspiracy" between order flow autocorrelation and impact decay is a market microstructure equilibrium, not a coincidence.

### The Obizhaeva-Wang Model (2013)

A tractable continuous-time model with transient impact:

```
dS = sigma dW - kappa (S - S_0) dt  +  theta * dX
```

where:
- `theta * dX` is the instantaneous execution cost (proportional to trade rate)
- `kappa * (S - S_0)` is the mean-reversion term (resilience)
- S_0 is the "fundamental price" to which the market reverts

In this model, **impact is fully transient** — it decays exponentially with rate kappa. The optimal execution problem has an explicit solution involving a combination of impulsive initial and terminal trades with a uniform rate in between (the "block + linear + block" structure).

### Comparison: Almgren-Chriss vs Bouchaud

| Feature | Almgren-Chriss (2001) | Bouchaud Propagator | Obizhaeva-Wang |
|---|---|---|---|
| Permanent impact | Yes, linear in v | Limiting value of G(inf) | None (fully transient) |
| Transient impact | Yes, instantaneous | Power-law decay | Exponential decay |
| Resilience | Instant (no memory) | Slow (tau^-0.5) | Exponential (kappa) |
| Closed form | Yes (alpha=1) | No (numerical) | Yes |
| NDA compatibility | Yes | Yes (if beta in (0,1)) | Yes |
| Optimal trajectory shape | Hyperbolic sine | Numerical | Block + linear + block |

### Implication for This Project

The Almgren-Chriss model implemented in this project assumes **instantaneous** temporary impact — the impact of a trade at time t does not persist to affect costs at time t+dt. This is a simplification. In reality, trades have memory. For a course project, the AC model is appropriate, but extensions toward:
1. Power-law decay kernel (Bouchaud): requires convolution integrals in cost computation
2. Exponential decay (Obizhaeva-Wang): adds a state variable (accumulated impact) to the HJB PDE

The transient impact extension would change `pde/hjb_solver.py` most significantly — the state space would need to expand from `(t, x)` to `(t, x, I)` where I tracks accumulated impact.

---

## 6. Key References

1. **Almgren, R. & Chriss, N. (2001).** "Optimal Execution of Portfolio Transactions." *Journal of Risk*, 3(2), 5-39.
   - Foundation paper. Derives closed-form solution for linear impact.

2. **Almgren, R. (2003).** "Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk." *Applied Mathematical Finance*, 10(1), 1-18.
   - Extends to power-law impact; shows alpha < 1 changes trajectory shape; provides numerical approach.

3. **Almgren, R., Thum, C., Hauptmann, E. & Li, H. (2005).** "Direct Estimation of Equity Market Impact." *Risk*, July 2005.
   - Large-sample empirical study; estimates alpha ~0.6; basis for industry impact models.

4. **Gatheral, J. (2010).** "No-Dynamic-Arbitrage and Market Impact." *Quantitative Finance*, 10(7), 749-759.
   - Derives the no-dynamic-arbitrage constraint; classifies which impact functions permit price manipulation.

5. **Bouchaud, J.-P., Gefen, Y., Potters, M. & Wyart, M. (2004).** "Fluctuations and Response in Financial Markets: The Subtle Nature of 'Random' Price Changes." *Quantitative Finance*, 4(2), 176-190.
   - Propagator model; establishes square-root law; shows impact decay scaling.

6. **Bouchaud, J.-P., Farmer, J.D. & Lillo, F. (2009).** "How Markets Slowly Digest Changes in Supply and Demand." In: *Handbook of Financial Markets: Dynamics and Evolution*, Chapter 2, North-Holland.
   - Comprehensive review of transient impact, order flow autocorrelation, and market microstructure equilibrium.

7. **Obizhaeva, A. & Wang, J. (2013).** "Optimal Trading Strategy and Supply/Demand Dynamics." *Journal of Financial Markets*, 16(1), 1-32.
   - Tractable transient impact model; "block + linear + block" optimal strategy.

8. **Lillo, F. & Farmer, J.D. (2004).** "The Long Memory of the Efficient Market." *Studies in Nonlinear Dynamics and Econometrics*, 8(3).
   - Documents long-range autocorrelation in trade signs; key input to Bouchaud propagator model.

9. **Hasbrouck, J. (1991).** "Measuring the Information Content of Stock Trades." *Journal of Finance*, 46(1), 179-207.
   - Decomposes impact into permanent (informational) and transient components; foundational microstructure paper.

10. **Torre, N. & Ferrari, M. (1999).** "Market Impact Model." *BARRA Research*, Berkeley, CA.
    - Practitioner square-root model; widely used in industry as the alpha=0.5 baseline.

---

*This note is part of the MF796 course project on Almgren-Chriss optimal execution. It informs extensions to the base model in `shared/cost_model.py` where `alpha` is already parameterized for nonlinear impact.*
