# Research Q&A: Regime Detection (Part E) & Architecture

**Project:** MF796 Optimal Execution
**Date:** 2026-03-26
**Scope:** Questions 13–18 covering HMM data requirements, spread proxies, regime-switch re-planning, statistical testing, permanent impact omission, and self-impact convention.

---

## Q13. HMM Data Requirements: Is 5 Days of Binance aggTrades Sufficient?

### Short Answer

5 calendar days of aggTrades is **borderline insufficient** for a production-quality 2-state GaussianHMM, but it can work if you aggregate to 1-minute OHLCV bars (~7,200 observations over 5 days on a 24/7 crypto exchange) and treat the result as a **provisional fit** subject to validation.

### Details

**Parameter count for a 2-state, 1-feature GaussianHMM:**

| Parameter block | Count |
|---|---|
| Transition matrix | 2 free parameters (2×2, rows sum to 1) |
| Initial state probabilities | 1 free parameter |
| Emission means (μ₁, μ₂) | 2 |
| Emission variances (σ₁², σ₂²) | 2 |
| **Total** | **7** |

By the rule-of-thumb "10 observations per free parameter" you need ~70 observations minimum. At the tick level 5 days of BTC aggTrades on Binance is **millions** of rows—far more than 70. However, parameter stability is not driven by raw observation count alone; it requires the chain to **visit and re-visit both states multiple times** during the training window.

**What the literature says:**

- The hmmlearn documentation warns that EM (Baum-Welch) converges to local optima; multiple random restarts (`n_init`) are recommended regardless of sample size. Monitor convergence via `model.monitor_.converged`.
- A paper from MIT (Anandkumar et al., JMLR 2018) on Baum-Welch convergence guarantees shows that convergence is linear in SNR, and the effective "mixing time" of the Markov chain must be much shorter than the sequence length. For a 2-state regime model with moderate persistence (e.g., expected regime duration ~30 min), 5 days at 1-minute resolution (~7,200 obs) gives ~240 expected regime transitions—adequate.
- Practitioners fitting HMMs to crypto daily returns typically use **6–24 months** of data (QuantStart, MDPI 2020). The key difference for your use case: you are using **intraday** features (e.g., realized volatility per 5-min bar, trade imbalance), not daily returns. The higher frequency dramatically increases the effective sample. See: "Regime-Switching Forecasts of Crypto Prices: Empirical Assessment of a Two-State Gaussian HMM on BTC, NEO, and RNDR" (Academia.edu, 2024).

**Practical recommendation:**

1. Aggregate aggTrades to **5-minute bars** (returns, realized vol, trade imbalance).
2. You will have ~1,440 bars over 5 days—enough for Baum-Welch to converge.
3. Run `n_iter=200`, multiple random inits, select best by log-likelihood.
4. Validate stability: re-fit on rolling sub-windows (days 1–3 vs days 2–4 vs days 3–5). Check that the regime labels are consistent across windows.
5. If the two regimes are not clearly separated in (mean, variance) space, the fit is unreliable regardless of sample size.

**Red flag:** If one regime is visited only in a single cluster (e.g., all high-volatility obs in the last hour of a single day), the emission parameters for that state will be poorly identified. In that case extend your training window.

### Sources

- [hmmlearn Tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html)
- [Market Regime Detection using HMMs in QSTrader — QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [Regime-Switching Forecasts of Crypto Prices (Two-State Gaussian HMM)](https://www.academia.edu/145026890/Regime_Switching_Forecasts_of_Crypto_Prices_Empirical_Assessment_of_a_Two_State_Gaussian_HMM_on_BTC_NEO_and_RNDR)
- [Statistical and Computational Guarantees for Baum-Welch (JMLR)](https://jmlr.org/papers/volume18/16-093/16-093.pdf)
- [Regime-Switching Factor Investing with HMMs (MDPI 2020)](https://www.mdpi.com/1911-8074/13/12/311)

---

## Q14. Spread Proxy from aggTrade Data

### Short Answer

aggTrades contain `price`, `quantity`, and `isBuyerMaker`. Three viable proxies in descending order of reliability for your setup:

1. **Roll (1984) estimator** — use first-order autocovariance of trade-price changes.
2. **Abdi-Ranaldo (2017) CHL estimator** — requires close/high/low per bar; directly constructible from aggTrades.
3. **Corwin-Schultz (2012) HL estimator** — requires high/low per interval; slightly lower accuracy than AR for intraday.

### (a) Roll Estimator

**Formula:**

$$\hat{s}_{Roll} = 2\sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}$$

where $\Delta p_t = p_t - p_{t-1}$ is the change in transaction price.

**Logic:** In an efficient market, transaction prices bounce between bid and ask. The bid-ask bounce induces negative serial correlation in price changes. Roll (1984) showed this first-order autocovariance equals $-(s/2)^2$ under the assumption of i.i.d. trade direction.

**Implementation from aggTrades:**
```python
import numpy as np
prices = agg_df['price'].values
dp = np.diff(prices)
cov = np.cov(dp[:-1], dp[1:])[0, 1]
roll_spread = 2 * np.sqrt(max(-cov, 0))  # set to 0 if cov > 0
```

**Known limitation:** When autocovariance is positive (trend-dominated periods), the estimator yields imaginary results. The fix (Harris 1990, Hasbrouck 2009) is to set negative-autocovariance instances to zero. At high frequency on a 24/7 crypto market, positive autocovariance from short-term momentum is common. Apply over rolling 5–15 minute windows.

**Performance:** Cross-sectional correlation with TAQ effective spread ≈ 0.56 (Goyenko, Holden & Trzcinka 2009). Lower than AR/CS but requires only trade prices.

### (b) Trade-Sign Autocorrelation

This is not a direct spread estimator but a **market condition indicator**. Under the Roll model, $q_t \in \{+1,-1\}$ (buyer-initiated vs seller-initiated). Since aggTrades include `isBuyerMaker`, you can reconstruct $q_t$ directly. Strongly negative sign autocorrelation indicates the spread is being crossed rapidly (thin book); near-zero autocorrelation indicates orderly flow. Use as a **regime feature input** for the HMM rather than as a spread estimate.

### (c) Corwin-Schultz (2012) High-Low Estimator

**Logic:** Daily high prices are almost always buyer-initiated trades (buy at ask); daily low prices are seller-initiated (sell at bid). The ratio of the high-low range over 1-day vs 2-day intervals allows decomposing variance from spread.

**Applicability to intraday aggTrades:** Corwin & Schultz explicitly extend the estimator to intraday data at intervals as short as 15 minutes ("An Application of the High-Low Spread Estimator to Intraday Data", Corwin & Schultz 2019 working paper). From aggTrades, construct OHLC bars at e.g. 5-minute intervals, then apply the CS formula over pairs of consecutive bars.

**Formula sketch:**
$$\beta = [\ln(H_t/L_t)]^2 + [\ln(H_{t-1}/L_{t-1})]^2$$
$$\gamma = [\ln(\max(H_t,H_{t-1})/\min(L_t,L_{t-1}))]^2$$
$$\alpha = \frac{\sqrt{2\beta}-\sqrt{\beta}}{3-2\sqrt{2}} - \sqrt{\frac{\gamma}{3-2\sqrt{2}}}$$
$$\hat{s}_{CS} = \frac{2(e^\alpha - 1)}{1+e^\alpha}$$

**Performance vs Roll:** CS has higher correlation with TAQ effective spread (~0.61 vs 0.56 for Roll). However it requires OHLC bars, not raw tick prices.

### (d) Abdi-Ranaldo (2017) CHL Estimator — Recommended

**Why preferred:** Abdi & Ranaldo (RFS 2017) showed their close-high-low estimator achieves average cross-sectional correlation of **0.74** with Daily TAQ effective spreads, outperforming Roll (0.56), CS (0.61), Gibbs/Hasbrouck (0.64–0.67). The `bidask` R package and Python ports implement this directly. It uses close, high, and low prices per interval—all constructible from aggTrades.

**Key reference:** Abdi & Ranaldo (2017) also demonstrate applicability to Binance crypto pairs (available crypto estimates in the bidask package).

### Recommendation for this project

For HMM feature construction, compute **Abdi-Ranaldo spread** at 5-minute resolution from aggTrade-derived OHLC bars. Use it as one of the emission dimensions in the 2-state GaussianHMM (alongside realized volatility and trade imbalance). Roll estimator is simpler and useful for a quick sanity check.

### Sources

- [Roll (1984): A Simple Implicit Measure of the Effective Bid-Ask Spread](https://www.bauer.uh.edu/rsusmel/phd/roll1984.pdf)
- [Roll Spread Estimator Lecture Notes — Ødegaard (2026)](https://www.ba-odegaard.no/teach/notes/liquidity_estimators/roll_spread_estimator/roll_lectures.pdf)
- [Corwin & Schultz (2012): A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2012.01729.x)
- [Abdi & Ranaldo (2017): Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices (RFS)](https://academic.oup.com/rfs/article/30/12/4437/4047344)
- [Hasbrouck (2009): Trading Costs and Returns for U.S. Equities](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2009.01469.x)
- [bidask R package: Efficient Estimation of Bid-Ask Spreads (OHLC)](https://cran.r-project.org/web/packages/bidask/vignettes/bidask.html)
- [Semiparametric Estimation of Bid-Ask Spread in Extended Roll Models (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0304407618301751)

---

## Q15. Re-Planning on Regime Switch: Re-Plan vs. Commit?

### Short Answer

**Re-plan from current inventory x(t) using new regime parameters.** The academic literature on regime-switching execution is unanimous: the optimal strategy is state-dependent and must be updated when the state changes. Committing to the initial plan in a new regime is suboptimal and can be significantly costlier.

### Theoretical basis

**Closed-form regime-switching execution (Nguyen & Tran, SIAM CT 2021):**

The paper "Execution Shortfall Algorithms under Regime Switching" (SIAM Conference on Control and Its Applications, 2021; DOI: 10.1137/1.9781611976847.7) formulates optimal execution as a discrete-time stochastic control problem with regime switching via a finite-state Markov chain. The key result: the **value function and optimal trading rates are regime-dependent**. The closed-form solution at time $t$ with remaining inventory $x(t)$ and current regime $r(t)$ is:

$$n_k^*(t, x(t), r(t)) = f(x(t), r(t), \text{remaining time})$$

There is no single trajectory that is jointly optimal across all regimes. Committing to the initial plan ignores the new regime's impact parameters, leading to suboptimal cost-risk tradeoff.

**Regime-switching market resilience (Cartea-Jaimungal variant, ScienceDirect 2019):**

"Optimal execution with regime-switching market resilience" (Journal of Economic Dynamics and Control, 2019) models the LOB resilience rate as a Markov chain. Result: "a trader would optimally place more aggressive market orders when the LOB switches from a low to a high resilience state." This is only possible if the agent re-plans at the switch point.

**Neural network + DP approach (arXiv 2306.08809):**

The 2023 paper proposes a four-step numerical framework combining dynamic programming and neural networks for multi-asset regime-switching execution. The framework explicitly solves the DP backward from terminal time, giving a **state-feedback policy** $\pi(t, x, r)$ rather than a fixed schedule. This is architecturally equivalent to re-planning at every step.

### Practical recommendation

**Implement a state-feedback controller:**

```
At each decision time k:
  1. Detect current regime r_k via Viterbi or posterior probability
  2. Look up pre-computed optimal rate n*(x_k, r_k, T-k) from the regime-k DP solution
  3. Execute n_k shares
```

If the DP is too expensive to pre-compute for all states, use the following approximation: when a regime switch is detected, **restart the Almgren-Chriss solver with remaining inventory x(t)**, remaining time T-t, and new regime parameters (σ_new, η_new). This gives a new TWAP-like schedule for the remainder of the horizon. The cost of this approach is an additional transaction if the re-plan changes urgency significantly.

**Regime hysteresis guard:** Avoid thrashing by requiring the HMM posterior to exceed a threshold (e.g., 0.75) before triggering re-plan. A brief "wait-and-see" window of 2–5 minutes prevents spurious re-planning.

### Sources

- [Execution Shortfall Algorithms under Regime Switching (SIAM 2021)](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976847.7)
- [ResearchGate abstract: Execution Shortfall Algorithms under Regime Switching](https://www.researchgate.net/publication/353100537_Execution_Shortfall_Algorithms_under_Regime_Switching)
- [Optimal execution with regime-switching market resilience (ScienceDirect 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0165188919300247)
- [Optimal Portfolio Execution in Regime-switching Market with Nonlinear Impact Costs (arXiv 2306.08809)](https://arxiv.org/abs/2306.08809)
- [Regime Shift Detection: Advanced MFT Alpha Generation and Execution Optimization (oboe.com)](https://oboe.com/learn/advanced-mft-alpha-generation-and-execution-optimization-1h6lzh4/regime-shift-detection-7)

---

## Q16. Statistical Significance of Regime-Conditional Execution

### Short Answer

Use a **paired Monte Carlo simulation test**: simulate many price paths from both the unconditional and regime-conditional model, compute IS cost for each strategy on each shared path, and test the paired cost differences with a t-test or Wilcoxon signed-rank test. Supplement with walk-forward out-of-sample backtesting.

### Detailed approach

#### Step 1: Paired MC simulation test

**Setup:**
1. Fit the regime-switching price model (GaussianHMM + impact parameters per regime) on training data.
2. Simulate $N = 5{,}000$–$10{,}000$ price paths using the fitted model.
3. On **each path $i$**, run both strategies:
   - **Strategy A:** Unconditional Almgren-Chriss (single set of parameters, no regime detection)
   - **Strategy B:** Regime-conditional (re-plans on regime switch using regime-specific parameters)
4. Compute implementation shortfall $IS_i^A$ and $IS_i^B$ for each path.

**Test:** The pair $(IS_i^A - IS_i^B)$ is the cost improvement from using regime detection on path $i$.

```
H0: E[IS^A - IS^B] = 0
H1: E[IS^A - IS^B] > 0  (regime-conditional is cheaper)
```

Use a **one-sided paired t-test** (if IS differences are approximately normal) or **Wilcoxon signed-rank test** (non-parametric, preferred for heavy-tailed IS distributions). With $N=5{,}000$ simulations and a true improvement of 2 bps, you will have power > 0.99 at $\alpha=0.01$.

**Reported by Nguyen & Tran (SIAM 2021):** Regime-switching strategies outperform VWAP benchmark by **0.10–8 basis points** on 5 NASDAQ stocks, validated via simulation with calibrated parameters. This provides a benchmark for what "realistic improvement" looks like.

#### Step 2: Walk-forward out-of-sample backtest

1. Divide the 5-day aggTrade dataset (or longer if available) into in-sample (fit HMM + AC params) and out-of-sample (evaluate).
2. Implement a **rolling window walk-forward**: fit on days 1–3, evaluate on day 4; fit on days 2–4, evaluate on day 5; etc.
3. Compute average IS for regime-conditional vs unconditional on the out-of-sample periods.
4. Apply a bootstrap test on the out-of-sample IS differences to obtain a p-value.

**Caveat:** With only 5 days of data, out-of-sample windows are very short. The MC simulation test is therefore the primary significance test for this project; walk-forward provides a sanity check.

#### Step 3: Avoid common pitfalls

- **Do not** compare in-sample IS—the regime model has more parameters and will appear better in-sample by construction.
- **Do control for** execution timing: both strategies must receive the same price path and the same market impact (same $\eta$, $\sigma$), differing only in the trading schedule used.
- **Report** the full distribution of improvements, not just the mean. A strategy with higher mean improvement but also higher variance may not be desirable.

### Sources

- [Execution Shortfall Algorithms under Regime Switching (SIAM 2021)](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976847.7)
- [Improving Robustness of Trading Strategy — Roncalli (Thierry-Roncalli.com)](http://www.thierry-roncalli.com/download/rbm_gan_backtesting.pdf)
- [Backtest Overfitting in the Machine Learning Era (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- [Monte Carlo Simulation Stress Test for Trading Strategies (backtestbase.com)](https://www.backtestbase.com/education/monte-carlo-stress-testing)
- [Pairs Trading with Markov Regime-Switching Model — Hudson & Thames](https://hudsonthames.org/pairs-trading-with-markov-regime-switching-model/)

---

## Q17. Permanent Impact Omission from HJB: When Does γ Matter?

### Short Answer

For linear permanent impact, omitting $\gamma$ from the HJB is **exactly correct** because the permanent impact term $\frac{1}{2}\gamma X_0^2$ is trajectory-independent and does not affect the optimal trading schedule. However, if you compare strategies by **total cost** (not just optimal schedule), $\gamma$ must be included. The approximation breaks down in terms of **strategy choice** only when permanent impact is **nonlinear**. The relevant ratio for when nonlinearity matters is $X_0 / V_{daily} \gtrsim 5\text{–}10\%$.

### Proof that linear γ is trajectory-independent

In Almgren & Chriss (2000), the **expected cost** of a liquidation schedule $\{n_k\}$ is:

$$E[IS] = \underbrace{\frac{1}{2}\gamma X_0^2}_{\text{permanent: trajectory-independent}} + \underbrace{\epsilon \sum_{k}|n_k|}_{\text{fixed cost}} + \underbrace{\eta \sum_{k} n_k^2 / \tau}_{\text{temporary: trajectory-dependent}}$$

The permanent impact term $\frac{1}{2}\gamma X_0^2$ depends only on the total quantity liquidated ($X_0$), not on how or when shares are traded. Therefore:
- **It does not enter the first-order conditions** for the optimal schedule.
- **It cancels out** when comparing two strategies that liquidate the same total quantity.
- **The HJB equation** for the optimal schedule correctly omits $\gamma$.

This is a standard result, confirmed by Almgren (2008, Encyclopedia of Quantitative Finance) and all major implementations.

### When does permanent impact affect strategy choice?

**1. Nonlinear permanent impact (power law)**

If $g(v) = \gamma |v|^\alpha$ with $\alpha \neq 1$ (empirically $\alpha \approx 0.6$, Almgren et al. 2005), then permanent impact IS trajectory-dependent. The optimal schedule is no longer the Almgren-Chriss hyperbolic-sine trajectory. In this case you cannot omit $\gamma$ from the HJB.

Almgren et al. (2005) fit their model using a large Citigroup dataset and find:
$$\text{Permanent impact} \approx \gamma \sigma \left(\frac{X_0}{V_{daily}}\right)^{0.6}$$

**2. Partial liquidation / repeated trading**

If the agent repeatedly impacts the same market (e.g., multiple execution slices per day), the accumulated permanent impact from prior trades shifts the reference price, affecting the marginal cost of subsequent trades. Here $\gamma$ matters even for linear impact.

**3. The X₀/V_daily threshold**

Empirical rule of thumb from Almgren et al. (2005) and Lillo (Imperial College lecture notes):

| $X_0 / V_{daily}$ | Regime |
|---|---|
| $< 1\%$ | Temporary impact dominates; permanent impact negligible even if nonlinear |
| $1\text{–}5\%$ | Permanent impact contributes but is roughly trajectory-independent under linear model |
| $5\text{–}10\%$ | Nonlinear permanent impact creates meaningful trajectory dependence; linear approximation breaks down |
| $> 10\%$ | Permanent impact dominates; full nonlinear model required; price can move several volatility units |

**Practical rule for this project:** If your order size $X_0$ is less than **5% of Binance BTC daily volume** (e.g., $X_0 < 100$ BTC for typical BTC daily volume of ~2,000 BTC on Binance spot), the linear $\gamma$ omission is valid for strategy optimization. Include $\frac{1}{2}\gamma X_0^2$ only in the **total cost comparison** between strategies.

### Sources

- [Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [Almgren (2008): Encyclopedia of Quantitative Finance — Market Impact](https://www.smallake.kr/wp-content/uploads/2016/03/eqf.pdf)
- [Market Impact Models and Optimal Execution Algorithms — Lillo (Imperial College)](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/cfm-imperial-institute-of-quantitative-finance/events/Lillo-Imperial-Lecture3.pdf)
- [Market Impact of Small Orders (arXiv 2201.02983)](https://arxiv.org/pdf/2201.02983)
- [Direct Estimation of Equity Market Impact — Almgren et al. (2005)](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf)
- [Optimal Execution: Almgren-Chriss and Beyond — Medium (Shivam Sharma)](https://medium.com/@shivam.sharma15/optimal-execution-almgren-chriss-and-beyond-9c192388262c)
- [Understanding the Almgren-Chriss Model — SimTrade blog](https://www.simtrade.fr/blog_simtrade/understanding-almgren-chriss-model-for-optimal-trade-execution/)

---

## Q18. Self-Impact Convention: Which Is Standard?

### Short Answer

The **standard Almgren-Chriss convention** (as in the original 2000 paper and most academic implementations) is that **permanent impact of trade $k$ affects the price for all subsequent trades $k+1, k+2, \ldots, N$**, i.e., $n_k$ is NOT included in the impact felt during its own execution. The cost_model convention you describe (where `n_k` multiplies a cumsum including itself) corresponds to a **concurrent self-impact** convention that overstates costs by approximately $\frac{1}{2}\gamma \sum n_k^2$ relative to the standard convention. The MC convention (no self-impact) understates by approximately the same amount in the opposite direction. The standard is **the original paper's convention: each trade impacts all future prices, not its own.**

### Original paper convention (Almgren & Chriss 2000)

The price dynamics in Almgren & Chriss are:

$$\tilde{S}_k = S_{k-1} + \sigma \tau^{1/2} \xi_k - \tau \cdot g(n_k/\tau)$$

where $g(\cdot)$ is the permanent impact function. The transaction in period $k$ causes a price shift of $-g(n_k/\tau) \cdot \tau$ that **takes effect before period $k+1$**. The execution price for trade $k$ is:

$$\tilde{p}_k = S_{k-1} - h(n_k/\tau) + \frac{1}{2}\sigma \tau^{1/2} \xi_k$$

where $h(\cdot)$ is the temporary impact. The key point: **$\tilde{p}_k$ does not include the permanent impact of $n_k$ itself**. Permanent impact from $n_k$ is reflected in $S_k = S_{k-1} - g(n_k/\tau)\tau + \sigma\tau^{1/2}\xi_k$, which affects the benchmark price for period $k+1$ onward.

**Expected cost formula:**

$$E[IS] = \frac{1}{2}\gamma X_0^2 + \epsilon \sum_k |n_k| + \frac{\eta}{\tau}\sum_k n_k^2 - \frac{1}{2}\gamma\tau\sum_k n_k^2$$

The term $-\frac{1}{2}\gamma\tau\sum_k n_k^2$ arises from the "self-impact correction": each trade's permanent impact is felt by all later trades but NOT by itself. This is the standard formula in all canonical references (Almgren & Chriss 2000 eq. 13; Almgren 2008 EQF).

### Your cost_model vs MC discrepancy

| Convention | Treatment of n_k's own permanent impact | Cost vs. standard |
|---|---|---|
| **Standard A&C (2000)** | n_k affects prices from k+1 onward only | Baseline |
| **cost_model (cumsum includes self)** | n_k affects its own execution price | Overstates by ~$\frac{1}{2}\gamma\sum n_k^2 \cdot \tau$ |
| **MC (no self-impact)** | n_k affects nothing during its own slice | Understates by ~$\frac{1}{2}\gamma\sum n_k^2 \cdot \tau$ |

For linear permanent impact with small $\gamma$ (thin books, small order), this discrepancy is negligible. For large $\gamma$ or large $n_k$, the difference can be material.

### Recommendation

Align both `cost_model` and `MC` to the **Almgren-Chriss 2000 standard**:
- Trade $k$'s price = mid-price at entry to period $k$ (reflecting all prior permanent impacts) minus temporary impact of $n_k$.
- Permanent impact of $n_k$ updates the reference price for period $k+1$.
- No "self-impact" on the current trade's own execution price.

In code terms, the permanent impact cumsum should be **lagged by one step**:

```python
# Standard A&C convention:
# cumulative_permanent_impact[k] = gamma * sum(n_0, n_1, ..., n_{k-1})
# NOT including n_k itself
cum_perm_impact = np.cumsum(np.insert(n[:-1], 0, 0)) * gamma
exec_prices = S0 - cum_perm_impact - eta * n / tau
```

If both `cost_model` and `MC` use this convention, the discrepancy disappears and the two implementations will agree on expected cost (up to simulation noise).

### Sources

- [Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions (full paper)](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [Deep Dive into IS: The Almgren-Chriss Framework — Anboto Labs (Medium)](https://medium.com/@anboto_labs/deep-dive-into-is-the-almgren-chriss-framework-be45a1bde831)
- [A Tale of Two Models: Implementing the Almgren-Chriss Framework (Bagourd, 2022)](https://www.arthur.bagourd.com/wp-content/uploads/2022/08/A_Tale_of_Two_Models__Implementing_the_Almgren_Chriss_framework_through_nonlinear_and_dynamic_programming.pdf)
- [Solving the Almgren-Chriss Model — Dean Markwick (2024)](https://dm13450.github.io/2024/06/06/Solving-the-Almgren-Chris-Model.html)
- [Target Close and Implementation Shortfall (arXiv 1205.3482)](https://arxiv.org/pdf/1205.3482)
- [acOptTxns: Almgren-Chriss in R blotter package](https://rdrr.io/github/braverock/blotter/man/acOptTxns.html)

---

## Summary Table

| Q | Issue | Recommendation |
|---|---|---|
| 13 | 5 days aggTrades for HMM | Sufficient if aggregated to 5-min bars (~1,440 obs); validate stability across rolling sub-windows |
| 14 | Spread proxy from aggTrades | Use Abdi-Ranaldo (2017) CHL on 5-min OHLC bars; Roll estimator for quick check |
| 15 | Regime switch re-plan vs commit | Re-plan from x(t) with new regime params; use hysteresis threshold to prevent thrashing |
| 16 | Significance test | Paired MC test (5,000+ paths), one-sided t-test or Wilcoxon; supplement with walk-forward OOS |
| 17 | Permanent impact omission (linear γ) | Valid for strategy optimization if X₀/V_daily < 5%; include γX₀²/2 in total cost comparison only |
| 18 | Self-impact convention | Standard is lagged cumsum (n_k affects k+1 onward, not itself); align cost_model and MC to this |
