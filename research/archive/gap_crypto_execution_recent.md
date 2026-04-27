# Gap Research: Crypto-Specific Optimal Execution (2020–2026)

**Date:** 2026-03-27
**Scope:** Fills knowledge gaps in three areas — crypto AC applications, practical Binance constraints, recent RL/ML advances — that are NOT covered by our existing research on the standard AC framework, Kyle's lambda, square-root law, Heston, HMM, or classical variance reduction.

---

## 1. Crypto-Specific Applications of Almgren-Chriss

### 1.1 Direct AC Applications to Cryptocurrency

**Kurz (2020) — "Optimal Execution in Cryptocurrency Markets"**
Claremont McKenna College senior thesis. Direct application of the AC model to BTC/USD on Binance and Coinbase. Key findings:
- Binance (wholesale matching engine) vs. Coinbase (retail quote-driven) produce materially different execution costs under the same AC parameters.
- Bid-ask spread is a larger driver of total execution cost in crypto than in equities, particularly on Coinbase.
- The thesis confirms that the AC linear temporary + permanent impact decomposition is structurally applicable to crypto, but warns that parameter values calibrated on equities should not be assumed to transfer.
- Source: https://scholarship.claremont.edu/cmc_theses/2387/

**Ramirez & Sanchez (2023) — "Optimal liquidation with temporary and permanent price impact, an application to cryptocurrencies"**
arXiv: https://arxiv.org/abs/2303.10043
This is the most technically rigorous crypto-specific extension of the AC framework to date. Key points:
- Uses order book data from the BNB/USDT pair on Binance to **empirically estimate** the functional forms of both temporary and permanent impact.
- Finds that neither impact is purely linear: three estimation scenarios (underestimation, overestimation, and average) yield different functional forms — power-law shapes emerge in the average case.
- Derives **closed-form recursions** for the linear temporary + linear or quadratic permanent impact case.
- For nonlinear impact, solves numerically via **finite differences and optimal policy iteration** (methodology directly relevant to our HJB solver work).
- Key result: the optimal liquidation trajectory changes substantially depending on which functional form is assumed, underscoring the sensitivity of the AC-derived strategy to calibration assumptions.
- This paper is the direct crypto analog of the AC model and should be treated as a primary reference.

**Springer Digital Finance (2024) — "Optimal trade execution in cryptocurrency markets"**
DOI: https://link.springer.com/article/10.1007/s42521-023-00103-y
Published in *Digital Finance* (Springer). Studies crypto-specific fee schedule structure and its interaction with optimal execution:
- Examines the asymmetric **maker-taker fee schedule with instantaneous tier updates** — unique to crypto and absent from equity AC derivations.
- Formulates optimal execution with explicit fee minimization under LOB depth uncertainty.
- Proves existence of optimal strategies under this fee schedule.
- Optimal strategy: distribute volume across price levels with **exponentially decaying allocation** away from best price; this can reduce total execution cost by more than **60%** vs. naive market order strategies.
- The instantaneous (trade-by-trade) fee tier updates on Binance mean the optimal mix of maker vs. taker orders is dynamic and path-dependent — not captured by AC.

### 1.2 Is the Square-Root Law (α ≈ 0.5) Valid in Crypto?

**Tóth et al. (2011, arXiv: 1412.4503 covers BTC extension)**
"A Million Metaorder Analysis of Market Impact on the Bitcoin"
arXiv: https://arxiv.org/abs/1412.4503
Empirically confirms the **square-root law for market impact holds on BTC/USD**, using over one million metaorders. Key findings:
- Impact scales as √(Q/V) where Q is metaorder size and V is volume — consistent with equity markets across four decades of data.
- The square-root law holds **throughout the trajectory**, not just at final execution.
- Decomposes order flow into "informed" vs. "uninformed"; uninformed flow shows near-complete long-term impact decay.
- Crucially, this evidence was obtained "quasi-absent of statistical arbitrage and market making strategies," suggesting the square-root scaling is a **mechanical, not informational, phenomenon**.
- Implication for our model: α ≈ 0.5 is empirically defensible for BTC, but the prefactor (liquidity parameter η) must be calibrated to crypto data — it will differ from equity estimates by an order of magnitude given higher crypto volatility.

### 1.3 Crypto-Specific Modifications Required

Based on the literature, the following structural differences from equity-market AC must be accounted for:

**24/7 trading and intraday seasonality:**
- Despite continuous operation, trading activity follows equity-market hours: highest volume and lowest spreads occur during US market hours (9:30–16:00 ET).
- Crypto futures on Binance show strong intraday seasonality in 2022–2024 data: "Seasonal patterns appear to be a stylized fact in crypto futures."
- Transition zones (Asian overnight hours) show materially lower liquidity — the same AC trajectory executed during off-hours incurs higher market impact.
- Source: Easley et al., Cornell SSRN-4814346, and ScienceDirect intraday periodicity studies.

**No hard close/open — VWAP curve estimation is harder:**
- VWAP is well-defined in equities because volume clustering around open/close is predictable. In crypto, there is no open/close anchor.
- Genet (2025), arXiv:2502.13722, shows that standard volume-curve-based VWAP scheduling systematically underperforms in crypto due to unstable volume predictions across regimes. Direct optimization of slippage via deep learning outperforms curve-following approaches on BTC/ETH/BNB/ADA/XRP Binance perpetual data (Jan 2020–Jul 2024).

**Higher and time-varying volatility:**
- Annual BTC volatility is typically 60–100%, vs. 20–30% for large-cap equities. The AC risk-aversion parameter λ has a much larger effect in crypto: the efficient frontier shifts dramatically depending on vol regime.
- HMM regime detection (which we already implement) is particularly well-motivated for crypto: large volatility spikes coincide with 2022 market structure events (Terra collapse, FTX) and 2023 regulatory events (BlackRock ETF filing, MiCA).

**Fragmented liquidity across venues:**
- BTC trades simultaneously on Binance (spot + perps), Bybit, Kraken, Coinbase, OKX, Deribit. Price discovery is distributed, unlike equities where a primary exchange dominates.
- Cross-venue arbitrage is extremely fast (<1ms via co-located bots), so permanent price impact propagates across venues rapidly, but the exchange-specific order book depth used for temporary impact calibration is venue-specific.
- For our simulation focused on Binance BTCUSDT, the relevant liquidity pool is the Binance order book, but the permanent impact component reflects global BTC price consensus.

---

## 2. Practical Execution Constraints on Binance

### 2.1 Minimum Lot Size and Order Filters

Binance enforces order-level constraints via **LOT_SIZE filter** on every trading pair. For BTCUSDT spot (as of 2024–2025):
- `minQty`: minimum order quantity in BTC
- `maxQty`: maximum order quantity in BTC
- `stepSize`: quantity increment (quantum), orders must be exact multiples

For BTCUSDT spot, the typical `stepSize` is **0.00001 BTC** (5 decimal places), and `minQty` is also **0.00001 BTC**. This means at BTC ≈ $85,000, the minimum notional order size is approximately **$0.85**. This is not a binding constraint for institutional execution (our simulation uses orders in the range of 0.001–0.1 BTC per slice), but it matters for sub-cent slicing at the tail of a liquidation schedule.

Additionally, a `NOTIONAL` filter may require minimum order value (e.g., $5 or $10 USDT equivalent). API users must query `GET /api/v3/exchangeInfo?symbol=BTCUSDT` before execution to retrieve current filter values, as Binance updated these in June 2024.

Source: https://developers.binance.com/docs/binance-spot-api-docs/filters
Source: https://www.binance.com/en/support/announcement/updates-on-minimum-order-size-for-spot-and-margin-trading-pairs-2024-06-07-4b419936509647a4896e65a48eef2c5e

### 2.2 Fee Structure and Impact on Optimal Trajectory

Binance spot fee structure (2024–2025):
- **Default (VIP 0):** 0.10% maker, 0.10% taker
- **With BNB fee discount:** 0.075% maker, 0.075% taker
- **High-volume tiers (VIP 6–9):** as low as 0.02% taker, with maker rebates of 0.01% at top tiers
- Fee tiers are based on **30-day trailing volume** and are updated **instantaneously** after each trade on Binance — unlike traditional equities where tier resets occur monthly

The maker-taker asymmetry matters for AC implementation:
- Pure market-order (taker) execution incurs 0.075–0.10% per slice on top of spread and impact.
- Posting limit orders (maker) can either pay a lower fee or receive a rebate at high-volume tiers.
- The Springer 2024 paper (above) formalizes this: the optimal AC-style strategy should blend market and limit orders to exploit the fee structure, with exponentially decaying limit order allocation.
- For our simulation assuming market orders only, the per-slice fee of 0.075–0.10% must be added to the cost model. For a 10-slice liquidation, total fee cost ≈ 0.75–1.0% of notional, which is comparable to market impact for modest-sized orders.

Sources: https://www.binance.com/en/fee/schedule
https://cryptopotato.com/binance-fees/

### 2.3 Latency and Real-Execution Discrepancies

**Albers, Cucuringu, Howison, Shestopaloff (2025) — "The good, the bad, and latency: exploratory trading on Bybit and Binance"**
*Quantitative Finance*, Vol. 25, No. 6, pp. 919–947
DOI: https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2515933
Oxford University Research Archive: https://ora.ox.ac.uk/objects/uuid:cdab1de2-7576-42e2-abae-ab12371eba76

This paper is the gold standard empirical study of actual vs. expected execution on Binance. Key results (from millions of live market orders):
- **Global failure rate:** 1.354% for FOK/IOC limit orders (order submitted but cancelled without fill because LOB changed during transmission delay).
- **Slippage direction:** Consistent adverse selection effect — the market moves against the trader between order submission and fill, even at sub-second latency.
- Slippage and failure rates are strongly correlated with: (1) **realized volatility** at submission time, (2) **exchange-side latency** (intermittent LOB update delays), (3) **LOB depth** at the targeted price level.
- Marketable limit orders (IOC) outperform pure market orders in terms of fill quality when latency is low, but fail more when LOB is thin.
- **Implication for our model:** Our simulation assumes instantaneous fills at the modeled impact price. In practice, 1–2 ms round-trip latency at Binance co-location means that at BTC volatility of 80% annual (≈0.5% per minute), a 5ms delay introduces ~0.003% adverse slippage per order in expectation. Over 10 slices this is 0.03% additional cost — small but non-negligible in a tight execution study.

**Additional latency context:**
- Binance WebSocket feed has ~1–3ms latency from Singapore co-location, ~20–50ms from US/EU retail connections.
- REST API order placement adds ~5–20ms round-trip at co-location.
- High-frequency arbitrageurs maintain sub-millisecond connectivity, so by the time retail/institutional orders are submitted against a stale LOB snapshot, prices have partially adjusted.
- For simulation purposes, a latency model can add a slippage term drawn from an empirical distribution conditioned on vol regime (following Albers et al. 2025 methodology).

---

## 3. Recent Advances in Execution Optimization (2022–2026)

### 3.1 Reinforcement Learning Approaches

**Key finding:** RL-based methods have advanced rapidly and now consistently outperform TWAP and basic AC strategies in benchmarks, but they require realistic market simulation environments and face a simulation-to-reality gap.

**Practical Deep RL for Optimal Trade Execution (MDPI FinTech, 2023)**
https://www.mdpi.com/2674-1032/2/3/23
- Uses **PPO with LSTM** networks.
- Achieves generalization across 50 stocks, execution horizons 165–380 minutes, with dynamic target volume.
- Outperforms TWAP and IS benchmarks in backtests. Limitations: trained on equity data, simulation uses historical replay without realistic impact feedback.

**Optimal Execution with RL in Multi-Agent Market Simulator (arXiv:2411.06389, Nov 2024)**
arXiv: https://arxiv.org/abs/2411.06389
- Uses **Double DQN** in the **ABIDES** multi-agent market simulator (realistic LOB with other agents).
- Benchmarked against the AC efficient frontier: RL strategies cluster near the theoretical frontier.
- Key result: RL outperforms standard TWAP/VWAP and comes close to AC-optimal in simulation, but the multi-agent environment is more realistic than simple price models.
- Limitation: ABIDES is calibrated to equity data; crypto application would require re-calibration.

**Right Place, Right Time: RL for Execution Optimisation (arXiv:2510.22206, Oct 2025)**
arXiv: https://arxiv.org/html/2510.22206v1
- Evaluates RL strategies explicitly against the **AC efficient frontier** (risk vs. cost plane).
- Shows that well-trained RL agents operate near (but not on) the efficient frontier.
- Defines the performance gap as the distance from the RL policy to the frontier: RL agents achieve ~85–95% of AC efficiency in diverse market conditions.
- Important contribution: defines a rigorous evaluation framework that can be applied to any execution policy.

**RL in Queue-Reactive Models (arXiv:2511.15262, Nov 2025)**
arXiv: https://arxiv.org/html/2511.15262
- Proposes a **Queue-Reactive Model (QRM)** of LOB dynamics — ergodic and analytically tractable.
- RL policy trained within QRM calibrated from real market data.
- Addresses the LOB state representation problem: rather than using raw order book features, the QRM provides a reduced-state parameterization.
- More relevant to limit order placement than bulk liquidation, but the calibration methodology transfers.

**Deep RL for Crypto Limit Order Placement (EJOR, 2022)**
https://ideas.repec.org/a/eee/ejores/v296y2022i3p993-1006.html
- Specifically applies RL to **cryptocurrency limit order placement**.
- Outperforms benchmark strategies in terms of execution quality on crypto data.
- One of the few RL papers explicitly validated on crypto exchange data (not equity simulation).

### 3.2 Neural Network / ML Surrogates for AC

**Chen, Ludkovski, Voß (2023) — "On Parametric Optimal Execution and Machine Learning Surrogates"**
arXiv: https://arxiv.org/abs/2204.08581
Published in *Quantitative Finance* (2023): https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2282657
GitHub: https://github.com/moritz-voss/Parametric_Optimal_Execution_ML

This paper is directly relevant to our project's variance reduction and calibration work. Key contributions:
- Studies optimal execution with **nonlinear transient price impact** and stochastic resilience.
- For **linear transient impact**: closed-form recursion (analytically tractable).
- For **nonlinear impact**: develops **actor-critic neural network** surrogates for both value function and feedback control.
- Critical innovation: **parametric learning** — the NN takes model parameters (impact, resilience, risk-aversion) as inputs alongside state variables, enabling a single trained model to solve the execution problem across a wide parameter range.
- This directly addresses the calibration uncertainty problem: instead of re-solving the HJB/DP for each parameter draw, the NN surrogate evaluates near-instantly at any parameter point.
- Implementation: Jupyter Notebook available on GitHub, reproducible.

**Relevance to our project:** Our existing policy iteration solver is computationally expensive per parameter set. The Chen et al. surrogate approach would allow rapid sensitivity analysis and Monte Carlo over the parameter uncertainty space — a direct complement to our QMC variance reduction work.

### 3.3 FlowOE — Imitation Learning with Flow Matching

**FlowOE (arXiv:2506.05755, Jun 2025)**
arXiv: https://arxiv.org/abs/2506.05755
- First application of **flow matching models** (normalizing flows variant) to optimal execution.
- Learns from an ensemble of traditional expert strategies (AC, TWAP, IS variants).
- Adaptive: selects the most appropriate expert behavior for current market conditions.
- A **refining loss function** enables FlowOE to improve upon the learned expert actions, not merely imitate.
- Evaluated under Heston stochastic volatility and concave market impacts — directly comparable to our model setup.
- **Outperforms** all individually calibrated expert models and benchmarks across market conditions.
- Significance: this is the first method that can adaptively combine AC-optimal, TWAP, and other strategies depending on the regime, without re-solving the HJB from scratch.

### 3.4 Deep Learning for VWAP Execution in Crypto

**Genet (2025) — "Deep Learning for VWAP Execution in Crypto Markets: Beyond the Volume Curve"**
arXiv: https://arxiv.org/abs/2502.13722
- Dataset: hourly data for BTC, ETH, BNB, ADA, XRP on Binance perpetuals, Jan 2020 – Jul 2024.
- Key insight: standard VWAP relies on **volume curve prediction**, which is noisy and unstable in crypto. The paper bypasses this by **directly optimizing VWAP slippage** using automatic differentiation and custom loss functions.
- Results: direct optimization consistently achieves lower VWAP slippage than curve-following methods, even using a simple linear allocation baseline.
- Follow-up (arXiv:2502.18177): dynamic VWAP with **Temporal Kolmogorov-Arnold Networks (T-KAN)** for real-time adaptation — captures temporal dependencies better than standard LSTM.

**Relevance:** For our project, this paper validates that (1) BTC Binance perp data 2020–2024 is the right dataset, (2) direct optimization of execution cost (as in AC) is superior to schedule-following heuristics, and (3) the volume forecast problem is a genuine difficulty unique to crypto that our model sidesteps by using an impact-minimizing framework rather than volume tracking.

### 3.5 Regime-Switching Execution (Directly Complementing Our HMM Work)

**Li & Mulvey (2023) — "Optimal Portfolio Execution in a Regime-switching Market with Non-linear Impact Costs: Combining Dynamic Program and Neural Network"**
arXiv: https://arxiv.org/abs/2306.08809
Published in *INFORMS Journal on Optimization*: https://pubsonline.informs.org/doi/10.1287/ijoo.2021.0053

This paper bridges our HMM regime detection with the nonlinear impact AC solver. Key contributions:
- Four-step numerical framework: (1) orthogonal portfolio decomposition, (2) DP for regime-conditioned schedules, (3) NN pre-training from DP solution, (4) NN fine-tuning on full model.
- Handles **Markov regime switching** in the impact cost parameters — directly analogous to our HMM state-switching model.
- Addresses the **curse of dimensionality** for multi-asset execution with regime changes.
- For 10-asset liquidation with quadratic impact costs: combined DP+NN significantly outperforms pure DP in both CRRA and mean-variance frameworks.
- **Direct implication:** Our single-asset BTC model with HMM-detected regimes can use this framework to switch between regime-specific AC parameters, with the NN surrogate replacing costly DP re-solves at each regime transition.

### 3.6 Adaptive Dual-Level RL (2024)

**Adaptive Dual-Level RL for Optimal Trade Execution (Expert Systems with Applications, 2024)**
https://www.sciencedirect.com/science/article/abs/pii/S0957417424011291
- Proposes a two-level RL hierarchy: **macro level** determines the execution schedule; **micro level** optimizes individual order placement.
- Adapts dynamically to changing market conditions without offline recalibration.
- Particularly suited to crypto's non-stationary regime structure.

---

## 4. Summary Table: Research Gap Coverage

| Gap | Paper(s) | Key Takeaway for Our Project |
|-----|----------|------------------------------|
| AC applied to crypto | Kurz 2020, Ramirez & Sanchez 2023 | Framework valid; power-law impact forms preferred over linear for BNB/BTC |
| Square-root law in crypto | Tóth et al. 2014 (BTC) | α ≈ 0.5 confirmed empirically for BTC; prefactor must be recalibrated |
| Fee schedule in AC | Springer Digital Finance 2024 | Maker-taker + instant tier update = 60%+ cost reduction possible with limit orders |
| 24/7 microstructure | Intraday seasonality literature | Liquidity peaks at US hours; off-hours execution has higher impact — time AC trajectory |
| Binance lot size | Binance API docs 2024 | stepSize = 0.00001 BTC; not a binding constraint for our simulation scale |
| Binance fees | Binance fee schedule 2025 | 0.075% per slice (BNB discount); add ~0.75% total fee to 10-slice cost model |
| Latency effects | Albers et al. 2025 (Quantitative Finance) | 1.354% failure rate; adverse selection ~0.003% per 5ms delay; volatile regimes worse |
| RL vs AC benchmark | arXiv:2411.06389, 2510.22206 | RL reaches ~85–95% of AC efficiency; RL benchmark against efficient frontier is standard |
| NN surrogates for parameters | Chen, Ludkovski, Voß 2023 | Parametric NN surrogate eliminates per-parameter HJB re-solve — applicable to our calibration uncertainty |
| Flow matching for execution | FlowOE arXiv:2506.05755 | Adaptively combines expert strategies; outperforms single-model AC under Heston + concave impact |
| DL for crypto VWAP | Genet 2025 arXiv:2502.13722 | Direct cost optimization beats volume-curve tracking on BTC/Binance data |
| Regime-switching + NN execution | Li & Mulvey 2023 arXiv:2306.08809 | DP+NN handles regime-switching nonlinear impact — blueprint for our HMM-AC integration |

---

## 5. Priority Actionable Findings for Our Model

**Immediate parameter corrections:**
1. When calibrating temporary impact parameter η for BTC, use Binance LOB depth data — do not scale from equity estimates. Ramirez & Sanchez (2023) provide BNB calibration methodology transferable to BTC.
2. Add fee cost explicitly: 0.075% × N_slices × notional as a fixed additive term in the cost functional.
3. Square-root law (α = 0.5) is empirically validated for BTC — our assumption is defensible.

**Model extensions worth citing:**
- The Springer 2024 fee-schedule paper provides formal justification for why our market-order-only model is a lower bound — the true optimal would blend maker/taker orders.
- Albers et al. (2025) provides the empirical grounding for why simulation-based execution ignores a real 0.03–0.1% adverse slippage cost — worth noting as a practical model limitation.

**For the ML/RL section of the project:**
- The RL-vs-AC efficient-frontier comparison framework (arXiv:2510.22206) is the standard benchmark methodology to cite when comparing our AC solution to RL baselines.
- Chen, Ludkovski, Voß (2023) is the most directly relevant "ML surrogate for AC" paper — their actor-critic parametric approach is the state of the art for handling impact parameter uncertainty, which is our main calibration challenge.

---

## References

1. Kurz, E. (2020). *Optimal Execution in Cryptocurrency Markets*. CMC Senior Thesis. https://scholarship.claremont.edu/cmc_theses/2387/

2. Ramirez, H.E. & Sanchez, J.F. (2023). *Optimal liquidation with temporary and permanent price impact, an application to cryptocurrencies*. arXiv:2303.10043. https://arxiv.org/abs/2303.10043

3. Tóth, B. et al. (2014). *A Million Metaorder Analysis of Market Impact on the Bitcoin*. arXiv:1412.4503. https://arxiv.org/abs/1412.4503

4. [Authors] (2024). *Optimal trade execution in cryptocurrency markets*. Digital Finance (Springer). https://link.springer.com/article/10.1007/s42521-023-00103-y

5. Albers, J., Cucuringu, M., Howison, S., & Shestopaloff, A.Y. (2025). *The good, the bad, and latency: exploratory trading on Bybit and Binance*. Quantitative Finance, 25(6), 919–947. https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2515933

6. Chen, T., Ludkovski, M., & Voß, M. (2023). *On parametric optimal execution and machine learning surrogates*. Quantitative Finance, 24(1). arXiv:2204.08581. https://arxiv.org/abs/2204.08581

7. [Authors] (2024). *Optimal Execution with Reinforcement Learning in a Multi-Agent Market Simulator*. arXiv:2411.06389. https://arxiv.org/abs/2411.06389

8. [Authors] (2025). *Right Place, Right Time: Market Simulation-based RL for Execution Optimisation*. arXiv:2510.22206. https://arxiv.org/html/2510.22206v1

9. [Authors] (2025). *Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution*. arXiv:2511.15262. https://arxiv.org/html/2511.15262

10. Li, X. & Mulvey, J.M. (2023). *Optimal Portfolio Execution in a Regime-switching Market with Non-linear Impact Costs: Combining Dynamic Program and Neural Network*. arXiv:2306.08809. https://arxiv.org/abs/2306.08809

11. [Authors] (2025). *FlowOE: Imitation Learning with Flow Matching for Optimal Execution under Heston Volatility and Concave Market Impacts*. arXiv:2506.05755. https://arxiv.org/abs/2506.05755

12. Genet, R. (2025). *Deep Learning for VWAP Execution in Crypto Markets: Beyond the Volume Curve*. arXiv:2502.13722. https://arxiv.org/abs/2502.13722

13. [Authors] (2024). *An Adaptive Dual-level Reinforcement Learning Approach for Optimal Trade Execution*. Expert Systems with Applications. https://www.sciencedirect.com/science/article/abs/pii/S0957417424011291

14. Binance API Documentation (2024). *Filters — LOT_SIZE, NOTIONAL*. https://developers.binance.com/docs/binance-spot-api-docs/filters

15. Binance (2024). *Updates on Minimum Order Size for Spot and Margin Trading Pairs 2024-06-07*. https://www.binance.com/en/support/announcement/updates-on-minimum-order-size-for-spot-and-margin-trading-pairs-2024-06-07-4b419936509647a4896e65a48eef2c5e

16. [Authors] (2024). *Practical Application of Deep Reinforcement Learning to Optimal Trade Execution*. FinTech (MDPI). https://www.mdpi.com/2674-1032/2/3/23
