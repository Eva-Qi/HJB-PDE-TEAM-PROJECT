# Almgren-Chriss Calibration: Research Q&A (Part A)

**Project**: MF796 Course Project — Optimal Execution on Binance BTCUSDT
**Date**: 2026-03-26
**Scope**: Three calibration questions for permanent impact γ, temporary impact η, and realized volatility σ.

---

## Q1: Kyle's Lambda vs OFI Regression for Permanent Impact γ

### What Kyle's Lambda Actually Estimates

Kyle (1985) defines the price-impact coefficient λ as the slope from regressing signed order flow on midprice changes. The standard empirical implementation (following Hasbrouck 2009 and Goyenko, Holden & Trzcinka 2009) runs:

```
ΔP_n = λ · S_n + ε_n
```

where `S_n` is signed square-root dollar volume (or signed raw volume). The slope λ captures the permanent, information-driven component of price impact: each unit of net order flow permanently shifts the price by λ. In the Almgren-Chriss (2001) framework, the permanent impact parameter γ plays exactly this role — a per-share price shift that does not revert. So **Kyle's lambda is conceptually the right object** for γ, but the regression must be estimated on a timescale long enough that the temporary (reverting) component has decayed.

**Key caveat — calendar-time bias**: Research on U.S. equities finds that calendar-time estimates of Kyle's lambda overstate trade-time estimates by 20–50%, especially around high-activity periods. For crypto tick data where trading is continuous 24/7 and highly clustered, this bias can be significant. The standard fix is to estimate on trade-time (bucket by trade count, not clock time) or use a two-pass regression that instruments with lagged order flow.

### The Cont-Kukanov-Stoikov (2014) OFI Alternative

Cont, Kukanov & Stoikov (2014) define Order Flow Imbalance as the net change in best-bid and best-ask queue sizes over an interval:

```
OFI = ΔQ_bid - ΔQ_ask
```

They regress midprice changes on OFI and document a mean R² of ~65% across 50 NYSE stocks, stable across time scales and robust to intraday seasonality. The slope of this regression is **also an estimate of Kyle's λ** — the authors themselves note it is a more direct empirical estimator of Kyle's fundamental liquidity parameter, because it captures the full queue dynamics rather than just executed trades.

For a Towards Data Science analysis of OFI applied to cryptocurrency order books, the linear model at 1-minute resolution achieves R² ≈ 40.5% for BTC, rising when aggregated further. A 2024 study of Binance BTCUSDT minute-by-minute data found that order-flow influences on returns were regime-dependent and inconsistent across weeks, partly due to a structural regime shift around October 2023.

### Pros and Cons for Crypto Tick Data

| Criterion | Kyle's Lambda (Cov/Var estimator) | OFI Regression (CKS 2014) |
|---|---|---|
| **Data required** | Trades + trade sign (Lee-Ready or aggressor flag) | L2 order book snapshots at best bid/ask |
| **Availability on Binance** | Readily available: `aggTrades` endpoint gives aggressor side | Requires L2 book data (Binance `depth` stream or Tardis.dev snapshots) |
| **Theoretical grounding** | Directly from Kyle (1985) equilibrium model | Empirically motivated; reduces to Kyle's λ under linear approximation |
| **R² / signal quality** | Lower; trade signs are noisy, especially for maker-initiated flow | Higher (~65% equity, ~40% crypto 1-min); accounts for passive queue changes |
| **Estimation noise in crypto** | High: spoofing, wash trading, fragmented venues inflate variance of signed flow | Moderate: spoofing inflates queue changes but effect partially cancels in net |
| **Calendar-time bias** | Up to 50% overestimate in high-activity periods | Less susceptible if OFI is computed per-interval not per-trade |
| **Stationarity** | Kyle's λ shifts with regime (e.g., post-Oct 2023 BTC structural break) | Same regime sensitivity; neither estimator is immune |
| **Crypto-specific literature** | Multiple papers apply it to BTC microstructure | Cross-interval OFI model validated on crypto (Springer 2021); LSTM extensions studied |

### Recommendation for This Project

**Use Kyle's lambda (Cov(ΔP, signed_flow) / Var(signed_flow)) as the primary estimator for γ**, because:

1. The `aggTrades` endpoint on Binance directly provides the aggressor flag, making signed flow trivial to compute. You do not need L2 book data for γ.
2. The formula is exactly what Almgren-Chriss (2001) Section 3 has in mind for the permanent impact slope.
3. For a course project, the computational overhead of full OFI (which requires synchronized L2 snapshots) is not justified by marginal accuracy gains.

**Mitigate calendar-time bias** by: (a) bucketing into fixed-trade-count windows (e.g., every 500 trades) rather than fixed clock windows, or (b) estimating over 30-second or 1-minute intervals and checking for intraday stability. If γ varies by more than 2x across the trading day, report this as a limitation.

**Do not use OFI as a replacement** unless you already have Tardis.dev L2 data for the temporary impact calibration (Q2 below), in which case you can run both regressions and compare R².

**Key references**:
- Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." *Econometrica* 53(6): 1315–1335.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events." *Journal of Financial Econometrics* 12(1): 47–88. ([arXiv:1011.6402](https://arxiv.org/abs/1011.6402))
- Hasbrouck, J. (2009). "Trading Costs and Returns for U.S. Equities." *Journal of Finance* 64(3).
- Revisiting U-shaped patterns in volatility and price impacts (2025): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1386418125000114)
- Explainable Patterns in Cryptocurrency Microstructure (2025): [arXiv:2602.00776](https://arxiv.org/html/2602.00776v1)

---

## Q2: Temporary Impact Calibration Without Order Book

### The Problem with Pure Trade-Data Power-Law Regression

The Almgren-Chriss temporary impact model is:

```
g(v) = η · (v / V)^α
```

where `v` is the instantaneous trade rate, `V` is average daily volume, η is the cost coefficient, and α is the concavity exponent. Fitting this from trade data alone (slippage ~ η · qty^α) is noisy because:

1. You observe only **realized execution prices**, which confound temporary and permanent impact.
2. On Binance spot, most retail trades are small and sit far from the regime where temporary impact is measurable against noise.
3. BTCUSDT has tight spreads (~0.01%), so slippage at small sizes is dominated by fee structure and price discretization, not market impact.

This is a well-known identification problem: without knowing what price would have prevailed without your trade, you cannot cleanly separate temporary from permanent impact from a single trade stream.

### Would Tardis.dev L2 Snapshots Help?

Yes, meaningfully, but with caveats:

**What L2 depth adds**:
- You can compute the **instantaneous cost of walking the book** for a given quantity at each snapshot: `slippage(qty, t) = (average_fill_price(qty, t) - midprice(t))`. This is a direct measure of the temporary market impact *at that moment*.
- By regressing `slippage(qty, t)` on `qty` across thousands of snapshots, you get a cleaner power-law fit because you are measuring the supply curve directly, not observed execution costs.
- The Tardis.dev `book_snapshot_25` feed for BTCUSDT provides top-25 levels at millisecond granularity. The open-source [`data-tardis`](https://github.com/nkaz001/data-tardis) repo already reconstructs market depth and computes imbalance from these feeds.

**Residual limitations**:
- L2 snapshots show posted liquidity, not *executable* liquidity (hidden orders, iceberg orders are invisible).
- During stress periods (liquidation cascades, news), the book thins dramatically and the power-law relationship breaks.
- For a course project, obtaining and processing Tardis.dev data has a cost (free tier is limited; historical Binance L2 is paid).

**Bottom line on Tardis.dev**: If you can access the data, running the book-walk regression will substantially reduce noise in η. If not, the literature fallback is acceptable.

### Is the Literature Fallback (α ≈ 0.5–0.6) Acceptable?

**Yes, for a course project**, with the following qualifications:

- **Almgren et al. (2005)** ("Direct Estimation of Equity Market Impact", Citigroup dataset) empirically rejected the square-root law (α = 0.5) for temporary impact and found α ≈ 0.6 (3/5 power law). This is from a large U.S. equity dataset, not crypto.
- **Bouchaud et al. (2004)** and the broader "square root law" literature find δ ≈ 0.5 for metaorder impact (total impact scales as √Q). The square-root arises from the OFI-to-price-change relationship and is confirmed across asset classes including crypto.
- **For BTC specifically**: The Anboto Labs analysis of Binance BTCUSDT applies the Almgren-Chriss framework and notes that the temporary impact function needs to be adapted to crypto's tighter spreads and higher liquidity relative to mid-cap equities. A 2022 Claremont thesis on cryptocurrency optimal execution directly applies the A-C model with standard power-law calibration.

**Practical recommendation**:

1. **Set α = 0.6** (Almgren 2005 estimate) as your baseline. This is defensible in a course context with a citation.
2. **Calibrate η from the bid-ask spread** using the Almgren-Chriss heuristic: assume executing 1% of average daily volume incurs temporary impact equal to one full spread. For BTCUSDT with spread ≈ 0.5 bps and ADV ≈ $20B/day:
   ```
   η = spread / (0.01 × ADV_in_shares)
   ```
   Convert to per-share units consistent with your price units.
3. **Report a sensitivity analysis**: show how the efficient frontier shifts when α ∈ {0.5, 0.6, 0.7} and η ±50%. This is standard practice and demonstrates robustness awareness.
4. If you do get even a few thousand aggressive fills with size information, run OLS log(slippage) ~ α·log(qty) + log(η) and report whether your fitted α is consistent with the literature priors. Even a noisy estimate with R² = 0.15 is more honest than purely assuming.

**Key references**:
- Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). "Direct Estimation of Equity Market Impact." *Risk* 18: 57–62. ([Semantic Scholar](https://www.semanticscholar.org/paper/Direct-Estimation-of-Equity-Market-Impact-Almgren-Thum/00777edc168f26633de9f5b9ff4c4f74bd9790e3))
- Bouchaud, J.-P., Gefen, Y., Potters, M., & Wyart, M. (2004). "Fluctuations and Response in Financial Markets." *Quantitative Finance* 4(2): 176–190.
- Gatheral, J. (2010). "No-Dynamic-Arbitrage and Market Impact." ([Baruch lecture notes PDF](https://mfe.baruch.cuny.edu/wp-content/uploads/2012/09/Chicago2016OptimalExecution.pdf))
- Tardis.dev documentation: [https://docs.tardis.dev/faq/data](https://docs.tardis.dev/faq/data)
- nkaz001/data-tardis (market depth reconstruction): [GitHub](https://github.com/nkaz001/data-tardis)
- Anboto Labs — Deep Dive into Almgren-Chriss: [Medium](https://medium.com/@anboto_labs/deep-dive-into-is-the-almgren-chriss-framework-be45a1bde831)
- Optimal Execution in Cryptocurrency Markets (Claremont thesis, 2022): [CMC Theses](https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=3566&context=cmc_theses)

---

## Q3: Realized Volatility Estimator for 24/7 Crypto Markets

### Your Current Approach and Its Issues

Annualizing 5-min realized variance via `σ_annual = σ_5min × √(seconds_per_year / 300)` is the **close-to-close realized variance** approach. This is:

```
RV = Σ r_t²   (sum of squared 5-min log returns)
σ_annual = √(RV × T_annual / T_sample)
```

where for 24/7 crypto, `T_annual / T_sample = 365 × 24 × 12` (number of 5-min bars per year). This is equivalent to what Coin Metrics and other data providers use as the industry standard for BTC annualization — **√365 scaling rather than √252**, because crypto trades all year round.

**Your formula is correct in convention.** The specific issue in practice is:

1. **Microstructure noise at 5-min**: At very high frequencies, bid-ask bounce, discrete tick sizes, and asynchronous trades inflate realized variance. The bias-variance tradeoff literature (Bandi & Russell 2006, Zhang et al. 2005) shows 5-minute sampling is close to the empirically optimal frequency for equities; for BTC on Binance, which has sub-second trades and ~$0.1 tick, microstructure noise is meaningful at 1-min but largely attenuates by 5-min.
2. **No jump filtering**: Crypto has frequent price jumps (FOMC, exchange outages, large liquidations). Squared returns at 5-min include both diffusion and jump variance, which you likely want to separate for the Almgren-Chriss σ parameter (which should proxy diffusion vol, not total variation).

### Range-Based Estimators: Which Apply to Crypto?

All four major range-based estimators use OHLC data from a bar (open, high, low, close). Their efficiency relative to close-to-close estimators:

| Estimator | Efficiency vs C-to-C | Handles Drift | Handles Overnight Jump | Crypto Applicable |
|---|---|---|---|---|
| **Parkinson (1980)** | ~5x | No (assumes zero drift) | No | Yes, but biased for trending BTC |
| **Garman-Klass (1980)** | ~7.4x | No | No | Yes, widely used for BTC |
| **Rogers-Satchell (1991)** | ~8x | **Yes** | No | **Best for intraday crypto** |
| **Yang-Zhang (2000)** | ~14x | Yes | **Yes** | Partial — overnight component irrelevant for 24/7 |

**Yang-Zhang for crypto**: The estimator combines overnight close-to-open variance with intraday Rogers-Satchell variance. For a 24/7 market like BTC, there is no meaningful "overnight" session break, so the overnight jump component adds noise rather than signal. Practitioners and the QuantConnect community explicitly note this limitation: for continuously-traded instruments, Rogers-Satchell is preferred over Yang-Zhang.

**Garman-Klass for crypto**: A 2024 study (Kristoufek et al., *Applied Economics Letters*) using Binance 5-minute OHLC data for BTC, ETH, BNB, XRP, and DOGE (July 2019–September 2022) found that the **Garman-Klass estimator clearly outperforms GARCH-family models** in both in-sample fit and out-of-sample forecasting. This is the most directly relevant empirical finding for your project setup.

**Rogers-Satchell for crypto**: Accounts for non-zero drift without the overnight gap assumption. A dedicated study on Bitcoin volatility (*Journal of FIRM*, 2022) applies the Rogers-Satchell range model directly to BTC and finds it well-suited to intraday BTC volatility measurement. This is the theoretically cleanest choice for a 24/7 asset.

### Optimal Frequency: 1-min, 5-min, or 15-min?

The standard result from microstructure econometrics (Bandi & Russell 2006) is that **5-minute sampling is the empirically validated sweet spot** for minimizing the bias-variance tradeoff in realized variance. Key evidence:

- At 1-min, bid-ask bounce and discrete ticking inflate variance on BTC by an estimated 10–30% (GitHub: LucasChaka/Bitcoin-Realized-Volatility-Analysis applies kernel-based RV to address this).
- At 5-min, microstructure effects are largely averaged out while still capturing intraday variation.
- At 15-min or 30-min, you lose too many observations, increasing estimation variance (especially in low-vol periods).
- The empirical crypto study above uses Binance 5-min bars specifically and confirms the frequency is appropriate.

For the Almgren-Chriss σ parameter (daily volatility in dollar terms), you want a **daily or sub-daily volatility** estimate that feeds the execution horizon. Best practice:

1. Compute 5-min Garman-Klass or Rogers-Satchell estimates.
2. Average over a rolling window (20 trading days × 24 hours × 12 bars = 5760 bars) to get a stable daily σ.
3. Annualize with `√(365 × 24 × 12)` for consistency with the 5-min bar count.

### Specific Annualization Formula

For 5-min bars on a 24/7 market:

```python
# Number of 5-min bars in a year (24/7)
bars_per_year = 365 * 24 * 12  # = 105,120

# Close-to-close realized variance over N bars
RV_5min = sum(r_t**2 for r_t in log_returns)  # sum over N bars

# Annualized volatility
sigma_annual = sqrt(RV_5min / N * bars_per_year)
```

This is equivalent to your `√(seconds_per_year / 300)` formula (since `seconds_per_year / 300 = 31,536,000 / 300 = 105,120`). **Your formula is correct.** The issue is not the annualization factor but whether to use close-to-close returns or a range-based estimator for each bar.

### Recommendation

**Primary estimator**: Use **Garman-Klass** on 5-minute OHLC bars. It has the strongest empirical validation specifically on Binance crypto data (Kristoufek 2024), is 7.4x more efficient than close-to-close, and does not require the overnight gap assumption.

**Implementation**:
```python
# Garman-Klass per bar
import numpy as np

def garman_klass(o, h, l, c):
    """o, h, l, c are log prices (or log of prices)"""
    return 0.5 * (np.log(h/l))**2 - (2*np.log(2) - 1) * (np.log(c/o))**2

# For each 5-min bar: GK_t = garman_klass(open_t, high_t, low_t, close_t)
# Rolling daily vol (288 bars per day):
gk_daily = rolling_mean(GK_series, window=288)
sigma_annual = np.sqrt(gk_daily * bars_per_year)
```

**Fallback**: If OHLC is unavailable (you only have trade-level data), close-to-close with 5-min bars and `√105,120` annualization is the defensible industry standard.

**Do not use Yang-Zhang** for BTC as the primary estimator. The overnight jump component is not meaningful for a 24/7 market and inflates the estimator's variance without reducing bias.

**Key references**:
- Kristoufek, L. et al. (2024). "Beyond GARCH in Cryptocurrency Volatility Modelling." *Applied Economics Letters* 32. ([Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/13504851.2024.2363295)) ([preprint PDF](https://library.utia.cas.cz/separaty/2024/E/kristoufek-0599014.pdf))
- Garman, M. B. & Klass, M. J. (1980). "On the Estimation of Security Price Volatilities from Historical Data." *Journal of Business* 53(1): 67–78.
- Rogers, L. C. G. & Satchell, S. E. (1991). "Estimating Variance from High, Low and Closing Prices." *Annals of Applied Probability* 1(4): 504–512.
- Yang, D. & Zhang, Q. (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open and Close Prices." *Journal of Business* 73(3): 477–492.
- Bandi, F. M. & Russell, J. R. (2006). "Microstructure Noise, Realized Volatility, and Optimal Sampling." ([ResearchGate](https://www.researchgate.net/publication/4817469_Microstructure_noise_realized_volatility_and_optimal_sampling))
- Coin Metrics — Realized Volatility methodology (√365 convention): [https://coinmetrics.io/company-news/realized-volatility/](https://coinmetrics.io/company-news/realized-volatility/)
- LucasChaka/Bitcoin-Realized-Volatility-Analysis (kernel-based RV, microstructure noise correction): [GitHub](https://github.com/LucasChaka/Bitcoin-Realized-Volatility-Analysis)
- Portfolio Optimizer — Range-Based Volatility Estimators Overview: [https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/](https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/)

---

## Summary Table

| Parameter | Method | Data Source | Citation |
|---|---|---|---|
| **γ (permanent impact)** | Kyle's lambda: OLS of ΔP on signed_flow, trade-time bucketing | Binance `aggTrades` | Kyle 1985; Hasbrouck 2009 |
| **η (temporary impact coeff)** | Literature prior: spread / (0.01 × ADV); sensitivity over α ∈ {0.5, 0.6, 0.7} | Binance 24h volume + spread | Almgren et al. 2005 |
| **α (temporary impact exponent)** | Fix at 0.6 (Almgren 2005 3/5 power law); check fit if fills available | — | Almgren 2005; Bouchaud 2004 |
| **σ (volatility)** | Garman-Klass on 5-min OHLC bars, annualized with √105,120 | Binance `klines` 5m | Kristoufek 2024; Garman-Klass 1980 |
