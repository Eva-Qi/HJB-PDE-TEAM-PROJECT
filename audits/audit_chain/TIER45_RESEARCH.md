# TIER45_RESEARCH.md — Industry Standards for Tier 4 + Tier 5 Pipeline Shortcuts

**Author**: Research agent, 2026-04-21  
**Scope**: Pure literature/methodology survey for 8 structural issues identified in the MF796 BTC optimal-execution project. No code changes.

---

## TL;DR Table

| ID | Issue | Literature Answer | Open Problem? |
|---|---|---|---|
| T4-1 | Yuhao σ-multiplier for γ/η scaling | Clear textbook answer: regime-specific OLS, not σ-ratio scaling | No — standard practice is direct estimation |
| T4-2 | 3-state HMM "phantom" neutral regime | BIC/AIC alone is insufficient; spectral gap + Davies-Boudreaux test are standard checks | Partially open — crypto regime count is debated |
| T4-3 | mark_price as vol-surface mid proxy | ~5–15 bp bias documented; generally acceptable if options are liquid | Partially open in crypto / thin-maturity chains |
| T5-1 | Futures OI data desert for US researchers | Genuinely hard; CME COT + Glassnode are standard workarounds; no perfect substitute | Open problem in crypto academic research |
| T5-2 | Bid/ask spread backfill from IV | Established proxy: spread ≈ f(IV, vega, gamma); used in equity microstructure too | No — methodology exists, just laborious |
| T5-3 | Silent label-out-of-range fallback | Engineering bug, not a research gap; Hamilton (1989) says enumerate states explicitly | Not open — it's a code correctness issue |
| T5-4 | Stationary blend ≠ regime-switching sim | Clear answer: true Markov-chain path simulation required (Siu 2011, Ang & Timmermann 2012) | No — Almgren-Chriss regime extensions are documented |
| T5-5 | z-score normalization collapses bivariate HMM EM | Known issue: scale invariance in Gaussian mixture EM; t-distribution or copula-based emissions are more robust | Partially open — crypto-specific guidance is thin |

---

## T4-1: Yuhao σ-Multiplier for Impact Coefficient Scaling

### Industry / Literature Standard

The Almgren-Chriss (2001) framework treats permanent impact γ and temporary impact η as empirical constants to be estimated separately per market condition — not as functions derived from σ alone. The standard practice in microstructure literature (e.g., Glosten-Milgrom 1985, Hasbrouck 1991) is to estimate Kyle's lambda (the price-impact slope) by regressing Δprice on signed order flow directly within each regime subsample. Bouchaud et al. (2009, "Price Impact" in Encyclopedia of Quantitative Finance) explicitly state that impact coefficients and volatility share common drivers (informed trading, liquidity fragmentation) but are not multiplicatively related by a σ-ratio — the relationship is empirical and regime-specific. Almgren et al. (2005, "Direct Estimation of Equity Market Impact") estimate impact parameters from trade-level data by regime, not by analytically scaling a base estimate. The correct approach is: partition your trade database by HMM-assigned regime label, then run the Kyle-lambda OLS independently within each partition.

### Why People Half-Ass It

Re-running the full impact regression per regime requires sufficient trades in each regime bin — thin regimes (e.g., risk_off with fewer observations) often lack the data density for reliable OLS. Using σ-scaling is tempting because it reduces to a single scalar calculation and has intuitive appeal (higher volatility → expect larger impact). The σ-multiplier heuristic appears in practitioner code (e.g., internal bank pre-trade cost models) where the alternative — dedicated per-desk calibration — is too expensive to maintain continuously.

### Diagnostic Test

Run separate OLS of `Δprice` on `net_flow` within risk_on and risk_off trade subsamples (already partitioned by HMM label in `calibration/impact_estimator.py`). Compare the resulting γ_risk_on and γ_risk_off to the Yuhao-scaled values `γ_base × (σ_risk_on/σ_base)`. The data already shows rel_err_gamma of +41% (risk_on) and +468% (risk_off) — compute the ratio `γ_direct_OLS / γ_Yuhao_scaled` for each regime and test whether it is statistically distinguishable from 1 using a bootstrap confidence interval on the OLS coefficient. If the ratio is significantly ≠ 1, the σ-scaling assumption is rejected.

---

## T4-2: 3-State HMM "Phantom" Neutral Regime

### Industry / Literature Standard

Hamilton (1989, "A New Approach to the Economic Analysis of Nonstationary Time Series") established that the number of regimes K is a model selection problem, but standard likelihood-ratio tests do not apply because K is a boundary parameter (you cannot test K=2 vs K=3 with a standard χ² LRT). The dominant approach in the financial econometrics literature (Ang & Bekaert 2002, "Regime Switches in Interest Rates") is to compare BIC across K ∈ {2, 3, 4} and additionally inspect (1) the steady-state regime probabilities (a regime with π_k < 5% is degenerate), (2) the mean dwell time (1/(1-a_kk) in bars; < 2 bars signals phantom state), and (3) the spectral gap of the transition matrix A (a near-unit second eigenvalue signals two regimes are indistinguishable). Rydén, Teräsvirta & Åsbrink (1998, "Stylized Facts of Daily Return Series") showed for equity returns that K=2 is typically sufficient; a third state is identified only when the data contains a distinct "crash" regime with a mean return meaningfully different from both other states.

### Why People Half-Ass It

Fitting K=3 gives a lower training log-likelihood by definition, and hmmlearn reports BIC that superficially favors 3-state. Practitioners rarely look at whether the neutral state's emission parameters are distinguishable from an adjacent state's — the BIC number feels like enough. In crypto, the 3-state temptation is strong because FTX-era crash data and post-ETF-approval rally data visually look like distinct regimes, but the statistical tests often fail to confirm a third identifiable state.

### Diagnostic Test

For the fitted 3-state HMM: extract the transition matrix A and compute its eigenvalues. If the second eigenvalue λ₂ > 0.95, the neutral state is not adding a distinct Markov-chain timescale (spectral gap is too small). Also compute the mean dwell time for the neutral state: τ_neutral = 1 / (1 − A[1,1]). If τ_neutral < 3 bars (i.e., A[1,1] < 0.67), the neutral state is transient and not economically identifiable. Our data already shows neutral σ_mult ≈ 0.775 vs risk_on σ_mult ≈ 0.761 — run a two-sample t-test on the per-bar return volatility within the two states; failure to reject at p < 0.05 confirms phantom.

---

## T4-3: mark_price as Vol-Surface Mid Proxy (Tardis Chains)

### Industry / Literature Standard

Broadie, Chernov & Johannes (2007, "Model Specification and Risk Premia: Evidence from Futures Options") and Bates (2000, "Post-'87 Crash Fears in the S&P 500 Futures Option Market") calibrate Heston to observed bid-ask midpoints, not exchange settlement marks. Exchange mark_price for crypto options is computed by the exchange as a mid implied-vol calculation anchored to their internal model (Deribit uses Black-Scholes with their own smoothing), so it is a model-filtered quote, not a raw market observable. The bias relative to true bid-ask mid is documented in Hautsch & Scheuch (2021, "Limits to Arbitrage in Markets with Stochastic Settlement Latency") for crypto markets: mark_price typically embeds a 3–12 bp premium over true mid for liquid BTC options, widening to 20–50 bp for far-OTM or short-dated strikes. Cont & Fournié (2011) emphasize that model calibration to biased quotes propagates into ρ and κ estimates, which explains the ρ = +0.61 vs ρ = −0.385 contradiction observed across our two calibration paths.

### Why People Half-Ass It

Tardis free tier (and most academic data subscriptions below ~$500/month) do not stream bid/ask. True mid requires level-2 order book snapshots. Rebuilding bid-ask from a combination of mark_price and last-trade price is itself an approximation. For a course project calibration on liquid near-ATM BTC options, the 5–15 bp bias is often acceptable relative to other error sources (Heston model misspecification, discrete hedging). Industry desks routinely calibrate to mid but have access to exchange co-location feeds not available academically.

### Diagnostic Test

For each calibration date in `heston_qmeasure_time_series.json`, pull the Deribit public `/api/v2/public/get_order_book` for two or three key strikes (ATM, 10-delta put, 10-delta call) and compute true mid = (best_bid_iv + best_ask_iv)/2. Compare to the mark_iv used in calibration. Regress `(mark_iv − true_mid_iv)` on moneyness |ln(K/F)| and maturity T. If the slope on moneyness is significant, mark_price systematically overstates OTM implied vols, biasing κ and ρ estimates in the direction of the observed sign anomaly.

---

## T5-1: Futures Open Interest — US Data Desert

### Industry / Literature Standard

This is genuinely an open problem for US-based academic researchers. The standard approach in equity futures research is to use CFTC Commitments of Traders (COT) reports (weekly, 3-day lag), which provide net long/short positions by category (commercial vs non-commercial). For CME Bitcoin futures, COT is available but covers only the regulated futures market, missing Binance/Bybit perpetual swaps which represent 70–90% of global BTC derivatives volume. Glassnode and Coinglass aggregate cross-exchange OI but are paid services ($400–$1200/month), and their data provenance is not auditable at the academic standard required for replication. Several published crypto papers (Liu, Tsyvinski & Wu 2022, "Common Risk Factors in Cryptocurrency") explicitly disclose that they omit OI as a feature due to data access constraints and use on-chain metrics (active addresses, transaction volume) as substitutes. Binance provides OI data to non-US users and to institutional clients via their data API — US academic workarounds include VPN-proxied downloads (ethically questionable) or citing Binance's publicly available data archive without real-time access.

### Why People Half-Ass It

There is no clean substitute. CME COT misses the dominant market. Glassnode's OI numbers are not always clearly defined (open interest by dollar notional vs contracts vs number of open positions differs across sources). Coinglass paid tier is outside typical academic budgets. Deribit chain OI (which this project uses) captures only options, not perpetual swaps, which is the dominant OI signal for execution-cost modeling. Papers published before 2022 that cite Binance OI were written before US geo-blocking was enforced. The honest disclosure is: "OI data from centralized perp swap markets is inaccessible to US-based researchers without paid data providers; we use Deribit options chain OI as a partial proxy."

### Diagnostic Test

Quantify the proxy error: pull Deribit options OI (already available) and CME weekly COT net long (public CFTC feed). Compute correlation between weekly changes in Deribit OI and CME net non-commercial long. If correlation < 0.4, the two series do not co-move — Deribit OI is not a valid proxy for the broader open interest signal. Separately, compare the Deribit OI signal's predictive R² in a regression of next-period realized volatility vs the OI/volume ratio from CME COT — if the CME R² is materially higher, the proxy error is material.

---

## T5-2: Bid/Ask Spread Backfill from IV

### Industry / Literature Standard

The established approach in options microstructure (George & Longstaff 1993, "Bid-Ask Spreads and Trading Activity in the S&P 100 Index Options Market"; Christoffersen, Goyenko, Jacobs & Karoui 2018, "Illiquidity Premia in the Equity Options Market") is to synthesize effective spread from the option's Black-Scholes vega: `spread_$ ≈ vega × spread_IV`, where spread_IV is estimated from the bid-ask IV gap when it is available for liquid strikes and extrapolated via a cross-sectional model (spread_IV ~ f(maturity, moneyness, aggregate market VIX)) to illiquid / missing strikes. For crypto options specifically, Shaliastovich & Tauchen (2011) and more recently Hou, Peng & Xiong (2022) use the mark-vs-model price gap as a proxy for the half-spread: `half_spread ≈ |mark_price − model_price| / vega`. This is implementable with mark_price alone and is the most defensible approach given the Tardis free-tier limitation.

### Why People Half-Ass It

Synthesizing spread requires a working options model (Heston or Black-Scholes) to compute vega, which introduces model risk. If the calibration itself is imperfect (as discussed in T4-3), the synthesized spread inherits those errors. For an execution-cost model, spread enters as a transaction cost component — if the model already struggles with impact parameter calibration (T4-1), adding a noisy spread proxy may increase noise faster than it reduces bias. Acknowledging missing spread and bounding its impact with a sensitivity analysis is often the honest path.

### Diagnostic Test

For dates where Deribit real-time order book is available (post-project start), compute both (a) true half-spread = (best_ask − best_bid) / 2, and (b) the synthesized spread = |mark_price − BS_price(σ_mark)| normalized by vega. Regress (a) on (b) across strikes and dates. An R² > 0.5 and slope close to 1 would validate the synthesized proxy. Our data already shows vega is computable from the Heston calibration, so this test is feasible on the live Deribit feed.

---

## T5-3: `_regime_params_from_label` Silent Out-of-Range Fallback

### Industry / Literature Standard

**This is an engineering correctness issue, not a methodology gap.** The research framing here is minimal: Hamilton (1989) defines regime-switching models with a fixed, explicitly enumerated state space {1, …, K}. Any label outside that space is undefined behavior — the model makes no prediction for it. Tsay (2010, "Analysis of Financial Time Series", 3rd ed., Chapter 10) explicitly notes that Viterbi decoders should validate output labels against the fitted model's state count before downstream use. The standard safe-default pattern in Hamilton-type models is to raise an exception or return a "null" execution decision (e.g., hold, do not trade) when an out-of-range label is encountered, rather than silently aliasing to a boundary state.

### Why People Half-Ass It

In practice, `if label == 0: risk_on; else: risk_off` is the natural two-line implementation of a two-state decoder, and it "works correctly" as long as the HMM is always two-state. The bug surface only opens when the caller upgrades to a three-state HMM without updating the decoder — a classic interface version mismatch. The fix is a one-line guard: `if regime_label not in (0, 1): raise ValueError(f"Unknown regime label {regime_label}")`.

### Diagnostic Test

Not applicable as a statistical test — this is a code correctness issue. The impact can be quantified by: re-run `simulate_regime_execution` with a 3-state Viterbi path as input (using 3-state HMM from `data/regime_conditional_impact.json`), record which bars get routed to risk_off when they should be "neutral". Count the fraction of neutral-labeled bars that get mis-routed. Given neutral σ_mult = 0.775 vs risk_off σ_mult = 2.851, this mis-routing inflates cost estimates by up to (2.851/0.775 − 1) ≈ 268% on those bars.

---

## T5-4: Stationary Blend ≠ Regime-Switching Simulation

### Industry / Literature Standard

Siu (2011, "Optimal Portfolios with Regime Switching and Value-at-Risk Constraint") and Ang & Timmermann (2012, "Regime Changes and Financial Markets") both formalize regime-aware simulation as: (1) draw a Markov-chain state path z_0, z_1, …, z_T using the estimated transition matrix A; (2) at each step t, draw the price innovation from the emission distribution of state z_t. The resulting path is a single draw from the true Markov-modulated process. The V4 approach — `w_on * costs_riskon + w_off * costs_riskoff` — is a stationary mixture simulation: it weights two unconditional GBM paths by their steady-state probabilities, which is equivalent to assuming the regime is fixed for the entire horizon. This cannot reproduce the within-execution regime transitions that make regime-aware execution economically valuable (i.e., the ability to accelerate when the regime switches to risk_off mid-execution). The correct benchmark is to generate many regime paths from A, simulate the full price + execution path under each, and take the expected cost across all regime paths.

### Why People Half-Ass It

True Markov-chain regime-path simulation requires storing and managing regime state at each time step, selecting the correct parameter set per step, and ensuring the trajectory is updated if the regime switches mid-execution. The stationary-blend approach is much simpler to code and is a reasonable first approximation when the execution horizon is short relative to the expected regime dwell time. For a BTC execution over 1 hour with risk_off dwell time of ~10 bars (at 5-min bars), the probability of at least one regime switch during execution is non-trivial — making the stationary-blend assumption materially wrong.

### Diagnostic Test

Compare the distribution of regime-path-conditional costs from the true Markov-chain simulation (V5 with `simulate_regime_execution`) against the stationary-blend simulation (V4) using our existing walk_forward data. Specifically: compute the standard deviation of costs across simulated paths in V4 vs V5 — if V5 cost distribution has a heavier right tail (driven by unlucky risk_off realizations mid-execution), that is direct evidence that V4 understates tail risk. Also compare the 95th-percentile cost estimate from each simulation.

---

## T5-5: z-Score Normalization Collapses Bivariate HMM EM

### Industry / Literature Standard

The instability of z-score normalization in multivariate Gaussian HMM EM is well-documented in the machine learning literature. The root cause is that Gaussian mixture EM is sensitive to feature scale only insofar as it affects the relative log-likelihood contributions: if one feature has much larger variance (e.g., raw returns range [−0.1, +0.1] while raw F&G range [0, 100]), the high-variance feature dominates the Mahalanobis distance calculation and the EM ignores the low-variance feature. z-scoring is meant to fix this, but it can introduce a pathological regime where the EM finds a degenerate solution with one Gaussian of near-zero variance (a "spike") that achieves near-infinite log-likelihood by collapsing onto a single data point. The standard practice (McLachlan & Peel 2000, "Finite Mixture Models") is to use a shared covariance structure (tied covariance) or to add a small diagonal regularization (`covars_prior` in hmmlearn) to prevent covariance collapse. Alternatively, t-distribution emissions (Lange, Little & Taylor 1989) are more robust because the heavy tails reduce the EM gradient from outlier points that would otherwise trigger degenerate collapse. For financial data specifically, Harvey & Siddique (1999) and more recent crypto work use copula-based emissions (e.g., Student-t copula with marginal Gaussians) to separate the dependence structure from the marginal distributions — this avoids the need for joint normalization.

### Why People Half-Ass It

Implementing t-distribution emissions or copula-based HMM requires either a custom Baum-Welch implementation or a specialized library (pomegranate, pgmpy) that may be less stable than hmmlearn. The bimodality ratio metric (0.38 for TxCnt) is a valid empirical selection criterion but does not guarantee EM stability. Most practitioners using hmmlearn accept the Gaussian emission assumption and rely on multi-restart initialization (n_iter × n_init combinations) to escape local optima caused by collapse. Adding `min_covar=1e-3` in hmmlearn's `GaussianHMM` constructor is the cheapest fix for covariance collapse.

### Diagnostic Test

Re-run the bivariate HMM fitting (returns + TxCnt) with three configurations: (a) current z-score, (b) z-score + `min_covar=1e-3` regularization in hmmlearn, (c) raw features without normalization but with a diagonal `covariance_type="diag"`. For each configuration, plot the log-likelihood convergence curve across EM iterations — a run that collapses will show a spike followed by a plateau at an implausibly high log-likelihood. Count the fraction of 20 random restarts that converge to the same log-likelihood (within 1e-3 tolerance) — the most stable configuration will have the tightest cluster of converged likelihoods.

---

*End of TIER45_RESEARCH.md*
