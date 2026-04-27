# HMM — Regime Detection for Optimal Execution

> **TL;DR / Current state (2026-04-26)**: We use a **2-state GaussianHMM on raw 5-min BTC log-returns** as the default regime spec for Part E. The pre-Apr-26 narrative claimed "3-state preferred by ΔBIC=−4,138,934" but that was a unit-error artifact — `hmmlearn.GaussianHMM.score(X)` already returns the total log-likelihood; the audit script was multiplying it by `T=28,223`. After fixing the bug and adding two restart filters (occupancy <1%, σ-ratio <1.15), 3-state on returns collapses on **14 of 15** random restarts — the surviving fit beats 2-state by only ΔBIC=+6 (Kass-Raftery "weak"). On a log-volatility feature `log(rolling_std(returns, window=6))`, 3-state cleanly separates with ΔBIC=−12,927 and three substantial regimes. Interpretation: BTC volatility itself is trimodal, but raw 5-min returns are too noisy for the HMM to identify the medium regime.
>
> **Key result**: 2-state on returns is the report's main spec; σ ratio 3.26 confirms non-degeneracy. 3-state on log_vol is shown as an ablation — defer to future work because changing the feature would invalidate all upstream Part E calibrations (V4/V5 paired tests, walk-forward).
>
> **Status**: stable. Bug fixed and ablation pushed to `main` (commits `cef075a`, `e930118`).

---

## Current state — what we are doing

### 1. Two-state GaussianHMM on 5-min log returns (default)

The HMM is fit on the cached series `data/btc_5min_log_returns_2026-01-01_to_2026-04-08.npy` (98 calendar days × 288 bars/day ≈ 28,223 observations after NaN drop). The emission distribution is univariate Gaussian; states are interpreted as `risk_on` (low σ) and `risk_off` (high σ) by sorting on emission variance.

Annualized per-state vol on the 98-day window (univariate fit):

| State | σ (annualized) | Occupancy |
|---|---|---|
| risk_on | 0.674 | ~94.5% |
| risk_off | 2.357 | ~5.5% |

Spread = 250%. Round-trip recovery is stable.

### 2. The state-count audit (commits `cef075a`, `e930118`)

The previous version of `data/hmm_2state_vs_3state.json` reported "3-state preferred by ΔBIC = −4,138,934". This was a **unit error** in `scripts/compare_2state_vs_3state_hmm.py:106`:

```python
log_likelihood = float(best_model.score(X)) * T  # WRONG
```

`hmmlearn.GaussianHMM.score(X)` runs the forward algorithm and returns the joint log-likelihood `log P(X | model)` of the entire sequence — it is **not** a per-observation average. Multiplying by `T=28,223` inflated BIC by the same factor. The reported log-likelihood was 4.222×10⁹; for 28k Gaussian observations with std ≈ 1.66e-3, total log-lik should be ~150k. A log-lik of 4 billion is physically impossible at this data size.

After fix:

```
Model      n_params   log-lik     BIC
2-state           7    146,071  -292,070
3-state          14    146,104  -292,064
                                 -------
                       ΔBIC (3-2) = +6.0  → 2-state preferred
```

ΔBIC = +6 is "weak" evidence by Kass-Raftery (1995) thresholds.

### 3. The restart audit — 14/15 collapses on 3-state

Each model fit ran 15 random initializations, with two filters applied per restart:
- **Occupancy filter**: reject if any state has <1% occupancy ("phantom state").
- **Duplicate-σ filter**: reject if any two states have emission σ within ratio 1.15 ("duplicate state"). The σ-ratio filter is feature-specific — see §5 below.

| Model | Converged | Occupancy reject | σ-ratio reject | Min σ-ratio observed |
|---|---|---|---|---|
| 2-state | 15/15 | 0 | 0 | 3.263 |
| 3-state | 15/15 | 11 | 3 | **1.005** |

A min σ-ratio of 1.005 means two of the three states have emission σ within 0.5% of each other — they are mathematically the same state. The EM algorithm is duplicating `risk_on` rather than discovering a third regime.

### 4. The vol-feature ablation (`scripts/compare_hmm_vol_feature.py`)

Replacing raw returns with `log_vol_t = log(rolling_std(returns, window=6))` (window=6 bars = 30 min) flips the result:

```
Model         n_params   log-lik         BIC
2-state              7   -21,374       42,820
3-state             14   -14,875       29,893
                                       -------
                          ΔBIC (3-2) = -12,927  → 3-state preferred (decisively)
```

Three substantial regimes on log_vol:

| Label | σ_state | Occupancy | Mean log_vol |
|---|---|---|---|
| risk_on | 0.254 | 43.4% | low (lowest mean) |
| neutral | 0.377 | 27.3% | mid |
| risk_off | 0.428 | 29.3% | high |

ΔBIC = −12,927 is "decisive" (Kass-Raftery: >10 = strong, this is 1300× past that). All three regimes have substantial occupancy, σ monotonically increasing, no phantom or duplicate state.

**Interpretation**: BTC volatility itself is trimodal (low/medium/high). Raw 5-min returns can't identify the medium regime because returns are zero-mean — Gaussian HMM is forced to use emission σ as the only separator, and σ_low ≈ σ_med on noisy 5-min data. log_vol gives the HMM emission means to separate on, not just variances.

### 5. Why σ-ratio filter is correct only for raw returns

The duplicate-σ filter rejected all 15 of the 2-state log_vol fits (because σ_state_on ≈ σ_state_off ≈ 0.46 even though their **means** are clearly separated). The σ-ratio filter is correct only for **zero-mean features** where regime separation must come from emission σ. For mean-separated features like log_vol, regimes can have similar emission σ but well-separated emission μ.

`scripts/compare_hmm_vol_feature.py` therefore disables the filter:
```python
result = _fit_and_score(log_vol, n_regimes=K, duplicate_sigma_ratio_threshold=0.0)
```

A more general "emission collapse" check would test whether two states have **both** μ within X% AND σ within Y%, but feature-specific thresholds are sufficient for this audit.

### 6. Recommendation for the report (carried into FINDINGS §5.4 and PRESENTATION_DRAFT slide 8)

| Spec | ΔBIC vs 2-state on returns | Decision |
|---|---|---|
| 2-state on raw returns | 0 (baseline) | **Main spec** |
| 3-state on raw returns | +6 | Reject — 14/15 collapse |
| 2-state on log_vol | n/a (different feature) | Reject — same info as raw |
| 3-state on log_vol | n/a | Defer — promising but changes upstream regime-conditional pipeline |

All existing regime-aware execution work (V4 invalidated, V5 paired test, walk-forward) uses 2-state on raw returns. Validated as non-degenerate (σ ratio 3.26). The bimodal risk-on/risk-off story is consistent with the Almgren-Chriss regime extension narrative.

### 7. Performance note — cached returns

The original `load_trades(directory, start, end)` concatenates **all** matching CSVs in the directory before filtering by date. With 111 daily CSVs at ~50 MB each, the global concat hit ~5 GB of pandas state and induced swap thrashing on the dev machine — the script appeared to run for 7+ minutes before being killed without ever reaching `model.fit`.

`scripts/cache_btc_5min_returns.py` loads each CSV one at a time, immediately resamples to 5-min mid-prices (~288 rows/day), and concatenates only the small series. Peak RAM ~50 MB, runtime ~30 s, output saved to `data/btc_5min_log_returns_<start>_to_<end>.npy`. All HMM ablations after the first load from this cache (~0.1 s) rather than re-loading aggTrades.

---

## Evolution & pivots

### Apr 26 — BIC unit-error fix + vol-feature ablation (TL;DR; current spec)

- Bug fixed in `compare_2state_vs_3state_hmm.py:106`. Added duplicate-σ filter.
- Result: on returns, 2-state preferred (ΔBIC=+6, weak); on log_vol, 3-state preferred (ΔBIC=−12,927, decisive).
- Cached-returns optimization avoids 5 GB pandas blow-up.

### Apr 21 — Multi-feature HMM rejected (FINDINGS §5.4)

The hypothesis was: "adding daily-frequency exogenous features (Fear & Greed, exchange FlowIn/FlowOut) would make regime switches interpretable without losing vol separation." Tested via `fit_hmm()` extended to accept 2-D feature matrices (commit `b032481`):

| Model | σ_on | σ_off | Spread |
|---|---|---|---|
| Univariate (log_return only) | 0.674 | 2.357 | **250%** |
| Bivariate (log_return + F&G) | 0.776 | 1.123 | **45%** |

Adding sentiment **dilutes** regime separation by 5.6×. Sonnet G's CoinMetrics FlowIn experiment (commit `cb2e6fa`) found the same — spread drops from 249.8% to ~46% across all three feature variants (raw, log-scale, +FlowOut).

**Generalized hypothesis**: the failure mode is structural, not signal-specific. Daily-frequency features (F&G, exchange FlowIn) are mismatched with 5-min vol regimes — the HMM gets pulled by the daily aggregation structure and loses intraday vol separation. Applies to any daily-sampled exogenous feature. A higher-frequency proxy (5-min on-chain whale-alert frequency, 5-min Twitter volume) remains untested but is out of scope.

### Apr 20 → Apr 21 — V1 → V5 paired-test progression (FINDINGS §2.2)

The Part E regime-aware vs single-regime execution test went through five iterations. **V4 is invalidated by V5** (commit `6bab5b1`):

| Version | Window | Params | Metric | Result | Status |
|---------|--------|--------|--------|--------|--------|
| V1 | ~7d | σ×1e-8/1e-6 magic | mean cost | p=0.84 | Invalidated — magic params |
| V2 (`1cc770e`) | ~7d | Yuhao multipliers | mean cost | p=0.84 | Null — wrong metric |
| V3 (`ec5fe1d`) | 98d | Yuhao multipliers | mean cost | p=0.84 | Null — wrong metric |
| **V4 (`f6c3ace`)** | 98d | Yuhao multipliers | CVaR₉₅ | **−14.0%, p<0.0001** | **INVALIDATED by V5** |
| **V5 (`6bab5b1`)** | 98d | True per-regime calibration | CVaR₉₅ | **+227.5%, p<0.0001 (reversed)** | Suspect — η fallback in risk-off |
| V6 | ~280d (pending) | True per-regime calibration | CVaR₉₅ | Worker I in flight | Pending |

**Why V4 was wrong**: V4 used Yuhao's σ-based multipliers to construct per-regime ACParams. Sonnet C (commit `bc7a47d`) discovered those multipliers overestimate risk-off γ by 41–469% and underestimate risk-off η by 79–90%. The inflated γ made risk-off execution look expensive, so the regime-aware scheduler shifted execution into risk-on — appearing as tail-risk reduction. It was an artifact of biased params, not a genuine regime benefit.

**Why V5 is also suspect**: when rebuilt with true per-regime calibration, risk-off η falls to literature fallback (η=1e-3) because the risk-off sub-sample is too small (5.5% of bars, ~1500 trades) to support stable regression (R²=0.10, α out of range). The +227.5% CVaR reversal may reflect η fallback noise rather than regime-aware behavior.

**Methodological learning — each version revealed a distinct bias layer**:
- V1: wrong model (σ×1e-8 magic scaling, no microstructure basis)
- V2: wrong sample size (7 days, HMM not stable)
- V3: wrong metric (mean cost; AC objective is cost + λ·risk)
- V4: right metric, wrong params (Yuhao multipliers inflated risk-off γ by 469%)
- V5: right params, wrong data window (risk-off sub-sample too small → η fallback)
- V6 (pending): extended 98→280-day window to resolve sub-sample size issue

Net status: Part E regime-aware benefit is currently **unresolved** on this data window.

### Apr 20 — Yuhao merge (commit `1cc770e`)

`extensions/regime.py` previously computed regime-specific γ and η as `sigma × 1e-8` and `sigma × 1e-6` — magic constants with no microstructure basis (V1 setup). Yuhao replaced these with dimensionless multipliers (~0.8–1.2× of the pooled estimate) derived from per-regime sub-sample calibration. Commit also added `simulate_regime_execution` in `sde_engine.py` (rule + pde modes) and extended `RegimeParams` with diagnostic fields. This unblocked V2 and the eventual V4 false positive.

### Mar 26 — Q&A research (was `qa_part_e_regime_arch.md`)

Specification-time questions, all consistent with the current spec:

- **Q13 (Data sufficiency)**: 5 calendar days of aggTrades is borderline-insufficient for production-quality 2-state HMM but works at 5-min aggregation (~1,440 obs/5d). At 98 days the Markov chain visits both states many times — adequate for parameter stability. Final spec uses 98 days.
- **Q14 (Spread proxy)**: Roll (1984) `s = 2√(−Cov(Δp_t, Δp_{t-1}))`; Abdi-Ranaldo (2017) CHL achieves 0.74 cross-sectional correlation with TAQ effective spread (best in class), but for HMM emission features it suffices to use realized vol; spread proxy was deferred.
- **Q15 (Regime-switch re-plan vs commit)**: re-plan from current inventory `x(t)` with new regime params. Closed-form regime-switching execution (Nguyen-Tran SIAM 2021) shows the optimal trading rate is regime-dependent — committing to the initial plan is suboptimal. Use posterior threshold (e.g., 0.75) to prevent thrashing. Implemented in `simulate_regime_execution`.
- **Q16 (Statistical significance)**: paired MC simulation test on common-random-numbers paths, one-sided paired t-test or Wilcoxon signed-rank. With N=5,000–10,000 simulations and a true 2 bps improvement, power > 0.99 at α=0.01. This is exactly the V1→V5 paired-test framework.
- **Q17 (Permanent-impact omission)**: for **linear** γ, the term `½γX₀²` is trajectory-independent and does not affect the optimal schedule (correctly omitted from HJB). Matters only for **nonlinear** permanent impact (α≠1) or when `X₀/V_daily ≥ 5–10%` (becomes nonlinear in practice). For BTC at 10–1000 BTC vs ~2000 BTC ADV, we are in the safe linear zone for the schedule — but include `½γX₀²` in total-cost reporting.
- **Q18 (Self-impact convention)**: the **standard A&C 2000 convention** is that trade `n_k` impacts prices for trades `k+1, k+2, ..., N` only — **NOT** its own execution price. The MC and `cost_model` had a ~0.05% discrepancy from a self-impact disagreement; align to lagged cumsum: `cum_perm_impact[k] = γ · sum(n_0, ..., n_{k-1})`. Aligning both makes the two implementations agree up to MC noise.

### Mar 22 — Initial methodology spec (was `regime_hmm.md`)

Full theory write-up is below in the Methodology section. Summary of the original spec choices that survived to the final report:

- **2-state GaussianHMM** (vs PELT, Markov-switching ARMA, GARCH regimes, k-means, TAR): wins for online execution because the forward algorithm gives O(K²) per-step filtering and the transition matrix encodes regime persistence. PELT is retrospective only and gives no posterior probability.
- **2 states (vs 3+)**: for parsimony, interpretability (risk-on / risk-off matches the AC narrative), and BIC. Validated post-fix: 3-state on returns collapses 14/15 restarts.
- **Univariate log returns** as the primary feature; multivariate (realized_vol + log_spread + OFI) deferred. The Apr-21 multi-feature experiment confirmed the univariate spec is the right call for 5-min crypto data.
- **Risk-off identified by `np.argmax(model.covars_)`** (state with higher emission variance).

---

## Methodology / theory

### 1. Why regime detection matters for AC execution

The Almgren-Chriss (2001) framework assumes market parameters are **stationary**: σ, γ, η fixed over the execution horizon. In practice:

| Regime | σ | Spread | Depth | Impact | Optimal response |
|---|---|---|---|---|---|
| Risk-off | High | Wide | Shallow | Large | **Trade faster** — reducing time-in-market cuts variance exposure |
| Risk-on | Low | Tight | Deep | Small | **Be patient** — impact dominates, spread risk over more intervals |

The AC urgency parameter `κ = √(λσ²/η)` shifts non-trivially:
- Risk-off: σ rises and η rises (illiquidity); `κ` increases → front-load more aggressively if `σ²` grows faster than η
- Risk-on: both compress; `κ` falls → trajectory flattens toward TWAP

A single static `κ` from time-averaged params gives a trajectory that is systematically wrong — too slow during stress, too aggressive during calm.

Empirical magnitudes (Makarov & Schoar 2020; Cont & Kukanov 2014):
- BTC/USDT 1-h realized vol oscillates between ~30% (calm) and 150%+ (crisis)
- Bid-ask spreads on Binance expand 3–8× during stress events
- Kyle's λ approximately doubles in high-vol regimes

These are large enough to change the optimal trajectory shape meaningfully.

### 2. HMM vs alternatives — why GaussianHMM wins for online execution

| Method | Type | Latency | Online? | Probabilistic? | Regime count |
|---|---|---|---|---|---|
| **GaussianHMM** | Probabilistic latent state | Low (Viterbi O(T·K²)) | Yes (filtering) | Yes | Fixed K |
| PELT | Changepoint detection | Moderate O(T log T) | No (retrospective) | No | Inferred |
| Markov-Switching ARMA (Hamilton 1989) | Regime-switching TS | Moderate | Yes | Yes | Fixed K |
| GARCH-based regimes | Vol clustering | Low | Yes | No | Usually 2 |
| k-means on vol features | Clustering | Fast | No | No | Fixed K |

**HMM wins because**:
- Online inference via forward algorithm (O(K²) per new obs)
- Transition matrix encodes regime persistence (risk-off can be sticky for days)
- Posterior `P(state=k | obs)` enables soft regime blending (vs hard switching)
- Baum-Welch (EM) gives ML estimates in closed form for Gaussian emissions

**Limitations**: K must be pre-specified (use BIC); Markov property cannot capture long-memory; Gaussian emission misspecified for fat tails (mitigate with mixtures or Student-t HMM).

### 3. Model specification (current spec, 2-state univariate)

```
Hidden state: S_t ∈ {0 (risk_on), 1 (risk_off)}

Transition matrix A:
    A[i,j] = P(S_t = j | S_{t-1} = i)
    Sticky: A[0,0] ≈ 0.95, A[1,1] ≈ 0.90 typical

Emission:
    r_t | S_t = k ~ N(μ_k, σ_k²)
```

Risk-off identification: `risk_off_state = np.argmax(model.covars_.flatten())`.

### 4. Per-regime parameter estimation

After Viterbi decoding the state sequence, per-regime AC params:

```python
# σ per regime (annualized for 5-min bars, 24/7 crypto)
def estimate_regime_vol(log_returns, state_sequence, regime,
                        annualize_factor=np.sqrt(288 * 365)):
    mask = (state_sequence == regime)
    return float(log_returns[mask].std() * annualize_factor)

# γ per regime: Kyle's λ regression on regime-filtered trades
# η per regime: aggregated 1-min cascade on regime-filtered trades
```

**Sub-sample stability**: risk-off is 5.5% of 98-day bars (~1,500 trades). At this size, Kyle's λ regression has R²≈0.10 in risk-off and α falls outside literature range, triggering η fallback to 1e-3. This is the V5 failure mode that motivated the 280-day extension (Worker I).

### 5. Transition matrix → regime duration

```
Expected duration of regime k:  E[T_k] = 1 / (1 − A[k,k])

Stationary distribution π:  left eigenvector of A at eigenvalue 1
                              (np.linalg.eig of A.T)
```

These feed `RegimeParams.probability` for soft regime blending in `simulate_regime_execution`.

### 6. Baum-Welch (E-M) summary

**E-step** — Forward-backward:
```
Forward:  α_t(k) = b_k(x_t) · Σ_j [α_{t-1}(j) · A[j,k]]
Backward: β_t(k) = Σ_j [A[k,j] · b_j(x_{t+1}) · β_{t+1}(j)]
Posterior: γ_t(k) = α_t(k)·β_t(k) / Σ_k [α_t(k)·β_t(k)]
ξ_t(j,k) = transition posterior
```

**M-step** — Gaussian emission update:
```
π_k = γ_1(k)
A[j,k] = Σ_t ξ_t(j,k) / Σ_t γ_t(j)
μ_k = Σ_t γ_t(k)·x_t / Σ_t γ_t(k)
Σ_k = Σ_t γ_t(k)·(x_t-μ_k)(x_t-μ_k)ᵀ / Σ_t γ_t(k)
```

**Viterbi** — single most probable state path (backtrack from `argmax_k δ_T(k)` where `δ_t(k) = b_k(x_t) · max_j [δ_{t-1}(j) · A[j,k]]`). hmmlearn handles all of this via `model.fit(X)` and `model.predict(X)`.

### 7. Online forward filter (for live execution)

```python
def online_regime_filter(model, new_obs, prev_alpha):
    A = model.transmat_
    b = np.array([multivariate_normal.pdf(new_obs.flatten(),
                                           mean=model.means_[k],
                                           cov=model.covars_[k])
                  for k in range(model.n_components)])
    alpha_t = b * (A.T @ prev_alpha)
    alpha_t /= alpha_t.sum()
    return alpha_t, int(np.argmax(alpha_t))
```

Single-step update is O(K²). Used by `simulate_regime_execution` in rule mode.

### 8. Re-plan vs commit on regime switch (Q15)

Re-plan from `x(t)` with new regime params. Closed-form regime-switching execution (Nguyen & Tran, SIAM CT 2021) and Cartea-Jaimungal regime-switching market resilience (JEDC 2019) both establish that the optimal trading rate is regime-dependent. Use a hysteresis threshold (posterior > 0.75) and 2–5-min wait window to prevent thrashing on noise.

### 9. Statistical significance (Q16)

Paired MC simulation test on common-random-numbers (CRN) paths:
1. Fit HMM + per-regime impact params on training data
2. Simulate N=5,000–10,000 price paths
3. On **each path** run both unconditional and regime-conditional strategy
4. Test pair `IS_i^A − IS_i^B` with one-sided paired t-test or Wilcoxon signed-rank

This is exactly the V1–V5 framework. With N=5,000 and a 2 bps true effect, power > 0.99 at α=0.01.

**Common pitfalls** (avoided in V1–V5):
- Compare OOS only, never IS (regime model has more parameters → looks better in-sample by construction)
- Both strategies receive the same price path and same η, σ — they must differ only in **schedule**
- Report full distribution of improvements, not just mean

### 10. Bivariate HMM — why F&G and FlowIn diluted regimes

Daily-frequency exogenous features (F&G, exchange FlowIn/FlowOut) compress σ spread from 250% to ~45% on the same window because:
- The HMM's emission Gaussian is fit jointly over all features
- Daily features have low-frequency content that the HMM picks up as "different" states even when 5-min vol is stable
- Net result: regime separation is dominated by daily structure, not the 5-min vol that AC actually cares about

Generalized rule: **frequency-mismatched features dilute regime separation** regardless of signal quality. A 5-min on-chain whale-alert proxy or 5-min Twitter volume might survive — out of scope for this project.

---

## References & cross-links

**Source notes preserved**:
- `research/archive/regime_hmm.md` (Mar 22, methodology spec)
- `research/archive/qa_part_e_regime_arch.md` (Mar 26, Q13–Q18 implementation Q&A)
- `research/archive/hmm_state_count_audit_apr26.md` (Apr 26, BIC unit fix + ablation)

**Canonical scripts**:
- `scripts/compare_2state_vs_3state_hmm.py` (BIC ablation, post-fix; commit `cef075a`)
- `scripts/compare_hmm_vol_feature.py` (vol-feature ablation; commit `e930118`)
- `scripts/cache_btc_5min_returns.py` (returns cache for fast reruns)
- `scripts/paired_test_regime_aware_v5.py` (current paired test)
- `extensions/regime.py` (regime-conditional ACParams)
- `montecarlo/sde_engine.py::simulate_regime_execution` (rule + pde modes)

**Canonical data**:
- `data/hmm_2state_vs_3state.json` (post-fix BIC table)
- `data/hmm_vol_feature_comparison.json` (vol-feature 3-state separation)
- `data/btc_5min_log_returns_2026-01-01_to_2026-04-08.npy` (cached returns)
- `data/paired_regime_v5_true_params.json` (V5 paired-test result)

**Cited in**:
- `FINDINGS.md` §1.5, §2.2, §5.3, §5.4 (regime-conditional impact, V1–V5 progression, multi-feature rejection)
- `PRESENTATION_DRAFT.md` slide 7 (V1–V5 learning narrative), slide 8 (bivariate HMM rejection)

**Foundational references**:
- Almgren & Chriss (2001), *Journal of Risk* 3(2)
- Hamilton (1989), *Econometrica* 57(2): Markov-switching regime
- Baum, Petrie, Soules, Weiss (1970): Baum-Welch EM
- Cont, Kukanov & Stoikov (2014), *Journal of Financial Econometrics* 12(1): OFI as price-impact predictor
- Ang & Bekaert (2002), *JBES* 20(2): 2-state regime evidence in financial markets
- Liu (2015), *JEDC* 53: 2-state HMM on equity vol
- Makarov & Schoar (2020), *JFE* 135(2): BTC microstructure
- Weiss et al. (2023), *JOSS*: hmmlearn library
- Killick, Fearnhead & Eckley (2012), *JASA* 107(500): PELT changepoint
- Rabiner (1989), *Proc. IEEE* 77(2): comprehensive HMM tutorial
- Nguyen & Tran (SIAM CT 2021): execution shortfall under regime switching
- Kass & Raftery (1995): BIC interpretation thresholds
