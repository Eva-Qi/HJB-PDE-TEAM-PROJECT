# HMM State-Count Audit — Bug Fix + Vol-Feature Ablation

**Course:** MF796 — Computational Methods in Finance
**Topic:** Part E — Regime detection, 2-state vs 3-state comparison
**Date:** 2026-04-26
**Status:** Bug fixed, ablation complete, results pushed to `main`

---

## TL;DR

Original `data/hmm_2state_vs_3state.json` reported **3-state preferred by ΔBIC = −4,138,934**. That number was an artifact of a unit error — `hmmlearn.GaussianHMM.score(X)` already returns the total log-likelihood of the sequence, but the audit script multiplied it by `T = 28,223` (treating it as a per-observation average). After fix:

| Feature | ΔBIC (3-state − 2-state) | Recommendation | 3-state non-degenerate? |
|---|---|---|---|
| Raw 5-min log returns | **+6.0** | **2-state** | No — 14/15 fits collapse |
| log(rolling 30-min std) | **−12,927** | **3-state** | Yes — three clean regimes |

Two-line summary for the report:
> "BTC volatility is trimodal (low/medium/high), but raw 5-min returns are too noisy to identify the medium regime — 14/15 of 3-state HMM restarts collapse on returns. On the explicit vol feature, the three regimes separate cleanly with ΔBIC = −12,927."

---

## 1. The Bug

**File:** `scripts/compare_2state_vs_3state_hmm.py:106` (pre-fix)

```python
log_likelihood = float(best_model.score(X)) * T  # WRONG — score already total
```

`hmmlearn.GaussianHMM.score(X)` runs the forward algorithm and returns
`log P(X | model)` — a single scalar that is the joint log-likelihood
of the entire sequence. Multiplying by `T = 28,223` inflates BIC values
by the same factor.

### Diagnostic smoke signal that should have caught this earlier

Reported log-likelihood: **4.222 × 10⁹** (4 billion). For 28k Gaussian
observations with std ≈ 1.66e-3, the per-observation log-likelihood is
roughly `log(1/(σ·√(2π))) ≈ 6.4`, so total log-lik should be ~150k. A
log-lik of 4 billion is physically impossible for this data size.

Lesson: when reporting log-likelihood values, sanity-check the order of
magnitude against `T · ln(1/σ)` — anything > 10× that is suspicious.

---

## 2. After the Fix — Raw Returns

```
Model         n_params   log-lik           BIC
2-state              7    146,071     −292,070
3-state             14    146,104     −292,064
                                       ──────
                          ΔBIC (3−2) = +6.0  → 2-state preferred
```

**Restart audit (15 random initializations each):**

| | Converged | Rejected: occupancy <1% | Rejected: σ-ratio <1.15 | Min σ ratio observed |
|---|---|---|---|---|
| 2-state | 15/15 | 0 | 0 | 3.263 |
| 3-state | 15/15 | **11** | **3** | **1.005** |

For 3-state on returns, **14 out of 15** restarts collapse to either a phantom
state (one regime with <1% occupancy) or duplicate states (two regimes with
σ ratio < 1.15). The single surviving fit barely beats 2-state by ΔBIC = +6,
which by Kass-Raftery (1995) thresholds is "weak" evidence at best.

**Observed σ ratio of 1.005** means two of the three states have
emission σ within 0.5% of each other — they are mathematically the same
state. EM is duplicating risk_on rather than discovering a third regime.

---

## 3. After the Fix — Log-Vol Feature

Replaced raw returns with `log_vol_t = log(rolling_std(returns, window=6))`,
where the 6-bar window is 30 minutes. Same `_fit_and_score` plumbing,
σ-ratio filter disabled (see §5).

```
Model         n_params   log-lik           BIC
2-state              7    −21,374       42,820
3-state             14    −14,875       29,893
                                        ─────
                          ΔBIC (3−2) = −12,927  → 3-state preferred
```

**3-state regimes on log_vol:**

| Label | σ_state | Occupancy | Mean log_vol |
|---|---|---|---|
| risk_on | 0.254 | 43.4% | low (lowest mean) |
| neutral | 0.377 | 27.3% | mid |
| risk_off | 0.428 | 29.3% | high |

All three regimes substantial, σ monotonically increasing, no phantom or
duplicate state. ΔBIC = −12,927 is "decisive" by Kass-Raftery thresholds
(>10 = strong, this is 1300× past that).

**Interpretation:** BTC volatility itself is trimodal. Raw returns can't
identify the third regime because returns are zero-mean — Gaussian HMM
is forced to use emission σ as the only separator, and σ_low ≈ σ_med
on noisy 5-min data. log_vol gives the HMM emission means to separate
on, not just variances.

---

## 4. Two New Filters (Code Changes)

### 4.1 Removed bug

`scripts/compare_2state_vs_3state_hmm.py:106`:
```python
# was: log_likelihood = float(best_model.score(X)) * T
log_likelihood = float(best_model.score(X))  # already total log-lik
```

### 4.2 Added duplicate-state σ-ratio filter

`scripts/compare_2state_vs_3state_hmm.py:_fit_and_score`:

```python
# Reject fit if any two states have σ ratio below threshold
state_sigmas = np.sqrt(model.covars_.reshape(K, -1)[:, 0])
sorted_sigmas = np.sort(state_sigmas)
if K > 1:
    ratios = sorted_sigmas[1:] / np.maximum(sorted_sigmas[:-1], 1e-12)
    if np.any(ratios < duplicate_sigma_ratio_threshold):  # default 1.15
        continue
```

Catches the actual collapse mode on BTCUSDT 5-min returns: phantom-occupancy
filter alone misses fits where two states have ~equal σ but reasonable
occupancy. Without this filter, the script ranks duplicate-state 3-state
fits as "valid" because each duplicate state passes occupancy individually.

### 4.3 Per-restart audit fields

New keys in `data/hmm_2state_vs_3state.json`:
- `n_converged`, `n_rejected_occupancy`, `n_rejected_duplicate`
- `min_sigma_ratio_observed` (across all converged fits)
- `best_model_min_state_frac`, `best_model_min_sigma_ratio`
- `rejected_no_valid_fit` (true if no restart passes both filters)

These let a reader audit *why* a model was preferred without re-running
the script.

---

## 5. Methodological Note — Filter Is Feature-Specific

The σ-ratio duplicate-state filter is correct for **zero-mean features**
where regime separation must come from emission σ (raw returns).

For **mean-separated features** like log_vol, regimes can have similar
emission σ but well-separated emission μ — applying a σ-ratio filter
incorrectly rejects valid fits.

**Symptom we hit during this audit:** running the unmodified
`_fit_and_score` on log_vol rejected all 15 of the 2-state fits because
σ_state_on ≈ σ_state_off (~0.46 each), even though their means were
clearly separated.

**Fix in `scripts/compare_hmm_vol_feature.py`:**
```python
result = _fit_and_score(log_vol, n_regimes=K, duplicate_sigma_ratio_threshold=0.0)
```

**General principle:** the filter should match how regimes separate in
the feature space. A more general "emission collapse" check would test
whether two states have *both* μ within X% AND σ within Y% — but for
this audit, feature-specific thresholds are sufficient.

---

## 6. Performance Note — Cached Returns

`load_trades(directory, start, end)` concatenates *all* matching CSVs in
the directory before filtering by date. With 111 daily CSVs at ~50MB
each, the global concat hit ~5GB of pandas DataFrame state and induced
swap thrashing on the dev machine — the script appeared to run for 7+
minutes before being killed without ever reaching `model.fit`.

**Fix:** `scripts/cache_btc_5min_returns.py` loads each CSV one at a
time, immediately resamples to 5-min mid-prices (~288 rows/day), and
concatenates only the small series. Peak RAM ~50MB, runtime ~30s, output
saved to `data/btc_5min_log_returns_<start>_to_<end>.npy`.

All HMM ablation runs after the first now load from this cache (~0.1s)
rather than re-loading aggTrades.

---

## 7. Recommendation for the Report

**Default spec:** 2-state HMM on raw 5-min log returns.

- All existing regime-aware execution work (V4, V5 walk-forward, paired
  tests) uses 2-state on returns.
- Validated as non-degenerate (σ ratio 3.26, no rejections).
- Bimodal regime story (risk_on / risk_off) is intuitive and consistent
  with Almgren-Chriss-style execution narrative.

**Ablation panel:** Show that 3-state on returns *does not* work and that
3-state on log_vol *does*, but stick with 2-state on returns for the main
spec to keep the regime-aware execution story coherent.

**Suggested table for the slides / report:**

| Spec | ΔBIC vs 2-state on returns | Decision |
|---|---|---|
| 2-state on raw returns | 0 (baseline) | Main spec |
| 3-state on raw returns | +6 | Reject — 14/15 collapse |
| 2-state on log_vol | n/a (different feature) | Reject — same info as raw |
| 3-state on log_vol | n/a | Defer — promising but changes upstream regime-conditional pipeline |

---

## 8. Files Changed

```
scripts/compare_2state_vs_3state_hmm.py    M   BIC fix + duplicate-σ filter
scripts/cache_btc_5min_returns.py          A   Daily-incremental returns cache
scripts/compare_hmm_vol_feature.py         A   Vol-feature 2 vs 3 state
data/hmm_2state_vs_3state.json             M   Regenerated with correct BIC
data/hmm_vol_feature_comparison.json       A   Vol-feature ablation results
data/btc_5min_log_returns_<window>.npy     A   Cached returns for fast reruns
```

Commits: `cef075a` (BIC fix + new scripts), `e930118` (vol-feature σ filter
disable). Pushed to `main`.

---

## 9. For Teammate Sync

If teammate reports "2-state HMM doesn't work, 3-state works," likely
explanations:
1. **Different `random_state`s** — out of 15 default seeds, some land on
   the rare non-collapsed 3-state fit.
2. **No duplicate-σ filter** — without it, phantom-σ fits look valid.
3. **No occupancy filter** — without it, <1% phantom states aren't caught.

**Sync action:** point teammate at this doc + the regenerated JSON. Have
them run `scripts/compare_2state_vs_3state_hmm.py` after `git pull` and
diff their `data/hmm_2state_vs_3state.json` against the version on `main`.
If the diffs are non-trivial, investigate config (data window, hmmlearn
version, random_state pool).
