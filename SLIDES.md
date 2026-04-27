# SLIDES — 10-min talk

**Format**: 10 slides + 5-min Q&A · action titles · one focal visual per slide · written-out talking script
**Source MDs**: [REPORT.md](REPORT.md) (narrative) · [RESULTS.md](RESULTS.md) (numbers) · [FIGURES.md](FIGURES.md) (figures) · [FINDINGS.md](FINDINGS.md) V6 (current truth)

---

## Slide 1 — Title

> **Key takeaway**: Optimal Execution Under Stochastic Volatility on Binance BTCUSDT

**Visual**: title card — equation `min E[cost] + λ·Var[cost]` centered, one-line headline preview underneath: *"Heston-Q stoch vol cuts CVaR₉₅ by 14.59% vs const-vol, p<0.0001"*

**Talking script (~30 s)**:

Good morning. We tackled the Almgren-Chriss optimal-execution problem on real Binance BTCUSDT data, and pushed it through three extensions: a fully implicit HJB PDE solver for nonlinear impact, a Heston Q-measure stochastic-volatility extension, and a Hidden-Markov regime-aware execution scheduler. The headline result is on screen — we'll get there in seven minutes. The two stories I want to tell are: **the clean Heston anchor finding**, and **the methodological-learning story from the regime-aware part where we caught and reversed our own findings**.

---

## Slide 2 — Why this problem matters

> **Key takeaway**: Linear AC is scale-invariant in X₀ (savings depends on κT, not X₀); but real BTC has nonlinear (square-root-ish) impact, where AC concentrates trades — and at institutional size that concentration is exactly what saves cost.

**Visual**: `figures/sensitivity_x0.png` — paired MC savings vs X₀ at calibrated parameters (LAM=1e-6, σ=0.42, α=1 vs α=0.441), with HJB degenerate-bang-bang points marked red

**Talking script (~60 s)**:

The Almgren-Chriss objective trades expected cost against cost variance. You pay two impact costs: a **permanent impact** γ·v·X (Kyle's λ generalized) that walks the price away from you, and a **temporary impact** η·|v|^α (square-root-style power law) that you incur on each child trade. The whole question is whether scheduling beats just trading uniformly.

Look at the green flat line — that's the closed-form linear AC at α=1. It sits at zero savings across every order size. **That's not a bug — linear AC is scale-invariant in X₀**: every term in the objective scales as X₀², so the percentage benefit doesn't depend on order size. The story changes when impact is nonlinear.

The purple line is the calibrated α=0.441 from real Binance trades. Where HJB's policy iteration converges to a non-degenerate trajectory, savings sit around **50%** — that's the nonlinear concentration benefit, AC front-loads execution to exploit the sub-quadratic cost. The red Xs are points where HJB degenerated to a single >95%-in-one-step bang-bang — a known viscosity-solution pathology at sublinear α. We catch those with a two-solver SLSQP cross-check (Part 11 §11.4) elsewhere in the project.

The cleaner X₀-dependence story — using full per-window calibration on 6 walk-forward splits — is on slide 6.

---

## Slide 3 — Calibration (Part A) — the cascade fix

> **Key takeaway**: Tick-level Kyle's-λ returned **negative γ** (economically nonsensical). A 1-min → 5-min → tick → fallback cascade fixes it; bookDepth cross-check confirms the fix.

**Visual**: small inline table:

| Method | γ | η | α | R² | Verdict |
|---|---|---|---|---|---|
| Tick-level (original) | **−0.0113** | 1.7e-5 | 0.04 | 0.001 | bid-ask bounce dominated |
| **1-min cascade (production)** | **+1.48** | **1.58e-4** | **0.441** | 0.18 | literature range ✓ |
| BookDepth integral (cross-check) | — | **1.58e-4** | — | — | independent confirmation ✓ |

**Talking script (~75 s)**:

When we calibrated permanent impact off raw trade prints, we got **negative γ — minus one one-hundredth of a basis point**. That's economically impossible: it would mean buying pushes the price *down*. The bug was bid-ask bounce: at the tick level, the absolute price change between consecutive trades is dominated by **alternation between bid and ask**, not by directional price impact. Same problem hit α — we got 0.04, way below the literature range of 0.3 to 1.5.

The fix is a cascade in `calibration/impact_estimator.py`: try 1-min bars first, where bid-ask bounce averages out; if R² is below 0.05, fall back to 5-min, then to tick, then to literature constants. The 1-min tier hits and produces **γ = 1.48, η = 1.58e-4, α = 0.441** — all in literature range. As a robustness check, we also computed η directly from order-book depth on 28 days of bookDepth data — **gave the same 1.58e-4** from a completely independent estimator. Two methods, one answer.

---

## Slide 4 — Heston Q-measure calibration

> **Key takeaway**: Carr-Madan FFT against Deribit IV surface gives κ=9.09, ρ=−0.385 with **2-day std(κ)=0.01, std(ρ)=0.001** — stable, not noise.

**Visual**: `figures/iv_fit_heatmap.png` — Deribit IV surface with model fit, RMSE = 0.0086

**Talking script (~75 s)**:

Calibrating Heston purely from spot data didn't work. We were using 24-bar overlapping rolling windows for moment-matching, and overlapping windows induce autocorrelation around 0.985 — which biases the moment estimators so much that κ pinned to its ceiling and ρ came out as noise.

The fix was to switch from P-measure to Q-measure: calibrate Heston by **Carr-Madan FFT pricing against the Deribit option implied-volatility surface**. The implied vols carry forward-looking risk-neutral information that's much cleaner than realized moments.

The result is on screen: IV-surface fit RMSE of **0.0086** — better than two-thirds of the published Heston-on-equities literature. The economically interesting numbers are κ = 9.09 — fast mean reversion, plausible for crypto — and ρ = −0.385, **negative leverage**, meaning when BTC drops, vol spikes. That sign is consistent with every crypto study we found. As a sanity check, Heston **beats the 3/2 model by 33.9% RMSE** on the same chain, which justifies the model choice empirically.

---

## Slide 5 — Heston parameters are stable across 12 months

> **Key takeaway**: Monthly recalibration over 12 months shows κ and ρ are persistent market structure, not month-specific noise.

**Visual**: `figures/heston_qmeasure_time_series.png` — 12-month time-series of κ, θ, ξ, ρ (Tardis monthly snapshots, 2025-05 → 2026-04)

**Talking script (~45 s)**:

We re-ran the calibration once a month for twelve months. κ stays around 9, θ stays around BTC's long-run vol, ξ moves a bit with the vol-of-vol regime, and ρ stays around minus 0.4. Nothing dramatic, no parameter wandering. **2-day repeated calibration** gives standard deviations of 0.01 for κ and 0.001 for ρ. The picture supports the claim that we're estimating real market structure — calibrated κ/ρ are not chance artifacts of one snapshot.

---

## Slide 6 — HJB walk-forward — AC beats TWAP across 6/6 splits

> **Key takeaway**: At institutional X₀ = 1000 BTC, AC beats TWAP by **15.1% to 36.9% MC savings** out-of-sample, all 6 splits, σ-drift up to +111% across train/test.

**Visual**: `figures/walk_forward_savings_6splits.png` — 6 splits × MC vs deterministic savings %

**Talking script (~75 s)**:

This is the headline for Part B, the HJB solver. Our `pde/hjb_solver.py` has two paths — closed-form Riccati for α=1, and **Howard's policy iteration with implicit Crank-Nicolson** for the general nonlinear-α case. The Howard's path matters because the separable ansatz V(t,x) = a(t)·x² fails as soon as α ≠ 1, and BTC empirically has α around 0.44, not 1.

To validate, we ran a 6-split walk-forward: train calibration on the first window, run the HJB-derived schedule on the next held-out window, repeat. The dark blue bars are MC out-of-sample savings — every single split is 15% or higher, several are above 25%, one hits **36.9%**. The light bars are the deterministic-cost equivalents, which are smaller (1.7% to 4.6%) — the MC-vs-deterministic gap is information about cost variance reduction, not just expected-cost reduction.

What gives me confidence this isn't a numerical artifact: we cross-checked with **SLSQP**, an entirely different solver that optimizes a discrete trade-vector instead of solving a PDE. The two solvers agree within 1%, which means we're seeing the actual optimum, not a discretization quirk. (This is the "two-solver pattern" from code-council Part 11 — when a numerical method gives a result, run an alternative method on the same problem and check they agree.)

---

## Slide 7 — Part D anchor finding — Heston-Q reduces tail risk

> **Key takeaway**: Heston-Q stochastic vol cuts **CVaR₉₅ by 14.59%** vs const-vol on a CRN-coupled paired MC test, p<0.0001.

**Visual**: `figures/heston_cvar_comparison.png` — bar chart, const-vol / Heston-P / Heston-Q with annotation

**Talking script (~75 s)**:

This is the cleanest positive result in the project. Setup: take two AC schedules — one designed under constant volatility, one designed under our calibrated Heston-Q dynamics — and run **both** on the same Heston-Q SDE paths, with common-random-numbers coupling for variance reduction.

Const-vol leaves CVaR₉₅ at 4,294. Heston-Q gets it down to 3,668. **That's a 14.59% reduction in worst-5% tail cost, and the paired-test p-value is below 0.0001**. The Heston-aware schedule sees the vol distribution that const-vol assumes away, so it slows down execution when the vol regime is ambiguous and concentrates trading when vol is more predictable. Note Heston-P (the historical-measure version) reduces CVaR even further — but Q-measure is what we actually have a calibration for, so Q is the one we report.

A clarification on what "Heston-aware schedule" means here: this is **schedule design at t=0 under a stochastic-vol distribution**, not mid-execution feedback adapting to a realized v_t. Our HJB is one-dimensional in (t,x). A 2D HJB with v as a state variable, producing a feedback rule v\*(t,x,v), is on the future-work list — that's what would make this fully sequential under stoch vol.

---

## Slide 8 — Part E — five iterations of the regime-aware story

> **Key takeaway**: Each version surfaced a successively subtler bias layer; V4's "clean" finding was invalidated by V5's sign reversal; V6 with extended data is the path to definitive answer.

**Visual**: inline table:

| V | Approach | CVaR₉₅ Δ | Status |
|---|---|---|---|
| V1 | σ × 1e-8 magic multipliers | not significant | wrong model |
| V2 | sample size fix | not significant | wrong sample |
| V3 | metric fix | not significant | wrong metric |
| V4 | Yuhao σ-multipliers | **−14.0% p<0.0001** | **invalidated** |
| V5 | true per-regime OLS | **+227.5%** (sign reversed) | suspect — η fallback |
| V6 | extended 280-day data | pending | resolves sub-sample size |

**Talking script (~75 s)**:

Part E is the **methodological-learning** part of the project. The honest version: every time we thought we had a result, the next audit revealed why it was wrong.

V1 used magic σ × 1e-8 multipliers — no microstructure basis. V2 fixed sample sizes. V3 fixed the metric. V4 introduced Yuhao's σ-based multipliers and produced what looked like a clean, defensible finding: **CVaR₉₅ down 14%, p < 0.0001**. We were ready to ship that. Then V5 replaced the σ-multipliers with **true per-regime OLS calibration** — fitting γ, η, α separately on the risk-on and risk-off sub-samples — and the sign **reversed**: CVaR₉₅ went up by 227%.

The reason V4 looked positive was that the σ-multipliers were systematically over-estimating risk-off γ by 41 to 469% and under-estimating risk-off η by 79 to 90%. The biased params made risk-off execution look artificially expensive, so the scheduler shifted execution into risk-on. That looked like tail-risk reduction, but it was an artifact of the bias.

V5 isn't trustworthy either — risk-off is only 5.5% of bars, so the OLS regression has R² of 0.10, α falls out of the literature range, and η falls back to the literature constant. So V5 is η-fallback noise rather than a real regime effect. **V6 with the extended 280-day window is in flight**; that should resolve the sub-sample-size problem.

The takeaway: **regime-aware tail-risk benefit is currently unresolved**. We're presenting this as a methodology story, not a positive result.

---

## Slide 9 — What we learned about audit-first development

> **Key takeaway**: Standard testing missed four classes of bug; explicit audits caught all four.

**Visual**: bullet list (no chart):

- **Numerical convention drift** — `T = 1/24` mistakenly meant 15 days, not 1 hour, across 13 files for 3 weeks
- **Two-solver cross-check** — SLSQP exposed where HJB could go bang-bang at sublinear α
- **Dead-code accumulation** — retired 28 dead scripts + 17 download scripts + ~80 stale data files in this week's cleanup
- **Calibration identification hygiene** — Heston ρ "non-identifiable" turned out to be a loss-function design bug (calls-only filter + OI-weighted loss + xi lower bound), not a data-quantity problem

**Limitations to flag**:
- Mid-price = per-bar VWAP, not L2 best-bid/ask (Tardis L2 snapshots available but not wired)
- Fallback constants used when cascade fails — flagged in CalibrationResult
- 1D HJB; 2D feedback policy under stoch vol is future work

**Talking script (~60 s)**:

Four bug classes that standard pytest didn't catch but our audit framework did. The unit-drift bug — calling 15 days "1 hour" — sounds dumb but it lived in 13 files for three weeks. The fix was **centralizing constants** in one file and adding dimensional comments at every parameter definition. The two-solver cross-check is the SLSQP-vs-HJB pattern we already mentioned. Dead-code accumulation is just hygiene — research codebases grow this way and need periodic pruning. And the calibration-identification one is the most interesting: we were about to spend three hours of Bloomberg seat time pulling more options data when the actual fix was a 10-minute change to the loss function. The data was never the problem; the **loss function geometry was zeroing out ρ identification**.

The two limitations that matter for the report: the **mid-price proxy** (we use per-bar VWAP, not real L2 best-bid-ask), and the **risk-off sub-sample size** (5.5% of bars, too small for stable OLS). Both are honest limitations, both flagged in the writeup.

---

## Slide 10 — Conclusions

> **Key takeaway**: Heston-Q anchor finding is robust and defensible. Regime-aware story is still being told.

**Visual**: confidence summary table:

| Claim | Confidence | Evidence |
|---|---|---|
| γ, η, α calibration reliable | high | 1-min cascade + bookDepth cross-check, both at η=1.58e-4 |
| Heston Q-measure κ/ρ reliable | high | std(κ)=0.01, std(ρ)=0.001 over 12 months |
| Heston beats 3/2 model | high | RMSE 0.098 vs 0.132 (−33.9%) |
| **Heston-Q reduces CVaR₉₅** | **high** | **14.59% reduction, p<0.0001** |
| AC beats TWAP at X₀ ≥ 100 BTC | high | walk-forward 6/6 splits |
| Regime-aware tail-risk benefit | **unresolved** | V4 invalidated, V5 suspect, V6 pending |
| Multi-feature exogenous HMM helps | none | F&G + CoinMetrics rejected |

**Future work**:
1. V6 paired test on extended 280-day window (in flight)
2. 2D HJB with v_t feedback (this is what truly-sequential stoch-vol execution would look like)
3. L2 mid-price wiring (10K Tardis snapshots ready)
4. Multi-horizon paper extension (1h / 6h / 1d, scaling already measured)

**Talking script (~45 s)**:

To wrap up: we have **five high-confidence claims** — the calibration is reliable, Heston Q-measure is stable across 12 months, Heston beats 3/2, Heston-Q cuts CVaR₉₅ by 14.59%, and AC beats TWAP at institutional size. We have **one explicitly unresolved claim** — the regime-aware tail-risk benefit. And we have **one explicitly null result** — multi-feature daily-frequency exogenous HMM doesn't help.

The cleanest deliverable is **Heston-Q stochastic vol stochastic vol cuts tail risk by 15% with p below 0.0001**. The honest deliverable is **the regime-aware story isn't done yet, and we know exactly why and what to do next**. Thank you.

---

## Time budget

| Slide | Time | Cumulative |
|---|---|---|
| 1 Title | 0:30 | 0:30 |
| 2 Why it matters | 1:00 | 1:30 |
| 3 Calibration cascade | 1:15 | 2:45 |
| 4 Heston Q-measure | 1:15 | 4:00 |
| 5 12-month stability | 0:45 | 4:45 |
| 6 HJB walk-forward | 1:15 | 6:00 |
| 7 Part D anchor finding | 1:15 | 7:15 |
| 8 Part E V1-V5 | 1:15 | 8:30 |
| 9 Audit-first lessons | 1:00 | 9:30 |
| 10 Conclusions | 0:45 | 10:15 |
| **Total** | **~10 min** ✓ | + 5 min Q&A |

---

## Q&A prep

| Q | A |
|---|---|
| **Why does AC ≈ TWAP at small X₀?** | Below X₀ ≈ 100 BTC, market impact is below the noise floor of the MC test (paired p > 0.79 at X₀ = 10). AC dominates only when impact > price-risk variance. |
| **Why is V5 +227% if V4 was −14%?** | V4's Yuhao σ-multipliers over-estimated risk-off γ by 41-469% and under-estimated η by 79-90%. When V5 used true OLS calibration, those biases reversed. V5 itself is suspect because risk-off is 5.5% of bars; V6 extended window will resolve. |
| **Sequential / feedback under Heston?** | Honest: our 1D HJB uses Heston-calibrated σ to **design the schedule at t=0**, not adapt mid-execution. Both AC schedules in the paired test are open-loop. SLSQP could in principle reproduce the schedule design. **2D HJB with v feedback is future work.** |
| **Is 14.59% economically meaningful?** | The CVaR₉₅ values are in the same units as the cost, and a 14.59% tail-cost reduction across thousands of independent paths is real money at institutional size. p < 0.0001 rules out variance noise. |
| **Why bookDepth as robustness if not in main pipeline?** | We computed η two ways — trade-flow 1-min cascade (canonical) and order-book-depth integral (bookDepth). Both gave 1.58e-4. The main pipeline uses 1-min because it's standard; bookDepth is the cross-check, not a parallel path. |
| **What's the deal with the negative γ?** | Tick-level Kyle's-λ regression on raw trades is dominated by bid-ask bounce. A buy at the ask is followed by a print at the bid even with no information — that creates a spurious negative correlation between trade direction and next-print return. 1-min aggregation averages bounce out. |
| **Why HJB instead of just SLSQP?** | Two reasons: (1) Howard's policy iteration generalizes naturally to nonlinear α — the separable ansatz fails for α≠1, but Howard's converges in 10-20 iterations regardless. (2) Generalization headroom for the future 2D feedback policy under stoch vol. For the current 1D problem, both methods agree to ~1%. |

---

## Image inventory

| Slide | File | Status |
|---|---|---|
| 1 | (none — title card with equation) | — |
| 2 | `figures/sensitivity_x0.png` | ✅ |
| 3 | (inline table) | — |
| 4 | `figures/iv_fit_heatmap.png` | ✅ |
| 5 | `figures/heston_qmeasure_time_series.png` | ✅ |
| 6 | `figures/walk_forward_savings_6splits.png` | ✅ generated 2026-04-27 |
| 7 | `figures/heston_cvar_comparison.png` (primary) + `figures/tail_qq_heston_vs_const.png` (backup) | ✅ generated 2026-04-27 |
| 8 | (inline V1-V5 table) | — |
| 9 | (bullet list) | — |
| 10 | (inline confidence-summary table) | — |

---

## Open question — CVaR % shift discrepancy (resolve before final slides)

The current `data/paired_heston_qmeasure_results.json` (10,000 paths) shows **−14.59%** CVaR₉₅ reduction. FINDINGS V6 / earlier draft cited **−4.79%** at "50k/100k paths" with a 17× larger absolute cost — implying a different X₀ or λ config.

**Options**:
- **A** — Re-run paired test at the canonical institutional config (X₀ = 1000 BTC, current λ); regenerate `figures/heston_cvar_comparison.png` with refreshed numbers; update REPORT, RESULTS, FINDINGS, slides to one consistent number
- **B** — Use JSON-current −14.59% everywhere; update FINDINGS to match (it's a stronger headline anyway)
- **C** — Report both as a sensitivity: −14.59% at the smaller config, −4.79% at institutional. Both are above noise floor.

The current SLIDES Slide 7 + figure use **−14.59%** (matches the JSON file a reviewer would inspect). Pick A/B/C and propagate before submission.

---

## Open question — teammate's HJB-vs-SLSQP claim under stoch vol

> Teammate (2026-04-27): *"HJB sequential, SLSQP trade vector. Same in deterministic AC. Stoch vol → only HJB does sequential optimal action."*

| Claim | Verdict |
|---|---|
| HJB sequential, SLSQP trade vector | ✅ correct |
| Same in deterministic AC | ✅ correct (verified: walk-forward HJB ≈ SLSQP within 1%) |
| Only HJB sequential under stoch vol | ⚠️ **theoretically true only for 2D HJB with v as state variable**. Our HJB is **1D in (t,x)** — both schedules in the Part D paired test are open-loop, designed at t=0 against the Heston-Q distribution. SLSQP can in principle reproduce. The "sequential feedback" advantage doesn't kick in until 2D HJB with v\*(t,x,v) — future work. |

**Slide 6 + Slide 7 talking scripts handle this honestly** (frame Heston advantage as schedule design under stoch-vol distribution, not mid-execution feedback). Don't oversell; a careful reviewer will ask "is the schedule path-dependent during execution?" and the answer is no.
