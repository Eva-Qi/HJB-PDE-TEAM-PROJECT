# SLIDES — 10-min talk outline

**Status**: Draft (2026-04-27). Use as scaffold; refine after teammate sync.
**Format**: 10 slides + 5-min Q&A · action titles · one key image per slide
**Source MDs**: `REPORT.md` (narrative), `RESULTS.md` (numbers), `FIGURES.md` (figure inventory), `FINDINGS.md` V6 (current truth)

---

## Open issues to resolve before presenting

### Issue 1 — Teammate's claim about HJB vs SLSQP (2026-04-27)

> "HJB solves the optimal allocation problem in sequential steps while SLSQP does it using a trade vector. Both have the same solution in a deterministic AC. When we introduce stochastic vol using HMM or Heston we need sequential optimal action to take place which only HJB does."

**Verdict**: partial. The first two sentences are correct. The third sentence is **theoretically true but does not match what we actually implemented**:

| Claim | Truth-status | Note |
|---|---|---|
| HJB sequential / SLSQP trade vector | ✅ correct | DPP backward induction vs open-loop optimization |
| Same in deterministic AC | ✅ correct | Verified: `data/walk_forward_results.json` (HJB) vs `data/walk_forward_results_slsqp.json` (SLSQP) agree within ~1% |
| "Only HJB does sequential under stoch vol" | ⚠️ misstates our implementation | True only for **2D HJB with state (t, x, v)** producing feedback policy v\*(t,x,v). Our `pde/hjb_solver.py` is **1D** (state = t, x). The Heston-Q advantage in our paired test comes from **schedule design under stochastic-vol distribution at t=0**, not from mid-execution v_t feedback. Both schedules in the paired test are open-loop (precomputed at t=0). SLSQP can in principle do the same thing under a Heston-distributed objective. |

**Implication for slides**: do not oversell "sequential feedback under stochastic vol" — a careful reviewer will ask "is your Heston-aware schedule path-dependent during execution?" Honest framing: "Heston-Q schedule is designed against the calibrated stoch-vol distribution at t=0; both solvers (HJB + SLSQP) could implement this open-loop schedule; HJB is our chosen primary path because Howard's policy iteration generalizes naturally to α≠1 and to a future 2D feedback extension." This sidesteps the overclaim while keeping HJB as the headline solver.

### Issue 2 — Walk-forward bar chart figure ✅ GENERATED 2026-04-27
Generated as `figures/walk_forward_savings_6splits.png` via `scripts/plot_slide_figures.py`. Shows 6/6 splits with MC savings 15.1% → 36.9% (matches REPORT §3 "+15-37%" claim verbatim). Used on Slide 6.

### Issue 3 — Part D CVaR comparison bar chart ✅ GENERATED 2026-04-27
Generated as `figures/heston_cvar_comparison.png` via the same script. Shows const-vol vs Heston-P vs Heston-Q with prominent 14.59% reduction annotation. Used on Slide 7.

### Issue 4 — ⚠️ CVaR % shift discrepancy (REQUIRES TEAM DECISION)
The current `data/paired_heston_qmeasure_results.json` (10,000 paths, σ_base=0.478) shows:
- const-vol CVaR₉₅ = **4,294**, Heston-Q CVaR₉₅ = **3,668**, **Δ = −14.59%**, p < 0.0001

FINDINGS V6 / REPORT / earlier SLIDES draft cites:
- const-vol CVaR₉₅ = **74,222**, Heston-Q CVaR₉₅ = **71,016**, **Δ = −4.79%**, "50k paths, robust at 100k paths"

The 17× absolute-cost gap implies these are **two different runs at different X₀ or λ** (smaller X₀ → smaller absolute cost; relative Δ% can differ a lot too). Both are valid runs of the same paired-test methodology.

**Decision needed before final slides**:
- Option A: stick with FINDINGS-cited 4.79% (representative of institutional X₀=1000 BTC). Re-run paired test to refresh `data/paired_heston_qmeasure_results.json` at the institutional config. Regenerate figure.
- Option B: use the JSON-current 14.59% (it's stronger anyway, and the reviewer will see the JSON file). Update REPORT/RESULTS/FINDINGS to match.
- Option C: report both — 4.79% at X₀=1000 (institutional anchor) and 14.59% at the smaller config (sensitivity check).

I currently plotted **Option B** numbers (the JSON values verbatim) so the figure matches `data/`. SLIDES Slide 7 below needs to be updated according to whichever option you pick.

---

## Slide 1 — Title

**Action title**: Optimal Execution Under Stochastic Volatility on Binance BTCUSDT

- Course / date / team (Eva-Qi, bp, Yuhao)
- Headline preview: "Heston-Q reduces CVaR₉₅ by **4.79%** vs const-vol on 100k MC paths, p<0.0001"
- Repo: github.com/Eva-Qi/HJB-PDE-TEAM-PROJECT

**Image**: none (or single equation: `min E[cost] + λ·Var[cost]`)
**Time**: 30 sec

---

## Slide 2 — Problem setup

**Action title**: Liquidate X₀ BTC over horizon T while balancing impact cost vs price-risk variance

**Image**: `figures/sensitivity_x0.png` — AC vs TWAP cost as X₀ scales (illustrates: small X₀ both equivalent; institutional X₀ AC saves 15-37%)

**Talking points (~60 s)**:
- Almgren-Chriss objective: `min E[cost] + λ·Var[cost]`
  - Permanent impact `γ·v·X` (Kyle's λ generalized)
  - Temporary impact `η·|v|^α` (square-root law generalized to power-law α)
- σ annualized; T in years (project audit caught a unit bug where `T=1/24` mistakenly meant 15 days, not 1 hour)
- Centralized constants: `shared/experiment_config.py` (`T_1H`, `LAM=1e-6`, `N_STEPS=250`, `SEED=42`)
- Two solvers cross-verified (deterministic AC): HJB PDE + SLSQP agree to ~1% — gives confidence in the optimum

---

## Slide 3 — Calibration (Part A): the cascade fix

**Action title**: Tick-level Kyle's-λ returned negative γ; a 1-min → 5-min → tick → fallback cascade fixes it

**Image**: inline table OR `figures/sensitivity_alpha.png` (impact-α sensitivity, since the bug was in α estimation)

**Talking points (~75 s)**:
- **The bug**: per-trade `abs_price_change` is dominated by bid-ask bounce → γ = **−0.0113** (negative permanent impact, economically nonsensical) and α ≈ 0.04 (well below literature [0.3, 1.5])
- **Fix**: `calibration/impact_estimator.py` cascade — tries 1-min bars first (R² > 0.05 + parameter in literature range), falls back to 5-min, tick, then literature constants
- **Production result**: γ = **+1.48** (R²=0.18, 141K buckets, 1-min), η = **1.58e-4**, α = **0.441**
- **Robustness check** (Part 13 audit): bookDepth-derived η on 28 days = 1.58e-4 — two independent estimators agree → robustness confirmed
- Per-regime OLS calibration (V5) replaced Yuhao's σ-multiplier heuristic, which invalidated V4 finding (Slide 8)

---

## Slide 4 — Heston Q-measure calibration

**Action title**: Carr-Madan FFT against Deribit BTC option IV surface gives κ=9.09, ρ=−0.385 with std(κ)=0.01, std(ρ)=0.001

**Image**: `figures/iv_fit_heatmap.png` — Deribit IV surface fit with RMSE=0.0086

**Talking points (~75 s)**:
- Original P-measure calibration (24-bar overlapping rolling windows) was unreliable: autocorrelation 0.985 biased moment-matching → κ pinned to ceiling, ρ noisy
- Fix: Q-measure via options. **Carr-Madan FFT pricing**, calibrated against Deribit live IV surface
- 12-month time-series (Tardis) confirms stability: **2-day std(κ)=0.01, std(ρ)=0.001** → reliable
- Heston **beats 3/2 model by 33.9% RMSE** on the same chain → empirical model justification
- Cross-source: IBIT (US ETF options) vs Deribit (BTC native) consistent — `figures/heston_cross_source_comparison.png`
- ρ identification was a separate bug (Part 13 audit caught calls-only filter + OI-weighted loss + xi=0.01 lower bound collectively zeroing out ρ signal); fixed before this slide's numbers

**Backup image**: `figures/heston_qmeasure_time_series.png` (12-month κ/ρ stability)

---

## Slide 5 — Heston Q-measure time-series stability

**Action title**: 12 months of monthly recalibration shows Heston parameters are stable, not chance

**Image**: `figures/heston_qmeasure_time_series.png` — 12-month κ, θ, ξ, ρ time-series (Tardis monthly snapshots 2025-05 → 2026-04)

**Talking points (~45 s)**:
- κ stable ~9, ρ stable ~−0.4 across all 12 months
- ρ < 0 confirms leverage effect (price down → vol up), consistent with crypto literature
- Validates that κ/ρ reflect persistent market structure, not artifacts of one snapshot

---

## Slide 6 — HJB PDE solver (Part B): walk-forward OOS validation

**Action title**: AC beats TWAP by +15-37% in MC savings across 6/6 walk-forward splits at X₀=1000 BTC, p<0.0001

**Image**: `figures/walk_forward_savings_6splits.png` — 6 splits, MC savings 15.1% → 36.9%, σ-drift +10% to +111%

**Talking points (~75 s)**:
- **Solver**: `pde/hjb_solver.py` — Riccati ODE for α=1, **Howard's policy iteration with implicit Crank-Nicolson** for α≠1
  - Separable ansatz V(t,x)=a(t)x² fails for nonlinear impact (`research/HJB.md` §4)
- **N=250 steps** (centralized; was N=50 before convergence audit)
- **Walk-forward**: 6 splits on 98 days of Binance aggTrades, 70% train / 30% test, MC paired test
- All 6 splits show AC > TWAP at X₀ = 1000, statistically significant
- **Retail boundary**: at X₀ ≤ 10 BTC, AC ≈ TWAP (paired p > 0.79). At X₀ ≥ 100 BTC, AC wins consistently
- **SLSQP cross-check** (`data/walk_forward_results_slsqp.json`) agrees with HJB to ~1% — confirms the schedule is the global optimum, not a numerical artifact (code-council Part 11 §11.4 two-solver pattern). Honest framing for stoch vol: see "open issues" §1 — both solvers can produce the Heston-aware schedule; HJB is our primary because of generalization headroom

---

## Slide 7 — Part D anchor finding: Heston-Q reduces tail risk

**Action title**: Heston-Q stochastic vol cuts CVaR₉₅ by 4.79% vs const-vol on 100k common-random-numbers paths, p<0.0001

**Image**: `figures/heston_cvar_comparison.png` (primary, bar chart) + `figures/tail_qq_heston_vs_const.png` (backup, distribution tails)

**Talking points (~75 s)**:
- Setup: two AC schedules — (a) designed under const-σ, (b) designed under Heston-Q calibrated dynamics. Both run on Heston-Q SDE paths with **CRN coupling** for variance reduction.
- **CVaR₉₅ reduction**: see Open issue §4 — pick the canonical run (4.79% at X₀=1000 from FINDINGS, or 14.59% from current 10k-path JSON) before final slides
- Significance: **p < 0.0001** in both runs
- Honest framing (Open issues §1): improvement comes from schedule design awareness of the stoch-vol distribution, not from mid-execution feedback. SLSQP can in principle reproduce.
- **Why this is the anchor finding**: clean causal chain — calibrated κ/ρ stable → Heston SDE simulation reliable → 100k CRN paired test rules out variance noise → effect size economically meaningful (~5% of tail risk)

---

## Slide 8 — Part E regime-aware: methodological-learning story

**Action title**: Five iterations each surfaced a successively subtler bias layer; V6 with extended data window is in flight

**Image**: V1-V5 bias-layer table (inline) — see `research/HMM.md` §evolution

**Talking points (~75 s)**:
- 2-state HMM (regime: risk-on / risk-off) on 5-min log-returns. After 2026-04-26 BIC-bug fix in `compare_2state_vs_3state_hmm.py` (the `score(X)·T` inflation factor reversed 2-vs-3 preference), **2-state preferred**
- V1-V5 narrative:
  - V1 magic σ × 1e-8 multipliers → wrong model
  - V2 sample size fix → wrong sample
  - V3 metric fix → wrong metric
  - V4 Yuhao's σ-multipliers → CVaR₉₅ **−14% p<0.0001 (LOOKED LIKE WIN)** — **invalidated by V5**
  - V5 true per-regime OLS → CVaR₉₅ **+227.5% (sign reversed!)** — also suspect because risk-off is 5.5% of bars (~1500 trades), forcing η literature fallback
  - V6 with extended 280-day data is in flight (Worker I)
- Honest takeaway: this is the **methodology learning** part of the project. Each fix surfaced the next bias. Multi-feature exogenous HMM (F&G, CoinMetrics FlowIn) tested and rejected (FINDINGS §5.4).

---

## Slide 9 — Methodological learnings + limitations

**Action title**: Audit-first development surfaced 4 classes of bug invisible to standard testing

**Image**: bullet list (no chart needed) — or a small "audit chain" diagram

**Talking points (~60 s)**:
- **Numerical convention drift** (Part 10): T-unit bug (1/24 = 15 days mistakenly labeled "1 hour") lived in 13 files for weeks; fix = centralized `shared/experiment_config.py` + dimensional comments
- **External-eyes / two-solver cross-check** (Part 11): teammate's SLSQP exposed HJB bang-bang risk at sublinear α
- **Dead-code accumulation** (Part 12 §P9-§P10): retired 28 dead scripts + 17 download scripts + ~80 stale data files this week (P1-2 cleanup pass)
- **Calibration identification hygiene** (Part 13): Heston ρ "non-identifiable" because of loss-function design (calls-only + OI-weighted + xi lower bound); fixed without pulling more data
- **Limitations** (must mention):
  - Mid-price = per-bar VWAP, not L2 best-bid/ask (10K Tardis L2 snapshots available, not wired)
  - Risk-off sub-sample 5.5% of bars (R²=0.10, η falls to literature 1e-3) — V6 extended window is the path forward
  - 1D HJB; 2D feedback policy v\*(t,x,v) under stoch vol is future work

---

## Slide 10 — Conclusions + future work

**Action title**: Heston-Q stochastic vol provides a clean tail-risk reduction; regime-aware execution remains an open question

**Image**: confidence summary table (from REPORT.md TL;DR / FINDINGS.md §9) — small inline

**Talking points (~45 s)**:
- **High confidence**: γ/η/α calibration (with bookDepth robustness), Heston Q-measure κ/ρ, Heston beats 3/2, Heston-Q reduces CVaR₉₅, AC beats TWAP at X₀≥100 BTC
- **Unresolved**: regime-aware tail-risk benefit (V4 invalidated, V5 reversed; V6 pending extended data)
- **None**: multi-feature daily-frequency exogenous HMM (F&G, CoinMetrics) — all rejected
- **Future work**:
  1. V6 extended-window paired test (in flight)
  2. 2D HJB with v_t feedback policy under Heston (this is what the teammate's "sequential optimal action" claim would actually need)
  3. L2 mid-price wiring (10K Tardis snapshots already in `data/`)
  4. Multi-horizon paper extension: 1h/6h/1d (multi_horizon scaling already measured, FINDINGS §7)

**Closing line**: "The Heston-Q anchor finding is robust. The regime-aware story is still being told."

---

## Image inventory (resolved against `figures/`)

| Slide | Image | Status |
|---|---|---|
| 1 | none | — |
| 2 | `figures/sensitivity_x0.png` | ✅ exists |
| 3 | `figures/sensitivity_alpha.png` (or table) | ✅ exists |
| 4 | `figures/iv_fit_heatmap.png` | ✅ exists |
| 5 | `figures/heston_qmeasure_time_series.png` | ✅ exists (Apr 25 canonical) |
| 6 | `figures/walk_forward_savings_6splits.png` | ✅ generated 2026-04-27 |
| 7 | `figures/heston_cvar_comparison.png` (primary) + `figures/tail_qq_heston_vs_const.png` (backup) | ✅ generated 2026-04-27 |
| 8 | V1-V5 inline table | — (text) |
| 9 | bullet list / audit-chain diagram | — (text) |
| 10 | confidence-summary inline | — (text) |

**Action item**: generate Slide 6 figure (walk-forward 6-split bar chart). Quick `matplotlib` plot from existing JSON, ~15 lines.

---

## Time budget

| Slide | Time | Cumulative |
|---|---|---|
| 1 Title | 0:30 | 0:30 |
| 2 Problem | 1:00 | 1:30 |
| 3 Calibration | 1:15 | 2:45 |
| 4 Heston calibration | 1:15 | 4:00 |
| 5 Heston time-series | 0:45 | 4:45 |
| 6 HJB walk-forward | 1:15 | 6:00 |
| 7 Part D anchor | 1:15 | 7:15 |
| 8 Part E V1-V5 | 1:15 | 8:30 |
| 9 Limitations | 1:00 | 9:30 |
| 10 Conclusions | 0:45 | 10:15 |
| **Total** | | **~10 min** ✓ |

5 min Q&A buffer.

---

## Q&A prep — likely questions

| Q | A |
|---|---|
| "Why did your AC ≈ TWAP at small X₀?" | At X₀ < 100 BTC, market impact is below noise floor of MC. Both schedules end up effectively similar. AC dominates only when impact > price-risk variance. |
| "Sequential feedback under Heston?" | Honest answer: our 1D HJB uses Heston-calibrated σ to **design** the schedule at t=0, not to **adapt** mid-execution. Both schedules are open-loop. SLSQP could replicate the schedule design. 2D HJB feedback is future work. |
| "Why is V5 +227% if V4 was −14%?" | V4's Yuhao multipliers overstated risk-off γ by 41-469% and understated η by 79-90%. When V5 used true OLS, those biases reversed. V5 is *also* suspect because risk-off has only 5.5% of bars. V6 extended window is the path to definitive answer. |
| "Is 4.79% economically meaningful?" | At institutional X₀=1000 BTC, this corresponds to ~3,200 USD of saved tail risk per execution. p<0.0001 across 100k paths makes it not noise. |
| "Why bookDepth in robustness if not in main pipeline?" | We computed η two ways: trade-flow 1-min cascade (canonical) and order-book depth integral (bookDepth). Both gave 1.58e-4 → independent confirmation. The main pipeline uses 1-min only because cascade is more standard; bookDepth is the cross-check. |
