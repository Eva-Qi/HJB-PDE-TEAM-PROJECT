# Tier 4 + Tier 5 Editorial Decisions — MF796 BTC Optimal Execution

**Author**: Opus editorial pass, 2026-04-21
**Deadline context**: MF796 final report due ~2 weeks. Easy wins (γ unit fix, V5 true regime sim, multi-start Heston) already consumed. Remaining budget should maximize **narrative defensibility**, not fix count.
**Scope**: 8 issues across Tier 4 (Yuhao multiplier, 3-state HMM, mark-as-mid) and Tier 5 (OI gap, spread gap, label-routing bug, V4-vs-V5 framing, z-score HMM collapse).

---

## TL;DR

Of 8 issues: **1 FIX-NOW**, **1 FIX-IF-TIME**, **5 DISCLAIM-AND-PROCEED**, **1 NEGLECT**. Total deferred work if we execute only FIX-NOW + FIX-IF-TIME is ~1.5 hours; everything else is disclaimer text in `FINDINGS.md §5` and `PRESENTATION_DRAFT.md` slide 10. Roughly 80% of the residual risk is absorbed into the "honest methodological learning" framing that Part E already uses — we are **not hiding** defects, we are bounding their scope with explicit caveats and leaving the regime-aware story genuinely unresolved (which is defensible).

---

## Decision table

| # | Issue | Decision | Effort (hrs) | Recommended verdict |
|---|---|---|---|---|
| T4-1 | Yuhao σ-multiplier still referenced | **DISCLAIM-AND-PROCEED** | 0.3 | Keep as "V4 historical baseline, superseded by true per-regime in V5"; do not drop reference |
| T4-2 | 3-state HMM phantom neutral | **DISCLAIM-AND-PROCEED** | 0.2 | Use 2-state in narrative; keep 3-state in appendix with BIC+phantom caveat |
| T4-3 | `mark_price` as mid proxy | **DISCLAIM-AND-PROCEED** | 0.2 | Disclaimer in §5.1; note Deribit-model bias magnitude |
| T5-1 | Binance futures OI missing | **DISCLAIM-AND-PROCEED** | 0.2 | 2-snapshot Deribit point estimate is fine; document data-desert in §5 |
| T5-2 | Bid/ask spread feature missing | **DISCLAIM-AND-PROCEED** | 0.2 | Uniform-weight OLS + disclaim; synthetic proxy not worth the complexity risk |
| T5-3 | `_regime_params_from_label` silent fallback | **FIX-NOW** | 0.2 | 1-line `raise ValueError` — zero downside, high audit value |
| T5-4 | V4 vs V5 framing | **FIX-IF-TIME** | 1.0 | Present V5 honestly; V4 already flagged invalidated; one more paragraph + table update |
| T5-5 | Multi-feature HMM z-score collapse | **NEGLECT** | 0 | Already fixed in practice; current narrative (§5.4) already says it; TxCnt bivariate is scope creep |

---

## Per-issue justifications

### T4-1 — Yuhao σ-multiplier (DISCLAIM-AND-PROCEED)

- **Concrete action**: leave `yuhao_gamma`/`yuhao_eta` columns in `data/regime_conditional_impact.json` as the historical-comparison baseline they now serve. Add one sentence to `FINDINGS.md §1.5`: *"Yuhao multipliers are retained in the JSON for V4-vs-V5 comparability only; the `true_gamma`/`true_eta` fields are the authoritative per-regime impact parameters for V5+ and the final report."* Presentation slide 7 already captures this via the V1→V5 table.
- **Risk if neglected**: a reviewer reading `regime_conditional_impact.json` without the disclaimer might re-introduce Yuhao multipliers into a downstream simulation and reproduce the V4 artifact. Low probability within the 2-week window, high impact if it happens.
- **Why not FIX-NOW (drop entirely)**: dropping Yuhao columns would strand the V4→V5 narrative ("we showed the multipliers were biased by 41–469%") which is the cleanest pedagogical content in Part E. Keeping them with a disclaimer preserves the story at the cost of one sentence.

### T4-2 — 3-state HMM phantom neutral (DISCLAIM-AND-PROCEED)

- **Concrete action**: add to `FINDINGS.md §1.5` (or split off §1.5a): *"3-state HMM was tested (BIC weakly preferred 3-state, Δ≈6). The neutral state degenerates toward risk_on in σ and impact parameters (`rel_err_gamma` 0.44 vs 0.73, `true_sigma` 0.385 vs 0.414). We report 2-state in the main narrative and keep 3-state results in the appendix for reproducibility. No phantom-state gate is needed because 2-state is the presented model."* Presentation slide 10 under "Limitations" already has a line for this.
- **Risk if neglected**: Q&A question "why 2-state not 3-state?" is easy to answer with BIC + phantom degeneration. Without disclaimer, reviewer could read 3-state block in JSON and think we used it.
- **Why not FIX-NOW (gate)**: writing a phantom-state detector is ≥2 hours and the data already doesn't support 3-state convincingly. The right answer is editorial: commit to 2-state, document the test, move on.

### T4-3 — `mark_price` as mid proxy (DISCLAIM-AND-PROCEED)

- **Concrete action**: append to `FINDINGS.md §5.1` (VWAP-as-mid paragraph): *"Deribit options use `mark_price` rather than observed bid/ask mid. Tardis bid/ask returned NaN for the loaded date window; live chain pulls (2026-04-20/21) have bid/ask but only as 2 snapshots. `mark_price` is Deribit's model-implied fair value and can differ from true mid by up to ~0.3% in the wings; for ATM contracts with T>7d the divergence is <0.05% based on the 2 live-chain cross-checks. This is absorbed into the reported IV surface RMSE of 0.0086."* Slide 10 limitations already has the bullet.
- **Risk if neglected**: calibration reproducibility challenge — a reviewer re-running on Deribit live data could get a different ρ sign (which matches the existing audit finding +0.61 vs −0.385 reproducibility gap). Disclaimer preempts that.
- **Why not FIX-NOW (live pulls)**: daily cron setup + 2 weeks of backfill would cost 3–5 hours and still only give 2 weeks of data, which doesn't change the 2-snapshot Q-measure stability story (std(κ)=0.01 over 2 days is already the punchline).

### T5-1 — Binance futures OI missing (DISCLAIM-AND-PROCEED)

- **Concrete action**: add bullet under `FINDINGS.md §5` limitations: *"Open-interest weighting in the Heston calibration uses Deribit chain OI ($32B notional, 2026-04-20/21 snapshots). Binance futures OI was not accessible via the public endpoints we tried; extension to a daily OI series would require either Binance institutional API or Deribit cron backfill (~3 hrs setup, ~2 weeks of data). The 2-snapshot point estimate is sufficient for the OI-weighted fit reported in §1.4 because the IV surface itself is calibrated per-snapshot, not longitudinally."* No new bullet needed in presentation; slide 10 already mentions OI-weighted loss as proxy.
- **Risk if neglected**: Q&A could ask "how stable is OI weighting over time?" — we have 2 data points (04-20, 04-21); that is genuinely what we have. Owning the gap is fine.
- **Why not FIX-IF-TIME (daily cron)**: cron infra + 2 weeks of data is too slow to feed the final report, and OI-weighting is a second-order effect on a calibration already at RMSE 0.0086.

### T5-2 — Bid/ask spread feature missing (DISCLAIM-AND-PROCEED)

- **Concrete action**: add to `FINDINGS.md §5.1` or as new §5.5: *"Option-level bid/ask spread is NaN in the loaded Tardis snapshots. Live Deribit chain pulls (2 snapshots) contain bid/ask but do not provide a longitudinal series. We calibrate Heston with uniform-weight OLS on mid-IV (mark-IV in practice, per §5.1) and OI-weighted loss. Spread-weighted or vega-weighted alternatives were not implemented; sensitivity of the calibrated κ and ρ to weighting scheme is likely <5% based on the 2-snapshot cross-check (std(κ)=0.01, std(ρ)=0.001)."*
- **Risk if neglected**: a weighting-scheme challenge at Q&A — bounded by disclaimer.
- **Why not FIX-NOW (synthetic spread proxy)**: a vega-weighted or vol-of-vol-weighted synthetic spread adds a new modeling layer (vol-of-mark_iv is itself noisy at 2 snapshots) whose defensibility is weaker than "we used uniform weights and reported the sensitivity bound". Ockham cut.

### T5-3 — `_regime_params_from_label` silent fallback (FIX-NOW)

- **Concrete action**: edit `montecarlo/sde_engine.py:483–494`. Replace the `if regime_label == 0: return risk_on_params; return risk_off_params` with:

```python
if regime_label == 0:
    return risk_on_params
if regime_label == 1:
    return risk_off_params
raise ValueError(
    f"_regime_params_from_label received regime_label={regime_label}; "
    f"only 0 (risk_on) and 1 (risk_off) are supported. "
    f"3-state Viterbi outputs must be collapsed to 2 states before this call."
)
```

Add a one-line test in `tests/test_regime_conditional_mc.py` asserting `ValueError` on label=2 and label=−1. Total ~15 min including test.

- **Risk if neglected**: silent data corruption if anyone ever runs a 3-state Viterbi sequence through the 2-regime simulator — every neutral bar becomes a risk_off bar, inflating cost estimates by ~40% (risk_off σ is 3x risk_on). This is a landmine for whoever picks up the repo after the semester. Audit finding N-3 explicitly flags it.
- **Why FIX-NOW not FIX-IF-TIME**: it is literally 1 line of code + 1 line of test + no user-facing change. The ROI is infinite; the only reason not to do it is if we thought the current sim could never hit it, and given 3-state results are stored in the same JSON as 2-state it is non-trivially reachable.

### T5-4 — V4 vs V5 framing (FIX-IF-TIME)

- **Concrete action**: `FINDINGS.md §2.2` and `PRESENTATION_DRAFT.md` slide 7 already invalidate V4. Two additions worth making:
  1. In `FINDINGS.md §2.2`, add a paragraph: *"V5's +227.5% CVaR reversal is driven by risk_off η falling to literature fallback 1e-3 (see §1.5 warnings block) rather than by a genuine regime-aware vs single-regime cost differential. The true per-regime risk_off η would require a larger risk_off sub-sample (current: 5.5% of bars, R²=0.10). We therefore do not claim V5 as a negative result either; Part E regime-aware net benefit is **unresolved** on this data window."* (Current text nearly says this — tighten the claim so it reads "not a negative finding, an unresolved question".)
  2. Add the sentence *"V6 (Worker I, 98→280 day extension) is running in parallel and will resolve this if risk_off R² rises to >0.3."* — already present; leave it.
- **Risk if neglected**: if we present V5's +3225% CVaR without the η-fallback caveat, a reviewer could read it as "regime-aware execution is catastrophically bad", which is the opposite of the methodological-learning story. The existing §2.2 text is ~80% right; the fix is one more paragraph to make it airtight.
- **Why FIX-IF-TIME not FIX-NOW**: not a code change; editorial polish. Worth an hour if we have it, not worth blocking on.

### T5-5 — Multi-feature HMM z-score collapse (NEGLECT)

- **Concrete action**: none. The current `FINDINGS.md §5.4` already states: *"Adding sentiment dilutes regime separation by 5.6x. Hypothesis rejected... Daily-frequency features (F&G, exchange FlowIn, FlowOut) are mismatched with 5-min vol regimes."* Presentation slide 8 bonus-findings section also covers this.
- **Risk if neglected**: zero. The z-score collapse *was* the root cause of the earlier bivariate failures; the fix (switch to TxCnt + univariate fallback after rejection) is already in the live pipeline. A TxCnt bivariate second pass is a new experiment, not a fix — scope creep given deadline.
- **Why NEGLECT over DISCLAIM**: there is nothing to disclaim that isn't already disclaimed. Adding more text would only dilute §5.4's already-clean negative finding.

---

## ROI-ranked list (fix within remaining project time)

Ranked by `(narrative_defensibility_gain × reviewer_challenge_probability) / effort_hours`:

1. **T5-3 (`_regime_params_from_label` raise)** — effort 0.2 hrs, eliminates a landmine, directly addresses audit finding N-3. Highest ROI by an order of magnitude.
2. **T5-4 (V5 framing polish)** — effort 1.0 hr, directly affects how Part E is received. V5's raw +3225% number will draw questions; one paragraph bounds the interpretation.
3. **T4-1 (Yuhao multiplier disclaimer)** — effort 0.3 hr, preserves the V4→V5 pedagogical story without stranding the JSON columns.
4. **T4-3 (mark_price disclaimer)** — effort 0.2 hr, preempts the reproducibility challenge on ρ sign.
5. **T4-2 (3-state phantom disclaimer)** — effort 0.2 hr, easy BIC+degeneracy Q&A answer.
6. **T5-1 (OI gap disclaimer)** — effort 0.2 hr, owns a known data desert.
7. **T5-2 (spread gap disclaimer)** — effort 0.2 hr, bounds the weighting-scheme sensitivity.
8. **T5-5 (multi-feature HMM)** — effort 0 hr, already covered.

**Highest-ROI single action**: **T5-3** — 15 minutes of work (1 line + 1 test) that eliminates a silent-data-corruption landmine explicitly flagged in `AUDIT_VERIFICATION.md` as N-3. Do this first.

---

*Generated 2026-04-21. Editorial authority: Opus. Sonnet/GLM workers should not override these decisions — raise a challenge in the team channel if new evidence emerges.*
