# Audit Chain

Historical audit MDs preserved for provenance. **Not the current source of truth** — see [`../../REPORT.md`](../../REPORT.md) for current narrative and [`../../FINDINGS.md`](../../FINDINGS.md) (V6) for current calibration/validation findings.

## Tracked files (committed to repo)

| File | Date | Role |
|---|---|---|
| PROGRESS.md | 2026-03-21 | W1 skeleton status snapshot — modules listed as "Skeleton/Partial" but project now complete |
| DATA_INTEGRITY_AUDIT.md | 2026-04-21 | 4 critical / 6 material / 5 informational issues identified by Sonnet R. Most fixed in subsequent commits. |
| AUDIT_VERIFICATION.md | 2026-04-21 | Opus U meta-audit of DATA_INTEGRITY_AUDIT — confirms 2/4 criticals, refines 1, finds 4 missed issues |
| TIER45_RESEARCH.md | 2026-04-21 | Literature survey for 8 Tier-4/5 structural issues |
| TIER45_DECISIONS.md | 2026-04-21 | FIX-NOW / FIX-IF-TIME / DISCLAIM / NEGLECT decisions for those 8 issues |
| TEAM_BRIEFING_APR22.md | 2026-04-22 | Team briefing — walk-forward OOS savings table now extracted to `../../RESULTS.md` §3 |

## Local-only (gitignored, included for local reference)

| File | Reason gitignored |
|---|---|
| PROJECT_AUDIT_REPORT.md (+ .pdf) | 2026-04-10 audit. **SUPERSEDED** by FINDINGS V6 — Part E V4 CVaR₉₅ −14% finding invalidated. SUPERSEDED banner added in-file. |
| WALKTHROUGH.md | Personal Chinese walkthrough notes (W1) |
| TECHNICAL_SUMMARY.md | Pre-audit methods scaffold — refreshed in REPORT.md §1-§4 |

## Why preserved here

1. **Audit-trail provenance** — academic reproducibility requires showing the chain of audits + fixes that led to current numbers
2. **Methodological-learning record** — Part E V1-V5 narrative (FINDINGS §2.2) cites these audit MDs as evidence for each invalidation
3. **TEAM_BRIEFING walk-forward table** — original breakdown of 6 walk-forward splits + savings %, copied verbatim into RESULTS.md §3
