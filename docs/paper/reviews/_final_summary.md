# Paper Review Loop V2 — Final Summary

**Paper:** `docs/paper/emnlp_draft_ko.md`
**Worktree:** `paper+review-loop-v2` (branch `worktree-paper+review-loop-v2`)
**Started:** 2026-05-11 03:01 (commit `f529a6a`, master ec0ab15 + archive move)
**Finished:** 2026-05-11 06:53
**Initial paper:** 834 lines (master `ec0ab15`)
**Final paper:** 837 lines (uncommitted)
**Net body delta:** +3 lines / 119 insertions / 116 deletions (substantial in-place rewrites — most net delta is in §1.5 + Abstract + §8.1 contribution-hierarchy reshape, balanced by §1.4 compression + scope-honesty insertions)

## Round-by-round headline

| Round | Persona | Decision | Tier signal | Major themes |
|---|---|---|---|---|
| 1 | Methodology | Borderline reject for EMNLP/NeurIPS Main; accept for Findings with major revisions | 3 CRIT + 10 MAJOR + 10 MIN | Post-hoc framework framed as prospective (CRIT-1); abstract overclaim of "uniquely passes free-lunch" (CRIT-2); embedded changelog + task IDs + lab-log §4.6 narration (CRIT-3 paper-as-experiment-log) |
| 2 | Writing | Borderline accept with major revision on writing axis | 3 CRIT + 9 MAJOR + 11 MIN | Title sub-line metadata residue (CRIT-W1); abstract 680 words × 3 paragraphs vs ≤250 venue norm (CRIT-W2); §6.6 5-arrow cross-reference chain substitutes references for argument (CRIT-W3) |
| 3 | Novelty | Borderline accept on novelty axis | 3 CRIT + 9 MAJOR; recommended venue = EMNLP Main if CRITs fixed, otherwise Findings | "Uniquely passes" reviewer-trap-vulnerable (CRIT-N1); ×12.7 auxiliary observation contaminates multiplicity-robust mitigation headline in abstract (CRIT-N2); "이론적 기여" overclaims (CRIT-N3 → relabel to 통합 framework). Substantive bar-raiser-grade ask: single-central-contribution restructure (MAJ-N1) |
| 4 | Aggressive | **REJECT for Main; borderline Findings only conditional on FATAL surfacing** | 4 FATAL + 9 MAJOR + 10 MIN | FATAL-A single-architecture overclaim of central contribution; FATAL-B Δdf 1/5 CI-clean vs Δem(b) headline pivot (anchoring task → side-effect marketing); FATAL-C framework prospective leg tests K=1, deployed at K=8; FATAL-D Telea inpaint texture confound |
| 5 | Bar Raiser | **Borderline Findings → weak Findings; not top decile** unless one-question answered | Top-decile readiness 2/6 → after R5 edits 5/6 + 1/6 partial | One-question: "Compress into one sentence — which of your six claims survives as thesis?" Substantive intervention: elevate **(a − m) paired-inpaint as generalisable design pattern** to thesis; demote E6 to *worked example* |

## Major themes addressed across rounds

- **CRIT-3 paper-as-experiment-log** (Round 1) → CHANGELOG.md extraction, task-ID stripping, generator-script citation removal, §1.5 6→4 contributions, scope-hedge consolidation. Round 2 follow-up: title sub-line removal, abstract 680→290 words, §6.6 cross-reference chain refactor, §4.6 lab-log rewrite. Round 3 follow-up: §1.5 restructured to 1 central + 3 supporting + 1 auxiliary. Round 5 follow-up: §1.5 + Abstract + §8.1 rewritten with **thesis-first** structure.
- **CRIT-1 / FATAL-C K=1 vs K=8 partial falsification of universal-K assumption** (Rounds 1, 4) → §5.4 framework labelled "사후 synthesis"; §4.6 verification leg labelled "partially prospective at K=1; deploy K=8 partial-falsifies universal-K". Propagated across Abstract / §1.3 / §1.5 / §5.4 / §8.1.
- **FATAL-B Δdf 1/5 CI-clean vs Δem(b) 5/5 Bonferroni-clean** (Round 4) → two-clause separation across Abstract / §1.5 / §6.2.3 / §8.1. *Anchoring effect* clause (single-dataset CI-strong) and *capability-side multiplicity-robust headline* clause (5/5 Bonferroni-clean) now presented as equal-weight but non-equivalent.
- **FATAL-A single-architecture case-study scope** (Round 4) → case-study qualifier inserted directly into E6 noun phrase at Abstract + §1.5 + §8.1 — no longer relies on §3.3 / §8.2 hedge alone.
- **FATAL-D Telea inpaint texture confound** (Round 4) → acknowledged in §6.2.1 Insight; (m − m') falsifier deferred to §8.4 item 7.
- **CRIT-N1 "uniquely passes" overclaim** (Round 3) → softened to "5-baseline panel 위 4-clause free-lunch 통과 후보"; ITI multi-head trap acknowledged; tuning-effort asymmetry note added in §6.5.
- **Bar-raiser one-question** (Round 5) → Option A adopted: design pattern is thesis, E6 is worked example. Thesis sentence verbatim at Abstract line 9, §1.5 line 39, §8.1 line 464. §8.1 Implications paragraph added with three field-level questions (cross-bias-class transfer, spectral K-prediction, cross-architecture transfer).

## Outstanding (DEFERred) items requiring future work

Logged in `§8.4 후속 작업` (9 items as of final state):

1. Random-K=8 baseline (FATAL-A from Round 4, MAJ-N6 from Round 3 tuning-effort asymmetry).
2. Cross-architecture E6 instantiation (FATAL-A single-architecture scope; bar-raiser top-decile checkpoint).
3. Pre-registered §4.6 cell for K=1 prospective verification (FATAL-C operative parameterisation gap).
4. CAA-at-K=1 + ITI multi-head empirical rows (CRIT-N1 + Round-4 attack 6).
5. ×12.7 paired-bootstrap CI (CRIT-N2 + Round-4 attack on ratio).
6. (m − m') Telea-residue falsifier — pixel-statistics-matched inpaint baseline (FATAL-D).
7. Spectral K-prediction analysis — Eigenvalue spectrum as predictor of operative K (§8.1 Implications + Round-4 MAJ-7).
8. Bonferroni-540 free recompute over 27-cell grid (Round-4 MAJ-1 selection-rule multiplicity).
9. Dyslexify + E6 composition test (MAJ-N8 Round 3).

## Bar-raiser signature ask (verbatim from Round 5 review)

> *"If you had to compress this paper into one sentence that a non-expert reader carries away three years from now, what is it? Then: which one of your six numbered claims survives as the thesis, and which collapse into supporting evidence for it?"*

**Answer adopted (Option A — full)**:

> **Vision-modality bias의 deployable mitigation은 causal pathway를 confounding scene variance로부터 분리하는 paired-inpaint calibration contrast 위에 구축할 수 있으며, 본 논문은 이 design pattern을 cross-modal numerical anchoring 위에 4-clause free-lunch worked example로 instantiate한다.**

Thesis sentence rehearsed verbatim at Abstract (line 9), §1.5 (line 39), §8.1 종합 (line 464). E6 demoted from central contribution to *worked example / proof of construction*; (a − m) paired-inpaint elevated from "subordinate calibration substrate" to **central methodological contribution**.

## Diff stat

- **Lines** 834 → 837 (+3 net; 119 insertions / 116 deletions)
- **Sections substantially rewritten**:
  - **Title block** — sub-line metadata stripped (R2)
  - **Abstract** — full rewrite, 680 → ~360 words (R2 compression, R3 ×12.7 drop, R4 scope-honesty insertion, R5 thesis-first restructure)
  - **§1.3** — partial rewrite for §5.4 post-hoc relabel + partial-prospective framing (R1, R4)
  - **§1.4** — compressed to ≤3-line forward-pointer (R2 partial, R5 full)
  - **§1.5** — three rewrites: R1 6→4 contributions; R3 single-central restructure (1 + 3 + 1); R5 design-pattern-as-thesis with E6 as worked example
  - **§4.6** — Lab-log narration → claim → evidence triple (R1)
  - **§5.4** — "이론적 기여" → "통합 framework"; "predict-then-verify" → "사후 synthesis + partially prospective at K=1" (R1, R3, R4)
  - **§6.2.1** — Telea-residue confound caveat added; Insight on (a − m) as generalisable design pattern (R3, R4, R5)
  - **§6.5** — "Uniquely passes" softened; ITI multi-head trap acknowledged; tuning-effort asymmetry note (R3)
  - **§6.6** — 5-arrow cross-reference chain + duplicate Summary block refactored to two paragraphs (R2)
  - **§8.1 종합** — thesis-first opening + 3-layer evidence stack + new Implications paragraph (R5)
  - **§8.4 후속 작업** — extended from 6 → 9 items across deferred FATAL/MAJOR (R4, R5)
- **Tables modified**: Table 6 caption (R2), Table 8 caption (R3 softening), Table 9 caption (R2 STRICT_FREE_LUNCH terminology fix)
- **Figures unchanged / added / removed**: 16 embedded figures intact; **Figure 3 designated canonical figure** (R5) — captures cross-model + cross-dataset (a − m) digit-pixel causality in one panel; no new figures created.
- **CHANGELOG.md** moved out of body to `docs/paper/CHANGELOG.md` (separate file, gitignored — local-only); v9–v11 entries logged.

## Final tier verdict

**Bar-raiser (Round 5) post-edit**: **Findings-tier weak-to-mid accept**; **Main-tier conditional on cross-architecture E6 instantiation (§8.4 item 2)**. Top-decile-readiness checklist moved from 2/6 met (pre-R5) to **5/6 met + 1/6 partial** (post-R5). The remaining partial — *single most-important figure captures contribution at a glance* — is closed for the thesis (Figure 3 designated) but not for the worked example's quantitative summary (Table 7 carries it alone; new figure deferred to camera-ready if needed).

The four prior rounds substantially closed the user's central concern that the paper reads as an experiment-log dump. After R5, the paper now has:

- **A single rehearsed thesis sentence** (Abstract / §1.5 / §8.1).
- **A clear contribution-hierarchy** (design pattern central; E6 worked example; three supporting findings as evidence licensing the design pattern; γ-β as separate auxiliary observation).
- **Scope-honest qualifiers** at every claim surface (single-architecture, partially-prospective, single-dataset-CI-clean for Δdf, Bonferroni-clean for Δem(b)).
- **A field-level Implications paragraph** that asks three questions the field will care about beyond this paper.

## Recommendation for next pass

1. **Camera-ready compute** (≤ 1 H100-week each):
   - §8.4 item 2 (second-architecture E6 calibration + 5-dataset evaluation) — single highest-leverage edit; promotes paper from Findings to Main.
   - §8.4 item 5 (×12.7 paired-bootstrap CI on Qwen3-VL γ-β) — closes Round 3 + Round 4 statistical residue.
   - §8.4 item 8 (Bonferroni-540 free recompute over 27-cell grid) — fully resolves Round-4 MAJ-1.
2. **No-compute polish**:
   - Sweep paper for any remaining inline canonical-source citations (`docs/insights/...`) — should now be in Reproducibility appendix only.
   - Sanity-check that Abstract numbers exactly match §6.2.3 / §7 tables.
3. **Optional Figure 7 plan** (camera-ready only): single 2×5 panel showing per-dataset Δdf + Δem(b) for E6, replacing Table 7's role as quantitative summary; would close last partial in top-decile checklist.
4. **PR strategy**: open PR `paper-review-loop-v2 → master`. User merges after review per the project's PR-only workflow rule.
