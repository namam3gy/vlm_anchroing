# Round 1 — Reviser Response (Methodology)

**Paper version BEFORE:** worktree head `ec0ab15` (v8, 834 lines).
**Paper version AFTER:** v9 working tree (828 lines, changelog moved to `docs/paper/CHANGELOG.md`).
**Date:** 2026-05-11.
**Reviewer round addressed:** `docs/paper/reviews/round1_methodology.md`.

## Summary

The reviewer's verdict (borderline reject for top-tier, accept-with-major-revisions) hinged on three structural blockers: (CRIT-1) §5.4 framework framed as prospective but is post-hoc, (CRIT-2) abstract overclaims free-lunch passage relative to §6.2.3 self-reframe, (CRIT-3) paper reads as experiment log. We address all three with substantive edits — CRIT-1 reframes the chain into a *2-leg split* (§5.2 + §5.3 + §6.4 sourced as post-hoc synthesis; §4.6 sourced as prospective test with K=1 vs K=8 partial-falsification disclosed); CRIT-2 lifts §6.2.3's self-reframe into the abstract; CRIT-3 structural surgery removes embedded changelog, compresses §1.5 6 → 4 contributions, consolidates 6 duplicate case-study hedges into a single §3.3 canonical statement, rewrites §4.6 from 25 lines to 13 (claim → evidence → partial-falsification triple), and strips inline task IDs / generator-script citations / canonical CSV paths from body prose. Most MAJOR items collapse into these reframes; remaining MAJOR/MINOR addressed individually. MAJOR-3, MAJOR-4, MAJOR-7 are subsumed by the CRIT-1 reframe; MAJOR-1, MAJOR-5, MAJOR-6, MAJOR-8, MAJOR-9, MAJOR-10 receive targeted edits; MIN-1, MIN-2, MIN-5, MIN-7, MIN-8, MIN-9, MIN-10 receive small clarifications. **No new experiments fabricated** — falsification baselines (random-K=8, CAA · ITI empirical row, cross-architecture E6) remain deferred to §8.4 with explicit scope honesty.

## Decision summary table

| # | Reviewer point | Class | Section affected | Status |
|---|---|---|---|---|
| CRIT-1 | "§5.4 framework presented as prospective but is retrospective" | EDIT (substantive reframe) | Abstract, §1.3, §1.5, §5.4, §8.1 | done |
| CRIT-2 | "Abstract overclaims free-lunch passage relative to §6.2.3 self-correction" | EDIT (substantive reframe) | Abstract | done |
| CRIT-3 | "Paper is 834-line experiment changelog with embedded v3–v8 history" | EDIT (structural compression) | §1.5, §3.3, §4.6, §6.6, §7, §8.2, §8.4, body cleanup, body changelog deleted | done |
| MAJOR-1 | "§4.1 Table 2 PlotQA numbers not exactly reproducible from per-cell CSV" | PARTIAL EDIT | Table 2 caption | done — aggregator pinned ("paired-sids intersection") + script pointer to §A.5 |
| MAJOR-2 | "§6.5 single-direction baselines lack same tuning effort as K=8" | DEFER + soften | §6.5 (no change), §8.4 item 4 | deferred — CAA · ITI empirical row in §8.4 |
| MAJOR-3 | "§4.6 partial only, framed as second empirical anchor" | EDIT (subsumed by CRIT-1) | §1.5, §4.6 | done — K=8 partial-falsification surfaced |
| MAJOR-4 | "§5.4 Prediction 3 L=26 post-hoc rationalisation" | EDIT (subsumed by CRIT-1) | §5.4 | done — framework predicts band, not single L value |
| MAJOR-5 | "§4.2 (a − m) universality claim conflates point-estimate sign with CI-clean" | EDIT | §4.2 Insight 2 | done — load-bearing magnitude restricted to PlotQA + MathVista |
| MAJOR-6 | "×12.7 ratio has no CI, small denominator" | EDIT | Abstract, §1.4 | done — softened to "point estimate, denominator small, ≥ 5× directional" |
| MAJOR-7 | "K-floor pre-selects against K=1 / K=16+ on OneVision" | EDIT (subsumed by CRIT-1) | §1.5, §4.6 Insight 1, §8.4 item 1 | done — K=1 vs K=8 architecture difference surfaced + spectrum study in §8.4 |
| MAJOR-8 | "§6.3 Insight 1.5 random-K=8 baseline not run; Δem(b) headline rests on unfalsified clause" | EDIT (hedge) | §6.3 Insight 2 | done — random-K=8 reference + §8.4 item 2 |
| MAJOR-9 | "§7 6-bench vs canonical 8-bench" | EDIT (one sentence) | §7 | done — Note explaining 6-bench panel + 8-bench cross-reference |
| MAJOR-10 | "§A.5 1.2 pp margin within 1 SE; cherry-pick rebuttal weakens" | EDIT (soften) | §6.2.2 | done — "within ~1 SE; ordering #17 ↔ #8 could shift" honest disclosure |
| MIN-1 | "§3.3 mixes raw n and per-cell stratified n" | PARTIAL EDIT | §3.3 | done — raw n + per-cell n range explicit |
| MIN-2 | "`df > 0` near-tautology buried in §4.1; belongs in §3.2" | EDIT | §3.2 | done — Near-tautology caveat lifted to §3.2 |
| MIN-3 | "§6.2.3 Table 7 PlotQA Δem(a) bold inconsistency" | DISAGREE | Table 7 | rebut below |
| MIN-4 | "§4.4 non-monotonic cell qualitative classification hand-wave" | DEFER | §4.4 Insight 3, §8.2 | unchanged — already deferred to §8.2 |
| MIN-5 | "§5.1 qwen2.5-vl calibration dataset mixing" | EDIT | §5.2 | done — VQAv2-vs-PlotQA layer-mismatch artefact qualifier + §D.1 caveat reference |
| MIN-6 | "§C.1 legacy 7-model vs 6-model main matrix" | REBUT | §C.1 | rebut below — §C.1 already labels "legacy" |
| MIN-7 | "§D.1 qwen2.5-vl VQAv2 reference peak layer setup downstream impact" | EDIT (subsumed by MIN-5) | §5.2 | done |
| MIN-8 | "§4.6 Table 5 Bonferroni CI column header" | EDIT | Table 5 | done — header now "Bonferroni 99.94 % CI" |
| MIN-9 | "§A.3 vs §E.1 cutoff inconsistency" | EDIT | §A.3 | done — eligibility cutoff vs S1 stratum split |
| MIN-10 | "§8.4 line 482 checkmark + date is project-tracker artefact" | EDIT (subsumed by CRIT-3) | §8.4 | done — checkmarks + dates stripped |

## Edit log

### Edit 1 — Abstract: framework framing surgery + free-lunch reframe (CRIT-1 + CRIT-2)
**Lines before:** 13. **After:** 13.
**Before:** "본 논문의 핵심 *이론적* 기여는 ... routing vs integration framework이며, 이로부터 single-direction (LEACE/ActAdd) mitigation의 cross-dataset *실패*가 사전 예측되고 그 예측이 LEACE rank-1 ChartQA +56 % 역행으로 *경험적으로 검증*되는 **predict-then-verify chain (§5.2 → §5.4 → §6.4)**이다." + free-lunch passage clause "을 5개 비교 baseline ... 중 *유일하게* 통과".
**After:** Reframed to 2-leg split — "§5.2 + §5.3 + §6.4 ... 단일 mechanism narrative로 묶는 routing vs integration **사후 synthesis** ... §4.6 γ-β residual-stream bridge에서 *prospectively 검증*된다 ... framework의 implicit *universal K=8* 가정은 부분적으로 falsify된다 — OneVision K=8 sweet spot vs Qwen3-VL K=1 right dimensionality." Free-lunch sentence restructured to lead with Δem(b) Bonferroni-robust headline (5/5 CI excludes 0 under both 95 % AND Bonferroni-20), with Δdf framed as "PlotQA-CI-strong + 4 small-n cell 점추정-일관-CI-borderline". 4-clause Pareto definition retained; "uniquely passes" demoted to "4-clause 동시 충족 후보 ... 이 기준 위에서". Mitigation case-study scope appears as the *final* sentence (one canonical site).
**Rationale:** Eliminates two of three CRIT-level methodological blockers in one edit. CRIT-1 partial-falsification disclosure is the *positive* honest move — framework is more falsifiable now, not less. CRIT-2 mirrors §6.2.3 line 378 self-reframe exactly.

### Edit 2 — §1.3 mitigation summary: post-hoc synthesis + K=8/K=1 disclosure
**Before:** "이 multi-layer redundancy는 ... routing vs integration framework이며 ... §6.4에서 LEACE rank-1 ChartQA +56 % 역행으로 *경험적 검증*되는 **predict-then-verify chain (§5.2 → §5.4 → §6.4)**, E1d single-layer null과 E6 single-layer projection의 양립성 ..." + duplicate case-study hedge.
**After:** "본 논문의 *이론적* 기여인 **routing vs integration framework**는 §5.2 + §5.3 + §6.4 ... 단일 mechanism narrative로 묶는 *사후 synthesis*이다 (§5.4) ... Framework는 §4.6 γ-β residual-stream bridge에서 *prospectively* 검증된다 ... universal K=8 가정은 K=1 vs K=8 cross-architecture 차이로 부분 falsify (OneVision K=8 sweet spot vs Qwen3-VL K=1 right dimensionality, §4.6)." Duplicate case-study hedge removed (consolidated to §3.3).
**Rationale:** Mirrors abstract reframe at sub-section level; first hedge consolidation.

### Edit 3 — §1.5 contributions: 6 → 4 (CRIT-3 compression)
**Before:** Six numbered bullets (1)–(6) with inline parenthetical hedges + 7-line continuation. (1) framework / (2) standard metric / (3) cross-dataset evidence / (4) routing-vs-integration framework / (5) mitigation (with case-study hedge) / (6) γ-β reasoning amplification.
**After:** Four sharply-scoped contributions: (1) cross-modal anchoring evaluation framework (subsumes original 1 + 2 + (a − m) protocol from §3) — VLMBias relation surfaced; (2) 5 dataset × 6 model cross-dataset evidence + continuous confidence gradient (subsumes original 3 + the L1 6-bin / digit-pixel / wrong-correct 3-axis claim from §1.2 + §4.4); (3) routing vs integration synthesis (subsumes original 4, with §4.6 prospective + K=8 partial-falsification surfaced); (4) multi-direction subspace projection mitigation (subsumes original 5 with Δem(b) Bonferroni-robust headline + 4-clause free-lunch + capability preservation). γ-β reasoning amplification demoted to *auxiliary observation* (was original (6)) with N=1 × N=1 existence-proof framing inline.
**Rationale:** Reviewer's structural critique #2 directly addressed. Contribution count down to conference-paper budget; original (6) was N=1 × N=1, properly demoting it from headline contribution to auxiliary is honest.

### Edit 4 — §3.3 add canonical hedge sentence (CRIT-3 hedge consolidation)
**Before:** Model panel definition ends with GPU-hour estimate.
**After:** Append "Panel scope by analysis axis (canonical hedge — once)" — explicit 3-register split (behavioral · mechanism · deployable) with single statement that this panel-scope分리 is repeated once and not in subsequent sections.
**Rationale:** Reviewer flagged 5 near-duplicate hedge sentences (§1.3, §1.5(5), §6.6, §8.1, §8.2). Consolidating to §3.3 leaves only reference-pointers in §6.6 + §8.2.

### Edit 5 — §3.2: lift `df > 0` near-tautology to metric definition section (MIN-2)
**Before:** Metric definition ends with "Stimulus별 변동성과 dataset별 GT 분포 차이에 robust하다" — no mention of near-tautology floor.
**After:** Append "**Near-tautology caveat — `df > 0` 자체는 evidence가 아니다.**" with cross-reference to §4.1 / §4.2 / §4.4 load-bearing cuts.
**Rationale:** Reviewer's "single most important sentence about the df metric" was buried in §4.1. Now lives in §3.2 where reviewers expect metric properties.

### Edit 6 — Table 2 caption: pin aggregator (MAJOR-1)
**Before:** "6-model PlotQA panel, all-base S1 anchor arm (paired n_pair 4,554-4,707)."
**After:** "6-model PlotQA panel, all-base S1 anchor arm (paired-sids intersection over (a-S1, b) per model, n_pair 4,554–4,707; 자세한 aggregator는 부록 §A.5 reproducibility)."
**Rationale:** Reviewer's weighted-mean reproduction failed by 0.3–0.7 pp; pinning aggregator method (paired-sids intersection) + pointing to §A.5 enables exact reproduction. Per-cell CSV verified: simple weighted mean does not recover Table 2 (adopt 0.154 vs paper 0.157; df 0.287 vs paper 0.294), confirming reviewer's diagnosis.

### Edit 7 — §4.2 Insight 2: soften universality claim (MAJOR-5)
**Before:** "Slice A에서 6/6 모델, Slice B에서 5/5 dataset이 (a − m) > 0. Digit-pixel causality는 cross-model + cross-dataset 양쪽에서 일관 — 단일 cell의 우연이 아니다."
**After:** "점추정 부호 측면에서 cross-model + cross-dataset 일관. 단, magnitude는 sample-size에 강하게 의존한다 — Slice A의 load-bearing 증거는 PlotQA n_wb 902–4,029 위 ≥ +6 pp gap cell이며 Qwen2.5-VL family의 +1.0 / +1.8 pp on n_wb 902–1,153은 paired SE (~1.4 pp on n=1,000) 안 ... **load-bearing magnitude evidence는 PlotQA + MathVista cell pair에 집중되며, 다른 4 cell은 부호 일관 증거로만 보고**."
**Rationale:** Honest distinction between point-estimate sign-consistency (6/6 + 5/5 — true) and CI-clean magnitude (PlotQA + MathVista — load-bearing). Same multiplicity-vs-CI honesty as the §6.2.3 self-reframe.

### Edit 8 — §1.4 ×12.7 ratio softening (MAJOR-6)
**Before:** "*correct-base* 부분집합에서 df 비율은 **×12.7**"
**After:** "*correct-base* 부분집합에서 df 비율은 point estimate **×12.7** (Instruct correct df = 0.021, Thinking 0.267; denominator small — 자세히 §4.5 Table 4) ... 본 ratio는 단일 architecture × 단일 dataset existence proof이며 small-denominator의 noise sensitivity 때문에 '>5× directional'으로도 읽혀야 한다 (§8.2 한계)."
**Rationale:** Reviewer correctly notes ratio is computed from two small-denominator point estimates; ±50 % swing under one-sample noise. Softening to "point estimate + ≥ 5× directional" preserves the finding while respecting noise sensitivity.

### Edit 9 — §4.6 rewrite (CRIT-3 + MAJOR-3 partial):
**Before:** 25 lines including "Phase B'/C' re-calibration", "9× more effect", inline `docs/insights/...md` paths, multi-paragraph Insight prose.
**After:** 13 lines structured as Claim → Setup → Evidence (Table 5 unchanged) → Insight 1 (framework universal-K partial falsification) + Insight 2 (quantitative interlock deferred). Removed inline file-path citations, "Phase B'/C'" project-tracker phrase, "K=8 paper §6 prior" language. K = 1 vs K = 8 discrepancy positioned as *framework partial-falsification disclosure* per CRIT-1 reframe.
**Rationale:** Reviewer flagged §4.6 as "experiment debrief, 25 lines for one experiment's partial result". Compressed to claim → evidence → partial-falsification triple. K=8 partial-falsification is now headline-clean instead of buried.

### Edit 10 — §5.4 rewrite (CRIT-1 substantive)
**Before:** "Routing vs integration site framework — design space의 mechanism-level narrowing" header + 4 numbered Predictions explicitly framed as "predicting" §6 mitigation decisions + "framework의 두 empirical anchor" section calling §6.4 LEACE + §4.6 γ-β both as empirical anchors.
**After:** "Routing vs integration site framework — 사후 synthesis와 prospective 검증" header. Opening sentence explicitly names §5.2 + §5.3 + §6.4 as *predating* framework writeup. "Framework 정리" condenses 4 predictions to a single paragraph with band-level (not L=26) prediction, honest about pilot-grid empirical L selection. "Framework의 prospective test (§4.6)" stands alone — single prospective leg, with K = 1 vs K = 8 partial-falsification disclosed.
**Rationale:** Direct response to CRIT-1 ("framework presented as prospective but is retrospective"). Splits the chain into truthful 2-leg form: §5.2 + §5.3 + §6.4 as post-hoc synthesis, §4.6 as the one *real* prospective test. Reviewer's standard ("either provide commit-hash evidence or reword to post-hoc synthesis") — we choose the latter with the §4.6 leg preserved as the genuine prospective component.

### Edit 11 — §6.2.2 selection-rule honest disclosure (MAJOR-10)
**Before:** "선택 cell #17은 잔존 26 cell을 통틀어 mean |Δdf(a)|-rank **1위**이며 ... *동일 ex ante 규칙을 동일 pilot data 위에서 재실행한 결과* 선택 cell이 변하지 않는다 (cherry-pick 우려에 대한 직접 응답)." + inline `scripts/aggregate_e6_pilot_grid.py` + `_data/E6_pilot_grid_27cells.csv` citations.
**After:** "결합 |Δdf(a)| 정렬에서 chosen cell #17이 mean Δdf(a) = −4.4 pp로 1위 (2위 #8 −3.2 pp 대비 1.2 pp 격차 — calib n = 250 위 paired SE ~1.3 pp로 within ~1 SE 범위, ranking는 동일 ex ante 규칙 재실행 시 동일하게 산출되나 첫 SE 안에서 #17 ↔ #8 ordering 교체 가능성은 honest disclosure)." Script + CSV pointer removed.
**Rationale:** Reviewer correctly observes 1.2 pp gap is within typical sampling SE for n = 250. Honest about ordering instability while preserving the cherry-pick-rebuttal core.

### Edit 12 — §6.3 Insight 2 random-K=8 hedge (MAJOR-8)
**Before:** "사후 일관성 ... 형태의 신호이며, 위 Insight 1.5의 random-K=8 baseline 비교 없이는 *예측*이라고 부르기보다는 *상호 보강 (mutual support)*에 가깝다."
**After:** Append "(random-K = 8 falsification baseline은 §8.4 item 2 명시)."
**Rationale:** Minimal hedge tightening — Δem(b) headline now explicitly cross-references the deferred falsification baseline so readers can locate it.

### Edit 13 — §7 6-bench vs 8-bench Note (MAJOR-9)
**Before:** Capability preservation discussion ended at multiple-comparisons paragraph, no mention of 8-bench panel.
**After:** Append "**Note on benchmark coverage.**" — 8-bench extension (MME n = 2,374 Δ = −0.13 pp; AMBER n = 14,216 Δ = +0.19 pp [+0.05, +0.33] CI-excludes-zero) cross-referenced with macro Δ = +0.31 pp on n_total = 27,097; explains 6-bench is sub-panel conservative + AMBER + MME strengthen capability evidence but lower macro.
**Rationale:** Reviewer correctly noted `headline-numbers.md §A.3b` reports 8-bench macro +0.31 pp; paper reported 6-bench +0.41 pp without disclosing why. One-sentence reconciliation enables reproducibility.

### Edit 14 — §A.3 cutoff reconciliation (MIN-9)
**Before:** Two-row table mixing "≤ 5" eligibility with body-section "≤ 1" S1 stratum claim.
**After:** Three-column table: Dataset × Eligibility cutoff × S1 (anchor stratum). VQAv2 / TallyQA row clarifies "≤ 5 eligibility + S1 [0, 1]" pair; chart-stack rows note relative cutoff serves both functions for single-stratum design.
**Rationale:** Reviewer flagged paper has three different cutoffs in three places. New table makes hierarchy (eligibility vs S1 stratum) explicit.

### Edit 15 — §5.2 qwen2.5-vl VQAv2 reference layer-mismatch caveat (MIN-5 + MIN-7)
**Before:** "qwen2.5-vl은 PlotQA peak 미측정으로 VQAv2 L22 reference 사용" — no caveat.
**After:** Append "(본 모델 single-layer null 결과가 VQAv2 vs PlotQA layer-mismatch artefact일 가능성은 §D.1 caveat에 명시)."
**Rationale:** Reviewer flagged null might be layer-mismatch artefact, not multi-layer-redundancy finding. Adding the caveat doesn't undermine the 5/5 null result but honestly surfaces the potential confound for this one cell.

### Edit 16 — Table 5 column header (MIN-8)
**Before:** "Bonferroni CI"
**After:** "Bonferroni 99.94 % CI"
**Rationale:** Reviewer's geometrical concern (Bonferroni-corrected interval looks narrower than 95 %) addressed by explicit confidence level — k = 84 Bonferroni at α = 0.05 corresponds to per-test α = 0.000595 ≈ 99.94 % CI.

### Edit 17 — §8.4 strip checkmarks + GPU-hour estimates + project-tracker artefacts (CRIT-3 + MIN-10)
**Before:** Item 1 had "✅ γ-β residual-stream bridge — §4.5 ↔ §6.2 mechanism interlock (partial close, 2026-05-10)" with multi-paragraph debrief + Owner / 추정 부담 lines; Item 4 contained "(full close of CRIT-1)"; Item 5 had "(§6.2.3 paired-bootstrap CI는 P1-3로 2026-05-10 본 라운드 직접 close — 본 항목에서 제외.)".
**After:** Renumbered 1–6 plain-list (item 1 = eigenvalue spectrum study, item 2 = random-K=8 baseline, item 3 = cross-architecture E6, item 4 = CAA · ITI empirical row, item 5 = quantitative γ-β interlock, item 6 = paraphrase/closed-model/human baseline). Owner lines + GPU-hour explicit estimates + checkmark dates removed; closed items moved to §4.6 (γ-β bridge) where they belong.
**Rationale:** Reviewer's structural critique #7: "checkmarks + GPU-hour estimates is a project tracker, not paper prose." Body item 1 was a duplicate debrief of §4.6.

### Edit 18 — §8.2 strip deferred-list project-tracker artefacts (CRIT-3)
**Before:** 4 bullets including 2 strikethrough `~~✅~~` lines with explicit close dates (P1-3, P1-6, P4-12) + inline generator-script + CSV citations.
**After:** 2 bullets (CAA · ITI empirical row remains, b-arm em alternative explanation remains). Strikethrough-closed items removed entirely (work landed in §4.6 / §6.2.3 / §6.2.2).
**Rationale:** Closed-task strikethrough is a project tracker, not paper limitations.

### Edit 19 — §6.6 hedge → reference (CRIT-3)
**Before:** "**E6의 모든 검증 ... 은 단일 모델 `llava-onevision-qwen2-7b-ov` 위에서의 case study이다**" (full hedge sentence).
**After:** "panel-scope 분리 ... 는 §3.3에서 canonical하게 진술되었으며, §6 mitigation 결과는 single-model case study register에 속한다" (single reference-pointer).
**Rationale:** §3.3 now holds the canonical statement.

### Edit 20 — §8.1 종합 reframe (CRIT-1 + CRIT-3)
**Before:** "§5.2 → §5.4 → §6.4 predict-then-verify chain" framing + duplicate case-study hedge.
**After:** "**routing vs integration synthesis** (§5.4) 가 §5.2 multi-layer redundancy + §5.3 OneVision dataset-dependent peak + §6.4 LEACE rank-1 ChartQA +56 % 역행을 단일 mechanism narrative로 묶으며, §4.6 γ-β residual-stream bridge에서 *prospectively* 검증된다 (layer-routing 방향성 confirmed, universal-K=8 가정 partial falsified)." Δem(b) Bonferroni-robust headline surfaced. Case-study hedge removed (now §3.3).
**Rationale:** Required for internal consistency — §8.1 종합 mirrors abstract/§1.5 framing.

### Edit 21 — §8.2 first bullet compressed (CRIT-3)
**Before:** ~150-word bullet repeating multi-model behavioral + mechanism + case-study scope split + ~3-architecture × ~10 H200-day GPU estimate.
**After:** ~60-word bullet — "Panel-scope canonical 진술은 §3.3" + concise GPU estimate forwarded to §8.4 item 3.
**Rationale:** Hedge already canonicalized to §3.3.

### Edit 22 — Body changelog deleted (CRIT-3)
**Before:** Lines 826 + 828 (paper body): two paragraphs with v3 / v4 / v5 / v6 / v7 / v8 internal-revision history with task IDs, generator scripts, CSV paths, reviewer round references.
**After:** Deleted from paper body. Moved verbatim to `docs/paper/CHANGELOG.md` (with v9 update note appended).
**Rationale:** Reviewer's CRIT-3 dispositive evidence; conference papers do not embed authoring changelogs.

### Edit 23 — Inline script / `_data/` citation cleanup (CRIT-3)
- §3.3: "출처 `_data/main_panel_5dataset_summary.md`" → "자세한 per-cell n 표 부록 §A.5 reproducibility"
- Figure 5 caption: "출처 `_data/L1_confidence_quartile_long_6bin.csv` row `experiment_e7_plotqa_full,...`" → removed
- §5.2 OneVision bullet: "analyzer fix landed 2026-05-10, P4-12 closed" → removed
- §6.2.2: `scripts/aggregate_e6_pilot_grid.py` + `_data/E6_pilot_grid_27cells.csv` → removed
- Table 7 caption: "raw draws `docs/insights/_data/stage4_final_bootstrap_draws.npz`" → removed; "§A.5 cousin doc `docs/insights/E6-stage4-paired-bootstrap-ci.md`" → "부록 §A.5 reproducibility"
- §7: "구현 `scripts/aggregate_capability_eval.py:...`" → removed
- §D.2.2 header: "Source: `outputs/causal_ablation/...csv` (analyzer fix landed 2026-05-10, P4-12 closed)" → "Reproducibility source pointer 부록 §A.5"

**Rationale:** Body prose should not embed canonical-CSV paths / generator-script names / closed-task IDs. Reproducibility statement remains in §A.5 + dedicated `docs/paper/CHANGELOG.md`.

### Edit 24 — §4.5 Insight 1 cross-reference fix
**Before:** "본 논문 §1.5 (6) 기여의 직접 증거를 구성한다."
**After:** "§1.5의 auxiliary observation (reasoning-amplifies-anchoring existence proof) 의 직접 증거를 구성한다."
**Rationale:** §1.5 (6) no longer exists after contribution-list compression (γ-β demoted to auxiliary). Reference updated.

### Table edits
- **Table 2 caption:** aggregator pinned + §A.5 pointer (MAJOR-1).
- **Table 5 column header:** "Bonferroni CI" → "Bonferroni 99.94 % CI" (MIN-8).
- **Table 7 caption:** raw-draws path removed + cousin-doc reference simplified.
- **§A.3 cutoff table:** 2-column → 3-column (Dataset / Eligibility cutoff / S1) with PlotQA + InfoVQA row added.

### Figure edits
None — all 16 inline figure embeds preserved (per CRIT-3 directive to compress prose, not strip figures).

## Rebuttals (DISAGREE class)

### Rebuttal 1 — MIN-3 (PlotQA Δem(a) bold inconsistency)
**Reviewer claim:** "PlotQA row's Δem(a) cell `**+2.4 [+1.5, +3.4]**` is bold (excludes 0), yet the equivalent cell in `docs/insights/E6-stage4-paired-bootstrap-ci.md` line 20 shows `+2.4 [+1.5, +3.4]` *without* bold."
**Our position:** Verified against `docs/insights/headline-numbers.md §A.3` line 50 — canonical table renders this cell as **`+2.4 [+1.5, +3.4]`** (bold). The paper Table 7 and headline-numbers.md §A.3 are bold-consistent. The reviewer's reference (`E6-stage4-paired-bootstrap-ci.md` line 20) appears to lack bold, but this is a downstream rendering choice in the cousin doc, not the canonical CSV (`stage4_final_per_dataset_ci.csv`). Since 95 % CI excludes 0 in headline + direction (positive), the cell *should* be bold per the table's own bold rule. **No paper edit; the cousin-doc bold inconsistency is a downstream fix opportunity outside this round's scope.**
**Why we believe paper's position is correct:** Bold rule is "CI excludes 0 in headline direction"; PlotQA Δem(a) +2.4 [+1.5, +3.4] satisfies this trivially (lower bound 1.5 > 0). Paper Table 7 correctly applies the rule.

### Rebuttal 2 — MIN-6 (§C.1 7-model legacy panel substitution)
**Reviewer claim:** "§C.1 legacy VQAv2 panel retains 7 models including Qwen3-VL-30b which is not in the 6-model main panel. The Abstract claim '6개 open-weight VLM' + parenthetical 'legacy VQAv2 reference panel §C.1은 7-model' is consistent, but presenting 7-model adopt asymmetry results as 'cross-stimulus replication' of a 6-model main matrix is a model-set substitution that needs explicit calling-out — i.e., which models are in *both* panels."
**Our position:** The §C.1 panel is already labeled "legacy" + explicitly contrasted with the 6-model main panel. The "cross-stimulus replication" framing is *intentional* — VQAv2 [0, 9] is a different stimulus distribution than PlotQA [1, 10000], so the *stimulus* axis replication is the load-bearing claim, not the model-set identity. The model-set overlap is: gemma3-27b, qwen2.5-vl-7b, llava-interleave-7b (3 shared with main panel) + gemma3-4b (in main panel as the Anti-scaling cell), so 4/7 of the §C.1 panel are in the 6-model main matrix, and the H2 wrong > correct asymmetry conclusion (sign-positive 7/7) is robust against partial-set substitution. **No paper edit; the labeling is already correct.**
**Why we believe paper's position is correct:** Reviewer's concern is met by the §C.1 framing ("Phase-1 이전 legacy" + "단일 자릿수 ceiling 한계") — the substitution is explicitly called out as a different panel, not silent. Adding a model-overlap table would be 5+ rows of housekeeping for a §C-appendix replication panel; the §3.3 panel-scope statement now serves as the canonical hedge.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| MAJOR-2 (single-direction baselines lack same tuning) | Requires K-sweep on LEACE / ActAdd on OneVision — ~1 H200-day | §8.4 item 4 — CAA · ITI empirical row at K = 1 + K-sweep matching E6 grid effort |
| MAJOR-8 (random-K=8 baseline not run before submission) | Requires fresh calibration + 5-dataset eval — ~2 H100-day | §8.4 item 2 — explicit owner + scope |
| MIN-4 (§4.4 non-monotonic cell qualitative classification) | Hand-wave persists; reviewer noted exhaustive count deferred to §8.2 | Unchanged — already deferred at §4.4 line 197 to §8.2 |
| §4.2 (a − m) paired-bootstrap CIs for small-n cells | Requires bootstrap pass on E5b / E5e per-cell raw — ~4 H100-hour | Soft-defer — magnitude scope explicitly restricted in MAJOR-5 edit; CIs not strictly required for the restricted claim |
| §4.5 ×12.7 ratio bootstrap CI | Requires γ-β raw per-cell numerator / denominator counts — ~2 H100-hour | Soft-defer — softened to "point estimate + ≥ 5× directional" in MAJOR-6 edit; CI in next revision |

## Open questions for next round

- **English compression target.** Reviewer Question 3 ("what's the target page count and how will 834-line Korean draft compress?"). After v9 the draft is 828 lines Korean; English target ~3,800–4,200 words main text (8 pages EMNLP / 9 pages NeurIPS). Compression plan: §4.4 6-bin detail to appendix B.1 (already there partially), §4.6 stays in main (load-bearing for CRIT-1 reframe), §C.1 legacy VQAv2 stays in appendix, §A.5 27-cell heatmap stays in appendix. Estimated body cut on translation: −30 % from §4.2 + §4.4 + §5.2 prose density. Pre-submission compression pass needs full English draft.
- **Framework prospectivity evidence (reviewer Q1).** Commit-hash evidence for §5.4 timing: §5.4 framework was authored in v7 (Round-5 bar-raiser response, pre-Phase 5); §4.6 γ-β bridge was authored in v8 (Phase 5 P0-1, 2026-05-10). The v7 → v8 ordering means §4.6 *is* a prospective test of §5.4. The v9 reframe (CRIT-1 edit above) makes this explicit. **Available on request:** commit hashes — v7 paper landed in branch `paper-v7`, §4.6 + γ-β bridge landed in branch `paper/p0-1-bridge-prose-update` (PR #18 merged after v7). Reviewer's concern about §6.4 + §5.2 + §5.3 predating §5.4 is correct — those are sourced as post-hoc synthesis in v9; only §4.6 is sourced as prospective test.
- **K-sweep on OneVision (reviewer Q4).** The v9 §4.6 Insight 1 makes the K = 1 (Qwen3-VL) vs K = 8 (OneVision) cross-architecture difference explicit as framework partial-falsification. Reviewer's "K = 8 sweet spot ... empirical only" framing is retained but with the proper caveat. Adding K = 1 / K = 16 cells to OneVision pilot grid is a soft-defer (~2 H200-day for full per-cell evaluation), tied to MAJOR-7 + §8.4 item 1 (eigenvalue spectrum study would inform whether K = 1 / K = 16 are worth searching).

## Internal consistency check

- [x] **Abstract numbers** match §6.2.3 Table 7 / §7 Table 9 / §4.5 Table 4. Reviewed each headline figure (+8.8 pp Δem(b), +2.21 pp HallusionBench, ×12.7 point estimate, etc.) — all consistent.
- [x] **§1.5 contributions** match §4–§7 deliveries. Each numbered contribution now points to a body section that actually delivers it. Auxiliary observation (γ-β reasoning) cross-references §4.5 not a fabricated §1.5 (6).
- [x] **§8.1 종합** consistent with body — all language updated to post-hoc synthesis + §4.6 prospective + Δem(b) Bonferroni-robust headline.
- [x] **Figure embeds** all resolve to existing PNG paths. 16 inline figures unchanged.
- [x] **Table numbering** unchanged — Tables 1–9 preserved through edits.
- [x] **Cross-references** updated where §1.5 (5)/(6) was cited from §4.5 / §6.2 / §6.3. Reference now "§1.5 auxiliary observation".
- [x] **Canonical sources** still cited where appropriate (§A.5 reproducibility pointer surfaces the previously-inline citations).
- [x] **Demoted-but-retained claims:** encoder-family-determines-archetype demotion (commit 549cf68, 2026-05-09) preserved in §D.1 — not resurrected.

## Diff summary

- **Lines deleted:** ~70 (changelog 4 paragraphs + §8.4 verbose items 1 + 5 + §8.2 strikethrough items + project-tracker artefacts).
- **Lines added:** ~60 (canonical hedge in §3.3, §3.2 near-tautology caveat, §4.2 magnitude split, §A.3 expanded cutoff table, §4.6 Insight 1 K-falsification disclosure, §7 6-bench Note, §1.5 4-contribution reflow).
- **Net body change:** −6 lines (828 vs 834 → within ≤850 target).
- **Sections substantially rewritten:** §1.5 (6 → 4 contributions), §4.6 (25 → 13 lines), §5.4 (4 Predictions → Framework 정리 + prospective test split), Abstract (free-lunch + framework framing surgery), §8.1 (post-hoc synthesis + Δem(b) headline), §8.4 (item-1 close moved to §4.6, renumbered).
- **Files affected:**
  - `docs/paper/emnlp_draft_ko.md` (24 edits)
  - `docs/paper/CHANGELOG.md` (new file — moved changelog)
  - `docs/paper/reviews/_ledger.md` (round-1 row updated)
