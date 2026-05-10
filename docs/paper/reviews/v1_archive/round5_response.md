# Round 5 — Author Response to Bar-Raiser

**Paper version BEFORE:** `docs/paper/emnlp_draft_ko.md` @ 581 lines, v6 changelog footer (post-Round-4).
**Paper version AFTER:** `docs/paper/emnlp_draft_ko.md` @ 604 lines, v7 changelog footer (post-Round-5).
**Date:** 2026-05-09.
**Reviewer round addressed:** `docs/paper/reviews/round5_bar_raiser.md` (Senior bar-raiser, top-10 % standards, forward-looking).

## Summary

This is the *final* round and unusual in posture. The bar-raiser is **forward-looking, not bug-finding** — they explicitly named a list of seven elements not to touch and asked one elevating question (the γ-β residual-stream bridge experiment) that, if landed, would push the paper from "Solid Findings, top of band" to "Main, lower half of accepted." Because the bridge experiment requires new GPU work (cheap form ~2 H100-day, clean form ~1 H200-week), the only *honest* response in this round is to acknowledge the question, frame it as the lead item in a new *load-bearing follow-up* §8.4, and sharpen the existing **§5.2 → §6.4 predict-then-verify chain** which the bar-raiser named as the most-citable 5-year finding. Seven items on the protect-list are untouched. No paper number changes.

Net posture: 4 EDIT (predict-then-verify framing sharpening + §8.4 follow-up section + §6.2.1 design-pattern sentence + v7 changelog) + 3 PARTIAL EDIT (bar-raiser secondary asks → §8.4 sub-items) + 0 REBUT + 4 DEFER (bridge experiment + eigenvalue spectrum + random-K=8 + cross-architecture E6, all framed as future work in §8.4 with owner / GPU-hour estimates).

## Decision summary table

| # | Reviewer point (verbatim) | Class | Section affected | Status |
|---|---|---|---|---|
| 1 | "Does the (a − m) K=8 subspace amplitude grow over Thinking-mode trace generation, and does that growth quantitatively predict the ×12.7 correct-base amplification?" — bar-raiser signature ask | EDIT (acknowledge + frame as future-work lead) + DEFER (experiment itself) | §8.4 (new) | done — added as lead item with cheap/clean form, positive/negative implications |
| 2 | 5-year most-citable finding identified as §5.2 → §6.4 predict-then-verify chain — currently presented as parallel findings, not interlocked | EDIT | abstract + §1.3 + §1.5 (4a) + §8.1 종합 | done — 4-callsite framing sharpening, named as *이론적 (theoretical)* contribution |
| 3 | Eigenvalue spectrum of D[:, L=26, :] — rank-8 elbow check (existing data, single plot) | DEFER | §8.4 item 2 | logged with owner / 4-H100-hour estimate |
| 4 | Random-K=8 subspace baseline (§6.3 Alt-1 falsification) — "cheapest available rigor improvement on the entire DEFER list" | DEFER | §8.4 item 3 | logged with owner / 2-H100-day estimate |
| 5 | Encoder-family-determines-archetype (§4.4 Insight 3 / §5.1 Insight 1) — promote from sub-bullet to top-line contribution | NO-OP (bar-raiser protect-list bias) | §1.5 (4b) | already split into (4a)/(4b) in Round 3; bar-raiser flagged as still buried but moving it would touch §1.5 hedge stack which is on the protect-list — defer to camera-ready |
| 6 | (a − m) contrast as generalizable design pattern — implicit suggestion in protect-list | EDIT (one sentence append, not rewrite) | §6.2.1 Insight | done — sentence appended naming "calibration contrast as paired difference isolating causal pathway from confounds" without modifying existing prose |
| 7 | Bar-raiser protect-list (7 items): (a − m) contrast / six-callsite hedge / §6.2.3 reframing / Δem(b) ≥ 0 clause / §1.5(1) hedge stack / §5.3 self-disclosure / §4.7 boundary case | NO-OP | various | all 7 items untouched as instructed |
| 8 | Cross-architecture E6 (CRIT-1 full close, R4 carryover) | DEFER | §8.4 item 4 | logged with 30-H200-day estimate (already in §8.2 — duplicated as future-work item with priority) |
| 9 | CAA · ITI empirical rows + §6.2.3 paired-bootstrap CI (R4 carryover) | DEFER | §8.4 item 5 | logged with 1-H200-week estimate |
| 10 | Paraphrase robustness · 폐쇄 모델 · human baseline (R1-R4 carryover) | DEFER | §8.4 item 6 | logged with separate work-unit framing |

## Edit log (every paper change in this round)

### Edit 1 — Abstract: predict-then-verify framing as theoretical contribution

**Reviewer point addressed:** #2 from decision table.
**Reviewer reasoning:** Bar-raiser names §5.2 → §6.4 chain as the single most-citable 5-year finding ("the headline a future activation-steering paper will cite to motivate going multi-direction"). Current abstract has it as a clause; needs to read as the paper's *theoretical* contribution.
**Before:**
> ... **single-layer mask ablation은 6-model 메커니즘 panel에서 6/6 null** — signal은 multi-layer redundant이다 ... 이 발견이 single-direction (LEACE/ActAdd) mitigation의 cross-dataset *실패*를 예측하고 검증한다 (§5.2 → §6.4; per-dataset mean-anchor direction이 측정 가능하게 다른 곳을 가리킴, cos ≈ 0.47-0.62).

**After:**
> ... **single-layer mask ablation은 6-model 메커니즘 panel에서 6/6 null** — signal은 multi-layer redundant이다 ... 본 논문의 핵심 *이론적* 기여는 이 multi-layer redundancy 발견이 single-direction (LEACE/ActAdd) mitigation의 cross-dataset *실패*를 사전 예측하고 그 예측이 LEACE rank-1 ChartQA +56 % 역행으로 *경험적으로 검증*되는 **predict-then-verify chain (§5.2 → §6.4)**이다 — mechanism analysis가 mitigation failure mode를 사전 진단하고 그 진단이 multi-direction subspace 설계를 유도하는 paradigm move (per-dataset mean-anchor direction이 측정 가능하게 다른 곳을 가리킴, cos ≈ 0.47-0.62).

**Rationale:** Names the chain as *이론적 기여* (theoretical contribution) and *paradigm move*. No metric numbers added or changed. The "+56 %" was already in §6.4; surfacing it here gives the abstract a concrete verification anchor. This is the highest-leverage single edit per the advisor's read — bar-raiser's 5-year finding now reads as the paper's central theoretical move from the abstract onward.

### Edit 2 — §1.3: predict-then-verify chain explicit in introduction

**Reviewer point addressed:** #2.
**Before:**
> Anchor attention은 encoder family별로 단일 peak layer를 가지지만 single-layer ablation은 6-model 메커니즘 panel에서 6/6 null — signal은 multi-layer redundant이다 (OneVision Main 확장은 분석기 수정 pending, §5.3). **E4** upper-half attention re-weighting은 ...

**After:**
> Anchor attention은 encoder family별로 단일 peak layer를 가지지만 single-layer ablation은 6-model 메커니즘 panel에서 6/6 null — signal은 multi-layer redundant이다 (OneVision Main 확장은 분석기 수정 pending, §5.3). 이 multi-layer redundancy는 **single-direction mitigation의 cross-dataset 실패를 *예측한다*** — 그리고 그 예측은 §6.4에서 LEACE rank-1 ChartQA +56 % 역행으로 *검증된다* (§5.2 → §6.4 predict-then-verify chain — 본 논문의 핵심 이론적 기여로, mechanism analysis가 mitigation failure mode를 사전 진단한 사례). **E4** upper-half attention re-weighting은 ...

**Rationale:** §1.3 is currently the "mechanism + mitigation" framing in the introduction. Inserting one sentence between the redundancy claim and E4 description makes the predict-then-verify chain a *headline theoretical move* in §1, not just a result statement in §6.4. Stays surgical — one sentence, no other changes.

### Edit 3 — §1.5 (4a): predict-then-verify named as theoretical contribution

**Reviewer point addressed:** #2.
**Before:**
> (4a) 6-model 메커니즘 panel에서 single-layer ablation이 6/6 null임을 보이고 그 *multi-layer redundancy*를 정량화한다.

**After:**
> (4a) 6-model 메커니즘 panel에서 single-layer ablation이 6/6 null임을 보이고 그 *multi-layer redundancy*를 정량화하며, 이 발견이 single-direction (LEACE/ActAdd) mitigation의 cross-dataset 실패를 사전 예측하고 §6.4에서 *경험적으로 검증*되는 **predict-then-verify chain (§5.2 → §6.4)**을 본 논문의 *이론적* 기여로 제시한다 — mechanism analysis로부터 mitigation failure mode를 사전 진단하고 그 진단이 multi-direction subspace 설계를 유도한다.

**Rationale:** Bar-raiser flagged that §1.5 has the architecture for the contribution but the chain is buried. Per advisor — "sharpen (4a)+(5) rather than create a new bullet." Edit extends (4a) without introducing (4c) or touching (4b) (which is on the protect-list). The "이론적 기여" label is now in three callsites (abstract / §1.3 / §1.5 (4a)) for consistency.

### Edit 4 — §8.1 종합: predict-then-verify chain restated in conclusion

**Reviewer point addressed:** #2.
**Before:**
> Cross-modal numerical anchoring은 VLM에서 실재하며, **세 gate의 conjunction** ... Mechanism 측은 *multi-layer redundant*이며 single-layer ablation은 null — 이로부터 single-direction mitigation의 실패가 *예측되고 검증*되며, *multi-direction subspace projection*이 두 failure mode를 동시 우회하는 유일 후보로 도출된다.

**After:**
> Cross-modal numerical anchoring은 VLM에서 실재하며, **세 gate의 conjunction** ... Mechanism 측은 *multi-layer redundant*이며 single-layer ablation은 null — 본 논문의 *이론적* 기여인 **§5.2 → §6.4 predict-then-verify chain** (multi-layer redundancy가 single-direction mitigation의 cross-dataset 실패를 *사전 예측*하고 LEACE rank-1 ChartQA +56 % 역행으로 *경험적 검증*) 으로부터 *multi-direction subspace projection*이 두 failure mode를 동시 우회하는 유일 후보로 도출된다 — mechanism analysis가 mitigation failure mode를 사전 진단하고 그 진단이 다음 단계 설계를 유도하는 paradigm move.

**Rationale:** Final callsite of the four-place framing pass (abstract / §1.3 / §1.5 (4a) / §8.1). The chain now reads as a single *paradigm move* across the paper's framing surfaces. Concrete instance ("LEACE rank-1 ChartQA +56 % 역행") preserved verbatim across all four callsites for internal consistency.

### Edit 5 — §6.2.1 Insight: (a − m) contrast as generalizable design pattern (one-sentence append)

**Reviewer point addressed:** #6.
**Reviewer reasoning:** Bar-raiser implicitly suggested in protect-list framing that the (a − m) contrast could be elevated to a *principle*. User instruction: one sentence at most, do not overclaim, do not modify existing protected prose.
**Before:**
> ... *§4.3의 (b, m, d) 통제 실험이 §6.2 subspace 설계를 직접 정당화*한다 — behavioral analysis가 mitigation 설계로 환원된 사례.

**After:**
> ... *§4.3의 (b, m, d) 통제 실험이 §6.2 subspace 설계를 직접 정당화*한다 — behavioral analysis가 mitigation 설계로 환원된 사례. 일반 design pattern으로 표현하면: **calibration contrast는 인과 통로 (causal pathway) 를 confounding variance로부터 *분리*하는 paired difference여야 한다** — 본 사례에서는 (digit pixel → answer shift) 통로를 (anchor scene background → general distraction) confound로부터 분리하기 위해 (a − m) paired-inpaint이 그 분리 구조를 정확히 제공한다.

**Rationale:** Strictly *appended* (no existing word changed). Names the principle in one sentence, then immediately grounds it in the (a − m) instance to avoid free-floating claim. Bar-raiser's "Protect it; do not 'improve' it" is honored — the existing Insight survives untouched, the addendum reads as a coda. Per advisor: "If you can't do it without touching existing prose, skip it"; here we did not touch existing prose.

### Edit 6 — §8.4 (new section) — 후속 작업 (load-bearing follow-up)

**Reviewer point addressed:** #1, #3, #4, #8, #9, #10.
**Reviewer reasoning:** Bar-raiser's signature ask is *forward-looking* — the experiment that would tip judgment from Findings to Main. Cannot be addressed in current revision (requires GPU work). Advisor's explicit guidance: "give it its own home, not buried in deferred bullets ... §8.4 후속 작업 / Future work subsection that leads with the bar-raiser's bridge experiment and includes the secondary asks as ranked sub-items."
**Before:** §8.3 윤리 → directly to References.
**After:** §8.3 윤리 → §8.4 후속 작업 (load-bearing follow-up) → References. New section contains 6 ranked items:

  1. **(Lead) γ-β residual-stream bridge** — bar-raiser signature ask. Cheap form (reuse OneVision K=8 V_K[L=26] as instrument on Qwen3-VL-Thinking traces) + clean form (Qwen3-VL-Instruct own (a − m) calibration). Positive outcome: §4.6 elevates from N=1 existence proof to residual-stream-level mechanism claim, K=8 subspace is dual-role (mitigation target + measurement instrument), partial close of CRIT-1. Negative outcome: §4.6 honestly retreats to "behavioral existence proof, mechanism unidentified." Both forms hedged with framework not fabricated outcomes. Owner: paper author. Estimate: cheap ~2 H100-day, clean ~1 H200-week.
  2. Eigenvalue spectrum of `D[:, L=26, :]` — rank-8 elbow check converts K=8 from grid-search artifact to data-property prediction; existing data, single plot. Owner: paper author. Estimate: ~4 H100-hour.
  3. Random-K=8 subspace baseline — head-to-head Alt-1 falsification for §6.3 b-arm em interpretation. Estimate: ~2 H100-day.
  4. Cross-architecture E6 replication (CRIT-1 full close, with sub-routing through item 2 if eigenvalue spectrum lands). Estimate: ~30 H200-day.
  5. CAA · ITI empirical rows + §6.2.3 paired-bootstrap CI. Estimate: ~1 H200-week.
  6. Paraphrase / closed-source / human baseline (§8.2 carryover).

Closing paragraph: "위 *1*은 본 논문의 elevating 가능성에 가장 큰 leverage를 가지며 ... *2-3*은 가장 cheap한 rigor 향상이며, *4*는 가장 큰 GPU 부담을 요구하는 generalisation 항목이다."

**Rationale:** New section, not a §8.2 limitations modification. §8.2 keeps its existing register (한계 + operational deferral); §8.4 carries the *forward-looking* register. Bar-raiser's "single experiment that would tip judgment" gets explicit prominence as item 1 with full positive/negative implications. Per advisor: "frame as conditional ('if positive ... if negative ...') matching the bar-raiser's own framing" — done verbatim. Item 4 (cross-architecture E6) is new in §8.4 *and* still in §8.2 deferred-list — intentional duplication: §8.2 is the limitations register (we acknowledge the gap), §8.4 is the prioritization register (we name when/how it would close).

### Edit 7 — Changelog: v7 entry

**Reviewer point addressed:** Bookkeeping for Round 5 final revision.
**Before:** Footer ends at v6 changelog block.
**After:** v7 changelog block appended documenting (a) predict-then-verify framing sharpening across 4 callsites, (b) §8.4 new section with γ-β residual-stream bridge as lead, (c) §6.2.1 design-pattern sentence append, (d) explicit confirmation that the bar-raiser 7-item protect-list is untouched.

**Rationale:** Maintains audit trail consistent with v3 / v4 / v5 / v6 footer pattern. Lists every R5 edit and explicitly enumerates the protect-list to make it visible to any future reviewer that R5 honored bar-raiser's protective duty.

### Table edits

None this round. No metric numbers changed. Tables 1–8 + Tables A.5 / E.1 / E.2 untouched.

### Figure edits

None this round. All 16 inline figure embeds preserved.

## Rebuttals (DISAGREE class)

None this round. The bar-raiser is forward-looking; their points are not bug-finding critiques to rebut. The closest thing to a contestable framing is the bar-raiser's tier verdict ("Solid Findings, top of band — not Top-10 %") — and the Round-4 author response itself already acknowledged "closer to strong Findings than weak-accept Main" (R4 honest assessment). Round 5 does not contest the tier verdict; it accepts and builds the §8.4 follow-up plan that the bar-raiser explicitly named as the path to elevation.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| Bridge experiment (γ-β residual-stream amplitude on Qwen3-VL-Thinking trace) | Requires new GPU run on Qwen3-VL pair; not scriptable in revision window | Owner: paper author. Estimate: cheap ~2 H100-day, clean ~1 H200-week. Listed as §8.4 item 1 (lead). |
| Eigenvalue spectrum rank-8 elbow check | Existing data, single SVD pass — not run in this revision | Owner: paper author. Estimate: ~4 H100-hour. Listed as §8.4 item 2. |
| Random-K=8 subspace baseline (§6.3 Alt-1 falsification) | Requires non-anchor calibration set + re-eval | Owner: paper author. Estimate: ~2 H100-day. Listed as §8.4 item 3 (cheapest rigor improvement). |
| Cross-architecture E6 (CRIT-1 full close, R4 carryover) | Requires per-archetype calibration + pilot grid + 5-dataset eval + capability eval | Owner: paper author. Estimate: ~30 H200-day. Listed as §8.4 item 4. |
| CAA · ITI empirical rows + Table 6 paired-bootstrap CI (R4 carryover) | Requires CAA at K=1 calibration + ITI head-level adaptation + bootstrap n=1,000 on Table 6 | Owner: paper author. Estimate: ~1 H200-week. Listed as §8.4 item 5. |
| Pre-registration registry document (R4 carryover MAJ-6) | Not retrospectively addable | Owner: future submissions. Camera-ready or future paper would pre-register on OSF / AsPredicted. |
| Encoder-family promotion to top-line contribution | Bar-raiser secondary ask flagged as still buried; moving it would touch §1.5 hedge stack on protect-list | Owner: camera-ready editor. The (4a)/(4b) split from R3 is preserved; further elevation defers to camera-ready. |
| Paraphrase robustness / closed-source defuse / human baseline (R1-R4 carryover) | Each a separate work unit | Listed as §8.4 item 6. |

## Open questions for next round

This is the final round; there is no next round in this review loop. Items that did not fit cleanly:

- **Whether the bridge experiment outcome would *retroactively* warrant abstract changes.** If item 1 lands positive, abstract / §1.3 / §1.5 (6) currently saying "first-evidence existence proof" would tighten to "residual-stream-level mechanism claim." This is conditional on a result we do not have; cannot be acted on this round but is the obvious post-experiment edit path.
- **§5.2 multi-layer redundancy formal definition.** Bar-raiser axis 3 / 5 raised this as a formalization upgrade ("distributed mass with each layer ≥ ε? Compositional conjunction? Eigen-spectrum of (a − m) with rank elbow at 8?"). The eigenvalue spectrum (§8.4 item 2) is the cheapest operational version; a fully formal definition (theoretical statement of redundancy) is a longer-term project.
- **Whether Δem(b) ≥ 0 clause should be renamed.** Bar-raiser preemptively defends it: "If a future reviewer pushes back, neutralize the *name* ('4-clause Pareto+') not the *substance*." Not acted on R5; logged for camera-ready discretion.

## Internal consistency check

After all R5 edits, the paper holds:

- [x] Abstract numbers still match §4-§7 tables. **No metric numbers changed in R5.** Only the framing of the §5.2 → §6.4 chain was sharpened.
- [x] §1.5 contributions still match §4-§7 deliveries. **(4a) sharpened with predict-then-verify framing — refers to §5.2 (delivered) + §6.4 (delivered).**
- [x] §8.1 종합 still consistent with body. **Predict-then-verify framing now matches abstract / §1.3 / §1.5 (4a).**
- [x] All figure embeds still resolve to existing PNG paths. **No figure edits.**
- [x] No figure or table renumbering issues introduced. **No table or figure changes.**
- [x] Canonical sources still cited where appropriate. **§6.4 LEACE +56 % cite preserved; abstract / §1.3 / §8.1 reference §5.2 → §6.4 chain consistently.**
- [x] §8.4 new section does not contradict §8.2. **§8.2 = 한계 register; §8.4 = forward-looking priority register. Cross-architecture E6 appears in both; §8.2 frames as limitation, §8.4 frames as prioritized follow-up — registers are complementary, not contradictory.**
- [x] Bar-raiser 7-item protect-list untouched. **Verified line-by-line:** (a − m) §6.2.1 Insight existing prose unchanged (one sentence appended only); R4 6-callsite hedge unchanged across abstract / §1.3 / §1.5 (5) / §6.6 / §8.1 / §8.2; §6.2.3 paired-sids reframe unchanged; Δem(non-anchored) ≥ 0 clause in §6.2.3 strict free-lunch definition unchanged; §1.5 (1) hedge stack unchanged; §5.3 OneVision dataset-dependent peak self-disclosure unchanged; §4.7 InternVL3 boundary case unchanged.

## Diff stat

- **Lines:** 581 → 604 (+23 net).
- **Word count delta:** ~+1,100 Korean characters (~+650 English-equivalent words). All additions are framing or new §8.4 section; no existing prose contracted.
- **Sections substantially modified:** Abstract (1 clause sharpened), §1.3 (1 sentence inserted), §1.5 (4a) (1 clause extended), §6.2.1 Insight (1 sentence appended), §8.1 종합 (1 clause sharpened), v7 changelog (new entry).
- **Sections newly added:** §8.4 후속 작업 (load-bearing follow-up) — 6 ranked items, ~600 Korean characters.
- **Sections fully rewritten:** None.
- **Tables modified:** None.
- **Figures modified:** None.
- **Numbers changed:** None.

**Single most impactful edit:** **§8.4 new section with γ-β residual-stream bridge as lead item.** The bar-raiser explicitly framed this as "the single experiment that, if added, would tip my judgment from 'Solid Findings, top of band' to 'Main, lower half of accepted.'" By giving it its own home in the paper (rather than burying as one bullet in §8.2 deferred-list), R5 commits the paper to the elevation path the bar-raiser identified. The four-callsite predict-then-verify framing sharpening (Edit 1-4) is the second-most impactful — locks in the bar-raiser's identified 5-year-citable finding as the paper's *theoretical* contribution rather than a §6.4 result statement.

**Final tier verdict (honest, post-R5):** **Solid Findings, top of band — borderline / weak-accept Main contingent on §8.4 item 1 (bridge experiment) landing positive in next revision.** This matches the bar-raiser's own verdict: "weak accept Findings without hesitation; borderline Main with a clear note that the bridge analysis is what separates this paper from the upper half of accepted Main." R4 honest verdict ("closer to strong Findings than weak-accept Main") and R5 honest verdict converge on the same anchor. The R5 revision (a) honors the bar-raiser's protective duty (7-item protect-list untouched), (b) elevates the §5.2 → §6.4 chain to its named citable-finding role across 4 callsites, (c) commits the paper to the bar-raiser's signature ask as a forward-looking item with explicit GPU-hour accounting. Whether the paper clears Main acceptance ultimately depends on whether the bridge experiment is run and lands positive before the venue submission date.
