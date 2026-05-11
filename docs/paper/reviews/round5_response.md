# Round 5 — Reviser Response (Bar Raiser)

**Paper version BEFORE:** 833 lines (post-Round-4 revision; commit 4d2392e + local working tree)
**Paper version AFTER:** 837 lines
**Date:** 2026-05-11
**Reviewer round addressed:** `docs/paper/reviews/round5_bar_raiser.md`

---

## Answer to the bar-raiser's one-question

**The thesis sentence — verbatim — that now opens Abstract, §1.5, and §8.1 종합:**

> *"Vision-modality bias의 deployable mitigation은 causal pathway를 confounding scene variance로부터 분리하는 paired-inpaint calibration contrast 위에 구축할 수 있으며, 본 논문은 이 design pattern을 cross-modal numerical anchoring 위에 4-clause free-lunch worked example로 instantiate한다."*

Translated for the non-expert / three-years-later reader: *"To mitigate vision-modality biases reliably, build the calibration step on a paired contrast (same scene, only the causal pixels swapped) so the intervention isolates the causal pathway from confounding scene variance — we demonstrate this design pattern on cross-modal numerical anchoring as a 4-clause free-lunch worked example."*

### Of the six numbered claims:

| Claim | Original framing | Post-revision role |
|---|---|---|
| (4) **(a − m) paired-inpaint calibration design pattern** | §6.2.1 Insight — subordinate as "supporting finding (iii) under E6" | **THESIS** — central contribution |
| (5) E6 4-clause free-lunch on `llava-onevision-qwen2-7b-ov` | §1.5 central contribution | **Worked example / proof of construction** for the thesis |
| (1) Anchoring is a graded continuous-confidence phenomenon | §4.1/§4.4/§4.5 | **Behavioural-layer evidence** licensing the thesis (the (a − m) gap demonstrates the design pattern's paired-isolation assumption holds at behavioural axis) |
| (2) Effect is gated by digit-pixel × uncertainty conjunction | Abstract / §4.2 / §1.2 | **Behavioural-layer evidence** — the (a − m) gap directly substantiates the design pattern's "scene-isolated paired contrast" structure |
| (3) Routing-vs-integration framework | §5.4 / §1.3 — co-equal claim | **Mechanism-layer evidence** for the design-pattern's *implementation choice*: multi-direction (not single-direction) subspace projection |
| (6) Reasoning amplifies anchoring (γ-β) | §1.4 standalone subsection + Abstract + §4.5 + §8.1 + §8.2 | **Auxiliary observation** — not supporting evidence for the thesis; separate observation about anchoring's reasoning-trace dependence (§1.4 compressed to forward-pointer; Abstract / §8.1 explicitly separate it from the thesis evidence stack) |

The thesis sentence does not claim "this design pattern transfers to other biases" — that is a *field-level question* surfaced in the new §8.1 Implications paragraph, properly hedged as future verification. The thesis claims (i) the design principle *exists*, (ii) we *instantiate* it as one worked example with formal 4-clause verification. Both parts are defended by current evidence; cross-bias-class and cross-architecture transfer remain deferred (§8.4 items 1, 3, 4).

---

## Decision on the design-pattern-as-thesis proposal: **Option A (full adoption)**

The bar-raiser identified Option A explicitly as the highest-leverage rewrite and labeled it the move that takes the paper from "bottom-half Findings" to "top-third Findings." The user's standing concern ("experiment-log feel") is fundamentally addressed only by Option A — Option B (equal-weight second contribution) preserves the multi-headline confusion the bar-raiser diagnosed; Option C (defer to §8.4) is the exact "competent log" failure mode flagged.

Three reasons not to elevate claim (3) (routing-vs-integration) as the thesis even though the bar-raiser mentions it as an alternative:

1. The paper itself labels (3) as "post-hoc synthesis" with only "partially prospective" verification (K=1 confirmed, K=8 partial-falsified). Elevating a framework the paper hedges as post-hoc invites a Round-6-equivalent shellacking.
2. Claim (4) is the move that *widens the contribution surface beyond anchoring*; claim (3) stays within the anchoring mechanism narrative.
3. The bar-raiser's "citable in 5 years" paragraph names (4) explicitly: *"Choi et al. 2026 introduced paired-inpaint calibration contrasts that isolate causal pathways from confounding scene variance for vision-modality bias mitigation."*

---

## Edit log

### Edit 1 — §1.5 first paragraph: full rewrite to thesis-first structure

**Lines affected:** `emnlp_draft_ko.md` lines 37–43 (replaced original §1.5 first + auxiliary paragraphs).

**Before (original first sentence + structure):**
> "본 논문의 단일 *central contribution*은 **multi-direction subspace projection을 사용하는 cross-modal anchoring mitigation (E6) — `llava-onevision-qwen2-7b-ov` 위 단일 architecture case study** 로, 형식 정의된 *4-clause free-lunch* 기준 ... 본 mitigation은 세 *supporting finding* 위에 build된다 — (i) ... (ii) ... (iii) calibration substrate로서 (a − m) digit-pixel paired-inpaint contrast — CAA-style paired-contrast paradigm에 vision-modality specific *인과 통로 분리* 구조를 도입하는 generalisable design pattern (§6.2.1)."

**After (new first sentence + thesis-first structure):**
> "**Vision-modality bias의 deployable mitigation은 causal pathway를 confounding scene variance로부터 분리하는 paired-inpaint calibration contrast 위에 구축할 수 있으며, 본 논문은 이 design pattern을 cross-modal numerical anchoring 위에 4-clause free-lunch worked example로 instantiate한다.** 본 논문의 *central contribution*은 이 design pattern과 그 첫 instantiation이다 — 즉 (i) §6.2.1의 (a − m) paired-inpaint calibration contrast가 *bias mitigation의 일반 design principle*로서 발화되며 ... (ii) 이 design pattern을 cross-modal numerical anchoring에 instantiate한 worked example이 `llava-onevision-qwen2-7b-ov` 단일 architecture 위 E6 — *proof of construction*으로서 ... **Figure 3이 design pattern의 두 직교 slice (cross-model + cross-dataset (a − m) digit-pixel causality) 를 한 panel로 carry하는 canonical figure이다.**"

**Rationale:** Thesis sentence is verbatim first sentence. (a − m) paired-inpaint is hoisted from "supporting finding (iii)" to *the* central contribution; E6 is reframed as worked example / proof of construction (case-study hedge stays because under design-pattern framing it becomes a *feature*, not a scope-honesty patch — one instantiation by definition). Supporting findings restructured: (i) §4 behavioural evidence *licenses* the paired-isolation assumption, (ii) §5 mechanism evidence *justifies* multi-direction over single-direction, (iii) evaluation protocol *makes the (a − m) contrast measurable*. Figure 3 explicitly designated canonical figure. Auxiliary observation paragraph rewritten to make γ-β reasoning-amplification *not* part of the thesis-evidence stack.

### Edit 2 — Abstract: thesis-first opening + content resequenced around it

**Lines affected:** `emnlp_draft_ko.md` line 9 (single-paragraph Abstract, ~290 words preserved).

**Before:** Opens with "본 논문은 vision-language model (VLM)이 질문과 무관한 두 번째 이미지에 그려진 단일 숫자에 의해 수치 응답에 체계적 편향을 받는 현상 — **cross-modal numerical anchoring** — 을 6개 open-weight VLM에서 보고한다."

**After:** Opens with the thesis sentence verbatim, then existing content (anchoring phenomenon, graded pull, digit-pixel × uncertainty conjunction, mechanism evidence, two mitigations, capability preservation, auxiliary observation) resequenced as *substrate for the worked example*.

**Rationale:** The bar-raiser's #1 load-bearing gap was "three different leads in Abstract / §1.5 / §8.1." Abstract now leads with the same sentence as §1.5 and §8.1. All numbers preserved verbatim — only resequencing, no fabrication. Word count ~290 (essentially unchanged).

### Edit 3 — §8.1 종합: thesis-first opening + 3-layer evidence stack + Implications paragraph

**Lines affected:** `emnlp_draft_ko.md` lines 462–466 (paragraph replaced + new "Implications" paragraph added).

**Before:** Opens with "Cross-modal numerical anchoring은 VLM에서 실재하며, **두 gate의 conjunction + plausibility 조건** ... 으로 정의된다."

**After:** Opens with the thesis sentence verbatim, then evidence stack reorganised into three labeled layers:
1. *행동 layer (§4)* — anchoring exists + graded gradient + (a − m) gap (the third item explicitly demonstrates the thesis's paired-isolation assumption holds at behavioural axis).
2. *Mechanism layer (§5)* — multi-layer redundancy + routing-vs-integration synthesis + §4.6 K=1 partially-prospective verification (this evidence *justifies the design choice* of multi-direction subspace projection in the instantiation).
3. *Worked example layer (§6 / §7)* — E6 as proof of construction with all the existing numbers (Δem(b) Bonferroni-clean, capability preservation +0.41 pp, etc.).

γ-β reasoning amplification explicitly labeled "auxiliary observation" and separated from the thesis-evidence stack.

**New paragraph added — §8.1 Implications.** Bar-raiser secondary ask #2 ("§8.1 종합 should end with one paragraph titled Implications"). Three field-level questions, mapped to §8.4 follow-up items but reframed as *field questions* not *author follow-ups*:
1. (a − m) paired-inpaint transferability to *other vision-modality biases* (familiar-subject counting / sycophancy / position bias / OCR-attack) — opens the design space beyond anchoring.
2. Operational hyperparameter universality: does spectral rank elbow predict K cross-architecture? (cites §4.6 K=1 vs K=8 9× ratio partial-falsification, points to §8.4 item 1.)
3. Cross-architecture worked-example transfer: does the calibration → SVD → projection 3-step pipeline generalise as automatable design recipe?

Closing sentence: "본 논문의 worked example이 thesis의 *유일한 instantiation*이라면 일반화는 가설; cross-bias-class transfer 또는 cross-architecture transfer가 같은 4-clause shape를 산출한다면 design pattern은 *generalisable substrate*로 hardening된다."

**Rationale:** This is the bar-raiser's headline checklist gap — "§8.1 is 'what we did,' not 'what this means.'" The Implications paragraph says explicitly what the field should ask if the thesis is correct. It surfaces three follow-up questions that already exist as §8.4 items but reframes them so the closing voice is "if this is right, the field should now think differently about X" — not "we plan to do X next."

### Edit 4 — §1.4 compression: collapsed to forward-pointer

**Lines affected:** `emnlp_draft_ko.md` lines 33–35 (4-line standalone paragraph compressed to 2 lines).

**Before:** Full paragraph explaining the γ-β result (Qwen3-VL Instruct vs Thinking, ×12.7 ratio, framework alignment) — one of five competing-centre appearances of claim (6) that the bar-raiser flagged.

**After:** 2-line forward-pointer noting auxiliary-observation status, pointing to §4.5 for detail and §8.2 for limits. Subsection title amended to "(auxiliary observation 요약)".

**Rationale:** Bar-raiser secondary ask #3 — "auxiliary observation should be one short sentence in Abstract or demoted to one paragraph in §4.5 only." The §4.5 detail stays; the Abstract reference stays (one sentence each in Abstract + §8.1 + §4.5 + §8.2); §1.4 standalone subsection compressed to forward pointer, reducing the dragging of an N=1 × N=1 existence proof on the thesis centring.

### Edit 5 — CHANGELOG: log v11 (Round-5 bar-raiser response)

**Lines affected:** `docs/paper/CHANGELOG.md` (v11 entry prepended).

**Content:** Documents Option A adoption, thesis sentence three-site insertion, §1.5/Abstract/§8.1 rewrites, §1.4 collapse, Figure 3 canonical designation, Implications paragraph addition. Notes "No new experiments / numbers / figures / tables introduced."

### Table edits

None. All tables preserved exactly. All numbers preserved verbatim.

### Figure edits

**Figure 3 — designated as canonical figure** in §1.5. No image edits, no caption changes — only the textual designation in §1.5 ("Figure 3이 design pattern의 두 직교 slice (cross-model + cross-dataset (a − m) digit-pixel causality) 를 한 panel로 carry하는 canonical figure이다"). Bar-raiser secondary ask #1 ("one figure that is the paper") closed via designation rather than creation; the existing Figure 3 already carries the design pattern at a glance (PlotQA × 6-model + OneVision × 5-dataset two-slice panel of (a − m) digit-pixel causality).

---

## New rehearsed take-away sentence

**Verbatim Korean:**

> "Vision-modality bias의 deployable mitigation은 causal pathway를 confounding scene variance로부터 분리하는 paired-inpaint calibration contrast 위에 구축할 수 있으며, 본 논문은 이 design pattern을 cross-modal numerical anchoring 위에 4-clause free-lunch worked example로 instantiate한다."

**Locations (verified via grep `Vision-modality bias의 deployable mitigation`):**

| Location | Line | Status |
|---|---|---|
| Abstract first sentence | line 9 | verbatim |
| §1.5 first sentence | line 39 | verbatim |
| §8.1 종합 first sentence | line 464 | verbatim |

The same sentence appears nowhere else in the paper (no accidental fourth-mention drag).

---

## Canonical figure designation

**Figure 3** (`docs/figures/paper_4_2_digit_pixel_causality.png`) — designated in §1.5 as *"design pattern의 두 직교 slice (cross-model + cross-dataset (a − m) digit-pixel causality) 를 한 panel로 carry하는 canonical figure."*

**Why this figure, not a new one:**

- Figure 3 already shows the (a − m) paired-inpaint contrast — the *design pattern itself* — across two orthogonal slices (PlotQA × 6-model panel from E7; LLaVA-OneVision × 5-dataset panel from E5b/E5e).
- It is the one figure in the paper that visualises the thesis's load-bearing experimental construct (paired-inpaint causal-pathway isolation) rather than a downstream consequence (mitigation Δdf).
- The bar-raiser's exact ask: "one figure you can show on a slide and say 'this is the paper'." Figure 3 *is* that figure once designated.

**On the bar-raiser's separate suggestion** (a new figure visualising E6 Table 7 — 5 datasets × Δdf, Δem(b) bars): we did *not* create this. Rationale: under the new thesis framing, Table 7 captures the *worked example's verification* of the design pattern. The canonical figure should depict the *design pattern* (Figure 3), not the worked example's quantitative verification (Table 7). The bar-raiser asked for "one figure that is the paper"; under the thesis framing, the paper *is* the (a − m) design pattern, which Figure 3 captures. Creating a new mitigation-visualization figure would re-centre the visual attention on the worked example. Acknowledged as a possible camera-ready addition if the worked-example layer needs separate visual support; not done in this revision.

---

## Top-decile-readiness checklist (after edits)

| Axis | Before R5 | After R5 |
|---|---|---|
| Central question that the field cares about for ≥ 3 years | **Partial** — deeper question (transferable calibration substrate) was sub-claim, not headline | **Closed** — thesis sentence now opens with "vision-modality bias의 deployable mitigation은 ... paired-inpaint calibration contrast 위에 구축할 수 있"; §8.1 Implications paragraph asks the field-level question explicitly |
| Central contribution that opens a new design space | **No** — design-space-opener subordinated to E6 | **Closed** — (a − m) design pattern is the thesis, E6 is worked example / proof of construction |
| Reproducibility at one-command level | **Yes, at venue norm** — §A.4 FLUX seed, §A.5 27-cell pilot replay, §6.2 raw output paths | **Yes** (unchanged; we did not regress) |
| Single most-important figure | **No** — Table 7 carried the contribution; no figure | **Partial-closed** — Figure 3 explicitly designated canonical figure in §1.5. Bar-raiser also suggested *new* visualisation of Table 7 (5 datasets × Δdf, Δem(b)); not created this round. Closed for the thesis (Figure 3 shows the design pattern); open for the worked example's quantitative summary (no new figure for that) |
| Retrievable lesson — one verbatim sentence repeated in Abstract / §1.5 / §8.1 | **No** (load-bearing gap, three different leads) | **Closed** — same sentence, lines 9 / 39 / 464 |
| Engagement with implications — *what this means for the field* | **No** — §8.1 was recap, not implications | **Closed** — §8.1 종합 now ends with **Implications** paragraph asking three field-level questions, reframed from §8.4 items |

**Score: 5/6 met + 1/6 partial** (vs 2/6 + 1/6 partial before). Per the bar-raiser's rubric: "top decile requires at minimum 5/6" — the paper now sits at threshold on this single rubric.

Per the bar-raiser's *own* paragraph "If the author answers the one-question well: the paper becomes a clean Findings paper with a memorable methodological contribution. With the thesis sentence in place + Secondary ask #1 (one figure that is the paper) + Secondary ask #2 (§8.1 Implications paragraph), the paper moves from bottom-half Findings to top-third Findings." All three of those moves are now in the paper. Top-decile / outstanding-paper still requires §8.4 item 3 (cross-architecture E6) + item 7 (Telea-residue (m − m') baseline), both deferred to pre-camera-ready / next revision; the bar-raiser acknowledged this is beyond this revision cycle's scope.

---

## Internal-consistency check

- [x] **Same thesis sentence verbatim at three locations.** Verified via grep — lines 9, 39, 464.
- [x] **Abstract numbers still match §4-§7 tables.** All numbers preserved verbatim (1.7-15.7 %, +19.5-23.5 pp, +19.0-34.4 pp PlotQA wrong-correct, ×12.7 ratio, Δem(b) 5/5 Bonferroni-clean, +0.41 pp macro, +2.21 pp HallusionBench, −0.06 pp POPE, K=1 14/84 cells, 9× ratio). No fabricated numbers introduced.
- [x] **§1.5 contributions still match §4-§7 deliveries.** §1.5 (i) design pattern → §6.2.1 Insight (unchanged); §1.5 (ii) worked example → §6.2.3 Table 7 (unchanged); supporting evidence (i) → §4; (ii) → §5; (iii) → §3 protocol (unchanged).
- [x] **§8.1 종합 consistent with body.** Three-layer evidence stack maps 1:1 onto §4 / §5 / §6+§7. Worked example numbers (Δem(b) Bonferroni-clean, +0.41 pp macro, HallusionBench excludes zero, POPE pinned) all match §6.2.3 and §7 tables.
- [x] **All figure embeds still resolve.** 16 inline figures unchanged. Figure 3 canonical designation is *textual* in §1.5; no Figure 3 caption or image change.
- [x] **No figure or table renumbering.** Tables 1-9, Figures 1-6 all in same positions.
- [x] **Canonical sources still cited where appropriate.** No reference changes; all citation labels preserved.
- [x] **§1.4 compression coherent with §1.5 + §4.5 + §8.1 + §8.2.** §1.4 now forward-pointer only; §4.5 still carries full γ-β detail; §1.5 auxiliary paragraph still distinguishes γ-β as auxiliary observation not supporting evidence; §8.1 evidence stack explicitly labels γ-β auxiliary; §8.2 reasoning-amplification bullet unchanged.
- [x] **E6 hedge "단일 architecture case study" still present and now *load-bearing*.** Under design-pattern framing, "one instantiation" is what a worked example *is*. The hedge becomes a feature of the thesis structure, not a scope-honesty patch.
- [x] **§8.4 follow-up list intact.** No items added or removed in §8.4 (items 1-9 preserved). §8.1 Implications paragraph cites items 1, 2, 3, 4 by number for cross-reference, but the §8.4 list itself is unchanged.
- [x] **No demoted claims resurrected.** Encoder-family-determines-archetype still demoted at §D.1; "uniquely passes" framing on E6 still removed; all Round-1 through Round-4 hedges preserved (case-study, two-clause Δdf/Δem(b), partially-prospective, Telea-residue caveat, axis-conditional capability disclosure).
- [x] **No fabricated experimental results.** No new numbers, tables, figures, or experiments introduced. Only rhetorical-discipline reframing of contribution hierarchy.

---

## Rebuttals (DISAGREE class)

None. Every substantive bar-raiser intervention was addressed via edit. The bar-raiser's verdict (weak Findings, not Main, not top decile in this revision cycle) is explicitly *consistent* with the post-edit state — the bar-raiser said the thesis-sentence + Implications + canonical figure moves take the paper from bottom-half Findings to top-third Findings, not to Main. We agree; we do not contest the tier verdict.

The bar-raiser's "What I would NOT change" list (two-clause Δdf/Δem(b) separation, partially-prospective framework label, Telea-residue caveat, §7 axis-conditional disclosure, §A.5 27-cell pilot heatmap, PlotQA single-dataset depth framing, γ-β auxiliary-observation qualifier) — all preserved unchanged in this round, as the bar-raiser requested.

---

## Deferred items

| Reviewer point | Reason for deferral | Status |
|---|---|---|
| Cross-architecture E6 replication (§8.4 item 3) | ~3-5 H100-day compute; bar-raiser explicitly noted this is out of scope for the rewrite ("either it is run pre-camera-ready or the paper is Findings") | §8.4 item 3 + reframed in §8.1 Implications as field-level "does the worked-example pattern generalise" question |
| Telea-residue (m − m') baseline (§8.4 item 7) | ~2 H100-hour; bar-raiser kept the §6.2.1 Telea caveat as load-bearing and did not push for in-round resolution | §8.4 item 7 |
| New figure visualising E6 Table 7 (5 datasets × Δdf, Δem(b)) | Bar-raiser secondary ask but not the bar-raiser's load-bearing intervention; ~1 hour matplotlib but would compete with Figure 3 for canonical-figure attention under design-pattern framing | Acknowledged as known gap for camera-ready if worked-example layer needs separate visual support |
| Top-decile-readiness still requires cross-architecture E6 + (m − m') baseline | Bar-raiser explicitly acknowledged ("paper is not in striking distance of top decile in this revision cycle") | Out of scope for this round |

---

## Open questions for next round

None requested. This is the final round of the 5-round loop. The bar-raiser's "one-question" was the entire substantive ask; the thesis-sentence + Implications + Figure 3 designation + §1.4 collapse close that single ask. Any further substantive work is camera-ready / pre-submission revision territory (cross-architecture E6, m − m' baseline, eigenvalue spectrum, paired-bootstrap CI for ×12.7) and is preserved in §8.2 limits + §8.4 follow-up.

---

## Diff summary

- **Lines:** 833 → 837 (net +4). Within the 840-line constraint.
- **Sections substantively edited:**
  - Abstract: paragraph rewritten with thesis-first opening (~290 word length preserved).
  - §1.4: 4-line paragraph compressed to 2-line forward-pointer.
  - §1.5: paragraph rewritten with thesis-first opening + Figure 3 designation + supporting evidence restructured as "licensing" the design pattern.
  - §8.1 종합: paragraph rewritten with thesis-first opening + 3-layer evidence stack labelled (behavioural / mechanism / worked example) + new Implications paragraph (~6 sentences).
  - CHANGELOG.md: v11 entry prepended.
- **Tables edited:** None.
- **Figures edited:** None (Figure 3 designated as canonical via §1.5 text only; image unchanged).
- **References edited:** None.
- **Numbers edited:** None — all numeric content preserved verbatim.
- **New experiments:** None.
- **Total word delta:** ~+250 net (Implications paragraph +~250, §1.4 collapse −~80, §1.5 rewrite ~+80, Abstract rewrite ~0, §8.1 rewrite ~+50 due to evidence-stack labels).
- **Contribution hierarchy:** completely reshaped. (a − m) paired-inpaint design pattern is now thesis; E6 is worked example; routing-vs-integration framework is mechanism evidence for the design choice; γ-β reasoning amplification is auxiliary observation (no longer competing with the thesis for centring).

---

**End of Round 5 Reviser Response. Paper is now ready for camera-ready / pre-submission revision phase pending §8.4 follow-up items (cross-architecture E6 + Telea-residue baseline + eigenvalue spectrum) per bar-raiser top-decile guidance.**
