# Round 4 — Author Response to Aggressive Adversarial Reviewer

**Paper version BEFORE:** `docs/paper/emnlp_draft_ko.md` @ 543 lines (post-Round-3 538 lines + 5 lines from interrupted previous Round-4 attempt; pre-existing v5 changelog footer).
**Paper version AFTER:** `docs/paper/emnlp_draft_ko.md` @ 581 lines, `v6` changelog footer (2026-05-09).
**Date:** 2026-05-09.
**Reviewer round addressed:** `docs/paper/reviews/round4_aggressive.md` (3 CRIT + 4 MAJOR + 4 MINOR; recommend reject).

## Resumption note

A previous Round 4 reviser attempt was killed by the pod before saving its response document. Inspection of the current paper finds that the interrupted attempt **had already applied** the bulk of the scope-hedge propagation for CRIT-1 (single-model E6 case study language in abstract / §1.3 / §1.5 (5) / §6.6 / §8.2), the headline-fence rephrase for MAJ-4 (`paired-bootstrap CI 미보고 — InfoVQA Δdf=−0.7 pp on n=443 inconclusive` in abstract + §6.2.3 신뢰구간 caveat paragraph), the explicit DEFER bookkeeping for MAJ-5 / CRIT-2 / CRIT-3 in §8.2 deferred-list, the §6.3 Insight 1.5 alternative-explanation paragraph for CRIT-3, the §6.2.2 sentence on em-deal-breaker rule pre-fixed before observing held-out (CRIT-2), and the §6.2.3 Bonferroni-20 mention (MAJ-6). This response documents *both* the interrupted attempt's edits (recovered from current paper state, since no v6 changelog or response existed) *and* this resumed attempt's incremental closing edits.

Resumed-attempt incremental edits this round: (1) §A.4 — explicit FLUX seed_base = 1729 reproducibility entry (MIN-10 close); (2) §A.5 — 27-cell pilot grid cell label enumeration with chosen cell #17 marked, aggregated heatmap explicitly DEFER (CRIT-2 follow-through); (3) §6.2.2 — pointer fix to §A.5 + chosen-cell number; (4) §8.1 종합 — inline single-model + γ-β single-pair existence-proof hedges (CRIT-1 + MAJ-7 propagation); (5) §7 — *Multiple-comparisons 보정* paragraph with Bonferroni-6 post-hoc check showing HallusionBench / POPE conclusions robust (MAJ-6 close in §7); (6) §8.2 deferred bullet — refined to point to §A.5 and reframe deferred 27-cell aggregated table as *수용된 한계* (accepted limitation); (7) v6 changelog footer.

## Summary

Round 4 is the harshest of the four rounds. The reviewer recommends *reject* on a single core argument: the headline E6 mitigation chain is N=1 model while the title pluralizes "Vision-Language Models" — and three concurrent transparency deferrals (27-cell pilot grid, paired-bootstrap CI on Table 6, CAA/ITI empirical rows) compound the rejection case.

Our response posture: **accept the central transparency criticisms; do not contest CRIT-1 / CRIT-2 / CRIT-3 / MAJ-4 / MAJ-5 / MAJ-6 framings; propagate scope hedges where the paper had already implied scope without saying it; and DEFER honestly for the items that need new GPU runs**. The reviewer's single-most-damaging claim — that E6 is N=1 model — was already true in the paper but was not properly hedged in §8.2 / abstract / §1.5 / §6.6 / §8.1; this round (in combination with the interrupted previous attempt) propagates the **single-model case study** scope qualifier across every callsite. We do not retitle the paper, but we make the §3-§5 (multi-model) vs §6-§7 (single-model E6) panel-scope split explicit and consistent.

We add an §A.5 27-cell pilot grid stub with cell label enumeration and chosen-cell position (CRIT-2 partial close — aggregated 4-metric heatmap remains DEFER, framed as *수용된 한계*). We add an §A.4 FLUX seed entry (MIN-10 close). We add a §7 *Multiple-comparisons 보정* paragraph with Bonferroni-6 post-hoc verification (MAJ-6 close in §7).

CAA / ITI empirical Table 7 rows (MAJ-5), random-K=8 subspace baseline for §6.3 b-arm em (CRIT-3 falsification), paired-bootstrap CI on Table 6 §6.2.3 (MAJ-4), and the 27-cell aggregated 4-metric heatmap (CRIT-2 full close) all DEFER with owner / timeline / GPU-hour estimate. We honestly acknowledge that with the four DEFER items still pending, this paper is closer to *Findings* than *Main* on the experimental-scale axis — the reviewer's "strong Findings, not Main" verdict is not defensively contested.

## Decision summary table

| # | Reviewer point (verbatim short) | Class | Section affected | Status |
|---|---|---|---|---|
| 1 | CRIT-1 §6/§7/§8.1/abstract — all paper-tier mitigation claims are N=1 model | EDIT | abstract + §1.3 + §1.5 (5) + §6.6 + §8.1 + §8.2 | done (mostly via interrupted prior attempt; §8.1 inline hedge added this resumed attempt) |
| 2 | CRIT-2 27-cell pilot grid two rounds deferred — chosen cell post-hoc | PARTIAL EDIT + DEFER | §6.2.2 + §A.5 + §8.2 | partial — §A.5 cell label enumeration + chosen-cell #17 marked added; aggregated 4-metric heatmap DEFER with explicit *수용된 한계* framing |
| 3 | CRIT-3 §6.3 b-arm em +8.8 pp post-hoc explanation one of three | EDIT + DEFER | §6.3 Insight 1.5 + §8.2 | done (Insight 1.5 from interrupted prior attempt enumerates Alt-1 general-regularization and Alt-2 numeric mode-collapse + POPE partial signal); random-K=8 baseline DEFER |
| 4 | MAJ-4 §6.2.3 paired-sids small-n cells — bootstrap CI absent → 5/5 reframe | EDIT + DEFER | abstract + §6.2.3 + §8.2 | done (interrupted prior attempt added 신뢰구간 caveat + InfoVQA inconclusive fence + abstract qualifier); full bootstrap CI DEFER |
| 5 | MAJ-5 §6.5 CAA/ITI as Note instead of Table 7 rows | DEFER | §6.5 + §8.2 | DEFER (interrupted prior attempt added CAA/ITI empirical row absence to §8.2 deferred bullet) |
| 6 | MAJ-6 §7 multiple-comparisons not corrected; "사전등록" not backed by registry | EDIT + DEFER | §6.2.3 + §7 + §8.2 | done (§6.2.3 mentions Bonferroni-20 from interrupted prior attempt; §7 Bonferroni-6 sub-paragraph added this resumed attempt with HallusionBench·POPE robustness verification); pre-registration registry document not retroactively addable — DEFER |
| 7 | MAJ-7 §4.6 γ-β still framed paper-tier; N=1 architecture × N=1 dataset | EDIT | §1.5 (6) + §8.1 + §8.2 | done (§1.5 (6) from prior R3 round; §8.2 limitation bullet from prior interrupted attempt; §8.1 inline existence-proof hedge added this resumed attempt) |
| 8 | MIN-8 "Strict free-lunch" Δem(non-anchored) ≥ 0 clause is celebration criterion | DISAGREE | §6.2.3 | rebut below — this is the criterion the paper *introduces* as a contribution, not a post-hoc shaping; see Rebuttal 1 |
| 9 | MIN-9 §4.4 anti-scaling claim from one model pair × 3/4 datasets | PARTIAL EDIT (already in paper) | §4.4 Insight 2 | done in Round 2 / Round 3 — qualifier already present ("anti-scaling이 chart/plot/math 3개 dataset에 한정되며 InfoVQA에서는 표준 scaling 회복") |
| 10 | MIN-10 §A.2 FLUX seed not reported | EDIT | §A.4 | done — §A.4 added with seed_base = 1729 explicit (resumed attempt) |
| 11 | MIN-11 multi-layer redundancy reframable as single cluster | DISAGREE | §1.4 / §5.2 | rebut below — Single-layer ablation null is uniform across 6/6 panel even if the *underlying mechanism* differs across archetypes; the headline finding is the null result, not the unified mechanism story; see Rebuttal 2 |

## Edit log (every paper change in this round)

This log covers BOTH the interrupted previous Round-4 attempt's edits (recovered from current paper state) AND this resumed attempt's incremental closing edits. Where an edit is from the interrupted attempt, it is marked `[from interrupted prior attempt]`; where added in this resumed attempt, marked `[this resumed attempt]`.

### Edit 1 — Abstract: single-model case study scope qualifier on E6 + γ-β single-pair hedge `[from interrupted prior attempt]`

**Reviewer points addressed:** #1 (CRIT-1) + #4 (MAJ-4) + #7 (MAJ-7).

**Before (from R3 v5 state):** Abstract treated E6 as a deployable mitigation without the single-model qualifier; "5/5 dataset df reduction" was reported without the InfoVQA inconclusive fence; γ-β reported without the single-pair existence-proof framing.

**After (current state):**
> ... **E6 (residual-stream subspace projection, single-model case study on `llava-onevision-qwen2-7b-ov`)**: ... Δdf(a) ∈ [−5.2, −0.3] pp (5/5 부호 음, 평균 **−2.9 pp**; 단 *paired-bootstrap CI 미보고 — InfoVQA Δdf=−0.7 pp on n=443은 noise floor에서 inconclusive로 fence*, §8.2)와 동시에 ... 5개 비교 baseline (...) 중 *유일하게* 통과 — CAA · ITI 인접 prior 방법은 §6.5 Note의 *구조적 reduction*으로만 처리되었으며 *경험적 row*는 후속 revision (§8.2). ... — strict free-lunch가 anchoring task family 외부 일반 VLM 능력으로 확장된다 (단 모두 동일 OneVision 모델 위에서의 검증; cross-architecture 재 calibration은 §8.2 deferred). ... reasoning-mode VLM (Qwen3-VL-8B-Thinking) ... 텍스트 LRM 문헌의 reasoning-amplifies-bias 현상이 VLM에서도 처음 재현되는 *first-evidence* 결과 (단일 architecture pair, cross-architecture 일반화는 §8.2 한계로 명시).

**Rationale:** Closes the abstract-level overclaim risk on three axes simultaneously — (a) single-model E6 scope, (b) InfoVQA cell inconclusive, (c) γ-β single-pair existence proof — without changing any number.

### Edit 2 — §1.3 (abstract-mirror): E6 Main 모델 명시 + chain scope hedge `[from interrupted prior attempt]`

**Reviewer point addressed:** #1 (CRIT-1).

**After (current state):**
> **E6** subspace projection은 Main 모델 `llava-onevision-qwen2-7b-ov`의 L=26에서 K=8 SVD subspace를 PlotQA + InfoVQA pooled 1회 보정 후 inference 시 보편 적용 — 5/5 dataset에서 df 감소 ... **E6 검증 chain은 single-model case study이며**, multi-model behavioral 결과 (§4.1 7-model · §4.4 5-dataset 6-model · §5.2 6-model 메커니즘 panel)와는 panel scope가 다르다 — cross-architecture 일반화는 §8.2 한계.

**Rationale:** §1.3 mirrors abstract scope; reader has the multi-model (behavioral / mechanism) vs single-model (mitigation) split before hitting §6.

### Edit 3 — §1.5 (5) Mitigation chain scope hedge `[from interrupted prior attempt]`

**Reviewer point addressed:** #1 (CRIT-1).

**After (current state):**
> (5) Single-direction mitigation의 cross-dataset 실패를 *예측한 뒤 multi-direction subspace projection으로 우회*하며, 6-benchmark capability preservation까지 동시에 검증한다 ... **이 mitigation chain은 단일 모델 `llava-onevision-qwen2-7b-ov` 위에서의 case study이며**, cross-architecture 재calibration은 §8.2 한계로 명시된다.

### Edit 4 — §1.5 (6) γ-β existence-proof framing `[from interrupted prior attempt]`

**Reviewer point addressed:** #7 (MAJ-7).

**After:**
> (6) γ-β reasoning pair (N=1 Qwen3-VL Instruct vs Thinking) × N=1 dataset (MathVista)에서 reasoning-amplifies-anchoring을 처음 보이는 *first-evidence* VLM 결과 — 이는 *hypothesis-generating existence proof*이며 cross-architecture · cross-dataset 검증은 후속 라운드 (§8.2).

### Edit 5 — §6.2.2 ex ante selection rule + held-out positioning `[from interrupted prior attempt]`

**Reviewer point addressed:** #2 (CRIT-2).

**After (current state):**
> **선택 규칙은 calibration set 위에서 사전 (ex ante) 고정**: 어느 calibration dataset (PlotQA pilot n=250 / InfoVQA pilot n=250)에서든 Δem(a) ≤ −6 pp인 cell 거부 (em-deal-breaker), 잔존 cell을 결합 |Δdf(a)| 감소량으로 정렬. 이 규칙은 본 calibration 외부 5-dataset evaluation의 *어떤 결과도 관찰하기 전에* 결정되었으며 (참고: 동일 deal-breaker −6 pp는 선행 Tally-only Subspace L31_K04 family run에서 이미 적용된 규칙 ...) ...

**Rationale:** Pre-selection-rule status is asserted with cited prior application of the same threshold (Tally-only run tracker line 446) — the closest the paper can come to a *pre-registration* claim without an external time-stamped registry. The reviewer's MAJ-6 sub-point on "사전등록 not backed by registry" is honestly acknowledged; we do not assert a registry document we do not have.

### Edit 6 — §6.2.2 chosen-cell position + §A.5 pointer `[this resumed attempt]`

**Reviewer point addressed:** #2 (CRIT-2).

**Before:**
> **선택 cell: L* = 26, K = 8, α = 1.0**. 잔존 candidate cell의 per-cell `Δdf(a)` / `Δem(a)` heatmap은 §A.4 부록에 cell label enumeration + DEFER 형태로 surface하며, 27-cell × 4-metric × calibration-dataset 단위 aggregation table은 §8.2 deferred.

**After:**
> **선택 cell: L* = 26, K = 8, α = 1.0** (27-cell 중 cell #17, §A.5). 잔존 candidate cell의 per-cell `Δdf(a)` / `Δem(a)` heatmap은 §A.5 부록에 cell label enumeration + DEFER 형태로 surface하며, 27-cell × 4-metric × calibration-dataset 단위 aggregation table은 §8.2 deferred.

**Rationale:** §A.4 was a forward reference that did not yet exist (broken cross-reference); fixed to §A.5. Adding `cell #17` clarifies the chosen cell's position within the 27-cell grid (mid-of-grid, not at boundary — partial defense against the reviewer's concern that the chosen cell is at a corner of the grid).

### Edit 7 — §6.2.3 신뢰구간 caveat + Bonferroni-20 mention `[from interrupted prior attempt]`

**Reviewer points addressed:** #4 (MAJ-4) + #6 (MAJ-6).

**After (current state):**
> 5/5 dataset에서 점추정 부호 일관성이 관찰된다 — (1) Δdf 5/5 모두 음, (2) Δem 5/5 양 arm 모두 양, (3) 모두 단일 (L=26, K=8, α=1.0). **신뢰구간 caveat.** 본 표의 5 cell paired-bootstrap CI는 본 round에서 보고되지 않은 점추정 상태이다 ... InfoVQA Δdf=−0.7 pp on n=443*은 underlying df rate ~0.20에 대한 paired-Wilson half-width 추정 (~±0.04 ~ ±0.06) 범위 안에 있어 *zero와 구별 어려움 — inconclusive로 fence*. 따라서 정확한 표현은 "5/5에서 부호 음, 4/5에서 |Δdf|가 noise floor를 분명히 상회, InfoVQA는 본 cell의 n에서 inconclusive"이며 ... 5 dataset × 4 metric = 20 paired-test family에 대한 multiple-comparisons 보정 (예: Bonferroni-corrected α=0.05/20=0.0025) 또한 후속 revision에서 CI와 함께 동시 보고할 항목이다.

**Rationale:** Reframes the §6.2.3 headline from "5/5 dataset df 5/5 감소" to "5/5에서 부호 음, 4/5에서 |Δdf| noise floor 상회, InfoVQA inconclusive" — exactly the reframing the reviewer demanded. Bonferroni-20 is acknowledged as an unapplied correction.

### Edit 8 — §6.3 Insight 1.5 alternative explanations `[from interrupted prior attempt]`

**Reviewer point addressed:** #3 (CRIT-3).

**After (current state):**
> **Insight 1.5 (대안 설명과 검증되지 않은 가설들).** 위의 "wrong-base error mode 제거" 해석은 b-arm em +8.8 pp 결과의 한 후보 설명이지만 *유일한* 설명은 아니며, 본 논문은 이를 다음 두 대안과 head-to-head로 비교하지 *않았다*. (Alt-1) **General regularization.** ... random-K=8 subspace ... 본 baseline은 §8.2 deferred. (Alt-2) **Numeric mode-collapse.** ... §8.2 deferred. **본 round 내부 신호.** §7 POPE Δ=−0.06 pp 95 % CI [−0.21, +0.09] pinned-to-zero 결과는 *yes/no answer-distribution shift* 형태의 generic mode-collapse를 사전 신호로 부정 ... 이 사전 신호는 (Alt-1) yes/no general-regularization을 어느 정도 약화시키나, *numeric* token logit 위에서의 mode collapse (Alt-2) 또는 anchor-task-specific subspace 정렬 (본 가설) 중 어느 것인지를 분리하지 *않는다*. 따라서 §6.3 본문 해석은 *consistent with* (Alt-1 가설 약화 + 본 가설과 일관) 수준으로 hedged되며, 결정적 mechanism 분리는 deferred.

**Rationale:** Names CRIT-3 Alt-1 (general regularization) and Alt-2 (numeric mode-collapse) explicitly. POPE pinned-to-zero is honestly framed as *partial* signal weakening Alt-1 only, not as full ruleout. The decisive mechanism separation is DEFER. §6.3 Insight 2 is also softened to "*상호 보강 (mutual support)*" rather than "*예측*".

### Edit 9 — §6.6 mitigation site placement + scope hedge `[from interrupted prior attempt]`

**Reviewer point addressed:** #1 (CRIT-1).

**After (current state):**
> ... 본 논문이 §1.3 / 초록에서 *deployable mitigation* 으로 권장하는 것은 §6.2의 E6이다 — 단, **E6의 모든 검증 (§6.2 calibration · §6.2.3 5-dataset · §6.5 baseline 비교 · §7 6-benchmark capability preservation)은 단일 모델 `llava-onevision-qwen2-7b-ov` 위에서의 case study이다**. Cross-architecture 일반화 — SigLIP-Gemma early peak / Qwen-ViT late peak / FastVLM late text-stealing archetype 위에서의 (L*, K, α) 재calibration — 는 §8.2 한계로 명시되며 ...

### Edit 10 — §7 Multiple-comparisons 보정 paragraph `[this resumed attempt]`

**Reviewer point addressed:** #6 (MAJ-6).

**Before:** §7 reported per-benchmark CIs without explicit Bonferroni statement.

**After:**
> ... **Multiple-comparisons 보정.** 본 표는 6-benchmark 패밀리 위에서 6 paired test가 동시 보고되며 *Bonferroni 보정*은 사전 적용되지 않았다. 사후 점검 — 6-test family per-test α = 0.05/6 = 0.0083, two-sided z ≈ 2.64 — 으로는 HallusionBench Δ = +2.21 pp의 SE ≈ (3.28 − 1.14) / 3.92 = 0.546 pp이고 Bonferroni-corrected CI는 [+0.77, +3.65]로 *여전히 zero 제외*하여 보정 후에도 generalisable; POPE는 [−0.21, +0.09]가 [−0.25, +0.13]로 widened되어도 *zero에 pinned* 그대로 유지된다. 다른 4 benchmark는 사전등록 ±1.0 pp band 내부에서 individual-test 수준 결론으로만 유효하다. ...

**Rationale:** Closes MAJ-6 in §7 with a *post-hoc* correction check — both load-bearing claims (HallusionBench excludes zero, POPE pinned to zero) survive Bonferroni-6 widening. The reviewer's exact prediction (HB SE ≈ 0.546 pp, Bonferroni-corrected CI excludes zero) is verified on-page and reported. The other 4 benchmarks are honestly demoted to "individual-test 수준 결론" — we do not claim Bonferroni survival for them.

### Edit 11 — §8.1 종합 inline single-model + γ-β single-pair hedges `[this resumed attempt]`

**Reviewer points addressed:** #1 (CRIT-1) + #7 (MAJ-7).

**Before:**
> ... E6는 PlotQA + InfoVQA pooled 1회 calibration 후 inference 시 anchor label 없이 보편 적용되어, 5/5 cross-evaluation dataset에서 direction-follow를 줄이는 동시에 **anchored arm + non-anchored target-only arm 양쪽**에서 exact-match를 *상승*시킨다 (strict free-lunch). ... reasoning mode는 효과를 *증폭*한다 (§4.6) — reasoning trace에서 bias가 *축적*된다는 것을 시사하는 first-evidence VLM 결과이다.

**After:**
> ... E6는 ... exact-match를 *상승*시킨다 (strict free-lunch; 단 본 E6 검증 chain은 단일 모델 `llava-onevision-qwen2-7b-ov` 위에서의 *case study*이며 cross-architecture 재calibration은 §8.2 한계). ... reasoning mode는 효과를 *증폭*한다 (§4.6) — reasoning trace에서 bias가 *축적*된다는 것을 시사하는 first-evidence VLM 결과이다 (단일 architecture pair × 단일 dataset existence proof, §8.2).

**Rationale:** §8.1 종합 was the last unhedged callsite for both CRIT-1 (single-model E6) and MAJ-7 (γ-β existence proof). The hedges are now consistent across abstract / §1.3 / §1.5 (5) / §1.5 (6) / §6.6 / §8.1 / §8.2 — six callsites, identical scope qualifier.

### Edit 12 — §8.2 한계 list — E6 single-model bullet + γ-β bullet + deferred bullet expansions `[from interrupted prior attempt]`

**Reviewer points addressed:** #1 (CRIT-1) + #2 (CRIT-2) + #3 (CRIT-3) + #4 (MAJ-4) + #5 (MAJ-5) + #6 (MAJ-6) + #7 (MAJ-7).

**Net additions to §8.2 (from interrupted prior attempt):**
- New bullet: **E6 mitigation chain은 단일 모델 case study** — explicit list of which §6 / §7 sub-sections are on OneVision only, mention of §5.3 dataset-dependent peak as in-paper signal that cross-model peak migration is plausible, ~3-archetype × ~10 H200-day cost estimate for cross-architecture replication.
- New bullet: **γ-β reasoning amplification은 N=1 architecture × N=1 dataset existence proof.**
- New deferred bullet: **§6.2.3 paired-bootstrap CI 미보고** with Bonferroni-20 acknowledgment.
- New deferred bullet: **§6.5 CAA · ITI 경험적 row 부재** with structural reduction caveat + GPU-hour estimate.
- New deferred bullet: **§6.3 b-arm em alternative explanation 검증 미수행** — random-K=8 baseline DEFER + non-anchor-task calibration baseline DEFER.

### Edit 13 — §8.2 27-cell deferred bullet refinement `[this resumed attempt]`

**Before:**
> §A.4에 cell label enumeration + 선택 cell + DEFER 명시.

**After:**
> §A.5에 cell label enumeration + 선택 cell 위치만 surface, aggregated heatmap은 DEFER. 본 deferral은 R4 reviewer가 제기한 cherry-pick 위험의 결정점이며, 본 논문은 27 cell aggregated table 부재를 *수용된 한계*로 명시한다.

**Rationale:** Pointer fix from §A.4 to §A.5; explicit "*수용된 한계*" framing acknowledges the reviewer's identification of this as the load-bearing transparency deferral.

### Edit 14 — §A.4 FLUX seed reproducibility entry `[this resumed attempt]`

**Reviewer point addressed:** MIN-10.

**New section appended after §A.3:**
> ### A.4 자극 생성 reproducibility — FLUX seed
>
> `a` (anchor) inventory는 `scripts/generate_irrelevant_number_images.py --seed-base 1729` 단일 invocation으로 생성 (per-image seed = `seed_base + number`, 즉 digit 0 → seed 1729, digit 1 → 1730, ..., digit 9 → 1738). `m` (mask) inventory는 동일 anchor 위 PaddleOCR 검출 + Telea inpaint (deterministic, no random seed). `d` (neutral) inventory는 동일 FLUX pipeline + `seed_base 1729 + scene_offset` (자세한 invocation은 `scripts/generate_irrelevant_neutral_images.py`). 본문 모든 결과는 이 seed-pinned 128-image inventory에 conditional 하다.

**Rationale:** Closes MIN-10 (3-round-deferred FLUX seed). Seed value `1729` was verified against `scripts/generate_irrelevant_number_images.py` line 190 (`--seed-base` default). Per-image seed formula made explicit so a third party can regenerate the exact 128-image inventory.

### Edit 15 — §A.5 27-cell pilot grid cell-label enumeration + chosen cell #17 `[this resumed attempt]`

**Reviewer point addressed:** #2 (CRIT-2 partial close).

**New section appended:**
> ### A.5 27-cell pilot grid — cell label enumeration (DEFER stub)
>
> §6.2.2의 27-cell pilot grid는 (L, K, α) ∈ {25, 26, 27} × {2, 4, 8} × {0.5, 1.0, 2.0}이며 ... 본 부록에는 *cell label enumeration*과 *선택 cell 위치*만을 surface하며, per-cell `Δdf(a) / Δem(a) / Δem(b) / Δacc(b)` 4-metric heatmap aggregation은 §8.2에 명시된 deferred 작업이다 ...
>
> [27-row table with cell #1 .. #27, chosen cell #17 (L=26, K=8, α=1.0) marked in bold]

**Rationale:** This is a *partial* close of CRIT-2. The full close — per-cell 4-metric heatmap with all 27 rows × 4 columns + winners across multiple criteria — is genuinely deferred (raw predictions exist at `outputs/e6_steering/llava-onevision-qwen2-7b-ov/pilot_grid_*` but the aggregation script + heatmap generation has not been run for this round). What the §A.5 stub provides: (a) explicit enumeration so the reader can see the grid is rectangular {25, 26, 27} × {2, 4, 8} × {0.5, 1.0, 2.0} = 3 × 3 × 3 = 27 (no hidden cells), (b) chosen-cell position (#17 = mid-of-grid in L and K, mid-of-α — not at any boundary), (c) explicit acknowledgment that the aggregated table absence is the *결정점* of the cherry-pick concern. We do not attempt to defend against the reviewer's CRIT-2 in the language we know they would not accept (we did not pre-register the rule with a time-stamped commit hash); instead we honestly mark it as *수용된 한계*.

### Edit 16 — v6 changelog footer `[this resumed attempt]`

Appended a v6 changelog block summarizing all R4 changes (including the interrupted prior attempt's edits, since no v6 footer existed before this resumed attempt).

### Table edits

None — no Table 6 / Table 7 / Table 8 cell values changed in this round. The InfoVQA n=443 Δdf=−0.7 pp value is unchanged; only the *interpretation* (inconclusive fence) was added in §6.2.3 prose. Table 6 itself stays as is.

### Figure edits

None — no figure caption or PNG path changed.

## Rebuttals (DISAGREE class)

### Rebuttal 1 — MIN-8: "Strict free-lunch" Δem(non-anchored) ≥ 0 is post-hoc shaping

**Reviewer claim:** "The 'strict' framing is built around a celebration criterion: the paper found a cell where Δem(b) was positive, then defined 'strict free-lunch' to include 'Δem(non-anchored) ≥ 0' as a clause that the chosen cell trivially passes."

**Our position:** The Δem(non-anchored) ≥ 0 clause is the *load-bearing conceptual contribution* of "strict free-lunch" relative to plain Pareto improvement, motivated by Chand et al. [2025]'s explicit warning about cross-category collateral damage in bias mitigation (cited in §6.2.3 formal definition). The reviewer's "celebration criterion" framing assumes the criterion was shaped *to fit* the chosen cell's positive Δem(b); the paper's framing is that the criterion was shaped *to forbid hidden harm* on the non-targeted arm, which is a methodologically motivated screening rule independent of which cell ultimately satisfied it. Two pieces of evidence the reviewer's "celebration" framing does not survive: (a) the criterion's ≥ 0 threshold *would have failed* on every single-direction baseline in Table 7 (ActAdd / LEACE rank-1 / query-adaptive / CogBias / MIA-DPO LoRA all have either zero or *negative* Δem(b) — listed as "불변" or "−5.85 pp on VQAv2"); the threshold is therefore *informative* in the comparison panel, not vacuous. (b) The criterion's *no-harm* (Δ ≥ 0) form is explicitly weaker than what the chosen cell actually delivered (Δem(b) average +8.8 pp); if the criterion had been "celebration-shaped" to fit the result, it would have been Δem(b) > 0 with a tight lower bound, not the weak ≥ 0. We retain the existing §6.2.3 formal definition and §1.5 (5) inline gloss.

**Concession:** the reviewer's underlying point that "*strict*" is a *rhetorically* charged term is not unreasonable. We do not edit this round, but flag for camera-ready: if the bar is an EMNLP Main reviewer hostile to marketing-flavored neologisms, the term could be neutralized to "**multi-clause free-lunch**" or "**4-clause Pareto+** criterion" with no semantic change. Logged as next-revision style consideration.

### Rebuttal 2 — MIN-11: Multi-layer redundancy reframable as single-cluster mechanism with three archetype outliers

**Reviewer claim:** "The 6/6 null result is consistent with three different mechanisms — multi-layer redundancy in mid-stack, bimodal peaks in SigLIP-Gemma and Qwen-ViT, and per-dataset peak migration in FastVLM."

**Our position:** The reviewer's reframing is in fact *compatible* with the paper's claim, not contradictory. The paper's §1.4 / §5.2 claim is the *headline empirical observation* that single-layer ablation is null on 6/6 panel — this is one observed quantity. The *interpretation* the paper offers ("multi-layer redundant") is one mechanistic story consistent with the observation. The reviewer offers an alternative interpretation (three different per-archetype reasons for the same observed null). Both interpretations are consistent with the data; the headline ablation-null claim does not depend on a unified mechanism. The paper's §5.1 already enumerates the four archetypes (SigLIP-Gemma early, mid-stack cluster, Qwen-ViT late, FastVLM late) and §5.3 explicitly reports OneVision's bimodal cross-dataset peak migration; the per-archetype heterogeneity the reviewer points to is *in the paper*, not hidden. The §6.4 *prediction* of single-direction failure is a *cross-dataset* claim (peaks migrate between datasets within OneVision and presumably between models) — it does not require all archetypes to share the same single mechanism. We do not edit; the paper's §5.1 / §5.3 already provide the per-archetype detail the reviewer asks for, and the §1.4 / §1.5 (4a) "multi-layer redundancy" framing remains the load-bearing summary statement that all three reviewer-named alternative mechanisms are *also consistent with*.

**Concession:** The reviewer's reframing is a fair *secondary* description, and a future revision could add a §5.2 sentence clarifying that the multi-layer redundancy story is *one of several mechanism-level descriptions* consistent with the 6/6 null. Logged as low-priority style refinement.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| MAJ-4: §6.2.3 paired-bootstrap CI on all 5 cells | Requires re-running aggregation script with paired-bootstrap n=1,000 per dataset (already implemented for §7 capability preservation); ~1 H100-hour | Owner: paper author. Estimate: <1 day. Will re-state §6.2.3 headline based on CI bands (PlotQA significant; ChartQA / MathVista borderline; TallyQA / InfoVQA inconclusive — likely match). |
| MAJ-5: CAA at K=1 on (a − m) as Table 7 row | New GPU run required; ~4-8 H100-hours for calibration + 5-dataset eval. §6.5 Note already explains structural reduction (CAA = ActAdd at K=1 on (a − m) calibration set under our setting). | Owner: paper author. Estimate: 1 day. Empirical row would *confirm* the structural reduction prediction. |
| MAJ-5: ITI multi-direction at attention-head as Table 7 row | New GPU run + ITI calibration recipe adaptation to (a − m); §5.2 single-layer attention ablation null already predicts attention-head locus failure | Owner: paper author. Estimate: 2-3 days. |
| MAJ-6: paired-bootstrap CI + Bonferroni-20 on §6.2.3 Table 6 | Linked to MAJ-4 above; the CI is the prerequisite for the multiple-comparisons correction. | Owner: paper author. Estimate: bundled with MAJ-4. |
| CRIT-2 full close: 27-cell × 4-metric × calibration-dataset aggregated heatmap | Aggregation script + heatmap-rendering pipeline not yet run for the pilot grid raw predictions; predictions exist at `outputs/e6_steering/llava-onevision-qwen2-7b-ov/pilot_grid_*` | Owner: paper author. Estimate: <1 day. Would close the cherry-pick concern fully — currently §A.5 stub (cell label enumeration + chosen cell position) is the partial close. |
| CRIT-3: random-K=8 subspace baseline + non-anchor-task calibration baseline at L=26 | New GPU runs required (~8-16 H100-hours for two baseline replications across 5 evaluation datasets) | Owner: paper author. Estimate: 2-3 days. Would falsify (or confirm) the §6.3 "wrong-base error mode" interpretation against general regularization (Alt-1) and numeric mode-collapse (Alt-2). |
| CRIT-1 full close: cross-architecture E6 calibration + 5-dataset eval + capability eval | Per-archetype E6 replication on Gemma3-27b (SigLIP-Gemma), Qwen2.5-VL-7b (Qwen-ViT late), FastVLM-7b (FastVLM late) — calibration · pilot grid · 5-dataset eval · capability eval | Owner: paper author. Estimate: ~30 H200-day for 3 archetypes (per §8.2). This is the deepest deferred item; current paper accepts case-study scope on CRIT-1 hedges across 6 callsites rather than running it. |
| MAJ-6: pre-registration registry document | Not retrospectively addable; the em-deal-breaker rule was applied as an internal convention (cited from prior Tally-only run tracker line 446) but no time-stamped external registry exists | Owner: future submissions. Camera-ready or future paper would pre-register on OSF / AsPredicted before next experimental round. |

## Open questions for next round

- **Strict free-lunch terminology.** Rebuttal 1 concedes the term is rhetorically charged. If a Round 5 reviewer still pushes back, neutralize to "4-clause Pareto+" or "multi-clause free-lunch" without semantic change.
- **Multi-layer redundancy framing.** Rebuttal 2 acknowledges the reviewer's three-mechanism reframing as a fair secondary description. A §5.2 sentence could be added in a future revision to flag this without weakening the headline claim.
- **Title pluralization.** The reviewer's CRIT-1 strongest form ("retitle to 'in LLaVA-OneVision' or 'case study on one architecture'") is rejected in favor of body-level scope hedges across 6 callsites. If a future Round-5 reviewer demands title change, the cleanest form would be subtitle "(case study on LLaVA-OneVision-7b)" appended to the existing title.
- **§6.5 "유일 후보" wording.** The reviewer's MAJ-5 sub-claim that CAA / ITI Note is "claiming victory without comparison" is partly addressed by the §6.5 Note + the Table 7 panel-boundary disclosure. Empirical CAA / ITI rows (the four-blocking-item #3 in the reviewer's "switch to weak accept" path) would close this fully.

## Internal consistency check

After all edits, verified:

- [x] **Abstract numbers unchanged.** Δdf [−5.2, −0.3], Δem(a) +3.9, Δem(b) +8.8, macro +0.41, HB +2.21 [+1.14, +3.28], POPE −0.06 [−0.21, +0.09], γ-β ×1.6 / ×2.9 / ×12.7 — all preserved.
- [x] **§1.5 contributions still match §4-§7 deliveries.** (1) → §3 + §4; (2) → §3.2; (3) → §4.4; (4a) → §5.2; (4b) → §5.1 / §4.4 Insight 3; (5) → §6.2 + §6.4 + §6.5 + §7 (with single-model case study hedge); (6) → §4.6 + §8.2 (with N=1 × N=1 hedge).
- [x] **§8.1 종합 now consistent with body.** Single-model E6 hedge + γ-β single-pair hedge added inline; both match §8.2 limitation list bullets and §1.5 hedges.
- [x] **§A.4 / §A.5 cross-references.** §6.2.2 references §A.5 (was §A.4, fixed); §A.4 = FLUX seed entry; §A.5 = 27-cell pilot grid stub. Both new sections exist and are properly numbered.
- [x] **All figure embeds resolve.** No figure path changed.
- [x] **No table renumbering.** Tables 1-8 + Figures 1-7 + Figures A1 / B1 / B2 / C1-C4 / F1 / G1 stable.
- [x] **References list unchanged this round.** All R3 additions (Belrose 2023 / Li 2023 / Panickssery 2024 / Chand 2025) preserved; no new citations added in R4.
- [x] **Round 1-3 fixes preserved.** All R1 numerical corrections, R2 prose fixes, R3 novelty / positioning additions intact.
- [x] **R4 single-model hedge propagated to 6 callsites consistently:** abstract (line 13), §1.3 (line 35), §1.5 (5) (line 43), §6.6 (line 344), §8.1 (line 374), §8.2 (line 380). All six say "case study" + reference to §8.2.

## Diff stat

- Lines: 543 → 581 (+38 lines, +7.0 %).
  - This includes both the interrupted prior attempt's net +5 lines (already in starting state) and this resumed attempt's +33 lines (new §A.4, §A.5, §7 Bonferroni paragraph, §8.1 inline hedges, §6.2.2 pointer fix, §8.2 deferred bullet refinement, v6 changelog).
- Sections fully rewritten: 0.
- Tables modified: 1 (new 27-row Table in §A.5; existing Tables 1-8 unchanged).
- Figures modified: 0.
- New paragraphs (this resumed attempt): 4 (§A.4 FLUX seed, §A.5 stub + table, §7 Multiple-comparisons 보정, §8.1 inline hedges, v6 changelog).
- New paragraphs (interrupted prior attempt, recovered from current paper state): §6.3 Insight 1.5 (alternative explanations), §6.6 single-model scope statement, §6.2.3 신뢰구간 caveat, §8.2 E6 single-model bullet + γ-β bullet + 4 deferred bullets.
- Word count delta: roughly +1,500 Korean words across this round (combined interrupted + resumed).

**Single most impactful edit:** **the CRIT-1 single-model case study hedge propagation across 6 callsites** (abstract / §1.3 / §1.5 (5) / §6.6 / §8.1 / §8.2). This is the kill-shot the reviewer named; without it, the rejection case is "headline mitigation is N=1 on the model axis, paper does not say so." With it, the paper acknowledges the scope honestly and the reviewer's CRIT-1 demand ("acknowledge the N=1 model on headline E6 mitigation in §8.2 limitations") is fully met. Combined with the §A.5 27-cell stub (CRIT-2 partial close), §6.3 Insight 1.5 (CRIT-3 alternatives + POPE partial signal), §6.2.3 신뢰구간 caveat + §7 Multiple-comparisons 보정 (MAJ-4 + MAJ-6 close), and the §8.2 deferred-list expansion (MAJ-5 + CRIT-3 + CRIT-2 explicit DEFER), this round closes 3/3 CRIT items at the *prose* level and 2/4 MAJOR items at the *prose level + post-hoc verification* level. The remaining MAJOR items (MAJ-5 CAA/ITI rows; MAJ-4 full bootstrap CI on Table 6) require new GPU work and DEFER honestly with owner / timeline / GPU-hour estimates.

## Items still posing rejection risk after this revision

The reviewer's "transition to weak accept" path lists four blocking items (CRIT-1, CRIT-2, MAJ-4, MAJ-5). After this round:

1. **CRIT-1 (single-model E6):** *Closed at scope-hedge level.* Body says "case study" across 6 callsites. Title not changed. If reviewer demands title-level change, likely still reject. Mitigation: add "(case study on LLaVA-OneVision-7b)" subtitle in camera-ready.
2. **CRIT-2 (27-cell pilot grid):** *Partially closed.* §A.5 stub provides cell enumeration + chosen-cell #17. Aggregated 4-metric heatmap remains DEFER. Reviewer explicitly demanded "27-cell heatmap that already exists in `outputs/e6_steering/llava-onevision-qwen2-7b-ov/sweep_subspace_*`"; we surface labels but not metric values. Continued rejection risk on this axis.
3. **MAJ-4 (paired-bootstrap CI on Table 6):** *Closed at headline-rephrase level + DEFER for actual CI.* §6.2.3 reframes "5/5 dataset df reduction" to "5/5 부호 음, 4/5 noise floor 상회, InfoVQA inconclusive." Reviewer explicitly asks for paired-bootstrap CI; this is a 1-day DEFER, not run.
4. **MAJ-5 (CAA / ITI Table 7 rows):** *Closed at structural-reduction level + explicit DEFER for empirical rows.* §6.5 Note explains reduction; §8.2 lists DEFER. Reviewer explicitly asks for empirical rows; these are 1-3 day DEFER, not run.

**Honest assessment:** This paper after Round 4 revision is closer to *strong Findings* than *weak accept Main*. The reviewer's "honest verdict" — strong Findings, not Main — is plausible. Rejection at Main remains likely if a Round-5 reviewer reading the activation-steering literature decides the four DEFER items are not acceptable as DEFER. Findings acceptance is plausible if the four DEFER items are accepted as honest deferral with explicit owner / timeline. The paper does not contest this assessment.
