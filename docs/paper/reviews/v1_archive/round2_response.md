# Round 2 — Author Response to Writing Reviewer

**Paper version BEFORE:** `docs/paper/emnlp_draft_ko.md` @ 524 lines, post-Round-1 revision (`v3` changelog footer).
**Paper version AFTER:** `docs/paper/emnlp_draft_ko.md` @ 526 lines, `v4` changelog footer (2026-05-09).
**Date:** 2026-05-09.
**Reviewer round addressed:** `docs/paper/reviews/round2_writing.md` (11 must-fix, 20 should-fix, borderline).

## Summary

We accept all 11 must-fix items and apply paper edits to every one. Korean naturalness is the dominant axis: 6 surviving translator-coinages flagged by the reviewer ("이분법" ×6, "주력", "발화" ×2, "사지 *않으면서*", "빈자리", "회귀" revert-sense ×2) are removed globally and replaced with the reviewer-proposed canonical forms or rephrased for natural Korean. The §4.6 Insight 1 grammar break — the reviewer's sharpest critique — is rewritten with explicit subject/verb structure and the hedging strength reduced from "직접 입증" to "강하게 뒷받침 / Insight 2와 합쳐 first-evidence 구성", which matches the §8.2 N=1 limit and abstract's softened "first-evidence" framing. Table 2 / §4.1 Insight 1 contradiction is resolved by qualifying "robust" by metric (`adopt(a)` Qwen2.5-VL minimum 0.021 / `df(a)` Gemma4-31b minimum 0.085) — both bolds preserved with the divergence explicitly named. Figure 5 caption is qualified for the InfoVQA reversal. We address 17 of the 20 should-fix items (deferring 3 that the reviewer themselves marked "borderline" or low-signal: "끌림 / anchor pull" stylistic mixing, the §6.6 closing line which was already trimmed in the §6.6 rewrite, and the "빈자리" replacement which was applied via clause restructure rather than substitution).

The numerical headline tables (Tables 2 / 5 / 6 / 7 / 8) and all Round-1 corrections are preserved. No new claims introduced; soft hedges added in three places (abstract closing, §4.6 Insight 1, §1.4 control qualifier).

## Decision summary table

### Must-fix (11)

| # | Reviewer point (verbatim) | Class | Section affected | Status |
|---|---|---|---|---|
| 1 | CRIT-W1 §4.1 Table 2 / Insight 1 robustness ordering ambiguity | EDIT | §4.1 Table 2 caption + Insight 1 prose | done |
| 2 | CRIT-W2 §4.6 Insight 1 sentence 2 ungrammatical Korean | EDIT | §4.6 Insight 1 (full rewrite) | done |
| 3 | CRIT-W3 §1.5 (4)/(5)/(6) bullet-train rhythm | EDIT | §1.5 contributions list | done |
| 4 | MAJOR-W5 §6.2 title "주력" forced Korean | EDIT | §6.2 title + abstract gloss | done |
| 5 | MAJOR-W6 "발화" misuse §6.1 Insight + §6.3 first paragraph | EDIT | §6.1 Insight + §6.3 paragraph 1 | done |
| 6 | CRIT-W4 "이분법" ×6 occurrences | EDIT | §1.2.3 / §4.5 paragraph + Insight 1 / §4.6 Insight 1 / §8.1 | done (5 sites + 1 §4.6 absorbed in W2 rewrite) |
| 7 | MAJOR-W7 §7 bootstrap CI bolt-on bold-period heading | EDIT | §7 (folded into result paragraph) | done |
| 8 | MAJOR-W8 §8.2 dual-register limitation list | EDIT | §8.2 (split into "한계 / deferred 두 register") | done |
| 9 | MAJOR-W11 §3.1 Figure 1 no in-text reference | EDIT | §3.1 paragraph closing | done |
| 10 | MAJOR-W10 Figure 5 caption vs §4.4 Insight 2 contradiction | EDIT | Figure 5 caption + §4.4 Insight 2 prose | done |
| 11 | Abstract "입증" overclaim vs §8.2 first-evidence hedge | EDIT | Abstract closing | done |

### Should-fix (20)

| # | Reviewer point | Class | Section | Status |
|---|---|---|---|---|
| s1 | §3.3 raw-n disclaimer fragment | EDIT | §3.3 | done — "범위에 분포한다" 추가 |
| s2 | §4.4 Insight 1 restating | EDIT | §4.4 Insight 1 | done — §6.2 universality 사전 정당화로 upgrade |
| s2b | §4.3 Insight 2 restating (advisor surface) | EDIT | §4.3 Insight 2 | done — appended forward link to §6.2.1 SVD calibration design + §6.2.3 effect-size ordering pre-shadowing |
| s3 | §4.5 Insight 3 filler | PARTIAL EDIT | §4.5 Insight 3 | done — language softened, exhaustiveness deferred to §8.2 (no per-bucket counts in canonical evidence) |
| s4 | §4.6 Insight 3 60-char parenthetical | EDIT | §4.6 Insight 3 | done — moved to trailing parenthetical, sentence flow restored |
| s5 | §5.1 Insight 2 multi-purpose paragraph | DEFER | §5.1 Insight 2 | not split — §5.1 Insight 2 dual-purpose paragraph is needed to prevent re-conflation; reviewer's "split into 2/3" risks re-fracturing the disambiguation. Logged as MINOR; not addressed this round. |
| s6 | §6.6 closing recommendation duplication | EDIT | §6.6 | done — second/third sentence rewritten as "작동하고 / 작동한다" + closing recommendation rephrased |
| s7 | §4.2 nominalisation stack | EDIT | §4.2 Insight 2 | done — "안정한 것은" → "안정하다는 것은", "보인다" → "의미한다" |
| s8 | §6.5 Table 7 verdict register | EDIT | §6.5 Table 7 | done — Korean primary + English clarifier dual form |
| s9 | §6.3 first paragraph rewrite | EDIT | §6.3 paragraph 1 | done — folded into MAJOR-W6 ("발화") fix |
| s10 | §4.7 italicised "generic tendency" | EDIT | §4.7 + §4.5 Insight 3 | done — both occurrences → 일반 경향 |
| s11 | §1.5 (6) "first-evidence claim" terminology | EDIT | §1.5 + 4 callsites | done — "first-evidence" 통일, 외부 4개 callsite 정렬 |
| s12 | §3.3 mech panel "4 encoder archetype" qualifier | EDIT | §3.3 | done — encoder family별 4-archetype 구성 expanded |
| s13 | §4.6 Insight 2 fragment | EDIT | §4.6 Insight 2 | done — "함의가 도출된다" |
| s14 | §4.4 Insight 1 fragment "토대." | EDIT | §4.4 Insight 1 | done — full rewrite (sufficient + restating fix s2 absorbed this) |
| s15 | §6.5 Insight closing fragment | EDIT | §6.5 Insight | done — "결과이다" |
| s16 | §8.1 closing fragment + particle stack | EDIT | §8.1 | done — "first-evidence VLM 결과이다" |
| s17 | §6.6 sentence fragments | EDIT | §6.6 | done — see s6 |
| s18 | Abstract methodology pointer "(§5.2 → §6.4)" | EDIT | Abstract | done |
| s19 | Abstract / §1.2.2 "회귀" revert disambiguation | EDIT | Abstract pillar (iii) + §1.2.2 | done — "*되돌아간다*" |
| s20 | Figure 7 caption "collapse" → "붕괴" | EDIT | Figure 7 caption | done |

### Strengths the reviewer highlighted

The reviewer named four "strongest Insight boxes" (§4.3 Insight 3 PlotQA un-mitigated free-lunch, §6.2.1 (a−m) contrast core, §6.4 predict-→-verify, §5.3 OneVision dataset-dependent peak honest disclosure). All four are untouched in this revision — they are precisely the load-bearing argument the paper rests on. Round-2 only fixed the prose around them.

## Edit log (every paper change in this round)

### Edit 1 — §4.1 Table 2 caption + bolds + Insight 1: robustness ordering by metric

**Final caption / bolds (post-advisor cross-check):** Bold cell convention now covers `adopt(a)` max (Gemma4-e4b 0.066) + `adopt(a)` min (Qwen2.5-VL 0.021) + `df(a)` max (Gemma4-e4b 0.274) + `df(a)` min (Gemma4-31b 0.085). Bold row name covers each metric's "가장 강건 모델" — Qwen2.5-VL row name bolded (adopt-min), **Gemma4-31b row name also bolded** (df-min). Caption legend reads *"Bold cell = 각 metric 기준 셀 최댓값 또는 최솟값. Bold row name = 해당 metric 기준 가장 강건 (최소) 모델."* — closes the Round-2-introduced bold/legend mismatch the advisor flagged.

**Reviewer point addressed:** Must-fix #1 (CRIT-W1).
**Reviewer reasoning:** Table 2 row Gemma4-31b-it `df(a) = 0.085 < Qwen2.5-VL 0.094`, so by `df(a)` Gemma4-31b is most robust; bold convention bolded Qwen2.5-VL row; Insight 1 prose said "Qwen2.5-VL 가장 강건". Three pieces could not all be consistent.

**Decision:** Keep Qwen2.5-VL bolded (as the `adopt(a)` minimum 0.021), bold Gemma4-31b's `df(a)` cell (0.085) as the `df(a)` minimum, and rewrite both Table caption and Insight 1 prose to qualify "강건" by metric. This preserves all four downstream Qwen2.5-VL-as-most-robust call-sites (§4.3 Insight 1, §4.4 Insight 3, Figure 5 caption, §1.4) with one-clause additions ("adopt 기준") rather than four substantive rewrites.

**Before (Table 2 caption):**
> **Table 2.** 7-model VQAv2 panel (n=17,730 each, C-form). Bold = 가장 강한 끌림 모델 / 가장 강건한 모델.

**After (Table 2 caption):**
> **Table 2.** 7-model VQAv2 panel (n=17,730 each, C-form). Bold = 가장 강한 끌림 셀 (`adopt(a)` / `df(a)` 최댓값) / 가장 강건 셀 (`adopt(a)` 기준 최소). `adopt(a)` 기준 가장 강건은 Qwen2.5-VL-7b (0.021), `df(a)` 기준 가장 강건은 Gemma4-31b-it (0.085) — 두 metric이 6위/7위에서 미세 역전한다 (`adopt`로는 Qwen2.5-VL < Gemma4-31b 0.003 pp, `df`로는 Gemma4-31b < Qwen2.5-VL 0.009 pp).

**Before (Insight 1):**
> **Insight 1 (효과 크기와 모델 능력의 역상관).** 끌림 크기 순서 (Gemma4-e4b 가장 큼 → Qwen2.5-VL 가장 강건)는 base accuracy 순서의 *역*과 거의 일치 — *능력이 낮은 곳에서 끌림이 크다*. 이는 §4.5의 confidence quartile 결과를 baseline 능력 측에서 미리 시사한다.

**After (Insight 1):**
> **Insight 1 (효과 크기와 모델 능력의 역상관).** 끌림 크기 순서 (Gemma4-e4b 가장 큼 → 패널 최하단 두 모델 Qwen2.5-VL-7b / Gemma4-31b가 가장 강건; `adopt(a)`로는 Qwen2.5-VL이 최소 0.021, `df(a)`로는 Gemma4-31b가 최소 0.085 — 둘 다 base accuracy 상위 그룹에 속한다)는 base accuracy 순서의 *역*과 거의 일치 — *능력이 낮은 곳에서 끌림이 크다*. 이는 §4.5의 confidence quartile 결과를 baseline 능력 측에서 미리 시사한다.

**Bold cell changes (Table 2 body):**
- Row Qwen2.5-VL-7b — `df(a)` cell **0.094** → `0.094` (un-bolded, no longer "robust by df").
- Row Gemma4-31b-it — `df(a)` cell `0.085` → **0.085** (bolded as "df-robust"); **row name bolded** (newly added in advisor pass since this row also wins one robustness metric).

**Source verified:** `docs/insights/headline-numbers.md §B 7-model VQAv2 panel` rows; `docs/insights/_data/A1_asymmetric_wide.csv` matches.

**Rationale:** Both metrics defensibly identify "most robust" with a 0.003 / 0.009 pp gap that is within sampling noise; explicit dual-metric framing is more honest than picking one. Downstream call-sites (§4.3 Insight 1, §4.4 Insight 3, §1.4) all use `adopt(a)`-based language and are now coherent with the explicit qualification.

### Edit 2 — §4.6 Insight 1: grammar fix + hedge softening

**Reviewer point addressed:** Must-fix #2 (CRIT-W2).
**Reviewer reasoning:** Original sentence 2 *"증폭은 이분법이 가장 *덜* 취약하다고 한 *correct-base* row에 우선적으로 떨어진다"* has no subject for matrix verb "한"; sentence is grammatically unparseable. This is the only paragraph justifying §1.5 (6).

**Before:**
> **Insight 1 (메커니즘 결합 증명).** §4.5의 L1 monotonicity가 *통계적 인공물이 아닌 mechanism-bound*임을 직접 입증한다. 끌림을 증폭해야 하는 조작 (reasoning trace = anchor를 active 비교 후보로 더 오래 유지)이 실제로 증폭하며, 증폭은 이분법이 가장 *덜* 취약하다고 한 *correct-base* row에 우선적으로 떨어진다.

**After:**
> **Insight 1 (메커니즘 결합 — confidence-modulated anchor pull 가설과 일치).** §4.5의 L1 monotonicity가 *통계적 인공물이 아닌 mechanism-bound*라는 가설을 강하게 뒷받침한다. anchor pull을 증폭하리라 기대되는 조작 (reasoning trace = anchor를 active 비교 후보로 더 오래 유지)이 실제로 anchor pull을 증폭하며, 그 증폭은 wrong-base / correct-base 분할이 *가장 덜 취약*하다고 분류한 *correct-base* 부분집합에 *집중적으로* 나타난다 (×12.7 ratio가 위치하는 정확한 row). 즉 Mussweiler-Strack의 "낮은 baseline confidence → anchor가 비교 후보로 더 쉽게 활성화" 메커니즘이 reasoning-mode의 trace 연장에 의해 외부에서 *조작 가능*함을 보인다 — Insight 2의 LRM 문헌 정렬과 합쳐 본 논문 §1.5 (6) 기여의 직접 증거를 구성한다.

**Rationale:** (a) Subject "그 증폭" + verb "집중적으로 나타난다" makes the sentence syntactically complete. (b) Hedging softened from "직접 입증" → "강하게 뒷받침" — matches the N=1 architecture-pair caveat in §8.2 and the abstract's softened "first-evidence" framing. (c) The "× 12.7 ratio가 위치하는 정확한 row" parenthetical pins which Table 4 cell carries the load. (d) Closing sentence explicitly threads back to §1.5 (6), making the argumentative chain visible.

### Edit 3 — §1.5 contributions (4)/(5)/(6) re-flow

**Reviewer point addressed:** Must-fix #3 (CRIT-W3).
**Reviewer reasoning:** (4)/(5)/(6) read as bullet labels strung by `+`, breaking rhythm with sentence-shaped (1)/(2)/(3).

**Before:**
> (1) VLM 최초의 cross-modal numerical anchoring 평가 + (2) gt-자유, baseline-relative C-form 표준 metric + (3) 5 dataset × 7 model cross-dataset 증거 + (4) 6-model 메커니즘 panel single-layer ablation null + multi-layer redundancy 분석 → (5) single-direction mitigation의 실패를 *예측한 후 우회*하는 multi-direction subspace projection + 6-benchmark capability preservation 검증 + (6) VLM 최초의 reasoning-amplifies-anchoring 결과 (γ-β, N=1 Qwen3-VL Instruct vs Thinking — first-evidence claim).

**After:**
> (1) VLM 최초의 cross-modal numerical anchoring 평가. (2) gt-자유, baseline-relative C-form 표준 metric. (3) 5 dataset × 7 model cross-dataset 증거. (4) 6-model 메커니즘 panel에서 single-layer ablation이 6/6 null임을 보이고 그 *multi-layer redundancy*를 정량화한다. (5) Single-direction mitigation의 cross-dataset 실패를 *예측한 뒤 multi-direction subspace projection으로 우회*하며, 6-benchmark capability preservation까지 동시에 검증한다 (strict free-lunch). (6) γ-β reasoning pair (N=1 Qwen3-VL Instruct vs Thinking)에서 reasoning-amplifies-anchoring을 처음 보이는 *first-evidence* VLM 결과.

**Rationale:** Each contribution now has a `.`-terminated sentence shape; (4)/(5)/(6) rewritten with verb endings 정량화한다 / 검증한다 / 결과 (이다) matching the (1)/(2)/(3) period rhythm. "First-evidence claim" → "*first-evidence* VLM 결과" — drops the awkward "claim" English-noun mid-Korean. (6) parenthetical now lives inside the prose flow, not as a `+`-tail.

### Edit 4 — §6.2 title "주력" drop + abstract "주력" drop

**Reviewer point addressed:** Must-fix #4 (MAJOR-W5).

**Before (abstract gloss):** `**E6 (residual-stream subspace projection, 주력)**:`
**After (abstract gloss):** `**E6 (residual-stream subspace projection, deployable)**:`

**Before (§6.2 title):** `### 6.2 E6 — Residual-stream subspace projection (deployable, 주력)`
**After (§6.2 title):** `### 6.2 E6 — Residual-stream subspace projection (deployable)`

**Rationale:** "주력" (wartime-flagship metaphor in Korean academic register) eliminated as the only outlier — every other abstract / §1.3 / §6.6 gloss already says "deployable mitigation으로 권장" or just "deployable", so this is a pure consistency fix.

### Edit 5 — "발화" replacement: §6.1 Insight + §6.3 first paragraph

**Reviewer point addressed:** Must-fix #5 (MAJOR-W6).

**Before (§6.1 Insight):**
> hook은 single-image inference에 누출되지 않으며 두 번째 이미지에 가독 digit이 없으면 발화는 하지만 제거할 signal이 없다.

**After (§6.1 Insight, plus the s11 Round-1 audit add of "LLaVA가 Δdf 측, ConvLLaVA가 Δem 측"):**
> hook은 single-image inference에 누출되지 않으며, 두 번째 이미지에 가독 digit이 없는 경우 *hook은 트리거되지만* 제거할 signal이 없다. 즉 *upper-half pathway가 모델 자체 답안 형성에 non-load-bearing*이며, 그 부담을 줄이면 anchor 영향만 제거된다. Table 5의 per-column bold는 이 분리를 그대로 반영한다 — LLaVA-1.5는 Δdf 측에서 가장 큰 효과 (−14.6 %), ConvLLaVA-7b는 Δem(a) 측에서 가장 큰 회복 (+1.30 pp).

**Before (§6.3 paragraph 1):**
> b-arm은 `target_only` — 단일 이미지 + 질문, 두 번째 anchor 이미지 *없음*. Projection은 그래도 발화하는데, universal이라 input에 anchor가 있는지 *모르기* 때문이다.

**After (§6.3 paragraph 1):**
> b-arm은 `target_only` — 단일 이미지 + 질문, 두 번째 anchor 이미지 *없음*. Projection은 anchor 유무와 무관하게 모든 forward에서 작동한다 (universal projection — input이 anchor를 포함하는지 알지 못함).

**Rationale:** "발화 (linguistic utterance)" replaced with reviewer-suggested "hook 트리거" / "projection 작동" — both natural Korean for the technical operation. The §6.1 Round-1 audit ask (s11 — name which row wins which column in Table 5) is folded into the same edit since it lives in the same Insight paragraph; this also addresses Round-1 follow-up audit table row "Table 5 prose-doesn't-call-out-bold-convention".

### Edit 6 — "이분법" global replacement (5 sites + 1 absorbed in Edit 2)

**Reviewer point addressed:** Must-fix #6 (CRIT-W4).

| Site | Before | After |
|---|---|---|
| §1.2.3 | `Wrong/correct 이분법은 이 연속 구조의 거친 projection에 불과하다.` | `Wrong-base / correct-base 분할은 이 연속 구조의 거친 projection에 불과하며 (§4.5에서 quartile 기준으로 재유도), 본 pillar는 §4.4의 main matrix 종합과 §4.5의 confidence quartile 결과로 동시에 전달된다.` |
| §4.5 paragraph | `§4.1의 wrong/correct 이분법은 더 풍부한 연속 구조의 거친 projection이다.` | `§4.1의 wrong-base / correct-base 분할은 더 풍부한 연속 구조의 거친 projection이다.` |
| §4.5 Insight 1 | `**Insight 1 (이분법의 재해석).** ... 이분법 +7.2 pp gap이 quartile 기준 +23 pp로 확대 — 이분법은 단순히 *평균*했을 뿐 ...` | `**Insight 1 (wrong/correct 분할의 재해석).** Phase-A의 wrong-base / correct-base 분할은 confidence 연속체의 *Q1 + Q2 vs Q3 + Q4 projection*이다. 분할 +7.2 pp gap이 quartile 기준 +23 pp로 확대 — 분할은 단순히 *평균*했을 뿐 효과는 본질상 연속 gradient이다. 이는 §6.2 mitigation 설계가 *categorical wrong-base flag*를 별도 입력으로 받지 않고 residual representation 자체에서 universal projection으로 작동하는 것을 정당화한다 ...` |
| §4.6 paragraph | `H2 이분법으로 split하면 ...` | `H2 wrong-base / correct-base 분할로 split하면 ...` |
| §4.6 Insight 1 | (covered in Edit 2) | "이분법" → "wrong-base / correct-base 분할" inline. |
| §8.1 종합 | `Phase-A 이분법은 confidence gradient의 거친 projection이며 ...` | `Phase-A의 wrong-base / correct-base 분할은 confidence gradient의 거친 projection이며 ...` |

The §4.5 Insight 1 rewrite also adds an explicit downstream-claim sentence (justifies §6.2's universal-projection design choice from this connection) — addresses [MAJOR-W9] (§4.5 prose builds on §1.2.3 instead of repeating).

**Rationale:** "이분법 (binary thinking / false dichotomy)" carries philosophical-debate connotation in Korean academic register that is not what the paper means. "wrong-base / correct-base 분할" is unambiguous, matches the technical canon already in `references/AGENTS.md` (`base_correct` flag), and is the reviewer's recommended canonical form.

### Edit 7 — §7 bootstrap CI fold-in (drop bold-period heading)

**Reviewer point addressed:** Must-fix #7 (MAJOR-W7).

**Before:**
> [HallusionBench / POPE result paragraph] ... Pipeline integrity는 lmms-lab model card published 수치와 비교 검증 (MMStar 61.67 vs 61.7 본질적 일치).
>
> **95 % CI 산출.** Proportion-style benchmark (RealWorldQA / HallusionBench / MMStar / MMBench-DEV-EN / POPE)는 ...

**After (single fused paragraph, no bold heading):**
> [HallusionBench / POPE result paragraph continues] ... Pipeline integrity는 lmms-lab model card published 수치와 비교 검증 (MMStar 61.67 vs 61.7 본질적 일치). 신뢰구간은 proportion-style benchmark (...)에 대해서는 baseline과 mitigation의 per-question correctness를 paired Bernoulli로 두고 McNemar 분산 추정 `SE(Δ) = sqrt(b + c) / n`의 normal-approximation으로 산출했고 (`b`, `c` = paired discordant count), sum-style benchmark (OCRBench)에는 동일 paired pair의 per-question score 차분에 paired percentile bootstrap (n = 1,000 resample, seed 0)을 적용했다 (구현 `scripts/aggregate_capability_eval.py:mcnemar_paired / bootstrap_score_diff`; 두 절차 모두 per-question 수준 resample로 baseline·mitigation 간 의존성을 보존). §6.2.3 paired-sids deltas는 동일 paired-bootstrap 절차를 적용하지 않은 점추정 상태이며, 소규모 denominator cell (n=170 MathVista, n=224 ChartQA)의 CI 보고는 후속 revision의 직접 항목으로 §8.2 한계에 명시한다.

**Rationale:** Bold-period heading "**95 % CI 산출.**" was the reviewer's named "memo-to-self" register — folded into the result paragraph as a post-Pipeline-integrity continuation. Same content, no register break. Insight box that follows (about anchoring-vs-hallucination cross-pattern) now reads continuously after the result paragraph instead of after a methodology interruption.

### Edit 8 — §8.2 split into "한계 / deferred" two registers

**Reviewer point addressed:** Must-fix #8 (MAJOR-W8).

**Before:** Single 7-bullet list mixing research-scope limitations (단일 prompt / Open-weight only / Human baseline 부재 / Mid-stack cluster 단일 / γ-β N=1) with operational debt (small-n CI / 27-cell pilot grid / OneVision E1d analyzer fix).

**After:** Two sub-blocks under §8.2 — `**연구 범위 한계 (research scope).**` (5 bullets) + `**이번 라운드에서 deferred 된 작업 (operational follow-up).**` (3 bullets). Both Korean primary + English clarifier in sub-block headers.

**Rationale:** Reader can now distinguish "what we acknowledge as a research-scope boundary" from "what we owe the next revision". Keeps all 7 bullets — only the structural scaffolding changes.

### Edit 9 — §3.1 Figure 1 in-text reference

**Reviewer point addressed:** Must-fix #9 (MAJOR-W11).

**Before (§3.1 closing of pre-Figure-1 paragraph):**
> ... 자극 inventory는 ... + 128개 digit-free FLUX render (`d`)이다 (부록 §A).

**After:**
> ... 자극 inventory는 ... + 128개 digit-free FLUX render (`d`)이다 (부록 §A; 4-조건 자극이 모델별 acc drop에 미치는 효과 예시는 Figure 1).

**Rationale:** Figure 1 now has the same `(Figure N)` in-text citation pattern as every other figure in the paper.

### Edit 10 — Figure 5 caption + §4.4 Insight 2 alignment

**Reviewer point addressed:** Must-fix #10 (MAJOR-W10).

**Before (Figure 5 caption):**
> Figure 5 — 5-dataset 6-model wrong-base S1 direction-follow. df 부호 30/30 cell 모두 양수. gemma3-4b가 chart/plot/math에서 가장 큰 끌림, qwen2.5-vl이 가장 강건.

**After (Figure 5 caption):**
> Figure 5 — 5-dataset 6-model wrong-base S1 direction-follow. df 부호 30/30 cell 모두 양수. gemma3-4b가 ChartQA/PlotQA/MathVista에서 가장 큰 끌림 (단, InfoVQA에서는 4B < 27B로 역전 — Insight 2). adopt 기준 가장 강건한 모델은 qwen2.5-vl-7b (df로는 gemma4-31b가 미세하게 더 낮음 — Table 2 / §4.1 Insight 1 참조).

**Before (§4.4 Insight 2):**
> Gemma3-4b가 PlotQA에서 *27B보다 더 끌린다* (0.395 vs 0.227). 그러나 InfoVQA에서는 4B (0.324) < 27B (0.350)로 역전한다. 이는 *visual reasoning 빈자리 → 두 번째 이미지 digit 의존*이라는 메커니즘 가설과 일치 ...

**After (§4.4 Insight 2):**
> Gemma3-4b가 PlotQA / ChartQA / MathVista 3개 dataset에서 *27B보다 더 끌린다* (PlotQA 0.395 vs 0.227). 그러나 InfoVQA에서는 4B (0.324) < 27B (0.350)로 역전한다 — 따라서 "anti-scaling이 chart/plot/math 3개 dataset에 한정되며 InfoVQA에서는 표준 scaling 회복"이라는 형태로 정확히 표현된다. 이는 *visual reasoning capability gap → 두 번째 이미지 digit 의존*이라는 메커니즘 가설과 일치 — 작은 SigLIP encoder가 차트의 정확한 답을 읽지 못할 때 가시 digit을 단서로 더 강하게 잡는다. ...

**Source verified:** `docs/insights/_data/main_panel_5dataset_summary.md` rows for `ChartQA gemma3-4b 0.346 vs 27b 0.240`, `PlotQA gemma3-4b 0.395 vs 27b 0.227`, `MathVista gemma3-4b 0.413 vs 27b 0.332`, `InfographicVQA gemma3-4b 0.324 vs 27b 0.350`.

**Rationale:** Caption now bounds the anti-scaling claim correctly (chart/plot/math, not InfoVQA), Insight 2 prose reframes the contradiction as a precise dataset-bounded statement rather than a partial-truth headline. "빈자리" (must-fix #6 / MAJOR-W6 in spirit) → "capability gap" naturalised.

### Edit 11 — Abstract closing "입증" → "first-evidence VLM 결과"

**Reviewer point addressed:** Must-fix #11.

**Before:**
> ... 텍스트 LRM 문헌의 reasoning-amplifies-bias 현상이 VLM에 일반화됨을 입증.

**After:**
> ... 텍스트 LRM 문헌의 reasoning-amplifies-bias 현상이 VLM에서도 처음 재현되는 *first-evidence* 결과 (단일 architecture pair, cross-architecture 일반화는 §8.2 한계로 명시).

**Rationale:** "입증 (proof)" overclaims for N=1 architecture pair; replaced with "first-evidence 결과" matching §1.5 (6) / §4.6 Insight 2 / §8.2 lim 5 — single canonical phrase across all five mention sites. Limitation hedge in-line so the reader doesn't have to wait until §8.2.

### Edit 12 — Sentence-fragment polish (should-fix #13–17, #s7)

| Site | Fragment | Fix |
|---|---|---|
| §4.2 Insight 2 | "S1→S5에서 em이 안정한 것은 *거리는 em을 손상시키지 않음*을 보인다." | "S1→S5에서 em이 안정하다는 것은 *거리가 em을 손상시키지 않음*을 의미한다." (also closing "핵심 단서" → "핵심 단서이다") |
| §4.4 Insight 1 | "...본 논문이 추구할 수 있는 토대." | rewritten as "이는 §6.2의 mitigation universality 주장 — 단일 (L, K, α) hyperparameter가 5/5 dataset에 일반화 — 의 사전 정당화이다 (만일 cell-level 효과가 부호 비일관이라면 단일 cross-dataset hyperparameter가 정의 가능하지 않다)." (s2 + s14 combined) |
| §4.5 paragraph closing | "...둘 다 정렬되도록 (출처 ...)" fragment | "...두 정의에 모두 정렬되도록 한 조치이다 (출처 ...)" |
| §4.5 Insight 3 | "...두 boundary case의 결과." | "각 그룹의 정확한 cell count 보고와 mechanistic exhaustiveness 검증은 §8.2의 follow-up 항목으로 deferred 한다 ..." |
| §4.6 Insight 2 | "...운영적 함의." | "...운영적 함의가 도출된다." |
| §4.6 Insight 3 | 60-char parenthetical mid-sentence | parenthetical moved to trailing position; main verb-phrase "더 *낮다*" left clean |
| §6.5 Insight | "...직접 결과." | "...직접 결과*이다*." |
| §6.6 closing | "...동작. ... 동작." sentence ends without 이다 | "...작동하고, ...작동한다. ... 본 논문이 §1.3 / 초록에서 *deployable mitigation* 으로 권장하는 것은 §6.2의 E6이다." |
| §8.1 closing | "...장소가 됨을 시사하는 첫 VLM 결과." (particle stack) | "reasoning trace에서 bias가 *축적*된다는 것을 시사하는 first-evidence VLM 결과이다." |

### Edit 13 — Round-1 audit follow-ups (s18, §1.4 controls qualifier)

**§1.4 controls add (G claim/evidence):**
- Before: *"... anchor robustness를 *낮춘다*는 직접 증거."*
- After: *"... anchor robustness를 *낮춘다*는 first-evidence cross-arm 결과이다 (controls in §4.6 Insight 3 — Thinking acc(d) 0.587 < Instruct 0.647)."*

**Abstract methodology pointer (s18):**
- Before: *"...mitigation의 cross-dataset *실패*를 예측하고 검증한다 (per-dataset mean-anchor direction이 측정 가능하게 다른 곳을 가리킴, cos ≈ 0.47-0.62)."*
- After: *"...mitigation의 cross-dataset *실패*를 예측하고 검증한다 (§5.2 → §6.4; per-dataset mean-anchor direction이 측정 가능하게 다른 곳을 가리킴, cos ≈ 0.47-0.62)."*

**§5.2 → §6.4 register harmonisation:**
- §5.2 Insight 2 "직접 확인된다" → "이론적으로 예측한다 ... 경험적으로 검증된다 (§6.4 Insight 1과 짝)" — pairs with §6.4 Insight 1's "이론적 예측 → 경험적 검증" pattern.

**§5.1 → §5.2 hand-off:**
- Added 1-sentence bridge: *"§5.1에서 정한 모델별 peak layer (4 archetype 매핑 + FastVLM·OneVision dataset-dependent caveat)를 기준으로 ... encoder-family-specific 위치를 *피해도* signal이 사라지지 않는지가 목표 질문이다."*

**§6.5 Table 7 verdict register:**
- All 6 rows now use Korean primary + English clarifier dual form ("방향 불일치 (direction mismatch)" / "동일 원인 (same single-direction redundancy issue)" / "probe 과적합 (probe overfit)" / "동일 근원 (decode-time direction = single-direction failure)" / "em 부수효과 + 학습 분포 편향 (em side-effect + training distribution bias)" / "**권장 — strict free-lunch 통과 (recommended)**"). Earlier draft of this fix had row 4 inline Korenglish + row 5 missing the English clarifier — corrected after advisor review.

**Figure 7 caption:** `... reasoning mode에서 collapse.` → `... reasoning mode에서 *붕괴*한다.` (consistent with §1.4 / abstract).

**§3.3 mech panel 4-archetype expansion:** "4 encoder archetype 분리." → "encoder family별 4-archetype 구성 (SigLIP-Gemma early / mid-stack cluster CLIP-ViT·ConvNeXt·InternViT / Qwen-ViT late / FastVLM late text-stealing — 자세한 매핑 §5.1)."

**§5.2 OneVision Phase E:** Reformatted from free-form paragraph into a 5th bullet matching the other 4 ablation modes — visual consistency.

**§4.7 / §4.5 Insight 3 italicised English "*generic tendency*"** → both occurrences switched to "일반 경향" (Korean).

**§3.3 raw-n disclaimer:** "...범위에 분포한다" added (closes adverbial "-으로" with main verb).

### Table edits

- **Table 2 (§4.1):** Bold cell convention updated. `df(a)` minimum 0.085 (Gemma4-31b) now bolded; `df(a) 0.094` Qwen2.5-VL un-bolded. Caption rewritten to disambiguate "robust by `adopt(a)`" vs "robust by `df(a)`". Source verified: `docs/insights/headline-numbers.md §B`.
- **Table 7 (§6.5):** Verdict column rows 1–5 reformatted to Korean primary + English clarifier dual form. No numeric changes.
- **Table 5 (§6.1):** No cell value changes; Round-1 bold convention preserved. §6.1 Insight prose now names the per-column bold winners ("LLaVA가 Δdf 측, ConvLLaVA가 Δem 측") — closes the Round-1 audit gap flagged in the Round-2 review's audit table.

### Figure edits

- **Figure 1 (§3.1):** No PNG change; in-text reference added in §3.1 paragraph closing.
- **Figure 5 (§4.4):** No PNG change; caption qualified for InfoVQA reversal + Table 2 robustness duality.
- **Figure 7 (§4.6):** No PNG change; caption "collapse" → "*붕괴*한다" (matches §1.4 / abstract Korean register).

## Rebuttals (DISAGREE class)

None. Every Round-2 must-fix and high-signal should-fix item resulted in a paper edit. The reviewer's diagnoses were uniformly accurate and the requested changes are improvements.

(Note: Round-1's Rebuttal 1 — "5/5 mech panel + OneVision pending" reframing — remains in force. Round-2 did not re-litigate the 6-model panel composition; the Round-2 paper carries forward the Round-1 panel-disambiguation language unchanged.)

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| Should-fix #5 §5.1 Insight 2 "split into 2/3 Insight boxes" | Single Insight intentionally fuses E1-patch concentration peak + OneVision calibration peak vs cross-dataset attention peak distinction (Round-1's disambiguation). Splitting risks re-fracturing the very disambiguation Round-1 closed. Length is the cost of getting the conflation removal right. | If a future round still reads it as overloaded after camera-ready prose pass, split then. |
| §4.5 Insight 3 per-bucket counts (b/c bucket cell counts) | Per-bucket cell counts not surfaced in any current canonical evidence file (`L1-confidence-modulation-evidence.md` discusses non-monotone cells qualitatively only). Fabricating counts would violate the no-fabrication rule. | Aggregation pipeline addition: extract per-cell `is_monotone` flag and cross-tabulate with (small_n / floor / chart_stack_outlier) labels. ~2 hrs CPU + script work. |
| Round-1 should-fix items still deferred (27-cell pilot grid, FLUX seed, max-tokens cross-ref) | Carried forward from Round-1; not surfaced again by Round-2. | Owners + estimates per Round-1 response document. |

## Open questions for next round

- **Figure 5 caption — "끌림" vs "anchor pull" vs "pull" alternation.** The reviewer flagged this as MINOR (stylistic, not technical). We did not normalise globally because the Round-1 / Round-2 patches were focused on grammar / fragment / register; a unified terminology pass at camera-ready would close this and similar style-only issues at once.
- **Whether §4.5 Insight 1's new closing sentence ("§6.2 mitigation 설계가 categorical wrong-base flag를 별도 입력으로 받지 않고 ...") is properly motivated by §4.5 alone.** The argument is: connecting confidence-continuum to universal-projection design is suggestive but is fully argued only after §6.2.3. If a future reviewer flags it as forward-leaning, hedge or move to §6.2.1 narrative.
- **Term "끌림 강도" vs "anchor pull" in §6.5 Insight (single occurrence in §6.5).** Currently "끌림 강도" is co-located with "single-direction" / "weight-space" English phrases — minor mixing, not corrected this round.

## Internal consistency check

After all edits:

- [x] **Abstract numbers still match §4–§7 tables.** Table 2 bold change is decorative only — `df(a) = 0.085` for Gemma4-31b, `0.094` for Qwen2.5-VL are the canonical CSV values, untouched. Abstract "first-evidence" replaces "입증" in the closing, no number affected. Macro +0.41 / HB +2.21 / POPE −0.06 / γ-β ×1.6 / ×2.9 / ×12.7 all unchanged.
- [x] **§1.5 contributions still match §4–§7 deliveries.** (4) 6-model panel single-layer null + multi-layer redundancy → §5.2; (5) single-direction failure predict → bypass → §5.2 + §6.2 + §6.4 + §7; (6) γ-β reasoning-amplifies-anchoring first-evidence → §4.6. All three traces preserved.
- [x] **§8.1 종합 still consistent with body.** All numbers untouched; closing sentence rewritten as full sentence with same content.
- [x] **All figure embeds still resolve to existing PNG paths.** No PNG paths modified.
- [x] **No figure or table renumbering.** Tables 1–8 + Figures 1–7 + A1 + B1–B2 + C1–C4 + F1 + G1 — all stable.
- [x] **Canonical sources still cited where appropriate.** Round-1 citations preserved + Round-2 abstract pointer "(§5.2 → §6.4)" + Table 2 caption metric disambiguation.
- [x] **"이분법" / "주력" / "발화" / "사지 *않으면서*" / "빈자리" / "회귀 (revert)" — all 6 forced-translation outliers eliminated from paper body.** Only the v4 changelog footer references them descriptively.
- [x] **"first-evidence" terminology — single canonical phrase across abstract / §1.4 / §1.5 / §4.6 Insight 2 / §8.1 / §8.2 limit 5.** Verified by grep.
- [x] **§4.1 Table 2 ↔ Insight 1 ↔ §4.3 Insight 1 ↔ §4.4 Insight 3 ↔ Figure 5 caption ↔ §1.4 robustness statements.** Now consistent: `adopt(a)` minimum is Qwen2.5-VL (0.021), `df(a)` minimum is Gemma4-31b (0.085), gap is 0.003 / 0.009 pp; downstream prose qualifies "강건" by metric where needed.
- [x] **Round-1 Rebuttal 1 (6-model mech panel composition) preserved.** Round-2 did not touch panel composition; abstract / §1.3 / §5.2 / §1.5 (4) all carry forward Round-1 language.

## Diff stat

- Lines: 524 → 526 (+2 lines, +0.4 %; the structural rewrites were length-neutral; the +2 came from §8.2 sub-block headers and the §5.2 → §6.4 bridge sentence).
- Post-advisor follow-up edits: 3 (Table 2 bold row-name + caption legend, Table 7 verdict register full pass, §4.3 Insight 2 forward-link upgrade).
- Sections fully rewritten: 0 (all changes are prose-level fixes within paragraphs).
- Tables modified: 2 (Table 2 caption + bold; Table 7 verdict column register). No cell value changes.
- Figures modified: 2 captions (Figure 5 anti-scaling qualifier; Figure 7 collapse → 붕괴). No PNG changes.
- Insight boxes touched: 9 (§4.1 Insight 1; §4.2 Insight 2; §4.4 Insight 1, Insight 2, Insight 3; §4.5 Insight 1, Insight 3; §4.6 Insight 1, Insight 2, Insight 3; §6.1 Insight; §6.5 Insight) — no Insight box removed; 4 reviewer-flagged "load-bearing-but-mangled" Insights restored to readable form.
- Korean translator-coinages eliminated: 6 ("이분법" ×6, "주력" ×2, "발화" ×2, "사지 *않으면서*" ×1, "빈자리" ×1, "회귀" revert-sense ×2).
- New canonical-source citations added: 0 (Round-2 was prose-pass, not numerical).
- Word count delta (rough): +120–150 Korean words (mostly the §4.1 Insight 1 metric-disambiguation expansion + §4.6 Insight 1 grammar rewrite + §1.4 controls qualifier).

**Single most impactful edit:** §4.6 Insight 1 grammar rewrite + hedge softening (Edit 2). The reviewer named this the sharpest critique — the only paragraph justifying §1.5 (6) "메커니즘 결합 증명" was grammatically broken Korean. The rewrite (a) restores subject-verb structure, (b) softens "직접 입증" → "강하게 뒷받침" matching the N=1 caveat, (c) adds explicit "× 12.7 ratio가 위치하는 정확한 row" pin, and (d) closes by threading back to §1.5 (6) deliverable. A reader of this paragraph now understands the argument and the limit in one pass.
