# Round 2 — Writing & Logical Flow Review

**Reviewer persona:** Editor-quality reader. Korean academic register + English technical terms.
**Paper version reviewed:** `docs/paper/emnlp_draft_ko.md` @ 524 lines, post-Round-1 revision (2026-05-09).
**Date:** 2026-05-09.
**Scope:** logical flow, terminology consistency, forward-reference fidelity, Insight-box clarity, awkward prose, table/figure-prose alignment, claim-evidence linking, Korean naturalness.
**Out of scope:** numerical correctness against canonical CSVs (handled by Round 1) — except where Round 1 fixed a number but the surrounding prose was left stale.

## Summary

Round 1's table-level fixes are applied, but several of those fixes leave the surrounding prose stale or internally inconsistent (Table 2 robustness ordering, §4.1 Insight 1, Figure 1 reference, §4.6 Insight 1). The Korean register is uneven: the v3 changelog (line 524) claims to have removed forced translations, but at least eight high-frequency awkward Korean coinages survive — most damagingly "이분법" (used 6 times in load-bearing sentences), "주력" (§6.2 title), "발화" (§6.1 Insight), "회귀" used in two distinct senses (revert / regression / backfire), and "끌림" alternating with "anchor pull" / "pull" without convention. The Insight boxes are unevenly weighted: ~30 boxes across §4–§7, of which I count 6 as filler or restating. The Round-1 bootstrap CI paragraph (§7) reads as bolted-on, breaking the §7 rhythm. The §8.2 limitations expansion from 4 to 7 bullets reads as a grab-bag of two registers. The §1.5 contributions list lost prose flow when (4) was rewritten — it is now a bullet train of run-on phrases.

The story remains publishable; the writing is one editing pass behind where it should be.

## Strengths

- **§4.3 Insight 3 (PlotQA un-mitigated free-lunch).** Verbatim: *"E7 panel의 *놀라움*: 7개 중 6개 모델이 PlotQA에서 `em(a) > em(b)` ... S1 cutoff가 anchor를 GT의 ±10 % 안에 두므로 anchor를 "그럴듯한 추측 단서"로 픽업하는 모델은 정확도를 *얻는다*. 이 패턴은 InfoVQA로 일반화하지 *않으며* ... *§6.2의 strict free-lunch mitigation이 이 baseline을 일반화 가능한 복구 메커니즘으로 변환하는 정확한 도메인*이다."* — load-bearing observation, threads §4.3 forward into §6.2 with a non-trivial mechanism claim. The cross-section pointer is explicit and accurate.
- **§6.2.1 Insight ((a − m) contrast의 핵심).** This is the single best Insight in the paper. Verbatim: *"§4.3의 (b, m, d) 통제 실험이 §6.2 subspace 설계를 직접 정당화*한다 — behavioral analysis가 mitigation 설계로 환원된 사례."* — explicit behavioral→mechanism→mitigation thread, no filler, terminologically clean.
- **§6.4 Insight 1 (예측 → 검증).** Verbatim: *"§5.2의 *multi-layer redundancy*가 single-direction 실패를 *이론적으로 예측*했다. §6.4는 그 예측의 *경험적 검증*."* — the paper's tightest mechanism→mitigation paragraph; backward-looking citation to §5.2 is exact.
- **§5.3 honest disclosure of OneVision dataset-dependent peak.** Verbatim: *"동일 encoder에서도 *데이터 분포*가 attention 분배를 바꾼다. 이 fragility는 single-direction mitigation이 cross-dataset에서 실패할 *추가* 이유를 보이며, subspace projection이 *모든 dataset의 signal을 결합 capture*해야 함 (§6.2)을 정당화한다."* — register is appropriately tentative, forward pointer is functional.

## Weaknesses (ordered by severity)

- **[CRIT-W1]** §4.1 Insight 1 prose contradicts Table 2 robustness column. Insight names "Qwen2.5-VL 가장 강건"; Table 2 row has Gemma4-31b-it df=0.085 < Qwen2.5-VL df=0.094, and the table caption "가장 강건한 모델"-bold is on Qwen2.5-VL, not Gemma4-31b-it. Either the bold convention is wrong or the prose is wrong; the reader cannot tell.
- **[CRIT-W2]** §4.6 Insight 1 contains an ungrammatical Korean sentence in a load-bearing position. Verbatim: *"증폭은 이분법이 가장 *덜* 취약하다고 한 *correct-base* row에 우선적으로 떨어진다."* The matrix verb "한" has no clear subject; "이분법이 ... 취약하다고 한"은 의미 불명. This is the *only* paragraph in §4.6 that justifies the "메커니즘 결합 증명" claim and a reader will stop here.
- **[CRIT-W3]** §1.5 contributions (4) — the Round-1 rewrite produced a string of phrases without grammatical glue. Verbatim: *"(4) 6-model 메커니즘 panel single-layer ablation null + multi-layer redundancy 분석 → (5) single-direction mitigation의 실패를 *예측한 후 우회*하는 multi-direction subspace projection + 6-benchmark capability preservation 검증 + (6) VLM 최초의 reasoning-amplifies-anchoring 결과 (γ-β, N=1 Qwen3-VL Instruct vs Thinking — first-evidence claim)."* Items (4)/(5)/(6) read as bullet labels strung by `+` rather than Korean sentences. Round 1 produced this when it patched in panel-disambiguation; the rhythm of (1)/(2)/(3) — "VLM 최초의 …", "gt-자유, baseline-relative …", "5 dataset × 7 model …" — is not preserved.
- **[CRIT-W4]** Korean register slippage: "이분법" used 6 times for what is everywhere else a "wrong/correct stratification" or "wrong/correct split". The word "이분법" in Korean academic register reads as "binary thinking" / "false dichotomy" — a *philosophical* term — and lands wrong here. §4.5 Insight 1 ("이분법의 재해석"), §4.6 Insight 1 ("이분법이 가장 *덜* 취약하다고 한"), §8.1 ("Phase-A 이분법은 confidence gradient의 거친 projection") all use it.
- **[MAJOR-W5]** §6.2 title — "**E6 — Residual-stream subspace projection (deployable, 주력)**". "주력" here means "main / flagship method" but in Korean academic register this reads as a wartime ("주력 부대") or commercial ("주력 상품") metaphor, not a method label. It is a forced Korean translation exactly of the type the user explicitly disliked.
- **[MAJOR-W6]** §6.1 Insight (Free-lunch의 메커니즘적 의미) — "발화" used metaphorically for `forward pass / hook fires`. Verbatim: *"hook은 single-image inference에 누출되지 않으며 두 번째 이미지에 가독 digit이 없으면 발화는 하지만 제거할 signal이 없다."* "발화 (utterance)" is a linguistic term; here it labels a hook firing on hidden states. A Korean reader will pause at "발화는 하지만". §6.3 has the same construction: *"Projection은 그래도 발화하는데..."*
- **[MAJOR-W7]** Round 1's §7 bootstrap CI paragraph is bolted on. The §7 prose currently runs Table 8 → result paragraph → **"95 % CI 산출."** stand-alone bold paragraph → Insight box. The bold heading "95 % CI 산출." with a period and bold-period is not used anywhere else as a paragraph label and reads as a memo to self. The methodology should be folded into the result paragraph as a parenthetical or moved to Appendix B (where M2 metric variants already live).
- **[MAJOR-W8]** §8.2 limitations is now 7 bullets in two registers. The original 4 — "단일 prompt", "Open-weight 모델만", "Human baseline 부재", "Mid-stack cluster 단일" — were one-line topical limitations. Round 1 appended 4 more — "γ-β N=1", "small-n CI 미보고", "27-cell pilot grid", "OneVision E1d analyzer" — which are *operational debt items* (after-the-fact accounting of what was deferred). The list now mixes "what we acknowledge as a research scope" with "what we owe the next revision" without separation. Either split into "Limitations" + "Deferred follow-ups" or rewrite the new 4 to the same register as the original 4.
- **[MAJOR-W9]** Forward reference §1.2.3 → §4.5 mismatch. §1.2 pillar 3 "Confidence-modulated" describes the gradient claim using "Wrong/correct 이분법은 이 연속 구조의 거친 projection에 불과하다" but §4.5 prose later opens with "**§4.1의 wrong/correct 이분법은**" — the same prose claim is made twice without the §1.2.3 version flagging that §4.5 will *re-derive* it from quartile data. As-is, §4.5 reads as repetition rather than delivery.
- **[MAJOR-W10]** §4.4 Figure 5 caption vs Insight 2 contradiction in granularity. Figure 5 caption: *"gemma3-4b가 chart/plot/math에서 가장 큰 끌림, qwen2.5-vl이 가장 강건."* Insight 2 (Anti-scaling): *"Gemma3-4b가 PlotQA에서 *27B보다 더 끌린다* (0.395 vs 0.227). 그러나 InfoVQA에서는 4B (0.324) < 27B (0.350)로 역전한다."* The caption claims gemma3-4b dominates "chart/plot/math" but the Insight says InfoVQA *reverses*. Either include InfoVQA in the caption qualification or rewrite the caption to match.
- **[MAJOR-W11]** §3.1 Figure 1 has no in-text reference. The §3.1 paragraph ends "...자극 inventory는 ... `d`)이다 (부록 §A)." then the Figure 1 embed appears below with no `(Figure 1)` mention in the prose. This is the only Figure in the paper with no in-text reference (every other figure is `(Figure N)`-cited before its embed). Either prose should add "(Figure 1 참조)" or reorder.
- **[MAJOR-W12]** §6.2 title says "주력"; §1.3 / abstract / §6.6 instead use "deployable mitigation으로 권장". Pick one. The §6.2 title is the only place "주력" is used; the rest of the paper consistently calls E6 "deployable" / "권장".
- **[MAJOR-W13]** "회귀" semantic overload. The paper uses 회귀 in three distinct senses without disambiguation: (a) §1.2 / §1.4 / abstract "anchor 쪽으로의 회귀" (regression-toward-anchor, used in cognitive sense), (b) §1.2.2 / abstract "generic distractor 수준으로 회귀" (revert), (c) §6.4 "ChartQA 역행 +56 %" (backfire — uses 역행 instead, fine). But (b) "수준으로 회귀" reads ambiguously: a Korean reader will see "regression" in the statistical sense and pause. Suggest "수준으로 *되돌아간다*" or "수준으로 *내려간다*" for (b).
- **[MINOR-W14]** "끌림" vs "anchor pull" vs "pull" — alternates without rule. §4.1 Insight 1 has "끌림 크기"; §4.6 Insight 1 has "끌림을 증폭"; abstract has "anchor pull을 *증폭*"; §1.4 has "anchor robustness"; §6.5 Insight has "끌림 강도". The difference is clearly stylistic, not technical, but the inconsistency is visible.
- **[MINOR-W15]** §4.5 sentence-end clause is a translator-fragment. Verbatim: *"본문 headline은 `log_prob_sum`을 보고하고 부록에서 두 proxy를 모두 표로 제공 — 현 운영적 confidence proxy 선택에 둘 다 정렬되도록 (출처 ...)"* The trailing "둘 다 정렬되도록 (출처 ...)" is grammatically incomplete — "정렬되도록" demands a verb that never arrives.
- **[MINOR-W16]** §4.6 Insight 3 parenthetical is too long for a sentence opener. Verbatim: *"Thinking은 acc(d) (no-anchor neutral arm baseline proxy, γ-β 단일-stratum 설정에서 별도 b-arm run 없음 — d-arm correct fraction 사용; Instruct 0.647 = 249/385, Thinking 0.587 = 226/385 출처 `experiment_e5e_mathvista_reasoning_per_cell.csv`)도 더 *낮다*."* The 60+ character parenthetical inserted between subject "Thinking은 acc(d)" and verb "도 더 낮다" breaks reading flow. Round 1 patched the label correctly but never re-flowed the sentence. Move parenthetical content to a footnote or rewrite as: "Thinking은 acc(d)도 더 낮다 (Instruct 0.647 = 249/385, Thinking 0.587 = 226/385; γ-β 단일-stratum 설정에서 d-arm correct fraction을 b-arm baseline proxy로 사용)."
- **[MINOR-W17]** §4.7 sentence "L1 monotonicity의 *generic tendency*는 6/7 main panel 모델에서 유지되며" — "generic tendency" is italicised English mid-Korean-sentence with no precedent term elsewhere in the paper. Either drop italics ("L1 monotonicity가 6/7 main panel 모델에서 일반 경향으로 유지되며") or make "generic tendency" a defined term in §4.5 first.
- **[MINOR-W18]** §1.5 references "(γ-β, N=1 Qwen3-VL Instruct vs Thinking — first-evidence claim)". "first-evidence claim" appears only here — §4.6 / §8.2 / abstract say "VLM 최초", "first VLM result", "VLM 최초의 reasoning-amplifies-anchoring 결과". "First-evidence claim" is fourth synonym for the same thing.
- **[MINOR-W19]** §3.3 — the new "**Mechanism panel 6 모델**: gemma4-e4b, llava-1.5-7b, ConvLLaVA-7b, InternVL3-8b, qwen2.5-vl-7b, fastvlm-7b — 4 encoder archetype 분리." The "4 encoder archetype 분리" qualification is dropped in here without explanation; §5.1 expands it. As written it reads telegraphically. Either move the explanation forward or drop the qualifier from §3.3.

## Specific issues by axis

### A. Logical flow

- **§1 → §4 promise/delivery.** §1.2 lists three pillars (graded / digit-pixel / confidence-modulated). §4 delivers all three but the *order* in §4 is graded (§4.1) → distance/plausibility (§4.2) → digit-pixel (§4.3) → main matrix overview (§4.4) → confidence (§4.5). §1.2 listed digit-pixel as pillar 2 and confidence as pillar 3, so §4.2 (distance/plausibility — which is *not* one of the three §1.2 pillars) and §4.4 (matrix overview — likewise not one) come *between* the pillars. §1.2 should either announce the four-element structure (graded / plausibility / digit-pixel / confidence-modulated) or §4.4 should be moved before §4.1 as setup.
- **§5.1 → §5.2 hand-off.** §5.1 Insight 2 names "OneVision Main의 *digit-bbox concentration peak*는 calibration set (VQAv2)에서 L20-L23 cluster" — Round 1's disambiguation. But §5.2 then opens "6-model 메커니즘 panel ... × 6 ablation mode" without ever referring back to the 4-archetype peak organisation. A reader leaving §5.1 with "encoder family chooses peak depth" finds §5.2 silently ablating peak-by-peak. One sentence at the start of §5.2 — *"§5.1에서 정한 model별 peak layer를 기준으로 ablate한 결과"* — would close the loop.
- **§5.2 → §6.4 prediction-verification chain.** §5.2 Insight 2 says single-direction mitigation failure is "*예측*" then "*직접 확인*된다 §6.4." §6.4 Insight 1 reciprocally says "§5.2의 *multi-layer redundancy*가 single-direction 실패를 *이론적으로 예측*했다." This is the strongest argumentative chain in the paper but the phrasing is asymmetric: §5.2 says "직접 확인" (direct empirical confirmation) and §6.4 says "이론적 예측" (theoretical prediction). Pick a register for both directions ("예측 ↔ 검증" or "예언 ↔ 확증").
- **§6.6 closing line repeats §1.3 / §6.2 / abstract.** The full sentence — *"본 논문은 **E6를 `llava-onevision-qwen2-7b-ov` 위 cross-dataset numerical anchoring 환경의 deployable mitigation으로 권장**한다"* — is the third recommendation in close proximity. Removing it does not weaken the section.

### B. Terminology consistency

| Term variant 1 | Term variant 2 | Term variant 3 | Locations | Suggested canonical |
|---|---|---|---|---|
| 끌림 | anchor pull | pull | §4.1 Insight 1 / Table 2 caption / Fig 5 caption / §4.6 Insight 1 (끌림); abstract / §4.6 / §6.5 Table 7 (anchor pull); §1.4 / §4.4 Insight 3 (pull) | `anchor pull` (English) — consistent with abstract |
| wrong/correct 이분법 | wrong-base/correct-base stratification | base_correct stratify | §4.5 / §4.6 Insight 1 / §8.1 (이분법); §3.3 / §4.1 (stratify); §4.1 (`base_correct`) | "wrong-base / correct-base 분할" or "(b)-base 정/오 분할" |
| 회귀 (revert) | 회귀 (regression) | 역행 (backfire) | §1.2.2 / abstract (revert); §1.1 / §1.2.1 (regression-toward-anchor metric); §6.4 (backfire); abstract `(*회귀*)` x2 | revert→`되돌아감`; regression→`회귀형 이동`; backfire→`역행`/`backfire` |
| 주력 | deployable | 권장 | §6.2 title (주력); abstract / §1.3 / §6.6 (deployable); §6.6 / §6.5 Table 7 (권장) | drop "주력"; use `deployable` |
| 발화 (hook fires) | (hook 동작) | (forward pass) | §6.1 Insight / §6.3 first paragraph (발화) | drop "발화" — use "hook이 트리거되지만" or "forward에서 동작하지만" |
| 사상 / 사영 / projection | mapping (linear-algebra sense) | — | only "projection" appears in §4.5 / §6 / §8.1 — `사상` correctly removed per v3 changelog | (no fix needed — kept clean) |
| Figure 5 "끌림" / Figure 7 "collapse" | — | — | Fig 5 caption (Korean), Fig 7 caption (English) | either Korean ("붕괴") or English ("collapse") consistently across captions |
| first VLM result / VLM 최초 / first-evidence claim | — | — | §1.5 (first-evidence claim); abstract (입증); §4.6 Insight 2 (첫 재현); §8.1 (첫 VLM 결과) | "VLM 최초 first-evidence" — pick one and use globally |
| 강건 / robust / robustness | — | — | §4.1 Insight 1 (강건); §4.4 Insight 3 (robustness); abstract (robustness) | English `robustness` once defined; otherwise "강건성" |

### C. Forward references — promises vs delivery

| Promise (where) | Delivery (where) | Match? |
|---|---|---|
| §1.2 pillar 1 "Graded vs categorical" | §4.1 Table 2 | ✅ pillar matches Table 2 prose |
| §1.2 pillar 2 "Digit-pixel causality" | §4.3 Table 3 + Fig 4 | ✅ pillar delivered |
| §1.2 pillar 3 "Confidence-modulated" | §4.5 Fig 6 | ⚠️ §4.5 first sentence repeats §1.2.3 verbatim ("wrong/correct 이분법은 ... 거친 projection") rather than building on it — see [MAJOR-W9] |
| §1.3 "single-layer ablation 6/6 null" | §5.2 | ✅ matches after Round-1 panel-disambiguation |
| §1.3 "multi-direction subspace ... cross-dataset" | §6.2 | ✅ |
| §1.3 "6-benchmark capability ... +0.41 pp" | §7 Table 8 | ✅ |
| §1.4 "Reasoning trace가 정확도를 사지 *않으면서*" | §4.6 Insight 3 (acc(d)) | ⚠️ §4.6 Insight 3 prose buries the link in a 60-char parenthetical (see [MINOR-W16]); a reader of §1.4 will not know that §4.6 Insight 3 is the citation |
| §1.5 (4) "6-model 메커니즘 panel single-layer ablation null + multi-layer redundancy 분석" | §5.2 | ⚠️ §5.2 prose says "Single-layer ablation: 6/6 모델 null" + "Multi-layer redundancy" appears as Insight 1 conclusion, but never the connector "ablation null *+ redundancy 분석*" matching §1.5 — §5.2 does the analysis but doesn't *frame* it as the §1.5 (4) deliverable |
| §1.5 (5) "single-direction mitigation의 실패를 예측한 후 우회" | §6.4 + §6.2 | ✅ §6.4 Insight 1 explicitly calls out the predict→verify chain |
| §1.5 (6) "VLM 최초의 reasoning-amplifies-anchoring 결과 (γ-β, N=1 Qwen3-VL ... — first-evidence claim)" | §4.6 + §8.2 limitation 5 | ⚠️ §4.6 prose "텍스트 LRM 문헌의 ... VLM에 일반화" softer than abstract "VLM에 일반화됨을 *입증*"; §8.2 limitation says "first-evidence 결과로, cross-architecture 일반화 ... 후속 라운드" — three different framings of the same claim |
| Abstract "5개 비교 방법 ... 중 *유일하게* 통과" | §6.5 Table 7 | ✅ Table 7 row count = 5 baselines + 1 ours; "유일" defended by "1/4 임계 미달" verdict on Query-adaptive / CogBias |
| Abstract "텍스트 LRM 문헌의 reasoning-amplifies-bias 현상이 VLM에 일반화됨을 *입증*" | §4.6 Insight 2 | ⚠️ §4.6 Insight 2 says "첫 재현" — much weaker than abstract "*입증*" — and §8.2 lim 5 says "first-evidence ... cross-architecture 일반화 후속 라운드". The abstract's "입증" overclaims relative to body. Soften abstract to "첫 VLM 재현" or upgrade body. |

### D. Insight clarity

For each Insight box in §4–§7. Class: load-bearing / restating / filler.

| Section | Insight # | Class | Note |
|---|---|---|---|
| §4.1 | 1 (역상관) | restating | "능력이 낮은 곳에서 끌림이 크다" — ordering visible in Table 2; prose-Table mismatch (see [CRIT-W1]) makes this load-bearing-but-broken |
| §4.1 | 2 (Mussweiler-Strack) | load-bearing | Connects Table 2 to cognitive-science prior, reframes the result |
| §4.2 | 1 (adopt vs df 분리) | load-bearing | Names two distinct gates ("admission" vs "sub-threshold pull") — non-trivial |
| §4.2 | 2 (em이 거리에 안정) | load-bearing | Forward-points to §6 with a specific design implication ("거리 분포를 재보정할 필요가 없음") |
| §4.3 | 1 (단조 ordering) | load-bearing | Establishes (a − m) gap as effect-size proxy — reused in §6.2.1 |
| §4.3 | 2 (E7 7-model 일반화) | restating | Just states "+0.014 ~ +0.139" range — would be enough as caption |
| §4.3 | 3 (PlotQA un-mitigated free-lunch) | load-bearing | Best Insight in §4 — see Strengths |
| §4.4 | 1 (효과의 보편성) | restating | "df 부호가 30/30 cell 모두 양" is exactly Figure 5 caption verbatim; the "토대" sentence does not add a new claim |
| §4.4 | 2 (Anti-scaling) | load-bearing | Specific contrarian observation (4B > 27B on PlotQA) with a concrete mechanism guess; conflict with caption noted at [MAJOR-W10] |
| §4.4 | 3 (Encoder family ordering) | load-bearing | Forward-points to §5 mechanism with a falsifiable ordering |
| §4.5 | 1 (이분법의 재해석) | restating | Same claim as §1.2.3 — see [MAJOR-W9] |
| §4.5 | 2 (Categorical capture 기각) | load-bearing | Provides a falsifiable contrast with explicit numbers (0.037 → 0.060 → 0.137 → 0.183) |
| §4.5 | 3 (Non-monotonic cell의 분류) | filler | "세 bucket으로 분류 가능" is ad-hoc post-hoc rationalisation — reads as defensive disclosure rather than insight; the closing sentence "*메커니즘 자체의 한계*가 아닌 *측정 정밀도 + 두 boundary case*의 결과" is unsupported by the paragraph itself (no per-bucket counts given). Either drop or move to §4.7 |
| §4.6 | 1 (메커니즘 결합 증명) | load-bearing-but-broken | The substantive content is the strongest argumentative move in the paper, but the second sentence is ungrammatical (see [CRIT-W2]) |
| §4.6 | 2 (LRM 문헌 정렬) | load-bearing | Single sentence; correctly attributed to Wang et al. [2025a]; serves bridge function |
| §4.6 | 3 (acc(d)도 낮다) | load-bearing | Round-1 patched the label; the parenthetical now overruns the sentence (see [MINOR-W16]) |
| §5.1 | 1 (Encoder가 위치를 정한다) | load-bearing | Names the canonical claim, ties it to mitigation design |
| §5.1 | 2 (E1-patch — digit pixel 자체) | load-bearing-but-rambling | Round-1 disambiguation made this paragraph 4 sentences long with two CSV citations and a §5.3 forward pointer; trying to do too much in one Insight |
| §5.2 | 1 (Peak ≠ causal site) | load-bearing | Crisp negative result with general-sounding mechanism implication |
| §5.2 | 2 (Single-direction failure 예측) | load-bearing | The strongest predict→verify Insight in the paper |
| §5.2 | 3 (Upper-half re-weighting 가능) | load-bearing | Direct motivation for §6.1 E4 |
| §6.1 | 1 (Free-lunch 메커니즘적 의미) | load-bearing-but-mangled | "발화" usage (see [MAJOR-W6]) and trailing em-dash chain make this hard to parse |
| §6.2.1 | 1 ((a − m) contrast 핵심) | load-bearing | Best Insight in §6 (see Strengths) |
| §6.2.3 | 1 (Effect size correlates with baseline) | load-bearing | Specific claim ("PlotQA −5.2, TallyQA −0.3"), forward-points to operational implication |
| §6.2.3 | 2 (단일 hyperparameter의 의미) | load-bearing | Resolves §5.3 dataset-dependent peak vs §6.2 single-(L,K,α) tension explicitly |
| §6.3 | 1 (두 효과의 분리) | load-bearing | Honestly disambiguates "anchor mitigation" claim from "general debiasing" — protects the strict-free-lunch claim |
| §6.3 | 2 (Capability preservation 사전 신호) | load-bearing | Frames §7 as confirmation of §6.3 prediction; tightens §6→§7 thread |
| §6.4 | 1 (예측 → 검증) | load-bearing | See Strengths |
| §6.4 | 2 (K=8 trade-off) | load-bearing | Names the K-selection rationale with specific failure modes for K=2/4/16 |
| §6.5 | 1 (single-direction vs weight-space failure modes) | load-bearing | Frames Table 7 as "two failure modes ... uniquely bypassed" — the right closing argument |
| §7 | 1 (Anchoring 외 hallucination 일반) | load-bearing | Names the cross-pattern hypothesis (anchor pull ↔ illusion-mode hallucination) and points to follow-up; appropriately tentative |

**Class totals: 21 load-bearing, 4 restating, 1 filler, 4 load-bearing-but-mangled.**

The mangled ones (§4.6 Insight 1 grammar; §4.6 Insight 3 parenthetical; §5.1 Insight 2 multi-clause; §6.1 Insight "발화") are all post-edit artefacts — substantively correct, surface-level rough.

### E. Awkward prose / register

- **§3.3 raw n disclaimer is a translator-fragment.** Verbatim: *"위 raw n은 stratification·eligibility 필터 *이전* count이며, 실제 본문 표에 사용된 per-cell n은 stratified 부분집합으로 ChartQA 129–517 / TallyQA 6,934–14,772 / PlotQA 926–4,610 / InfoVQA 218–865 / MathVista 127–274 (모델별 변동, 출처 ...)"* — reads as one 80-character sentence with no main verb after "n은 stratified 부분집합으로". Issue: "...부분집합으로 ChartQA 129–517..." ends in adverbial form "-으로" then lists numbers; needs a main verb. Suggested fix: *"실제 본문 표에 사용된 per-cell n은 stratified 부분집합 기준으로 ChartQA 129–517 ... MathVista 127–274 *범위에 분포한다*."*

- **§4.2 final paragraph nominalisation stack.** Verbatim: *"S1→S5에서 em이 안정한 것은 *거리는 em을 손상시키지 않음*을 보인다. anchor가 거부되면 d-arm em 비율로 복귀하고, 채택되면 redirect cost는 거리에 무관하게 지불된다."* The mid-sentence English-style nominalisation "*거리는 em을 손상시키지 않음*을 보인다" reads as direct translation of "distance does not damage em". Fix: *"S1→S5에서 em이 안정하다는 것은 *거리가 em을 손상시키지 않음*을 의미한다."* (just `이` → `이라는 것은`).

- **§5.2 enumerated list mid-paragraph.** The four bullets `- Single-layer ablation: 6/6 모델 null. - Lower-half ablation: heterogeneous ... - Upper-half ablation: 6/6 모델 −4.0~−10.5 pp - Full ablation: −5.0~−12.0 pp.` are followed by a free-form sentence "OneVision Main에 대한 4-dataset 확장 ... 통합." and then the Insight 1/2/3 boxes. The "OneVision Main에 대한" block is structurally a 5th bullet but presented as a paragraph; format is inconsistent.

- **§6.3 first paragraph.** Verbatim: *"b-arm은 `target_only` — 단일 이미지 + 질문, 두 번째 anchor 이미지 *없음*. Projection은 그래도 발화하는데, universal이라 input에 anchor가 있는지 *모르기* 때문이다."* The "발화하는데" + "universal이라" + "*모르기* 때문이다" sequence is a tangle. Fix: *"Projection은 anchor 유무와 무관하게 모든 forward에서 작동한다 (universal projection — input이 anchor를 포함하는지 알지 못함)."*

- **§6.5 Insight closing line.** Verbatim: *"이는 우연이 아니라 §5.2 multi-layer redundancy와 §6.2.1 (a − m) contrast 설계의 직접 결과."* — sentence-fragment ("결과." with no verb). Fix: *"이는 우연이 아니라 §5.2 multi-layer redundancy와 §6.2.1 (a − m) contrast 설계의 직접 결과*이다*."*

- **§8.1 종합 closing line.** Verbatim: *"reasoning trace가 bias가 *축적*되는 장소가 됨을 시사하는 첫 VLM 결과."* — particle stack 가 / 가 / 됨 / 을 / 첫 / 결과 reads garbled. Fix: *"reasoning trace*에서* bias가 *축적*된다는 것을 시사하는 첫 VLM 결과*이다*."*

- **§4.4 Insight 1 final clause.** Verbatim: *"이것이 *mitigation의 가능성*을 본 논문이 추구할 수 있는 토대."* — fragment without verb (`...토대.`). Fix: *"이것이 본 논문이 *mitigation*을 추구할 수 있는 토대*이다*."* (also drop the redundant *가능성*).

- **§6.6 first sentence run-on.** Verbatim: *"§6.1 (E4)와 §6.2 (E6)는 *상보적*이다. E4는 *mechanism* mitigation — attention pathway upper-half에서 동작. E6는 *representational* mitigation — residual stream에서 동작."* The second and third sentences both end in "동작." — present tense without 이다 (`동작한다` or `동작하는 mitigation이다`). Currently reads telegraphically.

- **§4.6 Insight 2 register.** Verbatim: *"*reasoning mode는 robustness 보강이 아니라 회귀의 위험원*이라는 운영적 함의."* — fragment ("*함의*."). Fix: *"... 회귀의 위험원이라는 *운영적 함의*가 도출된다."*

### F. Table / figure / prose alignment

- **Table 2 (§4.1) bold convention vs Insight 1 prose mismatch.** Table caption: *"Bold = 가장 강한 끌림 모델 / 가장 강건한 모델."* Bolded rows: Gemma4-e4b (df 0.274 — strongest) and Qwen2.5-VL-7b (df 0.094). But Gemma4-31b-it has df 0.085 < 0.094, so it is the most robust by `df(a)`. Insight 1 prose says "Qwen2.5-VL 가장 강건". *Either* the bold convention is wrong (should bold Gemma4-31b-it) *or* the metric for "robust" in Insight 1 is not df (e.g., adopt rate — Qwen2.5-VL 0.021 < Gemma4-31b 0.024 by 0.003 pp); but the prose / caption do not say which. **Match: ❌**.

- **Table 3 (§4.3) row 3 / row 6 zero-effect rows.** Verbatim row "VQAv2 | qwen2.5-vl-7b | 0.070 | 0.066 | +0.4 pp" and "TallyQA | qwen2.5-vl-7b | 0.033 | 0.037 | −0.5 pp" — these are the only non-bolded rows because Qwen2.5-VL is at the floor (per §4.3 Insight 1). The "−0.5 pp" *negative* gap is not addressed in any prose paragraph; a reader will wonder whether negative (a − m) gap is meaningful (anti-anchor effect?) or noise. The Insight 1 closing — *"양 arm이 모두 floor에 위치"* — implicitly waves it off but does not say "−0.5 pp is within noise". **Match: ⚠️**.

- **Table 5 (§6.1) post-Round-1 bold.** After Round-1 swap, bold row labels are "**LLaVA-1.5-7b**" and "**ConvLLaVA-7b**" with bold cells in different columns (LLaVA on Δdf, ConvLLaVA on Δem(a)). The caption says "Bold = 열 단위 가장 큰 효과" — per-column bold convention. The prose in §6.1 (which is one paragraph plus the Insight) does not call out which model wins which column; the reader has to deduce from the bold convention and the cell values. Surrounding §6.1 prose (verbatim) only says "Mid-stack cluster 3 모델 (LLaVA-1.5 / ConvLLaVA / InternVL3) Phase 2 full validation" — neutral. **Match: ⚠️ but defensible** (consider adding "LLaVA가 Δdf 측, ConvLLaVA가 Δem 측에서 각각 최대" to the prose).

- **Figure 1 (§3.1) — no in-text reference.** The §3.1 paragraph builds Table 1 then jumps to Figure 1 without "(Figure 1)" — see [MAJOR-W11]. **Match: ❌**.

- **Figure 5 caption vs §4.4 Insight 2.** Caption claims gemma3-4b dominates "chart/plot/math"; Insight 2 reverses for InfoVQA — see [MAJOR-W10]. **Match: ❌**.

- **Figure 7 caption.** Verbatim: *"H2 wrong > correct asymmetry가 reasoning mode에서 collapse."* English `collapse` mid-Korean caption; abstract uses "*붕괴*하다". Inconsistency only in this one caption. **Match: ⚠️**.

- **Figure C1 caption — VQAv2 wrong-base S1 anchor=0.138 vs masked=0.082.** This is the value Round 1 said was wrong on the *TallyQA gemma3-27b* row, but it *is* canonical for the *VQAv2 gemma3-27b* row (verified against `_data/E5c_per_cell.csv`: 0.1385 / 0.0816). Caption is correct. **Match: ✅**.

- **Table 6 (§6.2.3) "평균" row label.** The footer row label is `**평균**` but the column headers `n_paired | Δ adopt(a) | ...` have no n_paired total — leaving "n_paired = empty" in the average row. Implicit (averages are unweighted), but standard table convention is to mark "—" or "weighted" explicitly.

- **Table 7 (§6.5) verdict column register mismatch.** Verbatim verdicts: `direction mismatch | 동일 원인 | probe overfit | 동일 근원 | em 부수효과; gt 분포 학습 편향 | 권장 (strict free-lunch)`. Three are English ("direction mismatch", "probe overfit"), one is Korean+English ("em 부수효과; gt 분포 학습 편향"), one is Korean ("권장 (strict free-lunch)"). Pick one register. **Match: ⚠️**.

### G. Claim-evidence linking

- **§1.4 verbatim:** *"같은 자극에서 Qwen3-VL-8B-Thinking은 Instruct 변형 대비 adopt ×1.6, df ×2.9 — 그러나 *correct-base* 부분집합에서 df 비율은 **×12.7**로, main panel 전반에서 유지되는 H2 wrong > correct asymmetry가 *붕괴*한다. Reasoning trace가 정확도를 사지 *않으면서* anchor robustness를 *낮춘다*는 직접 증거."* — claim is "직접 증거" (direct evidence). Evidence cited in same paragraph: the ×12.7 ratio. Gap: *direct* evidence in cognitive-mechanism literature requires controlling for confounds (e.g., longer trace = longer parsing window = more chance for OCR pickup). The §1.4 paragraph does not mention the §4.6 Insight 3 acc(d)-stays-lower control that makes this argument live. Suggest qualifier: *"... *낮춘다*는 첫 cross-arm 증거 (controls in §4.6)."*

- **§1.5 (5) verbatim:** *"single-direction mitigation의 실패를 *예측한 후 우회*하는 multi-direction subspace projection."* — claim is "예측한 후 우회" (predicts, then bypasses). Evidence: §5.2 + §6.4. Strong claim warrants softer hedge in Insight; current Insight is OK. But abstract's "메커니즘 측에서 ... single-direction (LEACE/ActAdd) mitigation의 cross-dataset *실패*를 예측하고 검증한다" parallels §1.5 (5); both should have the same citation footing. Currently abstract has no parenthetical pointer to §6.4. Suggest abstract addition: *"... *실패*를 예측하고 검증한다 (§5.2 → §6.4)."*

- **§4.6 Insight 1 verbatim:** *"§4.5의 L1 monotonicity가 *통계적 인공물이 아닌 mechanism-bound*임을 직접 입증한다."* — claim is "직접 입증" (directly proves). Evidence: γ-β ratio (×12.7 on correct-base). Gap: ×12.7 establishes correlation between reasoning trace and anchor pull, *not* that L1 is mechanism-bound. The chain "L1 is mechanism-bound" → "manipulating reasoning trace amplifies anchor" is a fair *consistency* argument but not direct proof. Suggest soften: *"... mechanism-bound임을 *뒷받침한다*."* or *"... 가설과 *일치*한다."*

- **§5.2 Insight 1 verbatim:** *"*attention peak이 가장 큰 mass를 가진다*는 사실이 그 layer가 *causal site*임을 의미하지 *않는다*."* — solid hedge, evidence in same section, claim ↔ evidence well-bound.

- **§7 Insight verbatim:** *"K=8 subspace projection이 두 패턴 모두 건드린다는 것은 *VLM hallucination의 일부가 본 논문 anchoring mechanism과 representation space를 공유*함을 시사 — 후속 연구의 직접 진입점."* — properly hedged ("시사", not "입증") and forward-points to follow-up. Good.

- **Abstract verbatim closing:** *"텍스트 LRM 문헌의 reasoning-amplifies-bias 현상이 VLM에 일반화됨을 *입증*."* — claim "*입증*" (proves) is too strong for N=1 architecture pair. §8.2 lim 5 names this exact concern: *"single architecture pair ... cross-architecture 일반화 ... 후속 라운드"*. Abstract should say "첫 VLM 재현" or "first-evidence claim" matching §8.2's hedge. See also [Round 1 limitations / contribution claim consistency].

### H. Korean naturalness

The v3 changelog (line 524) claims forced translations were cleaned up. Verified the three named ("거리감쇠", "사상", "0결과") are absent — ✅. But several other forced/awkward Korean coinages survive:

| Surviving awkward term | Locations | Issue | Suggested fix |
|---|---|---|---|
| **이분법** | §1.2.3 / §4.5 / §4.5 Insight 1 / §4.6 §4.6 Insight 1 / §8.1 (6 occurrences) | Reads as philosophical "false dichotomy" rather than statistical "binary stratification"; user explicitly disliked translator-style Korean coinages | "wrong/correct 분할" or "wrong-base/correct-base stratification" |
| **주력** | §6.2 title (1) | Wartime/commercial metaphor; doesn't appear elsewhere — outlier | drop; rely on "deployable" (already in title) |
| **발화** | §6.1 Insight / §6.3 first paragraph (2) | Linguistic "utterance"; misuse for "hook fires / forward pass executes" | "hook이 트리거되지만"; "forward에서 작동하지만" |
| **회귀** (revert sense) | §1.2.2 / abstract pillar (iii) (2) | Ambiguous with statistical regression elsewhere in paper | "되돌아감" or "내려감" |
| **수치 단서** | §1.1 / §2 (2) | Tries to translate "numerical cue/anchor"; works but non-standard | could keep — borderline; flag for camera-ready review |
| **품 / 발화 / 누출 / 누수** | §4.2 Insight 1 ("누출"), §6.1 Insight ("누출"), abstract (none) | Vague metaphor for "signal leak"; English "leak" / "leakage" reads cleaner here | "leak" or `signal이 누출된다` → `signal이 새어 나간다` |
| **빈자리** | §4.4 Insight 2 ("visual reasoning 빈자리") | Casual register; "공백" or "한계" more academic | "*visual reasoning capability gap* → 두 번째 이미지 digit 의존" |
| **사지 *않으면서*** | §1.4 ("정확도를 사지 *않으면서*") | "사다 (buy)" as metaphor for "improve"; unusual register | "정확도를 *얻지 못하면서*" or "정확도 향상 없이" |
| **비추론 변형** | abstract / §4.6 (2) | "비추론" works; "변형" literally translates "variant" but reads as "modification" — slightly off | "비추론 mode" or "non-Thinking variant" |
| **결합형** | (not used post-Round-1, OK) | — | — |

Additional naturalness issues (sentence-rhythm, not term-level):

- **§4.4 Insight 2 final line.** *"단순 "큰 모델 = 강건" 직관과 어긋나며, 데이터셋의 *visual complexity*가 모델 크기보다 robustness를 더 결정함을 시사한다."* — long sentence with three clauses joined by `,` and `며`; the "직관과 어긋나며" + "결정함을 시사한다" register switch from colloquial ("직관") to formal ("시사한다") is jarring.
- **§4.5 first paragraph.** Three independent claims (panel scope, Q4-Q1 gap, monotonicity count, source citation) glued into one ~120-character sentence. Period-then-newline at "...51 / 85 (60 %)** (Figure 6)." would read better.
- **§6.2.1 method paragraph.** Code blocks intersperse Korean prose: `D[i, L, :] = h(...)` then `truncated SVD D[:, L, :] = U_L Σ_L V_L^T` then `h'(x, L*) = h(x, L*) − α · V_K[L*] V_K[L*]^T h(x, L*)`. The math is correct; Korean prose between blocks alternates between explanatory ("...stack한다") and mathematical-prescriptive ("retain한다", "Inference 시 선택된 layer L*에서"). Standard Korean academic register would be "...한다" throughout — current text is consistent on `한다` form, OK. But mid-paragraph particle `·`-stack ` U_L Σ_L V_L^T` should be in display block (already is — fine).

## Round-1 follow-up audit

The user asked specifically for verification that Round-1 fixes did not leave prose stale. Per-item audit:

| Round-1 fix | Prose stale? | Detail |
|---|---|---|
| Table 3 TallyQA gemma3-27b 0.138/0.117 → 0.074/0.053 | ✅ Round 1's prose impact preserved — §4.3 Insight 1 "큰 끌림 → 큰 gap" still holds (TallyQA gemma3-27b +2.1 pp gap unchanged) |
| Table 5 E4 Δem swap (LLaVA +0.77, ConvLLaVA +1.30) | ⚠️ Bold convention switched to per-column but §6.1 prose says only "Mid-stack cluster 3 모델 ... Phase 2 full validation" without naming which row wins which column. Reader has to deduce. Suggest one-sentence add. See [F]. |
| §7 bootstrap CI methodology paragraph | ❌ Reads as bolted-on. The "**95 % CI 산출.**" bold-period heading is not used elsewhere as a paragraph label; the paragraph interrupts §7's flow between result paragraph and Insight box. See [MAJOR-W7]. |
| §5.1 Insight 2 disambiguation (digit-bbox concentration peak vs answer-step attention peak depth) | ⚠️ Disambiguation correct, but the resulting Insight 2 paragraph is 4 sentences with two CSV citations and a §5.3 forward pointer — doing too much for one Insight box. See [D §5.1 Insight 2]. |
| §8.2 4 → 7 bullet expansion | ❌ Two registers mixed (research-scope limitation vs operational debt). See [MAJOR-W8]. |
| §1.5 contribution (4) rewrite | ❌ Bullet-train of phrases without grammatical glue. See [CRIT-W3]. |
| Abstract "5×7=85" → "6 dataset × 7 model heterogeneous coverage" | ✅ Three call-sites consistent (abstract / §1.2.3 / §4.5). |
| §6.5 Table 7 ActAdd "+57 %" → qualitative + LEACE +56 % source-pinned | ✅ Three call-sites consistent (Table 7 / §5.2 Insight 2 / §6.4 prose). |
| §6.5 Table 7 verdict column register | ❌ Mixed English/Korean across 5 baseline rows. See [F]. |

**Overall Round-1 prose impact:** mixed. Numerical fixes are clean; prose surrounding them often left without re-flow.

## Must-fix list

1. **§4.1 Table 2 / Insight 1 robustness ordering.** Either rebold Gemma4-31b-it as "가장 강건한 모델" (df 0.085 < Qwen2.5-VL 0.094) and rewrite Insight 1 to match, or qualify "강건" by metric ("adopt rate 기준 가장 강건" — Qwen2.5-VL 0.021 is the smallest by adopt). Right now the table caption, the Insight 1 prose, and the bolding cannot all be consistent. [CRIT-W1]
2. **§4.6 Insight 1 sentence 2 ungrammatical.** Replace *"증폭은 이분법이 가장 *덜* 취약하다고 한 *correct-base* row에 우선적으로 떨어진다."* with *"증폭은 이분법이 가장 *덜* 취약*하다고 분류한* *correct-base* row에 *집중된다*."* (or, cleaner: *"증폭의 효과는 wrong/correct 분할이 *덜 취약*하다고 분류한 correct-base 부분집합에 *집중된다*."*). [CRIT-W2]
3. **§1.5 contributions (4)/(5)/(6) rhythm.** Re-flow as bullet sentences matching the (1)/(2)/(3) register. Suggested:
   *"(4) 6-model 메커니즘 panel에서 single-layer ablation이 6/6 null임을 보이고 multi-layer redundancy를 분석한다. (5) Single-direction mitigation의 cross-dataset 실패를 *예측한 뒤 multi-direction subspace projection으로 우회*하며, 6-benchmark capability preservation을 검증한다. (6) γ-β (Qwen3-VL Instruct vs Thinking) N=1 pair에서 reasoning-amplifies-anchoring을 처음으로 보이는 *first-evidence* 결과."* [CRIT-W3]
4. **Drop "주력" from §6.2 title.** Change `### 6.2 E6 — Residual-stream subspace projection (deployable, 주력)` to `### 6.2 E6 — Residual-stream subspace projection (deployable)`. The "주력 / flagship" gloss is redundant with §1.3 / abstract / §6.6 already calling E6 "deployable mitigation으로 권장". [MAJOR-W5]
5. **Replace "발화" usage in §6.1 Insight and §6.3.** Rewrite §6.1 sentence to: *"hook은 single-image inference에 누출되지 않으며 두 번째 이미지에 가독 digit이 없으면 *hook은 트리거되지만* 제거할 signal이 없다."* §6.3 sentence to: *"Projection은 anchor 유무와 무관하게 모든 forward에서 작동한다 (universal projection — input의 anchor 포함 여부와 독립)."* [MAJOR-W6]
6. **Replace "이분법" globally.** 6 occurrences in §1.2.3 / §4.5 / §4.5 Insight 1 / §4.6 Insight 1 / §8.1 — replace with "wrong-base / correct-base 분할" (or, where space-tight, "(b)-base 정/오 분할"). The user explicitly disliked Korean translator-coinages of this kind; "이분법" is exactly that. [CRIT-W4]
7. **§7 bootstrap CI paragraph: re-integrate or move.** Either fold into the result paragraph as: *"... Pipeline integrity는 lmms-lab model card published 수치와 비교 검증 (MMStar 61.67 vs 61.7 본질적 일치). CI는 proportion-style 벤치마크에서는 McNemar paired SE, OCRBench는 paired percentile bootstrap (n=1,000 resample, seed 0; `scripts/aggregate_capability_eval.py`)으로 산출."* — single sentence, no bold heading. Or move to Appendix B as a numbered subsection. The current standalone bold paragraph reads as a memo. [MAJOR-W7]
8. **§8.2 limitation list register split.** Restructure as two sub-blocks:
    - *"한계 (research scope):"* 단일 prompt / Open-weight 모델만 / Human baseline 부재 / Mid-stack cluster 단일 / γ-β N=1 reasoning pair.
    - *"이번 라운드에서 deferred된 작업:"* §6.2.3 small-n cell CI / §6.2.2 27-cell pilot grid / OneVision E1d analyzer fix.
   Currently 7 bullets in one list mixing the two; reader cannot tell which limit is methodological vs operational. [MAJOR-W8]
9. **Add Figure 1 in-text reference in §3.1.** End the §3.1 paragraph with *"...자극 inventory는 ... `d`)이다 (부록 §A; Figure 1에 4-조건 자극 효과 예시)."* Currently no `(Figure 1)` mention exists in body prose. [MAJOR-W11]
10. **§4.4 Figure 5 caption vs Insight 2 alignment.** Either qualify caption — *"gemma3-4b가 chart/plot/math에서 가장 큰 끌림 (단, InfoVQA에서 4B < 27B로 역전; Insight 2)"* — or shorten Insight 2 to drop the InfoVQA reversal (less informative). The caption-Insight contradiction is a pure prose miss. [MAJOR-W10]
11. **Abstract closing claim soften: "입증" → "첫 VLM 재현".** Replace *"...VLM에 일반화됨을 입증."* with *"... 현상이 VLM에서도 *처음 재현*된다."* — matches §4.6 Insight 2 "첫 재현" and §8.2 lim 5 "first-evidence". The N=1 caveat in §8.2 directly contradicts abstract "입증". [G - claim/evidence + lim consistency]

## Should-fix list

1. **§3.3 raw-n disclaimer sentence.** Add main verb. *"...stratified 부분집합 *기준으로* ChartQA 129–517 ... MathVista 127–274 *범위에 분포한다*."* [E]
2. **§4.4 Insight 1 ("효과의 보편성") restating.** Either delete the Insight (the claim is already in Figure 5 caption verbatim) or rewrite to add a non-trivial inference (e.g., specific implication for §6 mitigation universality argument). [D]
3. **§4.5 Insight 3 ("Non-monotonic cell의 분류") filler.** Rewrite to give per-bucket counts (how many cells fall in each (a)/(b)/(c) bucket?) or move to §4.7 footnote. [D]
4. **§4.6 Insight 3 60-char parenthetical.** Move acc(d) computation detail to footnote. Body sentence should read: *"Thinking은 acc(d)도 더 *낮다* (Instruct 0.647 = 249/385, Thinking 0.587 = 226/385; γ-β 단일-stratum 설정에서 d-arm correct fraction을 b-arm baseline proxy로 사용)."* [MINOR-W16]
5. **§5.1 Insight 2 multi-purpose paragraph.** Split into Insight 2 (E1-patch concentration in 4-model perfect-square panel) and Insight 3 (OneVision Main calibration peak L20-L23 vs §5.3 attention peak distinction). [D]
6. **§6.6 closing recommendation duplication.** The recommendation appears in §1.3 / §6.2 / §6.6 closing. Drop §6.6 closing line (1 sentence). [A]
7. **§4.2 nominalisation stack fix.** Verbatim *"S1→S5에서 em이 안정한 것은 *거리는 em을 손상시키지 않음*을 보인다."* → *"S1→S5에서 em이 안정하다는 것은 *거리가 em을 손상시키지 않음*을 의미한다."* [E]
8. **§6.5 Table 7 verdict column register normalisation.** Pick one: all English ("direction mismatch / probe overfit / ...") or all Korean ("방향 불일치 / 동일 원인 / probe 과적합 / ..."). [F]
9. **§6.3 first paragraph rewrite.** *"Projection은 anchor 유무와 무관하게 모든 forward에서 작동한다 (universal — input이 anchor를 포함하는지 알지 못함). 첫 해석 — 'projection은 *오직* anchor 제거 연산일 *수 없다*' — 은 옳다."* [E]
10. **§4.7 italicised "generic tendency" English term.** Drop italics and switch to Korean "일반 경향" — current italicised English mid-Korean reads as direct translation residue. [MINOR-W17]
11. **§1.5 (6) "first-evidence claim" terminology.** Pick one of `VLM 최초 / first-evidence / 첫 VLM 결과 / 첫 재현` and use globally. [MINOR-W18]
12. **§3.3 mech panel "4 encoder archetype 분리" qualifier.** Either expand inline or remove until §5.1. [MINOR-W19]
13. **§4.6 Insight 2 sentence-fragment fix.** *"... 회귀의 위험원이라는 *운영적 함의*가 도출된다."* [E]
14. **§4.4 Insight 1 sentence-fragment fix.** *"이것이 본 논문이 *mitigation*을 추구할 수 있는 토대이다."* [E]
15. **§6.5 Insight closing fragment fix.** *"... §6.2.1 (a − m) contrast 설계의 직접 결과*이다*."* [E]
16. **§8.1 closing line fragment + particle stack fix.** *"reasoning trace*에서* bias가 *축적*된다는 것을 시사하는 첫 VLM 결과*이다*."* [E]
17. **§6.6 sentence-fragment fix in second/third sentences.** *"E4는 *mechanism* mitigation으로 attention pathway upper-half에서 *작동하고*, E6는 *representational* mitigation으로 residual stream에서 *작동한다*."* [E]
18. **Add abstract methodology pointer.** *"... mitigation의 cross-dataset *실패*를 예측하고 검증한다 (§5.2 → §6.4)."* — explicit cross-section pointer in abstract makes the Round-1 chain more legible. [G]
19. **Abstract / §1.2.2 "회귀" disambiguation.** Replace "*generic distractor 수준으로 회귀*" with "*generic distractor 수준으로 *되돌아간다**". Reduces overload with §1.1 / §1.2.1 statistical-regression sense. [B / H]
20. **Figure 7 caption "collapse" → "붕괴" or vice versa across all captions.** Pick one. [F]

## Final decision

**Decision:** **borderline** — writing-quality axis only.

The substantive argument flows correctly end-to-end (§1.2 pillars → §4 → §5 → §6 → §7) and the strongest Insight boxes (§4.3 PlotQA free-lunch, §6.2.1 (a−m) contrast, §6.4 predict-→-verify) carry the paper. But the Round-1 patch artefacts — bolted-on §7 bootstrap paragraph, §1.5 contributions bullet-train, §8.2 dual-register limitation list, §4.6 Insight 1 grammar, §4.1 Table 2 / Insight 1 mismatch — are at the level a careful camera-ready editor would catch, and there are at least 6 surviving forced Korean coinages ("이분법", "주력", "발화", "회귀" overload, "사지 않으면서", "빈자리") that the v3 changelog implicitly claimed to have removed. Round 2 needs a focused prose pass: 11 must-fix items (most are 1-sentence rewrites), 20 should-fix items (most are 1-clause polishes).

**One-line summary:** Round-1 fixed the numbers; Round-2 reveals that ~30 % of the prose around those fixes was not re-flowed, the Korean register has six surviving forced-translation outliers, and §1.5 / §8.2 lost their rhythm in the Round-1 patches.

**Sharpest single critique:** **§4.6 Insight 1 sentence 2 — *"증폭은 이분법이 가장 *덜* 취약하다고 한 *correct-base* row에 우선적으로 떨어진다"* — is grammatically broken Korean in the *only* paragraph that justifies the §1.5 (6) "메커니즘 결합 증명" claim**, in a section the abstract bills as the paper's "first VLM result on reasoning-amplified anchoring". A reader stops there.
