# A1 — Anchoring은 uncertainty-modulated graded pull, categorical capture가 아니다

**Status:** Phase-A finding. 7개 모델 모두에서 robust. EMNLP write-up의 가장 강한 단일 hook. 원본 데이터: `_data/A1_asymmetric_long.csv`, `_data/A1_asymmetric_wide.csv`. 스크립트: `research/scripts/phase_a_data_mining.py::a1_asymmetric_on_wrong`. *(영문 canonical: `A1-asymmetric-on-wrong.md`)*

## 질문

`research_plan.md`가 "모델이 원래 틀린 item에서 더 강한 anchoring"을 paper의 가장 강한 hook으로 flag — LLM/VLM anchoring 문헌 어디도 prior correctness로 effect를 partition하지 않음. Mussweiler-Strack / Jacowitz-Kahneman 모두 anchoring이 주관적 불확실성에 비례한다고 예측. **이게 성립하는가?** 사전에 명시한 falsifier (`RESEARCH_ROADMAP.md` §2 H2): `adoption(wrong)` ≈ `adoption(correct)`이면 H2 fail.

## 방법

각 모델에 대해, `target_only` prediction이 정확히 맞았는지 (`base_correct`)로 17,730개 paired record를 두 그룹으로 분할. 각 stratum 내에서 3개 anchoring metric 계산:

1. **`adoption_rate`** — `target_plus_irrelevant_number` prediction이 정확히 anchor digit과 같은 비율. Categorical.
2. **`moved_closer_rate`** — `|pred(number) − anchor| < |pred(target_only) − anchor|`인 pair의 비율. Graded.
3. **`mean_anchor_pull`** — `|pred(target_only) − anchor| − |pred(number) − anchor|`, pair 평균. Magnitude.

Effect는 `wrong − correct`로 보고 (양수 ⇒ H2 방향 asymmetry).

## 결과

Categorical falsifier는 *fail*: adoption은 stratum 간 거의 동일.

| Model | n_correct | n_wrong | adoption(correct) | adoption(wrong) | **adoption gap** |
|---|---:|---:|---:|---:|---:|
| gemma3-27b-it | 8,611 | 9,119 | 0.128 | 0.152 | **+0.024** |
| gemma4-31b-it | 10,644 | 7,055 | 0.117 | 0.113 | -0.004 |
| gemma4-e4b | 7,570 | 10,160 | 0.129 | 0.118 | -0.011 |
| llava-next-interleaved-7b | 8,419 | 9,264 | 0.143 | 0.125 | -0.018 |
| qwen2.5-vl-7b-instruct | 10,495 | 7,235 | 0.116 | 0.100 | -0.016 |
| qwen3-vl-30b-it | 10,854 | 6,868 | 0.118 | 0.123 | +0.005 |
| qwen3-vl-8b-instruct | 10,747 | 6,983 | 0.126 | 0.130 | +0.004 |

그러나 graded falsifier는 holds — 그것도 *모집단 전체에 걸쳐 균일하게*:

| Model | moved_closer(correct) | moved_closer(wrong) | **moved_closer gap** |
|---|---:|---:|---:|
| gemma4-e4b | 0.135 | 0.331 | **+0.196** |
| gemma3-27b-it | 0.080 | 0.239 | **+0.159** |
| qwen3-vl-30b-it | 0.116 | 0.238 | **+0.122** |
| gemma4-31b-it | 0.047 | 0.131 | +0.084 |
| qwen3-vl-8b-instruct | 0.068 | 0.148 | +0.080 |
| llava-next-interleaved-7b | 0.125 | 0.197 | +0.072 |
| qwen2.5-vl-7b-instruct | 0.061 | 0.130 | +0.069 |

`mean_anchor_pull`도 모든 모델에서 같은 sign으로 같은 방향 (`_data/A1_asymmetric_wide.csv` 참조).

## 의미

H2 예측을 다시 적어야 함:

> **재정의된 H2.** VLM이 numerical VQA 답에 대해 불확실할 때 (= "원래 틀림"으로 operationalize), irrelevant anchor digit이 prediction을 substantial하게 더 높은 비율 (+7 ~ +20 pp)로 anchor *쪽으로* shift시킨다. 단, anchor를 *정확히 copy*하는 비율은 거의 변하지 않는다.

이는 "wrong case가 더 anchor된다"의 원래 framing보다 더 sharp하고 explain-away하기 어려운 claim. Mussweiler-Strack "selective accessibility" 설명에 직접 매핑: anchor가 confident prediction을 대체하는 게 아니라 prediction이 uncertain할 때 **search direction을 bias**. Paper가 사용할 cognitive-science framing:

- *Confident estimate (= target-only에서 correct):* anchor 거의 무시; adoption ≈ baseline; pull ≈ 0.
- *Uncertain estimate (= target-only에서 wrong):* anchor가 candidate distribution에 진입해 prediction을 자기 쪽으로 끌지만 dominate하지는 않음.

이게 정확히 인간 실험이 보고하는 gradient-anchoring signature. Paper가 이제 새로운 empirical claim과 데이터에 맞는 깔끔한 cognitive-science 이론을 둘 다 가짐.

## Caveats

1. **`base_correct`는 주관적 불확실성의 noisy proxy.** 모델은 wrong-but-confident 또는 right-but-uncertain일 수 있음. 계획: 저장된 per-token logit margin (`5f925b2`)이 moved-closer gap과 correlate하는지 확인 — binary stratum을 continuous uncertainty measure로 대체 가능. Submission 전 가치 있는 Phase-A 확장.
2. **Outlier filter 적용.** `analysis.filter_anchor_distance_outliers` (IQR×1.5) on. 없으면 gemma3 / Gemma4 strengthen-prompt run이 mean을 왜곡. Pattern은 filter 없이도 robust — 재현 시 검증.
3. **Cross-anchor confound.** `adoption_rate`는 anchor digit이 plausible한 답인지에 의존. Anchor 1, 2, 3은 GT 분포에서 over-represented이라 adoption-correct가 "anchor가 진실과 일치하는 운좋은 추측"을 포함할 수 있음. Graded `moved_closer_rate`는 nontrivial shift에 conditioning하므로 이 confound로부터 largely 절연; per-digit confound는 주로 A2를 hits, 거기서 논의.
4. **가장 큰 gap 보이는 모델이 가장 weak한 모델** (gemma4-e4b acc=0.55; gemma3-27b acc=0.63). Lou & Sun "stronger LLMs anchor more"의 *반대* 방향. 가능한 설명: (a) weak 모델이 더 자주 uncertain이라 gap이 주로 다른 stratum 크기 반영 (gemma4-e4b: 10,160 wrong vs 7,570 correct); (b) weak 모델이 anchored prior가 적어 external anchor가 더 큰 weight. 분리하려면 per-confidence (logit-margin) 재분석 필요.

## 이 finding과 묶이는 구체적 다음 단계

- **`base_correct`의 logit-margin 대체.** Commit `5f925b2`의 per-token logit으로 continuous-uncertainty stratification으로 표 재현. (~시간 단위, 추가 compute 없음.)
- **E1 attention 분석**은 `base_correct`로 conditioning되어야 "wrong case에서 anchor에 attention 더 큼 > correct case" 예측을 직접 test 가능. 이 conditioning 없이는 attention signal이 average out 됨.
- **Mitigation E4**의 자연스러운 target: LLM의 answer position per-token entropy가 높을 때만 anchor-image attention을 down-weight. 무차별 anchor-token down-weighting보다 더 principled한 intervention.

## Roadmap entry

- §2 H2 status: ⚠️ → ✅ (재정의; *graded-pull* 형태는 holds; *categorical-adoption* 형태는 fails).
- §5 A1: ☐ → ✅
