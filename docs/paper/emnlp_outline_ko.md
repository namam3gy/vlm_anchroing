# Pulled by Pixels: Cross-Modal Numerical Anchoring in VLMs

**Authors.** {{Author list, affiliations}}

---

## Abstract

> 목표 분량: ~180 단어. 아래 5개 beat을 한 문단으로 압축.

- **Problem.** VLM에게 질문과 무관한 이미지를 함께 보여줄 때, 그 이미지에 숫자가 그려져 있다면 답이 그 숫자 쪽으로 끌리는가 — *cross-modal numerical anchoring*.
- **Finding 1 (현상).** 6개 open-weight VLM × 5 dataset에서 약 10–40 % 의 응답이 anchor 숫자 쪽으로 *점진적으로* 끌리고, 그 중 일부 (1.7–15.7 %) 는 anchor 를 그대로 베끼기까지 한다. 끌리는 정도는 base 답에 대한 모델 confidence 가 낮을수록 더 크다.
- **Finding 2 (causal gate).** 같은 anchor 이미지에서 숫자 픽셀만 가린 *masked* 짝을 만들어 비교하면 위 끌림이 일반 distractor 수준으로 사라진다 — 즉 anchoring 은 *(보이는 digit) × (모델 uncertainty)* 두 조건의 conjunction.
- **Method.** 본 논문은 anchored 와 masked 짝의 *paired contrast* 를 신호로 삼아 모델 내부의 anchoring representation 을 추정하고, inference 시 이를 제거하는 mitigation 을 제안한다.
- **Result.** 5 dataset 위에서 anchoring 효과 감소 + anchor 가 있는 경우와 없는 경우 모두 정확도 동시 상승, 6 held-out capability benchmark 평균 보존 (+0.41 pp).
- **Implication.** Mechanism 분석은 anchoring signal 이 단일 layer 가 아니라 모델 후반부 여러 layer 에 분산되어 존재함을 보여, vision-modality bias 가 LM 후반부에 *분리 및 제거 가능한 representation 단위* 로 존재해 retraining 이나 prompt-level 방어 없이 inference 시 직접 개입할 수 있음을 시사한다.

---

## 1 Introduction

### 1.1 동기 (Motivation)

- **Hook (model-behavior question).** Multi-image prompt 가 RAG / multi-screenshot Q&A / VLM-as-judge 등에서 일반화되는 가운데 — VLM 은 질문과 무관한 시각 입력을 *무시* 하는가, 아니면 그 정보가 답을 *흔드는가*?
- **Phenomenon 명명.** *Cross-modal numerical anchoring* — 무관한 이미지의 digit pixel이 모델 답을 끌어당긴다.
- **Cognitive grounding.** Anchoring 은 인간 [Tversky and Kahneman, 1974] 과 텍스트 LLM [Jones and Steinhardt, 2022; Echterhoff et al., 2024] 에서 잘 정립된 인지 편향. 그러나 *시각 modality* 위에서 — 즉 visual cue 가 모델의 답을 어떻게 끌어당기는지에 대해서는 체계적 평가가 부재. 위 deployment 시나리오 (RAG retrieval, multi-image Q&A, VLM-as-judge) 에서 무관한 시각 입력 노출이 자연스럽게 발생할 수 있어, 이 빈 칸은 cognitive 호기심을 넘어선 deployment relevance 를 가진다.

### 1.2 본 논문의 발견 (Findings)

> Abstract 의 Finding 1 / Finding 2 를 본문 톤으로 한 단계 풀어서 서술. 4-condition 자극 (b: target only / a: + anchor / m: + masked anchor / d: + neutral distractor) 으로 6 open-weight VLM × 5 dataset 측정.
- **F1 (graded pull).** 약 10–40 % 의 응답이 anchor 쪽으로 점진적으로 끌리고, 그 중 일부 (1.7–15.7 %) 는 anchor 를 그대로 베낀다 — 효과의 질량은 *literal copy* 가 아닌 *점진적 이동* 에 있다.
- **F2 (uncertainty modulation).** 끌리는 정도는 base 답에 대한 모델 confidence 가 낮을수록 크다 (5 dataset × 6 model L1 6-bin 위 단조 monotonic gradient).
- **F3 (digit-pixel causal gate).** Anchor 이미지에서 digit pixel 만 가린 *masked* 짝과 비교하면 끌림이 일반 distractor 수준으로 사라진다 — *digit pixel × uncertainty* 두 조건의 conjunction.
- **F4 (mechanism).** Anchoring 의 internal representation 은 LM 후반부 layer 에 형성된다 — 이 위치는 본 논문 mitigation 의 작용 site 가 된다.

### 1.3 접근과 결과 (Approach & Result)

- *(a − m) paired contrast* 를 calibration 신호로 활용 → 모델 내부의 anchoring representation 을 추정 → inference 시 projection 으로 제거.
- Calibration 은 별도 phase 에서 한 차례만 수행되고, 거기서 얻은 projection 은 이후 모든 inference 에 일괄 적용된다. 배포 단계부터는 입력 이미지에 anchor 가 포함되어 있는지 여부와 무관하게 동일한 projection 이 작동하므로, runtime 에 anchor 를 탐지하거나 라벨링할 필요가 없다 — 그대로 deployable 하다.
- 이 mitigation 을 적용하면 5 dataset 위에서 anchoring 효과가 감소하며, 동시에 anchoring 과 *무관한* 6 개 general capability benchmark (held-out) 의 성능도 손상되지 않는다 (평균 변화 +0.41 pp) — 즉 mitigation 이 모델의 일반 능력을 해치지 않는다.

### 1.4 기여 (Contributions)

- **C1 (Phenomenon).** Cross-modal numerical anchoring 을 5 dataset × 6 open-weight VLM 위에서 정량 보고 — 약 10–40 % 의 응답이 무관한 이미지의 digit 으로 끌리며 (일부는 그대로 베끼고), 끌림은 모델 confidence 에 비례해 graded 되며, anchor 이미지에서 digit pixel 만 가리면 효과가 사라짐. 행동 + causal gate 동시 검증.
- **C2 (Mitigation).** *(a − m) paired contrast* 를 calibration 신호로 활용해 LM 후반부의 anchoring representation 을 한 차례 추정하고, inference 시 모든 입력에 projection 으로 일괄 적용. anchor label 이나 runtime 탐지 불필요. 5 dataset 에서 anchoring 효과 감소 + 6 일반 capability benchmark 평균 +0.41 pp 보존.
- **C3 (Mechanism evidence).** Anchoring 의 internal representation 이 LM 후반부 layer 에 분산되어 형성됨을 mechanism 분석으로 입증 — 이 결과는 C2 mitigation 의 작용 site 와 *multi-direction subspace* 설계 선택을 모두 정당화한다.

---

## 2 Related Work

> 4 subsection, 각 끝 1–2 문장에 본 논문 differentiator woven-in. 별도 positioning subsection 없음.

### 2.1 Anchoring in cognition and text LLMs

{{Tversky-Kahneman 1974, Mussweiler-Strack 1999 (cognitive); Jones-Steinhardt 2022, Echterhoff 2024, Lou-Sun 2024 (LLM anchoring + prompt-level mitigation 실패), Wang 2025a (LRM judging bias 가 reasoning trace 통해 amplified — §4.5 Qwen3-VL Thinking ×12.7 호응), Huang 2025 (synthetic LLM mechanism — text-side mechanism 비교). 끝 1문장: 시각 modality 위 anchoring 미평가.}}

### 2.2 Cognitive bias and behavioral analysis of VLMs

{{VLMBias [Vo, Nguyen 2025] — familiar-subject counting; AIpsych [Liu 2025], CIVET [Rizzoli 2025], Tinted Frames [Fan 2026] — sycophancy / position / framing. 끝 1문장: 본 논문은 cue 가 question subject 와 분리된 *independent rendered-digit* 이미지 + open-ended numeric estimation 으로 두 축 모두에서 상보적.}}

### 2.3 Visual cue manipulation in VLMs

{{Goh 2021 multimodal neurons (mechanism foundation); Wang 2025b NAACL multi-image typographic attack; Gong 2025 FigStep visual jailbreak; Hufe 2025 Dyslexify (encoder-side defense for typographic). 끝 1문장: 본 논문은 클래스 라벨 / prompt 의 이미지화가 아닌 *수치값 단독* cue, 분류 뒤집기 / ASR 이 아닌 *open-numeric baseline-relative shift*; mitigation site 도 encoder 가 아닌 LM residual.}}

### 2.4 Representation-level intervention: activation steering and concept erasure

{{CAA [Panickssery 2024] — paired contrastive activation, single direction; ITI [Li 2023] — attention-head multi-direction (LM-only); LEACE [Belrose 2023] — closed-form linear erasure (rank-1 default); Weng 2024 EMNLP — VLM gender bias의 causal mediation → encoder-side mitigation (mechanism→mitigation chain venue-tier 선례); Chand 2025 "No Free Lunch in LM bias mitigation" — LM × discrete social bias × weight space 위 4-clause 동시 충족 *실패* 보고 (본 mitigation 은 *VLM × continuous numeric × inference activation* cross-axis positive). 끝 1문장: 본 mitigation 은 (i) CAA paired-contrast 패러다임을 *vision-modality (a − m) 인과 통로 분리* 로 확장, (ii) ITI attention-head 가 아닌 *residual stream*, (iii) single-direction (LEACE rank-1 / ActAdd) 의 cross-dataset 실패 위에서 *multi-direction subspace* 채택.}}

---

## 3 Cross-Modal Anchoring: Stimulus and Measurement

> Paradigm 도입. (a − m) paired contrast 는 §3.2 에 별도 격상 — §4 phenomenon, §5 mechanism, §6 mitigation 세 cluster 모두 이 substrate 를 재사용.

### 3.1 4-condition stimulus (b / a / m / d)

{{Target only / target + anchor / target + masked anchor / target + neutral distractor 의 정의 + 설계 의도. 자극 예시 + 자세한 spec 은 Appendix A.}}

### 3.2 (a − m) paired contrast — digit-pixel 인과 isolate

{{Anchor 이미지 안의 *다른 모든 변수* (이미지 존재, 추가 attention 부하, 배경 텍스처) 를 m 이 그대로 carry 하므로, (a − m) 차이는 *digit pixel* 만 isolate 한다는 design 논리. §4 의 digit-pixel causal-gate finding + §6 의 calibration 신호 양쪽의 substrate.}}

### 3.3 Anchoring 측정 (metrics)

{{Direction-follow (primary, gt-free) / adopt (literal copy) / exact-match. 정확한 정의 + 본문에서 어떤 metric 이 어떤 claim 을 carry 하는지.}}

### 3.4 Models and datasets (brief)

{{6 open-weight VLM × 5 dataset main panel 의 한 줄 summary. 자세한 사양 + filter / sampling 은 Appendix B.}}

---

## 4 Phenomenon

> *언제* 그리고 *무슨 조건* 에서 anchoring 이 일어나는가.

### 4.1 {{Graded pull across models and datasets}}

{{F1 — 약 10–40 % 응답이 anchor 쪽으로 점진적 끌림, 일부 (1.7–15.7 %) 는 그대로 베낌. 6 model × 5 dataset cross-table.}}

### 4.2 {{Confidence modulation}}

{{F2 — base prediction confidence 가 낮을수록 끌림이 큼. L1 6-bin monotonic gradient (5 dataset × 6 model 위 단조).}}

### 4.3 {{Digit-pixel causal gate via (a − m)}}

{{F3 — anchor 이미지의 digit pixel 만 가린 m 으로 비교 시 끌림이 일반 distractor 수준으로 사라짐. (a − m) gap 의 dataset/model 별 크기.}}

---

## 5 Mechanism Analysis

> *어디* 에서 anchoring representation 이 형성되는가.

### 5.1 {{Layer-wise localization}}

{{Layer × model heatmap, peak layer identification on calibration set.}}

### 5.2 {{Single-layer ablation + multi-layer redundancy}}

{{Single-layer null result (5/5), multi-layer redundancy 시사. §6 mitigation site / multi-direction subspace 설계의 motivation.}}

### 5.3 {{Late-layer formation site}}

{{Anchoring representation 이 LM 후반부 layer 에 형성된다는 통합 결론. §1.2 F4 의 evidence.}}

---

## 6 Mitigation

> Method + cross-dataset results + capability preservation 한 section 에.

### 6.1 {{Method: subspace projection from (a − m) calibration}}

{{(a − m) 차이 vector 들의 SVD top-K subspace 를 LM 후반부 residual 에 projection 으로 제거. Algorithm box.}}

### 6.2 {{Calibration setup + hyperparameters}}

{{Calibration set (보조 dataset 1–2개), projection rank K, layer 선택 근거.}}

### 6.3 {{Cross-dataset anchoring 감소}}

{{5 dataset 위 anchoring 효과 변화 (Δdf, Δadopt). Paired bootstrap CI.}}

### 6.4 {{Capability preservation (held-out benchmarks)}}

{{6 일반 capability benchmark (HallusionBench, AMBER, POPE, MME 등) 에서 평균 성능 변화 +0.41 pp.}}

---

## 7 Discussion

### 7.1 함의 (Implications)

{{}}

### 7.2 다른 접근과의 비교

{{}}

### 7.3 후속 작업 (Future work)

{{}}

### 7.4 Ecological validity — closed-API VLM-as-judge pilot

- §1.1 hook (deployment relevance) 의 보충 검증. 5 closed-API judge × 2 dataset × 3 arm × n=200 pilot.
- 결과 mixed: gpt-4o (digit-specific) / gemini-2.5-flash (distractor-general) 에서 anchoring 관찰; gpt-5.1 / gemini-2.5-pro / claude 는 robust.
- Take-home: phenomenon 이 open-weight 인공물이 아니며 frontier judge 에서도 부분적으로 surface — mitigation 의 deployment relevance 보강. 동시에 *모든* 시스템이 영향을 받는다고 overclaim 하지 않음.

---

## 8 Conclusion

{{한 문단 요약: 무엇을 했고, 무엇을 보였고, 왜 중요한가}}

---

## Limitations

> EMNLP 필수 섹션. 페이지 제한 *밖*. Honest, specific, non-defensive.
> 데이터/모델/일반화/통계/재현성 한계를 항목별로.

- {{Limitation 1}}
- {{Limitation 2}}
- {{Limitation 3}}

---

## Ethics Statement

> 권장 섹션. 페이지 제한 밖. 해당 사항 없으면 명시적으로 "no ethical concerns identified" 라고 적기보다, 데이터 출처·라이선스·잠재적 오용·환경 비용 정도는 짧게 다루는 것이 통상.

{{Data licensing, human subjects, potential misuse, compute / carbon footprint}}

---

## Acknowledgments

> Optional. 익명화 단계에서는 비워둘 것.

{{}}

---

## References

> BibTeX는 별도 `.bib` 파일에서 관리. 여기는 placeholder.

{{References}}

---

# Appendix

> 페이지 제한 밖. 본문에서 참조한 보충 자료.

## A Stimulus & prompt details

### A.1 Prompt template

모든 6개 모델, 모든 dataset, 모든 4-condition 에 대해 *동일한* system + user template 사용 (모델별 변동 없음). 이미지 슬롯 개수만 condition 에 따라 1 (b) / 2 (a, m, d) 로 변동.

**System:**
```
You are a visual question answering system.
Return valid JSON only in the form {"result": <number>}.
Use a numeric JSON value for <number>, not a string.
Do not output any other keys, words, explanation, or markdown.
If uncertain, still output the single most likely number in that JSON format.
```

**User template:**
```
Answer the question using the provided image(s).
Return JSON only in the form {"result": <number>}.
Question: {question}
```

Sampling: `temperature=0.0`, `top_p=1.0`, `max_new_tokens=16`.

### A.2 4-condition 자극 예시

Figure A.1 — 동일 question 위 4-condition 자극의 실제 입력 이미지 + 모델 답. 본 예시는 demo site (https://namam3gy.github.io/vlm_anchroing/) 의 첫 번째 샘플 (TallyQA, `2448100_24481_stratified`).

**Question:** *How many zebras are there standing in the water?*
**Ground truth:** 3
**Anchor value:** 4

각 condition 의 input 은 *target image + 두 번째 image (a/m/d 의 경우)* — 모델은 두 이미지를 하나의 prompt 에 동시에 받음.

| Condition | Input image(s) | Llava-OneVision-7B (Main) 답 |
|---|---|---|
| **b** (target only) | <img src="figures/demo_sample_01/target.png" alt="target" width="180"/> | **3** ✓ |
| **a** (target + anchor) | <img src="figures/demo_sample_01/target.png" alt="target" width="180"/> <img src="figures/demo_sample_01/anchor.png" alt="anchor" width="180"/> | **4** ← anchor 그대로 |
| **m** (target + masked anchor) | <img src="figures/demo_sample_01/target.png" alt="target" width="180"/> <img src="figures/demo_sample_01/masked.png" alt="masked" width="180"/> | **3** ✓ (회복) |
| **d** (target + neutral distractor) | <img src="figures/demo_sample_01/target.png" alt="target" width="180"/> <img src="figures/demo_sample_01/neutral.png" alt="neutral" width="180"/> | **3** ✓ |

> 이 한 sample 이 *(a − m) paired contrast* 의 직관을 그대로 보여준다 — anchor 이미지가 들어가면 답이 anchor (4) 로 끌리고, *동일한 anchor 이미지에서 digit 만 가린* m 으로 바꾸면 답이 base (3) 로 회복. Neutral distractor (d) 는 영향 없음. 6-model 전체의 같은 sample 위 답은 demo site 참조.

### A.3 Masked 자극 생성

Anchor 이미지의 digit bounding box 를 OpenCV `INPAINT_TELEA` [Telea, 2004] 로 채워 `m` 이미지 생성. Mask 외 영역은 원본 그대로 보존되어, *paired contrast (a − m)* 가 digit pixel 효과만 isolate.

---

## B Dataset details

| Dataset | Split | Source | Filter | Eligible samples (n) | GT range | GT distribution |
|---|---|---|---|---|---|---|
| ChartQA | test | full split (2,500) | numeric GT, range [0, 1000] | **705** | 0–1000 | small / round 값 skew |
| PlotQA | test | seed=42 stratified subset of 1,228,313 (1,000/bin × 5 GT bins (0,8] (8,20] (20,100] (100,1k] (1k,10k]) | range [0, 10000] | **5,000** | 0–10000 | bins 균등 (sampled by design) |
| InfoVQA | val | fetch-time numeric-only subset of 1,147 | range [0, 10000] | **1,147** | 0–10000 | natural, mild right-skew |
| MathVista | testmini | full split (1,000) | `answer_type=integer`, range [0, 1000] | **385** | 0–1000 | mixed (도형 counting + 산술) |
| TallyQA | test | full split (38,589) | numeric GT, range [0, 8] | **38,245** | 0–8 (counting) | strongly skewed small (1–4 dominant) |

- *Eligible samples (n)* = filter 통과한 sample 수, 모든 모델 공통. 이 pool 위에서 4-condition (b/a/m/d) 자극이 구성된다.
- 모든 dataset 에 공통으로 `require_single_numeric_gt=true` — GT 후보가 모두 동일 numeric value 인 sample 만 채택. `answer_range` cutoff 는 anchor inventory (Appendix C) 와 정합되도록 설정.

---

## C Anchor value 선정 (per dataset)

**Inventory.** 128 개의 pre-rendered single-/multi-digit 이미지 (`inputs/irrelevant_number/{value}.png`). Value 분포는 dense-low / sparse-high:

| 구간 | 간격 | Values | 개수 |
|---|---|---|---|
| 0 – 10 | 1 | 0, 1, 2, …, 10 | 11 |
| 15 – 100 | 5 | 15, 20, 25, …, 95, 100 | 18 |
| 200 – 10,000 | 100 | 200, 300, 400, …, 9900, 10000 | 99 |
| **합계** | | | **128** |

이 분포는 GT 가 작은 dataset (TallyQA, MathVista counting) 에서 close-distance anchor 가 풍부하고, GT 가 큰 dataset (PlotQA, InfoVQA) 에서도 100 단위로 plausible anchor 를 제공하도록 설계.

**Per-question stratification.** 각 question 의 GT 와 inventory 의 모든 candidate 사이 거리 |a − gt| 를 계산. 거리 stratum 별로 매칭되는 candidate 중 RNG 로 하나를 추출 — 한 question 당 stratum 수만큼의 anchor 가 sampling 됨.

**거리 stratum scheme 두 종류:**
- **absolute**: 고정 경계 [0,1], [2,5], [6,30], [31,300], [301,∞]. GT 범위가 좁은 dataset (TallyQA) 용.
- **relative**: 경계 hi 가 `max(absolute_floor, fraction × |gt|)` 로 GT 크기에 따라 스케일. fractions [10 %, 30 %, 100 %, 300 %, ∞]. Wide-GT dataset 용.
- **relative_s1**: relative scheme 의 *첫 stratum 만* 사용 (단일 close-distance: |a − gt| ≤ max(1, 0.10·gt)).

**Main 6-model panel 의 per-dataset 설정:**

| Dataset | Scheme | Stratum 수 | Effective range |
|---|---|---|---|
| TallyQA | absolute | [[0, 5]] (single) | \|a − gt\| ≤ 5 (GT 가 0–8 이므로 거의 모든 plausible anchor) |
| ChartQA | relative_s1 | 1 | \|a − gt\| ≤ max(1, 0.10·gt) |
| MathVista | relative_s1 | 1 | 동일 |
| PlotQA | relative_s1 | 1 | 동일 |
| InfoVQA | relative_s1 | 1 | 동일 |

→ **Main panel 은 close-distance single-stratum 만 사용** — anchor 가 GT 와 plausible 한 이웃 거리에 있는 cohort 위에서 효과를 측정 (anchoring 의 *worst-case* slot). 5-stratum full schedule 은 distance-decay 분석을 위한 OneVision-only extension (§{{section}}) 으로 별도 보고.
