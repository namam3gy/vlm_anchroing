# A2 — Per-anchor-digit pull은 비대칭이고 모델별로 다르다

**Status:** Phase-A finding. 원본 데이터: `_data/A2_per_anchor_value.csv`. 스크립트: `research/scripts/phase_a_data_mining.py::a2_per_anchor_value`. *(영문 canonical: `A2-per-anchor-digit.md`)*

## 질문

Bias가 anchor 이미지에 *어떤* digit이 렌더링됐는지에 의존하는가? Flat per-digit profile = "anchoring은 content-agnostic — 텍스트 있는 추가 이미지 그 자체"; sharply varying profile = "특정 digit이 더 sticky함".

## Chance baseline (no confound)

VQAv2 number subset은 `samples_per_answer=400`, `answer_range=8`로 구성되어 GT 분포가 거의 flat (digit 0-8 각각 ~11.3 % 등장, digit 7만 ~9.8 %). `anchor == GT`의 chance probability는 모든 anchor digit에 대해 ~11 % (검증됨 — digit별 `anchor_eq_gt`가 모든 모델에서 [0.097, 0.124]). **아래 per-digit adoption rate는 anchor-GT collision으로 설명되지 않는다.** Digit-specific structure가 보이면 진짜 bias.

## 결과

**Anchor digit별 adoption rate** (모델당 anchor bucket 크기 ≈ 1,920–2,045 item). Chance baseline (anchor == GT)은 모든 cell에서 ≈ 0.11.

| anchor → | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma3-27b-it | .094 | .148 | .172 | .145 | .175 | .095 | .190 | .101 | .142 |
| gemma4-31b-it | .109 | .138 | .129 | .125 | .137 | .116 | .147 | .072 | .065 |
| gemma4-e4b | .145 | .213 | .150 | .125 | .132 | .088 | .094 | .102 | .058 |
| llava-interleave-7b | .099 | .142 | **.300** | .180 | .157 | .095 | .143 | .023 | .052 |
| qwen2.5-vl-7b | .106 | .151 | .159 | .125 | .161 | .088 | .085 | .058 | .049 |
| qwen3-vl-30b-it | .112 | .140 | .176 | .125 | .164 | .124 | .126 | .064 | .044 |
| qwen3-vl-8b-instruct | .135 | .144 | .163 | .138 | .140 | .133 | .136 | .092 | .065 |

**세 가지 패턴:**

1. **Digit 1, 2, 4가 universally high-anchorability**, 7, 8 (그리고 대부분 0)이 low-anchorability. 모델 간 일관 — 가장 저항적인 Qwen2.5-VL도 같은 hump. 그럴듯한 설명: VQAv2 number subset에서 "how many X"의 *plausible* answer space가 작은 수에 dominate 되므로 모델 prior가 1–4에 몰려 있고, 이 분포에 이미 들어있는 candidate와 일치하는 anchor가 더 잘 winning.

2. **`llava-interleave-7b` × anchor=2 = 0.300**이 두드러지는 outlier — chance 거의 3배, 같은 anchor에서 차순위 모델의 ~2배. LLaVA family의 typographic-attack 문헌이 보고하는 in-image text 특별 취약성과 일치, 개별 조사 가치 (attention-mass 분석 E1을 이 slice에 먼저 돌릴 것).

3. **Anchor=8이 여러 모델에서 본질적으로 inert** (gemma4-e4b 0.058, gemma4-31b 0.065, qwen3-vl-8b 0.065). 8은 GT range 상한이라, 어떤 "8" anchor든 "답이 plausible count의 edge에 있음"을 함의 — 모델이 그 이유로 down-weight 가능. Cognitive-science 문헌의 "extreme anchor" effect와 동등 (Mussweiler & Strack: 비현실적 anchor는 reject됨).

## `mean_anchor_pull`이 방향 특이적 행동 드러냄

Adoption은 symmetric (prediction = anchor였나?). `mean_anchor_pull`은 signed — 양수 = 가까워짐, 음수 = 멀어짐. 아래 두 sign-flip 행이 주목:

- `qwen3-vl-30b-it` × anchor=3: mean_pull = **-4.24**, signed_shift = +4.21 (즉 anchor가 3일 때 prediction이 *위로* ~4 drift — prediction이 이미 > 3였으면 "anchor 쪽"의 반대).
- `qwen3-vl-30b-it` × anchor=7: mean_pull = -1.55, signed_shift = +1.32.
- `qwen3-vl-8b-instruct` × anchor=6: mean_pull = -2.89, signed_shift = +2.82.

모두 Qwen3-VL이 특정 digit에 *negative* anchoring (반발, 끌림 아님)을 보이는 case. Cognitive-science 용어로 "anti-anchoring"이라 부르며 인간 실험 일부에서 관찰. 두 Qwen3-VL 크기에서 (Qwen2.5-VL이나 Gemma 어떤 것에서도 안) 깔끔하게 reproduce되므로 데이터 노이즈가 아닌 Qwen3-RL-style 학습 artifact 시사. Bootstrap CI에서 살아남으면 paper에 한 단락 가치.

## Paper에 대한 함의

- **Universal한 "anchor = puller" effect를 주장하지 말라.** 일부 (모델, digit) cell이 repel. Paper framing은 "모델이 anchor된다"보다 *susceptibility profile*에 대해 talk.
- **`llava-interleave × anchor=2`가 가장 명확한 case study** — 단일 (모델, digit) 조합, chance 3배, 매우 큰 n. Paper의 정성적 case로 사용.
- **Mitigation 평가에 digit-aware 분석 필요.** E4가 "moved-closer rate 10% 감소" 보고하면, anti-anchoring case에서 sign을 *flip*시키지 않는지 (= 실제로 distortion 증가) 보장 필요.

## Caveats

- `anchor == GT`의 chance baseline이 flat (~11 %)이라 adoption rate를 chance-above-bias로 직접 해석 가능. 위 `_data` 계산에서 확인.
- Per-digit n은 ~1,920–2,045. 이 n에서 binomial proportion의 standard-error threshold를 통과하는 adoption rate 차이는 > 2 pp; < 2 pp 차이는 over-interpret 안 할 것.
- "Anti-anchoring" cell은 paper claim 전에 bootstrap CI 필요. Phase A 빠른 add-on.

## Roadmap entry

§5 A2: ☐ → ✅
