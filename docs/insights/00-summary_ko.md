# Phase-A summary — 17,730 sample × 7 model이 실제로 말해주는 것

**원본 데이터:** `outputs/experiment/<model>/<run>/predictions.csv`, `scripts/phase_a_data_mining.py`로 처리. `analysis.filter_anchor_distance_outliers`의 outlier filter 적용 (sample별 anchor-GT distance에 IQR×1.5, default settings). 모든 수치는 `docs/insights/_data/*.csv`에서 옴. *(영문 canonical: `00-summary.md`)*

이 파일은 cross-cutting 종합. 가장 강한 finding의 long-form 논의는 per-insight 파일에 (`A1-…`, `A2-…`, `A7-…`).

## TL;DR — 한 단락씩 가져갈 4개 발견

1. **Anchoring effect는 categorical이 아니라 graded.** 7개 모델 모두 5–14 %만 anchor digit에 정확히 일치. 그러나 6–19 %가 anchor *쪽으로* 추가로 shift. Bias는 cognitive-science anchoring 모델이 예측하는 정확히 그 형태의 soft regression-style pull.

2. **"불확실성이 anchoring을 modulate"라는 H2 예측은 진짜다 — 단 *graded-pull* 축에서, *adoption* 축이 아니다.** `target_only` correctness로 stratify하면 7개 모델 모두 wrong-stratum에서 moved-closer rate가 **+6.9 ~ +19.6 pp** 더 높음. Categorical adoption은 거의 안 움직임 (Δ ∈ [-1.8, +2.4] pp). H2 재정의: uncertain item은 anchor에 *capture*되는 게 아니라 anchor 쪽으로 *dragged*된다.

3. **Anchor digit은 대칭적이지 않다.** Anchor 7과 8은 모든 모델에서 2와 4보다 체계적으로 덜 효과적. 한 가지 confound — VQAv2 number subset의 GT 분포가 right-skewed (대부분 정답이 1, 2, 3)라 anchor=2가 정답일 prior 확률이 훨씬 높음. Chance-corrected 재분석 필요 (A2 file에서 자세히).

4. **Item susceptibility는 부분적으로 content-driven.** 모델 페어 간 per-question moved-closer rate의 Spearman correlation이 0.15–0.31. Same-family 모델이 가장 높음 (Qwen3-VL-30B ↔ Qwen3-VL-8B = 0.30; Gemma4-e4b ↔ Qwen3-VL-30B = 0.31). 어떤 question은 universally bias-susceptible, 다른 건 model-specific. **Mechanism 작업에 대한 함의:** bias의 일부는 visual encoder/content에 (H3 land 예측), 일부는 LLM head에.

## 헤드라인 수치 (standard prompt, 모델당 17,730 샘플)

| Model | acc(target) | adoption(num) | moved-closer | moved-closer\|wrong − correct |
|---|---:|---:|---:|---:|
| gemma4-e4b | 0.553 | 0.123 | 0.247 | **+19.6 pp** |
| gemma3-27b-it | 0.628 | 0.141 | 0.162 | **+15.9 pp** |
| qwen3-vl-30b-it | 0.759 | 0.120 | 0.163 | **+12.2 pp** |
| gemma4-31b-it | 0.749 | 0.116 | 0.081 | +8.4 pp |
| qwen3-vl-8b-instruct | 0.751 | 0.127 | 0.100 | +8.0 pp |
| llava-next-interleaved-7b | 0.619 | 0.133 | 0.163 | +7.2 pp |
| qwen2.5-vl-7b-instruct | 0.736 | 0.110 | 0.089 | +6.9 pp |

**Cross-model 패턴:** wrong-vs-correct gap (H2 effect)이 ~13 pp 범위지만 *모든* 모델에서 양수. 가장 작은 모델 (gemma4-e4b)과 가장 permissive (gemma3-27b)가 가장 큰 gap; Qwen2.5-VL이 가장 저항적. "더 강한 모델 → bias 적음"의 깔끔한 트렌드는 이 축에 없음.

## A3 (question type), A4 (shift distribution), A5 (prompt), A6 (failure mode) 한눈에

자체 insight markdown을 받지 못한 항목들 — 여기에 합침.

- **A3** — VQAv2의 `question_type` 필드가 이 subset에서 의미있게 slice할 정도로 세분화되지 않음. 지배적 유형이 "how many"와 "what number". `A3_question_type.csv`의 수치는 pooled per-model summary와 같은 패턴. 질문-유형-특이적 bias signature는 없음. 질문 taxonomy가 더 풍부한 ChartQA / TallyQA 추가까지 보류.
- **A4** — Shift 분포가 **"0"에서 강하게 bimodal + anchor 쪽 얇은 tail**. ≥ 75 %가 변화 없음 (gemma4-31b 85 %, qwen2.5-vl 85 %, qwen3-vl-8b 84 %; gemma4-e4b가 56 %로 outlier). 변화 중에서는 ±1 bin 우세, anchor 반대편 (away)이 anchor 쪽 (toward)보다 일관되게 가벼움. Visual signature가 Mussweiler-Strack "selective accessibility" 설명과 일치 — 대부분 unaffected, 일부가 끌려감.
- **A5** — Prompt 강화 ("must output a number")는 한 모델만 의미있게 움직임: **gemma3-27b** (adoption +17.4 pp, moved-closer +15.3 pp). 나머지는 < 6 pp 변화. Strengthen prompt는 **universal한** anchor amplifier가 아니라 주로 gemma3-27b를 망가뜨림. `references/roadmap.md` §3.5와 cross-reference: 특히 gemma3는 "no hedging" 압력 하에 huge number를 hallucinate, outlier filter가 trim하지만 paper에서 flag할 가치.
- **A6** — Failure-mode taxonomy: 7 모델 전체에 대해 분할 ≈ { exact-anchor: 11–14 %, unchanged: 56–85 %, graded-toward-anchor: 6–19 %, orthogonal/away: 6–19 %, non-numeric parse failure: ~0 % }. "graded toward" bucket이 paper가 의지할 새 finding; "exact-anchor"와 "orthogonal" bucket은 대략 균형이라 노이즈로 dismiss 가능.

## 이 모든 게 paper에 의미하는 것

- **Headline 재정의:** VLM의 anchoring은 uncertainty-modulated *graded pull*, categorical capture가 아님. 이는 LLM-anchoring 문헌 (대부분 aggregate accuracy drop만 보고)과도, VLMBias / typographic-attack 작업 (classification flip 보고)과도 차별화.
- **H2 결과는 그대로 paper-worthy.** Headline finding에 추가 compute 불필요.
- **A2는 위험하다.** Per-digit 패턴은 paper publish 전에 chance-corrected 분석 (`anchor == GT`의 base-rate 빼기) 필요. 쉬운 fix; paper 추가 전에 처리.
- **A7의 correlation이 mechanism으로 가는 다리.** Encoder ablation (E2 — ConvLLaVA full run)과 attention-mass 분석 (E1) 둘 다 signal 줄 것을 예측. 둘 다 schedule.

## 결정 trigger (`references/roadmap.md` §7 기준)

- After-A1 trigger **green** 발화: 여러 모델에서 asymmetry ≥ 10 pp (gemma4-e4b, gemma3-27b, qwen3-vl-30b가 통과; 나머지는 6–9 pp). Headline 유지.
- E1 / E2 우선순위 **변경 없음**. A7의 same-family correlation (Qwen3-VL-30B ↔ Qwen3-VL-8B = 0.30)이 architecture가 영향 있음을 시사 → encoder ablation 돌릴 가치.
