# E1b — Anchor는 family마다 stack의 다른 layer에서, 다른 메커니즘으로 LLM을 공격한다

**상태:** E1 attention 데이터 재분석 (추가 compute 없음). 전체 writeup: `docs/experiments/E1b-per-layer-localisation.md`. 원시 데이터: `outputs/attention_analysis/{model}/{run}/per_step_attention.jsonl`. 집계 테이블: `outputs/attention_analysis/_per_layer/{per_layer_deltas,peak_layer_summary,peak_budget_decomposition}.csv`. 스크립트: `scripts/analyze_attention_per_layer.py`. 재현 노트북: `notebooks/E1b_per_layer_localisation.ipynb`.

## 질문

E1은 `attention_anchor(number) − attention_anchor(neutral)`을 **layer 평균**해서 4개 encoder family 모두 answer-digit step에서 +0.004 ~ +0.007의 깨끗한 신호를 보고했다. 평균이 두 가지를 가리고 있었다:

1. **Anchor가 LLM stack 어디에서 실제로 읽히는가?** 28개 layer에 걸친 평균 0.005와 한 layer의 0.050 spike는 layer-평균에서는 구분되지 않지만 E4 개입 site가 완전히 다름.
2. **Anchor의 attention mass는 무엇을 밀어내는가?** Layer `l`에서 anchor가 gain하면 같은 layer의 다른 무언가가 lose해야 한다. Text? Target image? 그 구분이 메커니즘.

이 insight는 같은 E1 jsonl 파일에서 두 질문을 모두 답한다.

## 방법

각 triplet `(sample_instance × {base, number, neutral})`와 각 layer `l`에 대해, answer-digit step에서 `delta_l = anchor_mass(number, l) − anchor_mass(neutral, l)` 계산. Bootstrap 2,000 iter, 95 % CI, 모델당 유효 triplet ~200개 기준. 그리고 각 모델의 `argmax_l delta_l` layer에서 전체 budget 보고: `δ(anchor)`, `δ(target image)`, `δ(text)`, `δ(generated)`. 구조상 이 넷의 합은 0 (attention 정규화); 부호가 anchor가 어느 영역과 경쟁하는지 알려준다.

## 결과 1: 피크 layer가 encoder family 간 ~6× 차이

Answer-step 피크 layer와 delta, overall stratum:

| 모델 | 피크 layer | depth | 피크 δ | 95 % CI |
|---|---:|---:|---:|---|
| gemma4-e4b (SigLIP) | **5 / 42** | **12 %** | **+0.0501** | [+0.0477, +0.0523] |
| llava-1.5-7b (CLIP-ViT) | 16 / 32 | 52 % | +0.0188 | [+0.0144, +0.0231] |
| internvl3-8b (InternViT) | 14 / 28 | 52 % | +0.0193 | [+0.0135, +0.0254] |
| qwen2.5-vl-7b (Qwen-ViT) | **22 / 28** | **82 %** | +0.0153 | [+0.0084, +0.0231] |

세 개의 tier: SigLIP-Gemma 매우 이른 depth + 나머지 셋의 ~3× 크기; CLIP-ViT과 InternViT mid-stack; Qwen-ViT 늦음. Layer-평균 E1 수치 (≈ 0.005)는 그 값의 10×에 달하는 단일-layer 집중을 가리고 있었다.

## 결과 2: Budget 출처가 패널을 두 메커니즘으로 가른다

각 모델 피크 layer에서 anchor가 어디서 mass를 가져가는가:

| 모델 | 피크 | δ anchor | δ target | δ text | δ gen |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | +0.0501 | −0.0096 | **−0.0380** | −0.0024 |
| qwen2.5-vl-7b | 22 | +0.0153 | **−0.0097** | −0.0051 | −0.0005 |
| llava-1.5-7b | 16 | +0.0188 | +0.0070 | **−0.0294** | +0.0036 |
| internvl3-8b | 14 | +0.0193 | −0.0038 | **−0.0143** | −0.0012 |

두 가지 구별되는 메커니즘:

- **Text-stealing (Gemma, LLaVA-1.5, InternVL3).** Anchor가 주로 prompt text에서 mass를 가져감. LLaVA-1.5에서는 target image *도* 증가 (+0.007) — 두 이미지가 attention을 pool하고 함께 text에서 가져감. "Anchor가 또 하나의 visual evidence로 등록"되는 패턴과 일치.
- **Target-stealing (Qwen만).** Anchor가 mass를 주로 target 이미지에서 가져가고 text에서는 덜. Layer 22에서 answer-generation 회로가 "관련 이미지를 본다"에 고정된 예산을 가지고, 그 안에서 anchor가 target을 대체.

## 결과 3: Gemma의 layer 5 주변 "음의 delta" layer들은 대부분 mechanical이지 anti-anchor가 아니다

Layer 0–4, 6–10, 12, 17이 모두 `delta < 0`이고 CI가 0 제외 — 표면적으로는 "anchor-aversive". Per-layer budget 체크:

- Layer 0–10 (5 제외): `δ_anchor`와 `δ_target`이 같은 크기 반대 부호 (예: layer 2에서 δ_target = +0.0136, δ_anchor = −0.0135). **Anchor/target mass swap — text는 불변.** 능동적 suppression 아님; 두 이미지 사이에서 attention을 주고받을 뿐.
- Layer 5: `δ_anchor = +0.050, δ_text = −0.038, δ_target = −0.010`. Anchor가 **진짜로** text stream에서 끌어오는 **유일한** layer.
- Layer 17: `δ_anchor = −0.020, δ_text = +0.016`. Anchor에서 mass를 진짜 redirect — 하지만 target이 아니라 text로.

"Knife-edge spike" 비유는 layer 5에는 맞지만 주변 layer들에는 맞지 않는다. 대체 해석 (순수 budget redistribution artefact)은 layer 5의 pull이 특이적으로 text 방향이라는 점에서 기각.

## 메커니즘에 대한 함의

- **Bias가 stack 내에 사는 위치는 encoder-family-specific이지 universal이 아님.** "VLM은 anchor 이미지에 attention한다"는 논문-레벨 주장은 평균적으로는 true지만 설계에서는 misleading — 다른 encoder는 다른 depth에서 다른 경쟁 영역과 함께 그렇게 한다.
- **SigLIP-Gemma의 패턴은 typographic-attack read-off에 부합.** Early layer + text-stealing + susceptibility-gating 없음 (E1 Test 3)이 함께 "digit가 prompt 통합 중 이미지에서 텍스트처럼 읽히며, downstream utility와 무관"하게 보이며, 이는 SigLIP-family encoder의 publish된 failure mode.
- **Qwen-ViT의 패턴은 answer layer에서의 anchor-vs-target 경쟁.** Late layer + target-stealing + susceptibility-gated (동일 피크 layer에서 bottom-decile CI 0 포함)이 "어느 이미지를 볼지" 고정 예산을 가진 answer-generation 회로가 susceptible item에서 드러나는 패턴과 일관.
- **CLIP-ViT / InternViT는 중간에 위치.** Mid-stack + text-stealing + moderate susceptibility gating. 아직 깨끗한 메커니즘이 아님 — per-layer 프로파일이 두 극단보다 연속적.

## Experiment plan (`references/roadmap.md` §6)에 대한 함의

- **E4 (mitigation)은 이제 단일 개입을 설계할 수 없음.** Family별 후보 site (관찰만, E4에서 검증 예정):
  - SigLIP-Gemma: **input-side pre-layer-5** projection/KV patch. Answer-step re-weighting은 작동 불가 — 손상이 cache에 있음.
  - Qwen-ViT: **late-stack layer 22 ± 2** attention re-weighting, susceptibility로 gate. Anchor down-weight가 mass를 target 이미지로 되돌려야 (원하는 결과) — 예산이 anchor-vs-target이기 때문.
  - CLIP-ViT / InternViT: **mid-stack ~14–16** attention re-weighting. Mass를 target이 아니라 text로 되돌림 — 덜 이상적, 여전히 시도할 가치.
- **ConvLLaVA + FastVLM (inputs_embeds path)이 빠진 행.** "Location이 encoder family에 따라 다르다"를 주장하기 전에 논문에 필요한 6-model panel 완성. 추출 스크립트 확장이 다음 E1 단계, `references/roadmap.md` §6 참조.
- **Layer 5 pull의 causal test (E1 preliminary results의 open question 4).** Gemma의 layer 5 또는 그 이전에서 anchor-image token을 ablate하고 `direction_follow` delta 측정. 가장 싼 메커니즘-수준 주장이자 E4에 직접 입력.

## Caveat

- 모델당 step당 `n = 200`; InternVL3은 일부 생성에 digit token이 없어 answer step에서 `n = 135`로 떨어짐. Peak-layer CI는 타이트하지만 정확한 선택 (예: Gemma에서 5 vs. 6, LLaVA-1.5에서 16 vs. 17)은 CI overlap 내.
- `argmax delta`는 안정적이지만 multi-peak 분포를 무시함. 전체 per-layer trace figure (`fig_delta_by_layer_answer.png`)은 Gemma가 단일-spike이고 세 ViT-family 모델은 더 연속적임을 보여줌.
- Layer-wise mass는 head 평균. Head-level sparsity 주장 (예: "Gemma layer 5의 한 head가 신호를 운반")은 회로 이야기를 강화할 것이지만 본 insight의 범위 밖.
- Attention은 상관이지 인과 아님. E4가 인과 주장을 검증하는 곳.

## Roadmap 항목

§6 Tier 1 E1: "per-layer localisation" open question 체크. 남은 것: ConvNeXt/FastViT 확장, causal test.
