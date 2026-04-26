# E1b — Anchor는 family마다 stack의 다른 layer에서, 다른 메커니즘으로 LLM을 공격한다

**상태:** E1 attention 데이터 재분석 (추가 compute 없음). 전체 writeup: `docs/experiments/E1b-per-layer-localisation.md`. 원시 데이터: `outputs/attention_analysis/{model}/{run}/per_step_attention.jsonl`. 집계 테이블: `outputs/attention_analysis/_per_layer/{per_layer_deltas,peak_layer_summary,peak_budget_decomposition}.csv`. 스크립트: `scripts/analyze_attention_per_layer.py`. 재현 노트북: `notebooks/E1b_per_layer_localisation.ipynb`.

## 질문

E1은 `attention_anchor(number) − attention_anchor(neutral)`을 **layer 평균**해서 answer-digit step에서 +0.004 ~ +0.007의 깨끗한 신호를 보고했다 (처음에 4 모델, 이제 inputs_embeds-path의 ConvLLaVA, FastVLM 확장으로 6 모델). 평균이 두 가지를 가리고 있었다:

1. **Anchor가 LLM stack 어디에서 실제로 읽히는가?** 28개 layer에 걸친 평균 0.005와 한 layer의 0.050 spike는 layer-평균에서는 구분되지 않지만 E4 개입 site가 완전히 다름.
2. **Anchor의 attention mass는 무엇을 밀어내는가?** Layer `l`에서 anchor가 gain하면 같은 layer의 다른 무언가가 lose해야 한다. Text? Target image? 그 구분이 메커니즘.

이 insight는 같은 E1 jsonl 파일에서 두 질문을 모두 답한다.

## 방법

각 triplet `(sample_instance × {base, number, neutral})`와 각 layer `l`에 대해, answer-digit step에서 `delta_l = anchor_mass(number, l) − anchor_mass(neutral, l)` 계산. Bootstrap 2,000 iter, 95 % CI, 모델당 유효 triplet ~200개 기준. 그리고 각 모델의 `argmax_l delta_l` layer에서 전체 budget 보고: `δ(anchor)`, `δ(target image)`, `δ(text)`, `δ(generated)`. 구조상 이 넷의 합은 0 (attention 정규화); 부호가 anchor가 어느 영역과 경쟁하는지 알려준다.

## 결과 1: 피크 layer가 encoder family 간 ~6× 차이

Answer-step 피크 layer와 delta, overall stratum (6-모델 패널):

| 모델 | 피크 layer | depth | 피크 δ | 95 % CI | n |
|---|---:|---:|---:|---|---:|
| gemma4-e4b (SigLIP) | **5 / 42** | **12 %** | **+0.0501** | [+0.0477, +0.0523] | 200 |
| internvl3-8b (InternViT) | 14 / 28 | 52 % | +0.0193 | [+0.0135, +0.0254] | 135 |
| llava-1.5-7b (CLIP-ViT) | 16 / 32 | 52 % | +0.0188 | [+0.0144, +0.0231] | 200 |
| convllava-7b (ConvNeXt) | **16 / 32** | **52 %** | +0.0224 | [+0.0171, +0.0281] | 200 |
| qwen2.5-vl-7b (Qwen-ViT) | 22 / 28 | 82 % | +0.0153 | [+0.0084, +0.0231] | 200 |
| fastvlm-7b (FastViT) | **22 / 28** | **82 %** | **+0.0467** | [+0.0253, +0.0716] | 75 |

네 가지 원형이 드러남:

- **SigLIP-Gemma:** 매우 이른 peak (L5, 12 %), mid-stack 세 모델의 ~3× magnitude.
- **Mid-stack cluster (CLIP-ViT, InternViT, ConvNeXt):** 모두 layer 14–16 (~52 %)에서 peak, δ ≈ +0.019–0.022. **H3 "ConvNeXt < ViT" 형태는 확정적으로 falsified** — ConvLLaVA가 LLaVA-1.5와 동일한 peak layer, 거의 동일한 magnitude.
- **Qwen-ViT:** 늦은 peak (L22, 82 %), moderate δ.
- **FastViT:** 늦은 peak (L22, 82 %)지만 **Gemma 수준 magnitude** (+0.047). FastVLM n=75 (max_new_tokens=24로 JSON 프롬프트 하에서 62 % digit coverage; roadmap §9 caveat) — CI 더 넓음.

Layer-평균 E1 수치 (≈ 0.005)는 그 값의 10×에 달하는 단일-layer 집중을 가리고 있었다.

## 결과 2: Budget 출처가 패널을 두 메커니즘으로 가른다

각 모델 answer-step 피크 layer에서 anchor가 어디서 mass를 가져가는가:

| 모델 | 피크 | δ anchor | δ target | δ text | δ gen |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | +0.0501 | −0.0096 | **−0.0380** | −0.0024 |
| qwen2.5-vl-7b | 22 | +0.0153 | **−0.0097** | −0.0051 | −0.0005 |
| llava-1.5-7b | 16 | +0.0188 | +0.0070 | **−0.0294** | +0.0036 |
| internvl3-8b | 14 | +0.0193 | −0.0038 | **−0.0143** | −0.0012 |
| convllava-7b | 16 | +0.0224 | −0.0025 | **−0.0192** | −0.0006 |
| fastvlm-7b | 22 | +0.0467 | −0.0139 | **−0.0343** | +0.0016 |

두 메커니즘 — 이제 6개 중 5개가 text-stealing 버킷 (이전 4개 중 3개가 아님):

- **Text-stealing (6개 중 5개: Gemma, LLaVA-1.5, InternVL3, ConvLLaVA, FastVLM).** Anchor가 주로 prompt text에서 mass를 가져감. LLaVA-1.5에서는 target image가 *증가* (+0.007) — 두 이미지가 공-aggregate하고 함께 text에서 가져감. ConvLLaVA와 InternVL3는 약간의 target-loss (≈ −0.003); FastVLM은 text (−0.034)와 target (−0.014) 둘 다에서 끌어오지만 text 우세.
- **Target-stealing (Qwen 단독).** Qwen만이 mass를 주로 target 이미지 (−0.010)에서 가져가고 text (−0.005)에서는 덜. Layer 22에서 answer-generation 회로가 "관련 이미지를 본다"에 고정된 예산을 가지고, 그 안에서 anchor가 target을 대체.

"Depth × 메커니즘" 축 결합으로 네 원형:

1. **Early + text-stealing + large** — SigLIP (Gemma). Prompt integration 중 typographic-attack read-off.
2. **Mid + text-stealing + moderate** — CLIP-ViT (LLaVA-1.5), InternViT (InternVL3), ConvNeXt (ConvLLaVA). 서로 다른 세 encoder가 동일 프로파일로 수렴.
3. **Late + target-stealing + moderate** — Qwen-ViT (Qwen2.5-VL). 두 이미지가 answer-decision layer에서 경쟁하는 유일 모델.
4. **Late + text-stealing + large** — FastViT (FastVLM). Magnitude는 Gemma 수준; 위치는 Qwen과 동일; budget source는 Gemma와 동일. 패널 내 독특한 hybrid.

## 결과 3: Gemma의 layer 5 주변 "음의 delta" layer들은 대부분 mechanical이지 anti-anchor가 아니다

Layer 0–4, 6–10, 12, 17이 모두 `delta < 0`이고 CI가 0 제외 — 표면적으로는 "anchor-aversive". Per-layer budget 체크:

- Layer 0–10 (5 제외): `δ_anchor`와 `δ_target`이 같은 크기 반대 부호 (예: layer 2에서 δ_target = +0.0136, δ_anchor = −0.0135). **Anchor/target mass swap — text는 불변.** 능동적 suppression 아님; 두 이미지 사이에서 attention을 주고받을 뿐.
- Layer 5: `δ_anchor = +0.050, δ_text = −0.038, δ_target = −0.010`. Anchor가 **진짜로** text stream에서 끌어오는 **유일한** layer.
- Layer 17: `δ_anchor = −0.020, δ_text = +0.016`. Anchor에서 mass를 진짜 redirect — 하지만 target이 아니라 text로.

"Knife-edge spike" 비유는 layer 5에는 맞지만 주변 layer들에는 맞지 않는다. 대체 해석 (순수 budget redistribution artefact)은 layer 5의 pull이 특이적으로 text 방향이라는 점에서 기각.

## 메커니즘에 대한 함의

- **Bias가 stack 내에 사는 위치는 encoder-family-specific이지 universal이 아님.** "VLM은 anchor 이미지에 attention한다"는 논문-레벨 주장은 평균적으로는 true지만 설계에서는 misleading — 다른 encoder는 다른 depth에서 다른 경쟁 영역과 함께 다른 magnitude로 그렇게 한다.
- **H3 "ConvNeXt < ViT" 형태는 death.** ConvLLaVA의 per-layer 프로파일이 LLaVA-1.5와 거의 구분되지 않음 (같은 피크 layer, 같은 메커니즘, 유사한 magnitude). 순수 encoder architecture (Conv vs. ViT)는 anchoring을 예측하지 못함. *post-projection LLM-layer-depth* 축이 예측함.
- **SigLIP-Gemma의 패턴은 typographic-attack read-off에 부합.** Early layer + text-stealing + susceptibility-gating 없음 (E1 Test 3, A7 gap +0.001)이 함께 "digit가 prompt 통합 중 이미지에서 텍스트처럼 읽히며, downstream utility와 무관"하게 보이며, 이는 SigLIP-family encoder의 publish된 failure mode.
- **FastViT (FastVLM)가 새 원형을 추가: late + text-stealing + large + 가장 강한 A7 gating.** Peak δ +0.047은 다른 mid/late 모델의 3×이며 Gemma에 비견, 하지만 peak가 Qwen-ViT의 depth (L22)에 위치. 피크에서 A7 gap = +0.086 — Qwen의 +0.025의 3×, ConvLLaVA의 +0.013의 8×. n=75 caveat 전제 하에, FastVLM은 "susceptible item에 대해 typographic-read-off head로 변하는 late-stack decision layer"로 보임. 두 publish된 failure mode — typographic-attack (이미지 내 텍스트 읽기) vs "질문에 안 맞는 이미지" — 가 여기서 겹쳐 일어날 수 있음.
- **Qwen-ViT의 패턴은 answer layer에서의 anchor-vs-target 경쟁.** Late layer + target-stealing + susceptibility-gated (동일 피크 layer에서 bottom-decile CI 0 포함)이 "어느 이미지를 볼지" 고정 예산을 가진 answer-generation 회로가 susceptible item에서 드러나는 패턴과 일관.
- **Mid-stack cluster (CLIP-ViT, InternViT, ConvNeXt)가 공통 프로파일을 공유.** Mid-stack + text-stealing + moderate susceptibility gating. 패널에서 가장 깨끗한 "default VLM" 메커니즘 — 세 encoder 모두에서 replicate되어 가장 안정적 원형.

## Experiment plan (`references/roadmap.md` §6)에 대한 함의

- **E4 (mitigation)은 이제 단일 개입을 설계할 수 없음.** 이제 네 원형 걸친 후보 site (관찰만, E4에서 검증 예정):
  - SigLIP-Gemma: **input-side pre-layer-5** projection/KV patch. Answer-step re-weighting은 작동 불가 — 손상이 cache에 있음.
  - Mid-stack cluster (LLaVA-1.5 / InternVL3 / ConvLLaVA): **mid-stack ~14–16** attention re-weighting, answer-step. Mass를 text로 되돌림 — 덜 이상적이지만 *세 encoder replication*으로 패널 내 최고-leverage target — 한 개입이 세 모델 모두에 generalise될 수 있음.
  - Qwen-ViT: **late-stack layer 22 ± 2** attention re-weighting, susceptibility로 gate. Anchor down-weight가 mass를 target 이미지로 되돌림 (원하는 결과) — 예산이 anchor-vs-target이기 때문.
  - FastViT-FastVLM: **late-stack layer 22 ± 2** re-weighting, susceptibility gate. A7 gap이 여기서 매우 커서 (top-decile δ ≈ +0.10 vs bottom-decile ≈ +0.015) A7-gated rescale로 큰 direction-follow 감소 가능 — 하지만 n이 가장 작으므로 commit 전 더 큰 run으로 검증 필요.
- **Layer 5 pull의 causal test** (`docs/experiments/E1-preliminary-results.md`의 open question 4). Gemma의 layer 5 또는 그 이전에서 anchor-image token을 ablate하고 `direction_follow` delta 측정. 가장 싼 메커니즘-수준 주장이자 E4에 직접 입력.
- **inputs_embeds-path 확장 — 완료.** ConvLLaVA + FastVLM full n=200 run이 이제 패널에 포함 (open question 1 closed). 남은 E1 open question: head-level sparsity + causal test.

## Caveat

- Step 0에서는 모델당 `n = 200`. Answer step에서 InternVL3는 n=135, FastVLM은 n=75로 떨어짐 — 둘 다 일부 생성에 digit token이 없어서. 특히 FastVLM은 roadmap §9의 "prose before digit" issue에 해당 — 본 run은 `max_new_tokens=24` 사용 (다른 다섯은 8), 그럼에도 62 % record만 rescue. FastVLM의 answer-step 피크 CI가 넓음 ([+0.025, +0.072]).
- `argmax delta`는 안정적이지만 multi-peak 분포를 무시함. 전체 per-layer trace figure (`fig_delta_by_layer_answer.png`)은 Gemma가 단일-spike이고 네 mid/late 모델이 더 연속적이며 FastVLM은 layer 10–22에 걸쳐 시각적으로 monotonic하게 상승함을 보여줌.
- Layer-wise mass는 head 평균. Head-level sparsity 주장 (예: "Gemma layer 5의 한 head가 신호를 운반")은 회로 이야기를 강화할 것이지만 본 insight의 범위 밖.
- FastVLM은 `max_new_tokens=24` 사용, 다른 다섯은 8. Answer-step 인덱스의 *분포*에 영향을 주지만 mass 자체엔 영향 없음 — triplet 내에서 같은 step을 조건 간 비교.
- Attention은 상관이지 인과 아님. E4가 인과 주장을 검증하는 곳.

## Roadmap 항목

§6 Tier 1 E1: "per-layer localisation" + "ConvNeXt/FastViT extension" 두 open question 모두 closed. 남은 것: head-level sparsity 분석, causal test (각 family의 peak layer에서 anchor-image token ablate하고 `direction_follow` 측정).
