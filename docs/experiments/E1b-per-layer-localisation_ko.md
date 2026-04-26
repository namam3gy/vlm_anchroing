# E1b — anchor-attention gap의 per-layer 국소화

**상태:** `docs/experiments/E1-preliminary-results.md`의 후속. 이제 6-모델 패널 전체 (4 HFAttention + 2 inputs_embeds-path 확장), 각 n=200. 분석 스크립트: `scripts/analyze_attention_per_layer.py`. 원시 per-layer 테이블: `outputs/attention_analysis/_per_layer/per_layer_deltas.csv`. Figure: `outputs/attention_analysis/_per_layer/fig_delta_by_layer_{answer,step0}.png`.

## TL;DR — 6 모델에서의 세 가지 per-layer 관찰

1. **Anchor-attention gap은 layer 단위로 국소화되어 있고, 피크 layer가 encoder family에 따라 다르다.** Gemma-SigLIP은 **layer 5 / 42 (~12 % depth)**에서 delta **+0.050** (answer step)으로 피크; mid-stack cluster — CLIP-ViT (LLaVA-1.5), InternViT (InternVL3), ConvNeXt (ConvLLaVA) — 모두 layer 14–16 (~52 %)에서 δ ≈ +0.019–0.022로 피크; Qwen-ViT은 layer 22 / 28 (~82 %)에서 +0.015로 피크; FastViT (FastVLM)은 layer 22 / 28에서 **+0.047** — 패널 내 두 번째로 큰 magnitude, Qwen의 depth에서. E1의 layer-평균 수치 (≈ 0.005)는 최대 ~10×에 달하는 단일-layer 집중을 가리고 있었다.
2. **Gemma-SigLIP의 early-layer 피크는 음의 delta layer로 둘러싸여 있지만, 그 음들은 대부분 anchor/target trade-off이지 anti-anchor가 아니다.** Answer step에서 layer 0–4, 6–10, 12, 17은 모두 delta < 0 이고 CI 0 제외. Per-layer budget decomposition (아래 참조)을 보면 대부분이 두 이미지 사이의 anchor↔target mass swap (text는 불변)이고 능동적 anti-anchor layer가 아니다. Layer 5만이 실제로 text에서 anchor 이미지로 mass를 끌어오고 (δ_text = −0.038), layer 17만이 실제로 anchor에서 다시 text로 재분배한다. TL;DR의 "knife-edge" 비유는 layer 5의 *spike* 자체에는 맞지만 주변 layer들에는 맞지 않는다.
3. **A7 susceptibility gap은 피크 layer에 집중되고, magnitude는 강한 위계를 이룬다.** 각 모델의 answer-step 피크에서 top-decile-susceptible vs bottom-decile-resistant delta: **FastVLM +0.086** (엄청남, n=28 top-decile로 CI 넓음) → Qwen +0.025 → ConvLLaVA +0.013 → LLaVA-1.5 +0.010 → InternVL3 +0.006 → Gemma +0.001 (거의 0). FastVLM은 패널에서 가장 깨끗한 "susceptibility가 메커니즘을 gate"하는 사례. Gemma는 여전히 유일한 반전, 조건 없는 typographic-attack 프로파일과 일관.

## 설정

E1과 동일한 추출 파이프라인 + 두 가지 확장:

- **HFAttentionRunner path** (gemma4-e4b, qwen2.5-vl-7b-instruct, llava-1.5-7b, internvl3-8b): image-token id를 `input_ids`에서 스캔.
- **ConvLLaVA inputs_embeds path**: 텍스트 chunk embedding과 ConvNeXt vision feature를 concat하여 입력 구성; `scripts/extract_attention_mass.py`에서 splice 시점에 이미지 span 추적.
- **FastVLM -200-expand path**: `input_ids` 내의 `-200` 마커가 모델 forward 시점에 256개 FastViT patch embedding으로 치환됨; span은 step-0 attention k_len에서 역산.

각 triplet (sample_instance × {base, number, neutral})과 각 layer `l`에 대해:

```
delta_l = image_anchor_mass[l](number) − image_anchor_mass[l](neutral)
```

두 생성-step 라벨에서 평가: **answer** (decoded token이 숫자를 포함하는 첫 step — `per_step_tokens` 기반)과 **step 0** (첫 생성 토큰 — 보통 JSON 응답의 opening brace). Bootstrap 2,000 iter, 95 % CI.

Answer-step `n`이 두 모델에서 200 아래로 떨어짐: InternVL3 n=135, FastVLM n=75 — 둘 다 일부 생성에 digit token이 없어서. FastVLM은 `max_new_tokens=24` 사용 (다른 다섯은 8) — JSON 프롬프트 하에서 일반적으로 10–12 토큰 동안 prose 생성 후에야 (만약 나온다면) 답 digit 생성; roadmap §9 caveat.

## 모델별 피크 요약 (answer step, overall stratum)

| 모델 | 피크 layer | depth % | 피크 delta | 95 % CI | # sig-positive layers | n |
|---|---:|---:|---:|---|---:|---:|
| gemma4-e4b | **5 / 42** | 12 % | **+0.05007** | [+0.04772, +0.05229] | 19 / 42 | 200 |
| internvl3-8b | 14 / 28 | 52 % | +0.01931 | [+0.01348, +0.02535] | 24 / 28 | 135 |
| llava-1.5-7b | 16 / 32 | 52 % | +0.01883 | [+0.01441, +0.02310] | 26 / 32 | 200 |
| **convllava-7b** | **16 / 32** | **52 %** | **+0.02238** | [+0.01713, +0.02812] | — | 200 |
| qwen2.5-vl-7b | 22 / 28 | 82 % | +0.01527 | [+0.00842, +0.02307] | 22 / 28 | 200 |
| **fastvlm-7b** | **22 / 28** | **82 %** | **+0.04665** | [+0.02528, +0.07157] | — | 75 |

네 가지 tier (세 가지가 아님): SigLIP-Gemma 매우 이름 + 큼; mid-stack cluster (3개 encoder, L14–16에서 tight 피크 agreement) + moderate; Qwen-ViT 늦음 + moderate; FastVLM 늦음 + 큼 (Gemma magnitude에 근접).

ConvLLaVA 행이 원래 H3 가설에 대한 새 evidence의 핵심: ConvNeXt encoder임에도 피크 layer가 CLIP-ViT 기반 LLaVA-1.5와 정확히 일치, magnitude 20 % 이내, 메커니즘 동일. **H3의 "ConvNeXt가 ViT보다 덜 susceptible" 형태는 adoption 수준뿐 아니라 per-layer 수준에서도 falsified.**

## 모델별 피크 요약 (step 0, overall stratum)

| 모델 | 피크 layer | depth % | 피크 delta | 95 % CI |
|---|---:|---:|---:|---|
| gemma4-e4b | 6 / 42 | 15 % | +0.04505 | [+0.03817, +0.05177] |
| internvl3-8b | 3 / 28 | 11 % | +0.00760 | [+0.00665, +0.00858] |
| qwen2.5-vl-7b | 19 / 28 | 70 % | +0.00975 | [+0.00726, +0.01225] |
| llava-1.5-7b | 26 / 32 | 84 % | +0.01333 | [+0.01137, +0.01551] |
| **fastvlm-7b** | **27 / 28** | **96 %** | +0.01966 | [+0.01678, +0.02251] |
| **convllava-7b** | **31 / 32** | **100 %** | +0.02299 | [+0.01788, +0.02816] |

inputs_embeds-path 두 모델의 새 패턴: step-0 피크가 마지막 layer에 위치 (ConvLLaVA L31/32 = 100 %, FastVLM L27/28 = 96 %) — 자체의 answer-step 피크 (각각 L16, L22)와 다르며, 패널의 다른 모든 모델과도 다름. Causal test 없이는 해석이 speculative; 한 가능성은 inputs_embeds-path 모델들이 vision feature를 외부에서 처리하고 LLM의 final-layer integration step이 raw `input_ids` sequence에 vision token이 포함된 모델들과는 다른 역할을 한다는 것.

Gemma의 step-0 피크는 answer-step 피크와 사실상 같은 위치 (layer 5 vs. 6)이며, 신호는 39/42 layer에 걸쳐 broadcast (거의 모두 유의) — E1이 주장한 "Gemma는 prompt 통합 시점에 anchor를 인코딩하고 answer time에 같은 인코딩을 *재사용*한다"와 일치.

InternVL3은 두 step 간에 피크 layer가 바뀐다 (step-0 피크 layer 3, answer 피크 layer 14) — 초기 prompt-read은 shallow, answer-decision layer는 mid-stack. 이 패턴은 InternVL3을 "초기 등록 + mid-stack 결정"의 2단계 reader처럼 보이게 한다.

## 네 가지 encoder-family 원형 (6-모델 패널)

E1 Test 1 (전체 delta), E1 Test 3 (A7 stratum), 그리고 이번 layer 국소화를 종합:

| Family | 피크 layer depth | 피크 delta | 피크 A7 gap | 피크에서의 budget 출처 | Step-0 vs answer | 해석 |
|---|---|---|---|---|---|---|
| SigLIP (Gemma) | **매우 이른** (12 %) | 큼 (+0.050) | ~0 (+0.001) | **text** (−0.038) | 거의 동일 위치 | 이미지 속 텍스트를 조건 없이 이른 depth에서 등록. Anchor가 prompt text에서 mass를 가져옴 — SigLIP의 typographic-attack 계승과 일치. Susceptibility는 conditioner가 아님. |
| CLIP-ViT (LLaVA-1.5) | mid (52 %) | moderate (+0.019) | moderate (+0.010) | text (−0.029), target은 오히려 *증가* (+0.007) | answer mid / step-0 late | 두 이미지가 text에서 함께 attention을 pool. Anchor가 target과 경쟁하지 않고 "visual evidence"로 공-aggregate. |
| InternViT (InternVL3) | mid (52 %) | moderate (+0.019) | small (+0.006) | text (−0.014), target 약간 (−0.004) | step-0 early (L3) / answer mid | LLaVA-1.5처럼 text 중심이되 target도 약간 displacement. 2-stage: early 등록 + mid-stack 결정. |
| **ConvNeXt (ConvLLaVA)** | **mid (52 %)** | **moderate (+0.022)** | **moderate (+0.013)** | **text (−0.019)**, target 약간 (−0.003) | step-0 very late (L31/32 = 100 %) / answer mid | Answer step에서 LLaVA-1.5와 거의 구별 불가. Step-0 피크가 final LLM layer에 위치 — 해석은 열려 있음. |
| Qwen-ViT (Qwen2.5-VL) | **늦음** (82 %) | moderate (+0.015) | **큼** (+0.025, CI 분리) | **target** (−0.010) | 둘 다 늦음 co-located | Late-stack 조건부 anchor-vs-target 경쟁: answer-decision layer에서 anchor가 target 이미지를 밀어냄. |
| **FastViT (FastVLM)** | **늦음 (82 %)** | **큼 (+0.047)** | **엄청남 (+0.086)** | **text (−0.034)**, target (−0.014) | step-0 very late (L27/28 = 96 %) / answer late | Gemma-magnitude의 late-stack text-stealing에 패널 최고의 susceptibility gating. n=75 caveat 전제. |

6-모델 패널의 세 가지 read:

1. **"ViT vs. ConvNeXt" (원래 H3 축)은 "LLM stack 어디서 anchor가 읽히는가"보다 확정적으로 덜 정보적.** 세 encoder (CLIP-ViT, InternViT, ConvNeXt)가 layer 14–16에서 동일 메커니즘, 유사 magnitude로 수렴. 이 cluster 안에서는 architecture가 attention을 예측하지 않음.
2. **SigLIP과 FastViT은 구조적으로 다른 outlier지만 행동 지문(behavioural fingerprint)이 겹친다.** 둘 다 큰 magnitude (+0.050, +0.047)와 text-stealing budget source; *depth*에서 다름 (Gemma early, FastVLM late), *susceptibility gating*에서 다름 (Gemma 없음, FastVLM 패널 최고). 문헌에서 flagging하는 두 VLM failure mode — typographic attack (이미지 내 텍스트 read-off)과 "어느 이미지가 답에 해당" budget 혼란 — 이 FastVLM에서 co-firing하며, Gemma 혹은 Qwen 어느 쪽에서도 깨끗하게 격리되지 않는 방식으로 발생할 수 있음.
3. **Mid-stack cluster가 이제 패널 내 가장 깨끗한 E4 개입 target.** 서로 다른 세 encoder가 동일 프로파일 생성; mid-stack layer 14–16에서 architecture-blind intervention이 셋 모두에 작동할 수 있음. Gemma, FastVLM, Qwen은 각자의 설계 필요.

## 피크 layer budget decomposition: anchor는 어디서 attention을 훔치는가

각 layer의 attention mass는 1로 합해지므로, 특정 layer의 양의 `delta_anchor`는 같은 layer 내 다른 어딘가의 음의 delta로 상쇄되어야 한다. 그 상쇄 출처가 mechanistic 신호다 — anchor가 *target 이미지*, *prompt text*, 혹은 *이미 생성된 토큰*과 경쟁하는가?

각 모델의 answer-step 피크 layer에서 (number − neutral, answer step, triplet 평균, raw `sum_check ≈ 0`로 정규화 확인):

| 모델 | 피크 layer | δ anchor | δ target | δ text | δ generated |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | **+0.05007** | −0.00961 | **−0.03804** | −0.00242 |
| qwen2.5-vl-7b | 22 | +0.01527 | **−0.00972** | −0.00509 | −0.00046 |
| llava-1.5-7b | 16 | +0.01883 | +0.00702 | **−0.02940** | +0.00355 |
| internvl3-8b | 14 | +0.01931 | −0.00384 | −0.01427 | −0.00120 |
| convllava-7b | 16 | +0.02238 | −0.00254 | **−0.01920** | −0.00063 |
| fastvlm-7b | 22 | +0.04665 | −0.01393 | **−0.03427** | +0.00156 |

두 가지 구별되는 메커니즘이 여전히 나오고, 이제 6개 중 5개가 text-stealing 버킷:

- **Text-stealing (Gemma, LLaVA-1.5, InternVL3, ConvLLaVA, FastVLM).** Anchor 이미지가 주로 prompt text의 mass를 가져감. LLaVA-1.5에서는 target 이미지도 *증가* (+0.007) — 두 이미지가 attention pool하고 함께 text에서 가져감. ConvLLaVA와 InternVL3는 약간의 target-loss (≈ −0.003); FastVLM은 text (−0.034)와 target (−0.014) 둘 다 pull하지만 text 우세.
- **Target-stealing (Qwen 단독).** Qwen의 layer 22에서 anchor는 mass를 주로 target 이미지 (−0.010)에서 가져가고 text (−0.005)에서는 덜. 스택 말기에 이르면 답 생성 회로가 "관련 이미지를 본다"에 고정된 예산을 가지고, 그 안에서 anchor 이미지가 target을 대체.

이로써 위의 네 원형 테이블의 budget 컬럼이 확정:

- **Gemma (SigLIP):** early + text-stealing. Anchor 숫자가 초기 prompt-integration layer에서 prompt 상의 텍스트처럼 읽히며, pull은 text mass에서 옴 (target에서 아님).
- **Mid-stack cluster (LLaVA-1.5, InternVL3, ConvLLaVA):** mid + text-stealing (대체로). 셋 중 둘은 작은 target gain 혹은 작은 target loss — 두 이미지가 경쟁 아니라 협조 행동. A7 magnitude "중간" + layer 간 가장 연속적 프로파일 설명.
- **Qwen (Qwen-ViT):** late + target-stealing. Anchor가 answer-decision layer에서 target 이미지를 특이적으로 밀어냄.
- **FastVLM (FastViT):** late + text-stealing + 큼. 패널에서 독특. Gemma의 magnitude와 text-source + Qwen의 depth 조합.

## Gemma layer 5 재해석 — anti-anchor 회로가 아니라 국소적 text → anchor pull

주변 음의-delta layer들 (0–4, 6–10, 12, 17)은 **능동적 anti-anchor 회로가 아니다**. Per-layer budget 체크로 설명됨:

- Layer 0–10 (5 제외): δ_anchor와 δ_target이 거의 동일 크기 반대 부호 (예: layer 2에서 δ_target=+0.0136, δ_anchor=−0.0135). 이건 anchor/target trade-off layer — mechanical, non-directional. 두 이미지 사이에서 mass를 주고받을 뿐 text는 거의 영향 없음 (|δ_text| < 0.004).
- Layer 5: δ_anchor=+0.050, δ_text=−0.038, δ_target=−0.010. Text가 가장 많이 잃음. 여기 이 한 layer만이 anchor가 진짜 text stream에서 끌어당기는 곳.
- Layer 17: δ_anchor=−0.020, δ_text=+0.016 (CI 0 제외), δ_target ≈ 0. Anchor가 잃지만 target이 아니라 *text*로 감. 가장 anti-anchor에 가까운 layer지만 target-rebalance가 아니라 text-rebalance.

즉 TL;DR의 "음의 delta layer로 둘러싸인 knife-edge spike" 표현은 한 가지 자격 조건이 필요: layer 5 자체의 spike는 real/knife-edge지만, 주변 음의-delta layer들의 대부분은 mechanical anchor/target swap이지 능동적 suppression이 아니다. Layer 17만이 유의미하게 anchor에서 mass를 돌리는데, 그마저도 target이 아니라 text로 돌린다.

고려한 대체 해석: 순수 *budget redistribution* (방향성 pull 없이 spike에 대한 정규화 artefact 뿐). Layer 5에서 δ_text=−0.038라는 점이 이를 기각함 — anchor가 특이적으로 prompt text에서 끌어당김, 무작위 다른 영역이 아님. 이는 SigLIP의 typographic-attack 계승과 일관: digit가 이미지에서 텍스트처럼 읽히고, text mass가 자연스러운 출처.

E1 writeup은 Gemma에 대해 input-side 개입을 제안 (anchor의 visual-projection feature를 0으로, 혹은 input layer에서 KV cache 패치). Per-layer + budget 데이터가 구체적 target을 제공: **layer 5 이전 projection/KV patch**로 layer 5에서의 text-stealing pull을 차단. Answer step의 attention re-weighting은 작동하지 않는다 — 손상은 prompt 처리 중 일어나고 cache에 남아있기 때문.

## A7 susceptibility: per-layer gap

각 모델의 answer-step 피크 layer (두 stratum 모두 동일 layer)에서 top-decile-susceptible 에서 bottom-decile-resistant를 뺀 delta:

| 모델 | 피크 layer | top delta [CI] | bottom delta [CI] | gap |
|---|---:|---|---|---:|
| **fastvlm-7b** | 22 | **+0.10054 [+0.04728, +0.16007]** | +0.01454 [+0.00650, +0.02237] | **+0.08600** |
| qwen2.5-vl-7b | 22 | +0.02794 [+0.01532, +0.04271] | +0.00259 [−0.00204, +0.00602] | +0.02535 |
| convllava-7b | 16 | +0.02868 [+0.02043, +0.03836] | +0.01607 [+0.01053, +0.02148] | +0.01262 |
| llava-1.5-7b | 16 | +0.02369 [+0.01600, +0.03177] | +0.01398 [+0.00999, +0.01782] | +0.00970 |
| internvl3-8b | 14 | +0.02302 [+0.01273, +0.03766] | +0.01726 [+0.01064, +0.02362] | +0.00576 |
| gemma4-e4b | 5 | +0.05078 [+0.04758, +0.05417] | +0.04936 [+0.04628, +0.05260] | +0.00142 |

패널 내 두 개의 "가장 깨끗한 susceptibility gate":

- **FastVLM:** top-decile +0.10, bottom-decile +0.015 — 6.5× ratio, top-decile n=28 / bottom-decile n=47 (62 % digit-coverage filter 후). Gap (+0.086)이 다른 모델의 3×; top-decile CI는 넓지만 bottom-decile CI와 겹치지 않음. Caveat: n=28이므로 +0.10 point estimate가 낙관적일 수 있음 — 더 큰 run이 타이트하게 만들 것. 최고-leverage E4 개입 site 후보.
- **Qwen-ViT:** bottom-decile CI가 모델 자체의 피크 layer에서 *0을 포함* — resistant item에 대해서는 Qwen이 late-stack attention gap을 아예 발달시키지 않음. Susceptible item에 대해서는 같은 layer가 +0.028로 발화. 단일 layer가 A7 신호 거의 전부를 운반, susceptible item에만. 패널 내 가장 tight한 "susceptibility-gated answer-step 회로" (FastVLM의 넓은 CI 제외).

ConvLLaVA A7 gap (+0.013)은 LLaVA-1.5 (+0.010)과 Qwen (+0.025) 사이에 위치, mid-stack-cluster 소속과 일치하며 ConvNeXt 기반 모델도 일부 susceptibility-gated 구조를 발달시킨다는 것 확인 — 단 ViT 기반 모델보다는 덜.

## E4 (mitigation)에 대한 함의 — 후보 site (검증 전)

다음은 관찰된 per-layer + budget 데이터에서 생성한 가설이다; 아직 ablate하거나 re-weight하지 않음. E1 writeup의 결론은 유효 ("단일 범용 mitigation은 작동하지 않는다"); 6-모델 per-layer + budget 데이터는 네 원형에 걸친 E4의 *후보* 개입 site를 좁힌다:

- **SigLIP-Gemma:** input-side 후보. Layer 5 이전을 target으로, 해당 layer의 text→anchor pull을 차단하는 것이 구체 목표. 생성 시점의 attention re-weighting은 작동하지 않을 가능성 (피크가 이미 KV cache 안). 시도 옵션: 두 번째 이미지의 projection-feature masking, 또는 encoder / early LLM layer 경계에서 KV-cache patching.
- **Mid-stack cluster (LLaVA-1.5, InternVL3, ConvLLaVA):** layer ~14–16, answer-step의 mid-stack attention re-weighting 후보. Budget 출처가 주로 text이기 때문에 anchor down-weight는 mass를 target이 아니라 text로 되돌림. **패널에서 최고-leverage** — 세 encoder가 같은 프로파일 공유이므로 한 번 튜닝한 intervention이 CLIP-ViT, InternViT, ConvNeXt 모델 전반에 generalise될 수 있음.
- **Qwen-ViT:** layer 22 ± 2, answer-step only, per-item susceptibility 신호로 gate한 late-stack attention re-weighting 후보. 이 layer의 budget은 anchor-vs-text가 아니라 anchor-vs-target이므로, anchor down-weight는 mass를 text가 아니라 target 이미지로 되돌려야 함 — 원하는 결과.
- **FastVLM (FastViT):** layer 22 ± 2 late-stack re-weighting, susceptibility gate. A7 gap이 여기서 매우 커서 (top-decile δ ≈ +0.10 vs bottom-decile ≈ +0.015) A7-gated rescale로 큰 direction-follow 감소 가능 — 하지만 digit filter 통과한 top-decile triplet이 n=28뿐이므로 설계 노력을 commit하기 전 더 큰 run 또는 `max_new_tokens ≥ 32`로 gap 검증 필요.

패널은 이제 논문-레벨 가설과 부합한다: **cross-modal bias의 LLM stack 내 위치, magnitude, 그리고 경쟁 영역이 encoder-family-specific이며, architecture-aware mitigation 설계가 필요하다.** 이 가설이 causal E4 개입에서 살아남는지는 TBD — 이건 관찰된 mechanistic evidence이지 검증된 fix가 아니다.

## Caveat

- Step 0에서는 모델당 `n = 200`. Answer step에서 InternVL3은 n=135, FastVLM은 n=75로 떨어짐 (둘 다 일부 생성에 digit token이 없어서; FastVLM은 roadmap §9 "prose before digit" issue). FastVLM의 answer-step 피크 CI가 넓음 ([+0.025, +0.072]) — 더 큰 run 또는 `max_new_tokens ≥ 32`가 타이트하게 만들 것.
- "피크 layer"는 `argmax(delta_mean)`으로 선택. 피크의 CI가 타이트해서 안정적이지만, Gemma에서 5 vs. 6 또는 LLaVA-1.5에서 16 vs. 17의 정확한 선택은 CI overlap 내.
- FastVLM은 `max_new_tokens=24` 사용, 다른 다섯은 8. "answer step"으로 간주되는 step 인덱스에 영향을 주지만, digit이 나오는 step의 delta 자체엔 영향 없음 (비교는 triplet 내부).
- ConvLLaVA와 FastVLM 결과는 이번 round에 추가된 `inputs_embeds`-path 확장에서 나옴. 추출 스크립트 확장은 smoke-test 완료, span 계산은 expected expanded seq_len과 cross-check됨 (정규화된 mass의 `sum_check ≈ 0` 및 FastVLM span-reconstruction assertion 모두 record마다 통과). 여전히: 이는 최초 통합이므로 독립 cross-check까지는 수치를 preliminary로 취급.
- Layer-wise mass는 head 평균. Head-수준 sparsity (Gemma layer 5의 단일 head가 신호를 운반) 은 여기서 보이지 않으며 회로 주장을 더 강화할 것. E1b 범위 밖.

## Roadmap 업데이트

- §3 status: 6 모델 전반 E1b layer analysis done (이 파일). Cross-family 패턴이 이제 4-way: SigLIP early + 큼 / 세 encoder의 mid-stack cluster / Qwen-ViT late + target-stealing / FastViT late + 큼 + 최강 A7 gating.
- §6 Tier 1 E1: **두 open question closed** — "per-layer localisation" + "ConvNeXt/FastViT 확장". 남은 것: head-level sparsity, causal test (각 family의 peak layer에서 anchor token ablate하고 `direction_follow` 측정).
- §10 changelog entry 2026-04-24.
