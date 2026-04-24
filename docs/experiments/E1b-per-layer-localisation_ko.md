# E1b — anchor-attention gap의 per-layer 국소화

**상태:** `docs/experiments/E1-preliminary-results.md`의 후속. 동일한 4개 모델 × n=200 attention run을 재분석하되 layer-평균 delta를 per-layer trace로 분해. 분석 스크립트: `scripts/analyze_attention_per_layer.py`. 원시 per-layer 테이블: `outputs/attention_analysis/_per_layer/per_layer_deltas.csv`. Figure: `outputs/attention_analysis/_per_layer/fig_delta_by_layer_{answer,step0}.png`.

## TL;DR — 세 가지 per-layer 관찰

1. **Anchor-attention gap은 layer 단위로 국소화되어 있고, 피크 layer가 encoder family에 따라 다르다.** Gemma-SigLIP은 **layer 5 / 42 (~12 % depth)**에서 delta **+0.050** (answer step)으로 피크; Qwen-ViT은 **layer 22 / 28 (~82 % depth)**에서 +0.015로 피크; CLIP-ViT과 InternViT은 모두 **mid-stack (~52 % depth)**에서 ~+0.019로 피크. E1의 layer-평균 수치는 단일 layer가 균일 분포 가정 대비 ~3× 더 많은 일을 하고 있다는 사실을 감추고 있었다.
2. **Gemma-SigLIP의 early-layer 피크는 음의 delta layer로 둘러싸여 있지만, 그 음들은 대부분 anchor/target trade-off이지 anti-anchor가 아니다.** Answer step에서 layer 0–4, 6–10, 12, 17은 모두 delta < 0 이고 CI 0 제외. Per-layer budget decomposition (아래 참조)을 보면 대부분이 두 이미지 사이의 anchor↔target mass swap (text는 불변)이고 능동적 anti-anchor layer가 아니다. Layer 5만이 실제로 text에서 anchor 이미지로 mass를 끌어오고 (δ_text = −0.038), layer 17만이 실제로 anchor에서 다시 text로 재분배한다. TL;DR의 "knife-edge" 비유는 layer 5의 *spike* 자체에는 맞지만 주변 layer들에는 맞지 않는다.
3. **A7 susceptibility gap은 피크 layer에 집중되고, 그 크기는 family별로 다르다.** 각 모델의 answer-step 피크에서 top-decile-susceptible vs bottom-decile-resistant delta: Qwen +0.02535 (CI 분리 선명) → LLaVA-1.5 +0.00970 → InternVL3 +0.00576 → Gemma +0.00142 (거의 0). E1 Test-3의 A7 패턴 (3/4 유지, Gemma 반전)이 layer 단위 국소화로 *per-family 크기 위계*로 날카로워진다.

## 설정

E1과 동일한 `per_step_attention.jsonl` 파일 — 4 모델 × 600 attention records (3 조건 × 200 susceptibility-stratified 질문). 각 triplet (sample_instance × {base, number, neutral})과 각 layer `l`에 대해:

```
delta_l = image_anchor_mass[l](number) − image_anchor_mass[l](neutral)
```

두 생성-step 라벨에서 평가: **answer** (decoded token이 숫자를 포함하는 첫 step — `per_step_tokens` 기반)과 **step 0** (opening brace / 첫 생성 토큰). Bootstrap 2,000 iter, 95 % CI, 모델당 유효 triplet ~200개 기준.

InternVL3의 answer step은 n=135로 떨어지는데, ~65개 triplet이 number 또는 neutral 생성 시퀀스에 digit token이 없었기 때문 (JSON 프롬프트 하에서 간헐적으로 긴 prose로 degenerate).

## 모델별 피크 요약 (answer step, overall stratum)

| 모델 | 피크 layer | depth % | 피크 delta | 95 % CI | # sig-positive layers |
|---|---:|---:|---:|---|---:|
| gemma4-e4b | **5 / 42** | 12 % | **+0.05007** | [+0.04772, +0.05229] | 19 / 42 |
| llava-1.5-7b | 16 / 32 | 52 % | +0.01883 | [+0.01441, +0.02310] | 26 / 32 |
| internvl3-8b | 14 / 28 | 52 % | +0.01931 | [+0.01348, +0.02535] | 24 / 28 |
| qwen2.5-vl-7b | 22 / 28 | 82 % | +0.01527 | [+0.00842, +0.02307] | 22 / 28 |

Gemma-SigLIP의 피크 delta는 크기상 ~2.5–3× 크고, layer fraction은 세 ViT-family 모델보다 한 자릿수 얕다. 나머지 셋은 mid-to-late stack에 클러스터링.

## 모델별 피크 요약 (step 0, overall stratum)

| 모델 | 피크 layer | depth % | 피크 delta | 95 % CI | # sig-positive layers |
|---|---:|---:|---:|---|---:|
| gemma4-e4b | 6 / 42 | 15 % | +0.04505 | [+0.03817, +0.05177] | 39 / 42 |
| qwen2.5-vl-7b | 19 / 28 | 70 % | +0.00975 | [+0.00726, +0.01225] | 22 / 28 |
| llava-1.5-7b | 26 / 32 | 84 % | +0.01333 | [+0.01137, +0.01551] | 26 / 32 |
| internvl3-8b | 3 / 28 | 11 % | +0.00760 | [+0.00665, +0.00858] | 26 / 28 |

Gemma의 step-0 피크는 answer-step 피크와 사실상 같은 위치 (layer 5 vs. 6)이며, 신호는 39/42 layer에 걸쳐 brosadcast (거의 모두 유의) — E1이 주장한 "Gemma는 prompt 통합 시점에 anchor를 인코딩하고 answer time에 같은 인코딩을 *재사용*한다"와 일치.

InternVL3은 두 step 간에 피크 layer가 바뀐다 (step-0 피크 layer 3, answer 피크 layer 14) — 초기 prompt-read은 shallow, answer-decision layer는 mid-stack. 이 패턴은 InternVL3을 "초기 등록 + mid-stack 결정"의 2단계 reader처럼 보이게 한다.

## 세 가지 encoder-family 원형

E1 Test 1 (전체 delta), E1 Test 3 (A7 stratum), 그리고 이번 layer 국소화를 종합:

| Family | 피크 layer depth | 피크 delta | 피크 A7 gap | 피크에서의 budget 출처 | Step-0 vs answer | 해석 |
|---|---|---|---|---|---|---|
| SigLIP (Gemma) | **매우 이른** (12 %) | 큼 (+0.050) | ~0 (+0.001) | **text** (−0.038) | 거의 동일 위치 | 이미지 속 텍스트를 조건 없이 이른 depth에서 등록. Anchor가 prompt text에서 mass를 가져옴 — SigLIP의 typographic-attack 계승과 일치. Susceptibility는 conditioner가 아님. |
| CLIP-ViT (LLaVA-1.5) | mid (52 %) | moderate (+0.019) | moderate (+0.010) | text (−0.029), target은 오히려 *증가* (+0.007) | answer mid / step-0 late | 두 이미지가 text에서 함께 attention을 pool. Anchor가 target과 경쟁하지 않고 "visual evidence"로 공-aggregate. |
| InternViT (InternVL3) | mid (52 %) | moderate (+0.019) | small-moderate (+0.006) | text (−0.014), target 약간 (−0.004) | step-0 early / answer mid | LLaVA처럼 text 중심이되 target도 약간 displacement. 2-stage: early 등록 + mid-stack 결정. |
| Qwen-ViT (Qwen2.5-VL) | **늦음** (82 %) | moderate (+0.015) | **큼** (+0.025, CI 분리) | **target** (−0.010) | 둘 다 늦음 co-located | Late-stack 조건부 anchor-vs-target 경쟁: answer-decision layer에서 anchor가 target 이미지를 밀어냄. |

핵심: 원래 H3축 ("ViT vs. ConvNeXt")보다 "LLM stack 어디서 anchor가 읽히는가"가 더 정보적. SigLIP이 depth (얕음)와 susceptibility-gating (없음) 두 축 모두에서 outlier. Qwen이 반대 극단 (깊음, 강한 susceptibility gating). CLIP-ViT과 InternViT은 두 축 모두에서 중간.

## 피크 layer budget decomposition: anchor는 어디서 attention을 훔치는가

각 layer의 attention mass는 1로 합해지므로, 특정 layer의 양의 `delta_anchor`는 같은 layer 내 다른 어딘가의 음의 delta로 상쇄되어야 한다. 그 상쇄 출처가 mechanistic 신호다 — anchor가 *target 이미지*, *prompt text*, 혹은 *이미 생성된 토큰*과 경쟁하는가?

각 모델의 answer-step 피크 layer에서 (number − neutral, answer step, triplet 평균, raw `sum_check ≈ 0`로 정규화 확인):

| 모델 | 피크 layer | δ anchor | δ target | δ text | δ generated |
|---|---:|---:|---:|---:|---:|
| gemma4-e4b | 5 | **+0.05007** | −0.00961 | **−0.03804** | −0.00242 |
| qwen2.5-vl-7b | 22 | +0.01527 | **−0.00972** | −0.00509 | −0.00046 |
| llava-1.5-7b | 16 | +0.01883 | +0.00702 | **−0.02940** | +0.00355 |
| internvl3-8b | 14 | +0.01931 | −0.00384 | −0.01427 | −0.00120 |

두 가지 구별되는 메커니즘:

- **Text-stealing** (Gemma, LLaVA-1.5, InternVL3). Anchor 이미지는 주로 prompt text의 mass를 가져간다. LLaVA-1.5에서는 target 이미지도 *함께* 증가 (+0.007) — 두 이미지가 함께 text에서 가져감. Anchor가 "또 하나의 visual evidence"로 등록되고 target과 공유 이미지 슬롯을 두고 경쟁하지 않는 패턴과 일치.
- **Target-stealing** (Qwen만). Qwen의 layer 22에서 anchor는 mass를 주로 target 이미지 (−0.010)에서 가져가고 text (−0.005)에서는 덜 가져간다. 스택 말기에 이르면 답 생성 회로가 "관련 이미지를 본다"에 고정된 예산을 가지고, 그 안에서 anchor 이미지가 target 이미지를 대체.

이로써 세 family 원형이 더 갈라진다:

- **Gemma (SigLIP):** early + text-stealing. Anchor 숫자가 초기 prompt-integration layer에서 prompt 상의 텍스트처럼 읽히며, pull은 text mass에서 옴 (target에서 아님).
- **Qwen (Qwen-ViT):** late + target-stealing. Anchor가 answer-decision layer에서 target 이미지를 특이적으로 밀어냄.
- **LLaVA-1.5 / InternVL3:** mid + text-stealing (대체로). 두 이미지가 text에서 attention을 pool. 두 모델의 A7 magnitude가 "중간"이고 layer 간 프로파일이 가장 연속적인 이유 설명.

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
| qwen2.5-vl-7b | 22 | +0.02794 [+0.01532, +0.04271] | +0.00259 [−0.00204, +0.00602] | **+0.02535** |
| llava-1.5-7b | 16 | +0.02369 [+0.01600, +0.03177] | +0.01398 [+0.00999, +0.01782] | +0.00970 |
| internvl3-8b | 14 | +0.02302 [+0.01273, +0.03766] | +0.01726 [+0.01064, +0.02362] | +0.00576 |
| gemma4-e4b | 5 | +0.05078 [+0.04758, +0.05417] | +0.04936 [+0.04628, +0.05260] | +0.00142 |

Qwen의 bottom-decile 행이 특히 인상적: 모델 자신의 피크 layer에서 bottom-decile triplet은 CI가 *0을 포함*하는 delta를 보인다. Resistant item에 대해서는 모델이 late-stack attention gap을 아예 발달시키지 않음. Susceptible item에 대해서는 같은 layer가 +0.028로 발화. Qwen의 단일 layer가 A7 신호의 거의 전부를 운반하고, 그 신호는 susceptible item에만 존재. 4-모델 패널에서 가장 깨끗한 "susceptibility-gated answer-step 회로"이자, 이 family에 대한 자연스러운 E4 개입 부위.

## E4 (mitigation)에 대한 함의 — 후보 site (검증 전)

다음은 관찰된 per-layer + budget 데이터에서 생성한 가설이다; 아직 ablate하거나 re-weight하지 않음. E1 writeup의 결론은 유효 ("단일 범용 mitigation은 작동하지 않는다"); per-layer 데이터는 E4의 *후보* 개입 site를 좁힌다:

- **SigLIP-Gemma:** input-side 후보. Layer 5 이전을 target으로, 해당 layer의 text→anchor pull을 차단하는 것이 구체 목표. 생성 시점의 attention re-weighting은 작동하지 않을 가능성 (피크가 이미 KV cache 안). 시도 옵션: 두 번째 이미지의 projection-feature masking, 또는 encoder / early LLM layer 경계에서 KV-cache patching.
- **Qwen-ViT:** layer 22 ± 2, answer-step only, per-item susceptibility 신호로 gate한 late-stack attention re-weighting 후보. 이 layer의 budget은 anchor-vs-text가 아니라 anchor-vs-target이므로, anchor down-weight는 mass를 text가 아니라 target 이미지로 되돌려야 함 — 원하는 결과.
- **CLIP-ViT (LLaVA-1.5) 및 InternViT (InternVL3):** layer ~14–16, answer-step의 mid-stack attention re-weighting 후보. 이들의 budget 출처가 주로 text이기 때문에 anchor down-weight는 mass를 target이 아니라 text로 되돌림 — Qwen보다는 덜 이상적이지만 여전히 시도할 가치.

패널은 이제 논문-레벨 가설과 부합한다: **cross-modal bias의 LLM stack 내 위치 *그리고 경쟁 영역*이 encoder-family-specific이며, architecture-aware mitigation 설계가 필요하다.** 이 가설이 causal E4 개입에서 살아남는지는 TBD — 이건 관찰된 mechanistic evidence이지 검증된 fix가 아니다.

## Caveat

- 모델당 step당 `n=200`. 피크에서는 per-layer CI가 타이트하지만 tail은 더 noisy. 더 큰 n이 layer-선택 신뢰를 더 조일 것 (예: Qwen 피크가 22인지 20-23인지).
- "피크 layer"는 `argmax(delta_mean)`으로 선택. 피크의 CI가 타이트해서 안정적이지만, Gemma에서 5 vs. 6 또는 LLaVA-1.5에서 16 vs. 17의 정확한 선택은 CI overlap 내.
- 여전히 ConvNeXt / FastViT 데이터 없음 — ConvLLaVA와 FastVLM은 `inputs_embeds` 경로를 사용하며 추출 스크립트 확장 필요. E1-preliminary-results.md의 open question 1번은 여전히 열림; 그것이 다음 E1 단계.
- Layer-wise mass는 head 평균. Head-수준 sparsity (Gemma layer 5의 단일 head가 신호를 운반) 은 여기서 보이지 않으며 회로 주장을 더 강화할 것. E1b 범위 밖.

## Roadmap 업데이트

- §3 status: E1b layer analysis done (이 파일). Cross-family 패턴이 이제 3-way: SigLIP early / CLIP-Intern mid / Qwen late.
- §6 Tier 1 E1: E1-preliminary-results.md open-question의 "per-layer localisation" 항목 체크. 여전히 열림: ConvNeXt/FastViT 확장 (다음), causal test.
- §10 changelog entry 2026-04-24.
