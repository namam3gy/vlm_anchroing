# E1c — H3은 death: ConvNeXt는 adoption 수준뿐 아니라 per-layer 수준에서도 CLIP-ViT를 그대로 replicate

**상태:** 6-모델 E1b 패널(이번 라운드)에서 새로 나온 insight. 원시 데이터: `outputs/attention_analysis/{convllava-7b,llava-1.5-7b}/<run>/per_step_attention.jsonl`. 집계 테이블: `outputs/attention_analysis/_per_layer/per_layer_deltas.csv`. 6-모델 전체 컨텍스트: `docs/experiments/E1b-per-layer-localisation.md`.

## 가설과 그 falsifier

**H3** (`references/roadmap.md` §2): "Vision-encoder family가 susceptibility를 modulate. ConvNeXt/encoder-free는 CLIP/SigLIP-ViT보다 *덜* susceptible해야 함 (typographic-attack 상속)."

**Falsifier** (H3에 pre-stated): "ConvLLaVA / EVE / DINO-VLM이 CLIP-ViT VLM과 통계적으로 equivalent direction-follow gap을 보이면 H3 fail."

E2 pilot (1,125-sample)은 이미 ConvLLaVA adoption=0.156 vs LLaVA-1.5 0.181을 보고 — 통계적으로 CLIP/SigLIP cluster 안. 그건 *행동* falsification이었다. H3의 부분적 rescue 가능성: 행동이 동일하더라도 메커니즘은 다를 수 있다. **6-모델 E1b run이 그 rescue를 끝낸다.**

## 데이터

Answer-step 피크 layer에서 number − neutral anchor-mass delta, bootstrap 95 % CI, 같은 n=200 susceptibility-stratified 질문 세트:

| 모델 | encoder | 피크 layer | depth % | 피크 δ | 피크에서 budget source | A7 gap |
|---|---|---:|---:|---:|---|---:|
| **llava-1.5-7b** | CLIP-ViT | **16 / 32** | **52 %** | +0.0188 | text (−0.0294) | +0.0097 |
| **convllava-7b** | **ConvNeXt** | **16 / 32** | **52 %** | **+0.0224** | text (−0.0192) | +0.0126 |

- **같은 피크 layer** (32 LLM layer 중 L16).
- **같은 budget source** (둘 다 anchor mass를 주로 prompt text에서 가져옴).
- **같은 메커니즘 원형** (mid-stack text-stealing; 어느 쪽도 target-stealing 아님).
- **Magnitude 19 % 이내** (+0.0224 vs +0.0188).
- **A7 gap magnitude 30 % 이내** (+0.013 vs +0.010).

"CI overlap 수준으로 유사"가 아님 — 두 모델의 per-layer trace가 LLM stack 전체에 걸쳐 겹쳐 있음 (`fig_delta_by_layer_answer.png` 참조). ConvNeXt encoder를 CLIP-ViT로 교환해도 거의 동일한 per-layer attention 지문을 낸다.

InternVL3 (InternViT)도 L14에서 δ +0.019, A7 gap +0.006으로 같은 cluster. **서로 다른 세 encoder — CLIP-ViT, InternViT, ConvNeXt — 가 단일 "mid-stack text-stealing" 원형에 landing.**

## 예상 vs 실제

H3에 pre-registered된 예상: ConvNeXt가 CLIP-ViT 대비 anchor 효과를 suppress해야 함. Published typographic-attack 문헌이 SigLIP-/CLIP-ViT-family encoder를 메커니즘으로 인용. ConvNeXt의 spatial-hierarchy inductive bias가 덜 susceptible하다고 주장됨.

Per-layer 데이터가 보여주는 것: anchor가 정확히 같은 layer에서, 같은 budget source로, 거의 같은 magnitude로 도착. *Encoder architecture* (Conv vs. ViT)는 per-layer attention signature를 visibly modulate하지 않음. 6-모델 패널에 따르면 *modulate하는 것*은 **LLM stack 내 어디서** anchor가 읽히는가 — SigLIP-Gemma는 이르게, Qwen-ViT는 늦게, mid-stack cluster는 layer 14–16 근처에서. 그 축이 peak δ, budget source, A7 gap을 예측; encoder architecture는 그렇지 않음.

## 왜 이것이 논문 narrative를 바꾸는가

Roadmap의 H3 framing ("ConvNeXt < ViT")은 encoder-architecture 차이에 기반한 mechanistic claim의 paper hook이었다. H3이 adoption과 per-layer 두 수준 모두에서 falsified된 이상, 그 hook은 사라졌다 — 애도할 필요 없음: 더 날카로운 것으로 대체됨.

6-모델 패널에서의 새 framing (`docs/experiments/E1b-per-layer-localisation.md` 참조):

- Bias의 *LLM stack 내 위치*가 encoder-family-specific이지만, **H3가 예측한 방식으로는 아님**. Conv vs. ViT가 아니라 — SigLIP (early) vs. ViT/Conv cluster (mid) vs. Qwen-ViT (late) vs. FastViT (late + 큼).
- *Post-projection layer depth* 축이 peak δ, budget source, A7 gap을 예측; encoder-architecture (ViT vs. Conv) 축은 예측하지 않음.
- Mid-stack layer 14–16에서의 단일 E4 intervention이 plausible한 "encoder-agnostic" mitigation target — 이 프로파일을 공유하는 세 encoder (CLIP-ViT, InternViT, ConvNeXt)가 architecturally 서로 다르므로, L16에서 작동하는 intervention은 encoder property가 아닌 LLM-stack property에 key off하는 것이 plausible.

마지막 점이 falsification의 실제 유용한 논문-레벨 결과: mid-stack cluster가 **정확히 세 encoder가 수렴하기 때문에** 최고-leverage E4 target이 됨. 거기서 작동하는 좋은 mitigation은 vision encoder를 swap해도 살아남을 것.

## 말하지 않는 것

- **Encoder architecture가 절대 안 중요하다.** SigLIP-Gemma는 정말로 outlier고 그 typographic-attack 상속이 데이터에 잘 맞음 (early + unconditional + text-stealing + no susceptibility gating). Encoder architecture는 분포의 *tail* (SigLIP outlier, 그리고 가능하게는 FastViT의 거대한 A7 gap)에서는 중요, 중앙에서는 아님.
- **ConvLLaVA 사용이 안전.** ConvLLaVA가 무엇을 하든 이 task에서 LLaVA-1.5가 하는 것을 거의 정확히 한다. ConvLLaVA 사용이 anchoring-reduction 측면에서 이득 없음.
- **ConvNeXt가 절대 안 도움.** ConvLLaVA만 test됨. 다른 Conv+LLM 조합 — 다른 backbone + ConvNeXt, 혹은 다른 ConvNeXt variant — 은 다른 프로파일을 보일 수 있음. 주장은 test된 architecture pair에 한정.
- **행동이 메커니즘과 매칭, 메커니즘이 attention과 매칭.** Adoption, direction-follow, per-layer attention 모두 "ConvLLaVA ≈ LLaVA-1.5" read로 수렴. 다른 test (Phase C: paraphrase robustness, closed-model subset)은 아직 안 됨.

## Experiment plan에 대한 함의

- **H3을 depth-axis framing으로 대체, 논문에서.** Mechanistic 이야기가 "post-projection LLM depth가 anchor signature가 변하는 축이다; encoder architecture는 아니다"가 됨. 6-모델 패널이 이를 자연스럽게 지지.
- **E4 mitigation prototype 우선순위 — mid-stack cluster 먼저.** 세 encoder가 프로파일 공유이므로, 하나 (예: LLaVA-1.5)에서 튜닝한 단일 intervention을 나머지 두 개 (ConvLLaVA, InternVL3)에서 portability check로 test 가능. Port되면 그것이 논문의 "architecture-agnostic E4" 주장.
- **E2의 계획된 "encoder-ablation" 서브섹션 drop.** E2는 H3 test로 설계됨; H3 dead이므로 원래 설계의 E2는 증명할 게 없음. 해당 compute를 E5 (multi-dataset) 혹은 E7 (paraphrase robustness)으로 재할당.

## Caveat

- 모델당 n=200; ConvLLaVA run과 LLaVA-1.5 run이 susceptibility strata sampling에 다른 random seed 사용했으나 같은 susceptibility CSV에서 draw. 피크 layer 일치 (둘 다 16)가 CI overlap 안이고 sampling artefact일 가능성 낮음.
- 비교는 answer-step 피크에서. *Step-0* 피크는 실제로 다름 (ConvLLaVA L31/32 vs LLaVA-1.5 L26/32), 그리고 그 차이는 `docs/experiments/E1b-per-layer-localisation.md`에 flagged됨. Prompt-integration step은 answer step과는 다른 architecture-dependence 가짐; 이 insight는 answer step에 한정.
- ConvLLaVA EagerConvLLaVARunner는 첫 통합 (이번 라운드). 정규화된 mass sanity (layer와 step마다 sum = 1) 가 모든 record에서 통과하지만, 독립 cross-check까지는 수치를 preliminary로 취급해야 함.
- H3은 encoder-free VLM (EVE, DINO-VLM)에서는 여전히 untested. 헤드라인은 ConvNeXt specifically가 도움 안 된다는 것; 더 약한 read — "encoder architecture가 *outlier* architecture에서만 distinctive 프로파일 낸다 (SigLIP early, 가능하게는 FastViT late)" — 는 여전히 survive.

## Roadmap 항목

- §2 H3: 상태가 "⚠️ Pilot이 simple form 지지 안 함"에서 "❌ Adoption과 per-layer 두 수준 모두에서 Falsified. E1b의 depth-axis framing으로 대체됨"으로 변경. 이 insight에 cross-link 추가.
- §6 Tier 2 E2 encoder-ablation: 범위 축소 — H3 test 더는 불필요. Compute 예산을 E5 또는 E7로.
- §10 changelog entry 2026-04-24.
