# E1 — 4개 encoder family에 걸친 attention-mass 결과 (각 n=200)

**Status:** 4 모델 × 200 susceptibility-stratified question × 3 condition = 2,400 attention record. Main claim들이 cross-replication으로 결정됨. Plan: `research/experiments/E1-attention-mass.md`. 원본 데이터: `outputs/attention_analysis/{gemma4-e4b,qwen2.5-vl-7b-instruct,llava-1.5-7b,internvl3-8b}/<run>/per_step_attention.jsonl`. 분석: `research/scripts/analyze_attention_mass.py`. *(영문 canonical: `E1-preliminary-results.md`)*

## TL;DR — 4개 모델에서 각각 replicate되거나 깨끗하게 split되는 세 가지 발견

1. **Robust universal.** Anchor-image attention mass > neutral-image mass. Answer-digit step에서 mean delta가 4개 encoder family 걸쳐 +0.004 ~ +0.007 범위; 모든 모델에서 95 % CI가 0 제외. VLM이 neutral image보다 digit image를 일관되게 *noticed*.
2. **Robust null.** H2 mechanism 예측 — "모델이 원래 틀린 item에서 anchor에 더 attend" — **4/4 모델에서 fail**. `delta(wrong)` ≈ `delta(correct)` 어디서나, CI overlap. Phase A의 graded behavioural pull modulation이 mean-attention-mass level에서 반영 안 됨. Mitigation 설계에 직접 함의 있는 clean, publishable null.
3. **Partially replicated A7.** "Universally susceptible item이 resistant보다 더 많은 answer-step anchor attention 받음"이 3/4 모델 (Qwen, LLaVA-1.5, InternVL3 — top/bottom 비율 1.6 – 4×)에서 깨끗이 holds. Gemma만 reverses (bottom > top). SigLIP-family outlier가 signal을 answer generation이 아닌 step 0 (prompt processing)에 집중 — 같이 묶여 있는 두 architecture-specific 특이사항. SigLIP의 documented typographic-attack 상속이 다른 encoder와 달리 early kick in함을 시사.

## Setup

- **모델:** Phase-A susceptibility 축을 span하는 4 encoder family (wrong/correct moved-closer gap 큰 것 → 작은 것): gemma4-e4b (SigLIP + Gemma-4-E4B, 4B, gap +19.6 pp) → llava-1.5-7b (CLIP-ViT + Vicuna-7B, pilot에서만 implicit) → internvl3-8b (InternViT + InternLM3, main run 없음) → qwen2.5-vl-7b-instruct (Qwen-ViT + Qwen2.5-7B, gap +6.9 pp).
- **샘플:** 4 모델 모두 같은 200 question (top-decile susceptible 100 + bottom-decile resistant 100, `_data/susceptibility_strata.csv`). Question당 irrelevant-set variant 1개. 각 3 condition → 모델당 600 record.
- **Attention 추출:** `attn_implementation="eager"` (sdpa가 weight silently drop), `model.generate()` 시 `output_attentions=True`. 마지막 query position에서 4 region으로의 per-layer attention mass: target-image span, anchor-image span, prompt text, generated-so-far. Head 평균 → layer 평균; generated step마다 capture.
- **Join fix:** base_correct를 attention run의 자체 `target_only` decoded로 도출 — attention run의 set00 anchor가 predictions run의 set00과 다름 (variants_per_sample RNG divergence).

## Headline 수치 (4 모델, 각 n=200)

모든 delta는 `attention(number) − attention(neutral)` mean anchor-image attention mass (head 평균 → layer 평균). Bootstrap 2,000 iter, 95 % CI.

### Test 1 — overall

| Model | step | mean delta | 95 % CI | share > 0 |
|---|---|---:|---|---:|
| gemma4-e4b | step 0 | **+0.01169** | [+0.00988, +0.01343] | 0.82 |
| gemma4-e4b | answer | +0.00434 | [+0.00080, +0.00797] | 0.54 |
| qwen2.5-vl-7b | step 0 | +0.00196 | [+0.00136, +0.00253] | 0.70 |
| qwen2.5-vl-7b | answer | +0.00525 | [+0.00319, +0.00741] | 0.76 |
| llava-1.5-7b | step 0 | +0.00603 | [+0.00513, +0.00697] | 0.81 |
| llava-1.5-7b | answer | +0.00559 | [+0.00402, +0.00721] | 0.75 |
| internvl3-8b | step 0 | +0.00199 | [+0.00171, +0.00226] | 0.85 |
| internvl3-8b | answer | **+0.00670** | [+0.00483, +0.00870] | **0.95** |

Answer-step mean이 4 모델 모두 tight band (+0.004 ~ +0.007); 모든 cell에서 CI가 0 제외. InternVL3이 가장 높은 `share>0` (0.95 — answer step에서 135 triplet 중 128개 positive), pilot 프로파일 (low adoption이지만 high acc_drop → 두 번째 이미지가 item별 outsized effect)과 일관. Step-0 signal이 모델별로 다름: gemma (+0.012) 우세, llava-1.5 (+0.006), qwen/internvl3 (+0.002) minimal.

### Test 2 — base_correct stratified (H2 mechanistic 예측, answer step)

| Model | wrong n | wrong mean | wrong CI | correct n | correct mean | correct CI |
|---|---:|---:|---|---:|---:|---|
| gemma4-e4b | 100 | +0.00566 | [-0.00037, +0.01188] | 100 | +0.00301 | [-0.00079, +0.00694] |
| qwen2.5-vl-7b | 84 | +0.00494 | [+0.00192, +0.00853] | 116 | +0.00547 | [+0.00297, +0.00805] |
| llava-1.5-7b | 113 | +0.00515 | [+0.00308, +0.00734] | 87 | +0.00616 | [+0.00382, +0.00855] |
| internvl3-8b | 42 | +0.00743 | [+0.00429, +0.01178] | 93 | +0.00638 | [+0.00399, +0.00830] |

**H2 null, 4/4 replicates.** 모든 모델에서 `delta(wrong)`과 `delta(correct)`이 CI overlap. LLaVA-1.5와 Qwen에서는 nominal 방향이 뒤집힘 (correct > wrong). base_correct로 operationalise한 uncertainty가 4 encoder family 어디에서도 mean anchor-image attention을 modulate 안 함. "high-uncertainty case를 타겟"으로 하는 E4 mitigation에 대한 serious constraint.

### Test 3 — susceptibility_stratum stratified (A7 예측, answer step)

| Model | bottom n | bottom mean | bottom CI | top n | top mean | top CI |
|---|---:|---:|---|---:|---:|---|
| gemma4-e4b | 100 | **+0.00647** | [+0.00197, +0.01156] | 100 | +0.00220 | [-0.00332, +0.00762] |
| qwen2.5-vl-7b | 100 | +0.00191 | [-0.00021, +0.00352] | 100 | **+0.00859** | [+0.00526, +0.01238] |
| llava-1.5-7b | 100 | +0.00422 | [+0.00252, +0.00589] | 100 | **+0.00696** | [+0.00434, +0.00969] |
| internvl3-8b | 87 | +0.00563 | [+0.00328, +0.00734] | 48 | **+0.00865** | [+0.00545, +0.01289] |

**A7가 3/4 모델에서 holds, Gemma만 inverts.** Qwen, LLaVA-1.5, InternVL3에서는 top-decile susceptible question이 bottom-decile resistant보다 answer step에서 더 많은 anchor attention — 비율 1.6× ~ 4×, 세 모델 모두 CI separation 깨끗 (또는 거의). gemma4-e4b만 prediction flip. Test 1의 architecture ordering과 합치면 — Gemma step-0 signal이 LLaVA-1.5의 ~2배, Qwen/InternVL3의 ~6배 — 가장 깔끔한 읽기: **SigLIP-backed VLM (Gemma)은 anchor-encoding을 computation 초반에, 그리고 downstream item susceptibility에 conditioning 안 된 방식으로 수행**, 반면 ViT-backed VLM은 answer-generation step에 (또는 그 근처에) A7-예측 susceptibility ordering과 conform하는 방식으로 encode.

### Test 4 — combined base_correct × stratum (answer step)

`correct × top_decile_susceptible` cell이 3/4 모델에서 가장 큰 positive cell — 모델이 anchor를 "보고도" 답은 맞춤. Naive 예측 (`wrong × top_decile = strongest`)의 역이고, H2 null과 일관: attention이 anchor *notice*를 capture하지 anchor *capture*를 하지 않음. Gemma만 또 invert; 가장 강한 cell이 `wrong × bottom_decile_resistant` (n=23, paradoxical).

| Model | strongest cell | n | mean | 95 % CI |
|---|---|---:|---:|---|
| gemma4-e4b | wrong × bottom_decile | 23 | +0.01760 | [+0.00296, +0.03392] |
| qwen2.5-vl-7b | correct × top_decile | 32 | +0.01442 | [+0.00862, +0.02100] |
| llava-1.5-7b | correct × top_decile | 11 | +0.01998 | [+0.01073, +0.03001] |
| internvl3-8b | correct × top_decile | 15 | +0.00987 | [+0.00450, +0.01668] |

## 이 데이터가 말하는 것과 말하지 않는 것

**말하는 것 (4/4 또는 3/4 replicate):**
- Anchor image가 neutral image보다 더 많은 attention (4/4, clean CI).
- Uncertainty가 4개 architecture 어디에서도 mean anchor attention을 modulate 안 함. Attention-level 예측으로서 H2 robustly falsified.
- ViT-backed VLM (3/4)에서 susceptible item이 resistant보다 answer-step에서 더 많은 anchor attention. Attention mechanism이 예상 방향으로 행동적 susceptibility를 track — 단, answer-generation step에서만, prompt integration 때는 아님.
- SigLIP-Gemma가 두 축 모두에서 outlier: step-0-heavy signal, inverted A7. SigLIP typographic-attack 문헌이 obvious explanatory hook (in-image text의 early, unconditional encoding)을 제공하지만 이 해석은 confirming evidence 필요 (per-layer trace, logit lens).

**말하지 않는 것:**
- Anchor attention이 anchor pull을 *일으킨다*. Mean attention mass는 correlational measure; causal mediation (anchor token ablate → output shift 측정) 안 됨.
- ConvLLaVA (ConvNeXt)나 FastVLM (FastViT)으로 generalise. 둘 다 inputs_embeds path 필요해서 extraction script 확장 필요.
- Prompt 재작성이나 decoding 설정 간에 stable. 이 run에서 둘 다 fixed.
- 4 모델 A7가 다른 susceptibility 정의에서도 같은 크기로 holds. `moved_closer_rate` cross-model susceptibility 사용; 다른 per-model susceptibility는 다른 strata 줌.

## E4 (mitigation)에 대한 함의

H2-conditioned "uncertain할 때 downweight" design은 **4 architecture 모두에서 dead**. 모델 confidence에 key off하는 mitigation은 안 통함 — attention gap이 4/4에서 base_correct에 flat.

4-모델 데이터가 *지지*하는 것:

- **ViT-backed 모델용** (Qwen-ViT, CLIP-ViT, InternViT): intervention target 명확. Anchor-image attention을 answer-digit generation step에서, A7의 top-decile item susceptibility로 condition해서 rescale. A7-conforming 패턴이 tested VLM 3/4에 holds; 단일 intervention을 한 번 tune해서 family 전반에 작동 plausible.
- **SigLIP-backed 모델용** (Gemma): intervention target이 generation 아닌 *prompt-encoding*. Signal이 step-0-heavy이고 susceptibility 관계가 반전되므로, 이 family의 자연스러운 E4는 generation time의 attention-weight rescaling 아닌 input-side intervention (예: 두 번째 이미지 KV cache를 input layer에서 patch, 또는 projection feature를 zero).

Per-family 결과가 이제 E4 design constraint를 직접 명시할 만큼 강함: **단일 universal mitigation은 안 통할 것; paper는 두 intervention을 side-by-side 평가하거나 ViT-family intervention만 선택하고 SigLIP을 E4 평가에서 명시적으로 제외하며 이유를 설명할 것.**

E4 코드 작성 전 남은 blocker: per-family 패턴이 ConvLLaVA (ConvNeXt)와 FastVLM (FastViT)으로 확장되는지 검증. 둘 다 inputs_embeds path 사용해서 extraction script 확장 필요. 완료되면 attention 측면에서 6-model picture 확보하고 E4 design commit 가능.

## E4 전 답해야 할 open question

1. **Per-family 패턴이 ConvLLaVA / FastVLM에서 holds?** 둘 다 inputs_embeds — extraction script를 그 path 다루도록 확장 필요, 그 후 모델당 ~10분.
2. **Attention gap이 특정 layer에 집중?** Per-layer data 저장됨; focused 분석 pending. Qwen answer-step signal이 layer 10–20 vs 20–28에 살면 E4 intervention 좁아짐.
3. **Per-step attention profile이 `answer−1` (digit 나오기 직전 step)에서 spike?** 그렇다면 `answer−1` intervention이 `answer`보다 더 효과적.
4. **Causal test.** 식별된 layer/step에서 anchor-image token ablate 후 output shift 측정. 실제 mechanism claim이고 attention correlation을 causal evidence로 전환.
5. **Combined stratification에 더 큰 n.** Test 4 cell이 n=11–32. 400-question run (`--top-decile-n 200 --bottom-decile-n 200`)이 `correct × top_decile` cell의 CI 좁힘.

## Roadmap entry

- §6 Tier 1 E1: ☐ → ✅ primary claim 4 encoder family 걸쳐 settled. 추가 작업 (layer localisation, ConvNeXt/FastViT 확장, causal test)은 E4와 paper write-up scope, E1 자체 아님.
- H6 (anchoring vs distraction two-axis)이 attention data와 *consistent*하지만 거기서 직접 도출되진 않음 — cleaner paper story는 "anchor notice (attention)은 robust; anchor pull (행동)은 encoder-family-modulated; uncertainty는 pull을 modulate하지만 attention은 아님". 이게 H2 + H6 + E1가 joint하게 지지하는 3-claim structure.
