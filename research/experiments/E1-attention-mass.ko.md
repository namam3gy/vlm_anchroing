# E1 — Attention-mass 분석: anchor가 어디서 computation에 진입하는가?

**Status:** Plan only. 구현은 E2 pilot 뒤로 queued. Source: `RESEARCH_ROADMAP.md` §6 Tier 1 E1. *(영문 canonical: `E1-attention-mass.md`)*

## 질문

모델 출력이 anchor digit 쪽으로 shift할 때 language-model layer가 실제로 anchor-image token에 target-image token보다 더 *attend*하는가? 아니면 anchor의 영향이 다른 경로로 진입하는가 (예: LLM의 generic "second image" prior)?

이건 가능한 mechanistic 측정 중 가장 cheap하고 `research_plan.md`가 flag한 "왜 일어나는가" 리뷰어 컴플레인에 직접 답.

## 방법

Stratified 1k-sample subset (`base_correct` stratum 간 매칭 → H2 conditioning 작동):

1. 각 `(model, sample)` 페어를 `target_only`와 `target_plus_irrelevant_number` 조건 모두에서 재추론, `model.generate(...)`에 `output_attentions=True` 활성화.
2. 각 attention head, layer, decoded token에 대해 다음 영역의 attention mass 계산:
    - target-image token (image-1 patch position)
    - anchor-image token (있을 때 image-2 patch position)
    - prompt text token
    - generated text token (causal masked → upper-triangular only)
3. Layer/condition/(model, base_correct) stratum별 aggregate.

### A1과 묶이는 예측

Phase A가 bias가 *uncertain item에서의 graded pull*이라고 확립. 대응되는 attention 예측:

- 모델이 원래 **correct**한 item에서, anchor-image attention mass ≈ neutral-image attention mass (둘 다 noise distractor, 둘 다 largely ignored).
- 모델이 원래 **wrong**한 item에서, anchor-image attention mass가 neutral-image attention mass **위**에 — LLM이 uncertainty 하에 irrelevant 이미지를 evidence로 recruit.

이 예측이 holds하면 paper에 tight한 mechanistic story: "uncertainty → 두 image 모두로 visual attention 확장 → anchor digit이 answer-token decision에 encoded".

### A7과 묶이는 예측

A7가 item-level susceptibility가 모델 간 modest correlate. 따라서: high-susceptibility item (cross-model `moved_closer_rate` top-decile)이 low-susceptibility item보다 anchor-image attention mass가 systematically 높아야 함, layer-wise patching *전*에 — pure observational signal.

## 구현 outline

새 스크립트 `research/scripts/extract_attention_mass.py`:

1. `build_runner(...)` factory로 모델 1개 load.
2. `generate_number()`를 mirror하는 thin `_run_with_attention(...)` 메소드 추가, `output_attentions=True, return_dict_in_generate=True` 전달, `out.attentions` → list[layer] of tensor `[batch, heads, q_len, k_len]` unpack.
3. 각 step의 attention에서 processor의 `image_token_index` marker로 image-token span 식별 (모델별로 다름 — `<image>` → repeated visual token). Runner가 manual splice (FastVLM, ConvLLaVA)하는 곳은 adapter가 이미 알고 있는 splice point 재사용.
4. Step별 4개 bucket에 mass 계산 후 (model, sample_id, condition)당 parquet 작성.

Memory note: 1024-token context × 28 layer × 32 head의 full attention tensor는 generation당 ~50 MB. Per-bucket per-layer scalar만 저장하면 generation당 ~10 KB로 drop — 편안.

## Compute 예산

1k 샘플 × 3 condition × N 모델 × generation당 몇 초. H200 1장 4 모델 × 60 GB는 **단일 오후**, multi-day commitment 아님. E2 full run 종료 후 schedule.

## 산출물

- `research/scripts/extract_attention_mass.py`
- `outputs/attention_analysis/<model>/<run>/per_layer_mass.parquet`
- `research/insights/E1-attention-results.md`: per-stratum attention bar plot, per-layer trace, cross-susceptibility 비교.

## Mitigation E4와의 연결

E1이 anchor-image attention이 uncertain item에서 differentiator라고 보이면, 자연스러운 E4 prototype은 **conditional attention re-weighting** — 추론 시 LLM의 answer position per-token entropy가 threshold τ를 넘으면 anchor-image attention을 α < 1로 rescale. Parameter 2개, 둘 다 held-out split에서 tuneable. 가장 cheap한 E4 구현이고 E1에서 자연 연결 — 의도적으로 chain되도록 designed.
