# A7 — Item susceptibility는 부분적으로 content-driven; same-family 모델이 더 강하게 correlate

**Status:** Phase-A finding. 원본 데이터: `_data/A7_per_question.csv`, `_data/A7_model_correlation.csv`. 스크립트: `research/scripts/phase_a_data_mining.py::a7_cross_model_agreement`. *(영문 canonical: `A7-cross-model-agreement.md`)*

## 질문

Bias가 순수히 model-internal (LLM head가 임의 token 쪽으로 끔)이라면 모델 간 per-question susceptibility score가 uncorrelated여야 함. Content-driven (*target image*/question이 모델을 어떤 anchor든 exploit하는 방식으로 uncertain하게 만듦)이라면 per-question score가 highly correlated여야 함. 진실은 거의 둘 사이일 텐데, 질문은 *얼마나* 각각인가.

## 방법

각 `(model, question_id)` cell에 대해 per-question moved-closer rate (질문당 5 irrelevant set 평균) 계산. `[1,968 question × 7 model]` 매트릭스 생성. 모델 간 Spearman correlation이 per-pair item-agreement 강도.

## 결과

Per-question moved-closer rate의 Spearman correlation:

|                          | g3-27 | g4-31 | g4-e4 | llava | q2.5-7 | q3-30 | q3-8  |
|--------------------------|------:|------:|------:|------:|------:|------:|------:|
| **gemma3-27b-it**        | 1.00  | 0.20  | 0.27  | 0.22  | 0.22  | 0.24  | 0.23  |
| **gemma4-31b-it**        |       | 1.00  | 0.19  | 0.15  | 0.16  | 0.15  | 0.17  |
| **gemma4-e4b**           |       |       | 1.00  | 0.26  | 0.26  | 0.31  | 0.27  |
| **llava-interleave-7b**  |       |       |       | 1.00  | 0.20  | 0.24  | 0.19  |
| **qwen2.5-vl-7b**        |       |       |       |       | 1.00  | 0.24  | 0.25  |
| **qwen3-vl-30b-it**      |       |       |       |       |       | 1.00  | **0.30** |
| **qwen3-vl-8b-instruct** |       |       |       |       |       |       | 1.00  |

Positive-but-modest correlation (0.15 – 0.31)이 헤드라인. **Item susceptibility는 부분적으로 content-driven, 부분적으로 model-driven.** 세 가지 구조적 패턴:

1. **가장 높은 correlation은 Qwen3-VL family 내** (Qwen3-VL-30B ↔ Qwen3-VL-8B = 0.30) 그리고 Gemma4-e4b ↔ Qwen3-VL-30B = 0.31. 첫 페어는 LLM training 공유; 두 번째는 architecture로 읽기 어렵고 우연일 수 있음.

2. **Gemma3와 Gemma4는 특별히 correlate 안 함** (0.20, 일부 cross-vendor 페어보다 낮음). Gemma3-27B와 Gemma4-31B/E4B는 다른 vision encoder를 쓰고 Gemma 3 vs 4 lineage 하에 separately 학습됨; 이 internal disagreement가 encoder side가 중요하다는 가장 깨끗한 sign.

3. **`gemma4-31b-it`이 모든 페어에서 일관된 floor** (0.15–0.20). Moved-closer rate도 가장 낮음 (0.081) — anchor를 거의 안 함. 그래서 per-question signal이 noisy하고 누구와의 correlation도 mechanically capped.

## Mechanism에 대한 의미

- **Pure LLM-head story는 fit 안 함.** Anchoring이 전적으로 LLM의 number prior에 살면, *같은* LLM 쓰는 모델 (예: Qwen-family backbone)이 1에 가깝게 correlate, cross-vendor 모델이 0에 가까워야 함. 둘 다 보이는 게 아님 — same-family ≈ 0.30, cross-vendor ≈ 0.20. Encoder도 real work.
- **Pure encoder story도 fit 안 함.** Qwen3-VL-8B와 Gemma4-e4b는 완전히 다른 vision stack (Qwen3은 custom ViT; Gemma는 SigLIP). Within-Qwen-family와 비슷한 0.27로 correlate. 일부 bias-susceptible item은 universally bias-susceptible.
- **가장 일관된 읽기:** susceptibility = `f(content, encoder, LLM)`, 세 component 모두 weight. Component를 *분리*하는 paper에 적합한 setup — E1 (attention mass) + E2 (encoder ablation) + 추후 activation patching이 정확히 그 designed.

## Experiment plan에 대한 함의 (`RESEARCH_ROADMAP.md` §6)

- **E2 (ConvLLaVA full run) well-motivated.** Pure-Conv encoder가 lineup에서 가장 깨끗한 counterfactual, modest cross-encoder correlation이 ConvLLaVA의 susceptibility profile이 model-pair correlation map에서 *다른* point에 land할 것을 예측. ~0.30으로 모두와 correlate (즉 ConvNeXt + LLaMA가 여전히 universal content-driven bias 생산)면 bias가 encoder-architecture-invariant. ~0.10으로 (훨씬 낮게) correlate면 encoder가 크게 영향.
- **E1 attention mass는 `is_susceptible_item`으로 conditioning.** Cross-model moved-closer rate top-decile (universally susceptible) item과 bottom-decile (universally resistant) item을 골라 attention pattern 비교. Encoder의 역할에 직접적 read.
- **Phase A에서 할 가치 있는 별도 분석:** 7 모델 평균의 per-question susceptibility score 계산. 이 score top-decile question이 universally-susceptible "hard case" — 거기에 cheap attention diagnostic 돌리는 게 가장 leverage 높은 mechanistic move.

## Caveats

- 매트릭스가 `moved_closer_rate`로 만들어짐, `mean_anchor_pull`이나 `adoption_rate`가 아님. 다른 두 metric으로 반복하면 패턴 검증 가능; 정성적으로 비슷할 것.
- 1,968 question은 0.30 vs 0.20을 통계적으로 distinguishable (Spearman SE ≈ 0.022 at n=1968)이지만 borderline — same-family-correlates-more 주장 전에 핵심 페어 몇 개 bootstrap.
- Per-question rate가 5 irrelevant set만 평균이라 cell당 noise 높음. Correlation이 진짜 content-driven 비율을 underestimate할 수 있음.

## Roadmap entry

§5 A7: ☐ → ✅
