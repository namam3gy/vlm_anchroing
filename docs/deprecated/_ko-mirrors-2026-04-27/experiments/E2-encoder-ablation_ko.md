# E2 — Vision-encoder ablation: ConvLLaVA vs CLIP-ViT VLMs

**Status:** Pilot launched 2026-04-24 (`configs/experiment_encoder_pilot.yaml`, 4 모델 × 25 samples-per-answer = ~1,100 sample-instance/모델). Full run은 pilot 성공에 blocked. Source: `references/roadmap.md` §6 Tier 1 E2 + H3. *(영문 canonical: `E2-encoder-ablation.md`)*

## 검증할 가설

`references/roadmap.md` §2의 H3: vision-encoder family가 anchoring susceptibility를 modulate. CLIP/SigLIP-ViT VLM은 documented typographic-attack 약점 (arXiv:2508.20570)을 상속 — pixel 안 텍스트가 concept neuron 활성화. ConvNeXt-encoder VLM (그리고 encoder-free 모델)은 같은 학습 신호가 없어 cross-modal anchoring effect가 더 작아야 함.

**Falsifier:** 매칭된 compute scale에서 ConvLLaVA-7B의 `moved_closer_rate`와 `mean_anchor_pull`이 CLIP/SigLIP-ViT VLM과 통계적으로 동등 (둘 중 하나 metric에 95 % bootstrap CI 기준 ±10 % relative)이면 H3 fail, encoder-architecture story가 paper에서 drop.

## 실험 설계

### Pilot (현재 run)

- 모델: `convllava-7b` (ConvNeXt encoder), `llava-1.5-7b` (CLIP-ViT, vanilla baseline), `internvl3-8b` (InternViT), `fastvlm-7b` (Apple FastViT — 다른 ViT lineage). `experiment.yaml`과 같은 prompt + decoding.
- 25 samples per answer × 9 answer value × 5 irrelevant set × 3 condition ≈ 모델당 3,375 generation.
- 목표: (a) sanity-check pipeline이 신규 통합된 각 runner에서 end-to-end 작동; (b) cross-encoder direction-follow ordering의 fast read. Pilot은 bootstrap CI 불필요 — feasibility / 방향 read.

### Full run (pilot signoff 후)

- 같은 4 모델, `samples_per_answer=400` → 모델당 17,730 sample-instance, 기존 7 main run과 매칭.
- 기존 7-model panel에 H3 evidence 추가 (총 11 모델, 4–5 vision-encoder family span).
- 모든 record에 `token_info` (logit) 캡처 — separate run 없이 deferred A1 logit-margin 재분석 가능. *(`docs/insights/A1-asymmetric-on-wrong.md` "Concrete next steps" 참조.)*

### 비교 metric

Headline metric: `base_correct == False`로 stratified한 `moved_closer_rate` (Phase A가 가장 깨끗한 signal 보인 H2 "uncertain item" subset). Secondary: `adoption_rate`, `mean_anchor_pull`. 모두 bootstrap 95 % CI.

H3 side-by-side test:

```
For each model in {convllava, llava-1.5, internvl3, fastvlm} ∪ existing 7:
    moved_closer_rate(wrong) with 95% CI 계산
Encoder family로 group:
    ConvNeXt:  {convllava-7b}
    CLIP-ViT:  {llava-1.5-7b, llava-next-interleave-7b}
    SigLIP:    {gemma3-27b, gemma4-31b, gemma4-e4b}
    Qwen-ViT:  {qwen2.5-vl-7b, qwen3-vl-8b, qwen3-vl-30b}
    InternViT: {internvl3-8b}
    FastViT:   {fastvlm-7b}
Test: ConvNeXt mean이 CLIP-ViT, SigLIP mean보다 유의하게 낮은가.
```

Lineup이 unbalanced (ConvNeXt 1 모델만)이라 가장 강한 가능 claim은 "ConvLLaVA의 susceptibility가 CLIP/SigLIP 클러스터 95 % CI *밖*에 위치" — 모집단 claim 아닌 directional claim. Precedent (Weng et al. EMNLP 2024 Main이 그들의 causal-mediation ablation에 비슷한 rhetorical structure 사용)와 일관.

## 위험 / validity 위협

- **ConvLLaVA-7B는 SigLIP-Gemma 모델 대비 undertrained.** 낮은 baseline accuracy (target_only) → "wrong" item이 더 많음 → Phase A asymmetry metric을 mechanically inflate. 비교 시 `base_acc` 통제 필요.
- **Encoder family vs LLM family confound.** ConvLLaVA = Vicuna LLM, LLaVA-1.5 = 다른 Vicuna, Qwen3-VL = Qwen3. 차이를 encoder만에 깨끗하게 attribute 불가. Mitigation: writeup에서 "encoder family + LLM family 공동 변화"로 결과 제시, encoder-only effect overclaiming 회피. Same-LLM Qwen3-VL pair (Qwen3-VL-30B vs 8B)이 한 "LLM-only varies" 비교, ConvLLaVA vs LLaVA-1.5가 한 "encoder-only varies" 비교 (둘 다 Vicuna-7B).
- **FastVLM이 JSON-only prompt에도 산문 출력** (deleted integration note에 따라 — 모델 특성, pipeline bug 아님). Parser rescue rate < 100 %. `is_numeric_prediction` rate 보고, rate < 90 %면 어떤 metric 계산에서도 FastVLM 제외.
- **OpenGVLab InternVL3-8B가 `-hf` repo suffix 필요** — 이미 모델 dispatch에 wired됨.

## 성공 기준

- Pilot이 4 모델 모두 ≥ 90 % numeric-parse rate, pipeline crash 없이 완료.
- 둘 중 하나: (a) ConvLLaVA `moved_closer_rate(wrong)`가 CLIP-ViT 클러스터 명확히 아래 (H3 supported, full run high priority schedule) 또는 (b) 그렇지 않음 (H3 falls — encoder-ablation framing drop, attention mass / mitigation으로 refocus).
- 어느 쪽이든: 1-page result note 작성하고 `references/roadmap.md` §3 status table 갱신.

## 결과

*(pilot 종료 후 채움 — date는 roadmap §10 changelog 참조)*
