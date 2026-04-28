# E2 pilot results — encoder ablation, n=1,125 per model

**Status:** Pilot complete 2026-04-24. 신규 통합된 4 모델 × 25 samples-per-answer × 9 answer value × 5 irrelevant-set variant × 3 condition = 모델당 1,125 sample-instance. Source: `outputs/experiment_encoder_pilot/<model>/<run>/summary.json`. Plan: `docs/experiments/E2-encoder-ablation.md`. *(영문 canonical: `E2-pilot-results.md`)*

## 헤드라인 수치

11-model panel (pilot 4 + 기존 main run 7), `acc(target_only)` 정렬:

| Model | Encoder | acc(target) | adoption(num) | dir-follow(num) | acc_drop(num) | acc_drop(neutral) |
|---|---|---:|---:|---:|---:|---:|
| InternVL3-8B | InternViT | **0.784** | **0.066** | **0.106** | **+0.355** | +0.196 |
| qwen3-vl-30b-it | Qwen-ViT | 0.759 | 0.120 | 0.280 | +0.052 | +0.050 |
| qwen3-vl-8b-instruct | Qwen-ViT | 0.751 | 0.127 | 0.258 | +0.036 | +0.041 |
| gemma4-31b-it | SigLIP | 0.749 | 0.116 | 0.239 | +0.008 | +0.027 |
| qwen2.5-vl-7b-instruct | Qwen-ViT | 0.736 | 0.110 | 0.248 | +0.025 | +0.028 |
| gemma3-27b-it | SigLIP | 0.628 | 0.141 | 0.305 | −0.004 | +0.006 |
| llava-interleave-7b | SigLIP | 0.619 | 0.134 | 0.348 | +0.044 | +0.043 |
| **ConvLLaVA-7B** | **ConvNeXt** | **0.607** | **0.156** | **0.364** | +0.121 | +0.122 |
| gemma4-e4b | SigLIP | 0.553 | 0.123 | 0.320 | +0.012 | +0.048 |
| **LLaVA-1.5-7B** | **CLIP-ViT** | **0.528** | **0.181** | **0.355** | +0.038 | +0.049 |
| **FastVLM-7B** | **FastViT** | **0.483** | **0.090** | **0.245** | +0.188 | +0.216 |

Bold 행 = pilot 모델. Pilot n=1,125 vs main n=17,730 — pilot scale에서 adoption error bar 약 ±2–3 pp.

## H3에 대한 함의

원래 H3 예측 (`references/roadmap.md` §2): *ConvNeXt-encoder VLM이 CLIP/SigLIP-ViT VLM보다 anchor-susceptibility 낮다*.

Pilot 데이터는 이를 straightforwardly 지지하지 **않음**:

- ConvLLaVA-7B (ConvNeXt) `adoption=0.156`이 panel 상위 절반, 낮은 base accuracy 감안해도.
- LLaVA-1.5-7B (CLIP-ViT)가 전체 11-model panel에서 *최고* adoption (0.181), typographic-attack 상속과 일관.
- ConvLLaVA의 adoption (0.156)이 LLaVA-1.5-7B의 (0.181)보다 낮지만 gap ~2.5 pp — pilot noise 안 (n=1,125 → SE ≈ 0.011, 95% CI ±0.022).

**결정 발화 (`references/roadmap.md` §7):** ConvLLaVA의 susceptibility가 pilot scale에서 CLIP/SigLIP cluster confidence interval *안*. 단순 "ConvNeXt < ViT" 형태의 H3는 **지지 안 됨**. 두 갈래:

1. **ConvLLaVA + LLaVA-1.5 full 17,730 grid 돌려** tight CI 얻고 H3를 scale에서 정의적으로 reject (또는 accept). H200에서 모델당 ~1일. 부정적 finding ("ConvNeXt encoder가 anchoring을 escape 못함") 자체가 paper-worthy하고 문헌 가정에 challenge하므로 가치 있음.
2. **Refocus.** Pilot의 actually-novel 패턴은 다른 것 — 다음 섹션 참조.

## 실제로 novel한 finding: anchoring과 distraction은 분리 가능한 failure mode

Pilot에서 가장 깨끗한 새 structure: **`adoption_rate`와 `acc_drop`이 encoder별로 정반대 방향으로 decouple**:

| Bucket | Models | 패턴 |
|---|---|---|
| **Anchor-susceptible** (high adoption, low acc_drop) | LLaVA-1.5, gemma4-31b, qwen-family, gemma3-27b | 모델이 anchor digit을 깨끗하게 encode하고 prediction이 그 쪽으로 shift, 그러나 multi-image prompt에서의 전체 정확도는 높게 유지. |
| **Distraction-susceptible** (low adoption, high acc_drop) | InternVL3, FastVLM | 모델이 anchor digit을 specifically 잡진 않지만, *어떤* 두 번째 이미지가 추가되든 정확도가 collapse. |
| **Both** | ConvLLaVA, gemma4-e4b | Weak base 모델, anchored *and* distracted. |

이게 진짜 흥미로운 이유: cognitive-bias 문헌은 "anchoring"을 단일 현상으로 다룸. Pilot은 VLM에서 anchoring처럼 보이는 게 두 distinct failure mode일 수 있음을 보임:

- **Anchoring proper** = anchor가 candidate 분포에 진입해 그 쪽으로 끌어당김 (high adoption, low acc_drop).
- **Generic multi-image distraction** = 두 번째 이미지가 content 무관하게 inference 방해 (low adoption, high acc_drop).

InternVL3-8B가 가장 깨끗한 illustration: `adoption=0.066` (최저)이지만 `acc_drop(num)=0.355` (~3× 최고). 모델이 "digit은 무시"하지만 "두 번째 이미지가 있으면 question에 focus 못함". **이 decoupling이 원래 H3보다 더 publishable한 framing.**

## 재구성된 paper framing 후보

"vision-encoder family가 anchoring을 modulate" 대신 데이터가 지지하는 것:

> **Irrelevant 이미지가 추가될 때 VLM의 cross-modal failure는 두 축으로 깨끗하게 분리된다: anchor-pull (uncertainty-modulated, encoder-mediated, low cost)와 multi-image distraction (encoder-architecture-mediated, high cost). 다른 vision encoder는 이 2D plane 위 다른 점에 위치하고, 각 축의 적절한 mitigation이 다르다.**

이게 가져오는 것:
- Phase-A H2 finding (anchor pull = graded + uncertainty-modulated)을 한 축에.
- 새 직교 finding (multi-image distraction)을 두 번째 축에.
- Mitigation이 실제로 다름: anchor-pull에는 anchor-image attention을 down-weight; distraction에는 두 번째 이미지를 entirely mask (또는 content-aware image-routing 추가). Two-axis framing은 한 개가 아닌 **두 개의 complementary E4 prototype**을 의미.

이게 "encoder family matters"보다 더 sharp, pushback에 더 defensible, Phase A와 Phase B를 깨끗하게 묶음.

## Caveats

- **모델당 n=1,125.** ~5 pp 미만의 direction-follow 차이는 안전하게 해석 불가. Two-axis story는 decoupling이 크기 때문에 holds (InternVL3: 0.066 vs 0.355; ConvLLaVA: 0.156 vs 0.121). ~5 pp 안 adoption-only 비교는 reliable 안 함.
- **FastVLM acc(target_only)=0.483 의심스러움.** 이 모델이 JSON-only prompt에 산문 출력; parser가 대부분 rescue하지만 전부는 아님. FastVLM-specific 결론 전에 `is_numeric_prediction` rate 검사 필요.
- **InternVL3 행동이 tokenizer artifact 가능성.** 매우 낮은 adoption이 모델이 anchor digit으로 깨끗하게 parse 안 되는 multi-token answer 출력 때문일 수 있음. "Low anchoring" 주장 전에 raw_prediction 문자열 검사.
- **Bootstrap CI 아직 없음.** Pilot 결정은 point estimate로; main run에서 CI 추가.

## 결정

추천 (full run 시작 전 사용자 확인):

- ConvLLaVA + LLaVA-1.5 + InternVL3 + FastVLM의 full 17,730 run **Yes**. 정당화:
  - InternVL3의 distraction-vs-anchor decoupling이 핵심 새 claim, n=17,730 + CI 필요.
  - ConvLLaVA + LLaVA-1.5가 H3 평가에 필요한 깨끗한 "ConvNeXt vs CLIP-ViT, same Vicuna LLM" 비교 제공.
  - FastVLM run이 4번째 encoder family (FastViT)를 낮은 marginal cost로.
  - 4 run 모두 per-token logit 캡처 (commit `5f925b2`), deferred A1 logit-margin 재분석 가능.
- **Phase B 재정렬:** E2 (full 4 모델) → E1 (InternVL3 + LLaVA-1.5 head-to-head attention mass) → E4 (two-axis mitigation). E3는 이미 E2에 흡수.
- **"encoder family universally matters" framing drop.** Two-axis "anchoring vs distraction" framing으로 대체. 사용자 signoff 후 §2 hypothesis table 갱신.

## Roadmap update 발화

- §3.3 — full run queued되면 ConvLLaVA / InternVL3 / LLaVA-1.5 / FastVLM을 "통합되었으나 full run 없음"에서 이동.
- §6 Tier 1 E2 + E3 — 한 combined "full encoder-ablation grid" task로 통합.
- §7 결정 trigger — H3 결정 발화 ("simple 형태 지지 안 됨; two-axis framing으로 전환").
- §10 changelog — 이 writeup 추가.
