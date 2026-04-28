# 논문 스타일 요약 발표 (paper_summary.pptx) — 슬라이드별 설명

**Status:** 2026-04-29 작성. PPTX 파일: `docs/figures/paper_summary.pptx`.
PDF: `docs/figures/paper_summary.pdf`. 자료 출처:
`references/project.md §0`, `docs/insights/`. 한국어로 작성.

이 문서는 발표자가 각 슬라이드 옆에 두고 읽을 수 있는 발표 노트.
슬라이드의 핵심 메시지, 강조 포인트, 예상 질문 + 답변을 정리한다.

---

## 슬라이드 1 — Title

**핵심 메시지:** 본 연구의 한 줄 표제.

> Cross-modal Numerical Anchoring in VLMs — Uncertainty-modulated graded
> pull, digit-pixel causality, and a free-lunch mitigation.

**컨텍스트:**
- EMNLP 2026 Main 트랙 타깃, ARR May 25 마감.
- 자료 시점: 2026-04-29. `references/project.md §0` 의 paper outline에
  완전히 정렬된 정리.

**발표자 노트:** "오늘 발표는 지금까지의 모든 실험을 논문 구조에 맞춰
종합한 것입니다. 22 슬라이드 동안 4가지 축 — behavioural,
causal, mechanistic, mitigation — 을 각각 한 두 슬라이드씩 다룹니다."

---

## 슬라이드 2 — 헤드라인 클레임 (§1 Introduction)

**핵심 메시지:** 본 논문의 단 하나 sharp claim.

> Cross-modal numerical anchoring in VLMs is **uncertainty-modulated
> graded pull, not categorical capture**, and concentrates on a
> **digit-pixel cue** inside the anchor image.

**3 pillar:**
1. **Setup novelty** — stand-alone rendered-number 이미지를 cross-modal
   anchor로 open numerical VQA에 적용한 직접 선행 없음. (cf. project.md
   §"novelty verdict" 표; VLMBias / typographic attacks / FigStep / Tinted
   Frames 모두 다른 setup.)
2. **Graded vs. categorical** — Phase A `A1`: 7 모델 전부 wrong-base
   direction-follow가 correct-base보다 **+6.9 ~ +19.6 pp** 큼. paired
   adopt는 2~6 % — 모델은 anchor를 *복사*하기보다 *방향*을 끌린다.
3. **Cognitive science 정합** — Mussweiler-Strack의 selective
   accessibility 모델과 일치. uncertainty 비례 anchor pull.

**예상 질문:**
- *Q: 왜 "graded"가 중요한가?*
  A: anchor가 confident estimate를 *교체*하는 게 아니라 *search direction*을 편향시키는 것이라는 인지과학 가설을 테스트하는 게 핵심. categorical은 단순 typographic attack과 구별 안 됨.

---

## 슬라이드 3 — 현실 시나리오와 위험 (§1)

**핵심 메시지:** "왜 이 연구를 해야 하는가" — motivation 5 단계.

1. 사용자가 VLM에 여러 이미지를 동시 입력하는 빈도 증가 (앨범, 멀티
   스크린샷, 문서 첨부).
2. 정답과 무관한 이미지가 답변에 영향 → 위험.
3. 인지과학 prior — 사람도 무관한 숫자에 anchored.
4. LLM 선행 (Jones-Steinhardt 2022, Echterhoff 2024) — 텍스트 anchor는
   이미 확인.
5. **VLM 공백** — 시각적 anchor (rendered digit image)가 다른 이미지의
   수치 질문에 영향을 주는지는 미답.

---

## 슬라이드 4 — Related Work novelty 매트릭스 (§2)

**핵심 메시지:** 5 차별점 + Findings → Main 4 lever.

**표:** 7 row × 3 col (extension / 가까운 선행 / novelty 판정).
- core (cross-modal numerical anchoring): genuinely novel
- "stronger on wrong cases" 비대칭: genuinely novel — strongest hook
- dataset 확장: required, not novel (방법론 위생)
- ViT vs Conv ablation: novel angle
- Confidence-modulated (§6 신규): continuous proxy = novel projection
- Encoder-blind upper-half mitigation (§7): cross-architecture locus = novel

**Findings → Main 4 lever** (project.md 비평 기반):
attention mass · causal ablation · mid-stack mitigation · confidence proxy.

**예상 질문:**
- *Q: VLMBias와 어떻게 다른가?*
  A: VLMBias는 memorized-subject label을 cue로 사용. 우리 setup은
  *numerical value without semantic label*; 측정 대상은 *open-ended numeric
  estimation*; 측정 메트릭은 regression-style shift이지 classification flip이
  아님. 인지과학 framing 명시.

---

## 슬라이드 5 — 4 조건 setup (§3 Problem Definition)

**핵심 메시지:** 모든 측정의 토대 — 4 condition 카드.

| 조건 | 두 번째 이미지 | 역할 |
|---|---|---|
| **b** target_only | 없음 | baseline (pred_b) |
| **a** + anchor | 디지트 한 글자 그림 | manipulation (pred_a) |
| **m** + anchor_masked | anchor 이미지의 디지트 픽셀만 inpaint | digit-pixel control (pred_m) |
| **d** + neutral | 디지트 없는 FLUX 자연 장면 | 2-image distraction control (pred_d) |

3 종류의 gap이 분리해주는 것:
- (a − d) → anchoring vs distraction
- (a − m) → digit-pixel 인과
- (wrong-base − correct-base) → uncertainty 변조

**예상 질문:**
- *Q: 왜 4개? 3개로 충분하지 않나?*
  A: a/d만으로는 anchor effect ≈ "두 번째 이미지가 있으면 무조건 발생"
  과 구별 안 됨. m이 추가되어야 디지트 픽셀의 *인과적* 기여를 분리 가능.
  (E5c 결과: m을 추가했을 때 digit-pixel-only contribution +6.1 pp 확인.)

---

## 슬라이드 6 — Canonical metrics M2 (§3)

**핵심 메시지:** 메트릭 정의 4개 (수식).

```
adopt_rate          = #(pa == anchor AND pb != anchor) / #(pb != anchor)
direction_follow    = #( (pb-gt)·(pa-gt) > 0 AND pa != pb ) / #(numeric pair)
exact_match         = #(pa == gt) / #(numeric pair)
anchor_effect_M     = M(anchor arm) − M(neutral arm)
```

**표기 통일:** `pred_b / pred_a / pred_m / pred_d / anchor / gt`.
Boolean flags: `pb_eq_a / pa_eq_a / gt_eq_a / pa_ne_pb / pb_eq_gt`.

**왜 이 조합인가?** M2 evidence (다음 슬라이드)에서 18 변종 비교, 4가지
known signal preservation 기준으로 가장 깔끔.

---

## 슬라이드 7 — M2 evidence (figure)

**핵심 메시지:** 18 변종 중 `A_paired__D_paired` 가 가장 깔끔.

**그림:** 6 distinct adopt variants × `mean(wrong − correct)` gap on S0/S1
cells (n=22 cells). 막대 위 라벨은 `wins/n` (wrong>correct 보존 셀 수).

| Variant | mean(wrong − correct) | wins / 22 |
|---|---:|---:|
| A_paired__D_paired (M2) | **+0.040** | **22 / 22** |
| A_paired__D_all | +0.037 | 22 / 22 |
| A_paired__D_clean | +0.019 | 21 / 22 |
| A_clean__D_all | +0.009 | 17 / 22 |
| A_clean__D_paired | +0.007 | 16 / 22 |
| **A_raw__D_all** (pre-M1) | **−0.028** | 8 / 22 |

**핵심:** pre-M1 marginal 정의는 wrong > correct를 *역전*시킴 (gt==anchor
confound). M2 paired denominator는 confound 제거 + 신호 보존.

---

## 슬라이드 8 — Datasets + Anchor Inventory (§4)

**핵심 메시지:** 4 데이터셋 × 3 inventory (anchor / masked / neutral).

| Dataset | GT range | 샘플 | anchor selection rule | Status |
|---|---|---|---|---|
| VQAv2 number | 0–8 | 17,730 | range-restricted {0..9} | ✅ 7-model main |
| TallyQA | 0–8 | ~11,200 | absolute |a−gt|≤5 | ⏳ E5b 1-model + E5e 1/3 |
| ChartQA | 1–1000+ | ~5,400 | relative |a−gt|≤max(1, 0.10·gt) | ✅ E5e 3-model |
| MathVista (int) | 1–1000 | 385 | relative_s1 (same) | ✅ γ-α 3-model (2026-04-29) |

**Anchor 생성 파이프라인:**
1. FLUX → 480×480 anchor image ({0..10000} 중 128개)
2. OpenCV Telea inpaint → masked
3. FLUX → digit-free neutral (별도 생성)

---

## 슬라이드 9 — Distance × Plausibility Window (E5b)

**핵심 메시지:** anchor가 가까울수록 adoption ↑ (단조 감소).

**그림:** E5b VQAv2 + TallyQA wrong/correct base 분할 곡선. wrong-base에서
S1 → S5 단조 감소.

VQAv2 wrong S1 = **0.131**, S5 = 0.003 (44× 차이).
TallyQA wrong S1 = **0.092**, S5 = 0.000 (∞).

**결론:** anchor가 *plausible*할 때만 effect 발생 — plausibility window.

---

## 슬라이드 10 — Cross-dataset overlay (E5b)

**핵심 메시지:** 두 데이터셋의 base accuracy가 다름에도 (acc(b) 0.62 vs 0.21)
*adopt 곡선의 모양은 동일* — image domain에 의존하지 않는 효과.

---

## 슬라이드 11 — Digit-pixel causality (E5c)

**핵심 메시지:** anchor − masked gap이 디지트 픽셀의 인과를 입증.

**그림 두 개:**
- 왼쪽: anchor vs masked adopt rate per stratum
- 오른쪽: correct vs wrong base 분할 (anchor effect는 wrong-base 집중)

VQAv2 wrong-base S1: anchor 0.129 vs masked 0.068 → **gap +6.1 pp**.

---

## 슬라이드 12 — Direction-follow는 distance-invariant (E5c)

**핵심 메시지:** sign-based direction-follow는 anchor와 masked가 거의 같음
(S5에서도). 즉 sign-based df는 *generic 2-image distraction artifact*이지
anchor digit이 직접 만드는 게 아님.

→ **adopt가 진짜 anchor effect 메트릭**.

---

## 슬라이드 13 — Per-dataset cutoff (E5d)

**핵심 메시지:** ChartQA: ✅ S1-only relative cutoff 채택.
MathVista: ⚠ C3 FAIL (모든 stratum diffuse) → γ-α로 cross-model 재검증.

ChartQA wrong-base: S1 = 0.056, S4 = 0 (clean decay).
MathVista wrong-base: S1 = 0.110, S5 = 0.051 (faIL — 노이즈 floor 미도달).

**결정:** ChartQA는 S1-only 사용; MathVista는 single-model artifact 가능성
→ γ-α로 3-model panel 검증 (다음 슬라이드).

---

## 슬라이드 14 — Cross-model × cross-dataset (E5e)

**핵심 메시지:** 4 데이터셋 × 5 모델 wrong-base S1 adopt heatmap.

**채워진 셀:**
- VQAv2 (E5b) llava-interleave-7b: 0.131
- TallyQA (E5b) llava-interleave-7b: 0.092
- ChartQA (E5e): gemma3-27b 0.073, llava 0.037, qwen2.5-vl 0.042
- MathVista (γ-α): **gemma3-27b 0.194 (다크 색)**, llava 0.059, qwen 0.022

**비어있는 셀** = 진행 중인 cross-model 확장 작업 (E5b VQAv2/TallyQA cross-model,
E5e TallyQA gemma/qwen).

---

## 슬라이드 15 — MathVista γ-α highlight

**핵심 메시지:** **gemma3-27b adopt(a, wrong-base) = 0.194 — 본 프로그램에서
가장 큰 단일 셀.** anchor − masked gap = +15.2 pp.

**두 regime 발견:**
- **Graded-tilt** (VQAv2/TallyQA/ChartQA): df 큼, adopt 작음. anchor가
  search direction을 비례 이동.
- **Categorical-replace** (MathVista): df = 0, adopt 큼. anchor가 base
  답을 통째로 교체. 다른 경우 base 답 그대로.

(Plausible 원인: gemma3-27b의 high-acc 모델이 wrong-base를 *cleanly
delineated* uncertainty subset으로 만듦. SigLIP encoder의 typographic
weakness 가능성도. E1-patch에서 후속 검증.)

---

## 슬라이드 16 — Confidence-modulated anchoring (L1, §6)

**핵심 메시지:** base prediction의 entropy ↑ → anchor pull ↑ (graded
monotone).

**측정 방법:** 각 base inference에서 답 토큰의 top-k entropy 계산 →
모델·데이터셋 cell마다 4분위로 나눔. Q1 (가장 confident) → Q4 (가장
uncertain).

**헤드라인:**
- mean(adopt Q4 − Q1) = **+0.044**
- mean(direction_follow Q4 − Q1) = **+0.128**
- 18 / 34 cells가 fully monotone (Q1<Q2<Q3<Q4) on direction-follow.

`entropy_top_k`이 best proxy (vs softmax_top1_prob, top1−top2 margin).

---

## 슬라이드 17 — Q1 vs Q4 worked example

**핵심 메시지:** 구체적 셀 (E5c VQAv2 wrong-base S1 llava-interleave-7b)에서
quartile별 anchor effect 분포.

| quartile | base 정답률 | anchor adopt | direction-follow |
|---|---:|---:|---:|
| Q1 (most confident) | 0.77 | 0.077 | 0.040 |
| Q2 | 0.50 | 0.090 | 0.080 |
| Q3 | 0.27 | 0.110 | 0.090 |
| Q4 (least confident) | **0.07** | **0.147** | **0.113** |
| **Δ (Q4 − Q1)** | −0.70 | +0.070 (+7.0 pp) | +0.074 (+7.4 pp) |

**관계:** Phase A의 wrong/correct binary는 confidence quartile의 *coarse
projection*. Q1 mean exact_match=0.77 ∼ correct, Q4=0.07 ∼ wrong. 그러나
quartile은 confidently wrong / lucky correct를 더 정밀 분리.

---

## 슬라이드 18 — Attention 4-archetype (E1 + E1b, §7)

**핵심 메시지:** 6-model panel attention dump → 4 archetype 발견.

| Archetype | Encoder | Peak L | δ | 메커니즘 |
|---|---|---|---:|---|
| SigLIP-Gemma early | SigLIP-So (gemma4-e4b) | L5/42 (12%) | +0.050 | text-stealing |
| Mid-stack cluster | CLIP-ViT (llava-1.5) | L16/32 | +0.019 | text-stealing |
| Mid-stack cluster | InternViT (internvl3-8b) | L14/28 | +0.019 | text-stealing |
| Mid-stack cluster | ConvNeXt (convllava-7b) | L16/32 | +0.022 | text-stealing |
| Qwen-ViT late | Qwen-ViT (qwen2.5-vl-7b) | L22/28 (82%) | +0.015 | target-stealing |
| FastVLM late | FastViT (fastvlm-7b) | L22 | +0.047 | text-stealing (A7 +0.086) |

**핵심:** H3 ("ConvNeXt < ViT") falsified — 3개 다른 encoder가 같은
mid-stack text-stealing profile. 즉 **post-projection LLM stack depth**가
axis (H6: 2-axis decomposition).

---

## 슬라이드 19 — Causal ablation (E1d, §7)

**핵심 메시지:** single-layer null · stack-wide breaks fluency · upper-half
multi-layer만이 mitigation locus.

| 모드 | Δ direction_follow | Fluency | Use case |
|---|---|---|---|
| ablate_layer0 | null on 6/6 | clean | layer-0 control |
| ablate_peak (E1b 피크) | null on 6/6 | clean | multi-layer redundancy 입증 |
| ablate_lower_half | varies | clean → moderate | architecture별 다름 |
| **ablate_upper_half** | **−0.115 ~ −0.055 on 6/6** | **clean on 4/6** | ✅ E4 prototype locus |
| ablate_all | −0.22 ~ −0.11 universal | breaks on 3/6 | upper bound only |

**핵심:** anchor 효과는 stack 전반에 redundantly encoded. single-layer
attention-mask 차단은 무의미. upper-half multi-layer만이
architecture-blind mitigation locus.

---

## 슬라이드 20 — Mitigation "free lunch" (E4, §7)

**핵심 메시지:** mid-stack cluster 3-model Phase 2 full validation: df ↓ ·
em ↑ · acc invariant.

| Model | s* | Δ direction_follow | Δ exact_match | Δ acc(b) |
|---|---|---|---|---|
| llava-1.5-7b | −3.0 | −17.7 % rel | +0.77 pp | invariant |
| convllava-7b | −2.0 | −10.6 % | +1.30 pp | invariant |
| internvl3-8b | −0.5 | −5.8 % | +0.49 pp | invariant |

**"Free lunch" 의미:**
- df ↓ 함과 동시에 em이 *오히려 상승* (anchor에 끌렸던 wrong 답을 회복)
- base condition (no anchor) 의 acc는 *완전히 invariant*
- → mitigation은 anchor-condition-specific hook, single-image inference에 영향 0.

---

## 슬라이드 21 — Future Work (§8)

**핵심 메시지:** 3 방향 + 1번이 가장 priority.

1. **VLM vs LLM architectural diff (preferred)** — 동일 anchor를 VLM에는
   image, LLM에는 text로 줬을 때 layer-wise integration profile 비교 →
   architectural diff 입증.
2. **Image vs text anchor 비교** — cross-modal vs text-only LLM anchoring
   정량 비교.
3. **Reasoning-mode VLM (γ-β)** — Qwen3-VL thinking on/off × MathVista.
   reasoning chain 안에서 anchor가 amplify되는지 suppress되는지. VLMBias의
   "reasoning models can be more biased" cross-check.

---

## 슬라이드 22 — Conclusion

**한 문단 요약:**

> Cross-modal numerical anchoring in VLMs is **uncertainty-modulated
> graded pull**, concentrating on a **digit-pixel cue**, and mitigated by
> an **encoder-blind upper-half locus** — *without sacrificing accuracy*.

**4 takeaway:**

1. **Behavioural** — wrong > correct on direction-follow +6.9–19.6 pp · L1
   confidence quartile Q4-Q1 +0.128.
2. **Causal** — anchor − masked gap 0.5–15 pp on wrong-base S1 across 4
   datasets · digit-pixel는 effect의 인과.
3. **Mechanistic** — 4 archetypes의 attention peak · upper-half ablation
   −5.5 to −11.5 pp on 6/6 architecture-blind.
4. **Mitigation** — mid-stack cluster 3-model: df −5.8 to −17.7 % rel ·
   em +0.5 to +1.3 pp · acc invariant — "free lunch".

**Submission target:** EMNLP 2026 Main · ARR May 25.

---

## 발표자 부록 — 자주 묻는 질문 (FAQ)

**Q1. 왜 7B 모델에서 categorical-replace (MathVista) regime이 보이지 않고
27B에서만 보이는가?**
- gemma3-27b의 high-acc는 wrong-base subset을 cleanly delineated
  uncertainty subset으로 만듦. 7B는 wrong-base가 더 많지만 wrong이라는
  사실 자체가 noisy.
- 추가 가설: SigLIP encoder의 typographic weakness — gemma3-27b도
  SigLIP. E1-patch에서 후속 검증 (digit-pixel 집중도가 SigLIP에서 큰지).

**Q2. attention digit-pixel patch (E1-patch) 결과는?**
- POC 진행 중 (llava-1.5-7b + gemma4-e4b on n=400 stratified). 결과는
  `docs/insights/E1-patch-evidence.md` 에 추가될 예정.

**Q3. M2 metric refactor가 기존 publishing numbers를 바꾸는가?**
- adopt_rate는 ~10% 상승 (denominator 좁아짐). direction_follow_rate는
  큰 폭 하락 (no-change pair가 numerator에서 빠짐 → 진짜 movement만 카운트).
- 모든 변경은 mechanical, 새 데이터 아님. M1 backup 보존.

**Q4. 왜 closed model (GPT-4o, Gemini 2.5)을 안 쓰는가?**
- API 비용 + 5월 25일 마감 + Findings/Main 등급 결정 lever는 closed model
  보다 *mechanistic depth*. 500-sample reviewer-defuse run은 cheap하면
  추가 가능, 아니면 limitations 처리.

**Q5. Human baseline은?**
- 50명 Prolific run 가능했지만 cost / yield trade-off에서 yield 부족.
  cognitive science 인용으로 grounding하고 limitations로 명시.

---

*본 발표 자료는 `references/project.md §0` 의 paper outline + 모든
`docs/insights/` 의 evidence 문서를 종합한 것이다. 슬라이드 / 발표 노트 /
근거 데이터의 일관성은 `references/roadmap.md §3.3` 의 headline numbers
로 cross-check 가능.*
