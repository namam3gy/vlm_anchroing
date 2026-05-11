# Cross-Modal Numerical Anchoring in Vision-Language Models: Uncertainty, Plausibility, and Digit-Pixel Gates with a Deployable Subspace-Projection Mitigation

**시각언어모델의 cross-modal numerical anchoring — 불확실성·plausibility·digit-pixel gate와 배포 가능한 subspace projection mitigation**

---

## Abstract

**Vision-modality bias의 deployable mitigation은 causal pathway를 confounding scene variance로부터 분리하는 paired-inpaint calibration contrast 위에 구축할 수 있으며, 본 논문은 이 design pattern을 cross-modal numerical anchoring 위에 4-clause free-lunch worked example로 instantiate한다.** Worked example을 가능하게 하는 substrate는 6개 open-weight VLM의 행동·메커니즘 측정이다 — 본 논문은 질문과 무관한 두 번째 이미지에 그려진 단일 숫자가 수치 응답을 체계적으로 편향시키는 **cross-modal numerical anchoring** 현상을 보고하고, 효과가 categorical capture가 아닌 **불확실성에 비례하는 graded pull** (literal-anchor adoption 1.7-15.7 %, direction-follow B6−B1 6-bin gap +19.5-23.5 pp on 80 anchor cell across 5 dataset × 6 model) 이며 **digit-pixel × uncertainty 두 gate의 conjunction**으로 작동함을 (a − m) paired-inpaint 비교로 입증한다 — anchor scene 안의 digit pixel만 inpaint로 제거하면 효과가 generic distractor 수준으로 되돌아간다 (Figure 3). 이 (a − m) gap이 design pattern의 *paired-isolation 가정*을 행동 axis에서 충족하며, 동시에 §6.2 SVD calibration의 substrate로 발화된다. 메커니즘 측에서 single-layer ablation은 5-model panel + OneVision 5-dataset 확장 모두 5/5 null로 signal이 multi-layer redundant — 이는 design pattern을 구현할 때 *single-direction이 아닌 multi-direction subspace projection*을 선택해야 한다는 design choice를 정당화하며, §5.4 *routing vs integration 사후 synthesis*는 §5.2/§5.3/§6.4를 단일 narrative로 묶고 §4.6 Qwen3-VL γ-β residual-stream bridge에서 *partially prospective* 검증 (K=1 14/84 Bonferroni-clean cells; deploy K=8 partial-falsify at 동일 L=33 9× ratio) 을 받는다. Worked example은 두 mitigation을 instantiate한다 — **E4** (mid-stack attention re-weighting, df −9.6 ~ −14.6 %, em +0.77 ~ +1.30 pp) 와 **E6** (`llava-onevision-qwen2-7b-ov` 위 L=26에서 (a − m) calibration contrast로부터 K=8 SVD subspace를 1회 보정 후 inference 시 *anchor label 없이* 보편 projection); E6의 *anchoring effect* (Δdf < 0) 는 PlotQA n=2,306 위 95 % 및 Bonferroni-20 CI 모두 excludes 0 (single-dataset CI-clean) + 4 small-n cell 점추정-일관-CI-borderline, *capability-side multiplicity-robust headline* 은 **non-anchored arm Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 CI 모두 excludes 0**, 6-benchmark capability preservation 매크로 Δ = +0.41 pp (HallusionBench +2.21 pp [+1.14, +3.28] excludes zero; POPE pinned to zero; 단 3 cell negative point estimate, anchoring-adjacent axis dominant carrier). *Auxiliary observation*: Qwen3-VL-Thinking은 same continuous-confidence axis 위에서 anchor pull을 amplify하는 N=1 architecture × N=1 dataset *existence proof* (§4.5). Worked example은 단일 architecture instantiation이며 design pattern의 cross-architecture transfer는 §8.2 + §8.4 후속.

---

## 1 서론

### 1.1 동기와 자극 신규성

다중 이미지 prompt가 사용자 질의에서 흔해지는 가운데, 그중 *질문과 무관한 이미지가 하나*라면 — 우연이든 attacker가 의도적으로든 — 모델 응답에 영향을 주는가? 인지과학 문헌의 prior는 강하다: 인간은 무관함을 명시 통보받은 수치 단서에도 anchor를 내린다 [Tversky and Kahneman, 1974; Mussweiler and Strack, 1999]. LLM 문헌은 이를 *텍스트 anchor*에 대해 확립했다 [Jones and Steinhardt, 2022; Echterhoff et al., 2024]. 본 논문이 묻는 것은 — **의미 라벨 없는 단독 rendered-digit 이미지가 단서로 주어졌을 때, 동일한 효과가 이미지 modality에서도 일어나는가?**

기존 연구 중 *질문 대상과 무관한 independent rendered-digit 이미지*를 *cross-modal anchor*로 *open-ended numeric VQA*에 전달하여 *baseline-relative shift*를 측정한 사례는 없다. 가장 가까운 이웃 — VLMBias [Vo, Nguyen et al., 2025]는 *familiar subject (Adidas-style logo / 동물 / chess 등)*의 counting accuracy를, typographic attack [Wang et al., 2025b]은 *대상 이미지 위 오버레이*에 의한 분류 뒤집기를, FigStep [Gong et al., 2025]은 *prompt 텍스트의 이미지화*에 의한 jailbreak ASR을 측정한다. 본 논문은 cue를 *질문 subject로부터 분리*하고 metric을 *open-numeric-estimation의 baseline-relative shift*로 두는 보완적 paradigm을 도입한다.

### 1.2 핵심 주장 — 세 기둥

VLM의 cross-modal numerical anchoring은 **불확실성에 비례하는 graded pull**이며, **anchor 이미지의 digit pixel**이 효과의 인과 통로이다.

1. **Graded vs categorical.** 6개 모델 모두에서 paired adoption은 1.7-15.7 %로 모델이 anchor를 그대로 출력하는 일은 드물다. 효과의 질량은 *anchor 쪽으로의 점진적 이동*에 있다 (direction-follow 0.059-0.325 across 6 models on PlotQA, §4.1). 이동의 *크기*는 base-prediction confidence와 단조 관계 (pillar 3).
2. **Digit-pixel causality.** Anchor 이미지의 digit pixel만 inpaint로 가리면 paired adoption이 generic distractor 수준으로 *되돌아간다*.
3. **Confidence-modulated (continuous primary, binary projection consistent).** Direction-follow는 *base-prediction의 answer-span entropy*와 단조 monotonic 관계이다 (B6 − B1 6-bin gap **+19.5-23.5 pp** on 5 dataset × 6 model = 80 anchor cell, ≥ 4/5 strict pair-wise ↑ on 51-57 / 80 cells; §4.4). 통상의 *wrong-base vs correct-base* 분할은 이 연속 gradient의 거친 binary projection (B1+B2+B3 vs B4+B5+B6 평균에 해당) — 6-model PlotQA panel (§4.1) 에서 *df 기준* +19.0 ~ +34.4 pp 갭으로 동일 신호를 거친 형태로 운반한다 (legacy VQAv2 panel은 §C.1, +6.9 ~ +19.6 pp adopt 갭). γ-β reasoning 모드에서 이 binary projection이 *깨지는* 것은 (×12.7 ratio, §4.5) 연속 axis가 진짜 modulator라는 직접 증거이다.

### 1.3 메커니즘과 두 mitigation

Calibration dataset 위에서 모델별 단일 peak layer가 식별되지만 single-layer ablation은 5-model 메커니즘 panel에서 5/5 null — signal은 multi-layer redundant이다 (OneVision Main 5-dataset 확장에서도 single-layer 5/5 null로 *확장 검증*; §5.2 / §5.3). 본 논문의 **routing vs integration 통합 설명 framework** (사후 synthesis + §4.6 부분 prospective 검증)는 §5.2 multi-layer redundancy + §5.3 OneVision dataset-dependent peak + §6.4 LEACE rank-1 ChartQA +56 % 역행을 단일 mechanism narrative로 묶는 *사후 synthesis*이다 (§5.4) — multi-layer redundancy는 *attention pathway routing*의 속성이고, 그 routing의 결과가 *late residual stream에 통합*되어 single-layer site에서 접근 가능해진다는 view. Framework는 §4.6 γ-β residual-stream bridge에서 *partially prospective* leg을 받는다 — Qwen3-VL self-calibration K=1 subspace 위 within-Thinking paired Δ가 late-stack (L=29-34) positive + mid-stack (L=20) negative sign-reversal로 layer-routing *방향성* 예측 확인 (K=1 cell); 단 framework의 implicit *universal K=8* 가정은 동일 L=33에서 K=1 vs K=8 9× bridge ratio로 partial falsify (배포 K=8은 §6.2.2 OneVision K ∈ {2, 4, 8} grid 위 empirical sweet spot, framework prior 아님; §4.6). **E4** upper-half attention re-weighting은 mid-stack cluster (LLaVA-1.5 / ConvLLaVA) 2 모델에서 df −9.6 ~ −14.6 % 감소 + em +0.77 ~ +1.30 pp 상승의 free-lunch를 보이지만 inference 시 anchor token span을 요구한다. **E6** subspace projection은 Main 모델의 L=26에서 K=8 SVD subspace를 PlotQA + InfoVQA pooled 1회 보정 후 inference 시 보편 적용 — 5/5 dataset에서 df 감소 (평균 −2.9 pp, PlotQA만 CI excludes 0)와 동시에 *양 arm* em 상승 (anchored +3.9 pp, non-anchored +8.8 pp; **Δem(b) 5/5 cell × Bonferroni-corrected CI sign-clean**). 6-benchmark capability preservation 검증 (매크로 +0.41 pp, HallusionBench +2.21 pp excludes zero, POPE −0.06 pp pinned to zero)으로 free-lunch가 anchoring task 외부로 확장된다.

### 1.4 기여

**Vision-modality bias의 deployable mitigation은 causal pathway를 confounding scene variance로부터 분리하는 paired-inpaint calibration contrast 위에 구축할 수 있으며, 본 논문은 이 design pattern을 cross-modal numerical anchoring 위에 4-clause free-lunch worked example로 instantiate한다.** 본 논문의 *central contribution*은 이 design pattern과 그 첫 instantiation이다 — 즉 (i) §6.2.1의 (a − m) paired-inpaint calibration contrast가 *bias mitigation의 일반 design principle*로서 발화되며 (calibration contrast는 인과 통로 [digit pixel → answer shift] 를 confounding variance [anchor scene background → general distraction] 로부터 *paired difference*로 분리해야 한다), (ii) 이 design pattern을 cross-modal numerical anchoring에 instantiate한 worked example이 `llava-onevision-qwen2-7b-ov` 단일 architecture 위 E6 — *proof of construction*으로서 형식 정의된 4-clause free-lunch 기준 (Δdf < 0 ∧ Δem 양 arm ≥ 0 ∧ held-out capability ≥ −0.5 pp; §6.2.3) 을 5 evaluation dataset × 6 held-out capability benchmark 위에서 충족한다. **Figure 3이 design pattern의 두 직교 slice (cross-model + cross-dataset (a − m) digit-pixel causality) 를 한 panel로 carry하는 canonical figure이다.** Worked example의 *anchoring effect* (Δdf < 0 clause) 는 PlotQA n=2,306 single-dataset CI-clean (Bonferroni-20 후에도 excludes 0) + 4 small-n cell 점추정-일관-CI-borderline; *capability-side multiplicity-robust headline* 은 **non-anchored arm Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 corrected CI 모두 excludes 0** 이다 (§6.2.3 / §7) — 두 clause는 등가가 아니며 본 paper는 두 절을 별도 명시한다. Worked example의 cross-architecture transfer (다른 LM × encoder 위 design pattern instantiation) 는 design pattern의 *generalisability* 검증 과제로 §8.2 + §8.4 item 3에 정식 등록된다.

Design pattern과 worked example을 *evidence가 licensing* 한다 — (i) §4의 5 dataset × 6 model cross-dataset 위에서 anchoring을 wrong/correct binary projection이 아닌 *continuous confidence gradient*로 재해석하는 세 측정 axis 증거 (L1 6-bin gradient, (a − m) digit-pixel causality, wrong/correct binary stratification) 는 (a − m) paired-inpaint이 *실제로 인과 통로를 isolate한다*는 prerequisite을 제공하고 — digit pixel만 inpaint로 제거하면 효과가 generic distractor 수준으로 되돌아간다는 §4.2 (a − m) gap이 design pattern의 *paired-isolation 가정*을 행동 axis에서 충족; (ii) §5의 mechanism evidence — single-layer ablation이 5/5 null이고 signal이 multi-layer redundant라는 사실 + §5.4 *routing vs integration 사후 synthesis* + §4.6 layer-routing 방향성에 대한 K=1 partial-prospective 검증 — 은 design pattern을 구현할 때 *single-direction이 아닌 multi-direction subspace projection*을 선택해야 한다는 design choice를 정당화하며 (single-direction ActAdd + LEACE rank-1의 cross-dataset 실패 §6.4가 이를 부정 측 증거로 보강); (iii) 평가 protocol 자체 — independent-anchor open-numeric-estimation 4-condition (b/a/m/d) + gt-자유 direction-follow metric — 은 VLMBias [Vo, Nguyen et al., 2025]의 familiar-subject counting paradigm과 *cue independence × measurement type* 두 축에서 상보적이며 (§2 · §3), (a − m) paired contrast를 *측정 가능하게 만드는* substrate이다. 배포된 E6의 K=8 parameterization은 OneVision K ∈ {2, 4, 8} grid 위 empirical sweet spot (§6.2.2) 이며 framework의 implicit universal-K=8 가정은 동일 L=33 Qwen3-VL에서 K=1 vs K=8 9× bridge ratio로 partial falsify되어 — design pattern instantiation의 *operational hyperparameter*는 cross-architecture 재calibration을 요구한다는 운영적 결과로 §8.2 + §8.4에 명시된다.

---

## 2 관련 연구

**LLM anchoring 계보.** Tversky and Kahneman [1974]의 anchoring & adjustment heuristic과 Mussweiler and Strack [1999]의 selective accessibility 모델은 무관한 수치 단서가 *비교 대상으로 working memory에 진입*해 후속 판단에 점진적으로 혼합된다는 메커니즘 가설을 제공한다. LLM 시대에 들어 Jones and Steinhardt [2022], Echterhoff et al. [2024]는 텍스트 anchoring을 확립했고, Lou and Sun [2024]은 텍스트 LLM에서 anchoring을 재확인하면서 Chain-of-Thought, Thoughts-of-Principles, Ignoring-Anchor-Hints, Reflection 같은 *prompt-level mitigation들이 모두 불충분*함을 보고 — 이는 본 논문이 § 6에서 *representation-level (residual-stream subspace) intervention*으로 향하는 직접적 동기이다. Wang et al. [2025a]은 LRM에서 *judging bias가 reasoning trace를 통해 증폭*됨을 보고했다. Huang et al. [2025]은 합성 데이터로 메커니즘을 분해했다.

**Multimodal cognitive bias 평가.** VLMBias [Vo, Nguyen et al., 2025]는 가장 포괄적인 VLM cognitive bias benchmark으로, *모델이 사전 지식을 가진 visual subject* (Adidas-style logo, 동물, chess board, board game, optical illusion, patterned grid 등 7개 domain)를 stimulus로 사용하여 *familiar-subject counting accuracy* (예: stripe 개수)를 측정한다 — 평균 17.05 % counting accuracy + background 제거 시 +21.09 pp 회복 (arXiv:2505.23941). AIpsych [Liu et al., 2025], CIVET [Rizzoli et al., 2025], Tinted Frames [Fan et al., 2026]은 sycophancy / position bias / question framing을 다룬다. 본 논문은 이들과 *측정 대상*과 *cue source* 두 축에서 상보적이다 — VLMBias가 *familiar subject의 prior knowledge에 대한 counting accuracy*를 측정하는 데 비해 본 논문은 *질문 대상과 무관한 독립 rendered-digit anchor 이미지에 대한 open-ended numerical estimation의 baseline-relative shift*를 측정한다 (cue가 question 자체의 subject가 아니라 independent second image; metric이 closed-form counting이 아니라 arbitrary anchor에 대한 회귀형 이동). 두 paradigm 모두 "visual content가 numerical answer를 편향시킨다"는 같은 mechanism question을 공유하나, 측정 대상 (counting vs open estimation)과 cue dependency (subject-bound vs independent draw)에서 분리된다.

**Typographic attack과 mechanism 연구.** Goh et al. [2021] "Multimodal Neurons"는 CLIP에서 텍스트 픽셀이 의미 neuron을 활성화한다는 — typographic attack의 인과 기반 — mechanism을 처음 정립했다. Wang et al. [2025b, NAACL] multi-image typographic attack과 FigStep [Gong et al., 2025]은 *클래스 라벨* 또는 *prompt 텍스트*의 이미지화로 분류 뒤집기·jailbreak를 측정한다. 본 논문의 단서는 *수치값 단독* (클래스 정체성 없음), 표적은 *open-ended numerical estimation* (분류 뒤집기·ASR 아님). Hufe et al. [2025, Dyslexify]는 typographic attack에 대한 *encoder-side mechanistic defense* (CLIP 측 개입)를 제시 — encoder-side에서 작동하는 typographic-attack defense의 가장 가까운 mechanistic 이웃이다. 본 논문 E6는 이와 *상보적*으로 LM residual stream에서 작동한다. Weng et al. [2024, EMNLP Main]은 *gender bias*를 대상으로 causal mediation으로 image-encoder 기여를 식별하고 encoder-side feature blurring으로 22 % bias reduction (MSCOCO)을 보고한 EMNLP-Main mechanism→mitigation 사례이다. 본 논문은 *bias class*가 다르고 (numerical anchoring), mitigation의 *작용 site*도 다르다 (encoder가 아닌 LM residual stream); 공유하는 부분은 *mechanism→mitigation chain*이라는 venue-tier 형식이다.

**Activation steering과 concept erasure.** §6의 mitigation은 residual-stream intervention 계열에 속한다. CAA [Panickssery et al., 2024]는 *paired contrastive activation* — positive vs negative behavioral example pair의 residual-stream 차분 평균 — 으로 *single-direction* steering vector를 도출한다. ITI [Li et al., 2023]는 *attention head* 출력 수준에서 *multi-direction* 개입을 수행한다 (multiple head × direction; ITI 자체 평가는 TruthfulQA 위 LM-only로 수행되며 VLM × open-numeric anchoring으로의 직접 transfer는 미수행). LEACE [Belrose et al., 2023]는 closed-form linear concept erasure (rank-1 default 포함)로 baseline 표현 손상을 최소화하면서 선형 분류기가 concept을 검출하지 못하게 만든다. 본 논문 E6는 이 계열의 직접 후속 — (i) CAA의 paired-contrast 패러다임을 *multi-direction subspace* (K=8 SVD)로 확장하고, (ii) ITI의 attention-head locus 대신 *residual-stream* locus에서 작동하며, (iii) text-only steering 문헌에 없는 *vision-modality (a − m) paired-inpaint contrast* 구성을 도입한다. §6.2.1의 (a − m) paired-inpaint은 CAA의 behavioral paired-contrast paradigm에 *vision-modality specific 인과 통로 분리* 구조를 부여하는 일반 design pattern으로 positioned되며 (calibration contrast가 인과 통로를 confounding variance로부터 분리하는 paired difference여야 한다는 design principle, §6.2.1 Insight). §6.4에서 LEACE의 *rank-1 closed-form 인스턴스*를 직접 비교 baseline으로 평가하며 (LEACE 자체는 closed-form linear erasure framework이고 rank-1은 그 default 운영 모드), CAA·ITI는 §6.5 Table 8 footnote에서 single-direction failure mode (ActAdd 대등 — paired-contrast residual stream at K=1) 또는 multi-layer redundancy + dataset-dependent attention peak로부터 *방향성* 예측되는 multi-head locus 일반화 도전으로 reduce되는 점을 명시 — 본 작업의 differentiator는 *기법 class의 신규성이 아니라 multi-direction × residual-stream × (a − m) paired-inpaint 조합이 4-clause free-lunch 후보로 기능*한다는 점이다. 이 후보 자격은 'No Free Lunch in Language Model Bias Mitigation' [Chand et al., 2025]이 *LM × 이산 social bias × weight space*에서 보고한 4-clause 동시 충족 *실패*에 대한 *cross-axis positive result* — *VLM × 연속 numerical regression × inference-time activation projection*에서 4-clause 동시 충족이 가능함 — 으로 positioning된다 (§6.2.3 형식 정의 + §7 검증; 본 작업은 vision-token pruning / image-attention re-weighting 계열의 인접 prior class와는 *작용 site*에서 분리된다 — vision encoder 또는 cross-modal attention layer가 아닌 LM의 post-cross-attention residual stream 위에서 작동).

---

## 3 방법

### 3.1 자극 4-조건

`sample_instance` = (target_image, question, anchor_draw). 한 sample_instance마다 최대 4개 조건 (Table 1)을 평가하고 각 조건이 모델 예측 `pred_b / pred_a / pred_m / pred_d`를 산출한다.

**Table 1.** 4-조건 정의.

| 약어 | 라벨 | 두 번째 이미지 |
|---|---|---|
| `b` | `target_only` | 없음 (baseline) |
| `a` | `target_plus_irrelevant_number` | 단일 digit 이미지 (anchor) |
| `m` | `target_plus_irrelevant_number_masked` | 같은 anchor 이미지에서 digit pixel만 inpaint 제거 |
| `d` | `target_plus_irrelevant_neutral` | digit 없는 FLUX render (distractor control) |

세 조건 gap은 각각 (a − d) *anchoring vs generic distraction*, (a − m) *digit pixel vs anchor scene background*, ((a, base-wrong) − (a, base-correct)) *uncertainty modulation*을 분리한다. 자극 inventory는 128개 FLUX-rendered digit 이미지 (`a`) + 128개 OCR-검증된 Telea inpaint (`m`) + 128개 digit-free FLUX render (`d`)이다 (부록 §A; 4-조건 자극이 모델별 acc drop에 미치는 효과 예시는 Figure 1).

![Figure 1 — 4-조건 자극의 효과 예시. anchor / masked / neutral 조건에서 base 대비 accuracy drop을 모델별로 비교; anchor가 가장 큰 drop, masked와 neutral은 1-2 pp 안에서 구별 불가.](../figures/E5c_acc_drop_3way.png)

### 3.2 표준 metric (M2)

`pa = pred_a`, `pb = pred_b`로 두면

```
adopt_rate            = #(pa == anchor AND pb != anchor) / #(pb != anchor)
direction_follow_rate = #( (pa - pb) · (anchor - pb) > 0  AND  pa != pb )
                        / #(numeric pair AND anchor present)
exact_match           = #(pa == gt) / #(numeric pair)
```

`direction_follow_rate` 분자는 `pa`가 baseline `pb`에서 *anchor 쪽으로* 이동했는지를 측정한다 — baseline-relative shift. `pb`(`gt` 아님)을 reference로 사용함으로써 metric은 모델 출력과 anchor draw에만 의존하며 baseline 정답 여부와 `gt` 자체에 무관하다. Stimulus별 변동성과 dataset별 GT 분포 차이에 robust하다. 18개 numerator × denominator 변형의 known-signal preservation 분석은 부록 §B.

**Near-tautology caveat — `df > 0` 자체는 evidence가 아니다.** 본 C-form 분자 `(pa − pb)·(anchor − pb) > 0 ∧ pa ≠ pb` 는 anchor가 zero-effect인 null 하에서도 *대칭에 의해* `df ≈ 0.5 × P(pa ≠ pb) > 0`을 산출한다 — 즉 단순 "df > 0"은 stimulus / null 조건과 거의 무관하게 자동 만족된다. 본 paper에서 anchor-specific signal의 load-bearing 증거는 *모두 다른 cut*에서 나온다: wrong-base vs correct-base 비대칭 (§4.1, §4.4), confidence-bin gradient의 monotonicity (§4.4), (a − m) digit-pixel paired-inpaint gap (§4.2). 본 caveat은 §4.1 본문에서도 다시 cross-reference 된다.

### 3.3 데이터셋과 모델 패널

5개 1차 numeric VQA dataset: TallyQA (자연 이미지 카운팅, raw n ≈ 38 k), ChartQA (차트 정수 GT, raw n = 5,390), MathVista (testmini integer, n = 385), PlotQA (과학 plot V1, raw n ≈ 5,000), InfographicVQA (val numeric, n = 1,147). 위 raw n은 stratification · eligibility 필터 *이전* count이며, 실제 본문 표에 사용된 per-cell n은 stratified 부분집합 기준으로 ChartQA 129–517 / TallyQA 6,934–14,772 / PlotQA 926–4,610 / InfoVQA 218–865 / MathVista 127–274 *범위에 분포*한다 (모델별 변동). **§4.1 single-dataset depth panel = PlotQA** (5-dataset 중 GT-range 가장 넓고 anti-scaling / em-positive baseline / sample-size-robust (a-m) gap top-tier가 모두 가장 또렷이 분리되는 dataset; n_pair 4,554–4,707 per model). 레거시 7-model VQAv2 number subset (n=17,730 per model, GT range [0, 9])은 §C.1 부록 — adopt-기반 동일 wrong > correct 패턴 (+6.9 ~ +19.6 pp on 7/7 models) 을 단일 자릿수 분포에서 운반하는 cross-stimulus replication panel. **Main panel 6 모델**: `llava-onevision-qwen2-7b-ov` (Main, 28-layer Qwen2-7B), `google/gemma-3-4b-it`, `google/gemma-3-27b-it`, `llava-hf/llava-interleave-qwen-7b-hf`, `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct`. **Mechanism panel 5 모델**: gemma4-e4b, llava-1.5-7b, ConvLLaVA-7b, qwen2.5-vl-7b, fastvlm-7b — 5개의 서로 다른 visual encoder 조합 (SigLIP, CLIP-ViT, ConvNeXt, Qwen-ViT, FastViT) cover; 모델별 peak layer + cross-dataset variability 자세히 부록 §D.1. **γ-β**: Qwen3-VL-8B Instruct vs Thinking. 누계 ~1.6 M model generation, ~5,760 H200 GPU-hours.

**Panel scope by analysis axis.** 본 논문의 결과는 panel-scope 측면에서 세 register로 분리되어 보고된다 — *behavioral* 결과 (§4.1 PlotQA depth × 6-model, §4.3 5-dataset breadth × 6-model, §C.1 legacy VQAv2 × 7-model)는 multi-model panel 위에서, *mechanism* 결과 (§5.2 single-layer ablation × 5-model mech panel, §5.3 OneVision Main dataset-dependent peak)는 mech panel + OneVision 확장 위에서, *deployable mitigation* 결과 (§6.2 / §6.5 / §7 E6 chain calibration · 5-dataset eval · baseline 비교 · capability preservation)는 단일 모델 `llava-onevision-qwen2-7b-ov` 위에서의 *case study*로 한정된다. Cross-architecture E6 재calibration은 §8.2 한계 + §8.4 item 3 후속 작업.

---

## 4 행동 분석과 인사이트

### 4.1 Graded pull (PlotQA single-dataset depth panel) — confidence gradient의 binary projection

VLM의 cross-modal anchoring은 categorical capture가 아닌 **graded pull**이다: 6-model PlotQA panel (Table 2; n=5,000 base per model, S1 anchor `|a − GT| ≤ max(1, 0.10·GT)`) 에서 paired adoption은 1.7-15.7 %로 floor 수준인 반면 direction-follow는 0.059-0.325 범위로 그보다 1.5-8× 크다 — 효과의 질량은 *anchor 쪽 graded movement*에 있고, 모델이 anchor를 *그대로 출력*하는 일은 드물다.

***`df > 0`은 evidence가 아님 — metric construction의 near-tautology.*** C-form 분자 `(pa − pb)·(anchor − pb) > 0 AND pa ≠ pb`는 anchor가 zero-effect인 null 하에서도 *대칭에 의해 `df ≈ 0.5 × P(pa ≠ pb) > 0`*을 산출 — 즉 "df > 0"은 stimulus / null 조건과 거의 무관하게 자동 만족된다. 본 paper에서 anchor-specific signal의 load-bearing 증거는 *모두 다른 cut*에서 온다: (i) **wrong-base vs correct-base 비대칭** *df 기준* +19.0 ~ +34.4 pp on 6/6 models (§4.1 후속 단락) — confidence stratification에 따른 *비대칭* 자체는 null 대칭으로 설명 불가; (ii) **L1 confidence 6-bin gradient** B1<B2<...<B6, +19.5-23.5 pp B6-B1 gap; ≥ 4/5 strict pair-wise ↑ on 51-57 / 80 cells (§4.4) — null 하 monotonicity 기대 안 됨; (iii) **(a − m) digit-pixel gap** +6.2 pp on PlotQA OneVision wrong-base S1 (§4.2) — *같은 anchor scene*에서 디지트 픽셀만 inpaint했을 때 효과가 사라짐, scene-level + S1 trivial-recovery confound 동시 falsify (자세한 stratum별 plausibility window 검증은 §C). 본 단락의 magnitude (0.059-0.325) 는 *graded vs categorical* 시각적 사이즈를 제공하지만 anchor-effect의 *정성적* 증명은 (i)-(iii) 세 axis가 담당한다.

**Table 2.** 6-model PlotQA panel, all-base S1 anchor arm (paired-sids intersection over (a-S1, b) per model, n_pair 4,554–4,707). Bold cell = 각 metric 기준 셀 최댓값 또는 최솟값 (`adopt(a)` max + min, `df(a)` max + min, capability-bound caveat 모델 제외). Bold row name = 해당 metric 기준 가장 robust 모델. `adopt(a)` 기준 panel min = Qwen2.5-VL-7b (0.017, Qwen2.5-VL-32b와 동률), `df(a)` 기준 panel min = Qwen2.5-VL-7b (0.059, Qwen2.5-VL-32b와 동률) — 두 metric이 모두 Qwen2.5-VL family를 panel 최하단 robust group으로 지정한다.

| 모델 | n_pair | acc(b) | adopt(a) | df(a) | em(a) |
|---|---:|---:|---:|---:|---:|
| Gemma3-4b-it | 4,707 | 0.300 | **0.157** | 0.294 | 0.350 |
| LLaVA-Interleave-7b † | 4,554 | 0.119 | 0.105 | 0.325 | 0.116 |
| LLaVA-OneVision-7b *(Main)* | 4,699 | 0.481 | 0.069 | 0.130 | 0.502 |
| Gemma3-27b-it | 4,698 | 0.514 | 0.063 | 0.118 | 0.546 |
| Qwen2.5-VL-32b-instruct | 4,707 | 0.729 | 0.017 | 0.059 | 0.757 |
| **Qwen2.5-VL-7b-instruct** | 4,706 | **0.783** | **0.017** | **0.059** | 0.804 |

† **LLaVA-Interleave는 fixed-resolution image input만 받아 PlotQA chart를 강제 다운샘플** — `acc(b) = 0.119`는 anchor-vulnerability가 아닌 *resolution-bound capability ceiling*에서 비롯된다 (다운샘플된 chart 이미지는 사람 평가자도 정답 수치를 읽기 어려운 수준). 따라서 본 모델의 `df(a) = 0.325` panel-max는 *intrinsic encoder weakness*가 아닌 *input pipeline-induced low-capability data point*로 해석되어야 하며 — 단 anchor-effect 부호와 ordering은 panel 다른 5 모델과 정합 (Insight 1 능력↔anchor pull 역상관의 *low-capability anchor*로 기능). 따라서 표상 cell-level 굵은 highlight에서 제외하나 wrong-correct asymmetry 6/6 검증에는 포함된다.

**PlotQA single-dataset depth panel의 역할.** PlotQA는 본 paper main matrix의 **GT-range 가장 넓은 dataset** (정수 [1, 10000], 5-stratum sampled n=5,000)이며, §4.3 main matrix의 5-dataset breadth와 *상보적*인 *single-dataset depth* axis로 사용된다 — Phase-A H2 binary projection (§4.1 본 절) / (a − m) digit-pixel gap (§4.2 Table 3) / L1 6-bin gradient (§4.4) 의 *replication depth*가 PlotQA cell에서 selectively 가장 강하다. 5-dataset 중 anti-scaling (4B > 27B; §4.3 Insight 2) / em-positive baseline (em(a) > em(b) on 5/6 model; §4.2 Insight 3 — Slice A E7 PlotQA panel) / (a-m) gap top-tier (sample-size-robust; §4.2 Table 3 Slice B에서 PlotQA n_wb=2,107로 +6.2 pp, MathVista n_wb=152로 점추정 +6.6 pp가 동률 상위) 모두 PlotQA를 일차 evidence로 한다 — 즉 본 panel은 paper-wide pattern이 *가장 또렷이 분리되는* dataset에서 6-model breadth를 한 cell로 압축한다. (legacy VQAv2 number subset n=17,730 7-model panel은 §C.1 부록으로 이전 — 단일 자릿수 GT range [0,9] / skewed answer 분포로 인한 ceiling 한계 + S1 distance stratification 의미 약화 caveat 포함.)

본 panel과 §4.3 main matrix는 *동일 panel의 재실행이 아닌* 독립 axes로 — depth (PlotQA n=4,700 × 6 model, LLaVA-Interleave 포함) ↔ breadth (5 dataset × stratified per-cell × 6 model, LLaVA-Interleave 포함) — 두 axis가 *모든 정성적 claim의 cross-replication*을 제공한다.

`base_correct` (baseline 정답 여부) 로 stratify하면 *df 기준* 6/6 모델에서 wrong-base direction-follow rate가 correct-base보다 **+19.0 ~ +34.4 pp 더 크다** (Figure 2). adopt 기준에서도 6/6 sign-clean (gap +3.1 ~ +14.4 pp). 이는 §4.4의 continuous 6-bin gradient가 *B1+B2+B3 vs B4+B5+B6*로 평균 매핑된 binary projection이며 (§4.4 Insight 1에서 6-bin 기준 +28.9 pp로 확대 재유도, PlotQA × OneVision worked example), 본 panel은 그 projection이 *PlotQA chart-stack stimulus*에서도 — 즉 VQAv2 단일 자릿수 분포가 아닌 GT range [1, 10000] 의 본격 numerical question 위에서도 — 재현됨을 보이는 6-model 폭 검증이다.

![Figure 2 — Confidence-gradient의 coarse binary projection (PlotQA 6-model). df 기준 wrong-base > correct-base on 6/6 모델, gap +19.0-34.4 pp 양수. adopt 기준에서도 6/6 sign-clean. LLaVA-Interleave는 resolution-bound capability ceiling caveat 따라 *low-capability anchor*로 표시 — gap이 6/6 검증의 일부이지만 magnitude 해석은 footer † 참조.](../figures/paper_4_1_PlotQA_correct_vs_wrong_df.png)

**Insight 1 (능력↔anchor pull 역상관).** Anchor pull 크기는 base accuracy 순서의 *역*과 거의 일치한다 — *능력이 낮은 곳에서 pull이 크다*. resolution-caveat 모델 제외 시 ordering은 Gemma3-4b 0.294 → Gemma3-27b 0.118 → LLaVA-OneVision *(Main)* 0.130 → Qwen2.5-VL-7b/32b 0.059 동률 (panel min); `adopt(a)`도 Qwen2.5-VL family 0.017 panel min에서 같은 ordering. LLaVA-Interleave는 architecture-bound resolution ceiling이 부과한 low-capability cell로, df = 0.325 panel-max region에서 동일 prediction을 보인다 — capability ceiling의 *origin*에 무관하게 (intrinsic 모델 size든 input pipeline forcing이든) 역상관이 robust하다. 이는 §4.4의 confidence 6-bin 결과를 baseline 능력 측에서 미리 시사한다.

**Insight 2 (Mussweiler-Strack 직접 예측).** wrong-correct asymmetry가 *df 기준* 6/6 모두 양인 것은 우연이 아니다 — selective accessibility 모델의 *직접* 예측이다. baseline 신뢰도가 낮을수록 비교 후보로 anchor가 더 쉽게 활성화되며, 활성화된 anchor가 답안에 점진적으로 혼합된다. PlotQA 6-model 결과는 인지과학 가설의 VLM 측 외부 검증이며, 동시에 *GT range / encoder family / stimulus type 일반화* 검증이기도 하다 — VQAv2 [0,9] 단일 자릿수 분포에서 +6.9 ~ +19.6 pp adopt asymmetry로 재현되는 패턴 (§C.1 legacy panel) 이 PlotQA [1, 10000] 본격 chart-numeric 분포에서 +19.0 ~ +34.4 pp df asymmetry로 *증폭되어* 재현된다. 이는 H2 asymmetry가 GT range / question type / encoder family에 *무관한* mechanism-bound prediction임을 시사한다.

**Cross-dataset replication.** 본 절의 두 Insight는 TallyQA (n=24k–38k per model, 6-model panel) 와 InfoVQA (n=1,076 per model, 6-model panel) 에서도 재현된다 — Insight 1 능력↔anchor pull 역상관 부호 3/3 dataset (Qwen2.5-VL family가 모든 dataset에서 acc(b) 최고 + df 최저), Insight 2 wrong > correct df gap 부호 6/6 모델 × 3 dataset (TallyQA +7.0 ~ +12.1 pp, InfoVQA +11.3 ~ +31.7 pp). adopt 기준에서도 6/6 sign-clean on 3/3 dataset. 자세한 cross-dataset replication 표는 §C.3 (Table C.2 + C.3). 본문 6-row table을 PlotQA에 한정한 것은 §4.3 5-dataset main matrix breadth와의 axis 분업을 위해서이며 — *paper-wide pattern이 가장 또렷한 single dataset에서의 depth coverage* 와 *5-dataset breadth* 의 두 axis로 분리한다.

### 4.2 Digit-pixel causality

(`b`, `a`, `m`) 비교는 digit pixel의 효과 기여를 정량화한다. Mask는 같은 장면의 digit 영역만 Telea inpaint로 가린 것 — scene background는 유지 (Figure 3).

![Figure 3 — Digit-pixel causality, two orthogonal slices. (top) PlotQA 6-model cross-model panel (E7) — wrong-base × S1 paired adopt(a) vs adopt(m), 6/6 모델에서 (a−m) > 0 (range +1.0 ~ +12.8 pp). (bottom) LLaVA-OneVision Main × 5-dataset cross-dataset panel (E5b/E5e) — 5/5 dataset에서 (a−m) > 0 (range +0.7 ~ +6.6 pp). 두 슬라이스 모두 PlotQA를 포함하며 +6.1 pp (E7) / +6.2 pp (E5b)로 독립 run에서 일치 — cross-validation.](../figures/paper_4_2_digit_pixel_causality.png)

**S1 confound resolution.** 본 비교는 wrong-base × S1 (TallyQA `|a − GT| ≤ 1` absolute, chart-stack은 `max(1, 0.10·GT)` relative — 자세한 dataset별 cutoff + 거리 plausibility 감쇠 검증은 §C) 부분집합에서 수행된다. Wrong-base × S1에서 anchor가 gt에 plausibly 가까우면 "모델이 gt 쪽으로 복귀하는 것이 anchor 쪽으로 보이는" trivial 해석이 가능하다 — `(pa, pb, anchor)` 의 부호가 우연히 정렬될 수 있기 때문. (`a`, `m`) 페어는 *같은 anchor scene*에서 *디지트 픽셀만* 차이 — 거리·scene·plausibility는 동일하므로 이 confound는 두 arm에 *같은 정도로 작용*하고 (a − m) 차분에서 *상쇄*된다. 따라서 (a − m) gap > 0은 "S1에서 모델이 gt 회복" 가설로 설명 *불가* — 디지트 픽셀이 인과적이라는 *clean separation*이다.

**Table 3.** Wrong-base × S1 paired adoption + (a − m) gap. 두 직교 슬라이스로 제시 — (A) PlotQA 위에서 모델 breadth, (B) Main 모델 위에서 dataset breadth. PlotQA는 두 슬라이스에 공통 cell이며 두 독립 run (E7, E5b) 의 (a − m) gap이 +6.1 / +6.2 pp로 일치한다 (cross-run replication).

*Slice A — PlotQA × 6-model (E7 full panel, source `experiment_e7_plotqa_full_per_cell.csv`).*

| 모델 | n_wb | adopt(a) | adopt(m) | (a − m) |
|---|---:|---:|---:|---:|
| Gemma3-4b-it | 3,036 | 0.184 | 0.056 | **+12.8 pp** |
| Gemma3-27b-it | 1,939 | 0.099 | 0.037 | **+6.1 pp** |
| LLaVA-OneVision-7b *(Main)* | 2,106 | 0.090 | 0.028 | **+6.1 pp** |
| LLaVA-Interleave-7b † | 4,029 | 0.082 | 0.014 | **+6.8 pp** |
| Qwen2.5-VL-7b | 902 | 0.024 | 0.007 | +1.8 pp |
| Qwen2.5-VL-32b | 1,153 | 0.023 | 0.013 | +1.0 pp |

*Slice B — LLaVA-OneVision-7b *(Main)* × 5-dataset cross-dataset panel. Source: `experiment_e5b_5strat_<dataset>_onevision_per_cell.csv` (PlotQA / MathVista / ChartQA / InfoVQA), `experiment_e5e_tallyqa_full_per_cell.csv` (TallyQA OneVision backfill, Phase 1 P0 v3).*

| 데이터셋 | n_wb | adopt(a) | adopt(m) | (a − m) |
|---|---:|---:|---:|---:|
| MathVista | 152 | 0.105 | 0.039 | **+6.6 pp** |
| PlotQA | 2,107 | 0.087 | 0.025 | **+6.2 pp** |
| InfoVQA | 403 | 0.045 | 0.037 | +0.7 pp |
| ChartQA | 211 | 0.028 | 0.014 | +1.4 pp |
| TallyQA | 7,119 | 0.032 | 0.022 | +1.0 pp |

† LLaVA-Interleave는 fixed-resolution input으로 PlotQA chart를 강제 다운샘플 — Slice A에서 제공하는 (a − m) gap은 *low-capability anchor* 위치에서의 digit-pixel attribution이다 (§4.1 Table 2 footer 참조).

`adopt(m)` 해석 *— reader 혼란 방지*: m-arm은 anchor scene에서 *디지트 픽셀이 inpaint로 제거된* 조건이므로 모델이 anchor digit을 *볼 수 없다*. `adopt(m)` 분자 `pa_m == anchor_value`의 `anchor_value`는 *sample metadata로 보존된* (a-arm에서 보였을) anchor 값이며, m-arm에서는 모델이 그 값을 *anchor로서 채택한 것이 아니다*. `adopt(m)`이 양수일 수 있는 원인은 (i) 모델 prediction noise가 우연히 anchor_value와 일치 (~0.5 × P(numeric pair) baseline), (ii) anchor scene background에서 잔존하는 미세한 cue (Telea inpaint이 픽셀 레벨에서 완전 무 잔여 OCR 검증되었으나 representation level에서의 잔여 가능성), (iii) 모델의 prior digit 분포 편향 (예: 모델이 "3"을 자주 출력하는 경향) — 세 가지 모두 *anchor 효과가 아니다*. 따라서 `adopt(m)`은 *random-coincidence + scene-residual baseline floor*로 기능하며, `(a − m)` gap이 이 floor 위의 *digit-pixel-attributable adoption*을 분리한다.

(`b`, `m`, `d`) 통제는 *anchor scene background가 효과를 운반*하는 가설을 기각한다 — masked와 neutral이 correct-base 정확도에 끼치는 손실은 1-2 pp 안에서 구별 불가, scene 자체는 generic distractor와 동등.

**Insight 1 (단조 ordering — 능력↔anchor pull 역상관과 정렬).** Slice A에서 (a − m) gap은 모델 anchor pull 강도와 같은 부호로 움직인다 — Qwen2.5-VL-7b는 §4.1의 `adopt(a)` panel-min 모델 (PlotQA 0.017)이며 양 arm이 모두 floor에 위치하여 (a − m) gap도 noise 안에 머문다 (+1.8 pp); panel 끝의 Gemma3-4b는 `adopt(a)` 0.157 (panel-max)에서 (a − m) +12.8 pp로 가장 큰 gap을 보인다. 즉 (a − m) gap은 *digit pixel의 인과 기여를 측정하면서 동시에 모델별 pull 강도의 비례 함수*로 작동하며 — 메커니즘과 효과 크기가 같은 축에서 움직인다.

**Insight 2 (panel-wide 부호 일관성 + magnitude는 sample-size 의존).** Slice A에서 6/6 모델, Slice B에서 5/5 dataset이 (a − m) > 0 — *점추정 부호* 측면에서 cross-model + cross-dataset 일관. 단, magnitude는 sample-size에 강하게 의존한다 — Slice A의 load-bearing 증거는 PlotQA n_wb 902–4,029 위에서 Gemma 4b +12.8 pp / Interleave +6.8 pp / OneVision +6.1 pp / Gemma 27b +6.1 pp 같은 ≥ +6 pp gap cell이며 Qwen2.5-VL family의 +1.0 / +1.8 pp on n_wb 902–1,153 은 점추정 sign-positive 이지만 paired SE 안 (~1.4 pp on n=1,000). Slice B에서도 PlotQA n_wb=2,107 +6.2 pp + MathVista n_wb=152 +6.6 pp 두 cell만이 sample-size-robust 큰 gap이며 InfoVQA n_wb=403 +0.7 pp · ChartQA n_wb=211 +1.4 pp · TallyQA +1.0 pp 세 cell은 point-estimate sign-positive + magnitude noise-floor 안. **따라서 digit-pixel causality의 load-bearing magnitude evidence는 PlotQA + MathVista cell pair에 집중되며, 다른 4 cell은 부호 일관 증거로만 보고**한다. 이 panel-wide 부호 일관성은 §6.2.1의 SVD calibration이 (a − m) contrast를 *digit-pixel-specific principal direction* 추출의 paired difference로 사용하는 설계의 prerequisite (cell-by-cell sign-positive) 를 충족한다. **§6.2.3 5-dataset paired-sids Δdf 표 ordering (PlotQA Δdf = −5.2 pp largest / TallyQA −0.3 pp floor) 은 본 절 Slice B의 (a − m) gap magnitude ordering (PlotQA + MathVista top-tier ↔ TallyQA + InfoVQA floor-tier) 과 *개별 dataset 부호 일관*하며, 이 사전 정렬이 §6.2의 단일 (L, K, α) cell이 dataset-shared subspace direction을 capture한다는 §6.2.3 Insight 2의 메커니즘 측 prerequisite을 본 절에서 미리 충족한다.**

**Insight 3 (PlotQA un-mitigated free-lunch).** Slice A E7 PlotQA panel의 *놀라움*: 6개 중 5개 모델이 `em(a) > em(b)` (em delta +0.6 ~ +5.0 pp; per-model em table source 부록 §A.5). S1 cutoff가 anchor를 GT의 ±10 % 안에 두므로 anchor를 "그럴듯한 추측 단서"로 픽업하는 모델은 정확도를 *얻는다*. 이 패턴은 InfoVQA로 일반화하지 *않으며* (혼합 부호 — 같은 evidence file의 InfoVQA panel), *§6.2의 free-lunch mitigation이 이 PlotQA baseline 패턴을 5-dataset에 일반화 가능한 복구 메커니즘으로 변환하는 정확한 도메인*이다.

### 4.3 5-dataset main matrix 종합

전체 5 × 6 main matrix (Figure 4)에서 다음이 관찰된다.

![Figure 4 — 5-dataset 6-model wrong-base S1 direction-follow. df 부호 30/30 cell 모두 양수. gemma3-4b가 ChartQA/PlotQA/MathVista에서 가장 큰 끌림 (단, InfoVQA에서는 4B < 27B로 역전 — Insight 2). adopt + df 기준 가장 강건한 모델은 Qwen2.5-VL family (7b/32b 동률 — Table 2 / §4.1 Insight 1 참조).](../figures/paper_cross_dataset_summary.png)

**Insight 1 (효과의 보편성과 mitigation universality 사전 정당화).** 30/30 cell 부호 양수는 §6.2의 *단일 (L, K, α) hyperparameter가 5/5 dataset에 일반화*한다는 주장의 사전 prerequisite — cell-level 효과가 부호 비일관이라면 단일 cross-dataset hyperparameter가 정의 가능하지 않다.

**Insight 2 (Anti-scaling).** Gemma3-4b가 PlotQA / ChartQA / MathVista 3개 dataset에서 *27B보다 더 끌린다* (PlotQA 0.395 vs 0.227). 그러나 InfoVQA에서는 4B (0.324) < 27B (0.350)로 역전한다 — 따라서 "anti-scaling이 chart/plot/math 3개 dataset에 한정되며 InfoVQA에서는 표준 scaling 회복"이라는 형태로 정확히 표현된다. 이는 *visual reasoning capability gap → 두 번째 이미지 digit 의존*이라는 메커니즘 가설과 일치 — 작은 SigLIP encoder가 차트의 정확한 답을 읽지 못할 때 가시 digit을 단서로 더 강하게 잡는다. 단순한 "큰 모델 = 강건" 직관과 어긋나며, 데이터셋의 *visual complexity*가 모델 크기보다 robustness를 더 결정함을 시사한다.

**Insight 3 (Encoder family별 robustness ordering).** 5-dataset main matrix에서 robustness 순서 (낮은 df 순)는 encoder family와 정렬한다 — Qwen-ViT (7b/32b 모두 강건) > SigLIP-Gemma (27b) > InternViT (8b) > SigLIP-Gemma (4b) > AnyRes-SigLIP (OneVision). *Encoder의 typographic robustness*가 LM backbone 능력보다 anchor robustness를 더 결정한다는 가설을 행동 측 first-evidence로 지지한다 — 단 §5.3에서 보듯 동일 encoder OneVision Main 안에서도 dataset-dependent peak shift가 관측되어 *peak 위치* 차원의 일반화는 fragile하며, 본 ordering은 5-dataset average df 위에서의 cross-encoder pattern으로 한정한다.

### 4.4 Confidence 6-bin (L1) monotonic gradient

§4.1의 wrong-base / correct-base 분할은 더 풍부한 연속 구조의 거친 projection이다. 각 cell의 `target_only` row를 answer-span logit 기반 confidence proxy로 **6개 equal-frequency bin (B1 = 가장 confident, B6 = 가장 uncertain)** 으로 split한 후 bin별 adopt와 df를 계산한다.

**Headline.** 5 dataset {TallyQA, ChartQA, MathVista, PlotQA, InfoVQA} × 6 model heterogeneous-coverage panel의 **80 anchor cell에서 평균 B6 − B1 gap이 df +0.195** (`cross_entropy`, length-invariant paper-clean default) ~ **+0.235** (`log_prob_sum`, length-aware). **5 pair 중 4 pair 이상이 strict ↑인 cell은 cross_entropy 51 / 80 (64 %), log_prob_sum 57 / 80 (71 %)** — 1 bin-pair noise dip을 허용하면 panel 다수가 *substantively monotonic*. *fully strict 5/5 pairs* 기준은 21 / 80 ~ 24 / 80 cell로 더 엄격하게 잡힌다 (Figure 5).

**Proxy 비교.** Legacy `softmax_top1_prob` proxy 도 동일 6-model panel에서 일관된 신호를 보인다 (df B6 − B1 평균 +0.181, ≥ 4/5 strict 46 / 80 (58 %), fully strict 5/5 15 / 80 (19 %)). 본문 headline은 `log_prob_sum`을 보고하고 부록 §B.1에서 세 proxy 비교 표를 제공한다 — 운영적 confidence proxy 선택이 세 정의에 모두 정렬되도록 한 조치이다.

![Figure 5 — L1 6-bin confidence gradient. 가장 자신 있는 B1에서 가장 불확실한 B6로 adopt와 df 모두 monotonic 증가. Worked example PlotQA all-base × LLaVA-OneVision-7b *(Main)* S1 × cross-entropy proxy: df 0.000 → 0.000 → 0.028 → 0.128 → 0.238 → 0.289 (single-cell B6−B1 gap +28.9 pp, B1=B2=0 floor 후 sharp sigmoid rise). Panel-mean B6−B1 gap (80 anchor cell)은 본문 +19.5-23.5 pp.](../figures/paper_L1_confidence_quartile.png)

**Insight 1 (wrong/correct 분할의 재해석).** Phase-A의 wrong-base / correct-base 분할은 confidence 연속체의 *B1+B2+B3 vs B4+B5+B6 projection*이다. 분할 +7.2 pp gap이 6-bin 기준 +28.9 pp로 확대 — 분할은 단순히 *평균*했을 뿐 효과는 본질상 연속 gradient이다. 6-bin은 4-bin Q1=0.024 single floor를 *B1=B2=0 broad floor + sharp B3-B6 rise*로 분해하여 "high-confidence robust regime이 단일 점이 아닌 *broad cohort*"임을 시각적으로 드러낸다 — Insight 2의 categorical-capture 기각을 *floor → sigmoid → saturation* 3-단계 shape로 강화 (4-bin headline에서는 B1+B2 cohort가 단일 Q1에 평균되어 가려졌던 구조). 이는 §6.2 mitigation 설계가 *categorical wrong-base flag*를 별도 입력으로 받지 않고 residual representation 자체에서 universal projection으로 작동하는 것을 정당화한다 (입력 anchor 라벨이 필요 없는 design choice는 본질적으로 *연속 gradient* 가설에 정렬).

**Insight 2 (Categorical capture 기각).** "고불확실성에서 categorical capture"이라면 마지막 bin에서 갑작스러운 점프 — `adopt(B6) >> adopt(B5) ~ adopt(B4) ~ ... ~ adopt(B1)` — 가 보여야 한다. 실제는 *floor → sigmoid → saturation* 3-단계 부드러운 gradient (PlotQA S1 × LLaVA-OneVision-7b *(Main)* × `cross_entropy`: adopt 0.000 → 0.000 → 0.007 → 0.044 → 0.129 → 0.114, df 0.000 → 0.000 → 0.028 → 0.128 → 0.238 → 0.289). 6-bin 정밀도에서 *B1=B2=0 broad floor* (high-confidence cohort 33 % 가 anchor에 영향 받지 않음) → *B3-B5 sigmoid rise* → *B6 saturation* 의 3-단계 shape가 직접 노출되며 — Mussweiler-Strack의 *gradient-blending* 가설과 일치, categorical 가설 기각.

**Insight 3 (Non-monotonic cell의 정성적 분류).** ≥ 4/5 strict pair criterion을 통과하지 못하는 23-29 / 80 cell (29-36 %; cross_entropy 29 / 80, log_prob_sum 23 / 80) 의 정성적 분류는 두 그룹으로 매핑된다: (a) 작은 denominator (E5d ChartQA validation ~30개/bin 수준의 cell — 6-bin이 small-cell에서 noise floor가 다소 높음), (b) near-zero baseline anchor signal로 adopt floor가 noise에 잠긴 cell (qwen2.5-vl-7b TallyQA correct-base 같은 floor cell). 본 분류는 cell-by-cell 정성 검사 결과이며, 각 그룹의 정확한 cell count 보고와 mechanistic exhaustiveness 검증은 §8.2의 follow-up 항목으로 deferred 한다 — 본 절의 headline은 "L1 monotonicity가 monotonicity-supporting cell에서 *일반 경향*으로 유지된다"이며 non-monotonic cell의 origin 분해는 절 외부에서 다룬다. *fully strict 5/5 pair* 기준 (21-24 / 80 cell pass) 은 6-bin 정밀도에서 noise dip 1개도 허용 안 하는 hard criterion으로, ≥ 4/5 relaxed criterion이 본문 headline.

### 4.5 추론은 anchoring을 증폭한다 (auxiliary observation)

Qwen3-VL-8B-Thinking이 Instruct 변형 대비 anchor pull을 amplify한다 — MathVista S1 single-stratum (n=365 paired sids), all-base adopt(a) ×1.6 / df(a) ×2.9. Wrong / correct binary 분할에서 변화가 더 크다: instruct df(a) wrong=0.256 / correct=0.021 → thinking wrong=0.327 / correct=0.267, **correct-base ratio ×12.69, paired-bootstrap 95 % CI [×6.23, ×56.31]** (sid-level resample, arm-conditional base_correct filter, B=10,000, seed=42; instruct correct CI [0.0042, 0.0413] / thinking correct CI [0.2085, 0.3286]; data `docs/insights/_data/qwen3vl_x12_7_paired_ci.{csv,json}`). 하한 ×6.23은 instruct numerator (5 / 238 events) sparsity 때문에 percentile-bootstrap이 right-skewed (74 / 10,000 resample에서 instruct 분자=0 발생, ratio CI에서 제외) 상황에서도 ratio가 1로부터 결정적으로 분리됨을 보여준다. Wrong > correct asymmetry collapse (instruct +0.235 → thinking +0.060) 는 §4.4 framework의 직접 예측 — 연속 confidence axis가 증폭되면 binary projection 평균 차이가 축소되며 (correct-base가 push up하는 방향), Mussweiler-Strack "낮은 baseline confidence → anchor 활성화" 메커니즘이 reasoning trace 연장에 의해 외부 조작 가능함을 시사한다. Thinking은 acc(d)도 더 *낮다* (0.647 → 0.587; b-arm baseline proxy로 d-arm correct fraction 사용) — test-time-compute inverse-scaling [Bae et al., 2025] 및 Wang et al. [2025a] LRM judging-bias amplification과 정렬되며 운영 함의는 *reasoning mode가 robustness 보강이 아니라 anchor backslide 위험원*. 본 결과는 N=1 architecture × N=1 dataset *existence proof*로 thesis의 supporting evidence가 아니라 anchoring의 reasoning-trace 의존성에 대한 *auxiliary observation*; §4.6 residual-stream bridge가 layer-routing 측 prospective leg을 별도 제공한다. Cross-architecture / cross-dataset 일반화는 §8.2 / §8.4.

### 4.6 γ-β residual-stream bridge — framework의 prospective test

**Claim.** §5.4 routing vs integration framework는 anchor 정보가 mid-stack에서 V_K subspace를 *suppress*하다가 late-stack에서 *integrate*하는 layer-routed sign-reversal을 예측한다. 본 절은 framework 작성 *이후* 실행된 Qwen3-VL self-calibration 실험으로 이 예측을 직접 검증한다.

**Setup.** Qwen3-VL-Instruct를 PlotQA + InfoVQA + TallyQA pooled (a − m) wrong-base 잔차 (n=3,017) 위에서 self-calibrate해 V_K[L]을 얻은 뒤, γ-β stimuli (a-S1 anchor + d neutral) 위에서 Qwen3-VL-{Instruct, Thinking}의 per-generated-token 잔차를 사영하고 trace 평균/최대 진폭을 7 layer × 6 K (K ∈ {1, 2, 4, 8, 12, 16}) × 2 statistic = 84 cells L×K sweep으로 측정. Within-Thinking paired (T_a − T_d per sid) bootstrap B = 10,000, Bonferroni-corrected k = 84.

**Evidence — layer-routed sign-reversal confirmed (Table 5).** 14 / 84 cells가 Bonferroni-corrected CI에서 0 제외.

**Table 5.** L × K sweep within-Thinking 대표 cells (전체 84-cell table은 부록).

| layer | K | stat | n | within-Thinking | 95 % CI | Bonferroni 99.94 % CI |
|---|---:|---|---:|---:|---|---|
| **30** | **2** | **max** | 522 | **+0.866** | [+0.412, +1.330] | **[+0.115, +1.643]** |
| 30 | 1 | mean | 522 | +0.477 | [+0.254, +0.695] | [+0.082, +0.852] |
| 29 | 1 | mean | 522 | +0.446 | [+0.252, +0.635] | [+0.123, +0.793] |
| 33 | 1 | mean | 522 | +0.284 | [+0.188, +0.380] | [+0.113, +0.447] |
| 25 | 1 | mean | 522 | +0.213 | [+0.158, +0.270] | [+0.123, +0.314] |
| 20 | 1 | mean | 522 | **−0.152** | [−0.189, −0.116] | [−0.213, −0.094] |
| 20 | 4 | mean | 522 | −0.192 | [−0.232, −0.152] | [−0.269, −0.124] |
| 14 | 1 | mean | 522 | −0.041 | [−0.054, −0.028] | [−0.064, −0.020] |

Late-stack (L = 29, 30, 33) K = 1 mean이 positive (+0.21 ~ +0.48), mid-stack (L = 20) K = 1 / 2 / 4 / 8 mean이 negative (−0.11 ~ −0.19), early-mid (L = 14) 매우 작은 negative — framework가 예측한 mid-stack suppress / late-stack integrate sign-reversal과 직접 일치.

**Cell-selection scope honest note.** Framework prediction은 *방향성-수준* (direction-level) — mid-stack negative ↔ late-stack positive — 이며, 84 cell 중 *어느 (L, K, statistic) 조합이* Bonferroni-clean 할지를 사전 specify하지 *않는다*. 따라서 14/84 surviving cells는 framework의 directional prediction에 *consistent한 cell의 fraction*이지 *pre-registered cell의 verification rate*가 아니다 — single pre-registered cell의 hypothesis test로 framework의 prospective leg을 hardening 하는 것은 §8.4 item 8 (pre-registered §4.6 single-cell run) 으로 명시한다. 본 라운드는 framework의 *directional* prospective verification + *which K* dimensionality partial-falsification 두 측면을 보고하며 cell-level confirmatory test는 후속 작업.

**Insight 1 (Framework의 implicit universal-K 가정 partial falsification).** 동일 L = 33 + 동일 data + K = 1 vs K = 8 비교에서 bridge가 K=8에서 null (point estimate −0.05) 인 반면 K=1에서 Bonferroni-positive (+0.28 [+0.19, +0.38]) 으로 *qualitative sign-state가 변경된다* (K=8 zero-overlap → K=1 Bonferroni-positive) — §6의 K = 8 OneVision sweet spot이 cross-architecture universal이 아니다. Qwen3-VL의 sv7/sv8 elbow는 1.026 gradual decay라 K = 2..7 noise가 K = 1 anchor direction을 dilute. *Layer-routing 방향성은 framework-confirmed; dimensionality 보편성은 framework-partial-falsified* — framework의 falsifiability를 보장하는 핵심 disclosure이며, cross-architecture E6 응용 시 K-sweep이 운영적으로 의무화되는 근거 (§8.2).

**Insight 2 (Quantitative interlock 미해결).** within-Thinking magnitude (+0.5 ~ +0.9 amplitude units, baseline 위 ~0.2 ~ 0.4 % 상대 변화)는 §4.5의 correct-base df ×12.7 큰 폭 증가와 *정량적*으로 정렬되지 않는다 — K = 1 V_K[L=*]는 anchor 처리의 *one aspect*만 capture하며, 다른 잔차 차원·attention pathway·output-head dynamics가 behavioral gap의 대부분을 운반한다. *Qualitative bridge established / quantitative interlock deferred* (§8.2).

---

## 5 메커니즘과 인사이트

### 5.1 Mechanism panel + peak-layer setup

5-model 메커니즘 panel (gemma4-e4b, llava-1.5-7b, ConvLLaVA-7b, qwen2.5-vl-7b, fastvlm-7b) × 200 stratified 자극에서 각 모델의 (text → 두 번째 이미지) attention mass *peak layer*를 calibration dataset (PlotQA; qwen2.5-vl은 PlotQA peak 미측정으로 VQAv2 reference) 위에서 식별해 §5.2 ablation의 표적으로 사용한다. **OneVision Main은 본 panel에서 별도 처리한다** — Plot/Tally L=27 vs Info/VQAv2 L=14의 dataset-dependent peak (§D.1) 으로 인해 *단일 calibration peak으로 panel-level intervention site를 정의할 수 없기* 때문이며, OneVision Main의 5-dataset E1d 확장 결과는 §5.3 본문 + 부록 §D.2의 단일 cluster로 분리 보고. 모델별 peak layer 매핑 + FastVLM·OneVision의 cross-dataset peak variability + E1-patch digit-bbox attention concentration mechanism 측 보조 측정 자세히는 부록 §D.1.

### 5.2 Single-layer ablation null → multi-layer redundancy

§5.1 setup에서 식별한 모델별 peak layer (자세한 매핑 + FastVLM·OneVision dataset-dependent caveat 부록 §D.1) 를 기준으로, 5-model 메커니즘 panel (gemma4-e4b, llava-1.5, ConvLLaVA, qwen2.5-vl, fastvlm) × 200 자극 × 6 ablation mode (`baseline`, `ablate_peak`, `ablate_peak_window`, `ablate_lower_half`, `ablate_upper_half`, `ablate_all`)를 평가했다. Peak는 §D.1의 model별 peak (calibration dataset PlotQA 기준; qwen2.5-vl은 PlotQA peak 미측정으로 VQAv2 L22 reference 사용 — 본 모델 single-layer null 결과가 VQAv2 vs PlotQA layer-mismatch artefact일 가능성은 §D.1 caveat에 명시) — peak 위치를 *피해도* signal이 사라지지 않는지가 목표 질문이다.

- **Single-layer ablation (`ablate_peak`, `ablate_peak_window`)**: 5/5 모델 null.
- **Lower-half ablation**: heterogeneous (2/5 backfire — Gemma +0.27 / LLaVA-1.5 +0.165; 1/5 reduce; 2/5 flat — `E1d-causal-evidence.md`). 본문은 panel-mean ~0으로 보고하나 single-architecture-cluster 일반화 caveat 부록 §D.2 참조.
- **Upper-half ablation**: 5/5 모델 **−4.0 ~ −10.5 pp Δdf** (significant).
- **Full ablation**: −5.0 ~ −12.0 pp.
- **OneVision Main 5-dataset 확장 (n=200 stratified per dataset, B=2,000 bootstrap CI; per-dataset table 부록 §D.2)**: **Single-layer ablation `ablate_peak` / `ablate_peak_window`은 5/5 dataset 모두 null** (TallyQA · InfoVQA · ChartQA · MathVista · PlotQA, max |Δdf| = 1.5 pp on InfoVQA, 모든 95 % CI overlap 0) — multi-layer redundancy claim의 OneVision Main 위 *확장 검증*. 단, **upper-half ablation은 5-mech panel의 균일 −4.0 ~ −10.5 pp significant 와 달리 OneVision에서는 5/5 null at n=200** (point estimates ∈ [−3.9, +0.4] pp; PlotQA −3.9 pp [−9.4, +1.9]가 가장 가깝지만 0 포함) — 이 qualification은 §5.3 OneVision dataset-dependent peak (Plot/Tally L=27 vs Info/VQAv2 L=14)와 일관하며, *5-mech panel calibration 위에서 식별된 upper-half locus가 OneVision에서는 uniform 효과를 산출하지 않는다*는 mechanism-level 사실로서 §6.2의 *subspace projection over attention re-weighting* 도구 선택을 보강한다 (§5.4 routing vs integration framework로 정리).

**Insight 1 (Peak ≠ causal site).** Single-layer null은 메커니즘 해석에 직접적 결과를 가진다. *attention peak이 가장 큰 mass를 가진다*는 사실이 그 layer가 *causal site*임을 의미하지 *않는다*. Signal은 다층에 *redundant*하게 분산되어 어떤 한 layer를 잘라도 다른 layer가 그 부담을 받는다.

**Insight 2 (Single-direction mitigation 실패와의 *사후 일관성*).** Multi-layer redundancy 발견과 §6.4의 single-direction ActAdd cross-dataset 실패 + LEACE rank-1 ChartQA +56 % 역행은 모두 §5.4 framework 작성 이전 관측된 *post-hoc synthesis*의 일부로 §5.4 *routing vs integration framework*와 일관된 *짝 (paired observation)* 이며 (framework-level 해석은 §5.4가 사후 부여; timing 출처 §5.4 본문 + §8.4 참조), framework의 *load-bearing prospective leg*은 §4.6 layer-routing sign-reversal 검증 한 곳이다.

**Insight 3 (Upper-half는 re-weighting 가능).** 동시에 upper-half ablation의 significant 결과는 *upper-half attention pathway 전체가 signal의 일부를 운반*함을 보인다. Peak 한 layer는 causal하지 않지만 upper-half 다층의 *soft re-weighting*은 signal을 줄일 수 있다 — §6.1 E4의 직접 동기.

### 5.3 OneVision Main — dataset-dependent peak

E1d를 OneVision Main에 5개 dataset (TallyQA, InfoVQA, ChartQA, MathVista, PlotQA)으로 확장하면, **peak layer가 dataset-dependent**이다 — Plot/Tally에서 L=27, Info/VQAv2에서 L=14. 동일 encoder에서도 *데이터 분포*가 attention 분배를 바꾼다. 이 fragility는 single-direction mitigation이 cross-dataset에서 실패할 *추가* 이유를 보이며, subspace projection이 *모든 dataset의 signal을 결합 capture*해야 함 (§6.2)을 정당화한다.

E1d ablation 결과 (n=200 stratified per dataset, B=2,000 bootstrap CI; 부록 §D.2 표): single-layer `ablate_peak`는 5/5 dataset null (max |Δdf| = 1.5 pp on InfoVQA; 모든 95 % CI overlap 0) — *§5.2의 multi-layer redundancy 발견이 OneVision Main으로 확장 검증*. Upper-half ablation은 5/5 dataset 모두 95 % CI overlap 0 (point estimates ∈ [−3.9, +0.4] pp) — 5-mech panel의 균일 −4.0 ~ −10.5 pp significant와 *명확히 다른* heterogeneous pattern으로, peak가 datasets 간 L=27 ↔ L=14 사이에서 이동하므로 *upper-half라는 layer-band 내 평균* 자체가 dataset마다 다른 layer set을 capture하기 때문이다. Lower-half ablation은 MathVista에서 +7.5 pp [+1.6, +13.6] significant BACKFIRE (TallyQA boundary +5.0 pp [+0.0, +10.5]) — 5-mech panel의 2/5 backfire와 동일 heterogeneity pattern. 본 모든 결과는 *layer-level 단일 intervention이 OneVision의 cross-dataset signal을 capture하기 어렵다*는 단일 finding으로 수렴하며, §5.4가 §5.2 multi-layer redundancy와 본 절의 OneVision peak fragility를 단일 framework로 통합해 §6.1 / §6.2의 두 mitigation 도구 선택을 사전 예측한다.

### 5.4 Routing vs integration site framework — 사후 synthesis와 prospective 검증

§5.2의 multi-layer redundancy 결과 (5-mech panel single-layer ablation 5/5 null + upper-half multi-layer ablation 5/5 significant) + §5.3의 OneVision Main dataset-dependent peak (Plot/Tally L=27, Info/VQAv2 L=14) + §6.4의 LEACE rank-1 ChartQA +56 % 역행 — 이 세 mechanism finding은 본 framework 작성 *이전*에 모두 관찰되었으며, 본 절은 이 세 결과를 *사후*에 단일 mechanism narrative로 묶는 **synthesis**이다. Framework의 prospective test는 별도 — §4.6 γ-β residual-stream bridge가 framework 작성 이후 수행되어 layer-routing 방향성 예측을 직접 검증한다 (출처 timing은 §8.4 참조).

**Framework 정의.** Multi-layer redundancy는 *attention pathway*의 속성이다 — *residual stream*에서도 그런 것은 아니다. §5.2 E1d 결과는 어떤 단일 attention layer를 마스킹해도 다른 layer가 anchor 신호를 routing해 forward한다는 것이며, 이는 *routing layer*들이 redundant함을 의미한다. 이 redundant routing의 결과로 anchor 정보는 점진적으로 *residual stream에 누적*된다. 충분히 후반 layer에 도달하면 분산되어 있던 multi-layer attention 기여가 잔차 표현 안에 *통합된 형태*로 자리 잡으며, 이 통합 표현은 (1) low-dim이며, (2) single-layer 단일 site에서 접근 가능하다. *Attention layer는 routing site, residual stream의 후반 layer는 integration site*인 것이다.

**Framework 정리.** §5.4 framework는 네 mechanism-level 결정을 단일 narrative로 묶는다 — (i) routing site mitigation (mid-stack cluster attention pathway, E4) 과 integration site mitigation (OneVision residual stream, E6) 의 *두 상보적 site*, (ii) single-direction mitigation의 cross-dataset 실패는 multi-layer redundancy로부터 *통합 설명*되며 multi-direction subspace가 그 우회로, (iii) 단일 layer 개입이 강제될 경우 *통합이 일어난 late residual band* — 28-layer Qwen2 backbone에서는 L=26 부근의 *band* — 가 합당한 candidate. 단, 본 framework는 **band-level** 예측에 그친다 — 단일 L=26 값은 §6.2의 27-cell pilot grid가 L ∈ {25, 26, 27} 안에서 *empirical*하게 산출한 결과이며, framework는 그 band를 narrowing할 뿐 cell을 narrow하지 않는다. (iv) 도구 선택 측면에서 broad ablation은 잔차 정보 무차별 0화로 일반 능력 손상 위험을 가지지만 *subspace projection*은 K-dim만 제거하므로 그 외 분산이 보존된다 — 이 도구 선택의 정당성은 §7 capability preservation 결과 (HallusionBench excludes zero, POPE pinned to zero)에서 외부 검증된다.

**Framework의 partially prospective test (§4.6).** §5.2 + §5.3 + §6.4 mechanism observation을 사후 통합한 synthesis 위에 framework의 직접 layer-routing 예측 — anchor 정보가 mid-stack에서 V_K subspace를 *suppress*하다가 late-stack에서 *integrate*하는 sign-reversal — 을 §4.6 γ-β residual-stream bridge가 K=1 cell에서 *partially prospective* 검증한다 (framework 작성 이후 실행; cell-level confirmatory pre-registration은 §8.4 item 8). Qwen3-VL self-calibrated K=1 subspace에서 within-Thinking paired Δ가 mid-stack (L=20) negative + late-stack (L=29-34) positive sign-reversal로 14/84 Bonferroni-corrected cells에서 0 제외. 그러나 framework의 implicit *universal K=8 sweet spot* 가정은 부분 falsify — Qwen3-VL은 sv7/sv8 elbow가 1.026 gradual decay라 K=2..7 noise가 K=1 anchor direction을 dilute, 동일 L=33에서 K=1 vs K=8 ratio가 9× — *layer-routing 방향성은 framework-confirmed at K=1, dimensionality 보편성은 framework-partial-falsified at deploy K=8*. 본 honest disclosure는 framework의 falsifiability를 보장하는 핵심 element이다 (§4.6 Insight 2 + §8.2 한계).

§6은 두 mitigation site를 차례로 검증한다 (§6.1 E4 routing site, §6.2 E6 integration site), §6.4가 single-direction cross-dataset failure를 다루며, §7이 capability preservation 외부 검증을 한다.

---

## 6 Mitigation과 인사이트

### 6.1 E4 — Attention pathway re-weighting (mechanism demo)

§5.4 Prediction 1의 *routing site mitigation*을 직접 검증한다 — §5.2 5-mech panel에서 upper-half ablation이 5/5 significant라는 사실 위에 *soft re-weighting*을 얹는 form. a-arm에서 두 번째 이미지 (anchor)에 대한 text → image attention weight를 강도 `s`로 곱한다 (`s` ∈ {0.0, 0.25, ..., 1.5}). Mid-stack cluster 2 모델 (LLaVA-1.5 / ConvLLaVA) Phase 2 full validation (17,730 base question × 5 cond, 모델당 88,650 records).

**Table 6.** E4 Phase 2 결과, mid-stack cluster 2 모델 (LLaVA-1.5 / ConvLLaVA), 88,650 records / 모델. Δdf 상대 = (df_mit − df_base) / df_base; Δem(a)는 a-arm exact-match. acc(b)·acc(d)는 mitigation 외부 column으로 free-lunch 검증. Bold = 열 단위 가장 큰 효과.

| 모델 | Δdf 상대 | Δem(a) | acc(b) Δ | acc(d) Δ |
|---|---:|---:|---:|---:|
| **LLaVA-1.5-7b** | **−14.6 %** | +0.77 pp | 불변 | ±0.5 pp |
| **ConvLLaVA-7b** | −9.6 % | **+1.30 pp** | 불변 | ±0.5 pp |

**Insight 1 (Free-lunch의 메커니즘적 의미).** acc(b) target-only 불변 + acc(d) neutral arm ±0.5 pp 안 + em(a) 양수 — hook은 single-image inference에 누출되지 않으며, 두 번째 이미지에 가독 digit이 없는 경우 *hook은 트리거되지만* 제거할 signal이 없다. 즉 *upper-half pathway가 모델 자체 답안 형성에 non-load-bearing*이며, 그 부담을 줄이면 anchor 영향만 제거된다. Table 6의 per-column bold는 이 분리를 그대로 반영한다 — LLaVA-1.5는 Δdf 측에서 가장 큰 효과 (−14.6 %), ConvLLaVA-7b는 Δem(a) 측에서 가장 큰 회복 (+1.30 pp).

**Insight 2 (E4의 non-triviality — "anchor를 줄였으니 당연히 anchor 효과 감소" 반박).** "Attention을 anchor에 줄이면 anchor 효과가 당연히 떨어지는 것 아닌가" 라는 의문이 자연스럽다. 본 결과의 비-trivial성은 *세 측면*에서 입증된다.

(i) **Layer-restrictedness — lower-half는 anchor effect에 기여하지 않고 upper-half만이 active load-bearing.** §5.2 E1d ablation table (mech panel 5 모델, n=200 stratified per model)이 직접 보여준다 — `ablate_lower_half`는 5/5 모델에서 Δdf ≈ 0 (lower-half attention to anchor를 *전부* 제거해도 anchor effect 변화 없음); `ablate_peak` 단일 layer 마스킹도 5/5 null. 반면 `ablate_upper_half` 단독으로 **−4.0 ~ −10.5 pp Δdf** (5/5 통계적 유의), 이는 `ablate_all` (전 layer × 전 token 제거) 의 −5.0 ~ −12.0 pp Δdf의 *80-90 %* 수준이다. *Lower-half가 0 기여하고 upper-half가 거의 전부 운반*하는 명확한 비대칭은 anchor signal이 upper-half attention pathway에 *국소화*되어 있다는 mechanism-level finding이며, "anchor에 attention을 줄였으니 당연" 이라는 반론의 *정밀화* — anchor에 attention을 줄여도 lower-half에서 줄이면 효과 0, upper-half에서 줄여야만 효과 발생.

(ii) **Arm-selectivity — broadly suppressing이 아닌 selectively-firing.** Hook은 모든 forward pass에서 트리거되지만, target-only `acc(b)` 불변 + neutral arm `acc(d)` ±0.5 pp는 *제거할 anchor signal이 없으면 hook이 답안에 영향 안 줌*을 보인다. anchor arm에서만 effect 발생 — broadly attention을 깎는 것이라면 모든 arm에서 정확도 저하가 보였을 것.

(iii) **Non-tautological em rise — em이 단순 invariant가 아닌 *상승*.** Anchor 효과를 줄이는 어떤 개입이든 df는 줄어들 것이다 (input ablation의 trivial 한계). 그러나 `em(a)`가 +0.49 ~ +1.30 pp *상승*하는 것은 단순 input 차단으로 설명 안 됨 — 모델이 이미지 정보를 단순히 *덜 보는* 결과라면 em은 그대로거나 떨어졌을 것. em 상승은 *upper-half anchor-attention pathway가 anchor-arm em을 능동적으로 억제하고 있었다*는 직접 측정이며, "fluency-clean removable channel"이라는 mechanism-level claim의 핵심 증거.

종합하여 E4의 결과는 *input gating의 trivial limit*이 아니라 *upper-half attention pathway가 anchor signal을 redundantly route하는 active load-bearing site이고 lower-half는 그렇지 않다*는 mechanism finding이다. 단, E4는 inference 시 anchor token span을 요구 — adversarial 환경에서 그것은 정확히 *방어 대상* 정보이므로 — **deployable mitigation이 아닌 mechanism diagnostic**이다. 본 논문의 deployable claim은 §6.2의 E6 — input의 anchor 위치를 모르고 universal projection으로 작동 + anchor label 무관 + cross-arm 모두 적용 — 에서 별도 검증한다.

### 6.2 E6 — Residual-stream subspace projection (deployable)

§5.4의 routing-vs-integration framework가 본 절의 설계 선택을 사전 예측한다 — multi-layer attention routing을 모두 개입하는 것은 비현실적이지만, 그 routing의 결과가 누적되는 *residual integration site*에서 단일 layer 개입이 가능하다 (Prediction 1, integration site mitigation). 본 절은 이 framework의 구체화 — late residual layer L=26 (Prediction 3), K=8 부분공간 (Prediction 2의 multi-direction), projection (ablation 아닌; Prediction 4) — 으로, 네 framework 결정이 본 절의 네 hyperparameter 선택과 1:1 대응한다.

#### 6.2.1 방법

`h(x, L) ∈ R^d`를 input `x` 위에서 layer L의 마지막 input token residual로 둔다. Calibration set의 각 wrong-base sample i에 대해 (a-arm, m-arm) hidden-state pair를 capture해 difference matrix

```
D[i, L, :] = h(x_i^a, L) − h(x_i^m, L)
```

을 구성하고 N×L×d로 stack한다. Layer별 truncated SVD `D[:, L, :] = U_L Σ_L V_L^T`에서 top-K right singular vector `V_K[L] ∈ R^{K × d}`를 retain한다 (K-dim subspace = anchor 효과의 principal direction). Inference 시 선택된 layer L*에서 1회 residual projection:

```
h'(x, L*) = h(x, L*) − α · V_K[L*] V_K[L*]^T h(x, L*)
```

Projection은 *universal*이다 — 어떤 input 정보도 anchor-present를 anchor-absent와 구별하지 않는다. K-dim anchor subspace를 모든 forward pass에서 silently 제거한다.

**Insight ((a − m) contrast의 핵심).** 핵심 설계 결정은 `D = h^a − h^m`이지 `h^a − h^b` 또는 `h^a − h^d`가 아니라는 점이다. (a − m) contrast는 *같은 anchor scene*에서 *digit pixel만 제거*한 pair이므로 generic distraction은 자동 차감되고 결과 subspace가 *digit pixel 기여에 specific*하다. (a − b) 또는 (a − d)를 사용하면 일반 distraction signal이 섞여 K가 커야 충분 cover되고 non-anchor 분산까지 침식된다. *§4.2의 (b, m, d) 통제 실험이 §6.2 subspace 설계를 직접 정당화*한다 — behavioral analysis가 mitigation 설계로 환원된 사례. 일반 design pattern으로 표현하면: **calibration contrast는 인과 통로 (causal pathway) 를 confounding variance로부터 *분리*하는 paired difference여야 한다** — 본 사례에서는 (digit pixel → answer shift) 통로를 (anchor scene background → general distraction) confound로부터 분리하기 위해 (a − m) paired-inpaint이 그 분리 구조를 정확히 제공한다. **Telea-residue caveat.** (a − m) calibration substrate가 isolating 하는 것은 엄밀히는 *digit-pixel-or-Telea-residue-correlated* directions이다 — Telea inpaint는 픽셀 absence를 *OCR로* 검증했으나 (§3.1, §A.2), representation-level texture residue (frequency-domain artefact, color-bleeding around inpaint boundary, edge artefact 등) 가 0이라는 *control은 수행되지 않았다*. 따라서 §6.2 SVD는 digit-pixel 인과 통로 + Telea-residue texture direction을 *함께* capture할 가능성을 배제할 수 없다. 직접 falsification baseline은 (m − m') inpaint-noise-only SVD — 같은 scene의 두 독립 inpaint pass에서 도출한 K=8 subspace와 (a − m) subspace의 cosine similarity 비교 — 이며 §8.4 item 7에 명시한다.

#### 6.2.2 Calibration + hyperparameter grid

Main 모델 `llava-onevision-qwen2-7b-ov`을 **PlotQA + InfoVQA pooled** wrong-base set (N=5,000)으로 calibrate. 두 calibration dataset이 evaluation dataset 전체 GT 분포 (최대 ~1,000)를 결합 cover한다.

(L*, K, α) triple은 27-cell pilot grid (L ∈ {25, 26, 27} × K ∈ {2, 4, 8} × α ∈ {0.5, 1.0, 2.0})에서 선택한다. **선택 규칙은 calibration set 위에서 사전 (ex ante) 고정**: 어느 calibration dataset (PlotQA pilot n = 250 / InfoVQA pilot n = 250)에서든 Δem(a) ≤ −6 pp인 cell 거부 (em-deal-breaker), 잔존 cell을 결합 |Δdf(a)| 감소량으로 정렬. 이 규칙은 5-dataset evaluation의 *어떤 결과도 관찰하기 전에* 결정되었으므로 §6.2.3 표는 *held-out evaluation*의 위치를 갖는다.

**선택 cell: L* = 26, K = 8, α = 1.0** (27-cell 중 cell #17, §A.5). 27-cell 중 *어느 cell도* −6 pp 임계값을 위반하지 않아 deal-breaker 절은 본 grid 위에서 non-binding이며, 결합 |Δdf(a)| 정렬에서 chosen cell #17이 mean Δdf(a) = −4.4 pp로 1위 (2위 #8 −3.2 pp 대비 1.2 pp 격차). Calib n = 250 위 paired SE ~1.3 pp로 #17 ↔ #8 격차는 within ~1 SE 범위 — ranking는 동일 ex ante 규칙 재실행 시 산출되나 첫 SE 안에서 ordering 교체 가능성은 honest disclosure로 surface한다 (per-cell 4-metric heatmap 부록 §A.5).

#### 6.2.3 5-dataset cross-evaluation (paper headline)

선택 cell 동결 + 추가 tuning 없이 5개 dataset (full GT range, dataset당 n=5,000 wrong-base)에서 paired-sids deltas로 평가 (Figure 7, Table 7).

![Figure 7 — E6 Stage-4 (L = 26, K = 8, α = 1.0) 5-dataset paired-bootstrap deltas, 두 headline clause를 한 panel에 carry. (좌) Δdf(a) anchoring-effect clause — PlotQA n=2,306만 95 % CI excludes 0 ([−6.9, −3.4]); 나머지 4 small-n cell은 점추정 음 부호 일관이지만 CI가 0을 포함 (CI half-width sample-size에 비례). (우) Δem(b) non-anchored arm capability clause — 5/5 cell 95 % CI excludes 0 (Bonferroni-20 후에도 동일, 부록 §A.5). Row는 `n_paired` 내림차순; 강조색 = headline 방향으로 0 제외, 회색 = CI가 0 포함. Source `docs/insights/_data/stage4_final_per_dataset_ci.csv`, builder `scripts/build_paper_stage4_paired_ci_figure.py`.](../figures/paper_6_2_3_stage4_5dataset_paired_ci.png)

**Table 7.** E6 Stage 4-final, paired-sids paired wrong-base deltas with paired-bootstrap 95 % CI (B = 10,000, sid 단위 paired resampling, per-arm denominator/numerator 매 resample 재계산). Bold = 95 % CI excludes 0 in headline direction. 추가 Bonferroni-20 corrected (99.75 %) CI는 부록 §A.5 reproducibility.

| 데이터셋 | n_paired | Δ adopt(a) [95 % CI] | Δ df(a) [95 % CI] | Δ em(a) [95 % CI] | Δ em(b) [95 % CI] |
|---|---:|---:|---:|---:|---:|
| TallyQA | 4,978 | −0.6 [−1.1, +0.0] | −0.3 [−1.3, +0.6] | **+6.6 [+5.6, +7.5]** | **+13.8 [+12.9, +14.8]** |
| PlotQA | 2,306 | **−5.6 [−6.8, −4.4]** | **−5.2 [−6.9, −3.4]** | **+2.4 [+1.5, +3.4]** | **+4.7 [+3.8, +5.7]** |
| InfoVQA | 443 | +0.9 [−0.5, +2.5] | −0.7 [−4.7, +3.4] | **+3.4 [+0.5, +6.3]** | **+9.0 [+6.3, +11.7]** |
| ChartQA | 224 | **−3.3 [−6.0, −1.0]** | −4.0 [−9.8, +1.8] | **+4.0 [+0.0, +8.0]** | **+7.1 [+3.6, +10.7]** |
| MathVista | 170 | −1.5 [−6.9, +3.7] | −4.1 [−11.8, +3.5] | +2.9 [−2.4, +8.2] | **+9.4 [+4.7, +14.7]** |
| **평균** |   | **−2.0** | **−2.9** | **+3.9** | **+8.8** |

**Sign-clean count (CI excludes 0 in metric의 headline 방향):**

| Metric | 95 % CI | Bonferroni-20 (99.75 %) CI |
|---|:---:|:---:|
| Δ adopt(a) (− 방향) | 2 / 5 | 2 / 5 |
| Δ df(a) (− 방향) | 1 / 5 (PlotQA) | 1 / 5 (PlotQA) |
| Δ em(a) (+ 방향) | 3 / 5 | 2 / 5 (PlotQA, TallyQA) |
| **Δ em(b)** (+ 방향) | **5 / 5** | **5 / 5** |

**Δem(b)는 본 mitigation의 multiplicity-robust headline이다 — 5/5 cell에서 95 % 및 Bonferroni-20 (99.75 %) CI 모두 excludes 0.** Δdf(a)는 sample-size에 묶여 있다: PlotQA n=2,306만 95 % CI excludes 0 ([−6.9, −3.4]); 4 small-n cell은 점추정 부호 일관-CI-individually-inconclusive (ChartQA · MathVista CI half-width 5–8 pp; InfoVQA n=443 [−4.7, +3.4] fence; TallyQA baseline df floor). Δadopt(a) 부호 일관, Δem(a) 5/5 양 arm 모두 양 (3/5 cell 95 % CI excludes 0). Per-dataset paired-bootstrap CI 표는 부록 §A.5.

**Multiplicity-correction scope honest note.** 본 표의 Bonferroni-20 보정은 *선택 cell이 사전 등록된 (pre-registered) 조건* 하에 5 dataset × 4 metric = 20 paired-test family 위에서 strict하다. 그러나 §6.2.2의 27-cell pilot grid argmax 자체가 별도의 multiplicity 계층 — 27-fold cell selection — 을 형성한다 (§A.5 deal-breaker 규칙은 grid 상 non-binding). 27 × 20 = 540 family에 대한 strict Bonferroni 적용은 본 paper 표에 박지 않았으며 그 이유는 *empirical* 이다 — B = 100,000 paired-bootstrap + parametric normal CI 비교 diagnostic (부록 §A.5) 에서 small-n datasets (ChartQA n = 224, MathVista n = 170) 의 Bonferroni-540 lower bound가 *1/n empirical discretization floor* (ChartQA Δem(b) LO = +0.89 pp = 2/n; MathVista Δem(b) LO = +0.59 pp = 1/n) 에 박힌다. Parametric normal-approximation 99.99 % CI (`Δ ± 3.91 · SE`)는 동일 두 cell에서 0을 포함 (ChartQA LO = −0.05 pp, MathVista LO = −0.50 pp) 하여 bootstrap tail이 statistical 한계가 아니라 sample-size 한계에 도달했음을 알린다. PlotQA · TallyQA · InfoVQA (n ≥ 443) 에서는 bootstrap과 parametric 두 방법이 같은 부호로 일치하여 Δem(b) 3/5 cell + Δdf(a) PlotQA single cell이 540-family 하에서도 0을 제외한다. 즉 **27-cell selection을 포함한 strict 540-family correction은 large-n cell 에서 informative하나 small-n cell 에서는 noninformative** — 이 한계는 sample size에 기인하며 추가 bootstrap 재계산으로 해소되지 않는다. 본 paper는 Bonferroni-20 헤드라인 (Δem(b) 5/5, PlotQA Δdf 1/5) 을 유지하고 27-cell selection layer는 본 prose disclosure 및 §8.4 item 8에 명시한다.

**Free-lunch criterion 형식 정의 (4-clause).** 본 논문이 채택하는 *free-lunch* 기준은 *4-clause 동시 충족*을 요구한다 — *Δdf(anchoring task)* < 0 ∧ *Δem(anchored arm)* ≥ 0 ∧ *Δem(non-anchored arm)* ≥ 0 ∧ *Δ(held-out capability macro)* ≥ −0.5 pp (사전등록, §7). 본 기준은 통상의 Pareto-improvement 표현 (첫 번째 + 마지막 두 clause) 위에 *non-anchored arm em* 조항을 추가한다. 이 추가 clause는 bias mitigation의 *cross-category collateral damage* — Chand et al. [2025]가 LM debiasing에서 보고한 *31.5 % 비표적 dimension에서의 부수 손상* — 에 직접 대응하는 screening 기준으로, *anchoring task family를 벗어난 forward에서 mitigation이 representation을 손상시키지 않는다*는 측면을 *경험적으로 강제*한다 (vs Chand et al.의 negative result는 LM × 이산 social bias × weight space에서 4-clause 동시 충족이 성립하지 않음을 보임; 본 논문은 VLM × 연속 numerical regression × inference-time activation projection에서 *4-clause 동시 충족이 가능*함을 보고). 이후 절 (§6.5 비교 표 / §7 capability preservation / §8.1 종합)은 이 4-clause 기준을 일관 적용한다.

**Insight 1 (Effect size correlates with baseline).** Projection이 *dataset-shared subspace를 amplitude-dependent*하게 청소한다 — Δdf 감소량이 PlotQA (−5.2 pp, 가장 큰 baseline df)에서 가장 크고 TallyQA (−0.3 pp, df 거의 floor)에서 가장 작다는 ordering이 가설의 직접 시험. 단일 보정으로 효과 크기를 *예측 가능*하게 만든다는 운영 함의.

**Insight 2 (단일 hyperparameter의 의미).** E4가 모델당 `s*` tuning을 한 자릿수 차이로 필요로 한 데 비해 E6는 *단일 (L, K, α)*가 5 dataset에 일반화된다. 이는 (a − m) subspace가 *dataset 간 shared variance direction*을 capture함을 메커니즘적으로 뒷받침 — §5.3의 dataset-dependent attention peak에도 불구하고 *residual-stream representation 측에서는 shared axis가 존재*한다.

### 6.3 왜 non-anchored arm에서도 em이 오르는가

b-arm은 `target_only` — 단일 이미지 + 질문, 두 번째 anchor 이미지 *없음*. Projection은 anchor 유무와 무관하게 모든 forward에서 작동한다 (universal projection — input이 anchor를 포함하는지 알지 못함). 첫 해석 — "그러므로 projection은 *오직* anchor 제거 연산일 *수 없다*" — 은 옳다.

(a − m) calibration contrast는 wrong-base 부분집합에서 *digit-anchor arm을 visual-matched no-digit control과 구별하는 모든 variance direction*을 capture한다. 정의상 digit의 representational signature가 포함되지만, *wrong-base의 digit-anchor failure와 co-aligned된 어떤 error mode*도 함께 포함된다. K = 8 leading subspace projection은 둘 다 제거하며, b-arm em 이득은 그 방향들 중 일부가 *target_only arm에서 amplitude가 em을 억제하던 generic wrong-base error mode*를 운반했음을 드러낸다.

**Insight 1 (두 효과의 분리).** 선택 cell은 두 가지를 동시에 한다 — (a) a-arm anchor pull 감소, (b) b-arm wrong-base error mode의 우연한 debiasing. 두 효과는 같은 intervention을 공유하지만 (a)만이 *유일하게 anchor-targeted*이다. 본 논문은 (b)를 "anchor mitigation"으로 *주장하지 않는다* — deployable mitigation의 *경험적 부수효과*로 보고한다.

**Insight 1.5 (대안 설명과 검증되지 않은 가설들).** 위의 "wrong-base error mode 제거" 해석은 b-arm em +8.8 pp 결과의 한 후보 설명이지만 *유일한* 설명은 아니며, 본 논문은 이를 다음 두 대안과 head-to-head로 비교하지 *않았다*. (Alt-1) **General regularization.** 28-layer × d=4096의 residual stream에서 K=8 dim subspace 제거 (K/d ≈ 0.002)는 mild regularizer로 동작해 답안 token logit 분포를 *modal correct digit* 쪽으로 편향시킬 수 있다. 이 가설은 *random-K=8 subspace* (anchor-free non-anchor calibration set 위의 SVD 또는 random orthogonal projection)을 L=26에 동일 적용한 baseline에서 b-arm em 이득이 재현되는지로 falsify된다 — 본 baseline은 §8.2 deferred. (Alt-2) **Numeric mode-collapse.** L=26 K=8 hook이 답안 token 분포를 dataset-modal answer (counting / chart task에서 자주 0 / 1 / 2 / 3) 쪽으로 collapse시켜 *low-information* 질문에서 우연한 정답 일치 비율이 증가했을 가능성. 이는 b-arm em 이득의 GT-mode-frequency tertile 분해 (high-mode-coincidence 그룹에서 이득 집중 vs 균등 분포)로 falsify되며, 동일하게 §8.2 deferred. **본 round 내부 신호.** §7 POPE Δ=−0.06 pp 95 % CI [−0.21, +0.09] pinned-to-zero 결과는 *yes/no answer-distribution shift* 형태의 generic mode-collapse를 사전 신호로 부정 — POPE는 이항 yes/no benchmark이므로 modal-class 쪽 mass 증가가 있었다면 직접 검출되었을 것. 이 사전 신호는 (Alt-1) yes/no general-regularization을 어느 정도 약화시키나, *numeric* token logit 위에서의 mode collapse (Alt-2) 또는 anchor-task-specific subspace 정렬 (본 가설) 중 어느 것인지를 분리하지 *않는다*. 따라서 §6.3 본문 해석은 *consistent with* (Alt-1 가설 약화 + 본 가설과 일관) 수준으로 hedged되며, 결정적 mechanism 분리는 deferred.

**Insight 2 (Capability preservation의 사전 신호).** b-arm em 상승이 "target_only에서 anchor 없이도 모델이 더 정확해진다"는 사실은 §7 (E8)의 capability preservation이 *실패할 가능성이 낮음*과 *consistent*이다 — 만일 projection이 일반 representation을 *손상*시킨다면 b-arm em이 떨어졌을 것이라는 사후 일관성 (post-hoc consistency) 형태의 신호이며, 위 Insight 1.5의 random-K = 8 baseline 비교 없이는 *예측*이라고 부르기보다는 *상호 보강 (mutual support)*에 가깝다 (random-K = 8 falsification baseline은 §8.4 item 2 명시). §7에서 6-benchmark 매크로 +0.41 pp + HallusionBench +2.21 pp + POPE −0.06 pp pinned-to-zero가 관측된다.

### 6.4 왜 subspace이고 single direction이 아닌가

자연스러운 baseline은 single mean-anchor direction `v[L] = mean_i D[i, L, :]`. 두 단일-방향 방법 (ActAdd + LEACE) 모두 cross-dataset *실패*: ActAdd는 TallyQA-calibrated `v` self-test 자체가 α=1에서 backfire; LEACE [Belrose et al., 2023]를 *rank-1 closed-form 인스턴스*로 calibrate하면 (LEACE framework 자체는 closed-form linear concept erasure이며, rank-1은 본 비교에서 사용한 default 운영 모드) gt ∈ [0,8]로 제한해도 ChartQA에서 direction-follow를 +56 % *증가*시킨다. Per-dataset mean-anchor direction은 측정 가능하게 다른 곳을 가리킨다 — top-norm layer에서 `cos(v_tally, v_chartqa) ≈ 0.47-0.62`.

**Insight 1 (Single-direction failure와 multi-layer redundancy의 *사후 일관성*).** §6.4의 single-direction ActAdd backfire + LEACE rank-1 ChartQA +56 % 역행은 §5.2 multi-layer redundancy 결과와 함께 §5.4 *routing vs integration framework* 가 통합 설명하는 *두 관찰* 이다 — 둘 모두 framework 작성 *이전*에 관찰되었으며 (§5.4), 본 절은 §6.4를 §5.2의 prediction verification으로 *주장하지 않는다*. Framework의 사후 부여 해석은 dataset이 다르면 signal이 다른 layer 조합에 분산되어 single direction이 cross-dataset alignment를 잃는다는 것이며, 이 해석이 subspace projection으로의 도구 선택을 (사후) 정당화한다.

**Insight 2 (K=8의 trade-off).** 더 작은 K (2, 4)는 anchor signal을 더 누출, 더 큰 K (16)는 non-anchor variance를 제거 시작해 em 손상. K=8은 *shared variance를 충분히 cover하면서 non-anchor variance는 보존*하는 sweet spot.

### 6.5 부정적 결과 비교 — 5-baseline panel 위 4-clause free-lunch 통과 후보

**Table 8.** Multi-method 비교, LLaVA-OneVision Main + L=26 K=8 hook 단일 model run. "Cross-dataset 감소"는 5-dataset Δdf 요약 (본 작업 −0.3 ~ −5.2 pp on 5/5; baseline은 method 출처의 cross-dataset failure 라벨). Bold = 본 작업. *Note: 각 baseline은 method-source default 운영 모드로 평가되어 E6의 27-cell pilot grid 대비 *tuning effort가 비대칭*하다 (Insight 본문 Note 2 참조); fair-tuning 후 비교 + CAA / ITI empirical row는 §8.4 item 4 deferred.*

| 방법 | Cross-dataset 감소 | em on a-arm | em on b-arm | 판정 |
|---|---|---|---|---|
| Single-direction ActAdd | ❌ Cross-dataset 실패 (TallyQA-cal v → ChartQA self-test backfire α=1) | 불변 | 불변 | direction mismatch |
| LEACE closed-form (rank-1) | ❌ ChartQA backfire +56 % (gt ∈ [0,8]) | 불변 | 불변 | single-direction redundancy |
| Query-adaptive offset (PCA + Ridge) | ❌ 1/4 임계 미달 | 불변 | 불변 | probe overfit |
| CogBias decode-time | ❌ 1/4 임계 미달 | 불변 | 불변 | decode-time single-direction |
| MIA-DPO LoRA (weight space) | 부분적 df 감소 | **−5.85 pp on VQAv2** | 미보고 | em side-effect + training distribution bias |
| **Multi-direction subspace (이 작업, K=8)** | **−0.3 ~ −5.2 pp on 5/5** | **+2.4 ~ +6.6 pp** | **+4.7 ~ +13.8 pp** | **권장 — 4-clause 동시 충족 (이 5-baseline panel 위)** |

**Insight.** Single-direction 방법은 (i) *direction mismatch failure mode*에서 막히고, weight-space 방법 (MIA-DPO LoRA)은 (ii) *em 부수효과 + 학습 분포 편향*에서 막힌다. Multi-direction residual projection은 *이 5-baseline panel 위에서* 두 failure mode를 동시에 우회하는 유일한 cell이다 — *dataset 간 shared variance direction* + *inference 시 weight 보존*이 동시 충족된다. 이는 우연이 아니라 §5.2 multi-layer redundancy와 §6.2.1 (a − m) contrast 설계의 직접 결과*이다*.

*Note 1 (CAA / ITI baseline 처리는 구조적 reduction이며 empirical row는 deferred).* CAA [Panickssery et al., 2024]는 paired-contrast residual-stream steering을 *rank-1*으로 가중 합산하므로, *동일 (a − m) calibration set 위에 적용한 CAA-at-K=1 인스턴스*는 본 비교의 ActAdd 행과 구조적으로 동치이며 cross-dataset α=1 self-test backfire를 동일하게 상속하리라 *예측*된다. ITI [Li et al., 2023]는 *attention-head 출력*에서 multi-head × multi-direction 개입을 수행하므로 §5.2의 single-head single-layer null이 직접 cover하지 *않는다* — ITI는 multi-head cluster intervention이고 §5.2는 single-head ablation이다. 다만 §5.3의 OneVision dataset-dependent attention peak (Plot/Tally L=27 vs Info/VQAv2 L=14) 는 *attention-locus* 자체가 dataset 간 이동함을 보이므로, fixed-locus head-cluster 개입이 cross-dataset 일반화에 도전을 받을 *방향성*은 §5.2 + §5.3 조합에서 예측된다. 두 reduction은 *empirical*이 아닌 *구조적 추론*이며, CAA-at-K=1 + ITI multi-head empirical row는 §8.4 item 4로 후속 작업에 deferred — 따라서 본 표는 "*우리가 평가한 5-baseline panel 위에서* 4-clause free-lunch를 동시 충족하는 유일 cell"이라는 한정 statement으로 읽혀야 하며, 두 prior method를 panel-wide tuning effort 동등 수준으로 calibrate한 후의 비교 결과는 본 round에서 보고하지 않는다.

*Note 2 (baseline tuning effort asymmetry).* 본 5-baseline panel은 ActAdd / LEACE rank-1 / Query-adaptive offset / CogBias / MIA-DPO 각각을 method-source 문헌의 default 운영 모드로 평가했으며, E6는 27-cell pilot grid (§6.2.2) 위 ex ante selection rule을 거쳤다 — *tuning effort가 baseline 측에 비대칭하게 작다*. fair-tuning이 적용된다면 (i) ActAdd / LEACE rank-1의 cross-dataset failure quadrant는 *direction-mismatch 본질에서* 유래하므로 K-sweep으로 회복될 가능성이 낮으나 layer-sweep / α-sweep으로 일부 cell이 부호 회복할 수 있고, (ii) ITI multi-head empirical row가 attention-head locus K-sweep을 거치면 §5.3의 dataset-dependent peak이 cross-dataset 부분 평균으로 흡수되어 4-clause partial-pass가 가능할 *지점*이 존재할 수 있다. 본 round의 statement는 그러한 fair-tuning 수행 *이전*의 5-baseline panel 결과에 한정하며, 둘 모두 §8.4 item 4의 follow-up 항목으로 명시된다.

본 작업의 differentiator는 *기법 class 신규성*이 아니라 (multi-direction subspace는 ITI에서, paired-contrast residual-stream은 CAA에서 각각 prior) *multi-direction × residual-stream × (a − m) paired-inpaint × free-lunch 사전등록* 조합이 4-clause를 동시 충족하는 후보로 기능한다는 점이다.

### 6.6 두 mitigation의 정합과 deployable recommendation

E4 (routing site, mid-stack cluster attention re-weighting) 와 E6 (integration site, OneVision residual subspace projection) 는 §5.4 framework가 예측한 두 상보적 mitigation site의 직접 검증이다 — routing-redundant attention 전체에 개입하지 않고도 그 통합 결과를 cross-dataset universal하게 청소할 수 있다.

**Deployable recommendation.** 본 논문이 *deployable* mitigation으로 권장하는 것은 §6.2의 E6이다: inference 시 anchor token span을 알 필요가 없고 (E4는 요구), L=26의 단일 forward hook으로 작동하며, 5 dataset Δdf 부호 일관 + 양 arm em 상승 + 6-benchmark capability preservation을 동시 충족한다. E4는 mechanism diagnostic으로서 routing site의 active load-bearing 성질을 입증한다.

---

## 7 Capability preservation (E8)

§6의 free-lunch는 *anchoring task family 내부*에서 정의되었다. 이 hook이 일반 VLM 능력에서도 neutral-to-positive인지 검증하기 위해, LLaVA-OneVision-7b를 LLaVA-OneVision-Data instruction tuning composition에 *나타나지 않는* 6개 held-out benchmark에서 평가 (VLMEvalKit greedy decoding, baseline과 mitigation의 유일한 차이는 L=26 forward hook). 사전등록 임계: 벤치마크별 Δ ≥ −1.0 pp, 매크로 Δ ≥ −0.5 pp.

**Table 9.** E8 capability preservation, LLaVA-OneVision Main + L=26 K=8 hook, 6-benchmark held-out panel (n_total = 10,507). Δ = mit − baseline (pp); 95 % CI는 paired McNemar 또는 paired bootstrap (sum-style). Bold = 95 % CI excludes 0. 8-benchmark 확장 macro Δ = +0.31 pp는 본문 Note 참조.

| 벤치마크 | n | baseline | +mit | Δ (pp) | 95 % CI | 상태 |
|---|---:|---:|---:|---:|---|---|
| RealWorldQA | 765 | 69.80 | 71.11 | +1.31 | [−0.27, +2.89] | OK |
| OCRBench | 1,000 | 63.40 | 62.60 | −0.80 | [−1.68, +0.08] | OK |
| **HallusionBench** | 951 | 47.84 | 50.05 | **+2.21** | **[+1.14, +3.28]** | **OK** |
| MMStar | 1,500 | 61.67 | 61.80 | +0.13 | [−0.77, +1.04] | OK |
| MMBench-DEV-EN | 1,164 | 82.04 | 81.70 | −0.34 | [−0.82, +0.13] | OK |
| **POPE** | 5,127 | 92.16 | 92.10 | **−0.06** | **[−0.21, +0.09]** | OK |
| **Macro** | | | | **+0.41** | | **4-clause free-lunch** |

모든 벤치마크별 Δ가 사전등록 ±1.0 pp band 내, 매크로 +0.41 pp. **HallusionBench Δ = +2.21 pp 95 % CI [+1.14, +3.28] excludes zero** — K = 8 subspace가 anchor pull 억제와 hallucination diagnostic 성능 향상을 *동시에* 달성. **POPE Δ = −0.06 pp 95 % CI [−0.21, +0.09]** (n = 5,127)로 효과를 *영에 고정*. Pipeline integrity는 lmms-lab model card published 수치와 비교 검증 (MMStar 61.67 vs 61.7 본질적 일치). 신뢰구간은 proportion-style benchmark (RealWorldQA / HallusionBench / MMStar / MMBench-DEV-EN / POPE)에 대해서는 baseline과 mitigation의 per-question correctness를 paired Bernoulli로 두고 McNemar 분산 추정 `SE(Δ) = sqrt(b + c) / n`의 normal-approximation으로 산출했고 (`b`, `c` = paired discordant count), sum-style benchmark (OCRBench)에는 동일 paired pair의 per-question score 차분에 paired percentile bootstrap (n = 1,000 resample, seed 0)을 적용했다 — 두 절차 모두 per-question 수준 resample로 baseline · mitigation 간 의존성을 보존. **Multiple-comparisons 보정.** 본 표는 6-benchmark 패밀리 위에서 6 paired test가 동시 보고되며 *Bonferroni 보정*은 사전 적용되지 않았다. 사후 점검 — 6-test family per-test α = 0.05/6 = 0.0083, two-sided z ≈ 2.64 — 으로는 HallusionBench Δ = +2.21 pp의 SE ≈ (3.28 − 1.14) / 3.92 = 0.546 pp이고 Bonferroni-corrected CI는 [+0.77, +3.65]로 *여전히 zero 제외*하여 보정 후에도 generalisable; POPE는 [−0.21, +0.09]가 [−0.25, +0.13]로 widened되어도 *zero에 pinned* 그대로 유지된다. 다른 4 benchmark는 사전등록 ±1.0 pp band 내부에서 individual-test 수준 결론으로만 유효하다. **Note on benchmark coverage.** 본 표는 6-benchmark 결과를 보고하지만 cross-reference로 8-benchmark 확장 (MME n = 2,374, Δ = −0.13 pp; AMBER n = 14,216, Δ = +0.19 pp [+0.05, +0.33] CI-excludes-zero) 까지 포함한 macro Δ = +0.31 pp 가 별도로 산출되어 있다 (n_total = 27,097; contamination-resistant floor 강화 측면에서 4-clause free-lunch 판정은 두 panel에서 모두 유지). 본 표가 6-bench로 축약된 것은 사전등록 panel 정의의 conservatism이며, AMBER + MME 추가는 capability evidence를 strengthen하되 macro 수치는 conservative 쪽으로 변동한다.

**Insight (Anchoring 외 hallucination 일반 + per-benchmark heterogeneity).** HallusionBench의 양적 결과는 우연이 아니다 — §6.3의 (a − m) contrast가 *wrong-base의 generic error mode*까지 capture한다는 가설의 *외부 검증*이다. Hallucination (특히 illusion / depth)은 본질적으로 "약한 visual signal → 잘못된 confident 답안" 패턴이며, 이는 wrong-base anchor pull과 *동일 축*에서 표현될 가능성이 높다. K=8 subspace projection이 두 패턴 모두 건드린다는 것은 *VLM hallucination의 일부가 본 논문 anchoring mechanism과 representation space를 공유*함을 시사 — 후속 연구의 직접 진입점. **Per-benchmark heterogeneity honest disclosure.** 단 매크로 +0.41 pp는 6 cell 위 균일 positive가 *아니다* — RealWorldQA + HallusionBench 두 *anchoring-adjacent* benchmark가 dominant carrier (각 +1.31, +2.21 pp) 이고, OCRBench / MMBench / POPE / MMStar 4 *broad-VLM-capability* benchmark는 점추정 −0.80 ~ +0.13 pp ±1 pp pre-registered band 안의 mild-negative-to-neutral drift이다 (3/6 cell이 negative point estimate). 따라서 본 절의 작용 mode는 "anchoring/hallucination axis에서 positive + broad capability axis에서 neutral-to-mildly-negative" — 사전등록 ±1 pp / 매크로 ≥ −0.5 pp 두 임계 모두 충족하나 *균일 free-lunch가 아닌 axis-conditional free-lunch* 임을 본 disclosure에서 명시한다 (6-bench와 8-bench 매크로 모두 같은 axis-conditional shape 유지; 본 paper의 canonical capability headline은 6-bench 사전등록 +0.41 pp, 8-bench 확장 +0.31 pp는 contamination-resistance evidence 보강용 cross-reference).

---

## 8 토의 및 결론

### 8.1 종합

**Vision-modality bias의 deployable mitigation은 causal pathway를 confounding scene variance로부터 분리하는 paired-inpaint calibration contrast 위에 구축할 수 있으며, 본 논문은 이 design pattern을 cross-modal numerical anchoring 위에 4-clause free-lunch worked example로 instantiate한다.** 이 thesis를 떠받치는 evidence는 세 layer로 쌓인다. *행동 layer (§4)* — cross-modal numerical anchoring은 6 VLM 위에서 실재하며 **두 gate의 conjunction + plausibility 조건** (불확실성 × digit-pixel · plausibility window) 으로 정의된다; anchoring effect의 진짜 modulator는 *base-prediction의 연속 confidence gradient* (§4.4 L1 6-bin, B6-B1 +19.5-23.5 pp, ≥ 4/5 strict on 51-57/80 cells) 이고 wrong-base / correct-base 분할은 그 gradient의 거친 binary projection (§4.1 PlotQA panel df-기준 +19.0-34.4 pp; §C.1 legacy VQAv2 adopt-기준 +6.9-19.6 pp), 그리고 anchor scene의 digit pixel만 inpaint로 제거하면 효과가 generic distractor 수준으로 되돌아간다 (§4.2 (a − m) gap) — 이 마지막 axis가 thesis의 *paired-isolation 가정*을 행동 측에서 충족한다. *Mechanism layer (§5)* — single-layer ablation이 5/5 null이고 signal이 multi-layer redundant이며 (§5.2 + OneVision 5-dataset 확장), §5.4 **routing vs integration 사후 synthesis**가 §5.2 redundancy + §5.3 dataset-dependent peak + §6.4 LEACE rank-1 ChartQA +56 % 역행을 단일 narrative로 묶고 §4.6 γ-β residual-stream bridge에서 layer-routing 방향성에 한정한 K=1 *partially prospective* 검증을 받는다 — 이 mechanism evidence는 design pattern의 instantiation에서 *single-direction이 아닌 multi-direction subspace projection*을 선택해야 한다는 design choice를 정당화한다. *Worked example layer (§6 / §7)* — E6 (`llava-onevision-qwen2-7b-ov` 위 L=26, K=8 SVD on (a − m) calibration, α=1.0) 가 thesis의 *proof of construction*이다: PlotQA + InfoVQA pooled 1회 calibration 후 inference 시 anchor label 없이 보편 적용되어 5/5 cross-evaluation dataset에서 direction-follow 부호 일관 감소 + 양 arm em 상승, *anchoring effect* (Δdf < 0) PlotQA n=2,306 single-dataset CI-clean + 4 small-n cell 점추정-일관-CI-borderline, *capability-side multiplicity-robust headline* non-anchored arm **Δem(b) 5/5 cell × Bonferroni-corrected CI sign-clean**, 6 held-out benchmark *axis-conditional* free-lunch (anchoring-adjacent axis dominant positive: HallusionBench +2.21 pp excludes zero; broad-capability axis neutral-to-mildly-negative; 매크로 Δ = +0.41 pp 임계 충족). γ-β reasoning amplification (×12.7 correct-base ratio, §4.5; N=1 × N=1 existence proof) 은 *auxiliary observation*으로 thesis의 supporting evidence가 아니라 anchoring 자체의 reasoning-trace 의존성에 대한 별도 관찰이다.

**Implications.** Worked example이 thesis의 *one instantiation*이라는 framing은 field-level의 세 question을 직접 발화한다. 첫째, *(a − m) paired-inpaint calibration contrast의 transferability*: anchoring 외의 vision-modality bias — VLMBias가 다루는 familiar-subject counting의 *typographic identity vs visual content* gap, sycophancy의 *agreement cue vs visual evidence* gap, position bias의 *spatial pattern vs content* gap — 에서도 같은 paired-isolation 구조 (causal pathway scene과 confound scene이 *동일 paired-stimulus 안에서 isolate*되는 inpaint / mask 조작) 를 design할 수 있는가? Vision-modality bias mitigation 문헌은 대부분 single-modality intervention (encoder-side feature blurring [Weng et al., 2024], LM-side text steering [Panickssery et al., 2024]) 에 머물러 있으며, paired-inpaint substrate를 *cross-bias-class*로 확장하는 것은 본 논문이 발화하는 첫 design space이다. 둘째, *operational hyperparameter 보편성*: design pattern의 instantiation에서 K (subspace dimension) 가 cross-architecture universal인가, 아니면 architecture별 *spectral signature*가 K_predicted를 결정하는가? §4.6 K=1 vs K=8 9× ratio가 이미 universal-K 가정을 partial falsify했으며, §8.4 item 1의 eigenvalue spectrum 분석이 *spectrum-predicts-K* 형태로 K 선택을 falsifiable claim으로 변환하는 직접 경로를 제공한다. 셋째, *worked example pattern의 cross-architecture transfer*: design pattern을 다른 LM × encoder 조합 (SigLIP-Gemma, Qwen2.5-VL, FastVLM) 에 instantiate하려면 (L*, K, α) triple을 재calibrate해야 하며, calibration → SVD → projection 의 3-step pipeline 자체가 자동화 가능한 design recipe인가? 세 question 모두 *§8.4 후속 작업 항목* (item 1, 2, 3, 4) 으로 본 paper에 등록되어 있으나, 본 절은 이들을 *저자의 follow-up* 이 아니라 *thesis가 옳다면 field가 묻게 될 question*으로 재배열한다. 본 논문의 worked example이 thesis의 *유일한 instantiation*이라면 일반화는 가설; cross-bias-class transfer 또는 cross-architecture transfer가 같은 4-clause shape를 산출한다면 design pattern은 *generalisable substrate*로 hardening된다.

### 8.2 한계

**연구 범위 한계 (research scope).**

- **E6 mitigation chain은 단일 모델 case study.** Cross-architecture E6 재calibration (다른 LM × encoder 조합 위 (L*, K, α) 재calibration) 은 §5.3 dataset-dependent peak 패턴이 cross-model으로 어떻게 확장되는지 묻는 직접 후속 항목 (§8.4 item 3).
- **단일 prompt.** 모든 실험이 하나의 JSON-strict prompt. Paraphrase robustness (3-5 변종 × bootstrap CI)는 다음 단계.
- **Open-weight 모델만.** 폐쇄 모델 defuse (GPT-4o / Gemini 2.5의 ~500 sample)는 수정 단계에 시도.
- **Human baseline 부재.** 50명 Prolific 연구로 1-2 조건 replicate → 인지과학 framing 강화 (현 ARR 시계 외).
- **Mid-stack cluster 단일.** E4 free-lunch는 3 mid-stack cluster 모델에서 확립 — SigLIP-Gemma early / Qwen-ViT late로의 일반화 미해결. (E4·E6 모두 *각자의 panel scope* 안에서만 검증됨 — E4: 3 mid-stack 모델, E6: OneVision Main 1 모델.)
- **γ-β reasoning amplification은 N=1 architecture × N=1 dataset existence proof.** §4.5의 reasoning-amplifies-anchoring 결과는 단일 architecture pair (Qwen3-VL-8B Instruct vs Thinking) × 단일 dataset (MathVista) × 단일 stratum의 *N=1 × N=1 existence proof*로, cross-architecture · cross-dataset 일반화 검증은 *수행되지 않았다*. 본 결과는 hypothesis-generating existence proof로 분류해야 하며, 다른 thinking-mode VLM family (예: 가상의 Gemma-thinking pair, Qwen3-VL-30b Instruct vs Thinking) 또는 다른 reasoning benchmark (MMMU, MathVerse) 위에서의 검증은 후속 라운드.
- **γ-β residual-stream bridge는 qualitative만 establish, quantitative interlock 미해결.** §4.6은 Qwen3-VL self-calibration K=1 subspace에서 within-Thinking paired Δ가 14/84 Bonferroni-corrected cells에서 0 제외 — 즉 anchor 처리가 layer-routed 형태로 잔차에 흔적을 남긴다는 *qualitative bridge*는 establish되었다. 그러나 **§4.5의 ×12.7 correct-base behavioral df ratio는 잔차 amplitude 측정에서 정량적으로 재현되지 않았다** (within-Thinking magnitude +0.5~+0.9 amplitude units, baseline 위 0.2~0.4 % 상대 변화). K=1 V_K[L=*]는 anchor 처리의 *one aspect*만 capture하는 instrument이며, 다른 잔차 차원·attention pathway 차이·output-head dynamics가 ×12.7 gap의 대부분을 carry한다고 해석한다. Quantitative interlock 추구는 (i) digit-bbox-restricted (a − m) calibration, (ii) attention-head 단위 ITI-style locus 변경, (iii) output-head logit-level mediator 분석 등 후속 라운드 작업이다.

**이번 라운드에서 deferred된 작업 (operational follow-up).**

- **§6.5 CAA · ITI 경험적 row 부재.** Table 8 footnote는 *구조적 reduction* — CAA = (a−m) calibration의 rank-1 ActAdd 인스턴스, ITI = §5.2 single-layer attention null이 사전 예측하는 attention-head 수준 single-locus failure mode — 으로 두 prior 방법이 ActAdd / LEACE rank-1 cell의 failure quadrant에 속함을 *주장*하지만, 두 방법을 직접 calibrate하여 5-dataset Δdf 행을 산출한 *empirical 비교*는 수행되지 않았다. CAA at K = 1 on (a − m) calibration ~4–8 H200-hour, ITI head-level adaptation ~1–2 day 추가 GPU 부담으로 후속 revision에 추가.
- **§6.3 b-arm em +8.8 pp의 alternative explanation 검증 미수행.** §6.3 Insight 1은 b-arm em 이득을 "(a − m) contrast가 함께 capture한 wrong-base error mode 제거" 가설로 해석하지만, 이를 *random-K = 8 subspace* (residual stream의 non-anchor calibration으로부터 추출한 동일 차원 random direction) baseline + *non-anchor-task calibration* (예: POPE, RealWorldQA 위의 K = 8 subspace) baseline와 head-to-head 비교한 ablation은 수행되지 않았다. POPE Δ = −0.06 pp pinned-to-zero 결과는 *yes/no answer-distribution shift* 형태의 generic mode-collapse를 사전 신호로 부정하지만, *numeric mode-collapse* (modal-digit 쪽 logit 편향) 가설을 직접 falsify하지 않는다. Random-K = 8 baseline 추가는 후속 revision의 직접 항목 (§8.4 item 3).

### 8.3 윤리

자극은 합성 digit 이미지 (FLUX rendering)이며 사용된 VQA dataset은 공개 + 인간 주석. 사용자 데이터 미수집, 새 인간 annotation 미취득. 본 작업은 adversarial 환경에서 VLM 약점을 드러내는 *dual-use* 측면이 있으나, E6 mitigation의 즉시 deployability와 free-lunch + capability preservation이 수비 측을 강화한다. 자극 inventory · 코드 · 산물은 익명 저장소 책임 공개 가이드라인 준수.

### 8.4 후속 작업

§8.2가 한계와 운영적 deferral을 다룬다면, 본 절은 *논문의 elevating 가능성을 가진 forward-looking 실험*을 우선순위로 나열한다.

1. **Eigenvalue spectrum of `D[:, L = 26, :]`.** 기존 calibration data에서 단일 spectral plot. `σ_8 / σ_1` ratio가 rank-8 elbow를 보이면 K = 8이 *grid-search 산물*에서 *data-property prediction*으로 격상되며, cross-architecture transfer 질문이 "다른 모델 residual stream에서 rank elbow를 spectral로 예측하고 K_predicted ≈ K_chosen 검증"으로 재구조화된다. 또한 §6.4 Insight 2 ("K = 8 sweet spot, empirical")가 falsifiable spectral claim으로 변환된다.
2. **Random-K = 8 subspace baseline (§6.3 Insight 1.5 falsification).** §6.3에서 명시한 (Alt-1) general regularization 가설을 head-to-head로 falsify — non-anchor calibration set 위 동일 차원 random orthogonal projection을 L = 26에 적용해 b-arm em +8.8 pp이 재현되는지 검증. POPE pinned-to-zero 사전 신호를 보강하여 b-arm em 이득의 *anchor-task-specificity*를 결정적으로 분리한다.
3. **Cross-architecture E6 replication.** §3.3에서 명시한 panel-scope 한계의 직접 close — 다른 LM × encoder 조합 (SigLIP-Gemma, Qwen2.5-VL, FastVLM 등) 위 (L*, K, α) 재calibration. 항목 1의 eigenvalue spectrum이 양성이면 spectrum-predicts-cell 형태로 재구조화 가능.
4. **CAA · ITI 경험적 row.** §6.5 Table 8 footnote의 구조적 reduction을 경험적 row로 보강.
5. **Quantitative γ-β interlock.** §4.6 qualitative bridge 위에 quantitative behavioral ↔ residual amplitude 정렬을 위한 (i) digit-bbox-restricted (a − m) calibration, (ii) attention-head ITI-style locus 변경, (iii) output-head logit mediator 분석.
6. **Paraphrase robustness · 폐쇄 모델 · human baseline.** §8.2의 단일 prompt · open-weight only · human baseline 부재 항목들.
7. **(m − m') inpaint-noise-only SVD baseline (§6.2.1 Telea-residue control).** 같은 anchor scene의 두 독립 Telea inpaint pass (m, m')에서 도출한 K=8 SVD subspace와 본 paper의 (a − m) K=8 subspace의 cosine similarity 비교. (m − m')-only subspace가 (a − m) subspace를 부분적으로 span 한다면 §6.2 calibration substrate가 *digit-pixel-or-Telea-residue-correlated* directions를 함께 capture하고 있음이 직접 검증되며, 그렇지 않다면 §6.2.1 Insight의 "digit-pixel causality clean separation" claim이 representation-level까지 hardening된다. ~2 H100-hour (128 inpaint-twice neutrals 재생성 + SVD + cosine).
8. **Pre-registered single-cell hypothesis tests (§4.6 + §6.2.3 multiplicity hardening).** §4.6 14/84 Bonferroni-clean cells와 §6.2.3 Bonferroni-20을 cell-pre-registration 형태로 hardening — (a) §4.6에서 단일 (K, layer, statistic) cell을 사전 등록 후 단일 hypothesis test, (b) §6.2.3 mitigation evaluation을 *추가 large-n dataset* 위에서 반복하여 small-n bootstrap floor에 묶이지 않은 cell들로 27-cell selection layer를 strict하게 보정 (현재 paper의 540-family disclosure는 *prose-level* — 추가 large-n inference 없이는 ChartQA n=224, MathVista n=170의 1/n 분위수 한계를 우회할 수 없음). (a)는 free recompute, (b)는 ~5 H100-hour (PlotQA · TallyQA 외 추가 n ≥ 1,000 dataset 1-2개에서 chosen cell #17 inference). 본 paper의 directional findings를 confirmatory level로 hardening.
9. ~~**×12.7 ratio paired-bootstrap CI (§4.5 hardening).**~~ ✅ 해결 (2026-05-11). Paired-bootstrap (B=10,000, seed=42, sid-level resample × arm-conditional `base_correct` filter) 결과 **ratio 95 % CI [×6.23, ×56.31]**, instruct correct CI [0.0042, 0.0413] / thinking correct CI [0.2085, 0.3286]. 하한 ×6.23이 ratio가 1로부터 분리됨을 확정. §4.5 본문 prose에 정량 CI inline 삽입; data `docs/insights/_data/qwen3vl_x12_7_paired_ci.{csv,json}`. Round-1 MAJOR-6 / round-4 MAJ-7 요구사항 종결.

항목 1–2는 가장 cheap한 rigor 향상, 항목 7 (재분석 only) 및 8(a) (free recompute) 가 그 다음, 항목 9 + 8(b) (~2-5 H100-hour) 가 다음, 항목 3은 가장 큰 GPU 부담을 요구하는 generalisation 항목이다.

---

## References

[Tversky and Kahneman, 1974] Amos Tversky and Daniel Kahneman. 1974. Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157):1124–1131.

[Jacowitz and Kahneman, 1995] Karen E. Jacowitz and Daniel Kahneman. 1995. Measures of anchoring in estimation tasks. *Personality and Social Psychology Bulletin*, 21(11):1161–1166.

[Mussweiler and Strack, 1999] Thomas Mussweiler and Fritz Strack. 1999. Hypothesis-consistent testing and semantic priming in the anchoring paradigm: A selective accessibility model. *Journal of Experimental Social Psychology*, 35(2):136–164.

[Goh et al., 2021] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. 2021. Multimodal neurons in artificial neural networks. *Distill*.

[Jones and Steinhardt, 2022] Erik Jones and Jacob Steinhardt. 2022. Capturing failures of large language models via human cognitive biases. *NeurIPS 2022*.

[Belrose et al., 2023] Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Tamar Pichasov, and Stella Biderman. 2023. LEACE: Perfect linear concept erasure in closed form. *NeurIPS 2023*. arXiv:2306.03819.

[Li et al., 2023] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. *NeurIPS 2023 (spotlight)*. arXiv:2306.03341.

[Hagendorff et al., 2023] Thilo Hagendorff, Sarah Fabi, and Michal Kosinski. 2023. Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in ChatGPT. *Nature Computational Science*, 3:833–838.

[Echterhoff et al., 2024] Jessica Echterhoff, Yao Liu, Abeer Alessa, Julian McAuley, and Zexue He. 2024. Cognitive bias in decision-making with LLMs. In *Findings of EMNLP 2024*.

[Lou and Sun, 2024] Jiaxu Lou and Yifan Sun. 2024. Anchoring bias in large language models: An experimental study. arXiv:2412.06593.

[Panickssery et al., 2024] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2024. Steering Llama 2 via contrastive activation addition. *ACL 2024*. arXiv:2312.06681.

[Weng et al., 2024] Bingyang Weng et al. 2024. Images speak louder than words. *EMNLP 2024 (Main)*.

[Wang et al., 2025a] Qian Wang et al. 2025. Assessing judging bias in large reasoning models: An empirical study. *arXiv:2504.09946*.

[Wang et al., 2025b] Hanyi Wang, Xiaomeng Wang, Zhengyu Zhao, and Martha Larson. 2025. Typographic attacks in a multi-image setting. *NAACL 2025*. arXiv:2502.08193.

[Vo, Nguyen et al., 2025] Khoi Vo, An Vo, Mohammad Reza Taesiri, Daeyoung Kim, and Anh Nguyen. 2025. Vision language models are biased. *arXiv:2505.23941*.

[Liu et al., 2025] Liu et al. 2025. Investigating VLM hallucination from a cognitive psychology perspective. *arXiv:2507.03123*.

[Rizzoli et al., 2025] Massimo Rizzoli et al. 2025. CIVET: Systematic evaluation of understanding in VLMs. *arXiv:2506.05146*.

[Huang et al., 2025] Yiming Huang et al. 2025. Understanding the anchoring effect of LLM with synthetic data: Existence, mechanism, and potential mitigations. *HCAIR @ ICLR 2026*. arXiv:2505.15392.

[Hufe et al., 2025] Hufe et al. 2025. Dyslexify: A mechanistic defense against typographic attacks in CLIP. *arXiv:2508.20570*.

[Bae et al., 2025] Bae et al. 2025. Do reasoning vision-language models inversely scale in test-time compute? *arXiv:2511.21397*.

[Chand et al., 2025] Shireen Chand, Faith Baca, and Emilio Ferrara. 2025. No free lunch in language model bias mitigation? Targeted bias reduction can exacerbate unmitigated LLM biases. arXiv:2511.18635.

[Fan et al., 2026] Fan et al. 2026. Tinted Frames: Question framing blinds vision-language models. *arXiv:2603.19203*.

[Gong et al., 2025] Yichen Gong et al. 2025. FigStep: Jailbreaking large vision-language models via typographic visual prompts.

---

# Appendix

## A 자극과 prompt 세부

### A.1 Prompt template

JSON-strict template (greedy decoding, temperature 0, top_p 1, `max_new_tokens=8` 비추론 / `512` Qwen3-VL-8B-Thinking γ-β):

> *system:* You are a visual question answering system. Return valid JSON only in the form `{"result": <number>}`. Use a numeric JSON value for `<number>`, not a string. Do not output any other keys, words, explanation, or markdown. If uncertain, still output the single most likely number in that JSON format.
>
> *user:* Answer the question using the provided image(s). Return JSON only in the form `{"result": <number>}`. Question: `{question}`

### A.2 자극 inventory 세부

- **Anchor inventory (`a`)**: 128 FLUX-rendered digit 이미지 (1024 × 1024, 1-step inference, guidance 0). 단일 Arabic 숫자 + generated scene.
- **Mask inventory (`m`)**: 같은 scene 128 inpaint. Digit pixel 영역 (PaddleOCR detected dilated bbox + synthetic fallback)을 Telea inpaint (OpenCV) — OCR 검증 후 검출 digit 0.
- **Neutral inventory (`d`)**: 128 digit-free FLUX render, scene-style 매칭.

### A.3 데이터셋별 거리 cutoff

본 표는 *eligibility cutoff* (sample이 anchor-eligible로 포함되는 최대 distance) + S1 *anchor stratum* (가장 plausible distance bin) 두 양을 함께 명시. 본문 §4.2 의 S1 wrong-base 분석은 *S1 stratum* 위에서 수행되며, §E.1 의 5-strata decay (§E.2 / §E.3) 는 전체 cutoff 안에서 5-strata partition 위에서 수행된다.

| 데이터셋 | Eligibility cutoff | S1 (anchor stratum) |
|---|---|---|
| VQAv2 / TallyQA | 절대 `\|a − gt\| ≤ 5` | 절대 `\|a − gt\| ≤ 1` (S1 [0, 1]) — 5-strata 는 §E.1 |
| ChartQA / MathVista | 상대 `\|a − gt\| ≤ max(1, 0.10·gt)` (E5d 검증, Figure A1) | 동일 상대 cutoff = S1 (single-stratum) |
| PlotQA / InfoVQA | 상대 `\|a − gt\| ≤ max(1, 0.10·gt)` | 동일 상대 cutoff = S1 |

![Figure A1 — ChartQA 거리 cutoff 검증. 상대 cutoff `max(1, 0.10·gt)`이 S5에서 adopt ≤ 0.05를 만족.](../figures/E5d_chartqa_decay.png)

### A.4 자극 생성 reproducibility — FLUX seed

`a` (anchor) inventory는 `scripts/generate_irrelevant_number_images.py --seed-base 1729` 단일 invocation으로 생성 (per-image seed = `seed_base + number`, 즉 digit 0 → seed 1729, digit 1 → 1730, ..., digit 9 → 1738). `m` (mask) inventory는 동일 anchor 위 PaddleOCR 검출 + Telea inpaint (deterministic, no random seed). `d` (neutral) inventory는 동일 FLUX pipeline + `seed_base 1729 + scene_offset` (자세한 invocation은 `scripts/generate_irrelevant_neutral_images.py`). 본문 모든 결과는 이 seed-pinned 128-image inventory에 conditional 하다.

### A.5 Reproducibility — canonical evidence pointers + 27-cell pilot grid

**Body→appendix canonical-source pointers.** 본문 §1–§8의 inline `docs/insights/` 인용은 본 절로 일괄 이전되었다 (paper 본문은 `figures/` · `_data/*.csv` · `scripts/` 외에는 `docs/insights/` evidence file을 직접 인용하지 않는 정책).

- **§4.2 Insight 3 (PlotQA un-mitigated free-lunch).** Slice A E7 PlotQA panel에서 6개 중 5개 모델이 `em(a) > em(b)` (em delta +0.6 ~ +5.0 pp)이며, 동일 패턴이 InfoVQA에서는 부호 mix되어 일반화되지 *않는*다는 사실의 source: `docs/insights/E7-plotqa-infovqa-evidence.md` §3 (PlotQA per-model em(a) vs em(b) table) + §4 (InfoVQA mixed-sign generalization). Canonical CSV: `docs/insights/_data/experiment_e7_plotqa_full_per_cell.csv`, `docs/insights/_data/experiment_e7_infographicvqa_full_per_cell.csv`.
- **§6.2.3 multiplicity-correction honest note + §8.4 item 8 (Bonferroni-540 small-n floor).** B = 100,000 paired-bootstrap diagnostic + parametric normal-approximation 99.99 % CI (`Δ ± 3.91 · SE`) 직접 비교 — small-n datasets (ChartQA n = 224, MathVista n = 170) 의 Bonferroni-540 lower bound가 *1/n empirical discretization floor*에 박히는 반면 large-n cells (PlotQA · TallyQA · InfoVQA n ≥ 443) 는 540-family 하에서도 informative하게 남는다는 진단의 source: `docs/insights/E6-bonferroni540-smalln-floor.md`. 진단은 본 paper의 Bonferroni-20 헤드라인 유지 + 27-cell selection layer prose-only disclosure 선택의 *empirical* 정당성을 제공한다.

**27-cell pilot grid — 4-metric heatmap aggregation.**

§6.2.2의 27-cell pilot grid는 (L, K, α) ∈ {25, 26, 27} × {2, 4, 8} × {0.5, 1.0, 2.0}이며, raw prediction은 `outputs/e6_steering/llava-onevision-qwen2-7b-ov/pilot_grid_plotqa_n250/` (PlotQA pilot, 27 cell × baseline) 및 `pilot_grid_infographicvqa_n250/` (InfoVQA pilot, 동일)에 보존되어 있다. 본 부록은 (i) cell label enumeration, (ii) 선택 cell 위치, 그리고 (iii) Table 7과 같은 4 metric — Δadopt(a) · Δdf(a) · Δem(a) · Δem(b) — 의 27-cell × 2-calibration-dataset aggregated heatmap을 함께 surface한다 (`scripts/aggregate_e6_pilot_grid.py` 단일 pass; canonical CSV `docs/insights/_data/E6_pilot_grid_27cells.csv`, selection-rule replay markdown `docs/insights/_data/E6_pilot_grid_27cells_selection_replay.md`, heatmap figures `docs/figures/E6_pilot_grid_{plotqa,infographicvqa}_heatmap.png`).

**선택 규칙의 binding 여부.** §6.2.2의 ex ante 규칙은 *"어느 calibration dataset에서든 Δem(a) ≤ −6 pp인 cell 거부 후 결합 |Δdf(a)|로 정렬"*이다. 27 cell 중 *어느 cell도* 이 −6 pp 임계값을 위반하지 *않는다* (PlotQA pilot 위 27-cell min Δem(a) = −1.2 pp on cell #19 = L=27 K=2 α=0.5; InfoVQA pilot 위 min Δem(a) = +0.4 pp on cell #1; 자세한 per-cell 값은 `_selection_replay.md`). 따라서 deal-breaker 절은 본 grid 위에서 *non-binding* — pre-committed safety rail로만 작동했고 cell을 prune하지 않았다. 잔존 27 cell을 PlotQA + InfoVQA 평균 Δdf(a)로 정렬하면 *최상위 5*는:

| rank | cell # | (L, K, α) | mean Δadopt(a) | mean Δdf(a) | mean Δem(a) | mean Δem(b) |
|---:|---:|---|---:|---:|---:|---:|
| 1 | **17** | **(26, 8, 1.0)** ★선택 | −1.8 pp | **−4.4 pp** | +3.0 pp | +6.8 pp |
| 2 | 8 | (25, 8, 1.0) | −2.6 pp | −3.2 pp | +2.6 pp | +6.6 pp |
| 3 | 9 | (25, 8, 2.0) | −1.4 pp | −3.0 pp | +2.4 pp | +7.0 pp |
| 4 | 16 | (26, 8, 0.5) | +0.4 pp | −2.8 pp | +2.4 pp | +6.0 pp |
| 5 | 2 | (25, 2, 1.0) | −1.4 pp | −2.2 pp | +0.0 pp | +1.8 pp |

선택 cell #17은 잔존 26 cell을 통틀어 mean |Δdf(a)|-rank **1위**이며, 이는 reviewer가 제기한 *cherry-pick 위험*에 대한 직접 응답이다 — *동일 pilot data 위에서 동일 ex ante 규칙을 재실행한 결과* 선택된 cell이 변하지 않는다.

**Heatmap 패턴 요약.** `docs/figures/E6_pilot_grid_<calib>_heatmap.png`는 4 metric × 3 layer = 12 sub-panel, 각 panel은 K (2 / 4 / 8) × α (0.5 / 1.0 / 2.0) 9-cell heatmap이다. 두 calibration dataset 공통으로 (a) **K = 8 row가 강한 Δdf(a) 감소 영역을 차지**한다 (PlotQA K=8 row: Δdf min ∈ [−8.0, −2.8], K=4 row: ∈ [−2.4, +0.0], K=2 row: ∈ [−5.6, +0.4]), (b) **L = 27은 Δdf · Δadopt 모두 약화** (peak이 본 calib data 위에서는 L=26 즈음에 위치 — §5.3 OneVision peak migration narrative와 일관), (c) **Δem(b)는 grid 거의 전반에 걸쳐 양수** — b-arm em 이득은 chosen cell-specific이 아니라 K = 8 영역의 broad 속성이라는 사후 관찰이며, §6.3 "wrong-base error mode 제거" 해석을 broad-grid 사실로 보강한다.

| 셀 # | L | K | α | 선택 여부 |
|---:|---:|---:|---:|---|
| 1 | 25 | 2 | 0.5 | — |
| 2 | 25 | 2 | 1.0 | — |
| 3 | 25 | 2 | 2.0 | — |
| 4 | 25 | 4 | 0.5 | — |
| 5 | 25 | 4 | 1.0 | — |
| 6 | 25 | 4 | 2.0 | — |
| 7 | 25 | 8 | 0.5 | — |
| 8 | 25 | 8 | 1.0 | — |
| 9 | 25 | 8 | 2.0 | — |
| 10 | 26 | 2 | 0.5 | — |
| 11 | 26 | 2 | 1.0 | — |
| 12 | 26 | 2 | 2.0 | — |
| 13 | 26 | 4 | 0.5 | — |
| 14 | 26 | 4 | 1.0 | — |
| 15 | 26 | 4 | 2.0 | — |
| 16 | 26 | 8 | 0.5 | — |
| **17** | **26** | **8** | **1.0** | **선택 (chosen)** |
| 18 | 26 | 8 | 2.0 | — |
| 19 | 27 | 2 | 0.5 | — |
| 20 | 27 | 2 | 1.0 | — |
| 21 | 27 | 2 | 2.0 | — |
| 22 | 27 | 4 | 0.5 | — |
| 23 | 27 | 4 | 1.0 | — |
| 24 | 27 | 4 | 2.0 | — |
| 25 | 27 | 8 | 0.5 | — |
| 26 | 27 | 8 | 1.0 | — |
| 27 | 27 | 8 | 2.0 | — |

## B 표준 metric (M2) 변형 분석

C-form `(pa − pb) · (anchor − pb) > 0`을 두 alternative와 비교했다 (Figure B1).

- **anchor·gt form** `(pa − gt) · (anchor − gt) > 0`: M1-era code가 사용. `pa`와 `anchor`가 `gt`의 같은 쪽에 있는지를 측정 — stimulus별 변동에 non-robust. VQAv2에서 `gt ∈ {0..8}`와 `anchor ∈ {0..9}`이 부분 overlap이라 stimulus dependence가 두드러짐. *Discarded*.
- **pb·gt form** `(pb − gt) · (pa − gt) > 0`: anchor가 식에 *등장하지 않으므로* anchor pull이 아닌 pb-stickiness 측정. *Discarded*.

모든 paper-tier 주장은 C-form에서 보고되며, anchor·gt form 대비 migration 감사 (207 cells)에서 모든 정성적 주장이 *유지 또는 강화*된다.

![Figure B1 — C-form migration main panel. 5 dataset 6 model cell에서 anchor·gt → C-form direction-follow 변화.](../figures/C_form_migration_main_panel.png)

![Figure B2 — M2 18-variant 비교. Numerator × denominator 18개 변형의 known-signal preservation score. C-form `D_paired` denominator가 모든 signal을 동시 유지하는 유일 변형.](../figures/paper_M2_variant_comparison.png)

### B.1 L1 confidence-bin resolution — 4 vs 6 vs 10 bins

§4.4의 headline은 **6-bin** L1 confidence gradient다 (Figure 5). 사전등록 후보였던 4-bin / 6-bin / 10-bin 중 6-bin 채택의 정당성을 동일 high-n cell (PlotQA all-base × LLaVA-OneVision-7b *(Main)* × S1 × `cross_entropy` proxy, n_pair ≈ 4,700) 위에서 시각화한다 (Figure B3). 4-bin은 B1=B2 floor + B3=B4 plateau로 *coarse* — anchor pull의 graded movement가 4-step에서 분해되지 않고 binary-like step으로 collapse된다. 10-bin은 per-bin n이 ~470으로 떨어져 bootstrap 95 % CI half-width가 ~0.04~0.07로 vary하는 noise floor 위로 *signal이 잘 분리되지 않는다*. 6-bin은 sharp sigmoid-like rise (B1=B2=0 → B6=0.289)를 *6-step monotonic*으로 보여주면서도 per-bin n ≈ 780으로 CI가 stable. 본 cell에서 paper §4.4 headline을 6-bin으로 채택하며 4-bin / 10-bin은 각각 under-resolution / over-resolution failure mode로 보고된다.

![Figure B3 — L1 binning resolution 비교 (4 vs 6 vs 10 bins, PlotQA × LLaVA-OneVision-7b S1 × `cross_entropy`). 4-bin은 plateau로 graded structure를 잃고, 10-bin은 per-bin n이 작아 CI가 noisy. 6-bin은 sharp sigmoid rise + stable CI를 동시에 산출. 출처 `scripts/_compare_l1_binning.py`.](../figures/paper_4_4_binning_comparison.png)

## C Cross-stimulus replication + digit-pixel deeper view

### C.1 Legacy VQAv2 panel — 7-model cross-stimulus replication

§4.1의 PlotQA depth panel은 GT-range [1, 10000] / chart-numeric 분포에서 wrong > correct asymmetry를 *df 기준* +19.0 ~ +34.4 pp로 보고한다. 본 절은 동일 패턴이 **GT-range [0, 9] 단일 자릿수 분포 + 다른 model panel** (Phase-1 이전 legacy 7-model: Qwen3-VL / Gemma4 family + Qwen2.5-VL-7b / LLaVA-Interleave / Gemma3-27b) 위에서 *adopt 기준* +6.9 ~ +19.6 pp로 *동등하게* 재현됨을 보고한다 — H2 wrong > correct asymmetry가 GT-range / encoder-family / model-generation에 *무관한* mechanism-bound prediction임을 확정한다.

**Table C.1.** 7-model VQAv2 panel (n=17,730 per model, S0 unstratified). Bold cell = 각 metric 기준 셀 최댓값 / 최솟값. Bold row name = 해당 metric 기준 가장 robust 모델.

| 모델 | acc(b) | adopt(a) | df(a) |
|---|---:|---:|---:|
| **Gemma4-e4b** | 0.553 | **0.066** | **0.274** |
| LLaVA-Interleave-7b | 0.619 | 0.053 | 0.172 |
| Gemma3-27b-it | 0.628 | 0.053 | 0.167 |
| Qwen3-VL-30b | 0.759 | 0.039 | 0.170 |
| Qwen3-VL-8b | 0.751 | 0.033 | 0.104 |
| **Qwen2.5-VL-7b** | 0.736 | **0.021** | 0.094 |
| **Gemma4-31b-it** | 0.749 | 0.024 | **0.085** |

본 panel은 logit capture 이전 stimulus set이므로 §4.4의 연속 6-bin gradient 분석을 직접 적용할 수 없다. 대신 `base_correct` (baseline 정답 여부) coarse confidence proxy로 stratify하면 7/7 모델에서 wrong-base adopt가 correct-base보다 **+6.9 ~ +19.6 pp 더 크다** — §4.4 6-bin gradient (+19.5-23.5 pp B6-B1 gap)가 이 binary projection의 fine-grained 분해이며, §4.1 PlotQA panel df-기준 +19.0 ~ +34.4 pp gap이 동일 패턴의 chart-numeric stimulus 발현이다.

**Caveat**. VQAv2 numeric subset은 *GT range가 단일 자릿수 [0,9]에 한정*되며 answer 분포도 *highly skewed* (anchor digit "2"가 ~30 %+ 차지). S1 distance stratification (`|a − GT| ≤ 1`) 의 의미가 chart-stack [1, 10000] 만큼 selective하지 않아 anchor stratum 분리력이 약하다. 따라서 본 panel은 §4.1 main 자리에서 §C.1로 이전되었으며, *cross-stimulus replication 증거*로만 인용된다. Anti-scaling (4B > 27B; §4.3 Insight 2) / em-positive baseline (em(a) > em(b); §4.2 Insight 3 — Slice A E7 PlotQA) / sample-size-robust (a-m) gap top-tier (§4.2 Table 3 Slice B PlotQA n_wb=2,107 +6.2 pp) 의 모든 paper-wide pattern은 본 legacy panel의 단일 자릿수 ceiling 한계로 *명확히 분리되지 않으며*, 이 점이 §4.1 main을 PlotQA로 옮긴 일차 동기이다.

### C.2 Digit pixel — model별 deeper view

![Figure C1 — gemma3-27b digit pixel causality. VQAv2 wrong-base S1 anchor=0.138 vs masked=0.082 (gap +5.7 pp).](../figures/E5c_anchor_vs_masked_adopt_gemma3-27b-it.png)

![Figure C2 — qwen2.5-vl-7b digit pixel causality. 양 arm 모두 floor (anchor=0.070, masked=0.066, gap +0.4 pp). 가장 강건한 모델에서는 효과 자체가 검출 한계 아래.](../figures/E5c_anchor_vs_masked_adopt_qwen2.5-vl-7b-instruct.png)

![Figure C3 — Direction-follow 측면 같은 패턴.](../figures/E5c_anchor_vs_masked_df.png)

![Figure C4 — Acc-drop 3-way 비교. anchor / masked / neutral 셋에서 acc 손실. anchor가 가장 큼; masked와 neutral은 1-2 pp 안에서 구별 불가 — anchor scene *background*는 generic distractor와 동등.](../figures/E5c_acc_drop_3way.png)

### C.3 §4.1 Cross-dataset replication — TallyQA + InfoVQA

§4.1 Insight 1 (능력↔끌림 역상관) 과 Insight 2 (wrong > correct df asymmetry) 가 PlotQA 외 두 main-matrix dataset에서도 *동일 6-model panel* 위에서 재현됨을 표로 정리. 출처 `docs/insights/_data/section41_swap_analysis.csv` (생성 스크립트 `scripts/_analyze_section41_swap.py`).

**Table C.2.** Cross-dataset wrong > correct df gap (S1 anchor, base_correct stratify; *df 기준* 6/6 모델 sign-clean on 3/3 dataset).

| 모델 | PlotQA | TallyQA | InfoVQA |
|---|---:|---:|---:|
| Gemma3-4b-it | +34.4 pp | +12.1 pp | +25.8 pp |
| Gemma3-27b-it | +20.5 pp | +11.1 pp | +31.7 pp |
| LLaVA-Interleave-7b † | +23.1 pp | +8.7 pp | +14.7 pp |
| LLaVA-OneVision-7b *(Main)* | +21.8 pp | +7.6 pp | +13.6 pp |
| Qwen2.5-VL-7b-instruct | +20.9 pp | +7.0 pp | +11.3 pp |
| Qwen2.5-VL-32b-instruct | +19.0 pp | +9.2 pp | +14.2 pp |

**Table C.3.** Insight 1 (능력↔끌림 역상관) replication — Qwen2.5-VL family는 3/3 dataset에서 acc(b) 최고 + df(a) panel-min.

| 모델 | PlotQA acc(b) / df(a) | TallyQA acc(b) / df(a) | InfoVQA acc(b) / df(a) |
|---|---|---|---|
| Qwen2.5-VL-7b *(panel min)* | **0.783** / **0.059** | **0.803** / **0.029** | **0.794** / **0.032** |
| Qwen2.5-VL-32b | 0.729 / 0.059 | 0.807 / 0.036 | 0.794 / 0.038 |
| LLaVA-OneVision-7b *(Main)* | 0.481 / 0.130 | 0.787 / 0.039 | 0.582 / 0.114 |
| Gemma3-27b-it | 0.514 / 0.118 | 0.712 / 0.073 | 0.520 / 0.185 |
| Gemma3-4b-it | 0.300 / 0.294 | 0.614 / 0.098 | 0.355 / 0.245 |
| LLaVA-Interleave-7b † | 0.119 / 0.325 | 0.707 / 0.066 | 0.156 / 0.239 |

**Caveats**. (1) TallyQA gap magnitude는 PlotQA / InfoVQA의 절반 이하 — TallyQA의 0-15 GT range로 인한 panel-wide df compression (0.029-0.098). (2) †는 §4.1 Table 2 footer와 동일 caveat.

**Reading**. Insight 1 (Qwen2.5-VL family panel-min) 와 Insight 2 (df-기준 6/6 sign-clean) 모두 3/3 dataset에서 robust. PlotQA를 §4.1 main으로 선택한 정당화는 *signal magnitude* 기준 (PlotQA gap range +19.0~34.4 pp가 TallyQA +7.0~12.1 pp / InfoVQA +11.3~31.7 pp 양쪽 중간보다 widest); 다른 두 dataset에 대한 본문 6-row table 중복 확장은 §4.3 5-dataset main matrix와의 axis 분업을 깨므로 본 부록으로 한정한다.

## D Mechanism 보완

### D.1 모델별 peak layer (PlotQA calibration) + mechanism 측 보조 측정

Calibration dataset = **PlotQA** (`docs/insights/_data/cross_dataset_peaks.csv` answer-step). Cross-dataset stability matrix는 [`headline-numbers.md §A.4`](../insights/headline-numbers.md). qwen2.5-vl-7b은 PlotQA bbox-extraction run 부재로 **VQAv2 reference**를 사용 (top-decile susceptible stratum L=22; `_per_layer/peak_layer_summary.csv`).

| Archetype | 모델 | Peak layer | Source dataset |
|---|---|---|---|
| SigLIP-Gemma early | gemma4-e4b | L5 / L42 | PlotQA (4-dataset stable: VQAv2/Tally/Info/PlotQA 모두 L=5) |
| Mid-stack cluster (CLIP-ViT) | llava-1.5-7b | L14 / L32 | PlotQA (VQAv2/Tally/Info-only는 L=8) |
| Mid-stack cluster (ConvNeXt) | convllava-7b | L14 / L32 | PlotQA (VQAv2/Tally L=7, Info L=12) |
| Qwen-ViT late | qwen2.5-vl-7b | L22 / L28 | **VQAv2 reference** (PlotQA bbox run 미측정) |
| FastVLM late (text-stealing) | fastvlm-7b | L17 / L28 | PlotQA (VQAv2 L=22, Tally L=23, Info L=27 — cross-dataset variability §5.3) |

**Encoder-family clustering — PlotQA calibration 관찰, fragile.** PlotQA calibration 위에서 모델별 단일 peak가 SigLIP-Gemma early (L5) / mid-stack cluster (LLaVA-1.5·ConvLLaVA L14) / Qwen-ViT late (L22, VQAv2 reference) / FastVLM late (L17) 4-archetype으로 분리된다. 가장 직접적인 cross-cut 증거는 동일 LLaMA 계열 LM을 공유하는 LLaVA-1.5 (CLIP-ViT) + ConvLLaVA (ConvNeXt) 가 모두 mid-stack L14에 속한다는 것 (N=2 직접 비교). 단 본 archetype clustering은 *PlotQA calibration 위 단일 dataset 관찰*로, **FastVLM (PlotQA L17 / VQAv2 L22 / TallyQA L23 / InfoVQA L27 — `_data/cross_dataset_peaks.csv`)** 과 **OneVision Main (bimodal L14 / L27; §5.3)** 두 모델에서 cross-dataset peak shift가 관측되어 *encoder가 peak 위치를 고정한다*는 강한 claim은 fragile하다. 따라서 본 표는 §5.2 ablation의 표적 layer 식별을 위한 setup으로 기능하며, "encoder-family-determines-peak-archetype" 자체는 본 논문이 *contribution*으로 elevate하지 않는다 (§4.3 Insight 3의 *행동 측 5-dataset average df cross-encoder ordering* 은 peak 위치와 별개 register로 성립; §6.1 E4가 mid-stack cluster 두 모델에서만 검증된 panel-scope 한계는 §6.1 / §8.2 참조).

**E1-patch digit-bbox concentration (mechanism 측 보조 측정, §4.2 행동 측 (a − m) gap의 mechanism complement).** 4-model perfect-square panel (`gemma4-e4b`, `llava-1.5-7b`, `convllava-7b`, `qwen2.5-vl-7b`; non-perfect-square AnyRes 모델은 본 부록에 한정)에서 digit bbox 내부 attention 분배 비율이 0.468–0.631 — *fair-share assumption 대비 +24 ~ +40 pp 위*. OneVision Main의 digit-bbox concentration peak는 cross-dataset pooled extraction (n=1,205, `analyze_attention_patch.py`가 `outputs/attention_analysis/llava-onevision-qwen2-7b-ov/` 아래 모든 dataset run을 자동 합산)에서 L20–L23 cluster (`L20=0.507, L23=0.492, L25=0.459`, 출처 `_data/E1_patch_concentration_per_layer.csv`); 이 양은 attention-mass의 *answer-step peak depth*와 다른 양으로 후자는 dataset-dependent (§5.3). Attention이 *수치 단서를 운반하는 픽셀 영역에 우선 정렬*된다는 직접 측정으로, §4.2 digit-pixel causality의 행동 측 (a − m) gap에 대한 mechanism 측 complementary evidence.

### D.2 E1d ablation mode별 결과. Bold = significant (95 % bootstrap CI excludes 0).

**D.2.1 5-mech panel (gemma4-e4b, llava-1.5-7b, ConvLLaVA-7b, qwen2.5-vl-7b, fastvlm-7b; n=200 per model on VQAv2)**

| Mode | 5-model 평균 Δdf |
|---|---|
| `ablate_peak` (single layer) | ~0 (5/5 null) |
| `ablate_peak_window` (peak ± 1) | ~0 |
| `ablate_lower_half` | heterogeneous (2/5 backfire) |
| **`ablate_upper_half`** | **−4.0 ~ −10.5 pp** (5/5 significant) |
| `ablate_all` | −5.0 ~ −12.0 pp (5/5 significant) |

**D.2.2 OneVision Main extension (llava-onevision-qwen2-7b-ov; n=200 stratified per dataset, B=2,000 bootstrap CI).**

| Mode | TallyQA | InfoVQA | ChartQA | MathVista | PlotQA |
|---|---:|---:|---:|---:|---:|
| baseline df | 0.130 | 0.167 | 0.105 | 0.171 | 0.243 |
| Δ `ablate_peak` (pp [95 % CI]) | −0.5 [−5.0, +4.0] | +1.5 [−3.9, +7.0] | 0.0 [−4.0, +4.5] | 0.0 [−5.1, +5.5] | −0.6 [−6.2, +5.5] |
| Δ `ablate_peak_window` | +0.5 [−4.0, +5.5] | +0.4 [−4.6, +5.6] | +0.5 [−3.5, +5.0] | −0.5 [−5.5, +4.9] | −1.0 [−6.6, +5.2] |
| Δ `ablate_lower_half` | +5.0 [+0.0, +10.5] | −0.6 [−5.5, +4.7] | +2.6 [−2.0, +7.3] | **+7.5 [+1.6, +13.6]** | +2.4 [−3.7, +8.7] |
| Δ `ablate_upper_half` | −2.5 [−6.5, +2.0] | +0.4 [−4.7, +6.0] | −0.5 [−4.4, +4.0] | −2.6 [−7.1, +2.4] | −3.9 [−9.4, +1.9] |
| Δ `ablate_all` | −4.0 [−7.5, +0.0] | +0.8 [−4.2, +6.3] | +0.6 [−3.5, +5.1] | −4.5 [−9.0, +0.4] | −5.1 [−10.6, +0.5] |

**Reading.** Single-layer ablation은 5/5 null on OneVision Main (max |Δdf| = 1.5 pp on InfoVQA, 모든 95 % CI overlap 0) — 5-mech panel의 5/5 null과 일관, multi-layer redundancy claim의 *확장 검증*. 반면 upper-half ablation은 5-mech panel의 균일 −4 ~ −10.5 pp significant pattern과 달리 OneVision에서는 5/5 null at n=200 (point estimates ∈ [−3.9, +0.4] pp; PlotQA가 가장 가깝지만 95 % CI [−9.4, +1.9]로 0 포함) — §5.3 OneVision dataset-dependent peak (Plot/Tally L=27 vs Info/VQAv2 L=14)와 일관하는 heterogeneity로, *5-mech panel calibration 위에서 식별된 upper-half locus가 OneVision의 cross-dataset signal을 capture하기에 부족함*을 보이고 §6.2 subspace projection 도구 선택을 보강한다. Lower-half BACKFIRE는 1/5 significant (MathVista) + 1/5 boundary (TallyQA)로 5-mech panel의 2/5 backfire 패턴과 동일 (§5.2 line 229 참조). `mean_distance_to_anchor`는 OneVision의 hallucinated large-number 응답 (1e6 단위)으로 corrupted; C-form Δdf는 sign-only이므로 magnitude outlier 영향 없음 (`E1d-causal-evidence.md`).

## E Plausibility window (distance × stratum) — 전체 분석 + S1 confound resolution

본 부록은 본문 §4의 *plausibility 조건* (anchor가 gt에 가까운 stratum에서 효과 집중) 의 full empirical 검증과 S1 trivial-recovery confound 해소를 정리한다. Plausibility window 결과는 본 논문의 *3 axis 정성적 증거* (§1.3 (i)-(iii): wrong/correct asymmetry · L1 6-bin gradient · digit-pixel causality) 어느 항목의 load-bearing claim도 *아니지만*, Mussweiler-Strack selective accessibility 모델의 직접 예측 (anchor가 plausible 범위에 있어야 비교 후보로 진입) 의 행동 측 확인이며, §6 mitigation의 거리 분포 재보정 *불필요* 정당화 (em이 distance-stable) 의 근거이다.

### E.1 Per-dataset stratum cutoff

Anchor distance `|a − gt|` 의 stratum 분할은 dataset마다 다르다 (`docs/paper/sections/04_datasets.md` §4.1 표):

| 데이터셋 | Cutoff |
|---|---|
| VQAv2 / TallyQA | 절대 `|a − gt| ≤ 5`; 본문 5-strata는 S1 [0,1] / S2 [2,5] / S3 [6,30] / S4 [31,300] / S5 [301,∞) |
| ChartQA / MathVista | 상대 `|a − gt| ≤ max(1, 0.10·gt)` (E5d C3-validated; MathVista는 single-stratum 설계로 본문 §4.3 / γ-α 사용) |

### E.2 5-strata × 2-model × 2-dataset wrong-base adoption decay

LLaVA-Interleave-7b + Qwen2.5-VL-7b × VQAv2 + TallyQA × 5 strata, n=1,000 base/cell. Wrong-base `adopt_cond` (paired conditional adoption, source `_data/E5b_per_cell.csv`):

| Stratum | VQAv2 llava | VQAv2 qwen2.5 | TallyQA llava | TallyQA qwen2.5 |
|---|---:|---:|---:|---:|
| S1 [0,1] | **0.130** | **0.070** | **0.092** | **0.033** |
| S2 [2,5] | 0.032 | 0.014 | 0.006 | 0.015 |
| S3 [6,30] | 0.010 | 0.003 | 0.003 | 0.000 |
| S4 [31,300] | 0.010 | 0.003 | 0.000 | 0.000 |
| S5 [301,∞) | 0.003 | 0.003 | 0.000 | 0.000 |

LLaVA에서 두 자릿수 감쇠 (S1 0.130 → S5 0.003), Qwen에서 한 자릿수 (S1 0.070 → S5 0.003). 두 모델 × 두 dataset 모두에서 *S1 peak / S5 floor* monotonic decay (Figure F1, F2). Correct-base curve는 모든 stratum에서 평평 (`adopt_cond ≤ 0.10`) — *uncertainty gate × plausibility 조건의 conjunction*만이 효과를 산출.

![Figure F1 — Distance × stratum decay (paired adoption). Wrong-base에서 S1 [0,1] peak, S5 [301,∞) floor. LLaVA-Interleave + Qwen2.5-VL × VQAv2 + TallyQA 4 cell에서 monotonic. Correct-base (점선) 평평.](../figures/E5b_adopt_cond_curve.png)

![Figure F2 — Adopt rate overlay. 4 cell의 S1 peak / S5 floor decay 일관성 시각화.](../figures/E5b_adopt_cond_overlay.png)

### E.3 OneVision Main 5-strata × 4 dataset (full GT range, n=85,258)

본 부록의 §E.2 panel은 GT≤8 (VQAv2/TallyQA 교차 범위) 로 제한된다. plausibility window 주장이 *full GT range* 에서도 재현되는지 확인하기 위해 OneVision Main에 4-dataset (MathVista / ChartQA / InfoVQA / PlotQA) × 12-cond (b + a/m × {S1..S5} + d, relative cutoff scheme) 을 실행했다 (총 85,258 records on 2 × H200, ~3.5h wall, source `outputs/experiment_e5b_5strat_<ds>_onevision/<ts>/predictions.jsonl`).

| Dataset | n_샘플 | n_records (12-cond) | adopt(a) S1 → S5 | df(a) S1 → S5 |
|---|---:|---:|---|---|
| MathVista | 385 | 4,578 | 0.105 → **0.016** (6.6×) | 평평 (Δ ≈ +0.003) |
| ChartQA | ~705 | 8,260 | 0.028 → **0.004** (7.0×) | 0.18 → 0.14 (−22 %) |
| InfoVQA | ~1,147 | 13,546 | 0.045 → **0.002** (≥20×) | 0.19 → 0.15 (−21 %) |
| PlotQA | ~5,000 | 58,874 | 0.087 → **0.008** (11×) | 0.21 → 0.15 (−29 %) |

**결합 coverage**: 2 architecture (llava-interleave-7b §E.2 + llava-onevision-qwen2-7b-ov §E.3) × 6 dataset (VQAv2 + TallyQA + MathVista + ChartQA + InfoVQA + PlotQA) × GT range {0..8} ~ ≤1,000.

### E.4 S1 trivial-recovery confound 해소

S1 [0,1]은 anchor가 gt에 plausibly 가까운 stratum이라, "*모델이 gt 쪽으로 복귀한 결과가 우연히 anchor 쪽으로 보이는*" trivial 해석이 가능하다 — direction-follow C-form 분자 `(pa − pb)·(anchor − pb) > 0`이 anchor 효과 *없이도* `gt` 회복만으로 양수가 될 수 있기 때문 (anchor ≈ gt 인 S1 부분집합).

**(a, m) paired comparison이 이 confound를 직접 falsify한다.** 같은 wrong-base × S1 부분집합에서:
- a-arm: target + 디지트 anchor 이미지
- m-arm: 같은 anchor scene + Telea-inpaint된 디지트 (디지트만 제거, scene 보존)

두 arm의 *거리·scene·plausibility·gt 회복 가능성*은 동일하다 (anchor 이미지 자체가 같은 scene; 차이는 픽셀 한 곳뿐). 따라서 trivial-recovery confound가 작용한다면 양 arm에 *동일하게* 작용해 (a − m) 차분에서 *상쇠*되어야 한다.

본문 §4.2 Table 3 (두 슬라이스) 의 wrong-base × S1 paired conditional adoption gap에서 대표 cell:

- *Slice A — PlotQA × cross-model (E7).* Gemma3-4b PlotQA: `adopt(a) = 0.184` vs `adopt(m) = 0.056` → **(a − m) = +12.8 pp**; LLaVA-Interleave-7b PlotQA: 0.082 vs 0.014 → **+6.8 pp**; LLaVA-OneVision *(Main)* PlotQA: 0.090 vs 0.028 → **+6.1 pp**.
- *Slice B — Main × cross-dataset (E5b).* LLaVA-OneVision MathVista: 0.105 vs 0.039 → **+6.6 pp**; LLaVA-OneVision PlotQA: 0.087 vs 0.025 → **+6.2 pp** (E5b run, Slice A E7 PlotQA cell과 +6.1/+6.2 pp 독립 replication).

Slice A 6/6 모델, Slice B 5/5 dataset 모두에서 *디지트 픽셀이 있을 때만* effect가 발생, *디지트 픽셀을 제거하면* (a − m) gap이 양 — "S1에서 모델이 gt를 회복" 해석으로는 *(a − m) 차분이 0에 가까워야* 하지만 실제로 panel-wide 양수. 따라서 S1 peak는 *gt 회복 우연*이 아닌 *디지트 픽셀의 인과적 anchor pull* 에 기인한다.

### E.5 Plausibility window의 paper-narrative 위치

세 paper 결정에 거리 plausibility 결과가 함의를 가진다:

1. **Mitigation 측 거리-무관 작동 정당화 (§6).** S1→S5에서 em(a)이 안정 (Δem(a) ∈ [−0.018, +0.011] across 4 cross-datasets, §E.3) — *distance가 em을 손상시키지 않으며*, anchor가 거부되면 d-arm em으로 복귀, 채택되면 redirect cost는 거리에 무관. §6.2 E6 mitigation은 *distance 분포 재보정 없이* universal projection으로 작동하는 설계 결정이 plausibility 조건에 대한 *operational independence*에 의해 정당화된다.

2. **adopt vs df 측정 분리.** Distance는 *adopt*에 강한 효과 (S1→S5 6-22× 붕괴) 이지만 *df*에는 약한 (3/4 dataset에서 −21~−29 %; MathVista는 평평) 효과를 가진다. 이는 두 metric이 다른 게이트를 측정함을 시사 — adopt은 *후보 진입(admission) gate* (anchor가 답안 후보로 *허용*되는지 binary), df는 *sub-threshold pull* (후보로 거부되어도 방향 정보가 *누출*되는지 continuous). 멀리 떨어진 anchor는 답안 후보에서 거부되지만 (낮은 adopt) 그 방향은 잔여 표현에서 끌어온다 (0이 아닌 df).

3. **Mussweiler-Strack 인지과학 정렬.** Selective accessibility 모델은 anchor가 plausible 범위에 있어야 비교 후보로 active state에 진입한다고 예측한다. S1 peak / S5 floor는 그 예측의 행동 측 외부 검증 — 인지과학 가설의 cross-modal VLM 일반화이다 (§1 / §8.1 prose에서 reference).

## F MathVista 검증

![Figure G1 — MathVista 거리 검증 (E5d). MathVista에서는 S1-only cutoff가 ChartQA 같은 상대 cutoff 검증을 통과하지 못함 (C3 fail). 본 논문은 §4.3에서 MathVista를 single-stratum 설계 (γ-α)로 다루며, γ-β reasoning-mode 비교 (§4.5)는 별도 single-stratum 실행.](../figures/E5d_mathvista_decay.png)

