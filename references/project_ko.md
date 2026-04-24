# VLM의 cross-modal anchoring: EMNLP 2026 솔직 평가 *(영문 canonical: `project.md`)*

**결론 선제시.** 핵심 경험적 주장 — target VQA 이미지 옆에 렌더링된 *숫자 이미지*를 anchor로 주입했을 때, 효과가 **원래 모델이 틀린 item에서 비대칭적으로 더 강하게 나타난다**는 발견 — 은 실제로 novel하고, 기존 문헌에 대해 방어 가능하다. 하지만 현재 스코프(7 모델 × ~5 cognitive bias × VQAv2, mechanistic 분석 없음, mitigation 없음)는 EMNLP 2024–2025에서 **Findings로 accept된** 최근 cognitive-bias-in-LLM 논문들의 empirical profile과 일치 — Main에는 못 올라갔다. Findings-tier에서 Main-tier로 움직이려면 이 논문은 (1) breadth를 줄이고 *mechanistic depth*를 추가, (2) mechanism이 가이드하는 최소한의 mitigation 추가, (3) cognitive-bias 잡화점이 아니라 sharp한 단일 과학적 주장으로 reframe이 필요. 8×H200 + 한 달이면 가능 — 단, 확장 계획을 *좁혀야* 가능하다, 넓히는 게 아니라. 이하 섹션들은 prior-art landscape, 각 제안된 확장에 대한 novelty verdict, feasibility 계산, 구체적 권고사항을 정리.

## Novelty verdict: 강한 core, 약한 extension들

사용자의 중심 setup은 직접적 선례 없음. 어떤 published work도 standalone으로 렌더링된 숫자 이미지를 multi-image VLM prompt 안에 cross-modal anchor로 집어넣고 open numerical VQA에서 anchor 쪽으로의 regression-style shift를 측정하지 않는다. 가장 가까운 이웃들은 각각 중요한 축에서 다름: **Wang, Zhao, Larson (NAACL 2025, "Typographic Attacks in a Multi-Image Setting")** 는 target image 위에 overlay된 class-label 텍스트를 classification에 사용; **Nguyen et al. (VLMBias, ICML AI4MATH 2025)** 는 subject-identifying label을 주입해 counterfactual counting의 memorization prior를 probe, numerical anchoring 아님; **FigStep (Gong et al., AAAI 2025)** 는 jailbreak을 위해 text-as-image를 렌더, numerical bias 아님; "Biasing VLM with Visual Stimuli" 노트는 multiple-choice answer를 pre-mark, numerical value 아님. LLM-anchoring 부모 문헌 (Jones & Steinhardt 2022; Echterhoff et al. EMNLP Findings 2024; Lou & Sun 2024; O'Leary 2025) 은 전부 text-only.

**"원래 틀린 case에서 더 강함" asymmetry 역시 novel.** 어떤 prior LLM/VLM anchoring 논문도 prior correctness로 effect를 partition하지 않음. Lou & Sun (2024) 은 stronger model이 *더* anchor한다고 발견했지만 item-wise로 조건화하지 않음; Jones & Steinhardt는 correct/wrong을 aggregate "functional accuracy drop"에 합침. 이 asymmetry는 깨끗한 cognitive-science 근거 (Mussweiler & Strack; Jacowitz & Kahneman — anchoring은 subjective uncertainty에 비례) 가 있고 논문의 가장 강한 지적 hook.

**Dual Process Theory 프레이밍은 덜 탐험됐지만 empirically 논쟁적.** Hagendorff et al. (Nature Computational Science 2023) 은 LLM에 System 1/System 2를 적용; Brady et al. (Nature Reviews Psychology 2025) 는 LLM 맥락에서 그 프레임워크를 리뷰. VLM에 대해서는 published application 없음. 그러나 증거는 *엇갈림*: VLMBias는 reasoning VLM (o3, o4-mini) 이 non-reasoning보다 in-image textual cue에 **더** bias된다고 발견, Wang et al. "Judging Bias in Large Reasoning Models" (2025) 는 reasoning이 일부 bias를 증폭함을 보임. 사용자의 DPT 서사 ("System 2가 anchoring을 억제") 는 이 반례들을 조심스럽게 engage해야지 무시하면 안 된다.

각 제안된 확장의 novelty를 요약한 표:

| Extension | 가장 가까운 prior art | Novelty verdict |
|---|---|---|
| **Cross-modal numerical anchoring (core)** | VLMBias (다른 setup); typographic attack (다른 task) | **진짜 novel** |
| **"Wrong case에서 더 강함" asymmetry** | LLM이든 VLM anchoring이든 발견 못 함 | **진짜 novel, 가장 강한 hook** |
| **VQAv2 너머 데이터셋 확장** | 해당 없음 — 방법론적 위생 | 필요, novel하지 않음 |
| **Thinking vs. instruction-tuned VLM** | Lou & Sun 2024 (LLM); Wang 2025 LRM judging; VLMBias | VLM에서 moderately novel; 엇갈리는 증거 engage 필수 |
| **ViT vs. Conv-encoder VLM** | Typographic-attack mechanistic 작업 (arXiv 2508.20570) | **Novel angle** — ConvLLaVA / DINO / EVE 는 훌륭한 negative control |
| **프레이밍: 숫자에 대한 text vs. image modality** | Tinted Frames 2026 (question-form 프레이밍만); text-dominance 문헌 | **Novel** — 이미지 vs. 텍스트를 통한 Kahneman-Tversky gain/loss 프레이밍은 비어 있음 |
| **box/marker 를 통한 salience bias** | Shtedritski "red circle" 2023, FGVP, STER-VLM, spatial attention bias 2025 | **붐빔** — 차별화 어려움 |
| **Confirmation bias (base prediction 근처 anchor)** | VLMBias가 "memorization overrides vision" 프레이밍 소유 | **부분적** — 사용자 버전은 실제로 *anchoring의 하위 발견*이지 별개 bias 아님 |
| **Cognitive Load Theory / distractor** | Idis (2025), I-ScienceQA, MVI-Bench, MM-NIAH, MIHBench — 포화 | **약한 novelty** — 중복 큼 |
| **Dual Process Theory를 조직 프레임으로** | Hagendorff 2023, Brady 2025 (LLM만); VLM에 적용 안 됨 | **Novel 프레이밍** 이지만 empirically 논쟁적 |

## 논문이 무시할 수 없는 prior art

여덟 편은 인용하고 차별화해야 하며, 일부는 직접 비교 필요. **Nguyen et al.의 VLMBias (arXiv:2505.23941)** 는 가장 위험한 이웃이고 full paragraph의 차별화가 필요: 그들의 "cue"는 memorize된 subject label, 그들은 counterfactual canonical image에서의 counting을 측정 (arbitrary anchor 쪽으로의 numerical shift 아님), prior correctness로 조건화하지 않고, 다른 이론적 렌즈를 사용. **AIpsych (arXiv:2507.03123)** 는 최초의 psychology-grounded VLM cognitive-bias benchmark로, anchoring이 아니라 sycophancy/authority/logical consistency를 다루긴 하지만 가장 가까운 포괄적 전임자로 인용해야 한다. **Jones & Steinhardt (NeurIPS 2022), Echterhoff et al. (EMNLP Findings 2024), Lou & Sun (2024), and "Understanding the Anchoring Effect of LLM" (arXiv:2505.15392)** 는 canonical LLM-anchoring 계보; 마지막 것은 논문이 채택해야 할 A-Index / R-Error 메트릭을 도입. **Hagendorff, Fabi & Kosinski (Nature Computational Science 2023)** 는 canonical DPT-in-LLM 레퍼런스. **Goh et al.'s "Multimodal Neurons" (2021)** 과 **Wang, Zhao, Larson (NAACL 2025)** 은 논문이 synthesize하는 mechanistic 조상들. **Tinted Frames (arXiv:2603.19203)** 는 VLM에서 question-form 프레이밍을 다루며, 사용자가 제안하는 semantic gain/loss 프레이밍과 구분돼야 한다.

중요 red flag: 관련 작업들 — 특히 typographic-attack 문헌 — 은 같은 *causal mechanism* (pixel로 렌더된 텍스트가 concept neuron 활성화) 을 공유하지만 cognitive bias가 아니라 adversarial attack으로 프레이밍. **Reviewer는 물을 것: "이게 그냥 typographic attack을 이름만 바꾼 거 아님?"** 답은 crisp해야 함: (a) 주입된 콘텐츠는 *semantic label이나 class identity 없는 numerical value*, (b) target task는 *open-ended numerical estimation* 이고 측정된 regression 계수로 답함, classification flip이나 jailbreak ASR 아님, (c) 프레이밍이 testable prediction (uncertainty asymmetry, DPT effect) 을 가진 인간 cognitive science와 연결. 이 프레이밍은 implicit이 아니라 explicit해야 한다.

## 이 유형 논문에 대해 EMNLP Main이 실제로 요구하는 것

Empirical base rate는 명확. LLM의 cognitive bias에 대한 거의 동일한 논문들 — 예: Echterhoff et al. 2024는 13,465 prompt, 5 bias, 4+ 모델, *그리고* debiasing method — 은 Main이 아니라 **Findings**. "How Does Cognitive Bias Affect LLMs?" (2025) 도 Findings. CIVET (VLM의 position-understanding 체계적 평가) 도 Findings. 반면 Weng et al.의 "Images Speak Louder than Words" (EMNLP 2024 Main) 은 causal mediation analysis와 22% mitigation을 결합. **패턴은 결정적: black-box behavioral probing 만으로는 Main bar를 못 넘음, 스코프가 크고 발견이 흥미로워도.**

Main bar를 넘기는 것은: **mechanistic 또는 causal 분석** (attention patching, causal mediation, logit lens, SAE, probing classifier), 간단해도 좋은 **mitigation method**, 알려진 현상의 새로운 측정이 아니라 **새로운 방법론적 프레임워크**, 또는 엄격한 증거에 근거한 **진짜 놀라운 과학적 주장**의 어떤 조합. Behavioral 논문에서 reviewer들이 일관되게 cite하는 것: (1) mechanistic depth 부족, (2) prompt-sensitivity 우려, (3) "so what?" — 행동 가능한 takeaway 부재, (4) psychology 개념의 느슨한 사용, (5) human baseline 부재, (6) single-dataset 협소함, (7) frontier closed model 미포함. 스코프 기대치는 5–15 모델, ≥2 데이터셋, 통계 검정과 함께 3–5 prompt variant의 paraphrase robustness, 그리고 *진짜로 한계를 제한하는* limitations section.

## 8×H200 + 한 달로의 feasibility: compute는 병목이 아니다

Raw compute budget은 **~5,760 GPU-hour**, 전형적 EMNLP VLM 평가 논문 (총 50–500 GPU-hour) 을 훨씬 상회. 7B VLM은 vLLM과 H200 한 장에서 하루에 short-answer VQA sample 50k–200k 처리; 30B 모델은 10k–40k. Long CoT chain (>1k output token) 을 쓰는 thinking-mode 모델은 10–20× 느려서 진짜 병목. Naive 확장 plan — 7 모델 × ViT/Conv × thinking/instruction × 4+ bias × 2+ 데이터셋 × paraphrase — 는 대략 3–10M inference call로 폭발하고, 각 bias에 full dataset이 아니라 curated ~2k–5k subset을 쓸 때만 feasible.

**Compute는 풍부; scope discipline과 mechanistic depth가 binding constraint.** 마지막 한 달을 mechanistic layer 추가 대신 더 넓은 grid 돌리는 데 쓰는 것은 전략적 실수.

## Accept 확률에 대한 영향으로 랭크한 권고사항

**최고 영향: mechanistic 컴포넌트 추가.** 적당한 수준이라도 논문의 profile을 극적으로 변화시킴. 구체적 옵션을 effort-to-payoff 순으로:
- biased vs. neutral 조건에서 image-token vs. anchor-image-token vs. text-token 에 대한 attention을 비교하는 attention-mass 분석 — HuggingFace `output_attentions=True` 로 몇 시간 안에 실행, "왜 이런 일이 발생하는가" reviewer complaint에 직접 답.
- anchor 영향이 computation 안으로 *언제* 들어가는지 localize하는 layer-wise logit lens.
- **ConvLLaVA (pure ConvNeXt), EVE/Fuyu (encoder-free), DINO-based VLM** 을 사용한 vision-encoder ablation — 효과가 encoder-mediated (CLIP/SigLIP의 typographic 약점, arXiv:2508.20570 참조, 을 상속받는다면 예상되는 바) 인지 LLM-mediated 인지 직접 테스트. 과학적으로 흥미롭고 novel하며 empirically tractable.
- Weng et al. EMNLP 2024 Main 템플릿을 따라 biased vs. neutral 실행 간 activation patching 으로 bias를 causally localize.

**높은 영향: mechanism이 가이드하는 간단한 mitigation 추가.** Attention 분석이 anchor image가 불균형하게 attention을 끈다고 보이면, inference-time re-weighting 또는 contrastive-decoding intervention이 10–20% 효과 감소를 보이면 논문 profile을 "관찰"에서 "관찰 + 설명 + 해결"로 flip. Main으로 가는 가장 신뢰할 만한 단일 루트.

**높은 영향: 내러티브를 하나의 주장으로 sharpen.** 세 후보 프레이밍, reviewer-excitement 감소 순:
1. *"VLM의 cross-modal anchoring은 CLIP-family vision encoder에서 기원하며 cognitive-uncertainty 법칙을 따른다."* Novelty (cross-modal anchoring), mechanism (encoder-level), psychology (uncertainty-proportional bias) 를 결합.
2. *"Thinking VLM은 System-1 bias를 안정적으로 탈출하지 못한다: dual-process audit."* VLMBias의 반대 증거와 화해 필요; 사용자의 데이터가 이 주장을 지지하면 highly citable headline.
3. *"Vision-encoder 아키텍처가 VLM이 상속받는 cognitive bias를 결정한다."* ViT vs. Conv ablation을 조직 축으로 활용.

셋 중 어느 것이든 "N개 VLM을 5개 cognitive bias에 대해 테스트했다" 보다 낫다.

**중간 영향: bias breadth를 줄이고, model breadth는 줄이지 말 것.** Cognitive load / distractor (포화) 와 confirmation bias (anchoring core 발견과 대체로 중복) 를 drop. Anchoring을 centerpiece로 유지. **숫자에 대한 text vs. image modality 프레이밍** 을 가장 깨끗한 multimodal-native bias로 추가 — 이건 진짜 novel이고 reviewer들은 shopping list가 아니라 targeted extension으로 볼 것. 네 번째 bias를 원하면 VLM의 base-rate neglect 또는 availability bias 를 고려, 둘 다 진짜 빈 공간, cognitive load나 salience (둘 다 붐빔) 보다 낫다.

**중간 영향: scope discipline.** 편의로 샘플된 7 모델이 아니라 **orthogonal axis (size × encoder family × thinking vs. instruction) 를 따라 6–8 모델** 을 목표. subset에서 **최소 하나의 frontier closed model** (GPT-4o 또는 Gemini 2.5) 포함 — "open model만" 불만을 안정적으로 무력화. **≥2 데이터셋** 사용: 깨끗한 counting을 위해 TallyQA와 CountBench, in-image-number 충돌 (anchor가 target image 안의 legible 숫자와 경쟁 — 특히 compelling한 조건) 을 위해 ChartQA 또는 DocVQA, reasoning을 위해 MathVista. 3–5 paraphrase를 bootstrap confidence interval과 multiple-comparison correction과 함께 실행.

**낮은 비용, 높은 credibility: 최소한의 human baseline 추가.** bias 조건 중 한두 개를 복제하는 ~50-참가자 Prolific 연구는 cognitive-science 주장을 ground하고, reviewer 질문을 제기되기 전에 답하며, 일주일에 <$500 로 가능. Psychology-프레임 논문에 대해 불균형하게 효과적.

## 현재 plan의 red flag

현재 plan의 다섯 항목은 reviewer 공격을 끈다. 첫째, 제안된 **confirmation bias ("base prediction이 3이고 anchor가 4면 더 쉽게 당겨진다")** 는 classical Wason/hypothesis-testing 의미의 confirmation bias가 아님 — anchoring의 continuity-of-anchoring 발견이며, 논문은 별개 bias가 아니라 *anchoring의 sub-analysis* 로 제시해야. 잘못된 labeling은 psychology 배경의 reviewer들이 "cognitive-science 용어의 느슨한 사용"을 flag하도록 초대. 둘째, **distractor를 동반한 cognitive load theory** 는 Idis (arXiv:2511.21397), I-ScienceQA, MVI-Bench, MM-NIAH 와 크게 중복, 모두 최근이고 구체적으로 visual distractor에 관한 것; 차별화 비용이 novelty payoff를 초과할 가능성. 셋째, **marker 를 통한 salience bias** 는 붐빈 땅 (Shtedritski red-circle 작업, FGVP, STER-VLM, spatial attention bias) 이며 더 novel한 축과 경쟁하는 상당한 차별화를 요구. 넷째, **Dual Process Theory 프레이밍은 지적으로 매력적이지만 empirically 논쟁적** — VLMBias와 LRM-judging 작업은 reasoning 모델이 *더* biased일 수 있음을 보임, 논문은 DPT를 깨끗한 predictor로 제시하기보다 이를 engage해야. 다섯째, **VQAv2 만으로는 review를 통과 못 함**; behavioral claim에 대해 EMNLP에서 multi-dataset 확증은 non-negotiable.

Flag할 추가 risk 하나: 사용자는 인용된 모든 2026 arXiv ID가 실제로 paper로 resolve되는지 검증해야 한다. Aggregator search에서 나온 여러 ID (예: 2603.xxxxx, 2604.xxxxx) 는 진짜 2026 초 preprint 일 수도 있고 search-aggregator artifact 일 수도 — 인용 전 수동 검증 필수.

## 현실적 한 달 plan

Week 1: scope를 anchoring (core) + framing (image vs. text modality) + 하나의 추가 bias, orthogonal axis를 따르는 6–8 모델, 3 데이터셋 (TallyQA, CountBench, ChartQA/DocVQA), 3 paraphrase 로 lock. 자동화된 pipeline 구축. Generation과 함께 attention-mass 추출 실행. Week 2: main-grid run 완료; 더 작은 prompt set에서 vision-encoder ablation (ConvLLaVA, EVE, 하나의 DINO-based VLM) 시작. Prolific에서 50-참가자 human baseline 런칭. Week 3: mechanistic 분석 (logit lens, attention pattern, encoder ablation) 과 mitigation prototype (contrastive decoding 또는 anchor image token에 대한 attention re-weighting). closed-model subset (GPT-4o/Gemini 2.5) 을 ~500-예제 stratified 샘플에서 실행. Week 4: writing, multiple-comparison correction과 함께 통계 검정, limitations section, reviewer-question 선제 대응. May 25 ARR deadline 제출.

## 결론: accept 확률과 이를 움직이는 것

솔직한 평가: *현재* plan — analysis-only, single-dataset, DPT 아래 느슨하게 조직된 다섯 bias, mechanism 없음, mitigation 없음 — 은 거의 동일한 최근 선례에 근거해 현실적으로 **Findings-not-Main** 결과. 핵심 현상은 충분히 novel하고 흥미로워서 Findings accept은 plausible; 현재 스코프에서 Main accept은 아니다. Main-accept 확률을 의미 있게 움직이는 lever들은, 순서대로: (1) mechanistic 분석 추가, 특히 8×H200에서 과학적으로 흥미롭고 tractable한 ViT-vs-Conv-vs-encoder-free ablation; (2) 그 분석이 가이드하는 간단한 mitigation 추가; (3) 느슨하게 선택된 다섯 개가 아니라 잘 선택된 세 개의 bias로 좁히기; (4) 내러티브를 하나의 아키텍처 또는 mechanistic 주장으로 sharpen; (5) paraphrase robustness 동반의 multi-dataset 확증. Compute는 제약이 아니다 — 설계 discipline이 제약. 현재 preliminary result에서 가장 강한 단일 지적 자산은 **asymmetric anchoring-on-wrong-cases 발견** 으로, 진짜 novel하고, 깨끗한 cognitive-science 근거가 있으며, LLM이든 VLM 문헌이든 정확한 선례가 없다. Bias catalogue가 아니라 이 발견 주변에 논문을 구축하고, 남은 compute를 *bias 간 breadth* 가 아니라 *anchoring의 mechanistic depth* 에 투자하는 것이 EMNLP 2026 Main으로 가는 최고 기대값 경로.
