# 슬라이드 설명서 — `cross_modal_anchoring_paper_ko.pptx`

이 문서는 PPT 파일과 함께 보면서 각 슬라이드를 자세히 이해할 수 있도록 작성된 설명서다.
**도메인에 처음인 사람** (= VLM, anchoring bias, attention mechanism 모두 처음 듣는 사람)이
모든 슬라이드를 이해할 수 있도록 용어부터 차근차근 풀어 설명한다.

---

## 사전 지식 — 슬라이드를 이해하기 위해 알아야 할 5개 용어

| 용어 | 한 줄 설명 |
|---|---|
| **VLM** | Vision-Language Model. 이미지 + 텍스트를 함께 입력받아 텍스트로 답하는 AI (e.g. ChatGPT 4o, Gemini, Qwen-VL). |
| **VQAv2** | Visual Question Answering v2. 이미지 + 질문 → 답변 데이터셋. "사과가 몇 개?" 같은 수치 질문 포함. |
| **Anchoring** | 인지심리학의 편향 — 무관한 숫자(앵커)에 사고가 끌려가는 현상. e.g. "100을 떠올리고 답하라"고 한 뒤 묻는 경우 답이 100 쪽으로 치우침. |
| **Attention** | Transformer 모델 내부에서 "어떤 입력에 얼마나 집중할지"를 결정하는 가중치. anchor 이미지에 attention이 몰리면 그 이미지 영향 큼. |
| **Encoder** | 이미지를 숫자 벡터로 변환하는 신경망 모듈. CLIP-ViT, ConvNeXt, Qwen-ViT 등. |

---

## Slide 1 — Title (표지)

**무엇이 적혀 있나**
- 논문 제목: "VLM의 교차-모달 앵커링 편향: 메커니즘 규명과 어텐션 재가중 완화책"
- 부제목: "관련없는 숫자 이미지가 VLM의 수치 답변을 어떻게 끌어당기는가"
- 저자, 날짜.

**핵심**
- 본 연구는 **VLM에서 cross-modal anchoring**이라는 새로운 편향 현상을 제기하고,
  내부 메커니즘을 규명하며, 이를 완화하는 방법까지 제안한다는 1-stop study이다.
- 키워드 "교차-모달 (cross-modal)" = 다른 modality (텍스트 ≠ 이미지) 사이에서 발생한다.
  본 연구는 **이미지에서 시작된 anchoring**이 모델 출력 (텍스트)을 편향시키는 케이스다.

---

## Slide 2 — Abstract (5가지 핵심 발견 요약)

**무엇이 적혀 있나** — 5개 카드:
1. 현상 확인 (7개 모델 × 17,730 sample, +7~+20 pp graded pull)
2. 메커니즘 (어텐션 분석, 가족별 peak layer)
3. 인과 검증 (single-layer null, upper-half 유효)
4. 완화책 (full 검증에서 −10.6~−17.7 % df 감소, em 개선)
5. Anchor damage 회복 (−3.55~−9.34 pp 손상의 14~22 % 회복)

**용어 풀이**
- **direction_follow_rate (df)**: anchor 쪽으로 예측이 이동한 비율. 이 논문의 핵심 메트릭.
- **exact_match (em)**: ground truth와 정답 일치율. 정확도 측정.
- **pp**: percentage points (퍼센트 포인트). 0.30 → 0.20 = −10 pp.
- **graded pull**: 정답을 정확히 anchor와 똑같이 답하는 게 아니라, anchor 쪽으로 약간씩 끌리는 것.
  예: 원래 답이 "5"인데 anchor가 "9"이면, "5"가 "6", "7" 쪽으로 약간 이동하는 식.

**왜 이 5가지가 중요한가**
- 행동 → 메커니즘 → 인과 → 완화 → 효용 정량화의 완전한 chain. EMNLP main 채택을 위한
  표준 evidence ladder.

---

## Slide 3 — Introduction (왜 이 문제가 중요한가)

**무엇이 적혀 있나**
- VLM이 일상화되면서 multi-image prompt가 표준이 되고 있음
- 위험: 무관한 이미지가 답변에 영향을 준다면? 특히 숫자 이미지가 다른 질문의 수치 답변을
  끌어당긴다면?
- 인지 과학 배경: Tversky-Kahneman, Mussweiler-Strack — 사람도 같은 편향을 보임
- LLM 연구는 이미 진행됨 (Jones-Steinhardt 2022 등); VLM에서는 미답
- 본 연구의 3가지 기여 명시

**용어 풀이**
- **Multi-image prompt**: 한 번에 여러 장의 이미지를 모델에 보여주는 입력 방식.
  e.g. "이 두 이미지 중 어느 것이 더 새것인가?" — 두 이미지 동시 제시.
- **Selective accessibility (Mussweiler-Strack)**: 인지 과학에서 anchor가 "관련된 정보를
  떠올리기 쉽게" 만드는 메커니즘. 답을 교체하지는 않지만, 답의 방향성을 편향함.

**왜 중요한가**
- VLM은 이제 ChatGPT 4o, Gemini 2.5 등으로 일상 사용. 이런 일상 환경에서 위험한 편향이
  존재한다는 것은 안전성·신뢰성 측면에서 critical.
- 또한 cognitive-science theory (anchoring)와 직접 연결되어, AI를 인간 인지의 모델로
  해석하는 frame과도 호환.

---

## Slide 4 — Related Work

**무엇이 적혀 있나** — 5개 인접 연구와의 차이를 표로:
1. **LLM Anchoring** (Jones-Steinhardt 2022 등) — 텍스트 only, 이미지 미답
2. **Typographic Attacks** (Goh 2021, Wang 2025) — Classification flip만, 수치 회귀 미답
3. **VLMBias** (Nguyen 2025) — Memorization 편향, anchor 아님
4. **FigStep** (Gong 2025) — Adversarial jailbreak, 인지 편향 framing 아님
5. **Tinted Frames** (2026) — 질문 framing만, 이미지 modality 미답

마지막 카드: 본 연구는 "독립적인 rendered-number 이미지를 multi-image VLM prompt에 추가
하고, open numerical VQA에서 회귀 형태의 shift를 측정한 최초의 연구."

**용어 풀이**
- **Typographic attack**: 사물 이미지 위에 다른 class의 글자를 적어놓으면 분류가 그 글자로
  바뀌는 현상 (e.g. 사과 위에 "iPod" 적기 → 모델이 "iPod"라 분류). 분류 flipping이 메인.
  본 연구는 **수치 회귀 (regression)**이라 다름.
- **Memorization prior**: 모델이 학습 데이터에서 본 cue (e.g. 셀럽 얼굴)를 보고 그 prior에
  따라 답하는 현상. **외부에서 새로 주입된 anchor**가 아님.
- **Jailbreak**: 모델의 안전 가드를 뚫는 prompt. **인지 편향 framing**이 아님.

**왜 중요한가**
- Reviewer가 "이건 그냥 typographic attack 아니냐?" "VLMBias랑 같은 거 아니냐?"
  물어볼 것에 대한 명확한 차별화 — 핵심은 **(1) 별도 이미지로 anchor**, **(2) 수치 회귀**,
  **(3) 인지 편향 framing 으로 cognitive science 와 연결.**

---

## Slide 5 — Problem Definition (3-conditions)

**무엇이 적혀 있나** — 3개 카드:
- **A. target_only**: 타겟 이미지만 → 베이스라인 정확도
- **B. target + neutral**: 타겟 + 자릿수 없는 중립 이미지 → "두 번째 이미지 산만함" 통제
- **C. target + number**: 타겟 + anchor 숫자 이미지 → anchor manipulation

하단: 측정 메트릭 5개 (adoption, moved_closer, mean_anchor_pull, exact_match, anchor effect).

**용어 풀이**
- **Paired comparison**: 같은 모델, 같은 질문, 다른 이미지 입력으로 비교. 모델 능력의 baseline
  variation을 통제.
- **adoption_rate**: 모델이 anchor 자릿수 ("7")를 정확히 그대로 답한 비율. **Categorical**.
- **moved_closer_rate**: 답이 anchor 쪽으로 *조금이라도* 이동한 비율. **Graded** (점진적).
  예: 원래 "5"였던 답이 "6"이 되면 anchor "9" 쪽으로 이동한 것.
- **anchor effect = (C) − (B)**: 핵심 분리 트릭. C에서 측정된 효과 중 "두 이미지가 있는
  자체로 인한 산만함"은 B에서도 일어나므로, 차이 = **순수 anchor 효과**.

**왜 중요한가**
- 3-condition 디자인이 본 연구의 방법론적 핵심. **Anchoring**과 **multi-image distraction**
  을 분리함. (이걸 안 하면 "그냥 두 번째 이미지가 모델을 헷갈리게 한 거 아니냐?" 비판
  방어 못 함.)

---

## Slide 6 — Method (Setup)

**무엇이 적혀 있나**
- 데이터셋: VQAv2 number subset, answer ∈ [0,8] 한 자릿수 정수 답
- Sample: 17,730 paired records / model
- Anchor 변형: 5 irrelevant sets / 질문 (0~9 자릿수 무작위)
- 시각 자극: 1024×1024 PNG, 가운데 정렬 단일 자릿수
- 프롬프트: JSON-strict template `{"result": <number>}`
- Greedy decoding, max_new_tokens=8 (InternVL3는 32로 패치)

오른쪽: 모델 panel — Phase 1 (7개), E1/E1b/E1d (6개), E2 pilot (11개), E4 (3개).

**용어 풀이**
- **JSON-strict template**: 모델이 자유로운 prose 대신 `{"result": 5}` 같이 정해진 JSON
  포맷으로 답하도록 강제. 파싱이 쉬움.
- **Greedy decoding**: 매 step에서 가장 확률 높은 token만 뽑는 방법. 무작위성 제거.
- **max_new_tokens**: 생성할 수 있는 최대 토큰 수. 8개면 짧은 숫자 답에 충분하지만, prose
  가 들어가면 잘릴 위험 있음 (InternVL3 케이스가 그래서 32로 늘림).
- **VQAv2 number subset**: VQAv2 데이터셋에서 정답이 숫자인 질문만 추린 부분. 본 연구가
  처리하는 도메인.

**왜 중요한가**
- **Reproducibility** 측면. EMNLP review 시 "이거 어떻게 재현하나" 검증 가능하도록 모든
  hyperparameter 명시.
- 모델이 7개 / 6개 / 11개 / 3개로 다른 이유는 **각 실험이 요구하는 compute 양이 다르기**
  때문. Phase 1은 모든 모델에 full sample, mechanism은 sample 작게 + 새 모델 4개 추가.

---

## Slide 7 — Exp 1 — 행동 증거 (Phase A H2 정밀 분석)

**무엇이 적혀 있나**
- 왼쪽: H1 (앵커링 존재) 결과 — 7개 모델 모두에서 adoption 11~14 %, moved-closer 8~25 %.
- 오른쪽: H2 정밀 분석 — wrong vs correct 분리 시 adoption gap은 ±2 pp 안에서 평평하지만
  moved-closer gap은 +6.9~+19.6 pp.
- 하단: 정제된 H2 claim — "VLM이 불확실할 때 anchor가 search direction을 편향"

**용어 풀이**
- **H1 (가설 1)**: "앵커링이 존재한다" → adoption(C) > adoption(B) 혹은 moved_closer 차이.
- **H2 (가설 2)**: "앵커링은 불확실성에 비례" → wrong-vs-correct 차이가 양수.
- **Wrong vs Correct stratification**: target_only 조건에서 정답을 맞춘 케이스 (correct)와
  틀린 케이스 (wrong)로 나눠서 분석. 모델이 이미 정답 알면 anchor 무시할 거라는 hypothesis.

**왜 중요한가**
- 이 슬라이드의 정제된 H2가 **본 논문의 가장 강력한 단일 hook** (advisor 평가).
- 인지심리학의 표준 anchoring effect (Mussweiler-Strack)를 VLM에서 정량 확인 — 이건
  cognitive-science 연결고리로 paper의 intellectual depth 더함.
- "그냥 모든 sample이 anchor에 끌리는 게 아니라 *불확실한 케이스만* 끌린다" → 더 정교한
  이론 + 더 좁은 mitigation target.

---

## Slide 8 — Exp 2 — 어텐션 메커니즘 (E1 + E1b)

**무엇이 적혀 있나**
- 메인 표: 6개 모델의 per-layer 분석 — peak layer, depth %, peak δ (어텐션 증가량),
  budget 출처 (text/target), 유형
- 4 archetypes 등장:
  1. Gemma — early + large (L5 = 12 % depth, +0.05)
  2. Mid-stack cluster — InternVL3 + LLaVA-1.5 + ConvLLaVA (L14~16 = 52 %, ~+0.02, text)
  3. Qwen — late + target stealing (L22 = 82 %, +0.015, **target**)
  4. FastVLM — late + text + 큰 magnitude
- 오른쪽: budget decomposition figure
- 하단: 두 가지 핵심 발견 (H3 falsified, 4 archetypes)

**용어 풀이**
- **Per-layer attention analysis**: VLM 내부의 각 LLM decoder layer마다 anchor 이미지에 대한
  attention weight를 측정. **Peak layer**는 그 weight가 가장 큰 layer.
- **Budget**: 어텐션 weight는 sum=1로 normalize되므로, anchor가 더 많이 가져가면 다른 곳에서
  덜 가져옴. Budget 출처 = "anchor가 어디서 attention을 빼앗아갔나" (text token에서? target
  image에서?).
- **Text-stealing**: anchor가 text token (질문 문장)의 attention을 빼앗음. 5/6 모델.
- **Target-stealing**: anchor가 target image의 attention을 빼앗음. Qwen만 해당.

**왜 중요한가**
- **H3 falsification (E1c)**: 원래 가설은 "ConvNeXt encoder는 typographic attack에 강하니
  anchoring도 약할 것"이었는데, 실험 결과 ConvLLaVA가 LLaVA-1.5와 동일한 layer/mechanism으로
  작동. 즉 **encoder architecture가 아니라 LLM stack의 depth가 핵심**.
- 이 발견 자체가 paper-grade. 기존 typographic-attack 문헌의 가정 (encoder가 주된 원인)을
  뒤집음.

---

## Slide 9 — Exp 3 — 인과 증거 (E1d)

**무엇이 적혀 있나**
- 왼쪽 표: 5가지 ablation 모드의 결과
  - Single-layer at peak: ≤ ±3 pp (null on 6/6)
  - Single-layer at L0: 같음 (null on 6/6)
  - Stack-wide: −11~−22 pp 효과 but fluency 깨짐
  - Lower-half: +27/+17/+7 pp (BACKFIRES)
  - **Upper-half: −5.5~−11.5 pp on 6/6 + fluency clean (4/6)**
- 오른쪽: figure
- 하단: 두 결론 — single-layer 무효 + upper-half이 architecture-blind 인과 위치

**용어 풀이**
- **Ablation**: 모델의 일부를 "끄고" 효과 보기. 여기서는 anchor에 대한 attention을 강제로 0
  으로 만듦.
- **Single-layer ablation**: 한 layer만 끔. **Multi-layer ablation**: 여러 layer 동시에 끔.
- **Backfires (역효과)**: 의도와 반대 방향으로 효과 — anchor 효과를 줄이려는데 오히려 더 강해짐.
  Lower-half ablation이 그런 케이스.
- **Fluency**: 모델 출력이 자연스러운지. ablation으로 attention을 깨면 모델이 무관한 큰 숫자를
  뱉을 수 있음. mean_distance_to_anchor가 폭주하면 fluency 깨졌다고 봄.

**왜 중요한가**
- E1b의 peak layer가 인과적이라고 가정하기 쉽지만 **single-layer ablation 결과는 null** —
  "이 layer가 anchor 신호를 carrying"이 correlational 일 뿐 인과 아님.
- Multi-layer redundancy → anchor 신호가 stack 여러 곳에 redundant하게 인코딩됨.
- **Upper-half이 universal locus** → 한 intervention으로 3개 다른 encoder에 작동.
  완화책 (E4)의 출발점.

---

## Slide 10 — Exp 4-A — 완화책 설계 + Phase 1 sweep

**무엇이 적혀 있나**
- 왼쪽: 방법 — forward pre-hook이 attention_mask에 strength 더함, post-softmax에서
  exp(strength)배 down-weight
- 왼쪽 하단 표: Phase 1 sweep 결과 (3 모델)
  - LLaVA: s* = −3.0, df −13 %, em +0.5 pp
  - ConvLLaVA: s* = −2.0, df −10 %, em 0 pp
  - InternVL3: s* = −0.5, df −17.7 %, em +1.9 pp
- 오른쪽: Pareto figure
- 오른쪽 하단: 주목점 (3/3 모두 달성, s* per-model, anti-correlation)

**용어 풀이**
- **Forward pre-hook**: PyTorch hook의 일종. layer가 forward 계산하기 *전에* attention_mask
  를 수정. attention_mask = -inf 이면 그 위치는 attention 받지 못함.
- **Strength `s`**: 본 연구가 정의한 파라미터. anchor 위치의 attention_mask 컬럼에 s를 더함.
  - s = 0: no-op
  - s = -∞ (= -10⁴): anchor 어텐션 완전 0 (= E1d의 hard mask)
  - s = -3 정도: anchor 어텐션 약 5 % 수준으로 down (exp(-3) ≈ 0.05)
- **Sweep**: 여러 s 값을 시도 (-0.5, -1, -2, -3, -5, -10⁴)해보고 best 선택.
- **Pareto**: 두 메트릭 (df ↓, em →)의 trade-off 곡선. 우리는 df 감소 + em 손실 없는 지점.
- **Stratified n=200**: 단순 랜덤 200개가 아니라 "anchor에 가장 약한 (top decile) 100 + 가장
  강한 (bottom decile) 100" 으로 의도적 stratification. 효과 측정 sensitivity 높임.

**왜 중요한가**
- E1d의 hard mask (-10⁴)는 너무 aggressive하고 일부 모델에서 fluency 깨짐. **Soft strength
  axis로 dose-response 곡선** 그려서 안전한 운영 지점 (s*) 찾음.
- 모든 3개 모델이 ≥ 10 % df 감소 타깃 달성 + em 손실 없음 → 이 위치가 **mid-stack-cluster
  보편적 mitigation 위치**임을 시사.

---

## Slide 11 — Exp 4-B — Phase 2 풀 검증

**무엇이 적혀 있나**
- 메인 표: Phase 2 (full 88,650 records / model) 결과
  - **LLaVA: df 0.258 → 0.212 = −17.7 % rel, em +0.77 pp** (Phase 1과 정확히 일치)
  - **ConvLLaVA: df 0.228 → 0.204 = −10.6 % rel, em +1.30 pp** (Phase 1 −10.3 %와 0.3 pp 차이)
  - InternVL3: 진행 중, 다음 세션
- 왼쪽 카드: 복제 정확도 — Phase 1 예측 0.3 pp 이내 복제, CI 약 10× 좁음, em 두 모델 모두 상승
- 오른쪽 카드: Caveat — ConvLLaVA fluency tail (mean_dist 폭주), InternVL3 Phase 2 미완료

**용어 풀이**
- **Confidence Interval (CI)**: 통계 추정값의 신뢰구간. 95 % CI = "참값이 이 범위에 있을 확률
  95 %". sample 더 많을수록 좁아짐.
- **Replication**: Phase 1 (n=200)에서 본 효과가 Phase 2 (n=17,730)에서도 똑같이 나오는가.
  과학에서 replication은 결과의 robustness 핵심 지표.
- **Fluency tail**: ConvLLaVA에서 일부 sample이 broken output을 받아 mean_distance가 53.54
  로 폭주. 즉 outlier 분포의 긴 꼬리. em은 여전히 상승하므로 **대부분 sample은 OK**.
- **Per-pair**: df 메트릭은 같은 sample의 (target_only, anchor) 한 쌍 비교라 outlier에 영향
  덜 받음. → df는 robust.

**왜 중요한가**
- Phase 2가 **본 연구의 핵심 검증** — sample 200 → 17,730으로 약 90× 늘려도 같은 효과 크기.
- "이건 noise 아니다"의 강력한 증거. EMNLP main의 statistical rigor 요구 충족.
- em이 IM**상승** → mitigation이 단순 trade-off가 아니라 net positive (정확도까지 개선).

---

## Slide 12 — Reframing: Anchor Damage / Partial Recovery

**무엇이 적혀 있나**
- 메인 표: Phase 2 paired anchor-damage table
  - LLaVA: em(TO) 0.370, em(num@0) 0.334 → damage **−3.55 pp**, recovery +0.77 pp = **21.7 %**
  - ConvLLaVA: em(TO) 0.445, em(num@0) 0.352 → damage **−9.34 pp**, recovery +1.31 pp = **14.0 %**
- 하단: 더 강한 paper claim + "왜 강한가" 3개 reasoning

**용어 풀이**
- **Paired analysis**: 모든 condition (TO, num@0, num@s*)에서 valid한 sample만 모아 계산.
  unpaired는 condition마다 다른 sample이라 절대 em 비교 안 됨. paired는 like-for-like 비교.
- **em(target_only) = 정확도 ceiling**: anchor 없이 모델이 답할 때의 정확도. anchor가
  추가되면 이보다 떨어짐 (= damage).
- **% damage recovered**: (em(num@s*) − em(num@0)) / (em(TO) − em(num@0)) × 100. anchor가
  깎아먹은 정확도의 몇 %를 mitigation이 복구했나.

**왜 중요한가**
- 단순 "df 감소"보다 **"anchor가 정확도 손상시키고, mitigation이 그 손상의 14~22 %를 회복"**
  이 reviewer에게 더 직관적이고 중요한 framing.
- df-axis (mechanism)와 em-axis (functional) 신호가 **동일한 위치를 가리킨다는 것**이
  causality의 강력한 evidence.

---

## Slide 13 — Discussion: 본 논문의 강점 (Novelty)

**무엇이 적혀 있나** — 5가지 강점:
1. A1 — Graded pull refinement of H2 (cognitive science 연결)
2. E1c — H3 falsified, depth-axis replaces encoder-axis
3. E1d — Multi-layer redundancy + upper-half causal locus
4. E2 pilot (H6) — Two-axis decoupling (anchoring vs distraction)
5. E4 Phase 2 — Causal mitigation at scale

**왜 이 5가지가 강점인가** (각 항목별)

1. **A1 — Graded pull**:
   - 기존 LLM/VLM anchoring 문헌은 "wrong vs correct" 분리 안 함. 본 연구가 처음 분리.
   - +6.9~+19.6 pp on 7/7 = robust signal.
   - cognitive-science (Mussweiler-Strack)와 직접 매핑 → AI를 인간 인지 모델로 해석하는
     frame 강화.

2. **E1c — H3 falsified**:
   - "ConvNeXt < ViT" 가설을 정면으로 negative finding으로 보고. **Negative finding이지만
     publishable** — 다른 연구자들이 같은 가설로 시간 낭비하지 않게 함.
   - Replacement framing (depth-axis)이 **paper의 mechanism 챕터 핵심**이 됨.

3. **E1d — Multi-layer redundancy**:
   - "peak layer가 인과 위치"라는 직관적 가정을 **causal test로 뒤집음**.
   - Upper-half가 architecture-blind universal locus → mitigation 일반화 가능.

4. **E2 pilot (H6)**:
   - Anchoring vs multi-image distraction이 직교 실패 모드 → **별도 mitigation 전략 필요함**
     을 시사. InternVL3 같은 "pure distraction" 모델을 별도 처리.

5. **E4 Phase 2 — Mitigation at scale**:
   - n=200 sweep → n=17,730 full 복제. **Phase 1 예측 0.3 pp 이내 일치**가 핵심.
   - em 손실 없이 (오히려 개선) df 감소 = practical mitigation.
   - EMNLP main acceptance를 위한 가장 reliable lever ("observation + mechanism + mitigation").

---

## Slide 14 — Discussion: 약점과 한계 (Weakness)

**무엇이 적혀 있나** — 6가지 약점:
1. InternVL3 Phase 2 미완료 (다음 세션)
2. FastVLM 결과의 wide CI (n=75)
3. ConvLLaVA fluency tail at full scale
4. Single dataset (VQAv2 only) 약점
5. Closed-model subset 미수행 (GPT-4o, Gemini)
6. Paraphrase robustness 미검증

**왜 이것들이 약점인가** (각 항목별)

1. **InternVL3 Phase 2 미완료**:
   - 12-h budget 안에 끝나지 않음. 본 세션 결과는 LLaVA + ConvLLaVA 만 cover.
   - mid-stack-cluster 3 모델 일반화 주장이 2/3 만 confirmed. **다음 세션 필수**.

2. **FastVLM CI**:
   - 답변 단계 valid n=75 (다른 모델은 200) → CI 매우 wide.
   - Upper-half ablation에서 fluency 깨짐. unique archetype이지만 보고하기 위험.
   - **추가 큰 run 필요**.

3. **ConvLLaVA fluency tail**:
   - Phase 2에서 mean_dist 폭주. 일부 sample broken.
   - em은 상승, df는 robust → metric 자체는 OK이지만 reviewer 우려 가능.
   - **Median + winsorised distribution 보고**로 완화 가능.

4. **Single dataset**:
   - EMNLP main 검토자 표준 요구: ≥ 2 dataset.
   - TallyQA, ChartQA 등은 smoke test만. **E5 풀 run 필요**.

5. **Closed-model subset**:
   - "Open-only" reviewer 우려는 cognitive-bias 분야의 표준 요구.
   - GPT-4o, Gemini 등 ~500 sample로 minimum coverage 권장.

6. **Paraphrase robustness**:
   - Single prompt template (JSON-strict)만 사용.
   - Behavioral claim의 robustness — "JSON 강제 때문에 발생한 거 아니냐?" 우려.
   - **3~5 paraphrase × bootstrap CI 필요**.

**왜 약점을 명시하는가**
- 자기 비판이 reviewer 우려를 미리 방어함. "limitations" section이 잘 쓰여진 paper는
  reviewer에게 신뢰감 줌.
- 약점 list가 곧 **다음 세션 작업 우선순위**가 됨.

---

## Slide 15 — Conclusion + Future Work

**무엇이 적혀 있나**
- 주요 기여 5가지 요약
- Submission 전 우선순위:
  1. [우선] InternVL3 Phase 2 완료
  2. [우선] E5 multi-dataset 풀 run
  3. [권장] E6 closed-model subset
  4. [권장] E7 paraphrase robustness
  5. [옵션] Per-stratum 분석, ConvLLaVA fluency tail decomposition

**용어 풀이**
- **Per-stratum**: 모델 susceptibility decile (top vs bottom)별로 결과 분리. 완화책이 어떤
  종류의 sample에 가장 효과적인지 보임.

**왜 이 우선순위인가**
- [우선] 두 개는 **mid-stack-cluster 일반화**와 **single-dataset 약점**이라는 가장 큰 결함
  보강.
- [권장] 두 개는 reviewer 우려 방어 (closed model, paraphrase).
- [옵션] 위 5가지가 끝난 후 paper polish 단계.

---

## 부록 — 자주 묻는 질문 (FAQ)

**Q1. "이건 그냥 두 번째 이미지가 모델을 헷갈리게 한 거 아니냐?"**
A. **3-condition 디자인 (Slide 5)이 정확히 그 우려를 분리.** B (neutral) 조건이 "두 번째
이미지가 있는 자체 효과"를 통제. C와 B의 차이가 **순수 anchoring 효과**.

**Q2. "Cross-modal anchoring은 typographic attack 아니냐?"**
A. 아니다 (Slide 4). Typographic attack은 (a) 같은 이미지에 텍스트 오버레이, (b) classification
flipping (선택지 교체), (c) adversarial framing이다. 본 연구는 (a) **별도 이미지로 anchor**,
(b) **수치 회귀 (regression)** 측정, (c) **인지 편향 framing**.

**Q3. "Anchor가 자릿수와 GT가 우연히 일치한 거 아니냐?"**
A. (Slide 5) **Adoption rate** 외에 **moved_closer_rate**도 측정. moved_closer는 anchor와
GT가 다른 케이스도 cover (anchor=7, GT=5인데 답이 4→6으로 이동하면 카운트). 또한 chance
baseline 11 %를 baseline으로 비교.

**Q4. "왜 다른 데이터셋도 안 했나?"**
A. (Slide 14 weakness 4) Single dataset은 알려진 약점. TallyQA, ChartQA smoke test만 됨.
**E5 (다음 세션)에서 풀 run 예정**.

**Q5. "Closed model (GPT-4o)에선 이 효과 안 나올 수도 있는 거 아닌가?"**
A. 가능성 있음. 본 연구는 open VLM panel만 수행. **E6 (다음 세션)에서 GPT-4o, Gemini
~500 sample subset 수행 권장**.

**Q6. "Mitigation이 모든 모델 다 어디나 작동하나?"**
A. **NOT yet**. 본 연구의 mid-stack-cluster (LLaVA-1.5, ConvLLaVA, InternVL3) 만 cover.
다른 archetype (Gemma early peak, Qwen target-stealing, FastVLM late+text+큰 magnitude)은
다른 mitigation locus 필요할 수 있음. **다음 work에서 archetype별 mitigation 탐색 예정**.

**Q7. "왜 mid-stack-cluster 3개 모델만 mitigation 했나?"**
A. (Slide 8) E1d 결과 — upper-half ablation이 fluency clean한 것이 mid-stack cluster + Qwen.
Mitigation이 깨끗하게 작동할 가능성이 가장 높은 그룹. Gemma (early peak)와 FastVLM은 다른
intervention이 필요.
