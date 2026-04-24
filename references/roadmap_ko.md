# references/roadmap_ko.md — VLM의 cross-modal anchoring

**Single source of truth.** `vlm_anchroing/`에서 작업하기 전에 가장 먼저 이 파일을 읽습니다. task가 끝날 때마다 §3 (status), §5/§6 (체크리스트), §10 (changelog)를 갱신합니다. 이 내용을 다른 doc에 복제하지 말고, 항상 본 파일의 섹션을 link로 참조합니다. *(영문 canonical: `references/roadmap.md`. 항상 영문본을 먼저 갱신하고 그 다음 본 파일을 동기화합니다.)*

- **목표 venue:** EMNLP 2026 Main (현재 scope는 `references/project.md` 평가에 따르면 Findings-tier 수준; 본 roadmap은 plan이 권고하는 Main-tier 이행 동선을 중심으로 짜여 있음).
- **Compute envelope:** 8×H200 (그중 1장은 vLLM Qwen2.5-32B 서버와 공유 → 실효 60 GB/장), 1개월.
- **참고 doc:** `references/project.md` 는 candid feasibility review. 그것을 prior로 깔고, 이 파일은 그 위에서 굴러가는 운영 plan.

---

## 1. Research definition

**현상.** VLM에 numerical VQA 질문 + 타겟 이미지를 줍니다. *추가로* digit이 렌더링된 irrelevant 이미지(= **anchor**)를 두 번째 이미지로 함께 줍니다. 이 anchor가 모델의 numeric answer를 체계적으로 끌어당기는가?

**Within-sample 3 conditions** (`build_conditions` in `data.py`):

| Condition | Inputs | 역할 |
|---|---|---|
| `target_only` | 타겟 이미지 | 베이스라인 정확도 |
| `target_plus_irrelevant_neutral` | 타겟 + digit이 **없는** irrelevant 이미지 | "두 번째 이미지가 distract" 효과 통제 |
| `target_plus_irrelevant_number` | 타겟 + digit 1개가 들어있는 이미지 | anchor manipulation |

Neutral arm은 **anchoring**과 **단순 distraction**을 분리합니다. `number > neutral`로 보이는 모든 차이가 anchoring signal.

**왜 중요한가.** Multi-image VLM prompt에서 standalone-rendered-number를 cross-modal anchor로 두고 open numerical VQA에 대한 regression-style shift를 측정한 선행 연구는 없습니다. 가장 가까운 이웃 (VLMBias, Typographic Attacks, FigStep, Tinted Frames, LLM-anchoring lineage)은 모두 적어도 한 개 핵심 축에서 다릅니다. novelty 매트릭스는 `references/project.md` §1 참조.

## 2. Hypotheses (반증 가능한 예측 기준)

| ID | 가설 | Falsifier | 현재 증거 |
|---|---|---|---|
| **H1** | Anchor digit이 neutral-distractor baseline 이상으로 prediction을 끈다. | `direction_follow_rate(number)` ≤ chance(0.5) **그리고** `mean_distance_to_anchor(number)` ≈ random pairing baseline. | ✅ main run 7개 모델 전부: direction-follow ∈ [0.247, 0.348], adoption ∈ [0.110, 0.140]. |
| **H2** | Anchoring은 **비대칭**이다: 모델이 원래 틀렸던 항목(= 주관적 불확실성 proxy)에서 더 강하다. | `target_only`-correctness로 stratify하여 adoption_wrong > adoption_correct 기대. 같으면 H2 fail. | ✅ **재정의됨.** A1 (Phase A): adoption gap은 ≈ 0이지만, *graded* `moved_closer_rate` gap은 모든 모델에서 wrong > correct로 **+6.9 ~ +19.6 pp**. Bias는 uncertainty-modulated *graded pull*이지 categorical capture가 아님. `docs/insights/A1-asymmetric-on-wrong.md` 참조. |
| **H3** | Vision-encoder family가 susceptibility를 modulate한다. ConvNeXt/encoder-free는 CLIP/SigLIP-ViT보다 *덜* 취약 (typographic-attack 상속). | ConvLLaVA / EVE / DINO-VLM의 direction-follow gap이 CLIP-ViT와 통계적 동등이면 fail. | ⚠️ **Pilot (2026-04-24) 결과 단순 형태로는 지지 안 됨.** ConvLLaVA-7B `adoption=0.156`이 CLIP/SigLIP 클러스터 CI 안. `docs/experiments/E2-pilot-results.md` 참조. **아래 H6로 대체.** |
| **H4** | "Thinking"/instruction-tuned reasoning이 anchoring을 줄인다 (System-2 suppression). | 같은 모델 family에서 reasoning on/off가 동일한 direction-follow를 보이면 fail. | ❌ Untested. *주의*: VLMBias + LRM-judging 문헌은 reasoning이 일부 bias를 *증폭*시킬 수 있다고 보고 — 실험은 방향에 agnostic하게 짜야 함. |
| **H5** | Prompt 강화 ("number를 출력해라, hedge하지 마") → uncertain item에서 anchor pull 증가, 다른 item에서는 large-number hallucination 유발. | item별로 `experiment_anchor_strengthen_prompt`와 `experiment` 비교. | ⚠️ Suggestive: strengthen run에서 `mean_distance_to_anchor`가 gemma3-27b-it에 대해 2617 (같은 모델 standard run에서는 4.4)에 도달 → "must answer" 압력 하에서 모델이 huge number를 fabricate. 적절한 분석 필요. |
| **H6** | Cross-modal failure는 두 직교 축으로 분해된다: **anchor-pull** (uncertainty-modulated, encoder-mediated) + **multi-image distraction** (encoder-architecture-mediated, anchor를 encode하지 않고 정확도만 깎음). 다른 encoder family는 이 2D plane 위 다른 위치에 위치. | `adoption_rate`와 `acc_drop_vs_target_only`가 완전 상관(= single failure mode)이면 H6 fail. | ✅ **Pilot 강하게 시사.** InternVL3 = high acc_drop / low adoption; LLaVA-1.5 = low acc_drop / high adoption; ConvLLaVA = both. 두 축 decoupling이 새 headline candidate. Full-run CIs 필요. |

## 3. Status — 무엇이 돌아갔나

### 3.1 완료된 full run (각 17,730 샘플)

| Experiment | Models | 출력 위치 |
|---|---|---|
| `experiment` (standard prompt, VQAv2 number subset) | gemma3-27b-it, gemma4-31b-it, gemma4-e4b, llava-next-interleaved-7b, qwen2.5-vl-7b-instruct, qwen3-vl-8b-instruct, qwen3-vl-30b-it | `outputs/experiment/<model>/<run>/` |
| `experiment_anchor_strengthen_prompt` (no-hedging system prompt) | 동일 7개 | `outputs/experiment_anchor_strengthen_prompt/<model>/<run>/` |

Per-record 필드 (`predictions.jsonl` 키 참조): `condition`, `irrelevant_type`, `anchor_value`, `prediction`, `ground_truth`, `standard_vqa_accuracy`, `exact_match`, `anchor_adopted`, `anchor_direction_followed`, `numeric_distance_to_anchor`. Per-token logits도 캡처됨 (commit `5f925b2`).

### 3.2 Smoke run만 있는 것 (50 샘플)

| Experiment | Models | 출력 위치 |
|---|---|---|
| `experiment_tallyqa` | qwen2.5-vl-7b-instruct, qwen3-vl-8b-instruct, llava-next-interleaved-7b | `outputs/experiment_tallyqa/` |
| `experiment_chartqa` | qwen2.5-vl-7b-instruct (단독) | `outputs/experiment_chartqa/` |
| `experiment_mathvista` | qwen2.5-vl-7b-instruct (단독) | `outputs/experiment_mathvista/` |

### 3.3 통합은 됐지만 full run 없음

`build_runner`에 wired되어 있고 smoke 통과. Pilot (`outputs/experiment_encoder_pilot/`, 각 1,125 sample-instance, 2026-04-24) 완료; full 17,730 run은 사용자 signoff 대기:

- `llava-1.5-7b` — CLIP-ViT vanilla baseline. Pilot adoption=0.181 (11개 모델 중 최고).
- `internvl3-8b` — InternViT family. Pilot adoption=0.066 (최저), acc_drop=0.355 (최고). "distraction-not-anchoring" outlier.
- `fastvlm-7b` — Apple FastViT. Pilot acc=0.483 (낮음 — parse-rescue rate 검증 필요).
- `convllava-7b` — pure ConvNeXt encoder. Pilot adoption=0.156 (LLaVA-1.5와 근접; H3 단순 형태 지지 안 됨).

### 3.4 완료된 run의 헤드라인 수치

Standard prompt, VQAv2 number subset, 모델당 17,730 샘플. `direction_follow`는 `target_only` 답변 대비 anchor 쪽으로 prediction이 움직인 비율.

| Model | acc(target_only) | acc(neutral) | acc(number) | adoption(number) | direction_follow(number) | mean dist→anchor |
|---|---:|---:|---:|---:|---:|---:|
| gemma4-e4b | 0.553 | 0.505 | 0.541 | 0.123 | 0.320 | 4.17 |
| llava-interleave-7b | 0.619 | 0.577 | 0.576 | 0.134 | 0.348 | 3.20 |
| gemma3-27b-it | 0.628 | 0.623 | 0.633 | 0.141 | 0.305 | 4.45 |
| qwen2.5-vl-7b | 0.736 | 0.708 | 0.711 | 0.110 | 0.248 | 5.03 |
| qwen3-vl-8b | 0.751 | 0.709 | 0.715 | 0.127 | 0.258 | 3.43 |
| gemma4-31b-it | 0.749 | 0.723 | 0.741 | 0.116 | 0.239 | 6.16 |
| qwen3-vl-30b-it | 0.759 | 0.709 | 0.707 | 0.120 | 0.280 | 3.70 |

추가 분석 없이 즉시 보이는 두 패턴:
1. **Direction-follow ≈ 24–35 %** 인데 adoption (anchor와 정확히 같음)은 11–14 %에 그침. Bias가 *gradient*이지 categorical이 아님 — 대부분 케이스가 anchor "쪽으로 끌려"가지 anchor "로 setting"되지 않음. anchoring effect가 남길 정확히 그 signature.
2. **강한 모델일수록 direction-follow는 *덜***, 그러나 anchor digit 채택률은 비슷. Lou & Sun (2024) "stronger LLMs anchor more"의 반대 — 단정 전에 per-pair 재분석 필요.

### 3.5 Strengthen-prompt anomaly (preliminary)

`experiment_anchor_strengthen_prompt` 하에서 세 Gemma 모델이 비정상적인 `mean_distance_to_anchor`: gemma3-27b-it 2617.13, qwen2.5-vl-7b 1519.91, gemma4-31b-it 511.75 (standard prompt에서는 3–6). "must output a number" instruction이 large-number hallucination을 유발한 것이지 *anchor adoption*이 아님. 정량적 claim에 쓰기 전에 trim하거나 robust 통계로 보고해야 함.

## 4. 실험 설정 (현재 default)

- VQAv2 number subset: `answer_range=8`, `samples_per_answer=400`, `require_single_numeric_gt=True` → 모델당 17,730 sample-instance.
- 질문당 5개 irrelevant set (각 set은 number-image 1 + neutral-image 1, anchor digit ∈ {0..9} 샘플링).
- Greedy decoding, `max_new_tokens=8`.
- JSON-only system prompt (`{"result": <number>}`). Strengthen variant는 "no hedging"을 명시적으로 추가.
- Seed 42.

## 5. Phase A — 기존 데이터 deep re-analysis (no new compute)

`outputs/experiment/`와 `outputs/experiment_anchor_strengthen_prompt/`에 이미 들어있는 evidence를 추출. 각 항목 = 분석 스크립트 1 + 짧은 insight markdown 1 (`docs/insights/`).

| ID | 질문 | Output | Status |
|---|---|---|---|
| **A1** | **Wrong cases에서의 asymmetric anchoring.** `target_only` correctness로 pair stratify; correct vs wrong에 대해 adoption / direction-follow / pull magnitude 별도 계산. 모델별 + pooled 보고. H2 test이자 paper의 가장 강한 hook. | `docs/insights/A1-asymmetric-on-wrong.md` | ✅ 완료 — adoption symmetric, graded `moved_closer_rate` +6.9–19.6 pp |
| **A2** | **Anchor-value별 pull.** 각 anchor digit 0–9에 대해 mean signed shift `pred(number) − pred(target_only)`와 adoption rate 계산. Sticky한 digit이 있는가 (0, round number, 모델 frequent prior)? Pull vs anchor plot. | `docs/insights/A2-per-anchor-digit.md` | ✅ 완료 — anchor 1/2/4 sticky, 7/8 inert; LLaVA × anchor=2 = 0.30; Qwen3-VL은 digit 3/6/7에서 anti-anchoring |
| **A3** | **Question-type stratification.** 기존 `question_type` 컬럼 사용 ("how many", "what number" 등). 질문 형식별 anchoring 차이? 주의: VQAv2 question_type은 거칠다 — taxonomy 먼저 확인. | `00-summary.md`에 합침 (이 granularity에서는 signal 없음) | ✅ 완료 (negative — ChartQA/TallyQA 데이터 후로 보류) |
| **A4** | **Pair별 shift 분포.** mean 너머: `pred(number) − pred(target_only)` 히스토그램. 분포의 mass는 어디 있나? Bimodal (full adoption + no change) 인가 graded인가? H1이 "discrete capture"인지 "graded pull"인지 결정. | `00-summary.md`에 합침 | ✅ 완료 — strongly bimodal (≥75% no change) + thin pull-toward-anchor tail |
| **A5** | **Strengthen vs standard prompt.** Item별 두 prompt 비교. H5 test: strengthen prompt가 uncertain item에서 *anchor-driven* shift를 amplify하나, 아니면 그냥 hallucination을 inflate하나? §3.5 outlier 때문에 mean이 아닌 median + IQR 사용. | `00-summary.md`에 합침 | ✅ 완료 — gemma3-27b만 substantial하게 움직임 (+17.4 pp adoption); universal하지 않음 |
| **A6** | **Failure-mode taxonomy.** 각 `number`-condition prediction을 (a) exact anchor copy, (b) graded pull toward anchor, (c) `target_only`와 unchanged, (d) anchor-orthogonal hallucination, (e) refusal/non-numeric로 bucket. 모델별 보고. | `00-summary.md`에 합침 | ✅ 완료 — `_data/A6_failure_modes.csv` |
| **A7** | **Cross-model item agreement.** 한 모델이 anchor에 끌리는 item을 다른 모델도 anchor하나? Per-item susceptibility score 만들어 모델 간 correlation. Item-driven (high cross-model corr)이면 *content*-mediated; model-driven (low corr)이면 *parameter*-mediated — H3에 직접 정보. | `docs/insights/A7-cross-model-agreement.md` | ✅ 완료 — Spearman ρ ∈ [0.15, 0.31]; 부분적으로 content-driven; Qwen3-VL same-family pair = 0.30 |

**Phase A 산출물:** insight 파일 4개 (`00-summary.md`, `A1-…`, `A2-…`, `A7-…`) + numeric artifact `docs/insights/_data/`. A3/A4/A5/A6는 자체 writeup이 안 나올 정도 signal이라 summary에 흡수. 7개 분석 모두 단일 스크립트에서 돌림: `scripts/phase_a_data_mining.py`.

## 6. Phase B 이후 — 신규 실험

### Tier 1 (Main-tier acceptance에 결정적, `references/project.md` 권고)

| ID | Experiment | 컴퓨트 추정 | 왜 중요 | Status |
|---|---|---|---|---|
| **E1** | **Attention-mass 분석.** Stratified 샘플에 `output_attentions=True`. Anchor-image-token vs target-image-token vs text-token 어텐션 mass를 layer/head/condition별로 계산. | 모델당 시간 단위. 기존 7 모델 위에서 바로. | "왜 일어나는가" 리뷰어 질문에 직접 답. 가장 싼 mechanistic move. | ✅ **4 모델 완료** (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b; 각 n=200, 총 2,400 attention record). 세 claim 결정됨: (i) anchor > neutral attention **4/4 replicate** (answer step에서 mean +0.004 ~ +0.007, CI 0 제외); (ii) H2 `wrong > correct` attention asymmetry **4/4 falsified**; (iii) A7 `susceptible > resistant` **3/4 replicate** — Gemma-SigLIP이 outlier (inverts, step-0-heavy). 전체 writeup: `docs/experiments/E1-preliminary-results.md`. 남은 것: ConvLLaVA + FastVLM (inputs_embeds path 확장), causal test, per-layer localisation. |
| **E2+E3** (combined) | **신규 통합 4 모델 (ConvLLaVA, LLaVA-1.5, InternVL3, FastVLM) full 17,730 grid.** Pilot at 1,125 완료 — `docs/experiments/E2-pilot-results.md` 참조. Full run = (a) H6 two-axis hypothesis CI 좁힘, (b) per-token logit 캡처 → A1 logit-margin 재분석 가능, (c) paper용 11-model panel. | 7B 모델당 H200 1일; 4 모델 sequential. | 스케일에서 H6 two-axis hypothesis 결정. | **⏸ 보류** (사용자 2026-04-24): pilot 데이터만으로 진행; E1 attention + mitigation 우선. E1 결과가 full 4-model panel이 여전히 필요한지 알려주면 재평가. |
| **E4** | **Mitigation prototype.** E1 attention 분석이 시사하는 가장 단순한 intervention 선택 — 가장 가능성: number-image vs no-image contrastive decoding, 또는 anchor-image-token attention re-weighting. 목표: direction-follow ≥ 10 % 감소, accuracy 손실 ≤ 2 pp. | 며칠; E1에 의존. | Findings → Main으로 가는 가장 신뢰할 수 있는 leverage. | ☐ |

### Tier 2 (paper hardening)

| ID | Experiment | 메모 | Status |
|---|---|---|---|
| **E5** | **Multi-dataset full run.** TallyQA + ChartQA full scale (현재 50 샘플 smoke로는 부족). MathVista는 stretch. | TallyQA = 가장 깨끗한 counting 도메인; ChartQA = in-image-number conflict (특히 강력 — anchor가 타겟 이미지의 legible number와 경쟁). | ☐ |
| **E6** | **Closed-model subset.** GPT-4o 또는 Gemini-2.5에 대해 ~500 샘플 stratified slice. "open-only" 리뷰어 컴플레인 차단. | Token cost만. | ☐ |
| **E7** | **Paraphrase robustness.** 3–5개 question prompt 재작성 × bootstrap CI × multiple-comparison correction. | 모델별 effect 주장 전 필수. | ☐ |
| **E8** | **Position effect.** Anchor 이미지가 image[0]일 때 vs image[1]일 때. 일부 VLM은 위치-가중. | E2/E3 sub-experiment로. | ☐ |

### Tier 3 (optional, Tier 1–2 이후에만)

| ID | Experiment | 메모 | Status |
|---|---|---|---|
| E9 | 0–9 너머의 anchor value 범위 (10s, 100s). | Bias가 anchor magnitude에 scaling하는지 saturate하는지. | ☐ |
| E10 | Layer-wise logit lens — anchor가 *언제* prediction에 진입. | E1 보완. | ☐ |
| E11 | Human baseline (~50 Prolific 참가자, 작은 condition matrix). | Psychology-framed paper에 disproportionately credibility-positive. | ☐ |
| E12 | Reasoning 모드를 둘 다 지원하는 모델에서 thinking-VLM 비교 (Qwen3-VL with/without reasoning). | H4 test. | ☐ |

## 7. 순서와 결정 지점

1. **Phase A (완료)** — Phase-A insight 정리됨 (`docs/insights/`). 산출: A1이 H2를 graded 형태로 확인; A2/A7이 mechanism plan에 정보 제공.
2. **Phase B (현재)** — pilot 완료, E2+E3 full run은 사용자 결정으로 2026-04-24 보류. 활성 시퀀스: **E1 (기존 7 모델 attention mass) → E4 (mitigation prototype) → 4-model full run 재평가**. Pilot이 mitigation prototype 시작에 충분한 H6 evidence 제공; E1 attention이 두 failure mode를 mechanistically 분리 못 할 때만 E2+E3 full grid로 fallback.
3. **Phase C** — E1+E4가 publishable result 만들면 Tier-2 hardening (E5 multi-dataset, E7 paraphrase robustness, E6 closed-model subset).
4. **Phase D** — write-up. ARR May 25 deadline (`references/project.md` §"realistic one-month plan").

**결정 trigger** (trigger 발생 시 답을 적어둘 것):
- A1 후: asymmetry 진짜이고 큼 (≥ 10 pp gap)? 아니면 H3 또는 H5로 fallback.
- E1 후: anchor가 disproportionate attention 받음? 그렇다면 E4 straightforward; 아니면 mitigation을 LLM 쪽으로 향해야 함.
- E2 후: ConvLLaVA가 의미 있게 낮은 direction-follow? 그렇다면 H3가 paper section; 아니면 H3 drop, H2 + mechanism으로 통합.

## 8. Research artifact 파일시스템 컨벤션

첫 artifact 작성 시 lazy하게 만들기.

```
research/
  insights/                    # phase-A 재분석 결과 (insight 1개당 md 1개)
    00-summary.md
    A1-asymmetric-on-wrong.md
    ...
  experiments/                 # E1..E12의 plan + result writeup
    E1-attention-mass.md
    E2-encoder-ablation.md
    ...
  scripts/                     # insight를 만드는 일회성 분석 스크립트
                               # 재사용 스크립트는 ../scripts/
```

이 roadmap은 project root에 머무르며 program 전체의 status를 가진 **유일한** doc. Insight와 experiment doc은 각자 하나씩만 다룸.

## 9. Known caveats (모든 분석에 들고 갈 것)

- **Strengthen-prompt distance outlier** (§3.5) — 견고하게 trim하거나 median 사용.
- **Anchor digit ∈ 0–9 vs answer 지원 0–8** — anchor 분포가 GT 분포보다 넓음. "anchor 쪽으로 움직였다"를 계산할 때 anchor=9는 이 subset에서 절대 정답이 될 수 없다는 점을 통제.
- **VQA 깨진 이미지** (`inputs/vqav2_number_val/images/000000000136.jpg`) — file body가 JPEG가 아닌 filesystem garbage. Loader (`vlm_anchor.data.load_number_vqa_samples`)가 이제 `Image.verify()`를 호출해 decode 안 되는 파일을 silently skip하므로 더 이상 run을 죽이지 않음. 해당 image_id의 questions.jsonl 항목은 load 시 drop (1개 sample 적음).
- **`fastvlm-7b` 산문 출력** — JSON-only prompt에도 자주 산문 출력. `extract_first_number`가 대부분 살리지만 parse 실패율 0이 아님; 명시적으로 보고.
- **공유 GPU** — 같은 머신에서 vLLM `Qwen2.5-32B` 서버가 port 8000 (~55 % VRAM) 사용 중. 본 프로젝트의 효과적 per-GPU 예산 ≈ 60 GB.
- **Citation hygiene** — `references/project.md`가 일부 2026 arXiv ID는 resolve 안 될 가능성 flag. 제출 전 cite 검증.

## 10. Changelog

- **2026-04-24** — Roadmap 작성. Status: 7 모델 × full VQAv2 (standard + strengthen prompt) 완료; 신규 5 모델 통합되었으나 main run 없음; 3 데이터셋 확장 smoke만. Phase A queued.
- **2026-04-24** — Phase A 완료. Headline (H2): anchoring은 uncertainty-modulated **graded pull**, categorical capture가 아님 (`docs/insights/A1-asymmetric-on-wrong.md`). Per-digit asymmetry 확인 (A2). Cross-model correlation 0.15–0.31 (A7) → encoder와 content 둘 다 영향 → E1+E2 motivate. A3/A4/A5/A6는 `00-summary.md`에 통합. §7 결정 trigger 발화 — Phase B 순서 변경 없음.
- **2026-04-24** — E2 pilot (n=1,125 × 4 모델) 완료. **H3 단순 "Conv < ViT" 형태 지지 안 됨** — ConvLLaVA adoption=0.156이 CLIP/SigLIP 클러스터 CI 안. **새 H6 추가**: cross-modal failure가 두 직교 축 (anchor-pull vs multi-image distraction)으로 분해. InternVL3 = pure distraction (low adoption, high acc_drop), LLaVA-1.5 = pure anchoring (high adoption, low acc_drop), ConvLLaVA = both. Two-axis framing이 "encoder family universally matters"를 paper headline candidate로 대체. `docs/experiments/E2-pilot-results.md` 참조. 4 모델 full 17,730 run은 사용자 signoff 대기.
- **2026-04-24** — Bug fix: `vlm_anchor.data.load_number_vqa_samples`가 이제 `Image.verify()`를 호출해 decode 안 되는 이미지 silently skip. `000000000136.jpg` PIL crash가 미래의 multi-day run을 죽이지 못하게 함.
- **2026-04-24** — 사용자 결정 (option 2): E2+E3 full 4-model run 보류; E1 attention 추출 + E4 mitigation 우선. Pilot 데이터 + Phase A로 mitigation prototype 시작 충분. E1이 anchor-pull과 multi-image distraction을 mechanistically 분리 못 할 때만 E2+E3 재개. §7 Phase B 시퀀스 갱신.
- **2026-04-24** — Bilingual docs convention 채택. references/roadmap.md와 research/ 하위 모든 md가 `_ko.md` 한국어 미러 가짐. 영문 `.md`이 canonical (Claude가 먼저 읽고 편집); 한국어 버전은 lockstep으로 갱신. 메모리: `feedback_bilingual_docs.md`.
- **2026-04-24** — E1을 4 encoder family로 확장 (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b; 각 n=200). **4-모델 스케일에서 세 claim 결정됨:** (i) anchor>neutral attention이 4/4 robust (answer-step mean +0.004 ~ +0.007, CI 0 제외); (ii) H2 `wrong>correct` attention asymmetry가 4/4 falsified — uncertainty가 mean anchor attention modulate 안 함; (iii) A7 `susceptible>resistant`가 answer step에서 3/4 holds, Gemma-SigLIP에서 inverts (signal도 step 0에 집중, typographic-attack 상속과 일관). Paper candidate 3-claim structure 등장: anchor notice (attention)은 robust; anchor pull (behaviour)은 encoder-modulated; uncertainty가 pull을 modulate (Phase A)하지만 attention은 아님. `docs/experiments/E1-preliminary-results.md`.
