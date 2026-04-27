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
| **H3** | Vision-encoder family가 susceptibility를 modulate한다. ConvNeXt/encoder-free는 CLIP/SigLIP-ViT보다 *덜* 취약 (typographic-attack 상속). | ConvLLaVA / EVE / DINO-VLM의 direction-follow gap이 CLIP-ViT와 통계적 동등이면 fail. | ❌ **Adoption과 per-layer 두 수준 모두에서 Falsified.** Pilot (2026-04-24): ConvLLaVA adoption=0.156이 CLIP/SigLIP 클러스터 CI 안. 6-모델 E1b (2026-04-24): ConvLLaVA의 per-layer attention 지문이 LLaVA-1.5와 정확히 매칭 — 같은 피크 layer L16, 같은 text-stealing 메커니즘, magnitude 19 % 이내, A7 gap 30 % 이내. E1b의 depth-axis framing + H6 (two-axis decomposition)로 대체. `docs/insights/E1c-h3-falsified.md` 참조. |
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
| gemma4-e4b | 0.553 | 0.505 | 0.541 | 0.059 | 0.320 | 4.17 |
| llava-interleave-7b | 0.619 | 0.577 | 0.576 | 0.047 | 0.348 | 3.20 |
| gemma3-27b-it | 0.628 | 0.623 | 0.633 | 0.047 | 0.305 | 4.45 |
| qwen2.5-vl-7b | 0.736 | 0.708 | 0.711 | 0.019 | 0.248 | 5.03 |
| qwen3-vl-8b | 0.751 | 0.709 | 0.715 | 0.029 | 0.258 | 3.43 |
| gemma4-31b-it | 0.749 | 0.723 | 0.741 | 0.021 | 0.239 | 6.16 |
| qwen3-vl-30b-it | 0.759 | 0.709 | 0.707 | 0.034 | 0.280 | 3.70 |

추가 분석 없이 즉시 보이는 두 패턴:
1. **Direction-follow ≈ 24–35 %** 인데 paired anchor adoption은 **2–6 %** 에 그침. Bias가 *gradient*이지 categorical이 아님 — 대부분 케이스가 anchor "쪽으로 끌려"가지 anchor "로 setting"되지 않음. Paired 정의 (M1) 가 marginal 정의 (GT==anchor confound로 adoption 을 11–14 %로 부풀렸던) 대비 이 gap을 더욱 벌려 gradient-pull reading을 강화.
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

### 대기 중인 리팩터 (Tier 2 진입 전 필수)

| ID | Task | 왜 중요 | Status |
|---|---|---|---|
| **M1** | **Paired adoption metric 리팩터.** 현재 marginal `anchor_adopted = (pred == anchor_value)` (`src/vlm_anchor/metrics.py:40`)을 paired `(base_pred ≠ anchor_value) AND (pred == anchor_value)`로 교체. GT==anchor confound 제거: 현 정의는 `GT == anchor == pred` 케이스도 adoption으로 카운트하고, E5b stratum 1 `(0,1)`은 `\|a − gt\| = 0` 자체를 허용하며, main-run anchor inventory 0–9 vs GT support 0–8은 9/10 overlap. 구현: `evaluate_sample`에 `base_prediction` 인자 추가; runner는 sample-instance마다 `target_only`를 먼저 돌리고 그 prediction을 number/neutral 평가에 전달; `summarize_condition` mean shape 보존 (binary label 그대로 → rate처럼 읽힘). 완료된 run은 raw `predictions.jsonl`에서 재집계 (재추론 X). Marginal column 폐기 (raw `prediction` 보존 → downstream 재유도 가능). | §3.4 / Phase A / E1 / E4 mitigation 테이블의 모든 공개 수치가 의존. §6 Tier 2 hardening(E5/E7)이 공개 수치를 다시 만지기 전에 반드시 착륙. **세션 간 조율:** `metrics.py` 또는 `models.py` runner 순서를 동시 수정하는 작업은 M1과 정렬할 것. | ✅ landed (commits bbcc418..ce1928a; 54 dir 재집계 완료) |

### Tier 1 (Main-tier acceptance에 결정적, `references/project.md` 권고)

| ID | Experiment | 컴퓨트 추정 | 왜 중요 | Status |
|---|---|---|---|---|
| **E1** | **Attention-mass 분석.** Stratified 샘플에 `output_attentions=True`. Anchor-image-token vs target-image-token vs text-token 어텐션 mass를 layer/head/condition별로 계산. | 모델당 시간 단위. 기존 7 모델 위에서 바로. | "왜 일어나는가" 리뷰어 질문에 직접 답. 가장 싼 mechanistic move. | ✅ **6-모델 패널 + per-layer localisation + 인과 ablation 모두 완료** (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b, convllava-7b, fastvlm-7b; 각 n=200). E1 세 claim 결정됨; E1b 네 원형 식별: SigLIP-Gemma early+large (L5, δ +0.050, text-stealing); CLIP-ViT+InternViT+ConvNeXt의 mid-stack cluster (L14–16, δ ~+0.020, text-stealing); Qwen-ViT late+moderate (L22, δ +0.015, target-stealing); FastViT late+large+최강-A7 (L22, δ +0.047, text-stealing, A7 gap +0.086 with n=75 caveat). **H3 "ConvNeXt < ViT" 확정적으로 falsified** — ConvLLaVA가 LLaVA-1.5를 정확히 replicate. **E1d 인과 검증 (2026-04-25): 단일-layer ablation은 6/6 모두 null — E1b peak에서도, layer 0에서도; multi-layer redundancy 확인.** Upper-half ablation이 6/6에서 direction-follow를 줄이는 (−5.5 ~ −11.5 pp), 4/6에서 fluency 청결한 단일 architecture-blind locus. Writeup: `docs/experiments/E1-preliminary-results.md` (E1 4-모델 원본), `docs/experiments/E1b-per-layer-localisation.md` (E1b 전체 6-모델 writeup) + `docs/insights/E1b-per-layer-localisation.md` (요약), `docs/experiments/E1d-causal-ablation.md` (E1d detailed) + `docs/insights/E1d-causal-evidence.md` (요약). 남은 것: head-level sparsity; multi-layer combinatorial ablation. |
| **E2+E3** (combined) | **신규 통합 4 모델 (ConvLLaVA, LLaVA-1.5, InternVL3, FastVLM) full 17,730 grid.** Pilot at 1,125 완료 — `docs/experiments/E2-pilot-results.md` 참조. Full run = (a) H6 two-axis hypothesis CI 좁힘, (b) per-token logit 캡처 → A1 logit-margin 재분석 가능, (c) paper용 11-model panel. | 7B 모델당 H200 1일; 4 모델 sequential. | 스케일에서 H6 two-axis hypothesis 결정. | **⏸ 보류** (사용자 2026-04-24): pilot 데이터만으로 진행; E1 attention + mitigation 우선. E1 결과가 full 4-model panel이 여전히 필요한지 알려주면 재평가. |
| **E4** | **Mitigation prototype.** E1 attention 분석이 시사하는 가장 단순한 intervention 선택. **E1d (2026-04-25)가 E1b peak/layer 0에서의 단일-layer attention re-weighting을 배제**; multi-layer ablation 필요. 현재 후보 prototype: **mid-stack cluster (LLaVA-1.5 / ConvLLaVA / InternVL3)에 대한 upper-half attention re-weighting** — E1d에서 이 셋에 대해 −5.5 ~ −9.8 pp direction-follow, fluency hit 없음. 목표: direction-follow ≥ 10 % 감소, accuracy 손실 ≤ 2 pp. | 며칠; E1d에 의존. | Findings → Main으로 가는 가장 신뢰할 수 있는 leverage. | ⏳ **Phase 1 sweep (n=200, strength 7개 × 조건 3개) 진행 중.** llava-1.5-7b 완료 — `s* = −3.0`이 타깃 만족 (df 0.305 → 0.265, −13 % 상대; em 0.365 → 0.370, +0.5 pp); em(target_only)이 0.435로 불변하여 hook이 anchor-condition-specific 임을 확인. convllava-7b sweep GPU 0에서 진행 중 (2026-04-25 15:20 시점 ~50 % 완료); internvl3-8b 대기. Phase 2 풀 스케일 (n=17,730) 검증을 chosen `s*`에서 수행, 12-h 세션 예산을 고려해 llava-1.5-7b 우선. Writeup: `docs/experiments/E4-mitigation.md` (+ _ko mirror, 진행 중), `docs/insights/E4-mitigation-evidence.md` (+ _ko mirror, 진행 중). |

### Tier 2 (paper hardening)

| ID | Experiment | 메모 | Status |
|---|---|---|---|
| **E5** | **Multi-dataset full run.** TallyQA + ChartQA full scale (현재 50 샘플 smoke로는 부족). MathVista는 stretch. | TallyQA = 가장 깨끗한 counting 도메인; ChartQA = in-image-number conflict (특히 강력 — anchor가 타겟 이미지의 legible number와 경쟁). | ☐ |
| **E5b** | **Anchor-distance robustness sweep.** TallyQA + VQAv2에서 per-question 5-stratum anchor sampling (dataset당 500 base question, llava-interleave-7b). Anchor selection 규칙이 paper 헤드라인 figure에 load-bearing인지 검증. | ~50 min wall (1 모델). 산출: 곡선 figure 1개 (effect-vs-distance) + cross-dataset overlay. 헤드라인 figure가 near-anchor subset에서 reporting할지 결정. | ☐ |
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
2. **Phase B (현재)** — E1 attention + E1b per-layer + E1d 인과 ablation 모두 6-모델 패널에서 완료. 활성 시퀀스: **E4 (E1d에 따라 mid-stack cluster에 대한 upper-half multi-layer mitigation prototype)**. Pilot 데이터 + Phase A + E1d로 mitigation prototype 시작 충분; E2+E3 full 4-모델 grid는 E4가 mid-stack cluster에서 다른 원형으로 generalise 못 할 때에만 재개.
3. **Phase C** — E4가 publishable mitigation result 만들면 Tier-2 hardening (E5 multi-dataset, E7 paraphrase robustness, E6 closed-model subset).
4. **Phase D** — write-up. ARR May 25 deadline (`references/project.md` §"realistic one-month plan").

**결정 trigger** (trigger 발생 시 답을 적어둘 것):
- A1 후: asymmetry 진짜이고 큼 (≥ 10 pp gap)? 아니면 H3 또는 H5로 fallback. **Green 발화** — Phase A에서 graded-pull asymmetry +6.9 ~ +19.6 pp가 7/7 모델에서 확인.
- E1 후: anchor가 disproportionate attention 받음? 그렇다면 E4 straightforward; 아니면 mitigation을 LLM 쪽으로 향해야 함. **Green 발화** — anchor>neutral attention이 answer step에서 6/6 robust. *E1d caveat:* per-layer attention 집중은 상관관계일 뿐 인과적 single-layer site가 아님 — E4는 multi-layer intervention 사용 필요.
- E1d 후: per-family peak layer가 single-layer 인과 site? 그렇다면 family별 targeted E4 prototype. 아니면 multi-layer 또는 다른 intervention class. **No 발화** — single-layer ablation이 peak *과* layer 0에서 6/6 null; 현재 E4 후보는 mid-stack cluster에 대한 upper-half multi-layer ablation.
- E2 후: ConvLLaVA가 의미 있게 낮은 direction-follow? 그렇다면 H3가 paper section; 아니면 H3 drop, H2 + mechanism으로 통합. **No 발화** — H3 retired (E1c).

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
- **`anchor_adopted`는 이제 paired (M1 landed 2026-04-27)** — `(base_pred ≠ anchor) AND (pred == anchor)`. M1 이전 marginal 정의 (`pred == anchor_value`만 보고 base prediction 무관)는 `GT == anchor == pred` 케이스를 adoption 으로 카운트하던 문제 — 특히 E5b stratum 1 `(0,1)`, main-run anchor 0–9 vs GT 0–8 9/10 overlap에서 가시 — 때문에 폐기. 기존 54개 predictions.jsonl 모두 `scripts/reaggregate_paired_adoption.py`로 재집계; 원본 marginal-era artefact는 `*.marginal.bak.*`로 보존.

## 10. Changelog

- **2026-04-24** — Roadmap 작성. Status: 7 모델 × full VQAv2 (standard + strengthen prompt) 완료; 신규 5 모델 통합되었으나 main run 없음; 3 데이터셋 확장 smoke만. Phase A queued.
- **2026-04-24** — Phase A 완료. Headline (H2): anchoring은 uncertainty-modulated **graded pull**, categorical capture가 아님 (`docs/insights/A1-asymmetric-on-wrong.md`). Per-digit asymmetry 확인 (A2). Cross-model correlation 0.15–0.31 (A7) → encoder와 content 둘 다 영향 → E1+E2 motivate. A3/A4/A5/A6는 `00-summary.md`에 통합. §7 결정 trigger 발화 — Phase B 순서 변경 없음.
- **2026-04-24** — E2 pilot (n=1,125 × 4 모델) 완료. **H3 단순 "Conv < ViT" 형태 지지 안 됨** — ConvLLaVA adoption=0.156이 CLIP/SigLIP 클러스터 CI 안. **새 H6 추가**: cross-modal failure가 두 직교 축 (anchor-pull vs multi-image distraction)으로 분해. InternVL3 = pure distraction (low adoption, high acc_drop), LLaVA-1.5 = pure anchoring (high adoption, low acc_drop), ConvLLaVA = both. Two-axis framing이 "encoder family universally matters"를 paper headline candidate로 대체. `docs/experiments/E2-pilot-results.md` 참조. 4 모델 full 17,730 run은 사용자 signoff 대기.
- **2026-04-24** — Bug fix: `vlm_anchor.data.load_number_vqa_samples`가 이제 `Image.verify()`를 호출해 decode 안 되는 이미지 silently skip. `000000000136.jpg` PIL crash가 미래의 multi-day run을 죽이지 못하게 함.
- **2026-04-24** — 사용자 결정 (option 2): E2+E3 full 4-model run 보류; E1 attention 추출 + E4 mitigation 우선. Pilot 데이터 + Phase A로 mitigation prototype 시작 충분. E1이 anchor-pull과 multi-image distraction을 mechanistically 분리 못 할 때만 E2+E3 재개. §7 Phase B 시퀀스 갱신.
- **2026-04-24** — Bilingual docs convention 채택. references/roadmap.md와 research/ 하위 모든 md가 `_ko.md` 한국어 미러 가짐. 영문 `.md`이 canonical (Claude가 먼저 읽고 편집); 한국어 버전은 lockstep으로 갱신. 메모리: `feedback_bilingual_docs.md`.
- **2026-04-24** — E1을 4 encoder family로 확장 (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b; 각 n=200). **4-모델 스케일에서 세 claim 결정됨:** (i) anchor>neutral attention이 4/4 robust (answer-step mean +0.004 ~ +0.007, CI 0 제외); (ii) H2 `wrong>correct` attention asymmetry가 4/4 falsified — uncertainty가 mean anchor attention modulate 안 함; (iii) A7 `susceptible>resistant`가 answer step에서 3/4 holds, Gemma-SigLIP에서 inverts (signal도 step 0에 집중, typographic-attack 상속과 일관). Paper candidate 3-claim structure 등장: anchor notice (attention)은 robust; anchor pull (behaviour)은 encoder-modulated; uncertainty가 pull을 modulate (Phase A)하지만 attention은 아님. `docs/experiments/E1-preliminary-results.md`.
- **2026-04-24** — **E1b per-layer localisation 완료** (동일 4 모델 × n=200). 피크 layer가 encoder family별로 선명히 다름: SigLIP-Gemma **layer 5/42** (12 % depth, δ +0.050, anchor/target trade-off layer로 둘러싸인 spike), Qwen-ViT **layer 22/28** (82 %, δ +0.015, A7 gap +0.025이고 bottom-decile CI가 0 포함), CLIP-ViT (LLaVA-1.5) **layer 16/32**, InternViT (InternVL3) **layer 14/28** (둘 다 mid, δ ~+0.019). Layer-평균 E1 수치는 단일 layer에 ~3× 집중되어 있던 사실을 감추고 있었음. **두 번째 축 — budget decomposition:** 피크에서 Gemma/LLaVA-1.5/InternVL3는 anchor mass를 *text*에서 가져오고 (δ_text −0.014 ~ −0.038), Qwen은 *target 이미지*에서 가져옴 (−0.010, text −0.005). 두 개의 구별되는 메커니즘: text-stealing vs target-stealing. **Family별 E4 개입 후보 site (검증 전):** Gemma → input-side pre-layer-5 KV/projection patch (text→anchor pull 차단); Qwen → susceptibility gate한 layer 22±2 late-stack anchor attention re-weighting (mass를 target으로 되돌림); CLIP/Intern → mid-stack ~14–16 (mass를 text로 되돌림 — 덜 이상적이지만 여전히 시도 가능). 이들은 관찰 기반 추측; E4가 어느 것이 실제로 `direction_follow`를 줄이는지 검증할 예정. `docs/experiments/E1b-per-layer-localisation.md` (detailed) + `docs/insights/E1b-per-layer-localisation.md` (요약).
- **2026-04-24** — **E1 inputs_embeds-path 확장 완료; 6-모델 패널 완성.** `scripts/extract_attention_mass.py`에 `EagerConvLLaVARunner` / `EagerFastVLMRunner` subclass 추가로 ConvLLaVA (ConvNeXt encoder, inputs_embeds generate path)와 FastVLM (FastViT, -200-marker expansion path)을 attention 추출 파이프라인에 통합. 두 모델 모두 full n=200 run 완료. **두 가지 주요 신규 발견:** (i) **H3 "ConvNeXt < ViT"이 per-layer 수준에서 확정적으로 falsified** — ConvLLaVA의 피크 layer가 L16 (LLaVA-1.5와 동일), 메커니즘이 text-stealing (동일), magnitude +0.022 (LLaVA-1.5의 20 % 이내). 세 encoder (CLIP-ViT, InternViT, ConvNeXt)가 이제 tight한 "mid-stack text-stealing" cluster 형성. (ii) **FastVLM은 새 원형:** late 피크 (L22, Qwen depth와 일치) + text-stealing budget (−0.034, Gemma 유형과 일치) + Gemma 수준 magnitude (+0.047) + 패널 최대 A7 gap (+0.086, n=75 및 wide CI caveat 전제). 두 publish된 VLM failure mode — typographic attack과 anchor-vs-target budget 혼란 — 이 FastVLM에서 co-firing하는 것으로 보임. 4-모델 E1b의 3-원형 이야기가 4-원형으로 refine됨. E4 design이 이제 family별 개입 site로 진행 가능; mid-stack cluster가 최고-leverage target (하나의 intervention으로 세 encoder generalise). 업데이트된 6-모델 패널은 `docs/experiments/E1b-per-layer-localisation.md` 참조.
- **2026-04-24** — **H3 공식 retire; depth-axis framing으로 대체.** 요약 insight 작성: `docs/insights/E1c-h3-falsified.md` (+ _ko mirror). H3의 "ConvNeXt < ViT" 가설이 행동 (E2 pilot adoption)과 메커니즘 (E1b per-layer) 두 수준 모두에서 fail. Architecturally 다른 세 encoder (CLIP-ViT / InternViT / ConvNeXt)가 같은 mid-stack text-stealing 프로파일로 수렴. 논문 narrative가 "encoder architecture가 anchoring을 modulate"에서 "post-projection LLM stack depth가 축"으로 이동. 결과: 원래 계획된 E2 "encoder-ablation" 서브섹션 불필요; compute를 E5 (multi-dataset) 또는 E7 (paraphrase robustness)로 재할당 가능. §2 H3 status ⚠️ → ❌.
- **2026-04-25** — **E1d 인과 anchor-attention ablation 6-모델 패널 완료.** Driver `scripts/causal_anchor_ablation.py`, 분석 `scripts/analyze_causal_ablation.py`, 모델당 n=200 stratified, 7가지 ablation mode (`baseline`, `ablate_layer0`, `ablate_peak`, `ablate_peak_window`, `ablate_lower_half`, `ablate_upper_half`, `ablate_all`). **세 가지 발견.** (i) **단일-layer ablation은 6/6 모델 모두 null — E1b peak에서도, layer 0에서도** (peak에서 `Δ direction_follow ∈ [−0.032, +0.020]`, layer 0에서 `[−0.027, +0.005]`; 모든 CI가 baseline과 겹침). Layer-0 control이 (b) "peak이 상관관계이고 다른 단일 layer가 인과 site" 해석을 배제 — Gemma (E1b가 L0–4 anchor↔target swap을 보고했던 모델)에서도 (Gemma layer-0 Δ = +0.005). Multi-layer redundancy 확인: anchor의 효과가 LLM stack 전체에 redundant하게 인코딩되어 있어 어떤 단일-layer attention-mask ablation도 답을 변경하지 않음. (ii) **Stack 전체 ablation은 보편적으로 `direction_follow`을 11–22 pp 감소시키지만 3/6에서 fluency 깨뜨림** (Gemma/LLaVA-1.5/ConvLLaVA에서 mean-distance 4–6× 폭발, FastVLM에서 ~3 자릿수). 11–22 pp drop은 인과 anchor pathway의 *upper bound*이지 target이 아님. (iii) **Upper-half attention ablation이 단일 architecture-blind mitigation locus** — 6/6에서 direction-follow 감소 (−5.5 ~ −11.5 pp), 4/6에서 fluency 청결 (mid-stack cluster + Qwen). Mid-stack cluster가 가장 leverage 높은 E4 prototype target — 세 encoder, 한 공유 upper-half-clean 반응. **Caveat sub-finding:** ConvLLaVA와 LLaVA-1.5가 같은 E1b peak/메커니즘 공유에도 불구하고 lower-half ablation에 *반대로* 반응 (ConvLLaVA Δ = −0.120, LLaVA-1.5 Δ = +0.165) — same-attention-signature가 same-causal-structure를 함의하지 않음. **Roadmap 효과:** §6 E1 행 "causal test" open question close; §6 E4 행은 mid-stack cluster의 upper-half multi-layer prototype 명시, single-layer 배제로 갱신. 새 open question: head-level sparsity; multi-layer combinatorial ablation. Writeup: `docs/experiments/E1d-causal-ablation.md` (+ _ko mirror), `docs/insights/E1d-causal-evidence.md` (+ _ko mirror).
- **2026-04-25** — **E4 Phase 1 strength-sweep 시작; llava-1.5-7b 완료.** Driver `scripts/e4_attention_reweighting.py`, 분석 `scripts/analyze_e4_mitigation.py`, n=200 stratified, strength 7개 × 조건 3개 = 모델당 4,200 records. **llava-1.5-7b:** 베이스라인 df_num=0.305 → s=−3.0 df_num=0.265 (−13 % 상대; ≥ 10 % 타깃 만족); em_num 0.365 → 0.370 at s=−3.0 (≤ 2 pp 예산 안), 포화에서 (s=−10⁴) 0.395로 상승; em(target_only)이 모든 strength에서 0.435로 불변 — hook이 anchor-condition-specific (single-image 추론에 leakage 없음). **convllava-7b** sweep GPU 0에서 진행 중 (15:25 시점 ≈ 60 % 완료), partial 베이스라인 df_num=0.126 (llava보다 낮음 — 이 stratified 세트에서 convllava가 더 anchor-resistant), em_num=0.563 (더 높음); convllava의 효과 크기는 절대값으로 더 작을 것. **internvl3-8b** 대기. 3 sweep 완료 후 Phase 2 풀 스케일 (n=17,730 × 조건 3 × mode 2 ≈ target_only-skip 최적화 후 88 k generations) 우선순위 (advisor 기준 llava-1.5-7b 우선 — E1d 시그널 가장 깨끗, caveat 없음). Phase 2 design은 resumable (append-only JSONL + completed-key skip)이라 12-h 세션 경계를 넘어 계속 진행 가능. Writeup: `docs/experiments/E4-mitigation.md` (+ `_ko.md`), `docs/insights/E4-mitigation-evidence.md` (+ `_ko.md`); 둘 다 풀 검증 land 전까지 "Phase 1 in progress"로 flag.
- **2026-04-27** — **E5b 설계 + plan commit; pipeline 구현 및 smoke validated.** Anchor-distance robustness sweep을 E5의 새 sub-experiment으로 추가. Stratified anchor sampling (5 strata by `|a − GT|`: [0,1] / [2,5] / [6,30] / [31,300] / [301,∞)), TallyQA + VQAv2 dataset당 500 base questions, llava-interleave-7b 단독. 새 driver path는 YAML의 `inputs.anchor_sampling: stratified`로 진입; legacy 3-condition path 그대로 유지 (5-sample smoke로 regression test). 3개 smoke run (VQAv2 stratified / TallyQA stratified / legacy) 모두 통과: stratum별 평균 distance가 S1<S2<S3<S4<<S5로 monotonic 증가, condition counter 정확, `anchor_stratum_id` field 존재 (legacy row에선 None). Specs: `docs/experiments/E5b-anchor-distance-design.md` (+ _ko mirror), plan: `docs/experiments/E5b-anchor-distance-plan.md` (+ _ko mirror). Full run (T9) 과 reproducible notebook (T10) 예정.
- **2026-04-27** — **M1 추가: §6 대기 중인 리팩터 — paired adoption metric.** `anchor_adopted = (pred == anchor_value)` (`src/vlm_anchor/metrics.py:40`)이 base 조건 prediction을 무시하여 `GT == anchor == pred` 케이스도 adoption으로 카운트 — anchor inventory가 GT support와 overlap하는 모든 곳에서 비율을 silently 부풀림 (E5b stratum 1 `(0,1)`이 `\|a − gt\| = 0` 자체를 허용; main run anchor 0–9 vs GT 0–8 = 9/10 overlap). Paired `(base_pred ≠ anchor_value) AND (pred == anchor_value)`로 교체: `evaluate_sample`에 `base_prediction` 인자 추가, runner는 sample-instance마다 `target_only` 먼저 돌리고 그 pred를 number/neutral 평가에 전달, `summarize_condition` mean shape 보존 (binary label → rate처럼 읽힘). 완료된 run은 raw `predictions.{jsonl,csv}`에서 모델별 재집계 (재추론 X); raw `prediction` 컬럼 보존이라 downstream 재유도 가능. Marginal 정의 폐기, paired로 단일화. M1은 §6 Tier 2 hardening(E5/E7)이 §3.4 / Phase A / E1 / E4 공개 수치를 다시 만지기 전에 반드시 착륙. **세션 간 노트:** `metrics.py` 또는 `models.py` runner 순서를 동시 수정하는 작업은 이 리팩터와 정렬할 것 — §9 caveat 참조.
- **2026-04-27** — **M1 landed: paired anchor-adoption metric.** `evaluate_sample` 이 이제 `base_prediction` 을 require하고 `anchor_adopted = (base_pred ≠ anchor) AND (pred == anchor)` 계산 (commit `bbcc418`). Driver는 sample-instance 마다 target_only 의 parsed prediction을 다음 conditions에 thread (`9c07f2e`). 일회성 `scripts/reaggregate_paired_adoption.py` (`220dc4b`, ablation/e4 schema 위해 `ce1928a`로 확장) 가 기존 54개 predictions.jsonl 모두에서 adoption 재계산 (35 standard + 13 causal_ablation + 6 e4_mitigation) — re-inference 불필요; raw prediction 보존. §3.4 헤드라인 paired adoption rate: 0.019–0.059 (이전 marginal 0.110–0.141 대비 ~75–90 % 상대 감소). direction-follow 와 accuracy 는 변동 없음. Stale smoke-only output dir 삭제 (`experiment_tallyqa`, `experiment_mathvista`, `experiment_smoke_check`, ChartQA 5-sample smoke). §6 Pending refactors의 M1 status → ✅. 이제 Tier 2 hardening (E5/E5b/E7) 가 metric drift 없이 진행 가능.
