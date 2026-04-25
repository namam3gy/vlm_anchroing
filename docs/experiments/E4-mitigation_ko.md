# E4 — 어텐션 재가중 완화: 결과

**상태:** Phase 1 strength sweep 진행 중 (llava-1.5-7b 완료; convllava-7b·internvl3-8b 진행 중).
Phase 2 풀 검증: Phase 1 완료 후 시작. 사용자 12-h 세션 예산을 고려해 llava-1.5-7b를 먼저
돌리고, 나머지 모델은 다음 세션에서 resumability 프로토콜로 이어짐.

**소스 데이터:** `outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl` (Phase 1),
`outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl` (Phase 2).
**드라이버:** `scripts/e4_attention_reweighting.py`. **분석:** `scripts/analyze_e4_mitigation.py`.
**디자인 문서:** `docs/experiments/E4-mitigation-design.md` (Phase 2 종료 시 본 문서로 대체).

## 목표 & 타깃

`references/roadmap.md` §6 Tier 1 기준: VQAv2 number subset에서 mid-stack-cluster VLM 3종
(LLaVA-1.5, ConvLLaVA, InternVL3) 대상으로 `direction_follow_rate` ≥ 10 % 감소 동시에
표준 VQA 정확도 ≤ 2 pp 하락. 이 클러스터는 E1b/E1d에서 같은 어텐션 시그니처를 공유하면서도
upper-half ablation 응답이 가장 깨끗하게 나타난 가족이다.

## 방법론

E1d에서 단일 레이어 개입은 모두 null로 판명났고, 패널 전체에서 fluency를 깨지 않으면서
`direction_follow_rate`를 떨어뜨린 유일한 모드는 *upper-half* multi-layer ablation
(strength = −10⁴, hard mask)이었다. E4는 그 위치를 그대로 두고 strength 축을 추가한다:
upper-half 레이어에서 anchor 어텐션을 0으로 만드는 대신, `exp(strength)` 배수로
다운가중. forward pre-hook이 `[n_layers/2, n_layers)`의 각 LLM decoder 레이어에서
`attention_mask`의 anchor span 컬럼에 `strength`를 더해주며, softmax 이후 anchor 어텐션
가중치가 `exp(strength)` 배가 된 뒤 KV mix가 진행된다.

**Strength sweep (Phase 1):** strength 7개 × 조건 3개 × 200 stratified samples per model.
Strength 격자는 `[0, −0.5, −1, −2, −3, −5, −10⁴]` — multiplier(`exp(strength)`)가 의미 있는
구간(≈ 1.0 → 0.0067 → 0)을 로그 스페이싱.

**샘플 stratification:** E1b/E1d와 동일한 200-sample 세트
(상위-decile-susceptible × 100 + 하위-decile-resistant × 100, 출처
`docs/insights/_data/susceptibility_strata.csv`).

**(model, strength)별 메트릭:**
- `direction_follow_rate(target_plus_irrelevant_number)` — 주 메트릭; 예측이
  `target_only` 베이스라인보다 anchor에 더 가까이 이동했는지.
- `adoption_rate(target_plus_irrelevant_number)` — anchor 자릿수와 정확히 일치.
- `exact_match(target_plus_irrelevant_number)` — ground truth와 일치 (표준 VQA 타깃 프록시).
- `mean_distance_to_anchor(target_plus_irrelevant_number)` — fluency 모니터 (큰 값 = 개입이
  단순한 anchor 가중치 다운이 아니라 생성을 깨고 있음을 시사).
- `target_only` 및 `target_plus_irrelevant_neutral`에 대한 `exact_match` — 정합 통제;
  hook은 `target_only`에서 no-op (anchor span 비어 있음), neutral에서는 거의 동일
  (anchor 위치에 가독 자릿수 없음).

부트스트랩 95 % CI (n_boot = 2,000) per cell.

**Strength 선택 규칙:** 다음 두 조건을 동시에 만족하는 가장 작은 `|s|`:
`direction_follow_rate(target_plus_irrelevant_number, s) ≤ 0.9 ×
direction_follow_rate(target_plus_irrelevant_number, 0)` 그리고
`exact_match(target_plus_irrelevant_number, s) ≥ exact_match(target_plus_irrelevant_number, 0)
− 0.02`. 만족하는 `s`가 없으면 더 조밀한 격자 또는 `ablate_upper_quarter`로 escalate.

## Phase 1 — strength sweep

### llava-1.5-7b (베이스라인 df=0.305)

| strength | n | df_num | adopt_num | em_num | em_target_only | em_neutral | mean_dist |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.305 [0.24, 0.37] | 0.175 | 0.365 [0.30, 0.44] | 0.435 | 0.360 | 3.58 |
| −0.5 | 200 | 0.290 [0.23, 0.36] | 0.170 | 0.365 [0.30, 0.44] | 0.435 | 0.355 | 3.63 |
| −1.0 | 200 | 0.290 [0.23, 0.35] | 0.155 | 0.365 [0.30, 0.44] | 0.435 | 0.365 | 3.63 |
| −2.0 | 200 | 0.280 [0.22, 0.34] | 0.120 | 0.375 [0.31, 0.45] | 0.435 | 0.360 | 3.72 |
| **−3.0** | 200 | **0.265 [0.21, 0.33]** | **0.125** | **0.370 [0.30, 0.44]** | 0.435 | 0.360 | 3.77 |
| −5.0 | 200 | 0.250 [0.19, 0.31] | 0.095 | 0.395 [0.33, 0.47] | 0.435 | 0.365 | 4.25 |
| −10⁴ | 200 | 0.250 [0.19, 0.31] | 0.085 | 0.395 [0.33, 0.47] | 0.435 | 0.360 | 4.72 |

**선택:** `s* = −3.0` (두 타깃을 만족하는 최소 |s| — df 상대 13 % 감소, em 델타 +0.5 pp,
예산 내).

**주목할 점:** `em_num`이 어떤 strength에서도 떨어지지 않고 0.365 (베이스라인) → 0.395
(포화)으로 *향상*된다. 완화책이 정확도에 중립이 아니라 *상승*시킨다는 뜻.
`em_target_only`은 0.435로 불변 (정합 통제 통과 — hook이 single-image 입력에서 no-op).
`em_neutral`도 0.355–0.365로 거의 불변 — neutral 이미지에도 hook이 firing 하지만
가독 자릿수가 없으므로 예측이 변하지 않음.

### convllava-7b (베이스라인 df=0.290)

| strength | n | df_num | adopt_num | em_num | em_target_only | em_neutral | mean_dist |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 0.290 [0.23, 0.36] | 0.175 | 0.375 [0.31, 0.45] | 0.500 | 0.380 | 3.18 |
| −0.5 | 200 | 0.285 [0.23, 0.35] | 0.180 | 0.370 [0.31, 0.44] | 0.500 | 0.375 | 3.19 |
| −1.0 | 200 | 0.275 [0.22, 0.34] | 0.165 | 0.370 [0.30, 0.44] | 0.500 | 0.380 | 3.25 |
| **−2.0** | 200 | **0.260 [0.20, 0.32]** | **0.160** | **0.375 [0.31, 0.45]** | 0.500 | 0.375 | 3.30 |
| −3.0 | 200 | 0.255 [0.20, 0.32] | 0.155 | 0.375 [0.31, 0.44] | 0.500 | 0.390 | 3.32 |
| −5.0 | 200 | 0.240 [0.19, 0.30] | 0.125 | 0.400 [0.33, 0.47] | 0.500 | 0.390 | 3.44 |
| −10⁴ | 200 | 0.235 [0.18, 0.30] | 0.120 | 0.405 [0.34, 0.47] | 0.500 | 0.385 | 3.46 |

**선택:** `s* = −2.0` (두 타깃을 만족하는 최소 |s| — df 상대 10.3 % 감소, em 델타 0 pp,
정확도 여유 충분).

**주목할 점:** `em_num`이 또한 어떤 strength에서도 떨어지지 않음 — 0.370–0.405 범위에서
flat 하며 포화값이 상승. `em_target_only` 0.500으로 불변 (정합 통제 통과). ConvLLaVA의
베이스라인 df가 E1d 베이스라인 (0.29)과 정확히 일치 — eager attention 파이프라인이
런 간 정확히 재현됨을 확인. Strength 응답이 단조적이며 fluency-clean —
`mean_distance_to_anchor`이 strength 전 범위에서 3.18 → 3.46, ≤ 0.3 unit drift, E1d
convllava upper-half ablation의 "no fluency hit" 발견과 일치.

### internvl3-8b (대기)

convllava-7b 완료 즉시 Phase 1 sweep 시작 (동일한 격자).

## Phase 2 — 풀 검증 (대기)

Phase 1에서 유효한 `s*`가 나온 모델(들)에 대해 풀 VQAv2-number 스케일에서 실행
(n=17,730 sample-instances × 5 irrelevant sets × 3 conditions × 2 modes ≈ 100 k generations
per model). Resumable — 모델별 단일 canonical JSONL, append-only flush, 시작 시 완료된
`(sample_instance_id, condition, mask_strength)` 키 집합을 읽어 skip.

**12-h 세션 예산 하의 우선순위 (advisor 호출 기반):** llava-1.5-7b 우선 (E1d 시그널 가장
깨끗, 인과 효과 가장 큼, caveat 없음). 다른 모델은 후속 세션에서 resumability 프로토콜로
이어짐.

## Caveat

- **n=200 CI는 넓다.** Sweep 스케일에서 부트스트랩 CI가 strength들 사이에서 크게 겹친다
  (예: 베이스라인 df 0.305 [0.24, 0.37] vs s=−3.0 df 0.265 [0.21, 0.33]). Strength 축
  단조성은 정보적이지만 strength별 델타는 아직 load-bearing 하지 않다. n=17,730 Phase 2가
  헤드라인 수치를 carrying.
- **ConvLLaVA causal-structure caveat (E1d 출처).** ConvLLaVA와 LLaVA-1.5는 E1b peak·메커니즘
  은 같지만 lower-half ablation에 *반대* 응답 (E1d). 같은 어텐션 시그니처가 같은 인과 구조를
  의미하지는 않는다. Phase 2 수치가 LLaVA-1.5/InternVL3와 크게 다르면 헤드라인
  mid-stack-cluster 주장에서 ConvLLaVA를 demote 하고 discussion caveat로 강등할지
  writeup 시점에 결정.

## 후속 follow-up (Phase 2 후)

- Per-model vs. shared optimal strength — Phase 1에서 3 모델이 비슷한 `s*`를 고르면 단일
  shared strength를 architecture-blind prototype으로 보고; 그렇지 않으면 per-model.
- [−5, 0] 범위에서 어떤 모델도 타깃을 만족하지 못하면 escalate: (a) `ablate_upper_quarter`
  (`[3n/4, n)`), (b) 다른 intervention class.

## Writeup 태그

- `docs/experiments/E4-mitigation.md` (영어 정본).
- `docs/experiments/E4-mitigation_ko.md` (이 파일 — 한국어 미러).
- Phase 2 후 distilled insight pair: `docs/insights/E4-mitigation-evidence.md` + `_ko.md`.
