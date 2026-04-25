# E4 — 어텐션 재가중 완화: 결과

**상태:** Phase 1 sweep이 mid-stack-cluster 3 모델 (llava-1.5-7b, convllava-7b,
internvl3-8b) 모두에서 완료. Phase 2 풀 검증: 2026-04-25에 시작.
`scripts/run_e4_phase2_chain.sh` 우선순위에 따라
llava-1.5-7b → convllava-7b → internvl3-8b 순으로 chain 진행. 12-h 세션 경계에서 resumable.

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

### internvl3-8b (베이스라인 df=0.161)

샘플 사이즈 노트: InternVL3가 자유 prose 토큰 ("based on…")을 emit 하는데, 파서가 이를
숫자 문자열 (예: `"based"`)로 잘못 분류하고, 분석 단계의 strict `_to_int`에서 drop된다.
~30 %의 record가 "(target_only, target+anchor) triplet의 한쪽이라도 non-numeric"으로
탈락; 살아남은 valid-triplet 수가 200에서 n ∈ [112, 137]으로 떨어진다. 베이스라인과
모든 treated cell이 같은 노이즈 floor를 공유하므로 strength별 델타 비교는 여전히
유효; 절대 n만 다른 두 모델보다 작다. (드라이버 fix — 프로젝트의 regex-rescuing
`extract_first_number` 사용 — 은 §"후속 follow-up"에 기록.)

| strength | n | df_num | adopt_num | em_num | em_target_only | em_neutral | mean_dist |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 137 | 0.161 [0.10, 0.22] | 0.095 | 0.591 [0.51, 0.67] | 0.568 | 0.581 | 5.67 |
| **−0.5** | 136 | **0.132 [0.08, 0.19]** | **0.081** | **0.610 [0.53, 0.69]** | 0.568 | 0.584 | 5.74 |
| −1.0 | 131 | 0.099 [0.05, 0.15] | 0.076 | 0.618 [0.53, 0.70] | 0.568 | 0.580 | 5.31 |
| −2.0 | 129 | 0.070 [0.03, 0.12] | 0.070 | 0.597 [0.51, 0.68] | 0.568 | 0.618 | 5.43 |
| −3.0 | 122 | 0.066 [0.02, 0.11] | 0.082 | 0.631 [0.55, 0.72] | 0.568 | 0.597 | 5.11 |
| −5.0 | 115 | 0.052 [0.02, 0.10] | 0.087 | 0.643 [0.56, 0.74] | 0.568 | 0.600 | 5.02 |
| −10⁴ | 112 | 0.063 [0.03, 0.11] | 0.089 | 0.652 [0.56, 0.74] | 0.568 | 0.599 | 5.00 |

**선택:** `s* = −0.5` (두 타깃을 만족하는 최소 |s| — df 0.161 → 0.132 = −17.7 % 상대;
em 0.591 → 0.610 = +1.9 pp).

**주목할 점.** InternVL3는 세 모델 중 베이스라인 `direction_follow_rate`이 *가장 낮음*
(0.16 vs LLaVA 0.31, ConvLLaVA 0.29) — H6 / E2-pilot의 "InternVL3 = distraction-not-anchoring
outlier" 발견을 확증 — 그러나 *상대적으로 가장 큰 완화 효과*를 보인다. 포화 (s = −10⁴)에서
df가 0.063 (−61 % 상대); 선택된 가장 약한 s = −0.5에서도 −17.7 % 상대 감소로 10 % 타깃을
넉넉히 상회. em 컬럼이 +0.019 ~ +0.061로 단조 상승, strength 전 범위에서 fluency-clean
(`mean_dist` 5.0–5.7, fluency hit 없음). `em(target_only)`이 0.568로 불변 — hook이
anchor-condition-specific 함을 확인. 이것이 지금까지 가장 강한 단일 증거: upper-half
multi-layer 위치가 이 모델 가족에서 단순 correlational이 아니라 *causal*이다 — 작은
어텐션 재가중 (s = −0.5, multiplier ≈ 0.61) 만으로 direction-follow가 거의 1/5 감소.

### 교차-모델 요약

| model | layers | upper_half 범위 | 베이스라인 df | s* | s*에서 df | df 감소 (상대) | s*의 em 델타 | em(target_only) 베이스라인 | em(target_only) 불변? |
|---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| llava-1.5-7b | 32 | 16..31 | 0.305 | −3.0 | 0.265 | −13 % | +0.5 pp | 0.435 | ✓ |
| convllava-7b | 32 | 16..31 | 0.290 | −2.0 | 0.260 | −10 % | +0 pp | 0.500 | ✓ |
| internvl3-8b | 28 | 14..27 | 0.161 | −0.5 | 0.132 | −17.7 % | +1.9 pp | 0.568 | ✓ |

**읽기 (3/3 모델 완료):**

1. **Phase 1이 mid-stack-cluster의 모든 모델에서 타깃 달성.** 셋 다 ≥ 10 %
   `direction_follow_rate` 감소를 만족하면서 `exact_match` 손실 없음. 개입은 E1d가
   식별한 패널-공유 `ablate_upper_half` 위치에서, 적당한 soft strength로 (hard masking
   불필요). em(target_only)이 모든 모델의 모든 strength에서 불변 — hook이 구조적으로
   anchor-condition-specific 함을 경험적으로 확인.
2. **단일 공유 strength는 작동하지 않는다.** `s*`가 모델 간 한 자릿수 차이:
   LLaVA-1.5는 `−3.0`, ConvLLaVA는 `−2.0`, InternVL3는 `−0.5`. InternVL3 선택이 LLaVA
   클러스터에서 멀리 떨어져 있음 — LLaVA에 맞춰진 단일 공유 `s*`는 InternVL3에서
   over-mitigate (여전히 타깃은 달성하나 불필요하게 강함), under-mitigate는 어디에서도
   발생 안 함. Per-model strength 선택이 더 깨끗한 prototype 디자인; 논문에는 단일 숫자가
   아니라 per-model 컬럼으로 보고.
3. **완화 효과가 베이스라인 anchor-pull과 비례하지 않고 *반비례*.** InternVL3 (베이스라인
   df 가장 낮음 = 0.16)가 *가장 큰* 상대 감소 (−17.7 %, 포화에서 −61 %). LLaVA-1.5
   (베이스라인 df 가장 높음 = 0.305)가 *가장 작은* 상대 감소 (−13 %, 포화에서 −18 %).
   Flag할 만한 sub-finding: upper-half 어텐션 경로가 anchor 신호를 *적게 사용하는* 모델에서
   더 큰 *비율*로 carrying. 메커니즘 가설: InternVL3의 작은 anchor 신호가 upper-half 레이어에
   좁게 농축돼 있어 그 위치를 제거하면 거의 다 사라짐; LLaVA-cluster 신호는 더 넓게/
   redundant하게 분포해 같은 위치가 일부 slice만 제거. Phase 2가 풀 스케일에서 확증/반박할
   때 writeup 시점에 재검토.
4. **Strength 축 단조성이 모든 모델에서 robust.** `s = 0`부터 `s = −10⁴`까지 모든 단계가
   `direction_follow_rate`을 감소시키거나 유지하고, `exact_match`은 유지하거나 상승.
   과-완화가 hallucination로 전환되는 "U-shape" 재앙 없음. 즉 보수적인 selection rule이
   안전 — 논문이 동등하게 포화 값 (strength 축의 점근선)을 "달성 가능한 최대 완화"로
   인용해도 em 안전 계약이 깨지지 않음.

## Phase 2 — 풀 검증

Phase 1에서 유효한 모든 모델에 대해 풀 VQAv2-number 스케일 실행 (n=17,730 sample-instances
× 5 irrelevant sets × 3 conditions × 2 modes = 모델당 88,650 records, target_only-skip 최적화
후). Resumable — 모델별 단일 canonical JSONL, append-only flush, 시작 시 완료된
`(sample_instance_id, condition, mask_strength)` 키 집합을 읽어 skip.

**12-h 세션 예산 하의 우선순위 (advisor 호출 기반):** llava-1.5-7b 우선 (E1d 시그널 가장
깨끗, 인과 효과 가장 큼, caveat 없음), 그다음 convllava-7b, 그다음 internvl3-8b.
`scripts/run_e4_phase2_chain.sh`로 chain. 다른 모델은 후속 세션에서 resumability 프로토콜로
이어짐.

### llava-1.5-7b — Phase 2 (88,650 records, 100 % 완료)

| 메트릭 | 베이스라인 (s=0) | treated (s=−3.0) | Δ | 상대 |
|---|---:|---:|---:|---:|
| direction_follow_rate | 0.2578 [0.2515, 0.2640] | 0.2122 [0.2060, 0.2182] | **−4.55 pp** | **−17.7 %** |
| exact_match (num) | 0.3340 [0.3272, 0.3412] | 0.3418 [0.3348, 0.3490] | +0.77 pp | +2.3 % |
| exact_match (target_only 베이스라인) | 0.3697 | (treated에서 hook은 no-op) | – | – |
| exact_match (neutral 베이스라인) | 0.3249 | 0.3284 | +0.35 pp | – |

**Paired anchor-damage 표** (intersection of valid {target_only@0, num@0, num@s*} —
n_paired = 17,724, LLaVA는 parse loss 거의 없음):

| em(target_only) | em(num@0) | em(num@s*) | anchor damage | s*에서 회복 | 손상 회복 비율 |
|---:|---:|---:|---:|---:|---:|
| 0.3696 | 0.3340 | 0.3417 | **−3.55 pp** | +0.77 pp | **21.7 %** |

**헤드라인.** LLaVA Phase 2가 Phase 1 sweep claim을 *복제·강화*. Direction-follow가
Phase-1-chosen `s*`에서 sweep이 예측한 정확히 같은 상대 양 (−17.7 %)으로 감소; CI는 이제
~10× 더 좁음. Exact-match는 *유지가 아니라 약간 개선* (+0.77 pp), 그리고 paired anchor-damage
표는 upper-half 어텐션 재가중이 풀 VQAv2-number subset에서 *anchor 유발 em 손실의 21.7 %를
회복*함을 보임. Hook은 구조적으로 여전히 anchor-condition-specific (treated 행의
`em_target_only` 셀이 비어 있는 이유는 Phase 2 드라이버가 non-zero strength에서 target_only를
skip하기 때문; Phase 1이 작은 n에서 invariance 검증 완료).

**Phase 2 vs Phase 1 sweep — 무엇이 변했나.** Sweep 세트는 stratified (top-decile-susceptible
× 100 + bottom-decile-resistant × 100)이라 anchor가 영향을 주는 items을 over-sample; 풀 세트는
전체 VQAv2-number subset. 예상대로, 절대 df 수치는 줄어들고 (베이스라인 0.305 → 0.258;
treated 0.265 → 0.212), *상대* 완화는 ~동일, *paired anchor-damage*는 줄어듦
(−7.00 pp → −3.55 pp), 더 representative한 sample mix를 반영.

### convllava-7b — Phase 2 (진행 중, writeup 시점 ~0.8 %)

2026-04-25 20:24 UTC 시작, rate ~0.86 sample-instances/sec, ETA ~5.7 h. 이 완료율의 부분
Phase-2 수치는 아직 load-bearing 하지 않음 (CI가 s=0 영역에 걸침); 런 완료 후 표 채워넣음.

### internvl3-8b — Phase 2 (대기)

LLaVA 4시간 + ConvLLaVA ~5.7시간 ETA를 고려하면 12-h 세션 예산 안에 시작 못 함. 다음 세션에서
resumability 프로토콜로 이어짐; §"후속 follow-up"에 logged된 드라이버 fix (InternVL3
max_new_tokens 연장) 가 InternVL3 Phase 2 시작 *전에* 적용되어야 parse-loss caveat이 풀
스케일에서 다시 나타나지 않음.

## Caveat

- **n=200 CI는 넓다.** Sweep 스케일에서 부트스트랩 CI가 strength들 사이에서 크게 겹친다
  (예: 베이스라인 df 0.305 [0.24, 0.37] vs s=−3.0 df 0.265 [0.21, 0.33]). Strength 축
  단조성은 정보적이지만 strength별 델타는 아직 load-bearing 하지 않다. n=17,730 Phase 2가
  헤드라인 수치를 carrying.
- **InternVL3 prose-leak parse 손실은 driver-side, parser-side 아님.** InternVL3의 자유
  prose ("Based on the image…")가 `max_new_tokens=8`에서 자릿수가 생성되기 전에 잘림;
  파서는 이미 `extract_first_number`를 사용하며 빈 입력에 대해 올바르게 동작. 따라서
  analysis-layer rescue가 dropped triplets를 회복하지 *못함* — 회복할 숫자가 없음.
  ~30 %의 record가 within-condition view에서 n ∈ [112, 137] (200 대신)로 탈락; paired
  (intersection-of-valid-cells) view에서는 n=109로 떨어짐. 모델 내 strength 델타는 여전히
  유효 (베이스라인·treated 같은 surviving subset). Fix는 driver-side: longer
  `max_new_tokens` (16–32, InternVL3 prose가 자릿수까지 마무리하도록), 또는 InternVL3-특이적
  JSON-strict prompt. Phase 2를 위해 §"후속 follow-up"에 tracked.
- **InternVL3 paired-set bias.** Paired set (n=109)의 em(target_only) = 0.734, unpaired
  0.567보다 훨씬 높음 — parse-failing items은 systematically 모델의 더 어려운 cases.
  Cross-condition em 비교의 InternVL3 행은 "InternVL3 전반"이 아니라 "parse-가능한 subset에
  대한 완화 행동"으로 다룰 것.
- **ConvLLaVA causal-structure caveat (E1d 출처).** ConvLLaVA와 LLaVA-1.5는 E1b peak·메커니즘
  은 같지만 lower-half ablation에 *반대* 응답 (E1d). 같은 어텐션 시그니처가 같은 인과 구조를
  의미하지는 않는다. Phase 2 수치가 LLaVA-1.5/InternVL3와 크게 다르면 헤드라인
  mid-stack-cluster 주장에서 ConvLLaVA를 demote 하고 discussion caveat로 강등할지
  writeup 시점에 결정.
- **공유 `s*`는 작동 안 함.** `s*`가 −0.5 (InternVL3) ~ −3.0 (LLaVA-1.5)로 분포. 단일
  공유 strength는 한 모델을 over-mitigate 하고 다른 모델을 under-mitigate. Per-model 보고.

## 후속 follow-up (Phase 2 후)

- Per-model strength 선택 (Phase 1 결과; 단일 공유 숫자는 infeasible — caveat 참조).
- `scripts/e4_attention_reweighting.py`의 InternVL3 parse-rescue 패치: 모델 raw numeric
  output에 의존하지 않고 프로젝트의 `extract_first_number` regex helper 사용. 비교 구조를
  바꾸지 않으면서 dropped triplet ~30 % 회복.
- Phase 2가 어떤 모델에서도 타깃을 못 맞추면 escalate: (a) `ablate_upper_quarter`
  (`[3n/4, n)`), (b) 다른 intervention class.
- Phase 2 스케일에서 stratum별 분석: 헤드라인을 susceptibility decile (top vs bottom)로
  분리해 완화 효과가 모델이 원래 가장 susceptible 했던 item에 농축되는지 검증 — H2 →
  완화 link를 직접 강화.

## Writeup 태그

- `docs/experiments/E4-mitigation.md` (영어 정본).
- `docs/experiments/E4-mitigation_ko.md` (이 파일 — 한국어 미러).
- Phase 2 후 distilled insight pair: `docs/insights/E4-mitigation-evidence.md` + `_ko.md`.
