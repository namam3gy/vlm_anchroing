# E4 — upper-half 어텐션 재가중이 mid-stack cluster의 anchor pull을 줄인다

**상태:** mid-stack-cluster 3 모델 (llava-1.5-7b, convllava-7b, internvl3-8b) Phase 1
sweep 모두 완료. Phase 2 풀 검증 진행 중 (2026-04-25 시작; chained
llava → convllava → internvl3). 소스 데이터:
`outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl`. 집계 표:
`outputs/e4_mitigation/_summary/{sweep_pareto.csv, chosen_strength.json}`. 상세 writeup:
`docs/experiments/E4-mitigation.md`.

## 주장과 검증

E1d는 `ablate_upper_half`가 패널 6/6 모델에서 `direction_follow_rate`를 떨어뜨리면서
mid-stack cluster의 fluency를 깨지 않은 유일한 multi-layer 어텐션 마스크 개입임을 보였다.
E4는 hard masking을 부드러운 strength 축(`exp(strength)` multiplier on anchor attention)
으로 대체하고, n=200 stratified 샘플에서 sweep 한 뒤, ≥ 10 % 상대적 `direction_follow_rate`
감소와 ≤ 2 pp 표준 VQA 정확도 손실을 만족하는 가장 작은 |strength|를 선택한다.

## 결과 (Phase 1, 모델당 n=200 stratified)

| model | 베이스라인 df | s* | s*에서 df | s=0의 em(num) | s*의 em(num) | em(target_only) 불변? |
|---|---:|---:|---:|---:|---:|:---:|
| llava-1.5-7b | 0.305 | **−3.0** | 0.265 (−13 %) | 0.365 | 0.370 (+0.5 pp) | ✓ (0.435) |
| convllava-7b | 0.290 | **−2.0** | 0.260 (−10 %) | 0.375 | 0.375 (+0.0 pp) | ✓ (0.500) |
| internvl3-8b | 0.161 | **−0.5** | 0.132 (−17.7 %) | 0.591 | 0.610 (+1.9 pp) | ✓ (0.568) |

**헤드라인:** 완화책이 mid-stack-cluster 모든 모델의 n=200에서 타깃을 달성. 셋 다
≥ 10 % `direction_follow_rate` 감소를 만족하면서 em이 flat 또는 상승. 결정적으로
`em(target_plus_irrelevant_number)`가 어느 모델에서도 *떨어지지 않음* — 포화 (`s = −10⁴`)
에서 +0.030 ~ +0.061 상승. 완화책이 정확도에 안전할 뿐 아니라 약간 *유익*함. Phase 2
풀 스케일이 좁은 CI로 이를 확증할 것이다.

**반비례 surprise.** 완화 효과가 베이스라인 anchor-pull과 비례하지 않고 *반비례*. InternVL3
가 가장 낮은 베이스라인 df (0.161, H6의 "distraction-not-anchoring" 모델)인데도 가장 큰
상대 감소 (선택된 s*에서 −17.7 %, 포화에서 −61 %). LLaVA-1.5가 가장 높은 베이스라인 df
(0.305)인데 가장 작은 상대 감소 (s*에서 −13 %, 포화에서 −18 %). Upper-half 어텐션 경로가
*적게 사용하는* 모델에서 anchor 신호의 *더 큰 비율*을 carrying — InternVL3에선 anchor 신호가
upper-half 레이어에 좁게 농축되어 있고, LLaVA-cluster에선 넓게/redundant하게 분포한다는
가설과 일치. Phase 2 스케일에서 재검토.

## 조건별 정합 통제 (llava-1.5-7b, n=200)

Strength hook이 두 번째 이미지 토큰 span에 부착됨; `target_only`(두 번째 이미지 없음)에서
hook은 구조적으로 no-op, `target_plus_irrelevant_neutral`에서는 hook이 firing 하지만
두 번째 이미지에 가독 자릿수 없음. Phase 1 확인:

- `em(target_only)`: **0.435 불변**, 모든 7 strength에서 동일 — hook이 single-image
  추론에 leak 하지 않음.
- `em(target_plus_irrelevant_neutral)`: 0.355–0.365 (Δ ≤ 0.01) — hook이 firing 하지만
  제거할 anchor 신호가 없으므로 예측이 변하지 않음.
- `em(target_plus_irrelevant_number)`: 0.365 → 0.395 (+3 pp) — 예측이 hook 아래에서
  변하는 유일한 조건이며, 변화 방향은 정답 쪽.

## 더 강한 프레이밍 — anchor 손상과 부분 회복

`em_target_only`을 "anchor 없는" 상한으로 읽으면 em 컬럼들이 "완화책이 정확도에 안전"보다
더 날카로운 claim을 가능하게 한다:

| model | em(target_only) | em(num) at s=0 | anchor 유발 em 손실 | em(num) at 포화 s=−10⁴ | 포화 회복 |
|---|---:|---:|---:|---:|---:|
| llava-1.5-7b | 0.435 | 0.365 | −0.070 (상대 16 %) | 0.395 | +0.030 (손실의 43 % 회복) |
| convllava-7b | 0.500 | 0.375 | −0.125 (상대 25 %) | 0.405 | +0.030 (손실의 24 % 회복) |
| internvl3-8b | 0.568 | 0.591 | +0.023 (손상 없음) | 0.652 | +0.061 (anchor 조건 em 더 상승) |

InternVL3는 다르게 읽힌다: s=0에서 em(num) ≥ em(target_only) — 이 stratified 세트에서
anchor가 em을 손상시키지 않는다. (Caveat: em 컬럼들은 InternVL3의 parse-leak 때문에 다른
surviving sample 부분집합에서 계산됨, n=137 vs n=200; 조건 간 절대 em 수준 비교는 근사적.
load-bearing 신호는 target_plus_irrelevant_number 내의 strength 축 단조성.) 따라서
anchor-damage / partial-recovery 프레이밍은 LLaVA-cluster (LLaVA-1.5, ConvLLaVA)에서만
깨끗하게 holds — anchor가 정확도를 손상시키고, upper-half 어텐션 재가중이 손상의 일부를
회복. InternVL3에서는 같은 개입이 anchor damage 신호가 0으로 lower-bounded인 모델에서도
em을 상승시킴 — upper-half 위치가 유용한 일을 하고 있다는 뜻.

선택된 운영 지점 `s*`에서는 회복이 더 작음 (LLaVA / ConvLLaVA에서 em delta ≈ 0 pp,
InternVL3에서 +1.9 pp); 가시적 회복은 포화에서 발생. Phase 2가 (a) `s*`가 너무 약해 운영
지점을 strength 축의 더 깊은 곳으로 옮겨야 하는지, 또는 (b) 포화가 과도하고 `s*`에서의
적당한 df 감소가 올바른 운영 지점인지 결정.

## 의의

E1d가 한 열린 질문(E1b의 per-layer 어텐션 peak는 correlational, 인과 경로는 multi-layer)
을 닫고 완화 질문을 열었다. E4는 이 모델, 이 스케일에서 그것을 닫는다. Phase 2가 유지되면
잠재적 paper-level claim 두 가지:

1. **단일 architecture-blind 개입 위치(upper-half LLM 레이어)가 VLM mid-stack cluster의
   cross-modal anchoring을 줄인다.** Mid-stack cluster(LLaVA-1.5, ConvLLaVA, InternVL3)는
   같은 E1b peak·메커니즘과 fluency-clean upper-half-ablation 응답을 공유 (E1d).
2. **완화책이 정확도 손실을 동반하지 않는다.** E1d의 hard mask가 풀 ablation에서 fluency
   리스크를 표출했지만, E4의 soft strength 영역에는 direction-follow가 떨어지면서
   exact-match가 안정/상승하는 운영 지점이 존재.

## Caveat

- **n=200 CI는 넓다.** Sweep 스케일에서 strength들 사이의 부트스트랩 CI가 크게 겹친다.
  Strength 축 단조성은 정보적이지만 strength별 델타는 아직 load-bearing 하지 않다.
  n=17,730 Phase 2가 헤드라인 수치를 carrying.
- **InternVL3 prose-leak parse 손실.** InternVL3가 prose 토큰을 emit하고 파서가 이를
  숫자 문자열로 잘못 분류해 ~30 %의 record가 valid-triplet에서 탈락. 모델 내 strength별
  델타 비교는 여전히 유효 (베이스라인·treated 같은 노이즈 공유), 하지만 anchor-damage 표의
  조건 간 절대 em 비교는 근사적. Phase 2를 위해 드라이버 fix (regex parse-rescue,
  `scripts/e4_attention_reweighting.py`) 로깅됨.
- **ConvLLaVA causal-structure caveat (E1d 출처).** 같은 어텐션 시그니처 ≠ 같은 인과 구조.
  ConvLLaVA의 lower-half ablation 응답은 LLaVA-1.5의 *반대*. Phase 2에서 ConvLLaVA의 E4
  응답이 LLaVA-1.5/InternVL3와 다르게 발산하면 discussion caveat로 강등.
- **n=200 선택이 n=17,730에서 변할 수 있다.** Phase 2가 모델별로 다른 `s*`를 고를 수 있음.
  Phase 2는 Phase 1-chosen `s*`와 풀 부트스트랩 분포를 모두 보고.
- **Per-model `s*`, 공유 아님.** `s*`가 −0.5 (InternVL3) ~ −3.0 (LLaVA-1.5) 분포; 단일 공유
  strength는 한 모델 over-mitigate, 다른 모델 under-mitigate. 완화책은 *위치 + selection
  rule*로 일반화되며, 단일 strength 상수로는 아님.

## 후속 follow-up (Phase 2 후)

- 베이스라인 anchor-pull과 완화 효과 크기의 반비례 (InternVL3 = 가장 낮은 베이스라인 / 가장
  큰 완화; LLaVA-1.5 = 가장 높은 베이스라인 / 가장 작은 완화)가 풀 스케일에서 holds 하는지.
  holds 하면 upper-half 위치가 mid-stack cluster 전반에 anchor 신호를 어떻게 공유하는지에
  관한 paper-grade 발견.
- 포화에서 정확도 상승이 anchor-특이적인지 (두 번째 이미지 어텐션을 떨어뜨릴 때 generic
  accuracy bump가 아님을 배제).
- 완화책이 작동하면 VQAv2-number 외에 ChartQA/TallyQA(E5)에서도 일반화되는지.
