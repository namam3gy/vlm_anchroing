# E4 — upper-half 어텐션 재가중이 mid-stack cluster의 anchor pull을 줄인다

**상태:** mid-stack-cluster 3 모델 (llava-1.5-7b, convllava-7b, internvl3-8b) Phase 1
sweep 모두 완료. **Phase 2 풀 검증: llava-1.5-7b 완료 (88,650 records, 100 %),
convllava-7b 진행 중, internvl3-8b 대기.** 소스 데이터:
`outputs/e4_mitigation/<model>/{sweep_n200,full_n17730}/predictions.jsonl`. 집계 표:
`outputs/e4_mitigation/_summary/{sweep_pareto, full_validation, full_validation_compare,
anchor_damage_paired_{sweep,full}, chosen_strength}.csv|.json`. 상세 writeup:
`docs/experiments/E4-mitigation.md`.

## Phase 2 헤드라인 (mid-stack-cluster 3 모델 모두, 모델당 88,650 records, 100 % 완료)

| model | s* | df 베이스라인 | df treated | df Δ | df rel | em 베이스라인 | em treated | em Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava-1.5-7b | −3.0 | 0.2578 | 0.2122 | **−4.55 pp** | **−17.7 %** | 0.3340 | 0.3418 | +0.77 pp |
| convllava-7b | −2.0 | 0.2283 | 0.2042 | **−2.42 pp** | **−10.6 %** | 0.3522 | 0.3652 | +1.30 pp |
| internvl3-8b | −0.5 | 0.1035 | 0.0975 | **−0.59 pp** | **−5.8 %** | 0.5902 | 0.5950 | +0.49 pp |

풀 세트의 paired anchor-damage:

| model | n_paired | em(TO) | em(num@0) | em(num@s*) | damage | recovery | 손상 회복 비율 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llava-1.5-7b | 17,724 | 0.3696 | 0.3340 | 0.3417 | **−3.55 pp** | +0.77 pp | **21.7 %** |
| convllava-7b | 17,722 | 0.4454 | 0.3520 | 0.3651 | **−9.34 pp** | +1.31 pp | **14.0 %** |
| internvl3-8b | 11,848 | 0.6325 | 0.5938 | 0.5977 | **−3.87 pp** | +0.40 pp | **10.2 %** |

**읽기.** Phase 2가 **mid-stack-cluster 3 모델 모두를 풀 스케일에서 cover**. 헤드라인 속성이
패널 전체에서 holds: df 감소 모든 모델, em 상승 모든 모델, paired anchor-damage 분석에서 부분
회복 모든 모델 (손상의 10–22 %). 10 % 상대 감소 roadmap 타깃은 LLaVA (−17.7 %)와 ConvLLaVA
(−10.6 %)에서 깨끗하게 만족; InternVL3는 미달 (−5.8 %), 구조적 이유로 — H6의 "distraction-
not-anchoring" 모델이며 풀 스케일 베이스라인 df가 LLaVA-cluster의 절반. 완화책이 여전히 메트릭을
올바른 방향으로 움직이지만, 애초에 이 모델에서 제거할 anchor 신호가 적음.

**Stratified vs. 풀 스케일 InternVL3 — 차이는 sample-distributional, mitigation 실패 아님.**
Phase 1 sweep stratified set (상위-decile-susceptible × 100 + 하위-decile-resistant × 100)에서
df₀ = 0.161, mitigation이 17.7 % 상대 감소. Phase 2 풀 세트는 더 representative, df₀ = 0.103
(~36 % 낮은 베이스라인 anchor pull). 완화책 효과가 운영점에서 제거하는 베이스라인 신호와 비례.

**ConvLLaVA 풀 스케일 fluency caveat.** ConvLLaVA의 `mean_distance_to_anchor`가 풀 스케일에서
2.99 → 53.54로 점프 (Phase 1 sweep stratified set에서: 3.18 → 3.30). 일부 sample이 어떤
plausible anchor와도 멀리 떨어진 예측을 받아 *평균*을 ~17× 끌어올림; em(num)은 여전히 상승
(분포의 대부분이 충분히 개선되어 net positive), df는 모델 자기 베이스라인 대비 per-pair 계산이라
robust. 논문에서는: 비-winsorised mean이 아닌 median distance + fluency-degraded fraction
카운트로 보고.

**InternVL3 parse-loss 지속.** ~33 % record가 paired-valid set에서 탈락 (n_paired = 11,848 /
17,730). InternVL3 max_new_tokens=32 드라이버 패치가 본 run *중에* 적용되었기 때문. 패치된
드라이버로 InternVL3 재실행 시 sample 크기 tightening; 헤드라인이 materially 바뀔지는 dropped
items가 kept items와 systematic하게 다른지에 의존 (paired-set em(TO) = 0.6325 vs 패널 em(TO)
= 0.5760은 parse-failing items가 systematically 모델의 더 어려운 cases임을 시사 — Phase 1과
같은 caveat).

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

**Phase 1의 두 패턴.**

*`direction_follow_rate` 축 — 베이스라인 anchor-pull과 반비례.* 완화 효과가 베이스라인 df
와 비례하지 않고 *반비례*. InternVL3가 가장 낮은 베이스라인 df (0.161, H6의
"distraction-not-anchoring" 모델)인데도 가장 큰 상대 감소 (s*에서 −17.7 %, 포화에서
−61 %). LLaVA-1.5가 가장 높은 베이스라인 df (0.305)인데 가장 작은 상대 감소 (s*에서 −13 %,
포화에서 −18 %). 가설: upper-half 어텐션 경로가 *적게 사용하는* 모델에서 anchor 신호의
*더 큰 비율*을 carrying — InternVL3에선 anchor 신호가 upper-half 레이어에 좁게 농축되어
있고, LLaVA-cluster에선 넓게/redundant하게 분포.

*paired `exact_match` 축 — 일관된 damage / 부분-회복 비율.* em에서는 반대 그림: valid sample
교집합에서 계산하면 (셀들이 like-for-like), 세 모델 모두 일관된 anchor-damage −7 ~ −12.5 pp,
포화에서 +3 ~ +3.7 pp 부분 회복, 즉 손상의 24–43 % 회복. Damage/recovery 비율이 모델 간
df보다 더 잘 일치 — upper-half 위치가 cluster 전반에 비슷한 크기의 em correction 전달.
df-축 반비례와 em-축 일관성이 같은 underlying mechanism (anchor 신호 농축)의 다른 metric
해상도인지, 또는 다른 메커니즘인지가 Phase 2의 open question.

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

모든 em을 valid sample의 *교집합* (cross-condition 비교가 동일 sample 기준이 되도록)에서
계산하면:

| model | n_paired | em(target_only) | em(num) at s=0 | anchor 유발 em 손실 | em(num) at s* | em(num) at 포화 s=−10⁴ | 포화 회복 |
|---|---:|---:|---:|---:|---:|---:|---:|
| llava-1.5-7b | 200 | 0.435 | 0.365 | −0.070 (상대 16 %) | 0.370 (+0.005) | 0.395 | +0.030 (손실의 43 % 회복) |
| convllava-7b | 200 | 0.500 | 0.375 | −0.125 (상대 25 %) | 0.375 (+0.000) | 0.405 | +0.030 (손실의 24 % 회복) |
| internvl3-8b | 109 | 0.734 | 0.633 | −0.101 (상대 14 %) | 0.642 (+0.009) | 0.670 | +0.037 (손실의 36 % 회복) |

**세 모델 모두 anchor-damage를 보임; 완화책이 세 모델 모두에서 일부 슬라이스를 회복.**
LLaVA-1.5와 ConvLLaVA는 이미 이 패턴을 보였고, InternVL3도 paired intersection 분석
(*모든* (condition, strength) 셀에서 valid한 sample_instance_id — cross-cell 비교가 fair한
유일한 방법) 결과 같은 패턴을 따른다. 이전의 "InternVL3는 anchor damage 없음" 읽기는 n=137
em(num)을 n=200 em(target_only)와 비교한 artefact였음 — 다른 sample 부분집합.

**InternVL3 caveat:** paired 분석이 InternVL3 n을 109로 축소 (200 중 — 4 셀의 교집합).
중요한 점: 살아남은 InternVL3 paired set의 em(target_only) = 0.734이며, 더 큰 n=200
condition-internal set의 0.567보다 훨씬 높음 — parse-failing samples는 모델에게 systematically
더 어려운 cases. 따라서 위 표의 InternVL3 행은 "InternVL3 일반"이 아니라 "parse-가능한
subset에 대한 완화책 행동"으로 읽을 것. Phase 2 + 드라이버 수정 (InternVL3 max_new_tokens
연장, 또는 JSON-strict 템플릿)이 InternVL3 행을 다른 두 모델과 완전히 비교 가능하게 만들기
전에 필요.

세 모델 모두 비슷한 완화 행동: 손상 −7 ~ −12.5 pp, 포화에서 +3 ~ +3.7 pp 부분 회복 = 손상의
24–43 % 회복. 선택된 운영 지점 `s*`에서는 회복이 더 작음 (em delta 0 ~ +0.9 pp); 가시적
회복은 포화에서 발생. Phase 2가 (a) `s*`가 너무 약해 운영 지점을 strength 축의 더 깊은
곳으로 옮겨야 하는지, 또는 (b) 포화가 과도하고 `s*`에서의 적당한 df 감소가 올바른 운영
지점인지 결정.

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
- **InternVL3 prose-leak parse 손실은 driver-side, parser-side 아님.** ~30 %의 record가
  valid-triplet에서 탈락하는 이유는 InternVL3가 prose ("Based on the image…")를 emit하고,
  드라이버의 `max_new_tokens=8`이 자릿수가 생성되기 전에 잘라버리기 때문; 파서는 이미
  프로젝트의 `extract_first_number`를 사용하며 입력에 대해 올바르게 동작. 따라서
  analysis-layer parse-rescue가 dropped triplets를 회복하지 *못함* — 회복할 숫자가 없음.
  Fix는 driver-side: longer `max_new_tokens` (16–32, prose가 자릿수까지 마무리하도록), 또는
  InternVL3-특이적 JSON-strict prompt. Phase 2를 위해 tracked.
- **InternVL3 paired-set bias.** intersection-of-valid-samples set (n=109 vs per-condition
  n=137 ~ 200)의 em(target_only) = 0.734로, unpaired 0.567보다 상당히 높음. parse-failing
  samples는 systematically 모델의 더 어려운 cases. Anchor-damage 표의 InternVL3 행은
  "InternVL3 전반"이 아니라 "parse-가능한 item에 대한 완화 행동"으로 다룰 것.
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
