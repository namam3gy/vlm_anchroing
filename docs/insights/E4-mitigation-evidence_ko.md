# E4 — upper-half 어텐션 재가중이 mid-stack cluster의 anchor pull을 줄인다

**상태:** llava-1.5-7b Phase 1 sweep 완료; convllava-7b·internvl3-8b sweep 진행 중.
Phase 2 풀 검증 대기. 소스 데이터:
`outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl`. 집계 표:
`outputs/e4_mitigation/_summary/{sweep_pareto.csv, chosen_strength.json}`. 상세 writeup:
`docs/experiments/E4-mitigation.md`.

## 주장과 검증

E1d는 `ablate_upper_half`가 패널 6/6 모델에서 `direction_follow_rate`를 떨어뜨리면서
mid-stack cluster의 fluency를 깨지 않은 유일한 multi-layer 어텐션 마스크 개입임을 보였다.
E4는 hard masking을 부드러운 strength 축(`exp(strength)` multiplier on anchor attention)
으로 대체하고, n=200 stratified 샘플에서 sweep 한 뒤, ≥ 10 % 상대적 `direction_follow_rate`
감소와 ≤ 2 pp 표준 VQA 정확도 손실을 만족하는 가장 작은 |strength|를 선택한다.

## 결과 (예비, n=200)

| model | 베이스라인 df | s* | s*에서 df | s=0의 em(num) | s*의 em(num) | em(target_only) 불변? |
|---|---:|---:|---:|---:|---:|:---:|
| llava-1.5-7b | 0.305 | **−3.0** | 0.265 (−13 %) | 0.365 | 0.370 (+0.5 pp) | ✓ (0.435) |
| convllava-7b | 0.290 | **−2.0** | 0.260 (−10 %) | 0.375 | 0.375 (+0.0 pp) | ✓ (0.500) |
| internvl3-8b | _대기_ | _대기_ | _대기_ | _대기_ | _대기_ | _대기_ |

**헤드라인 (현재까지):** 완화책이 llava-1.5-7b의 n=200에서 작동하고 roadmap 타깃을
만족한다. 결정적으로 `em(target_plus_irrelevant_number)`가 *떨어지지 않는다* — 포화
(`s = −10⁴`)에서 0.365 → 0.395로 *상승*. 정확도에 안전할 뿐 아니라 약간 *유익*함을
시사. Phase 2 풀 스케일이 이게 실제인지를 결정.

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
- **ConvLLaVA causal-structure caveat (E1d 출처).** 같은 어텐션 시그니처 ≠ 같은 인과 구조.
  ConvLLaVA의 lower-half ablation 응답은 LLaVA-1.5의 *반대*. Phase 2에서 ConvLLaVA의 E4
  응답이 LLaVA-1.5/InternVL3와 다르게 발산하면 discussion caveat로 강등.
- **n=200 선택이 n=17,730에서 변할 수 있다.** Phase 2가 모델별로 다른 `s*`를 고를 수 있음.
  Phase 2는 Phase 1-chosen `s*`와 풀 부트스트랩 분포를 모두 보고.

## 후속 follow-up (Phase 2 후)

- mid-stack 3 모델 모두에 universal한 패턴인지 LLaVA-1.5만 인지.
- 포화에서 정확도 상승이 anchor-특이적인지 (두 번째 이미지 어텐션을 떨어뜨릴 때 generic
  accuracy bump가 아님을 배제).
- 완화책이 작동하면 VQAv2-number 외에 ChartQA/TallyQA(E5)에서도 일반화되는지.
