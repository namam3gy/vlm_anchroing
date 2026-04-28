# E4 — attention re-weighting mitigation prototype: 설계

**상태:** 구현 전 설계. 사용자 승인 2026-04-25. `references/roadmap.md` §6 Tier 1의 open E4 행을 대체. Phase 1 + 2 완료 시 결과 writeup `docs/experiments/E4-mitigation.md`로 교체됨.

## 목표

E1d finding ("upper-half ablation이 fluency를 깨지 않으면서 `direction_follow_rate`을 줄이는 단일 architecture-blind locus")를 따라 mid-stack-cluster VLM (LLaVA-1.5, ConvLLaVA, InternVL3)에 대해 동작하는 inference-time mitigation 구현.

**목표치** (roadmap §6 기준): VQAv2 number subset에서 `direction_follow_rate` ≥ 10 % 감소 + standard VQA accuracy ≤ 2 pp 하락.

## 왜 이 설계

E1d는 single-layer intervention을 배제 (peak ablation, layer-0 control 모두 6/6 null). 남은 viable class는 multi-layer attention re-weighting. E1d는 또한 lower-half ablation도 배제 (3/6 BACKFIRE). Hard mask (−1e4)의 upper-half ablation이 패널 전체에서 작동한 유일한 모드이고, mid-stack cluster에 대해서는 fluency 청결 (mean_distance_to_anchor가 baseline에서 ~1 단위 이내). E4는 그 결과를 두 가지 변경으로 확장:

1. **Strength knob** — hard masking 대신 가변 강도의 soft re-weighting. Hard mask는 anchor attention을 ≈ 0으로 죽임; softer mask는 `exp(strength)`만큼 down-scale. Strength axis를 탐색하면 direction-follow 감소와 accuracy 보존을 trade-off 가능, roadmap target을 만족하는 운영점 결정.
2. **Accuracy 메트릭** — E1d는 `direction_follow_rate`과 `mean_distance_to_anchor`만 trace. E4 target은 standard-VQA-accuracy 측면에서 정의되므로 스크립트가 `exact_match`를 ground truth와 비교해 기록해야 함.

## Phase 1 — Strength sweep (n=200, mid-stack 3 모델)

### 무엇

각 모델 {LLaVA-1.5, ConvLLaVA, InternVL3}에 대해 E1b-stratified n=200 question set (top-decile-susceptible × 100 + bottom-decile-resistant × 100)을 `target_only` / `target_plus_irrelevant_number` / `target_plus_irrelevant_neutral` × **7개 strength 값**의 upper-half attention re-weighting 하에 실행:

| strength | post-softmax anchor-attention 곱셈자 `exp(strength)` | 의미 |
|---:|---:|---|
| 0 | 1.000× | baseline (mask 없음) |
| −0.5 | 0.607× | gentle |
| −1.0 | 0.368× | moderate |
| −2.0 | 0.135× | strong |
| −3.0 | 0.050× | very strong |
| −5.0 | 0.0067× | near-zero |
| −1e4 | ≈ 0 | hard mask (E1d control) |

Mask 개입은 `[n_layers/2, n_layers)`의 각 LLM decoder layer에 forward pre-hook으로 구현, anchor-image-token span의 `attention_mask` 칼럼에 `strength`를 더함. 수학: `attention_mask`는 softmax 전에 더해지므로 anchor 칼럼에 `s`를 더하면 post-softmax anchor-attention weight가 `exp(s)`만큼 곱해짐. Strength 값은 의미 있는 범위에 logarithmically 분포 — `−5` 이상에서는 attention weight가 이미 baseline의 < 1 %.

### 메트릭 (각 (model, strength, condition) 셀, n=200)

- `direction_follow_rate` (E1d에서 carry over)
- `adoption_rate`
- `mean_distance_to_anchor` (fluency monitor)
- **`exact_match` rate** — `# pred_number == ground_truth ÷ valid triplets`. Roadmap-target 분모로 사용되는 standard-VQA-accuracy proxy. (Full VQAv2 10-annotator soft accuracy는 본 subset에 없음; exact_match가 가장 해석 가능한 proxy.)
- 모두 bootstrap 95 % CI (2,000 iter).

### 출력

Pareto 그래프 (x-axis = strength, y-axis = `direction_follow_rate`과 `exact_match(target_plus_irrelevant_number)`을 별개 scale로). 다음을 만족하는 **가장 작은 |strength|** 선택:

```
direction_follow_rate(target_plus_irrelevant_number, strength)
  ≤ 0.9 × direction_follow_rate(target_plus_irrelevant_number, strength=0)
AND
exact_match(target_plus_irrelevant_number, strength)
  ≥ exact_match(target_plus_irrelevant_number, strength=0) − 0.02
```

Accuracy criterion이 `target_plus_irrelevant_number`에서 평가되는 이유는 mitigation이 실제로 발동하는 condition이기 때문 — upper-half hook은 anchor span의 attention만 수정하고, anchor span이 `target_only`에서는 비어있음 (=(0, 0)). Sanity check로 모든 strength에서 `exact_match(target_only, strength) ≈ exact_match(target_only, strength=0)` 검증 — 거기서 drift가 있다면 hook이 single-image inference로 새고 있다는 뜻이고, 그건 버그.

어느 strength도 두 criterion을 만족 못 하면 다음으로 escalate: (a) 더 dense한 strength 격자, (b) `ablate_upper_quarter` (`[3n/4, n)`) — 더 좁은 layer 밴드, (c) 모델별 strength 선택.

### 컴퓨팅

모델당: 200 sample × 3 condition × 7 strength = 4,200 generation × 약 0.5–1 s = 35–70 분. 3 모델 sequential on GPU 0 = **약 2–4 시간 총**.

## Phase 2 — Full-scale 검증 (n=17,730, mid-stack 3 모델, 단일 최적 strength)

### 무엇

Phase 1이 모델별 최적 strength `s*`를 고른 후 (혹은 모델별 optima가 일치하면 한 공유 strength), full VQAv2 number subset (n=17,730 sample-instance, 질문당 5 irrelevant set)을 3 condition × 2 mode (baseline vs upper-half re-weighting at `s*`)로 실행.

### 컴퓨팅

모델당: 17,730 × 3 × 2 ≈ **106 k generation** × 약 0.5–1 s ≈ **15–30 시간/모델**. 3 모델 sequential = **45–90 시간**.

### Resumability 요건

Phase 2는 한 번의 끊김 없는 세션으로 합리적으로 완료할 수 없음. 스크립트가 kill/crash/재시작에 걸쳐 resumable해야 함:

- **출력 구조**: `outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl` — (model, phase) 당 canonical 파일 1개. 모든 record append.
- **Resume 프로토콜**: 시작 시 스크립트가 기존 JSONL (있다면)을 읽어 완료된 `(sample_instance_id, condition, mode_strength)` key set을 빌드, iteration 중 그 키는 skip. 새 record는 같은 파일에 append.
- **Crash 안전성**: 각 record는 `fh.write(json.dumps(...) + "\n")` 후 `fh.flush()`. Kill 시 최대 마지막 (쓰기 중인) 라인이 partial. Reader는 `json.loads(line)` 주위에 try/except로 malformed 마지막 라인을 silently skip.
- **Run-once semantics**: 같은 명령 재실행은 파일 완료 시 no-op. Append-only write로 이미 완료된 run을 recompute 없이 다시 로드 가능.
- **종료 시 검증**: 스크립트가 종료 시 expected vs. actual record count 출력 (e.g., "completed 106,380 / 106,380"). 종료 시 ≥ 1 record 부족이면 missing key 출력.

### 메트릭 (각 (model, mode, condition))

Phase 1과 동일한 4개 메트릭, full-scale n에 대한 bootstrap 95 % CI.

## 코드 구조

신규 스크립트 2개 + writeup 페어 1개.

- **`scripts/e4_attention_reweighting.py`** — `scripts/causal_anchor_ablation.py`의 hook/anchor-span/runner-build plumbing 기반 (import, 중복 아님). 추가:
  - `--strength FLOAT` (default `-1e4`); `0`이 아닐 때 upper-half hook 설치
  - `--phase {sweep,full}`이 sample-set 출처 (sweep은 n=200 stratified, full은 full VQAv2 number subset)와 출력 디렉터리 레이아웃 결정
  - Resumability (Phase 2 hard 요건; Phase 1에서는 무해 overhead)
  - JSONL에 기존 E1d 필드와 함께 `parsed_number`, `ground_truth`, `exact_match` 기록
- **`scripts/analyze_e4_mitigation.py`** — `(model, strength, condition)` 셀별로 bootstrap CI와 함께 집계; Pareto 그래프, per-strength summary CSV, full-validation CSV 생성; Phase 1의 strength-selection rule 인코딩 후 선택된 `s*` 출력.
- **`docs/experiments/E4-mitigation.md`** (+ `_ko.md`) — Phase 1이 strength 선택을 산출하면 작성; full validation 끝나면 Phase 2 숫자로 update. **본 설계 doc (`E4-mitigation-design.md` + `_ko.md`)는 그 시점에 결과 writeup으로 대체**되지만 git history에 보존.

### 출력 디렉터리

```
outputs/e4_mitigation/
  <model>/
    sweep_n200/
      predictions.jsonl     # Phase 1: 4,200 record (3 condition × 7 strength × 200)
    full_n17730/
      predictions.jsonl     # Phase 2: ~106k record (3 condition × 2 mode × 17,730)
  _summary/
    sweep_pareto.csv
    sweep_pareto.png
    chosen_strength.json    # {model: chosen_strength_s*}
    full_validation.csv
    full_validation_summary.md
```

## Writeup 단계에서 재검토할 open question

- **Paper-grade 결과에서 ConvLLaVA 포함 여부.** ConvLLaVA를 Phase 1과 Phase 2에 모두 포함했지만, E1d sub-finding ("LLaVA-1.5와 동일한 E1b peak/메커니즘에도 불구하고 lower-half ablation 행동이 *반대*")이 causal-structure caveat를 표시. Phase 2 숫자가 ConvLLaVA의 E4 반응이 LLaVA-1.5/InternVL3와 substantially 다르거나 unstable하면 **writeup 시점에 paper의 헤드라인 mid-stack-cluster 클레임에서 drop할지 결정**, discussion caveat로 강등. 결정은 `E4-mitigation.md`에 문서화.
- **모델별 vs 공유 최적 strength.** Phase 1이 3 모델에 걸쳐 비슷한 `s*`를 고르면 (e.g., 전부 {−1, −2} 안), 단일 공유 strength를 architecture-blind prototype으로 보고. `s*`가 의미 있게 갈라지면 (e.g., LLaVA-1.5는 −0.5, InternVL3은 −3), 모델별 strength가 prototype spec의 일부가 됨 — "encoder를 가로지르는 하나의 mitigation" 클레임을 약화시키지만 죽이지는 않음.
- **실패 시 escalation path.** [−5, 0] 어느 strength도 어느 모델에서도 target을 못 만족하면 Phase 1 결과로 다음을 결정: (a) `ablate_upper_half` 대신 `ablate_upper_quarter`를 시도, (b) 다른 intervention class로 이동 (contrastive decoding, vision-token re-projection — E1d가 untested but plausible로 flag), (c) "≥ 5 pp" 느슨한 target을 받아들이고 문서화.

## 완료 시 roadmap update

- §6 Tier 1 E4 행: status가 `☐` → `✅`로, Phase 2 완료 시.
- §10 changelog 항목 완료 시점에 dated.
