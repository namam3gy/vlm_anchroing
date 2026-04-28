# E5b — anchor-distance robustness sweep: 설계

**Status:** 구현 전 설계. 사용자 승인 2026-04-27. E5(multi-dataset full runs)의 sub-experiment이며 E1 아래 E1b/c/d 명명 패턴과 같음. 실험 종료 후 결과 writeup인 `docs/experiments/E5b-anchor-distance.md`로 대체됨.

## 목표

cross-modal anchoring 강도가 anchor digit과 질문의 ground-truth 답 사이 절대 거리에 따라 어떻게 변하는지 정량화하고, 이 관계가 질문 형태는 같지만 이미지 도메인이 다른 두 dataset (TallyQA — natural-photo counting; VQAv2 number subset — open-domain numeric VQA) 에서 일관적인지 확인. 산출물은 dataset별 anchor effect vs distance bin figure와 cross-dataset overlay.

**의사결정 영향.** Distance 곡선이 paper 헤드라인 figure에서 stratified anchor sampling rule을 정당화 (또는 차단) 합니다. 효과가 distance에 따라 급격히 감쇠하면 헤드라인은 near-anchor subset에서 reporting. flat이라면 anchor selection이 load-bearing이 아니므로 paper 모든 section에 single random-anchor methodology면 충분.

## 이 설계를 택한 이유

reviewer는 anchor 효과 주장에 대해 두 가지로 공격합니다 — *(i) 효과가 선택된 특정 anchor digit의 feature 아니냐, (ii) scale이 바뀔 때(TallyQA가 VQAv2보다 큼) 살아남느냐*. 두 질문 모두 "anchor가 plausible answer 대비 어느 거리에 있느냐가 중요하냐"로 환원됩니다. E5b는 첫 질문은 stratified distance sweep으로 직접 답하고, 두 번째 질문은 cross-dataset 비교로 답합니다.

세 가지 후보 design 검토:

1. **0..10000 anchor 인벤토리에서 uniform random sampling.** 거부 — 인벤토리의 78%가 typical TallyQA/VQAv2 GT (≈ 0–3) 로부터 d > 30 거리 → 가장 강한 신호를 갖는 near-distance bin이 starved.
2. **Per-question 5-nearest-to-prediction selection.** 거부 — 모든 데이터를 d ≤ 5 영역에 집중시켜 곡선을 그릴 수 없게 만들고, 종속변수를 직접 최적화하는 형태. reviewer-hostile.
3. **Per-question 5-stratum anchor sampling, stratum당 random 1개.** 채택 — stratum 내 random sampling은 유지("selection" 공격 방어 가능)하면서 모든 distance regime에 균등 coverage 보장 (효율적인 곡선 추정).

## 실험 설계

### Datasets와 N

| Dataset | Subset 규칙 | N (base questions) |
|---|---|---:|
| VQAv2 number val | `answer_range = 8`, `samples_per_answer = 100`, `max_samples = 500`, `require_single_numeric_gt = True`, GT ≤ 10000 | 500 |
| TallyQA test | `answer_type_filter = ["number"]`, `answer_range = 8`, `samples_per_answer = 100`, `max_samples = 500`, `require_single_numeric_gt = True`, GT ≤ 10000 | 500 |

`answer_range=8`은 GT class 0..8 (총 9 class) 유지 (`data.load_number_vqa_samples` line 77 filter). `samples_per_answer = 100`은 class당 base question 100개로 cap; `max_samples = 500`은 base question 500개에 도달하면 iteration 종료시키는 hard total cap. 결합 동작: dataset 순회 순서대로 class들이 채워지다 per-class cap (100) 또는 total cap (500) 이 먼저 트리거 → 보통 total cap이 먼저 → 실제 class 분포는 dataset 자연 순서를 반영. N = dataset당 정확히 500 base questions. N은 sample-instance가 아닌 **base question** 단위로 reporting: stratified 모드에서 각 base question은 anchor 5개 (stratum당 1개) + target_only baseline 1개 = 6 generation 기여. Subset 규칙 α (GT ≤ 10000) 는 사용자 결정대로 — 본 filter에서 no-op (TallyQA/VQAv2 모두 GT ≤ 8) 이지만, 동일 config가 GT 10000 초과 가능한 ChartQA/MathVista 로 E5b 확장 시 그대로 적용되도록 명시 기록.

### Anchor sampling — 5 strata, GT 기반 reference

각 sample-instance의 ground-truth answer `g`에 대해, 인벤토리 `inputs/irrelevant_number/{0..10000}.png` (128 PNG)에서 stratum별 1개씩 anchor 5개 `(a₁..a₅)` 독립 추출:

| Stratum | `|a − g|` 범위 | 역할 |
|---|---|---|
| **S1** | [0, 1] | near-peak (literature가 맞다면 효과 가장 강함) |
| **S2** | [2, 5] | 인접 mid |
| **S3** | [6, 30] | 중간 감쇠 |
| **S4** | [31, 300] | far |
| **S5** | [301, ∞) | very far / saturation tail |

구현: `vlm_anchor.data`에 새 pure function (`sample_stratified_anchors`) 추가. `(gt: int, inventory: list[int], rng: random.Random)` 받아서 stratified anchor 5개 반환. GT ∈ 0..8과 inventory `{0..10, 15, 20, ..., 100, 200, ..., 10000}` (128개) 조합에서, 본 실험 모든 GT × stratum 쌍에 매칭 ≥ 4개 보장 → 본 실험에서 empty-stratum branch는 logged-and-skipped로 두지만 실제 발화 안 함. 향후 ChartQA/MathVista 확장에서 발화할 가능성에 대비해 branch는 그대로 구현.

### Sample당 conditions

| Condition | Inputs | Anchor 값 |
|---|---|---|
| `target_only` | target image만 | — |
| `target_plus_irrelevant_number_S1` | target + anchor PNG | `a₁` |
| `target_plus_irrelevant_number_S2` | target + anchor PNG | `a₂` |
| `target_plus_irrelevant_number_S3` | target + anchor PNG | `a₃` |
| `target_plus_irrelevant_number_S4` | target + anchor PNG | `a₄` |
| `target_plus_irrelevant_number_S5` | target + anchor PNG | `a₅` |

**Neutral arm은 E5b에서 drop** (사용자 결정): S5 condition (very far anchor) 가 "anchor 정보 ≈ 0" reference 역할을 동일하게 하고, target_only가 이미 "두 번째 이미지 없음" baseline 제공.

총: **base question 당 6 generations** (1 target_only + anchor stratum 5개; stratum 간 target_only 중복 없음).

기존 pipeline에서는 `assign_irrelevant_images`가 `irrelevant_sets_per_sample`마다 sample-instance 1개를 만들고 `build_conditions`가 sample-instance당 3 condition을 yield → 같은 base question의 replica들에 걸쳐 `target_only`가 중복 실행됨. `stratified=True` 모드에서는 각 base question이 list-valued `anchor_strata` field (anchor 5개) 를 가진 sample-instance 1개로 매핑되고; `build_conditions`는 그 하나의 sample-instance에 대해 6 condition을 yield, YAML의 `irrelevant_sets_per_sample`은 무시.

### Model

단일 모델: **`llava-interleave-7b`** (`llava-hf/llava-interleave-qwen-7b-hf`).

근거:
- 17,730-record VQAv2 main-run baseline 이미 존재 → distance 곡선을 random-anchor adoption (0.134), direction-follow (0.348) 수치에 대해 sanity-check 가능.
- 7B-class panel에서 direction-follow 최고치 (0.348) → distance 곡선의 dynamic range 가장 큼.
- Multi-image native (interleave training) → 두 번째 이미지 anchor pathway가 모델의 설계 의도 그대로, stress mode 아님.
- 기존 TallyQA smoke run config에 이미 포함.

### Sampling

- `temperature = 0.0`, `top_p = 1.0`, `max_new_tokens = 8`. Greedy, JSON-only system prompt — `experiment.yaml`과 동일.
- `seed = 42`.

## Driver와 config 변경

### `src/vlm_anchor/data.py`

추가:

```python
def sample_stratified_anchors(
    gt: int,
    inventory: list[int],
    rng: random.Random,
    strata: list[tuple[int, int]] = ANCHOR_DISTANCE_STRATA,
) -> list[int | None]:
    """Stratum당 random anchor 1개 반환, 매칭 없으면 None."""
```

`ANCHOR_DISTANCE_STRATA` 상수 같은 module: `[(0, 1), (2, 5), (6, 30), (31, 300), (301, 10**9)]`.

`assign_irrelevant_images`에 `stratified=True` flag 추가. set이면 anchor PNG path를 `sample_stratified_anchors` 출력에서 resolve (sample당 anchor 5개, stratum당 1개) — 기존 uniform-random-from-pool path 대신. Number-image filename convention: `inputs/irrelevant_number/{anchor_value}.png`.

### `src/vlm_anchor/data.py` — `build_conditions`

5개 anchor 값이 있을 때 stratum당 1 condition씩 emit하도록 확장 — condition 이름 `target_plus_irrelevant_number_S{1..5}`. `stratified=True`일 때 neutral arm 생략. 기존 3-condition 동작 (target_only / neutral / number) 은 main run 호환성 위해 `stratified=False`에서 그대로 유지.

### Configs

`configs/experiment_tallyqa.yaml` 스타일을 따르는 새 file 2개:

- `configs/experiment_distance_vqa.yaml`
- `configs/experiment_distance_tally.yaml`

둘 다 `inputs.anchor_sampling: stratified`, `inputs.distance_strata: [[0,1],[2,5],[6,30],[31,300],[301,1000000000]]`, `inputs.irrelevant_neutral_dir: null` (neutral skip), `models: [llava-next-interleaved-7b]`, sampling block 동일. 차이는 `vqa_dataset.local_path` (`inputs/vqav2_number_val` vs `inputs/tallyqa_test`) 와 `samples_per_answer = 63`.

### Driver script — 신규 파일 없음

`scripts/run_experiment.py` 재사용. stratified flag는 YAML에서 읽고; per-condition emission과 record당 `anchor_stratum_id` field가 기존 pipeline 통과. `metrics.summarize_condition`은 condition 이름에 generic이라 per-stratum summary block 자동 생성.

## 분석과 산출물

### Notebook

`notebooks/E5b_anchor_distance.ipynb` — 합쳐진 6,000-record dataframe (500 samples × 6 conditions × 2 datasets) 에 대한 pandas 분석:

1. **Per-stratum summary table**: `direction_follow_rate`, `adoption_rate`, `exact_match`, `mean_distance_to_anchor` per (dataset, stratum). per-record bootstrap으로 95 % CI.
2. **Distance 곡선 figure**: x = stratum midpoint (1, 3.5, 18, 165, 650), y = direction-follow gap vs target_only baseline. 두 line (TallyQA, VQAv2) + 음영 CI.
3. **Cross-dataset overlay**: 같은 figure에 두 line을 한 axis에. d* point — S1 효과의 50 % 이상이 살아있는 가장 큰 stratum — 를 paper 헤드라인 subset 후보 cutoff로 표시.
4. **Sanity check**: VQAv2 S1+S2 pooled direction-follow vs 기존 17,730-record main-run direction-follow (0.348). anchor-distance 분포가 일치할 때만 pooled ≈ main; 아니면 expected mismatch로 기록.

### Writeups

- `docs/experiments/E5b-anchor-distance.md` (+ `_ko.md` mirror) — 결과 도착 후 full 실험 writeup. 이 design doc을 canonical E5b reference로 대체.
- `docs/insights/E5b-anchor-distance-evidence.md` (+ `_ko.md` mirror) — distilled one-claim insight.

### Roadmap

- 완료 시 `references/roadmap.md` §3 (status), §6 Tier 2 (E5b row 추가), §10 changelog 업데이트.
- Korean mirror도 lockstep으로 업데이트.

## 시간 추정

- Dataset당: 500 samples × 6 generations = 3,000 records.
- Dataset 2개: 총 6,000 records.
- llava-interleave-7b 2-image inference: empirical wall ≈ 0.5 sec/record (이전 panel 중간 — ChartQA 0.12, VQAv2 main ≈ 1.6 — interleave의 two-image inference는 그 사이).
- 예상 wall: **총 ~50분** (1 모델, 두 dataset 순차).

## Risks와 caveats

- **TallyQA stratum coverage** — TallyQA의 `answer_range=8`이 GT를 0..8로 truncate, VQAv2 number와 동일. 따라서 cross-dataset 비교는 **image-domain의 anchoring 효과**를 testing — GT-scale 효과 아님 (진짜 scale 비교에는 ChartQA/MathVista 필요, deferred). E5b 헤드라인은 "image-domain robustness"로 framing해야 — "scale robustness" 아님.
- **Anchor 인벤토리 mid-range gap** — [10, 100] 범위 anchor가 5 step (1 step 아님), GT가 10..100일 때 S1 안에서 d=1,2 해상도 손실. TallyQA/VQAv2는 GT가 거의 0..3이라 본 실험에 영향 없음, 미래 ChartQA/MathVista 확장 시 위해 logged.
- **S5 floor가 zero 아님** — S5는 "anchor very far"를 capture하지만 진짜 anchor-information-zero condition 아님. paper가 후에 strict zero-information control 필요하면 follow-up에서 neutral arm 재추가. 본 sanity check는 target_only가 진짜 zero-information baseline 역할 하고 record됨.
- **Single-model claim scope** — llava-interleave-7b 단독 결과로는 distance dependence가 일반화된다고 단정 못 함. roadmap 업데이트에서 "mid-stack cluster (LLaVA-1.5 / ConvLLaVA / InternVL3) 확장"을 자연스러운 follow-up으로 명시; cost ≈ 3 × 50분 = 2.5h 순차.
- **Driver back-compat** — `stratified=True` flag가 기존 어떤 config의 동작도 바꾸면 안 됨. smoke test (`uv run python -m pytest`) + `configs/experiment.yaml` 5-sample run이 launch 전 regression check.

## Out of scope (deferred)

- **기존 데이터에 대한 Phase B post-hoc bin** — 사용자 결정 2026-04-27: 분석 측 시간 절약이 paper에 두 가지 methodology를 적는 비용 대비 의미 없으므로, post-hoc bin 표 산출 안 함. 모든 헤드라인 figure는 기존 main run의 random-anchor 0..9 sampling 그대로 사용.
- **ChartQA/MathVista distance sweep** — anchor 인벤토리 cap=10000은 ChartQA GT ≤ 10000 이미 cover. E5b가 distance 의존성을 확인하면 follow-up E5c로 확장.
- **Pred-based reference point** — 사용자 결정으로 GT 기반 선택. Pred 기반은 (i) two-pass 실행, (ii) 모델별 anchor 분기로 item-paired 분석 깨짐, (iii) 재현성 불안정. Cognitive framing (uncertainty-modulated) 동기는 이미 Phase A의 A1 finding이 다른 stratification 축 (target_only correctness) 으로 cover.
