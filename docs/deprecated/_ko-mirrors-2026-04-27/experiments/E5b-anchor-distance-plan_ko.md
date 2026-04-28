# E5b Anchor-Distance Sweep 구현 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (권장) 또는 superpowers:executing-plans 로 task 단위 구현. 체크박스 (`- [ ]`) 로 진행 추적.

**목표:** stratified-by-distance anchor sampling 모드 추가하고 TallyQA + VQAv2 (각 500 base questions, llava-interleave-7b)에서 E5b 실행 — anchor 효과가 거리에 따라 감쇠하는지 검증.

**Architecture:** `vlm_anchor.data`에 pure helper 2개 추가 (`sample_stratified_anchors`, `assign_stratified_anchors`); `build_conditions`는 `anchor_strata` key dispatch — stratified sample은 6 condition (1 baseline + 5 strata) yield, 기존은 3. Driver는 YAML의 `inputs.anchor_sampling`로 분기, record당 `anchor_stratum_id` 기록. 새 driver script 없음 — 기존 `scripts/run_experiment.py` 확장.

**Tech Stack:** Python 3.10+, `vlm_anchor` 패키지, `unittest`/pytest, `uv`, HuggingFace `transformers` (이미 `vlm_anchor.models`로 wired). Project convention: `pathlib.Path`, public function 타입 힌트, snake_case modules, `references/AGENTS.md` 따라 영문 `.md` + `_ko.md` mirror.

**Spec:** `docs/experiments/E5b-anchor-distance-design.md`. 시작 전 읽기.

**Out of scope:** 기존 데이터 Phase B post-hoc bin; ChartQA/MathVista distance sweep; pred-기반 reference. Spec §"Out of scope" 참조.

---

## 파일 구조

| File | 동작 | 역할 |
|---|---|---|
| `src/vlm_anchor/data.py` | Modify | `ANCHOR_DISTANCE_STRATA`, `sample_stratified_anchors`, `assign_stratified_anchors` 추가; `build_conditions`에 `anchor_strata` dispatch 확장 |
| `tests/test_data.py` | Modify | `SampleStratifiedAnchorsTest`, `AssignStratifiedAnchorsTest`, `BuildConditionsStratifiedTest` 추가 |
| `scripts/run_experiment.py` | Modify | `cfg["inputs"]["anchor_sampling"]` 분기; record dict에 `anchor_stratum_id` 추가 |
| `configs/experiment_distance_vqa.yaml` | Create | VQAv2 stratified config |
| `configs/experiment_distance_tally.yaml` | Create | TallyQA stratified config |
| `references/roadmap.md` | Modify | §3.2 / §6 Tier 2에 E5b row 추가; §10 changelog |
| `references/roadmap_ko.md` | Modify | Korean mirror |

---

## Task 1: `ANCHOR_DISTANCE_STRATA` 상수 + `sample_stratified_anchors` 함수 추가

영문 plan의 Task 1과 동일. 코드 블록·테스트·커밋 메시지는 영문 plan 그대로 사용.

요지:
- pure function. `(gt: int, inventory: list[int], rng: random.Random)` 받아서 stratum당 1 anchor (없으면 None) 5개 반환.
- 5 stratum: `(0,1), (2,5), (6,30), (31,300), (301, 10**9)` — 양 끝 inclusive.
- Test: stratum 별 distance 정확성 / empty stratum None / seeded RNG 재현성.

---

## Task 2: `assign_stratified_anchors` 함수 추가

영문 plan의 Task 2 그대로.

요지:
- 각 sample에 `anchor_strata` field 부착 — 5개 dict 리스트 (`stratum_id`, `stratum_range`, `anchor_value`, `irrelevant_number_image`).
- `inputs/irrelevant_number/` PNG 파일명 (`{value}.png`)에서 inventory 구성.
- GT 정수 아니면 ValueError.
- Test: anchor_strata 구조 / 빈 stratum None / 빈 디렉토리 FileNotFoundError.

---

## Task 3: `build_conditions`를 `anchor_strata` dispatch 로 확장

영문 plan의 Task 3 그대로.

요지:
- `anchor_strata` key 있으면 1 baseline + 각 stratum 1 condition (None인 stratum은 skip).
- Condition 이름: `target_plus_irrelevant_number_S{1..5}`.
- `anchor_stratum_id` field 동시 부착.
- Legacy path (3-condition) 기존 동작 그대로 유지 — anchor_strata 없는 sample.
- Test: stratified 6 condition / None stratum skip / legacy 3-condition 보존.

---

## Task 4: `scripts/run_experiment.py`에 stratified 모드 wire-up

영문 plan의 Task 4 그대로.

요지:
- import에 `assign_stratified_anchors` 추가.
- `cfg["inputs"].get("anchor_sampling")`이 `"stratified"`면 새 함수 호출, 아니면 기존 `assign_irrelevant_images` 호출.
- Per-record dict에 `"anchor_stratum_id": cond.get("anchor_stratum_id")` 추가 (`anchor_value` 다음 줄).
- 기존 test 통과 확인.

---

## Task 5: `configs/experiment_distance_vqa.yaml` 생성

영문 plan의 Task 5 YAML 그대로 사용.

핵심 keys:
- `vqa_dataset.local_path: inputs/vqav2_number_val`
- `vqa_dataset.answer_range: 8`, `samples_per_answer: 100`, `max_samples: 500`
- `inputs.anchor_sampling: stratified`
- `models: [llava-next-interleaved-7b]`
- `irrelevant_neutral_dir`는 다른 config과 parity 위해 남겨두지만 stratified 모드에선 무시됨.

---

## Task 6: `configs/experiment_distance_tally.yaml` 생성

영문 plan의 Task 6 YAML 그대로.

VQAv2 config과 차이:
- `vqa_dataset.local_path: inputs/tallyqa_test`
- `vqa_dataset.answer_type_filter: ["number"]`

---

## Task 7: 두 config의 smoke validation

영문 plan의 Task 7 그대로.

요지:
- `--max-samples 3` 으로 smoke run × 2 (vqa + tally).
- 출력 record 18개 (3 × 6 condition) 확인. condition 분포 확인.
- 기존 `configs/experiment.yaml` 5-sample run으로 legacy path 무사한지 확인 (15 record = 5 × 3).
- Smoke 디렉토리 정리 (`outputs/`는 gitignored).

---

## Task 8: Roadmap 업데이트

영문 plan의 Task 8 그대로.

요지:
- `references/roadmap.md` §6 Tier 2에 E5b row 추가 (E5 row 다음).
- `references/roadmap.md` §10 changelog에 2026-04-27 entry 추가.
- `references/roadmap_ko.md`에 동일 위치 Korean translation 적용.
- Commit.

---

## Task 9: E5b 두 dataset에 대해 end-to-end 실행

영문 plan의 Task 9 그대로.

요지:
- VQAv2 distance config full 500 (~25분)
- TallyQA distance config full 500 (~25분)
- 각 3,000 records 산출 검증 (500 base × 6 conditions)
- `outputs/`는 gitignored — commit 없음. 두 timestamp dir 이름 기록 (Task 10에서 필요).

## Task 10: 분석 script + 재현 가능 notebook (executed outputs 포함)

영문 plan의 Task 10 그대로. 코드 블록은 영문 그대로 사용.

요지:
- `scripts/analyze_e5b_distance.py` — heavy lifting (load → per-stratum stats with bootstrap CI → plots).
- `scripts/build_e5b_notebook.py` — nbformat으로 notebook 구성.
- `notebooks/E5b_anchor_distance.ipynb` — thin caller, 기존 E1b notebook 패턴.
- 산출물: per_stratum CSV (`docs/insights/_data/`), figure 2개 (`docs/figures/`), executed notebook.
- `jupyter nbconvert --execute --inplace`로 cell outputs 박아넣음.
- 모든 code cell에 outputs 박혀있는지 검증.

## 최종 검증

- [ ] **Step 1: Test suite green 확인**

Run: `uv run python -m pytest -v`
Expected: 모든 test 통과 (기존 + Task 1–3에서 추가한 8개).

- [ ] **Step 2: Commit log 깨끗한지 확인**

Run: `git log --oneline -15`
Expected: 새 commit 9개 (Task 1–6 source/config, Task 8 roadmap, Task 10 analysis+notebook). Task 7 (smoke), Task 9 (full run)은 output-only — commit 없음. `git status` clean.

- [ ] **Step 3: 사용자에게 hand off**

구현 + 실행 + 재현 notebook 모두 완료. 사용자에게 반환:
- 추가된 commit 참조
- Executed notebook 경로 (`notebooks/E5b_anchor_distance.ipynb`)
- summary CSV의 headline 수치 (dataset별 peak direction-follow + 선택된 `d*` band)
- 다음 단계 제안: notebook 검토 후 결과 writeup `docs/experiments/E5b-anchor-distance.md` (+ `_ko.md`) + distilled insight `docs/insights/E5b-anchor-distance-evidence.md` (+ `_ko.md`) 작성.
