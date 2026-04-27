# E5b Anchor-Distance Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stratified-by-distance anchor sampling mode and run E5b on TallyQA + VQAv2 (500 base questions each, llava-interleave-7b) to test whether anchor effect decays with distance.

**Architecture:** Two new pure helpers in `vlm_anchor.data` (`sample_stratified_anchors`, `assign_stratified_anchors`); `build_conditions` dispatches on the `anchor_strata` key so a stratified sample yields 6 conditions (1 baseline + 5 strata) instead of the existing 3; the driver branches on `inputs.anchor_sampling` in YAML and records `anchor_stratum_id` per record. No new driver script; existing `scripts/run_experiment.py` is extended.

**Tech Stack:** Python 3.10+, `vlm_anchor` package, `unittest`/pytest, `uv` for env, HuggingFace `transformers` (already wired via `vlm_anchor.models`). Project convention: `pathlib.Path` over strings, type hints on public functions, snake_case modules, English `.md` + `_ko.md` mirror per `references/AGENTS.md`.

**Spec:** `docs/experiments/E5b-anchor-distance-design.md`. Read it before starting.

**Out of scope:** Phase B post-hoc bin on existing data; ChartQA/MathVista distance sweep; pred-based reference point. See spec §"Out of scope".

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/vlm_anchor/data.py` | Modify | Add `ANCHOR_DISTANCE_STRATA`, `sample_stratified_anchors`, `assign_stratified_anchors`; extend `build_conditions` to dispatch on `anchor_strata` |
| `tests/test_data.py` | Modify | Add `SampleStratifiedAnchorsTest`, `AssignStratifiedAnchorsTest`, `BuildConditionsStratifiedTest` |
| `scripts/run_experiment.py` | Modify | Branch on `cfg["inputs"]["anchor_sampling"]`; add `anchor_stratum_id` to per-record dict |
| `configs/experiment_distance_vqa.yaml` | Create | VQAv2 stratified config |
| `configs/experiment_distance_tally.yaml` | Create | TallyQA stratified config |
| `references/roadmap.md` | Modify | Add E5b row to §3.2 / §6 Tier 2; add §10 changelog entry |
| `references/roadmap_ko.md` | Modify | Korean mirror update |

---

## Task 1: Add `ANCHOR_DISTANCE_STRATA` constant + `sample_stratified_anchors` function

**Files:**
- Modify: `src/vlm_anchor/data.py` (append near top after imports)
- Test: `tests/test_data.py` (append new test class)

- [ ] **Step 1: Write the failing test**

Open `tests/test_data.py`. Add `import random` near the existing stdlib imports and extend the existing `from vlm_anchor.data import ...` line to also import `ANCHOR_DISTANCE_STRATA, sample_stratified_anchors`. The resulting top-of-file imports should look like:

```python
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from vlm_anchor.data import (
    ANCHOR_DISTANCE_STRATA,
    assign_irrelevant_images,
    load_number_vqa_samples,
    sample_stratified_anchors,
)
```

Then append the new test class to the end of the file (after the existing test classes, before any `if __name__ == "__main__":` guard if present):

```python
class SampleStratifiedAnchorsTest(unittest.TestCase):
    def test_returns_one_anchor_per_stratum_in_correct_distance_band(self) -> None:
        gt = 3
        inventory = list(range(0, 11)) + [15, 20, 25, 50, 100, 200, 500, 1000, 5000, 10000]
        rng = random.Random(42)

        anchors = sample_stratified_anchors(gt, inventory, rng)

        self.assertEqual(len(anchors), len(ANCHOR_DISTANCE_STRATA))
        for (lo, hi), value in zip(ANCHOR_DISTANCE_STRATA, anchors):
            self.assertIsNotNone(value, f"stratum [{lo},{hi}] yielded None unexpectedly")
            self.assertTrue(lo <= abs(value - gt) <= hi)

    def test_returns_none_when_stratum_has_no_inventory_match(self) -> None:
        gt = 3
        inventory = [3, 4, 5]
        rng = random.Random(0)

        anchors = sample_stratified_anchors(gt, inventory, rng)

        self.assertEqual(anchors[0], 3 if anchors[0] in (3, 4) else 4)
        self.assertIsNone(anchors[2])
        self.assertIsNone(anchors[3])
        self.assertIsNone(anchors[4])

    def test_seeded_rng_is_reproducible(self) -> None:
        gt = 5
        inventory = list(range(0, 101))
        a = sample_stratified_anchors(gt, inventory, random.Random(1234))
        b = sample_stratified_anchors(gt, inventory, random.Random(1234))
        self.assertEqual(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_data.py::SampleStratifiedAnchorsTest -v`
Expected: ImportError or AttributeError on `ANCHOR_DISTANCE_STRATA` / `sample_stratified_anchors`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/vlm_anchor/data.py` after the `_select_image_variants` function:

```python
ANCHOR_DISTANCE_STRATA: list[tuple[int, int]] = [
    (0, 1),
    (2, 5),
    (6, 30),
    (31, 300),
    (301, 10**9),
]


def sample_stratified_anchors(
    gt: int,
    inventory: list[int],
    rng: random.Random,
    strata: list[tuple[int, int]] = ANCHOR_DISTANCE_STRATA,
) -> list[int | None]:
    """Return one randomly-chosen anchor per stratum, keyed off |a - gt|.

    Returns None for strata with no inventory match. Strata are matched
    on absolute distance from `gt`; bounds are inclusive on both sides.
    """
    out: list[int | None] = []
    for lo, hi in strata:
        candidates = [a for a in inventory if lo <= abs(a - gt) <= hi]
        out.append(rng.choice(candidates) if candidates else None)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_data.py::SampleStratifiedAnchorsTest -v`
Expected: 3 passes.

- [ ] **Step 5: Commit**

```bash
git add src/vlm_anchor/data.py tests/test_data.py
git commit -m "Add ANCHOR_DISTANCE_STRATA + sample_stratified_anchors primitive"
```

---

## Task 2: Add `assign_stratified_anchors` function

**Files:**
- Modify: `src/vlm_anchor/data.py` (append after existing `assign_irrelevant_images`)
- Test: `tests/test_data.py` (append new test class)

- [ ] **Step 1: Write the failing test**

Extend the existing `from vlm_anchor.data import (...)` block in `tests/test_data.py` to also include `assign_stratified_anchors`. Then append the new test class to the end of the file:

```python
class AssignStratifiedAnchorsTest(unittest.TestCase):
    def test_attaches_anchor_strata_field_with_one_entry_per_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            number_dir = root / "irrelevant_number"
            number_dir.mkdir()
            for v in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 100, 500, 1000]:
                (number_dir / f"{v}.png").write_bytes(b"png")

            samples = [
                {
                    "question_id": 101,
                    "image_id": 202,
                    "question": "How many?",
                    "image": root / "target.png",
                    "ground_truth": "3",
                    "answers": ["3"] * 10,
                    "question_type": "how many",
                }
            ]

            enriched = assign_stratified_anchors(
                samples,
                irrelevant_number_dir=number_dir,
                seed=42,
            )

            self.assertEqual(len(enriched), 1)
            row = enriched[0]
            self.assertIn("anchor_strata", row)
            self.assertEqual(len(row["anchor_strata"]), 5)

            for entry in row["anchor_strata"]:
                self.assertIn("stratum_id", entry)
                self.assertIn("stratum_range", entry)
                self.assertIn("anchor_value", entry)
                self.assertIn("irrelevant_number_image", entry)
                if entry["anchor_value"] is not None:
                    self.assertTrue(Path(entry["irrelevant_number_image"]).exists())

            stratum_ids = [e["stratum_id"] for e in row["anchor_strata"]]
            self.assertEqual(stratum_ids, ["S1", "S2", "S3", "S4", "S5"])

    def test_skips_strata_with_no_inventory_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            number_dir = root / "irrelevant_number"
            number_dir.mkdir()
            for v in [3, 4, 5]:
                (number_dir / f"{v}.png").write_bytes(b"png")

            samples = [{
                "question_id": 1, "image_id": 1,
                "question": "Q?", "image": root / "t.png",
                "ground_truth": "3", "answers": ["3"], "question_type": "",
            }]

            enriched = assign_stratified_anchors(samples, number_dir, seed=0)
            entries_by_id = {e["stratum_id"]: e for e in enriched[0]["anchor_strata"]}
            self.assertIsNotNone(entries_by_id["S1"]["anchor_value"])
            self.assertIsNone(entries_by_id["S3"]["anchor_value"])
            self.assertIsNone(entries_by_id["S5"]["anchor_value"])

    def test_raises_when_inventory_directory_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            number_dir = Path(tmpdir) / "irrelevant_number"
            number_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                assign_stratified_anchors([], number_dir, seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_data.py::AssignStratifiedAnchorsTest -v`
Expected: ImportError on `assign_stratified_anchors`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/vlm_anchor/data.py` after `assign_irrelevant_images`:

```python
def assign_stratified_anchors(
    samples: list[dict],
    irrelevant_number_dir: str | Path,
    seed: int = 42,
    strata: list[tuple[int, int]] = ANCHOR_DISTANCE_STRATA,
) -> list[dict]:
    """Per-question stratified anchor sampling.

    Each output sample carries an `anchor_strata` field: a list of dicts
    in the order of `strata`, each containing `stratum_id` ("S1".."Sn"),
    `stratum_range` (the (lo, hi) tuple), `anchor_value` (int or None),
    and `irrelevant_number_image` (str path or None).
    """
    rng = random.Random(seed)
    number_images = list_images(irrelevant_number_dir)
    if not number_images:
        raise FileNotFoundError(f"No number images found in {irrelevant_number_dir}")

    inventory_by_value: dict[int, Path] = {}
    for img in number_images:
        v = extract_first_number(img.stem)
        if v and v.lstrip("-").isdigit():
            inventory_by_value[int(v)] = img

    inventory_values = sorted(inventory_by_value.keys())

    enriched: list[dict] = []
    for sample in samples:
        gt_str = sample["ground_truth"]
        if not gt_str.lstrip("-").isdigit():
            raise ValueError(f"sample {sample.get('question_id')} has non-integer ground_truth {gt_str!r}; stratified mode requires integer GT")
        gt = int(gt_str)
        anchor_values = sample_stratified_anchors(gt, inventory_values, rng, strata=strata)
        anchor_strata: list[dict] = []
        for idx, ((lo, hi), value) in enumerate(zip(strata, anchor_values), start=1):
            entry = {
                "stratum_id": f"S{idx}",
                "stratum_range": (lo, hi),
                "anchor_value": value,
                "irrelevant_number_image": str(inventory_by_value[value]) if value is not None else None,
            }
            anchor_strata.append(entry)
        enriched.append({
            **sample,
            "sample_instance_id": f"{sample['question_id']}_{sample['image_id']}_stratified",
            "sample_instance_index": 0,
            "anchor_strata": anchor_strata,
        })
    return enriched
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_data.py::AssignStratifiedAnchorsTest -v`
Expected: 3 passes.

- [ ] **Step 5: Commit**

```bash
git add src/vlm_anchor/data.py tests/test_data.py
git commit -m "Add assign_stratified_anchors for per-question 5-stratum sampling"
```

---

## Task 3: Extend `build_conditions` to dispatch on `anchor_strata`

**Files:**
- Modify: `src/vlm_anchor/data.py` (replace `build_conditions` body)
- Test: `tests/test_data.py` (append new test class)

- [ ] **Step 1: Write the failing test**

Extend the existing `from vlm_anchor.data import (...)` block in `tests/test_data.py` to also include `build_conditions`. Then append the new test class to the end of the file:

```python
class BuildConditionsStratifiedTest(unittest.TestCase):
    def test_stratified_sample_yields_one_baseline_plus_one_per_stratum(self) -> None:
        sample = {
            "question_id": 1, "image_id": 1,
            "question": "Q?", "image": Path("/tmp/t.png"),
            "ground_truth": "3", "answers": ["3"], "question_type": "",
            "sample_instance_id": "1_1_stratified", "sample_instance_index": 0,
            "anchor_strata": [
                {"stratum_id": "S1", "stratum_range": (0, 1),   "anchor_value": 3, "irrelevant_number_image": "/tmp/3.png"},
                {"stratum_id": "S2", "stratum_range": (2, 5),   "anchor_value": 7, "irrelevant_number_image": "/tmp/7.png"},
                {"stratum_id": "S3", "stratum_range": (6, 30),  "anchor_value": 20, "irrelevant_number_image": "/tmp/20.png"},
                {"stratum_id": "S4", "stratum_range": (31, 300),"anchor_value": 100,"irrelevant_number_image": "/tmp/100.png"},
                {"stratum_id": "S5", "stratum_range": (301, 10**9),"anchor_value": 1000,"irrelevant_number_image": "/tmp/1000.png"},
            ],
        }

        conds = list(build_conditions(sample))

        self.assertEqual(len(conds), 6)
        self.assertEqual(conds[0]["condition"], "target_only")
        self.assertEqual(conds[0]["irrelevant_type"], "none")
        self.assertIsNone(conds[0]["anchor_value_for_metrics"])

        for i, sid in enumerate(["S1", "S2", "S3", "S4", "S5"], start=1):
            cond = conds[i]
            self.assertEqual(cond["condition"], f"target_plus_irrelevant_number_{sid}")
            self.assertEqual(cond["anchor_stratum_id"], sid)
            self.assertEqual(cond["irrelevant_type"], "number")
            self.assertEqual(len(cond["input_images"]), 2)

    def test_stratified_skips_strata_with_none_anchor(self) -> None:
        sample = {
            "question_id": 1, "image_id": 1,
            "question": "Q?", "image": Path("/tmp/t.png"),
            "ground_truth": "3", "answers": ["3"], "question_type": "",
            "anchor_strata": [
                {"stratum_id": "S1", "stratum_range": (0, 1), "anchor_value": 3, "irrelevant_number_image": "/tmp/3.png"},
                {"stratum_id": "S2", "stratum_range": (2, 5), "anchor_value": None, "irrelevant_number_image": None},
                {"stratum_id": "S3", "stratum_range": (6, 30), "anchor_value": 20, "irrelevant_number_image": "/tmp/20.png"},
            ],
        }
        conds = list(build_conditions(sample))
        condition_names = [c["condition"] for c in conds]
        self.assertIn("target_only", condition_names)
        self.assertIn("target_plus_irrelevant_number_S1", condition_names)
        self.assertNotIn("target_plus_irrelevant_number_S2", condition_names)
        self.assertIn("target_plus_irrelevant_number_S3", condition_names)

    def test_legacy_sample_without_anchor_strata_yields_three_conditions(self) -> None:
        sample = {
            "question_id": 1, "image_id": 1,
            "question": "Q?", "image": Path("/tmp/t.png"),
            "ground_truth": "3", "answers": ["3"], "question_type": "",
            "irrelevant_number_image": "/tmp/3.png",
            "irrelevant_neutral_image": "/tmp/n.png",
            "anchor_value": "3",
        }
        conds = list(build_conditions(sample))
        self.assertEqual(len(conds), 3)
        self.assertEqual([c["condition"] for c in conds], [
            "target_only", "target_plus_irrelevant_number", "target_plus_irrelevant_neutral",
        ])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_data.py::BuildConditionsStratifiedTest -v`
Expected: 2 failures (stratified tests fail) + 1 pass (legacy test).

- [ ] **Step 3: Replace `build_conditions` in `src/vlm_anchor/data.py`**

Locate the existing `def build_conditions(sample: dict) -> Iterator[dict]:` block and replace with:

```python
def build_conditions(sample: dict) -> Iterator[dict]:
    if "anchor_strata" in sample:
        yield {
            **sample,
            "condition": "target_only",
            "input_images": [sample["image"]],
            "anchor_value_for_metrics": None,
            "irrelevant_type": "none",
            "irrelevant_image": None,
            "anchor_stratum_id": None,
        }
        for entry in sample["anchor_strata"]:
            if entry["anchor_value"] is None:
                continue
            yield {
                **sample,
                "condition": f"target_plus_irrelevant_number_{entry['stratum_id']}",
                "input_images": [sample["image"], entry["irrelevant_number_image"]],
                "anchor_value_for_metrics": str(entry["anchor_value"]),
                "irrelevant_type": "number",
                "irrelevant_image": entry["irrelevant_number_image"],
                "anchor_stratum_id": entry["stratum_id"],
            }
        return

    yield {
        **sample,
        "condition": "target_only",
        "input_images": [sample["image"]],
        "anchor_value_for_metrics": None,
        "irrelevant_type": "none",
        "irrelevant_image": None,
    }
    yield {
        **sample,
        "condition": "target_plus_irrelevant_number",
        "input_images": [sample["image"], sample["irrelevant_number_image"]],
        "anchor_value_for_metrics": sample["anchor_value"],
        "irrelevant_type": "number",
        "irrelevant_image": sample["irrelevant_number_image"],
    }
    yield {
        **sample,
        "condition": "target_plus_irrelevant_neutral",
        "input_images": [sample["image"], sample["irrelevant_neutral_image"]],
        "anchor_value_for_metrics": None,
        "irrelevant_type": "neutral",
        "irrelevant_image": sample["irrelevant_neutral_image"],
    }
```

- [ ] **Step 4: Run all data tests to verify legacy unchanged + stratified passes**

Run: `uv run python -m pytest tests/test_data.py -v`
Expected: all data tests pass (existing + new).

- [ ] **Step 5: Commit**

```bash
git add src/vlm_anchor/data.py tests/test_data.py
git commit -m "Extend build_conditions to dispatch on anchor_strata key"
```

---

## Task 4: Wire stratified mode through `scripts/run_experiment.py`

**Files:**
- Modify: `scripts/run_experiment.py`

- [ ] **Step 1: Replace the assignment block in `main()`**

Locate the call to `assign_irrelevant_images(...)` in `scripts/run_experiment.py` (currently lines around the data-loading region) and replace the contiguous block:

```python
    samples = assign_irrelevant_images(
        samples,
        irrelevant_number_dir=resolve_path(cfg["inputs"]["irrelevant_number_dir"], base_dir=project_root),
        irrelevant_neutral_dir=resolve_path(cfg["inputs"]["irrelevant_neutral_dir"], base_dir=project_root),
        seed=cfg["seed"],
        variants_per_sample=int(cfg["inputs"].get("irrelevant_sets_per_sample", 1)),
    )
```

with:

```python
    anchor_sampling = cfg["inputs"].get("anchor_sampling", "uniform")
    if anchor_sampling == "stratified":
        samples = assign_stratified_anchors(
            samples,
            irrelevant_number_dir=resolve_path(cfg["inputs"]["irrelevant_number_dir"], base_dir=project_root),
            seed=cfg["seed"],
        )
    else:
        samples = assign_irrelevant_images(
            samples,
            irrelevant_number_dir=resolve_path(cfg["inputs"]["irrelevant_number_dir"], base_dir=project_root),
            irrelevant_neutral_dir=resolve_path(cfg["inputs"]["irrelevant_neutral_dir"], base_dir=project_root),
            seed=cfg["seed"],
            variants_per_sample=int(cfg["inputs"].get("irrelevant_sets_per_sample", 1)),
        )
```

- [ ] **Step 2: Update the `from vlm_anchor.data import ...` line**

Replace:
```python
from vlm_anchor.data import assign_irrelevant_images, build_conditions, load_number_vqa_samples
```
with:
```python
from vlm_anchor.data import (
    assign_irrelevant_images,
    assign_stratified_anchors,
    build_conditions,
    load_number_vqa_samples,
)
```

- [ ] **Step 3: Add `anchor_stratum_id` to the per-record dict**

In the per-condition record-building section (the `row = { ... "model": ..., ...}` dict), add the line `"anchor_stratum_id": cond.get("anchor_stratum_id"),` between `"anchor_value": cond["anchor_value_for_metrics"],` and `"standard_vqa_accuracy": ...`.

The resulting dict slice should read:

```python
                    "anchor_value": cond["anchor_value_for_metrics"],
                    "anchor_stratum_id": cond.get("anchor_stratum_id"),
                    "standard_vqa_accuracy": sample_eval.standard_vqa_accuracy,
```

- [ ] **Step 4: Run the existing test suite to confirm no regression**

Run: `uv run python -m pytest`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_experiment.py
git commit -m "Wire stratified anchor mode through driver + record anchor_stratum_id"
```

---

## Task 5: Create `configs/experiment_distance_vqa.yaml`

**Files:**
- Create: `configs/experiment_distance_vqa.yaml`

- [ ] **Step 1: Create the file**

Write `configs/experiment_distance_vqa.yaml`:

```yaml
seed: 42
output_root: outputs

vqa_dataset:
  local_path: inputs/vqav2_number_val
  answer_range: 8
  samples_per_answer: 100
  max_samples: 500
  require_single_numeric_gt: true

inputs:
  irrelevant_number_dir: inputs/irrelevant_number
  irrelevant_neutral_dir: inputs/irrelevant_neutral
  anchor_sampling: stratified

models:
  - name: llava-next-interleaved-7b
    hf_model: llava-hf/llava-interleave-qwen-7b-hf

sampling:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 8

prompt:
  system: |
    You are a visual question answering system.
    Return valid JSON only in the form {"result": <number>}.
    Use a numeric JSON value for <number>, not a string.
    Do not output any other keys, words, explanation, or markdown.
    If uncertain, still output the single most likely number in that JSON format.
  user_template: |
    Answer the question using the provided image(s).
    Return JSON only in the form {"result": <number>}.
    Question: {question}
```

> Note: `irrelevant_neutral_dir` is kept in the YAML for parity with other configs but is unused in stratified mode. `irrelevant_sets_per_sample` is omitted because it's ignored by the stratified path.

- [ ] **Step 2: Commit**

```bash
git add configs/experiment_distance_vqa.yaml
git commit -m "Add VQAv2 distance-sweep config (stratified anchor mode)"
```

---

## Task 6: Create `configs/experiment_distance_tally.yaml`

**Files:**
- Create: `configs/experiment_distance_tally.yaml`

- [ ] **Step 1: Create the file**

Write `configs/experiment_distance_tally.yaml`:

```yaml
seed: 42
output_root: outputs

vqa_dataset:
  local_path: inputs/tallyqa_test
  answer_type_filter: ["number"]
  answer_range: 8
  samples_per_answer: 100
  max_samples: 500
  require_single_numeric_gt: true

inputs:
  irrelevant_number_dir: inputs/irrelevant_number
  irrelevant_neutral_dir: inputs/irrelevant_neutral
  anchor_sampling: stratified

models:
  - name: llava-next-interleaved-7b
    hf_model: llava-hf/llava-interleave-qwen-7b-hf

sampling:
  temperature: 0.0
  top_p: 1.0
  max_new_tokens: 8

prompt:
  system: |
    You are a visual question answering system.
    Return valid JSON only in the form {"result": <number>}.
    Use a numeric JSON value for <number>, not a string.
    Do not output any other keys, words, explanation, or markdown.
    If uncertain, still output the single most likely number in that JSON format.
  user_template: |
    Answer the question using the provided image(s).
    Return JSON only in the form {"result": <number>}.
    Question: {question}
```

- [ ] **Step 2: Commit**

```bash
git add configs/experiment_distance_tally.yaml
git commit -m "Add TallyQA distance-sweep config (stratified anchor mode)"
```

---

## Task 7: Smoke validation on both configs

**Files:**
- (no source changes; runs the pipeline end-to-end)

- [ ] **Step 1: Smoke run on VQAv2 distance config (3 samples)**

Run: `uv run python scripts/run_experiment.py --config configs/experiment_distance_vqa.yaml --max-samples 3`

Expected outputs under `outputs/experiment_distance_vqa/llava-next-interleaved-7b/<timestamp>/`:
- `predictions.jsonl` with **18 records** (3 samples × 6 conditions)
- `summary.json` with keys for all 6 conditions: `target_only`, `target_plus_irrelevant_number_S1`..`_S5`

- [ ] **Step 2: Verify the smoke output**

Run:
```bash
ls -la outputs/experiment_distance_vqa/llava-next-interleaved-7b/
SMOKE_DIR=$(ls -t outputs/experiment_distance_vqa/llava-next-interleaved-7b/ | head -1)
wc -l outputs/experiment_distance_vqa/llava-next-interleaved-7b/$SMOKE_DIR/predictions.jsonl
python -c "
import json
from collections import Counter
recs = [json.loads(l) for l in open('outputs/experiment_distance_vqa/llava-next-interleaved-7b/$SMOKE_DIR/predictions.jsonl')]
print('records:', len(recs))
print('conditions:', Counter(r['condition'] for r in recs))
print('stratum_ids:', Counter(r.get('anchor_stratum_id') for r in recs))
"
```

Expected:
- `records: 18`
- `conditions: Counter({'target_only': 3, 'target_plus_irrelevant_number_S1': 3, ..._S5: 3})`
- `stratum_ids: Counter({None: 3, 'S1': 3, 'S2': 3, 'S3': 3, 'S4': 3, 'S5': 3})`

- [ ] **Step 3: Smoke run on TallyQA distance config (3 samples)**

Run: `uv run python scripts/run_experiment.py --config configs/experiment_distance_tally.yaml --max-samples 3`

Same verification as Step 2 with `experiment_distance_tally`.

- [ ] **Step 4: Confirm legacy configs still work**

Run: `uv run python scripts/run_experiment.py --config configs/experiment.yaml --max-samples 5 --models qwen2.5-vl-7b-instruct`

Expected: existing 3-condition behaviour unchanged; predictions.jsonl has 15 records (5 × 3).

- [ ] **Step 5: Commit any fixes; otherwise just delete smoke outputs**

If any step required fixes, stage and commit. Then clean up the smoke directories:

```bash
rm -rf outputs/experiment_distance_vqa outputs/experiment_distance_tally
```

(`outputs/` is gitignored — no commit needed for cleanup.)

---

## Task 8: Roadmap update

**Files:**
- Modify: `references/roadmap.md`
- Modify: `references/roadmap_ko.md`

- [ ] **Step 1: Add E5b row to §6 Tier 2 in `references/roadmap.md`**

Locate the `### Tier 2 (paper hardening)` table. Add a new row below the existing `**E5**` row:

```markdown
| **E5b** | **Anchor-distance robustness sweep.** Per-question 5-stratum anchor sampling on TallyQA + VQAv2 (500 base questions each, llava-interleave-7b). Validates whether anchor selection rule is load-bearing for paper headline figures. | ~50 min wall (1 model). | One figure (effect-vs-distance) + cross-dataset overlay. Decides whether headline figures should report on a near-anchor subset or remain full-set. | ☐ |
```

- [ ] **Step 2: Add changelog entry to §10**

Append to the bottom of the `## 10. Changelog` section in `references/roadmap.md`:

```markdown
- **2026-04-27** — **E5b design + plan committed.** Anchor-distance robustness sweep added as new sub-experiment of E5. Stratified anchor sampling (5 strata by `|a − GT|`: [0,1] / [2,5] / [6,30] / [31,300] / [301,∞)), 500 base questions per dataset on TallyQA + VQAv2, llava-interleave-7b only, ~50 min wall. Goal: justify (or block) the use of a near-anchor subset for paper headline figures. New driver path keyed off `inputs.anchor_sampling: stratified` in YAML; legacy 3-condition path untouched. Specs: `docs/experiments/E5b-anchor-distance-design.md` (+ _ko mirror), plan: `docs/experiments/E5b-anchor-distance-plan.md` (+ _ko mirror).
```

- [ ] **Step 3: Mirror both edits in `references/roadmap_ko.md`**

Apply the equivalent Korean translations to the matching §6 Tier 2 row and §10 changelog section in `references/roadmap_ko.md`. Maintain the same structural placement.

Korean §6 row template:
```markdown
| **E5b** | **Anchor-distance robustness sweep.** TallyQA + VQAv2에서 per-question 5-stratum anchor sampling (dataset당 500 base question, llava-interleave-7b). Anchor selection 규칙이 paper 헤드라인 figure에 load-bearing인지 검증. | ~50 min wall (1 모델). | 곡선 figure 1개 (effect-vs-distance) + cross-dataset overlay. 헤드라인 figure가 near-anchor subset에서 reporting할지 full-set에서 유지할지 결정. | ☐ |
```

Korean §10 entry template:
```markdown
- **2026-04-27** — **E5b 설계 + plan commit.** Anchor-distance robustness sweep을 E5의 새 sub-experiment으로 추가. Stratified anchor sampling (5 strata by `|a − GT|`: [0,1] / [2,5] / [6,30] / [31,300] / [301,∞)), TallyQA + VQAv2 dataset당 500 base questions, llava-interleave-7b 단독, ~50분 wall. 목표: paper 헤드라인 figure에서 near-anchor subset 사용을 정당화 (또는 차단). 새 driver path는 YAML의 `inputs.anchor_sampling: stratified`로 진입; legacy 3-condition path 그대로 유지. Specs: `docs/experiments/E5b-anchor-distance-design.md` (+ _ko mirror), plan: `docs/experiments/E5b-anchor-distance-plan.md` (+ _ko mirror).
```

- [ ] **Step 4: Commit roadmap updates**

```bash
git add references/roadmap.md references/roadmap_ko.md
git commit -m "Roadmap: add E5b anchor-distance sweep to §6 Tier 2 + §10 changelog"
```

---

## Task 9: Run E5b end-to-end on both datasets

**Files:**
- (no source changes; runs the pipeline on real data)

This is the heavy step (~50 min total wall). Run it on a free GPU. Outputs land in `outputs/experiment_distance_*/llava-next-interleaved-7b/<timestamp>/`.

- [ ] **Step 1: Run VQAv2 distance config (~25 min wall)**

Run: `uv run python scripts/run_experiment.py --config configs/experiment_distance_vqa.yaml`

Expected:
- Output dir: `outputs/experiment_distance_vqa/llava-next-interleaved-7b/<timestamp>/`
- `predictions.jsonl`: 3,000 records (500 base questions × 6 conditions)
- `summary.json`: 6 condition keys (`target_only`, `target_plus_irrelevant_number_S{1..5}`)
- Wall: ~25 min (~0.5 sec/rec on llava-interleave 2-image inference)

- [ ] **Step 2: Run TallyQA distance config (~25 min wall)**

Run: `uv run python scripts/run_experiment.py --config configs/experiment_distance_tally.yaml`

Expected (same shape as Step 1 but under `outputs/experiment_distance_tally/...`).

- [ ] **Step 3: Verify both runs landed**

Run:
```bash
ls -la outputs/experiment_distance_vqa/llava-next-interleaved-7b/
ls -la outputs/experiment_distance_tally/llava-next-interleaved-7b/
python -c "
import json, pathlib
for ds in ['experiment_distance_vqa', 'experiment_distance_tally']:
    runs = sorted(pathlib.Path(f'outputs/{ds}/llava-next-interleaved-7b').iterdir())
    latest = runs[-1]
    recs = [json.loads(l) for l in open(latest/'predictions.jsonl')]
    print(f'{ds}: n={len(recs)}, conditions={sorted(set(r[\"condition\"] for r in recs))}')
"
```

Expected: each dataset shows `n=3000` and 6 conditions.

(No commit — `outputs/` is gitignored. Record the two timestamp dir names; they'll be needed in Task 10.)

---

## Task 10: Build analysis script + reproducible notebook (with executed outputs)

**Files:**
- Create: `scripts/analyze_e5b_distance.py`
- Create: `notebooks/E5b_anchor_distance.ipynb`
- Create: `docs/insights/_data/E5b_per_stratum.csv`
- Create: `docs/figures/E5b_distance_curve.png`
- Create: `docs/figures/E5b_cross_dataset_overlay.png`

The analysis script does all the heavy lifting (load → stratum stats with bootstrap CI → plots → save artefacts). The notebook is a thin top-to-bottom reproducer that imports and calls the script — same pattern as `notebooks/E1b_per_layer_localisation.ipynb`.

- [ ] **Step 1: Write the analysis script**

Create `scripts/analyze_e5b_distance.py`:

```python
"""E5b — per-stratum direction-follow / adoption / em / distance with CIs + plots.

Loads the latest run from `outputs/experiment_distance_vqa/<model>/` and
`outputs/experiment_distance_tally/<model>/`, computes per-stratum stats
with bootstrap CIs, draws two figures (per-dataset distance curve,
cross-dataset overlay), and writes a tidy CSV summary.

Usage: `uv run python scripts/analyze_e5b_distance.py [--model llava-next-interleaved-7b]`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "VQAv2": "experiment_distance_vqa",
    "TallyQA": "experiment_distance_tally",
}
STRATUM_ORDER = ["S1", "S2", "S3", "S4", "S5"]
STRATUM_MIDPOINT = {"S1": 0.5, "S2": 3.5, "S3": 18.0, "S4": 165.0, "S5": 650.0}
STRATUM_LABEL = {"S1": "[0,1]", "S2": "[2,5]", "S3": "[6,30]", "S4": "[31,300]", "S5": "[301,∞)"}
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _latest_run_dir(experiment_dir: str, model: str) -> Path:
    base = PROJECT_ROOT / "outputs" / experiment_dir / model
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run dirs under {base}")
    return runs[-1]


def _load_records(model: str) -> pd.DataFrame:
    frames = []
    for ds_label, ds_dir in DATASETS.items():
        run_dir = _latest_run_dir(ds_dir, model)
        recs = [json.loads(l) for l in (run_dir / "predictions.jsonl").open()]
        df = pd.DataFrame(recs)
        df["dataset"] = ds_label
        df["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.nanmean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_stratum_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in records["dataset"].unique():
        ds = records[records["dataset"] == dataset]
        baseline = ds[ds["condition"] == "target_only"]
        baseline_df_mean = float(np.nanmean(baseline["anchor_direction_followed"].astype(float)))
        for stratum in STRATUM_ORDER:
            cell = ds[ds["condition"] == f"target_plus_irrelevant_number_{stratum}"]
            if cell.empty:
                continue
            df_vals = cell["anchor_direction_followed"].astype(float).to_numpy()
            adoption_vals = cell["anchor_adopted"].astype(float).to_numpy()
            em_vals = cell["exact_match"].astype(float).to_numpy()
            dist_vals = cell["numeric_distance_to_anchor"].astype(float).to_numpy()
            df_lo, df_hi = _bootstrap_ci(df_vals)
            rows.append({
                "dataset": dataset,
                "stratum": stratum,
                "stratum_range": STRATUM_LABEL[stratum],
                "stratum_midpoint": STRATUM_MIDPOINT[stratum],
                "n": len(cell),
                "direction_follow_rate": float(np.nanmean(df_vals)),
                "direction_follow_ci_lo": df_lo,
                "direction_follow_ci_hi": df_hi,
                "adoption_rate": float(np.nanmean(adoption_vals)),
                "exact_match": float(np.nanmean(em_vals)),
                "mean_distance_to_anchor": float(np.nanmean(dist_vals)) if not np.isnan(dist_vals).all() else float("nan"),
                "df_minus_baseline": float(np.nanmean(df_vals)) - baseline_df_mean,
            })
    return pd.DataFrame(rows)


def plot_distance_curve(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dataset in zip(axes, ["VQAv2", "TallyQA"]):
        ds = summary[summary["dataset"] == dataset].sort_values("stratum_midpoint")
        x = ds["stratum_midpoint"].to_numpy()
        y = ds["direction_follow_rate"].to_numpy()
        lo = ds["direction_follow_ci_lo"].to_numpy()
        hi = ds["direction_follow_ci_hi"].to_numpy()
        ax.plot(x, y, "o-", color="C0")
        ax.fill_between(x, lo, hi, color="C0", alpha=0.2)
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xticks(ds["stratum_midpoint"].tolist())
        ax.set_xticklabels(ds["stratum"].tolist())
        ax.set_xlabel("Anchor distance stratum (|a − GT|)")
        ax.set_title(f"{dataset}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("direction-follow rate (95% CI)")
    fig.suptitle("E5b — anchor effect vs distance (llava-interleave-7b, n≈500/dataset)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_dataset_overlay(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for dataset, color in [("VQAv2", "C0"), ("TallyQA", "C1")]:
        ds = summary[summary["dataset"] == dataset].sort_values("stratum_midpoint")
        x = ds["stratum_midpoint"].to_numpy()
        y = ds["direction_follow_rate"].to_numpy()
        lo = ds["direction_follow_ci_lo"].to_numpy()
        hi = ds["direction_follow_ci_hi"].to_numpy()
        ax.plot(x, y, "o-", color=color, label=dataset)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18)
    ax.set_xscale("symlog", linthresh=2)
    midpoints = sorted(STRATUM_MIDPOINT.values())
    ax.set_xticks(midpoints)
    ax.set_xticklabels(STRATUM_ORDER)
    ax.set_xlabel("Anchor distance stratum (|a − GT|)")
    ax.set_ylabel("direction-follow rate (95% CI)")
    ax.set_title("E5b — cross-dataset anchor-distance comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(model: str = "llava-next-interleaved-7b") -> dict:
    records = _load_records(model)
    summary = per_stratum_summary(records)

    out_csv = PROJECT_ROOT / "docs" / "insights" / "_data" / "E5b_per_stratum.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    fig_dir = PROJECT_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_distance_curve(summary, fig_dir / "E5b_distance_curve.png")
    plot_cross_dataset_overlay(summary, fig_dir / "E5b_cross_dataset_overlay.png")

    return {
        "summary": summary,
        "n_records": len(records),
        "out_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "figures": [
            "docs/figures/E5b_distance_curve.png",
            "docs/figures/E5b_cross_dataset_overlay.png",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava-next-interleaved-7b")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = run(model=args.model)
    print(out["summary"].to_string(index=False))
    print(f"\nwrote {out['out_csv']}")
    for f in out["figures"]:
        print(f"wrote {f}")
```

- [ ] **Step 2: Sanity-run the analysis script**

Run: `uv run python scripts/analyze_e5b_distance.py`

Expected: prints a 10-row table (2 datasets × 5 strata) + writes `docs/insights/_data/E5b_per_stratum.csv` + 2 PNGs under `docs/figures/`. Each line in the table has finite numeric `direction_follow_rate` and CI columns.

- [ ] **Step 3: Build the notebook via nbformat**

Create `scripts/build_e5b_notebook.py`:

```python
"""Construct notebooks/E5b_anchor_distance.ipynb via nbformat.

Run once to (re)generate the notebook source; then execute the notebook
itself (`jupyter nbconvert --to notebook --execute --inplace`) to embed
output cells. Same builder + execute pattern as other E*-notebooks.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(nbf.v4.new_markdown_cell(
        "# E5b — Anchor-distance robustness sweep (reproducer)\n"
        "\n"
        "Top-to-bottom reproducer for `docs/experiments/E5b-anchor-distance-design.md`.\n"
        "Reads the latest VQAv2 + TallyQA stratified runs under "
        "`outputs/experiment_distance_*/llava-next-interleaved-7b/<timestamp>/predictions.jsonl` "
        "and regenerates:\n"
        "\n"
        "1. Per-stratum summary table (direction-follow / adoption / EM / mean distance, with 95% bootstrap CI).\n"
        "2. Distance curve per dataset.\n"
        "3. Cross-dataset overlay.\n"
        "\n"
        "All heavy lifting lives in `scripts/analyze_e5b_distance.py` — this notebook just invokes it and displays the outputs."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT / 'scripts'))\n"
        "from analyze_e5b_distance import run\n"
        "out = run()\n"
        "summary = out['summary']\n"
        "print(f\"loaded {out['n_records']} records, wrote {out['out_csv']}\")"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Per-stratum summary\n"
        "\n"
        "Each row = one (dataset, stratum) cell. `direction_follow_rate` is the headline number; "
        "`df_minus_baseline` subtracts the per-dataset target_only direction-follow (≈0 by construction)."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "pd.set_option('display.float_format', '{:0.3f}'.format)\n"
        "summary"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Distance curve per dataset"
    ))

    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5b_distance_curve.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Cross-dataset overlay\n"
        "\n"
        "Two lines on one axis. The cutoff `d*` for paper-headline subset is the largest stratum where "
        "the effect remains > 50% of S1's effect."
    ))

    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5b_cross_dataset_overlay.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Sanity check vs main run\n"
        "\n"
        "VQAv2 main run (random anchor 0–9, n=17,730) had `direction_follow_rate = 0.348` on llava-interleave-7b. "
        "If E5b's anchor-distance distribution were uniform random over 0–9, the pooled effect would land near 0.348. "
        "The stratified sweep oversamples far strata (S4/S5), so we expect the pooled E5b direction-follow to be **lower** than 0.348."
    ))

    cells.append(nbf.v4.new_code_cell(
        "vqa_pool = summary[summary['dataset'] == 'VQAv2']\n"
        "n_total = vqa_pool['n'].sum()\n"
        "weighted_df = float((vqa_pool['direction_follow_rate'] * vqa_pool['n']).sum() / n_total)\n"
        "print(f'E5b VQAv2 pooled direction-follow (n={n_total}): {weighted_df:.3f}')\n"
        "print(f'reference — main run random-anchor 0..9 (n=17,730):  0.348')"
    ))

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).resolve().parents[1] / "notebooks" / "E5b_anchor_distance.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)
    print(f"wrote {out_path.relative_to(out_path.parents[1])}")
```

- [ ] **Step 4: Run the builder**

Run: `uv run python scripts/build_e5b_notebook.py`
Expected: writes `notebooks/E5b_anchor_distance.ipynb` (no outputs yet — just the source).

- [ ] **Step 5: Execute the notebook in-place to embed outputs**

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/E5b_anchor_distance.ipynb`

Expected: every code cell runs; `summary` DataFrame and the two PNGs appear as embedded outputs in the .ipynb file.

- [ ] **Step 6: Verify the notebook has executed outputs**

Run:
```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/E5b_anchor_distance.ipynb', as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == 'code']
without_output = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
print(f'{len(code_cells)} code cells, {len(without_output)} without outputs: {without_output}')
"
```

Expected: `0 without outputs: []`. If any code cell has empty outputs, debug it and re-execute.

- [ ] **Step 7: Commit notebook + analysis script + figures + summary CSV**

```bash
git add scripts/analyze_e5b_distance.py scripts/build_e5b_notebook.py \
        notebooks/E5b_anchor_distance.ipynb \
        docs/insights/_data/E5b_per_stratum.csv \
        docs/figures/E5b_distance_curve.png \
        docs/figures/E5b_cross_dataset_overlay.png
git commit -m "Add E5b analysis script + reproducible notebook with executed outputs"
```

---

## Final Verification

- [ ] **Step 1: Confirm test suite is green**

Run: `uv run python -m pytest -v`
Expected: all tests pass (existing + 8 new tests across Tasks 1–3).

- [ ] **Step 2: Confirm new commits are clean**

Run: `git log --oneline -15`
Expected: 9 new commits (Tasks 1–6 source/configs, Task 8 roadmap, Task 10 analysis+notebook). Task 7 (smoke) and Task 9 (full run) are output-only and don't commit. `git status` clean.

- [ ] **Step 3: Hand off to user**

Implementation + run + reproducible notebook all complete. Hand back to user with:
- Reference to commits added
- Path to executed notebook (`notebooks/E5b_anchor_distance.ipynb`)
- Headline numbers from the summary CSV (per-dataset peak direction-follow + chosen `d*` band)
- Suggested next step: writing the results writeup `docs/experiments/E5b-anchor-distance.md` (+ `_ko.md`) and the distilled insight `docs/insights/E5b-anchor-distance-evidence.md` (+ `_ko.md`) once the user has reviewed the notebook.
