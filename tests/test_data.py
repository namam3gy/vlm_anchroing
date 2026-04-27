from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from vlm_anchor.data import (
    ANCHOR_DISTANCE_STRATA,
    assign_irrelevant_images,
    assign_stratified_anchors,
    build_conditions,
    load_number_vqa_samples,
    sample_stratified_anchors,
)


class AssignIrrelevantImagesTest(unittest.TestCase):
    def test_expands_each_sample_into_distinct_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            number_dir = root / "irrelevant_number"
            neutral_dir = root / "irrelevant_neutral"
            number_dir.mkdir()
            neutral_dir.mkdir()

            for idx in range(12):
                (number_dir / f"{idx}.png").write_bytes(b"png")
                (neutral_dir / f"neutral_{idx}.png").write_bytes(b"png")

            samples = [
                {
                    "question_id": 101,
                    "image_id": 202,
                    "question": "How many objects are visible?",
                    "image": root / "target.png",
                    "ground_truth": "3",
                    "answers": ["3"] * 10,
                    "question_type": "how many",
                }
            ]

            enriched = assign_irrelevant_images(
                samples,
                irrelevant_number_dir=number_dir,
                irrelevant_neutral_dir=neutral_dir,
                seed=7,
                variants_per_sample=10,
            )

            self.assertEqual(len(enriched), 10)
            self.assertEqual([row["sample_instance_index"] for row in enriched], list(range(10)))
            self.assertEqual(len({row["sample_instance_id"] for row in enriched}), 10)
            self.assertEqual(len({Path(row["irrelevant_number_image"]).name for row in enriched}), 10)
            self.assertEqual(len({Path(row["irrelevant_neutral_image"]).name for row in enriched}), 10)


class LoadNumberVqaSamplesTest(unittest.TestCase):
    def test_filters_by_answer_range_and_samples_per_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            image_dir.mkdir()

            rows = [
                self._build_row(question_id=1, image_id=11, answer="0"),
                self._build_row(question_id=2, image_id=12, answer="1"),
                self._build_row(question_id=3, image_id=13, answer="1"),
                self._build_row(question_id=4, image_id=14, answer="1"),
                self._build_row(question_id=5, image_id=15, answer="2"),
                self._build_row(question_id=6, image_id=16, answer="2"),
                self._build_row(question_id=7, image_id=17, answer="2"),
                self._build_row(question_id=8, image_id=18, answer="10"),
            ]

            for row in rows:
                (image_dir / Path(row["image_file"]).name).write_bytes(b"png")

            questions_path = root / "questions.jsonl"
            with open(questions_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            loaded = load_number_vqa_samples(
                dataset_path=root,
                max_samples=None,
                require_single_numeric_gt=True,
                answer_range=2,
                samples_per_answer=2,
            )

            self.assertEqual([row["question_id"] for row in loaded], [1, 2, 3, 5, 6])
            self.assertEqual([row["ground_truth"] for row in loaded], ["0", "1", "1", "2", "2"])

    def test_accepts_non_number_answer_type_when_filter_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            image_dir.mkdir()

            rows = [
                self._build_row(question_id=1, image_id=11, answer="3", answer_type="integer"),
                self._build_row(question_id=2, image_id=12, answer="7", answer_type="text"),
                self._build_row(question_id=3, image_id=13, answer="1.5", answer_type="float"),
                self._build_row(question_id=4, image_id=14, answer="Yes", answer_type="text"),
            ]
            for row in rows:
                (image_dir / Path(row["image_file"]).name).write_bytes(b"png")
            with open(root / "questions.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            loaded = load_number_vqa_samples(
                dataset_path=root,
                max_samples=None,
                require_single_numeric_gt=True,
            )
            self.assertEqual([row["question_id"] for row in loaded], [1, 2])
            self.assertEqual([row["ground_truth"] for row in loaded], ["3", "7"])

    def test_answer_type_filter_limits_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            image_dir.mkdir()

            rows = [
                self._build_row(question_id=1, image_id=11, answer="3", answer_type="integer"),
                self._build_row(question_id=2, image_id=12, answer="7", answer_type="text"),
                self._build_row(question_id=3, image_id=13, answer="2", answer_type="number"),
            ]
            for row in rows:
                (image_dir / Path(row["image_file"]).name).write_bytes(b"png")
            with open(root / "questions.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            loaded = load_number_vqa_samples(
                dataset_path=root,
                max_samples=None,
                require_single_numeric_gt=True,
                answer_type_filter=["integer", "number"],
            )
            self.assertEqual(sorted(row["question_id"] for row in loaded), [1, 3])

    def test_rejects_invalid_filter_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "questions.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "answer_range"):
                load_number_vqa_samples(dataset_path=root, max_samples=None, answer_range=-1)

            with self.assertRaisesRegex(ValueError, "samples_per_answer"):
                load_number_vqa_samples(dataset_path=root, max_samples=None, samples_per_answer=0)

    @staticmethod
    def _build_row(question_id: int, image_id: int, answer: str, answer_type: str = "number") -> dict:
        return {
            "question_id": question_id,
            "image_id": image_id,
            "question": f"How many items are in image {image_id}?",
            "image_file": f"images/{image_id}.png",
            "multiple_choice_answer": answer,
            "answer_type": answer_type,
            "question_type": "how many",
            "answers": [{"answer": answer} for _ in range(10)],
        }


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

        self.assertIn(anchors[0], (3, 4))
        self.assertIsNone(anchors[2])
        self.assertIsNone(anchors[3])
        self.assertIsNone(anchors[4])

    def test_seeded_rng_is_reproducible(self) -> None:
        gt = 5
        inventory = list(range(0, 101))
        a = sample_stratified_anchors(gt, inventory, random.Random(1234))
        b = sample_stratified_anchors(gt, inventory, random.Random(1234))
        self.assertEqual(a, b)


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

    def test_stratified_with_no_valid_anchors_yields_only_baseline(self) -> None:
        # Covers two edge cases: empty anchor_strata, and all entries with
        # anchor_value=None. Both should produce exactly one target_only record.
        for strata in [[],
                       [{"stratum_id": "S1", "stratum_range": (0, 1),
                         "anchor_value": None, "irrelevant_number_image": None}]]:
            sample = {
                "question_id": 1, "image_id": 1,
                "question": "Q?", "image": Path("/tmp/t.png"),
                "ground_truth": "3", "answers": ["3"], "question_type": "",
                "anchor_strata": strata,
            }
            conds = list(build_conditions(sample))
            self.assertEqual(len(conds), 1)
            self.assertEqual(conds[0]["condition"], "target_only")
            self.assertIsNone(conds[0]["anchor_stratum_id"])

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


if __name__ == "__main__":
    unittest.main()
