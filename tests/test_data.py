from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vlm_anchor.data import assign_irrelevant_images, load_number_vqa_samples


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


if __name__ == "__main__":
    unittest.main()
