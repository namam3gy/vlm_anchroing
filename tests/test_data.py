from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm_anchor.data import assign_irrelevant_images


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


if __name__ == "__main__":
    unittest.main()
