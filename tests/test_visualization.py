from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

skimage_module = types.ModuleType("skimage")
skimage_transform_module = types.ModuleType("skimage.transform")
skimage_transform_module.resize = lambda *args, **kwargs: None
skimage_module.transform = skimage_transform_module
sys.modules.setdefault("skimage", skimage_module)
sys.modules["skimage.transform"] = skimage_transform_module

from vlm_anchor.visualization import _compute_paired_effects, overlay_attention_on_image


class ComputePairedEffectsTest(unittest.TestCase):
    def test_keeps_multiple_variants_for_the_same_question(self) -> None:
        records = [
            {
                "sample_instance_id": "11_21_set00",
                "question_id": 11,
                "condition": "target_only",
                "standard_vqa_accuracy": 1.0,
                "exact_match": 1,
                "prediction": "5",
                "ground_truth": "5",
                "anchor_value": None,
            },
            {
                "sample_instance_id": "11_21_set00",
                "question_id": 11,
                "condition": "target_plus_irrelevant_number",
                "standard_vqa_accuracy": 0.0,
                "exact_match": 0,
                "prediction": "3",
                "ground_truth": "5",
                "anchor_value": "3",
            },
            {
                "sample_instance_id": "11_21_set00",
                "question_id": 11,
                "condition": "target_plus_irrelevant_neutral",
                "standard_vqa_accuracy": 1.0,
                "exact_match": 1,
                "prediction": "5",
                "ground_truth": "5",
                "anchor_value": None,
            },
            {
                "sample_instance_id": "11_21_set01",
                "question_id": 11,
                "condition": "target_only",
                "standard_vqa_accuracy": 1.0,
                "exact_match": 1,
                "prediction": "7",
                "ground_truth": "7",
                "anchor_value": None,
            },
            {
                "sample_instance_id": "11_21_set01",
                "question_id": 11,
                "condition": "target_plus_irrelevant_number",
                "standard_vqa_accuracy": 1.0,
                "exact_match": 0,
                "prediction": "10",
                "ground_truth": "7",
                "anchor_value": "10",
            },
            {
                "sample_instance_id": "11_21_set01",
                "question_id": 11,
                "condition": "target_plus_irrelevant_neutral",
                "standard_vqa_accuracy": 0.0,
                "exact_match": 0,
                "prediction": "8",
                "ground_truth": "7",
                "anchor_value": None,
            },
        ]

        paired = _compute_paired_effects(records)

        self.assertEqual(paired["pair_count"], 2)
        self.assertAlmostEqual(paired["accuracy_delta_number_vs_target_only"], -0.5)
        self.assertAlmostEqual(paired["accuracy_delta_neutral_vs_target_only"], -0.5)
        self.assertAlmostEqual(paired["moved_closer_to_anchor_rate"], 1.0)
        self.assertAlmostEqual(paired["mean_anchor_pull"], 2.5)


class OverlayAttentionOnImageTest(unittest.TestCase):
    def test_resizes_token_grid_to_full_image_resolution(self) -> None:
        image = Image.new("RGB", (10, 8), color="white")
        heatmap = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)

        overlay = overlay_attention_on_image(image, heatmap)

        self.assertEqual(overlay.shape, (8, 10, 3))
        self.assertGreaterEqual(float(overlay.min()), 0.0)
        self.assertLessEqual(float(overlay.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
