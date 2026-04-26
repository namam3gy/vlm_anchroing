from __future__ import annotations

import unittest

from vlm_anchor.visualization import _compute_paired_effects


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


if __name__ == "__main__":
    unittest.main()
