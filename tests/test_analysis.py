from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_anchor.analysis import (
    build_paired_dataframe,
    filter_anchor_distance_outliers,
    load_experiment_records,
    parse_int_like,
    summarize_anchor_distance_outliers,
)


class ParseIntLikeTest(unittest.TestCase):
    def test_accepts_integer_like_values_only(self) -> None:
        self.assertEqual(parse_int_like("7"), 7)
        self.assertEqual(parse_int_like("-4"), -4)
        self.assertEqual(parse_int_like("8.0"), 8)
        self.assertIsNone(parse_int_like("3.5"))
        self.assertIsNone(parse_int_like("three"))


class BuildPairedDataFrameTest(unittest.TestCase):
    def test_derives_anchor_and_neutral_shift_metrics(self) -> None:
        rows = [
            {
                "experiment_root": "/tmp/root",
                "experiment_name": "root",
                "model": "demo-model",
                "model_root": "/tmp/root/demo-model",
                "sample_instance_id": "11_21_set00",
                "sample_instance_index": 0,
                "question_id": 11,
                "image_id": 21,
                "question": "How many objects?",
                "question_type": "how many",
                "condition": "target_only",
                "condition_label": "target only",
                "prediction": "3",
                "prediction_int": 3,
                "ground_truth": "4",
                "ground_truth_int": 4,
                "anchor_value": None,
                "anchor_int": None,
                "standard_vqa_accuracy": 1.0,
                "exact_match": 0,
                "anchor_adopted": 0,
                "anchor_direction_followed": 0,
                "numeric_distance_to_anchor": None,
                "input_image_paths": ["/tmp/root/images/target.png"],
                "irrelevant_image": None,
                "raw_prediction": '{"result": 3}',
                "answers": ["4"] * 10,
                "backend": "huggingface",
                "irrelevant_type": "none",
                "is_numeric_prediction": True,
                "is_numeric_ground_truth": True,
                "is_numeric_anchor": False,
            },
            {
                "experiment_root": "/tmp/root",
                "experiment_name": "root",
                "model": "demo-model",
                "model_root": "/tmp/root/demo-model",
                "sample_instance_id": "11_21_set00",
                "sample_instance_index": 0,
                "question_id": 11,
                "image_id": 21,
                "question": "How many objects?",
                "question_type": "how many",
                "condition": "target_plus_irrelevant_number",
                "condition_label": "+ irrelevant number",
                "prediction": "6",
                "prediction_int": 6,
                "ground_truth": "4",
                "ground_truth_int": 4,
                "anchor_value": "10",
                "anchor_int": 10,
                "standard_vqa_accuracy": 0.0,
                "exact_match": 0,
                "anchor_adopted": 0,
                "anchor_direction_followed": 1,
                "numeric_distance_to_anchor": 4.0,
                "input_image_paths": ["/tmp/root/images/target.png", "/tmp/root/images/10.png"],
                "irrelevant_image": "/tmp/root/images/10.png",
                "raw_prediction": '{"result": 6}',
                "answers": ["4"] * 10,
                "backend": "huggingface",
                "irrelevant_type": "number",
                "is_numeric_prediction": True,
                "is_numeric_ground_truth": True,
                "is_numeric_anchor": True,
            },
            {
                "experiment_root": "/tmp/root",
                "experiment_name": "root",
                "model": "demo-model",
                "model_root": "/tmp/root/demo-model",
                "sample_instance_id": "11_21_set00",
                "sample_instance_index": 0,
                "question_id": 11,
                "image_id": 21,
                "question": "How many objects?",
                "question_type": "how many",
                "condition": "target_plus_irrelevant_neutral",
                "condition_label": "+ irrelevant neutral",
                "prediction": "5",
                "prediction_int": 5,
                "ground_truth": "4",
                "ground_truth_int": 4,
                "anchor_value": None,
                "anchor_int": None,
                "standard_vqa_accuracy": 0.0,
                "exact_match": 0,
                "anchor_adopted": 0,
                "anchor_direction_followed": 0,
                "numeric_distance_to_anchor": None,
                "input_image_paths": ["/tmp/root/images/target.png", "/tmp/root/images/neutral.png"],
                "irrelevant_image": "/tmp/root/images/neutral.png",
                "raw_prediction": '{"result": 5}',
                "answers": ["4"] * 10,
                "backend": "huggingface",
                "irrelevant_type": "neutral",
                "is_numeric_prediction": True,
                "is_numeric_ground_truth": True,
                "is_numeric_anchor": False,
            },
        ]

        paired_df = build_paired_dataframe(pd.DataFrame(rows))
        self.assertEqual(len(paired_df), 1)
        row = paired_df.iloc[0]

        self.assertEqual(row["number_shift"], 3)
        self.assertEqual(row["neutral_shift"], 2)
        self.assertEqual(row["neutral_abs_shift"], 2)
        self.assertEqual(row["anchor_gt_distance"], 6)
        self.assertEqual(row["anchor_pull"], 3)
        self.assertEqual(row["signed_anchor_movement"], 3)
        self.assertEqual(row["moved_closer_to_anchor"], 1)
        self.assertEqual(row["changed_toward_anchor"], 1)
        self.assertEqual(row["anchor_distance_bin"], "[5,10)")


class AnchorOutlierFilterTest(unittest.TestCase):
    def test_filters_entire_sample_instances_when_anchor_distance_is_extreme(self) -> None:
        rows = []
        for sample_instance_id, gt, anchor in [
            ("sample_keep_1", 3, 4),
            ("sample_keep_2", 4, 6),
            ("sample_keep_3", 5, 8),
            ("sample_keep_4", 6, 10),
            ("sample_outlier", 5, 80),
        ]:
            base_prediction = str(gt)
            number_prediction = str(anchor)
            neutral_prediction = str(gt)
            rows.extend(
                [
                    {
                        "experiment_root": "/tmp/root",
                        "experiment_name": "root",
                        "model": "demo-model",
                        "model_root": "/tmp/root/demo-model",
                        "sample_instance_id": sample_instance_id,
                        "sample_instance_index": 0,
                        "question_id": 1,
                        "image_id": 2,
                        "question": "How many objects?",
                        "question_type": "how many",
                        "condition": "target_only",
                        "condition_label": "target only",
                        "prediction": base_prediction,
                        "prediction_int": int(base_prediction),
                        "ground_truth": str(gt),
                        "ground_truth_int": gt,
                        "anchor_value": None,
                        "anchor_int": None,
                        "standard_vqa_accuracy": 1.0,
                        "exact_match": 1,
                        "anchor_adopted": 0,
                        "anchor_direction_followed": 0,
                        "numeric_distance_to_anchor": None,
                        "input_image_paths": ["/tmp/root/images/target.png"],
                        "irrelevant_image": None,
                        "raw_prediction": '{"result": 3}',
                        "answers": [str(gt)] * 10,
                        "backend": "huggingface",
                        "irrelevant_type": "none",
                        "is_numeric_prediction": True,
                        "is_numeric_ground_truth": True,
                        "is_numeric_anchor": False,
                    },
                    {
                        "experiment_root": "/tmp/root",
                        "experiment_name": "root",
                        "model": "demo-model",
                        "model_root": "/tmp/root/demo-model",
                        "sample_instance_id": sample_instance_id,
                        "sample_instance_index": 0,
                        "question_id": 1,
                        "image_id": 2,
                        "question": "How many objects?",
                        "question_type": "how many",
                        "condition": "target_plus_irrelevant_number",
                        "condition_label": "+ irrelevant number",
                        "prediction": number_prediction,
                        "prediction_int": int(number_prediction),
                        "ground_truth": str(gt),
                        "ground_truth_int": gt,
                        "anchor_value": str(anchor),
                        "anchor_int": anchor,
                        "standard_vqa_accuracy": 0.0,
                        "exact_match": 0,
                        "anchor_adopted": int(anchor == gt),
                        "anchor_direction_followed": 1,
                        "numeric_distance_to_anchor": float(abs(anchor - gt)),
                        "input_image_paths": ["/tmp/root/images/target.png", f"/tmp/root/images/{anchor}.png"],
                        "irrelevant_image": f"/tmp/root/images/{anchor}.png",
                        "raw_prediction": '{"result": 4}',
                        "answers": [str(gt)] * 10,
                        "backend": "huggingface",
                        "irrelevant_type": "number",
                        "is_numeric_prediction": True,
                        "is_numeric_ground_truth": True,
                        "is_numeric_anchor": True,
                    },
                    {
                        "experiment_root": "/tmp/root",
                        "experiment_name": "root",
                        "model": "demo-model",
                        "model_root": "/tmp/root/demo-model",
                        "sample_instance_id": sample_instance_id,
                        "sample_instance_index": 0,
                        "question_id": 1,
                        "image_id": 2,
                        "question": "How many objects?",
                        "question_type": "how many",
                        "condition": "target_plus_irrelevant_neutral",
                        "condition_label": "+ irrelevant neutral",
                        "prediction": neutral_prediction,
                        "prediction_int": int(neutral_prediction),
                        "ground_truth": str(gt),
                        "ground_truth_int": gt,
                        "anchor_value": None,
                        "anchor_int": None,
                        "standard_vqa_accuracy": 1.0,
                        "exact_match": 1,
                        "anchor_adopted": 0,
                        "anchor_direction_followed": 0,
                        "numeric_distance_to_anchor": None,
                        "input_image_paths": ["/tmp/root/images/target.png", "/tmp/root/images/neutral.png"],
                        "irrelevant_image": "/tmp/root/images/neutral.png",
                        "raw_prediction": '{"result": 3}',
                        "answers": [str(gt)] * 10,
                        "backend": "huggingface",
                        "irrelevant_type": "neutral",
                        "is_numeric_prediction": True,
                        "is_numeric_ground_truth": True,
                        "is_numeric_anchor": False,
                    },
                ]
            )

        records_df = pd.DataFrame(rows)
        paired_df = build_paired_dataframe(records_df)
        outlier_df, summary = summarize_anchor_distance_outliers(paired_df, iqr_multiplier=1.5)

        self.assertEqual(summary["outlier_count"], 1)
        self.assertEqual(
            outlier_df.loc[outlier_df["is_outlier"], "sample_instance_id"].tolist(),
            ["sample_outlier"],
        )

        filtered_records_df, filtered_paired_df, _, filtered_summary = filter_anchor_distance_outliers(
            records_df,
            paired_df,
            iqr_multiplier=1.5,
        )
        self.assertEqual(filtered_summary["outlier_count"], 1)
        self.assertEqual(set(filtered_paired_df["sample_instance_id"].tolist()), {"sample_keep_1", "sample_keep_2", "sample_keep_3", "sample_keep_4"})
        self.assertEqual(set(filtered_records_df["sample_instance_id"].tolist()), {"sample_keep_1", "sample_keep_2", "sample_keep_3", "sample_keep_4"})


class LoadExperimentRecordsTest(unittest.TestCase):
    def test_reads_model_roots_under_experiment_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "demo-model"
            model_dir.mkdir()
            rows = [
                {
                    "model": "demo-model",
                    "sample_instance_id": "1_2_set00",
                    "sample_instance_index": 0,
                    "question_id": 1,
                    "image_id": 2,
                    "question": "How many?",
                    "question_type": "how many",
                    "condition": "target_only",
                    "irrelevant_type": "none",
                    "irrelevant_image": None,
                    "ground_truth": "3",
                    "answers": ["3"] * 10,
                    "backend": "huggingface",
                    "raw_prediction": '{"result": 3}',
                    "prediction": "3",
                    "anchor_value": None,
                    "standard_vqa_accuracy": 1.0,
                    "exact_match": 1,
                    "anchor_adopted": 0,
                    "anchor_direction_followed": 0,
                    "numeric_distance_to_anchor": None,
                    "input_image_paths": ["/tmp/target.png"],
                }
            ]
            with open(model_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            records_df = load_experiment_records(root)
            self.assertEqual(records_df["model"].tolist(), ["demo-model"])
            self.assertEqual(records_df["prediction_int"].tolist(), [3])
            self.assertEqual(records_df["experiment_name"].tolist(), [root.name])


if __name__ == "__main__":
    unittest.main()
