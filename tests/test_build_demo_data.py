import csv
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_demo_data as bdd  # noqa: E402


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/build_demo_data.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "build site/data/demo.json" in result.stdout.lower()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic_outputs(tmp_path: Path) -> Path:
    """Two models × one dataset × two samples, all four conditions present."""
    root = tmp_path / "outputs"
    base_row = {
        "model": "", "sample_instance_id": "", "sample_instance_index": "0",
        "question_id": "", "image_id": "", "question": "How many?",
        "question_type": "free_form", "condition": "", "irrelevant_type": "",
        "irrelevant_image": "", "ground_truth": "4", "answers": "['4']",
        "backend": "huggingface", "raw_prediction": "", "prediction": "",
        "answer_token_id": "", "answer_token_text": "",
        "answer_token_logit": "", "answer_token_probability": "",
        "token_info": "", "anchor_value": "5",
        "anchor_stratum_id": "", "anchor_stratum_range": "",
        "standard_vqa_accuracy": "1", "exact_match": "1",
        "anchor_adopted": "0", "anchor_direction_followed": "0",
        "numeric_distance_to_anchor": "1",
        "input_image_paths": "['/abs/inputs/vqav2_number_val/images/000000000136.jpg']",
        "anchor_direction_followed_moved": "0",
        "pred_b_equal_anchor": "0", "pred_diff_from_base": "0",
    }
    for outdir in ("llava-onevision-qwen2-7b-ov", "qwen2.5-vl-7b-instruct"):
        rows: list[dict] = []
        for sid in ("S1", "S2"):
            for cond, pred in (
                ("target_only", "4"),
                ("target_plus_irrelevant_number_S1", "5"),
                ("target_plus_irrelevant_number_masked_S1", "4"),
                ("target_plus_irrelevant_neutral", "4"),
            ):
                rows.append({
                    **base_row,
                    "model": outdir,
                    "sample_instance_id": sid,
                    "condition": cond,
                    "prediction": pred,
                })
        run_dir = root / "experiment_synthetic" / outdir / "20260510-000000"
        _write_csv(run_dir / "predictions.csv", rows)
    return root


def test_load_predictions_pivots_long_to_wide(synthetic_outputs):
    by_model = bdd.load_predictions(
        outputs_root=synthetic_outputs,
        anchor_stratum="S1",
    )
    assert set(by_model) == {"llava-onevision-7b", "qwen2.5-vl-7b"}
    samples = by_model["llava-onevision-7b"]
    assert "S1" in samples and "S2" in samples
    s1 = samples["S1"]
    assert s1["b"] == 4
    assert s1["a"] == 5
    assert s1["m"] == 4
    assert s1["d"] == 4
    assert s1["meta"]["dataset"] == "VQAv2"  # inferred from input_image_paths
    assert s1["meta"]["question"] == "How many?"
    assert s1["meta"]["gt"] == 4
    assert s1["meta"]["anchor"] == 5


def test_load_predictions_skips_before_c_form(tmp_path):
    forbidden = tmp_path / "outputs" / "before_C_form" / "x" / "y" / "predictions.csv"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("model\nignored\n")
    by_model = bdd.load_predictions(outputs_root=tmp_path / "outputs", anchor_stratum="S1")
    assert by_model == {}


def test_load_predictions_drops_samples_missing_b(tmp_path):
    """Samples that lack the b (target_only) condition are dropped entirely."""
    root = tmp_path / "outputs"
    base_row = {
        "model": "llava-onevision-qwen2-7b-ov", "sample_instance_id": "",
        "sample_instance_index": "0", "question_id": "", "image_id": "",
        "question": "How many?", "question_type": "free_form",
        "condition": "", "irrelevant_type": "", "irrelevant_image": "",
        "ground_truth": "4", "answers": "['4']", "backend": "huggingface",
        "raw_prediction": "", "prediction": "", "answer_token_id": "",
        "answer_token_text": "", "answer_token_logit": "",
        "answer_token_probability": "", "token_info": "", "anchor_value": "5",
        "anchor_stratum_id": "", "anchor_stratum_range": "",
        "standard_vqa_accuracy": "1", "exact_match": "1",
        "anchor_adopted": "0", "anchor_direction_followed": "0",
        "numeric_distance_to_anchor": "1",
        "input_image_paths": "['/abs/inputs/vqav2_number_val/images/x.jpg']",
        "anchor_direction_followed_moved": "0",
        "pred_b_equal_anchor": "0", "pred_diff_from_base": "0",
    }
    # S1 has b/a/m/d (full); S2 has only a/m/d (missing b)
    rows = []
    for sid, conds in (
        ("S1", [("target_only", "4"), ("target_plus_irrelevant_number_S1", "5"),
                ("target_plus_irrelevant_number_masked_S1", "4"),
                ("target_plus_irrelevant_neutral", "4")]),
        ("S2", [("target_plus_irrelevant_number_S1", "5"),
                ("target_plus_irrelevant_number_masked_S1", "4"),
                ("target_plus_irrelevant_neutral", "4")]),
    ):
        for cond, pred in conds:
            rows.append({**base_row, "sample_instance_id": sid,
                         "condition": cond, "prediction": pred})
    run_dir = root / "experiment_x" / "llava-onevision-qwen2-7b-ov" / "20260510-000000"
    _write_csv(run_dir / "predictions.csv", rows)

    by_model = bdd.load_predictions(outputs_root=root, anchor_stratum="S1")
    samples = by_model["llava-onevision-7b"]
    assert "S1" in samples
    assert "S2" not in samples  # dropped — no b condition
