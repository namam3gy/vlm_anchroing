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


def _make_sample(b, a, m, d, gt, anchor, dataset="VQAv2", question="Q?",
                 target="/abs/inputs/vqav2_number_val/images/x.jpg"):
    return {
        "b": b, "a": a, "m": m, "d": d,
        "meta": {"question": question, "gt": gt, "anchor": anchor,
                 "dataset": dataset, "target_image_path": target},
    }


def test_eligible_samples_require_all_5_models_all_4_conditions():
    by_model = {mid: {} for mid in bdd.MAIN_PANEL}
    # S1 has all 5 models and all 4 conditions
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["S1"] = _make_sample(4, 5, 4, 4, gt=4, anchor=5)
    # S2 has only 4 models
    for mid in list(bdd.MAIN_PANEL)[:4]:
        by_model[mid]["S2"] = _make_sample(4, 5, 4, 4, gt=4, anchor=5)
    # S3 has all models but missing the m condition for one
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["S3"] = _make_sample(4, 5, 4, 4, gt=4, anchor=5)
    by_model["qwen2.5-vl-7b"]["S3"].pop("m")
    eligible = bdd.eligible_samples(by_model)
    assert eligible == ["S1"]


def test_score_sample_rewards_correct_base_and_anchor_pull():
    by_model = {mid: {} for mid in bdd.MAIN_PANEL}
    # SAMPLE A: 5 correct on b, 5 pulled to anchor on a, 5 recover on m
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["A"] = _make_sample(4, 5, 4, 4, gt=4, anchor=5)
    # SAMPLE B: 0 correct on b, 0 pulled, 0 recover
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["B"] = _make_sample(7, 7, 7, 7, gt=4, anchor=5)
    score_a = bdd.score_sample(by_model, "A")
    score_b = bdd.score_sample(by_model, "B")
    assert score_a > score_b
    assert score_a > 0


def test_score_sample_prefers_full_trajectory_over_clean_base_only():
    """A 'b correct, a pulled, m recovered' signature beats 'b correct everywhere'.

    Without the trajectory weighting the picker loved samples where every
    model was simply correct on every condition (no anchoring effect at
    all) — useless for an anchoring demo.
    """
    by_model = {mid: {} for mid in bdd.MAIN_PANEL}
    # FULL: every model walks b=4 → a=5 (anchor) → m=4 (recover) → d=4
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["FULL"] = _make_sample(4, 5, 4, 4, gt=4, anchor=5)
    # CLEAN: every model just stays at gt across all conditions (no anchor effect)
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["CLEAN"] = _make_sample(4, 4, 4, 4, gt=4, anchor=5)
    score_full = bdd.score_sample(by_model, "FULL")
    score_clean = bdd.score_sample(by_model, "CLEAN")
    assert score_full > score_clean


import json

from PIL import Image


def _write_image(path: Path, color: tuple[int, int, int], size=(64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_pick_samples_prefers_dataset_diversity():
    samples = {
        "S1": ("VQAv2", 10.0),
        "S2": ("VQAv2", 9.5),
        "S3": ("ChartQA", 9.0),
        "S4": ("ChartQA", 8.5),
        "S5": ("PlotQA", 8.0),
        "S6": ("MathVista", 7.5),
        "S7": ("VQAv2", 7.0),
    }
    chosen = bdd.pick_samples(samples, n=4)
    # Top-1 must be the highest scorer; subsequent picks should diversify.
    assert chosen[0] == "S1"
    chosen_datasets = {samples[sid][0] for sid in chosen}
    assert len(chosen_datasets) >= 3


def test_build_writes_demo_json_and_images(tmp_path):
    site = tmp_path / "site"
    inputs = tmp_path / "inputs"
    target = inputs / "vqav2_number_val" / "images" / "x.jpg"
    anchor = inputs / "irrelevant_number" / "5.png"
    masked = inputs / "irrelevant_number_masked" / "5.png"
    neutral = inputs / "irrelevant_neutral" / "0.png"
    _write_image(target, (255, 0, 0))
    _write_image(anchor, (0, 255, 0))
    _write_image(masked, (0, 0, 255))
    _write_image(neutral, (128, 128, 128))

    by_model = {mid: {} for mid in bdd.MAIN_PANEL}
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["S1"] = _make_sample(
            4, 5, 4, 4, gt=4, anchor=5, dataset="VQAv2",
            target=str(target),
        )

    chosen = ["S1"]
    bdd.build_site_artifacts(
        chosen=chosen, by_model=by_model,
        inputs_root=inputs, site_root=site,
        anchor_stratum="S1", max_image_px=32,
    )
    demo_json = json.loads((site / "data" / "demo.json").read_text())
    assert {m["id"] for m in demo_json["models"]} == set(bdd.MAIN_PANEL)
    s1 = demo_json["samples"][0]
    assert s1["id"] == "S1"
    assert s1["dataset"] == "VQAv2"
    assert s1["predictions"]["llava-onevision-7b"] == {"b": 4, "a": 5, "m": 4, "d": 4}
    for kind in ("target", "anchor", "masked", "neutral"):
        rel = s1["images"][kind]
        assert (site / rel).exists()


def test_eligible_samples_drops_degenerate_anchor_equal_gt():
    """Samples where anchor == gt are uninformative for an anchoring demo."""
    by_model = {mid: {} for mid in bdd.MAIN_PANEL}
    # S1: anchor != gt → eligible
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["S1"] = _make_sample(4, 5, 4, 4, gt=4, anchor=5)
    # S2: anchor == gt → degenerate, should be dropped
    for mid in bdd.MAIN_PANEL:
        by_model[mid]["S2"] = _make_sample(7, 7, 7, 7, gt=7, anchor=7)
    eligible = bdd.eligible_samples(by_model)
    assert eligible == ["S1"]


def test_first_path_handles_double_quoted_json_arrays():
    """Live CSVs serialize input_image_paths with double quotes (JSON)."""
    real = '["/mnt/abs/inputs/plotqa_test/images/000000031236.png", "/mnt/abs/inputs/irrelevant_number/300.png"]'
    assert bdd._first_path(real) == "/mnt/abs/inputs/plotqa_test/images/000000031236.png"


def test_first_path_handles_single_quoted_python_repr():
    """Synthetic / legacy rows use Python repr() with single quotes."""
    legacy = "['/abs/inputs/vqav2_number_val/images/000000000136.jpg']"
    assert bdd._first_path(legacy) == "/abs/inputs/vqav2_number_val/images/000000000136.jpg"
