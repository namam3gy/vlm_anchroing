"""Build outputs/paper2/cross_model_cross_dataset/manifest.json for H1.

`paper_cross_model_cross_dataset.ipynb` reads this manifest at L451 for
the generation metadata + hyperparameters block (system / user prompt,
sampling, dataset filters, anchor selection) AND for the per-cell
records (source_exp_dir / source_run / n_b_eligible / sha256 prefix).

Starts from the legacy `outputs/paper/.../manifest.json` schema, then
patches:
  - hyperparameters.prompt_system / prompt_user_template → Candidate A
  - dataset_filters.TallyQA.n_eligible: 38245 → 5000 (H1 cap)
  - canonical_source path label stays the same
  - cells[]: regenerated from the actual outputs/paper2/ predictions

Usage (after Stage 2 fully complete, 30/30 markers present):
  .venv/bin/python scripts/_build_h1_manifest.py
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO.parent.parent.parent / "outputs" / "paper" / "cross_model_cross_dataset" / "manifest.json"
H1_ROOT = REPO / "outputs" / "paper2" / "cross_model_cross_dataset"
H1_PRED = H1_ROOT / "predictions"
H1_MANIFEST = H1_ROOT / "manifest.json"

# Dataset slug → display (matches notebook expectations).
DATASETS = [
    ("tallyqa", "TallyQA",        "experiment_e5e_tallyqa_full"),
    ("chartqa", "ChartQA",        "experiment_e5e_chartqa_full"),
    ("mathvista", "MathVista",    "experiment_e5e_mathvista_full"),
    ("plotqa", "PlotQA",          "experiment_e7_plotqa_full"),
    ("infographicvqa", "InfographicVQA", "experiment_e7_infographicvqa_full"),
]
MODELS = [
    "llava-onevision-qwen2-7b-ov",
    "llava-next-interleaved-7b",
    "qwen2.5-vl-7b-instruct",
    "qwen2.5-vl-32b-instruct",
    "gemma3-4b-it",
    "gemma3-27b-it",
]


def sha256_prefix(path: Path, nbytes: int = 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[: nbytes * 2]


def count_base_rows(jsonl: Path) -> int:
    n = 0
    seen = set()
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("condition") == "target_only":
                sid = rec.get("sample_instance_id")
                if sid and sid not in seen:
                    seen.add(sid)
                    n += 1
    return n


def main() -> None:
    # Start from legacy manifest as template — patches replace H1-affected fields.
    if not LEGACY.exists():
        raise SystemExit(f"legacy manifest not found at {LEGACY}")
    m = json.loads(LEGACY.read_text())

    # Top-level metadata patches.
    m["purpose"] = (
        "Paper §4 main panel (cross-model × cross-dataset) — H1 raw-number "
        "prompt + DF eps=0 form. Source of truth for outline Appendix D.1/D.2."
    )
    m["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    m["spec_reference"] = (
        "docs/paper/emnlp_outline_ko.md §3.4 (panel) + §B (filters) + §C "
        "(anchors) + §A.1 (prompt, H1 raw-number) + [[option-h1-prompt-df-full-rerun]]"
    )

    # Hyperparameters → Candidate A prompt.
    m["hyperparameters"]["prompt_system"] = (
        "You are a visual question answering system.\n"
        "Output a single number only. No words, no units, no explanation, no markdown.\n"
        "If uncertain, still output the single most likely number."
    )
    m["hyperparameters"]["prompt_user_template"] = (
        "Answer the question using the provided image(s).\n"
        "Output a single number.\n"
        "Question: {question}"
    )

    # TallyQA cap.
    m["dataset_filters"]["TallyQA"]["n_eligible"] = 5000
    m["dataset_filters"]["TallyQA"]["note"] = (
        "H1 cap: first 5,000 in questions.jsonl order (audit receipt at "
        "docs/insights/_data/tallyqa_5k_sids.json)"
    )

    # Metrics canonical → eps=0.
    if "metrics_canonical" in m:
        m["metrics_canonical"]["direction_follow"] = (
            "DF_a = P[(pa - pb)(z - pb) > 0 | pb != z] "
            "(eps=0, strict >, excludes pb==z rows from denominator)"
        )

    # Regenerate cells from H1 predictions.
    cells: list[dict] = []
    missing: list[str] = []
    for slug, display, exp_dir in DATASETS:
        for model in MODELS:
            cell_dir = H1_PRED / slug / model
            jsonl = cell_dir / "predictions.jsonl"
            if not jsonl.exists():
                missing.append(f"{display}/{model}")
                continue
            n_b = count_base_rows(jsonl)
            cells.append({
                "dataset": display,
                "dataset_slug": slug,
                "model": model,
                "source_exp_dir": exp_dir,
                "source_run": "h1",  # H1 doesn't use timestamped run dirs
                "predictions_size_bytes": jsonl.stat().st_size,
                "predictions_sha256_prefix": sha256_prefix(jsonl),
                "n_b_eligible": n_b,
            })
    m["cells"] = cells

    if missing:
        print(f"[warn] {len(missing)} cells missing predictions.jsonl:")
        for x in missing:
            print(f"  - {x}")

    H1_ROOT.mkdir(parents=True, exist_ok=True)
    H1_MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {H1_MANIFEST.relative_to(REPO)}")
    print(f"  cells: {len(cells)}/30")
    print(f"  generated_at: {m['generated_at']}")


if __name__ == "__main__":
    main()
