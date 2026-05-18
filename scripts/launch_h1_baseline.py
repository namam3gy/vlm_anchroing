"""Option H1 baseline launcher — 6 models x 5 datasets, GPU-pinned.

Spawns one background process per GPU. Each GPU is assigned a list of
(model, dataset) cells; cells run sequentially within a GPU. Marker
file at `outputs/paper2/cross_model_cross_dataset/predictions/<dataset>/
<model>/_done.marker` makes a re-run skip finished cells.

Predictions land at
`outputs/paper2/cross_model_cross_dataset/predictions/<dataset>/<model>/`
matching the layout `notebooks/paper_cross_model_cross_dataset.ipynb` +
`scripts/_build_paper_main_panel_notebook.py` read from.

Datasets executed small -> large so prompt regressions surface in
~30min on MathVista rather than ~5h into TallyQA.

Usage:
    uv run python scripts/launch_h1_baseline.py
Watch logs in outputs/paper2/_logs/<gpu>_<model>.log; resume with the
same command after a crash.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "outputs" / "paper2" / "cross_model_cross_dataset"
LOG_DIR = REPO / "outputs" / "paper2" / "_logs"

# 5 datasets x config slug used by paper main-panel notebook.
DATASETS = [
    # (slug, config)        — order = MathVista -> ChartQA -> InfoVQA -> PlotQA -> TallyQA
    ("mathvista",       "configs/experiment_e5e_mathvista_full.yaml"),
    ("chartqa",         "configs/experiment_e5e_chartqa_full.yaml"),
    ("infographicvqa",  "configs/experiment_e7_infographicvqa_full.yaml"),
    ("plotqa",          "configs/experiment_e7_plotqa_full.yaml"),
    ("tallyqa",         "configs/experiment_e5e_tallyqa_full.yaml"),
]

# GPU assignment (5 GPUs, 6 models; GPU 0 takes the two fastest sequentially)
# Each list value: (model_name, hf_model_id)
GPU_PLAN: dict[int, list[tuple[str, str]]] = {
    0: [
        ("gemma3-4b-it",              "google/gemma-3-4b-it"),
        ("llava-next-interleaved-7b", "llava-hf/llava-interleave-qwen-7b-hf"),
    ],
    1: [("qwen2.5-vl-7b-instruct",     "Qwen/Qwen2.5-VL-7B-Instruct")],
    2: [("qwen2.5-vl-32b-instruct",    "Qwen/Qwen2.5-VL-32B-Instruct")],
    3: [("gemma3-27b-it",              "google/gemma-3-27b-it")],
    4: [("llava-onevision-qwen2-7b-ov","llava-hf/llava-onevision-qwen2-7b-ov-hf")],
}


def cell_done(dataset_slug: str, model_name: str) -> bool:
    out_dir = OUT_ROOT / "predictions" / dataset_slug / model_name
    return (out_dir / "_done.marker").exists()


def run_cell(dataset_slug: str, config: str, model_name: str, hf_model: str, gpu: int) -> bool:
    out_dir = OUT_ROOT / "predictions" / dataset_slug / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "_done.marker"
    log_path = LOG_DIR / f"gpu{gpu}_{model_name}_{dataset_slug}.log"

    # Use the worktree's venv directly — skip `uv run` re-resolution cost
    # (~30s per cell × 30 cells = ~15min saved).
    py = str(REPO / ".venv" / "bin" / "python")
    cmd = [
        py, "scripts/run_experiment.py",
        "--config", config,
        "--models", model_name,
        "--output-dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[GPU{gpu}] {model_name}/{dataset_slug} starting -> {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w") as fp:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fp, stderr=subprocess.STDOUT)
    dur_min = (time.time() - t0) / 60.0
    if proc.returncode == 0:
        marker.touch()
        print(f"[GPU{gpu}] {model_name}/{dataset_slug} done ({dur_min:.1f} min)", flush=True)
        return True
    print(f"[GPU{gpu}] {model_name}/{dataset_slug} FAILED rc={proc.returncode} ({dur_min:.1f} min) — log {log_path}", flush=True)
    return False


def worker(gpu: int) -> None:
    """Sequentially handle all (model, dataset) cells assigned to this GPU."""
    for model_name, hf_model in GPU_PLAN[gpu]:
        for slug, config in DATASETS:
            if cell_done(slug, model_name):
                print(f"[GPU{gpu}] {model_name}/{slug} skip (marker)", flush=True)
                continue
            ok = run_cell(slug, config, model_name, hf_model, gpu)
            if not ok:
                print(f"[GPU{gpu}] giving up on {model_name} after failure", flush=True)
                break


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    n_total = sum(len(models) for models in GPU_PLAN.values()) * len(DATASETS)
    n_done = sum(
        1 for gpu in GPU_PLAN for model_name, _ in GPU_PLAN[gpu]
        for slug, _ in DATASETS if cell_done(slug, model_name)
    )
    print(f"Plan: {n_total} cells, {n_done} already complete (will skip)")

    if "--single-gpu" in sys.argv:
        gpu = int(sys.argv[sys.argv.index("--single-gpu") + 1])
        worker(gpu)
        return

    # Fork-and-wait: one child per GPU. Stdin/stdout inherited so log lines
    # interleave at the parent's terminal; per-cell verbose output stays in
    # the dedicated per-cell log file under outputs/paper2/_logs/.
    children: list[int] = []
    for gpu in sorted(GPU_PLAN):
        pid = os.fork()
        if pid == 0:
            try:
                worker(gpu)
            finally:
                os._exit(0)
        else:
            children.append(pid)
    for pid in children:
        os.waitpid(pid, 0)
    print("All GPUs done.")


if __name__ == "__main__":
    main()
