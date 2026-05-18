"""Option H1 baseline launcher — job-queue dispatch across N GPUs.

Replaces the earlier per-GPU model-pin scheme (which left small-model
GPUs idle for hours while the 32B model finished). Cells are pushed
into a queue, workers (one per GPU) pull whatever is available next.
Heavy cells dispatched first so small-cell workers keep churning while
the big ones run.

Each "job" = (model, dataset) pair. Predictions land at
`outputs/paper2/cross_model_cross_dataset/predictions/<dataset>/<model>/`
matching the layout the `paper_cross_model_cross_dataset.ipynb`
reproducer reads.

Marker resume: cells with `_done.marker` are skipped — works after
SIGINT, OOM, or container swap.

Usage:
    .venv/bin/python scripts/launch_h1_baseline.py            # autodetect GPU count
    .venv/bin/python scripts/launch_h1_baseline.py --gpus 8   # force GPU count
    .venv/bin/python scripts/launch_h1_baseline.py --dry-run  # plan only, no execution

Per-cell logs at `outputs/paper2/_logs/gpu{g}_{model}_{dataset}.log`.
"""
from __future__ import annotations
import argparse
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "outputs" / "paper2" / "cross_model_cross_dataset"
LOG_DIR = REPO / "outputs" / "paper2" / "_logs"
PY = str(REPO / ".venv" / "bin" / "python")

# (slug, config) — dataset order is fail-fast (smallest first) so prompt
# regressions surface within minutes rather than hours.
DATASETS = [
    ("mathvista",       "configs/experiment_e5e_mathvista_full.yaml"),
    ("chartqa",         "configs/experiment_e5e_chartqa_full.yaml"),
    ("infographicvqa",  "configs/experiment_e7_infographicvqa_full.yaml"),
    ("plotqa",          "configs/experiment_e7_plotqa_full.yaml"),
    ("tallyqa",         "configs/experiment_e5e_tallyqa_full.yaml"),  # cap 5000
]

# (model_name, hf_id, rough_param_count_billions)
MODELS = [
    ("gemma3-4b-it",               "google/gemma-3-4b-it",                  4),
    ("llava-onevision-qwen2-7b-ov","llava-hf/llava-onevision-qwen2-7b-ov-hf", 7),
    ("llava-next-interleaved-7b",  "llava-hf/llava-interleave-qwen-7b-hf",  7),
    ("qwen2.5-vl-7b-instruct",     "Qwen/Qwen2.5-VL-7B-Instruct",           7),
    ("gemma3-27b-it",              "google/gemma-3-27b-it",                27),
    ("qwen2.5-vl-32b-instruct",    "Qwen/Qwen2.5-VL-32B-Instruct",         32),
]

# Approximate per-cell sample count (after the 4-condition fan-out).
# Used only to sort the queue heavy-first.
DATASET_N_SAMPLES = {
    "mathvista":      385,
    "chartqa":        705,
    "infographicvqa": 1147,
    "plotqa":         5000,
    "tallyqa":        5000,
}


@dataclass
class Job:
    model: str
    hf_model: str
    params_b: int
    dataset: str
    config: str

    def weight(self) -> int:
        """Heuristic time order — larger value = run earlier in the queue."""
        return self.params_b * DATASET_N_SAMPLES[self.dataset]

    def out_dir(self) -> Path:
        return OUT_ROOT / "predictions" / self.dataset / self.model

    def marker(self) -> Path:
        return self.out_dir() / "_done.marker"

    def log_path(self, gpu: int) -> Path:
        return LOG_DIR / f"gpu{gpu}_{self.model}_{self.dataset}.log"

    def __str__(self) -> str:
        return f"{self.model}/{self.dataset}"


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for slug, config in DATASETS:
        for model_name, hf_id, params_b in MODELS:
            jobs.append(Job(model_name, hf_id, params_b, slug, config))
    return jobs


def cell_done(job: Job) -> bool:
    return job.marker().exists()


def run_cell(job: Job, gpu: int) -> bool:
    out_dir = job.out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.log_path(gpu)

    cmd = [
        PY, "scripts/run_experiment.py",
        "--config", job.config,
        "--models", job.model,
        "--output-dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Enable DataLoader prefetch — overlaps CPU input prep with GPU
    # generate (HFAttentionRunner exposes the split prepare/generate API,
    # which all 6 H1 panel models use). ~25 % wall-time gain.
    env["VLM_ENABLE_PREFETCH"] = "1"

    print(f"[GPU{gpu}] {job} starting -> {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w") as fp:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fp, stderr=subprocess.STDOUT)
    dur_min = (time.time() - t0) / 60.0
    if proc.returncode == 0:
        job.marker().touch()
        print(f"[GPU{gpu}] {job} done ({dur_min:.1f} min)", flush=True)
        return True
    print(f"[GPU{gpu}] {job} FAILED rc={proc.returncode} ({dur_min:.1f} min) — log {log_path}", flush=True)
    return False


def worker(gpu: int, q: "queue.Queue[Job]", stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            if cell_done(job):
                print(f"[GPU{gpu}] {job} skip (marker)", flush=True)
            else:
                run_cell(job, gpu)
        finally:
            q.task_done()


def detect_in_flight_cells() -> set[tuple[str, str]]:
    """Return {(model, dataset_slug), ...} for cells already running in
    another launcher process. Used to avoid double-dispatching a cell
    when a second launcher is started in parallel (e.g., to retry
    failed cells while the original launcher's long cells finish)."""
    try:
        ps = subprocess.run(["ps", "-ef"], check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    in_flight: set[tuple[str, str]] = set()
    for line in ps.splitlines():
        if "scripts/run_experiment.py" not in line or "grep" in line:
            continue
        parts = line.split()
        model = config = None
        for i, tok in enumerate(parts):
            if tok == "--models" and i + 1 < len(parts):
                model = parts[i + 1]
            elif tok == "--config" and i + 1 < len(parts):
                config = parts[i + 1]
        if not (model and config):
            continue
        # Derive dataset slug from `configs/experiment_*_full.yaml`
        cfg_base = Path(config).stem.replace("experiment_", "").replace("_full", "")
        for slug, _ in DATASETS:
            if slug in cfg_base:
                in_flight.add((model, slug))
                break
    return in_flight


def detect_gpu_count() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return len([line for line in out.splitlines() if line.strip()])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: autodetect via nvidia-smi)")
    ap.add_argument("--gpus-list", type=str, default=None,
                    help="Explicit comma-separated GPU indices to use, e.g. '1,3,6'. "
                         "Overrides --gpus. Useful when other GPUs are already busy "
                         "with a different launcher run.")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, don't execute")
    ap.add_argument("--light-first", action="store_true",
                    help="Run smallest cells first instead of heaviest. Use when an interruption "
                         "is imminent (pod swap, etc.) so you bank cheap marker wins before kill.")
    args = ap.parse_args()

    if args.gpus_list is not None:
        gpu_indices = [int(g.strip()) for g in args.gpus_list.split(",") if g.strip()]
    else:
        n_gpus = args.gpus if args.gpus is not None else detect_gpu_count()
        gpu_indices = list(range(n_gpus))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_jobs = build_jobs()
    if args.light_first:
        # Smallest first — bank marker wins before an imminent interruption.
        all_jobs.sort(key=lambda j: j.weight())
    else:
        # Heaviest first to fill GPU slots early so small-model GPUs are not stranded.
        all_jobs.sort(key=lambda j: -j.weight())

    in_flight = detect_in_flight_cells()
    pending = [
        j for j in all_jobs
        if not cell_done(j) and (j.model, j.dataset) not in in_flight
    ]
    done = [j for j in all_jobs if cell_done(j)]
    print(f"Plan: {len(all_jobs)} cells, {len(done)} done, "
          f"{len(in_flight)} in-flight elsewhere, {len(pending)} pending")
    if in_flight:
        print("Skipping (already running in another launcher):")
        for m, d in sorted(in_flight):
            print(f"  {m}/{d}")
    print(f"GPUs: {gpu_indices}")
    print()
    print("Queue order (heavy first):")
    for j in pending[:15]:
        print(f"  weight={j.weight():>6}  {j}")
    if len(pending) > 15:
        print(f"  ... +{len(pending) - 15} more")

    if args.dry_run or not pending:
        return

    q: "queue.Queue[Job]" = queue.Queue()
    for j in pending:
        q.put(j)

    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for gpu in gpu_indices:
        t = threading.Thread(target=worker, args=(gpu, q, stop_event), name=f"gpu{gpu}", daemon=False)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupted — stop signaling workers (in-flight cells will complete current run_experiment subprocess)")
        stop_event.set()
        for t in threads:
            t.join()

    n_done_final = sum(1 for j in all_jobs if cell_done(j))
    print(f"\nFinal: {n_done_final}/{len(all_jobs)} cells complete.")


if __name__ == "__main__":
    main()
