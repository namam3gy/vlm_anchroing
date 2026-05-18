"""Build the three §5 reproducibility notebooks.

Outputs:
  notebooks/paper_section_5_1_attention_peaks.ipynb
  notebooks/paper_section_5_2_subspace_sweep.ipynb
  notebooks/paper_section_5_3_routing_integration.ipynb

§5.1 — re-runs attention extraction across the 5-model mechanism panel
on PlotQA + TallyQA + InfoVQA, then aggregates the per-(model, dataset)
peak layer table and renders the §5.1 figure.

§5.2 — calibrates the (a − m) subspace on PlotQA + InfoVQA pooled
(OneVision Main), then sweeps the pilot (L × α × K) grid on PlotQA and
the 5-dataset layer sweep at L=26 (K=8 vs K=1 fallback).

§5.3 — narrative synthesis only; reads §5.1 / §5.2 outputs and lays out
the routing-and-integration framework.

All three notebooks share the same `setup` cells (paths, GPU sharding,
subprocess helper) so the layout is symmetric with the §4 reproducers.
"""
from __future__ import annotations
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(s: str) -> dict:
    return nbf.v4.new_markdown_cell(s.lstrip("\n"))


def code(s: str) -> dict:
    return nbf.v4.new_code_cell(s.lstrip("\n"))


def _setup_cells(title: str, subtitle: str, scope_doc: str) -> list[dict]:
    return [
        md(rf"""
# {title}

{subtitle}

**Spec source-of-truth.** `docs/paper/emnlp_outline_ko.md` — {scope_doc}.

This notebook drives the heavy inference stages by `subprocess`-invoking
the existing sharded drivers in `scripts/`. The `RUN_INFERENCE = False`
toggle below lets a reviewer read the entire pipeline without GPU
access. Full reproduction targets the **8 × H200** cluster and uses
`--gpus 0,1,2,3,4,5,6,7` end-to-end.
"""),
        md("## 1 · Setup — paths, GPU sharding, subprocess helper"),
        code(r"""
from __future__ import annotations
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_main_worktree() -> Path:
    # Gitignored artifacts (inputs/, outputs/, docs/insights/_data/) live in
    # the main worktree even when this notebook runs from a linked worktree.
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], cwd=Path.cwd(), text=True
    ).strip()
    return Path(common).resolve().parent


def find_worktree_root() -> Path:
    # Current worktree's working tree top (== main when not in a linked worktree).
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), text=True
    ).strip()).resolve()


MAIN     = find_main_worktree()
WORKTREE = find_worktree_root()

# Outputs live under MAIN (gitignored); figures land under WORKTREE so they ride the active branch.
# Scripts + configs come from WORKTREE so the active branch's edits are used
# (subprocess invocations would otherwise hit the stale main-worktree copies
# until the active PR merges).
SCRIPTS    = WORKTREE / "scripts"
CONFIGS    = WORKTREE / "configs"
DATA_DIR   = MAIN / "docs" / "insights" / "_data"
PRED_ROOT  = MAIN / "outputs" / "paper2" / "cross_model_cross_dataset" / "predictions"

# §5.2 e6_steering input root selection (same toggle as §5.1 above):
#   - RUN_INFERENCE=False: read pre-existing sweep dirs from legacy
#     `outputs/e6_steering/` (the aggregators have always filtered by
#     subdirectory name, so the legacy tree won't pollute results even
#     when read-only).
#   - RUN_INFERENCE=True: write new calibrate / sweep dirs to an isolated
#     tree so this run doesn't commingle with the legacy pool.
E6_ROOT_LEGACY = MAIN / "outputs" / "e6_steering"
E6_ROOT_FRESH  = MAIN / "outputs" / "paper2" / "section_5_e6_steering"

# §5.1 attention input root selection:
#   - RUN_INFERENCE=False: read pre-existing bbox-with runs from
#     legacy outputs/attention_analysis/ (the docstring header in
#     `extract_attention_mass.py` confirms each model has multiple
#     timestamped runs carrying `image_anchor_digit`; the analyzer
#     auto-skips bbox-less records via field-presence check).
#   - RUN_INFERENCE=True: write new runs to a fresh isolated tree so
#     they do not commingle with the legacy pool.
ATT_ROOT_LEGACY = MAIN / "outputs" / "attention_analysis"
# `section_5_attention` is the n=400 root (n=400 spec was the first run).
# `section_5_attention_n1000` is the n=1000 extension. After n=1000 completes,
# n=400 is to be retired and only the n=1000 tree kept.
ATT_ROOT_FRESH  = MAIN / "outputs" / "paper2" / "section_5_attention_n1000"
PEAKS_CSV       = MAIN / "outputs" / "paper2" / "section_5_attention_n1000" / "_data" / "cross_dataset_peaks.csv"
BBOX_FILE       = MAIN / "inputs" / "irrelevant_number_bboxes.json"

ATT_ROOT_FRESH.mkdir(parents=True, exist_ok=True)
PEAKS_CSV.parent.mkdir(parents=True, exist_ok=True)
assert BBOX_FILE.exists(), f"missing digit-pixel bbox JSON: {BBOX_FILE}"

PDF_OUT = MAIN     / "outputs" / "paper2" / "section_5_figures"
PNG_OUT = WORKTREE / "docs"    / "figures"
PDF_OUT.mkdir(parents=True, exist_ok=True)
PNG_OUT.mkdir(parents=True, exist_ok=True)

GPUS = os.environ.get("VLM_ANCHOR_GPUS", "0,1,2,3,4")  # 5 GPUs by default
RUN_INFERENCE = False  # set True to invoke the heavy sharded drivers.

# Pick the attention + e6_steering input roots.
# `ATT_ROOT_FRESH` (section_5_attention_n1000) holds the canonical §5.1
# inference output, so always read from it — the RUN_INFERENCE toggle
# only gates whether a new launch script *adds* to that tree.
ATT_ROOT = ATT_ROOT_FRESH
E6_ROOT  = E6_ROOT_FRESH if RUN_INFERENCE else E6_ROOT_LEGACY
E6_ROOT_FRESH.mkdir(parents=True, exist_ok=True)

print(f"MAIN     = {MAIN}")
print(f"WORKTREE = {WORKTREE}")
print(f"GPUS     = {GPUS}")
print(f"RUN_INFERENCE = {RUN_INFERENCE}")
"""),
        code(r"""
def run_cmd(cmd: list[str] | str, *, dry: bool = False, env: dict | None = None) -> int:
    # Print and (optionally) execute a shell command from the main worktree.
    printable = " ".join(shlex.quote(c) for c in cmd) if isinstance(cmd, list) else cmd
    print(f"$ {printable}")
    if dry:
        print("  (dry — RUN_INFERENCE=False)")
        return 0
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(cmd, cwd=MAIN, env=full_env,
                          shell=isinstance(cmd, str)).returncode


def save_figure(fig, stem: str):
    pdf = PDF_OUT / f"{stem}.pdf"
    png = PNG_OUT / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
"""),
    ]


# =============================================================================
# Notebook 1 — §5.1 Attention peaks
# =============================================================================

cells_51: list[dict] = _setup_cells(
    title="Paper §5.1 — Layer-wise attention peaks",
    subtitle=(
        "Reproduces the per-`(model, dataset)` attention peak layer table that "
        "underpins paper §5.1 (Per-model peak heterogeneity) and Appendix G."
    ),
    scope_doc="§5.1 (Layer-wise probes)",
)

cells_51 += [
    md(r"""
## 2 · Mechanism panel + datasets

§5.1 reports the **digit-pixel region** attention peak per
`(model, dataset)` — i.e., the layer at which the model's
text→second-image attention concentrates *on the digit-pixel patches
of the anchor*, not on the full anchor image span. The bbox of the
digit pixels is taken from `inputs/irrelevant_number_bboxes.json`
(produced once by `scripts/compute_anchor_digit_bboxes.py`).

Inside `extract_attention_mass.py --bbox-file ...` each per-step record
gains an `image_anchor_digit` field (per-layer attention mass restricted
to the bbox patches); `analyze_cross_dataset_peaks.py --region
image_anchor_digit` argmaxes the layer over the (anchor − neutral)
delta on that field.

**Panel (5 models, 5-GPU cluster):**

| Model | HF id | Encoder family |
|---|---|---|
| Gemma3-4b           | `google/gemma-3-4b-it`                    | SigLIP2 + Gemma (42L)  |
| Qwen2.5-VL-7b       | `Qwen/Qwen2.5-VL-7B-Instruct`             | window-attn ViT + Qwen2 (28L) |
| LLaVA-OneVision-7b  | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | AnyRes-CLIP + Qwen2 (28L) |
| Gemma3-27b          | `google/gemma-3-27b-it`                   | SigLIP2 + Gemma (~46L) |
| Qwen2.5-VL-32b      | `Qwen/Qwen2.5-VL-32B-Instruct`            | window-attn ViT + Qwen2 (~64L) |

LLaVA-Interleave is run separately (its multi-image AnyRes span is not
perfect-square for the current bbox routing, requiring a follow-up
patch). All five panel models are pre-verified to emit
`image_anchor_digit` under the default perfect-square assumption.

**Datasets (5, in execution order):**

PlotQA → InfoVQA → ChartQA → MathVista → TallyQA.

Each model runs all 5 datasets sequentially within its dedicated GPU;
the 5 models run in parallel across the 5 GPUs.

Hyperparameters identical to §4: `seed=42`, greedy decoding,
`max_new_tokens=16` (TallyQA: 8). §5.1 additionally forces
`attn_implementation=eager` so the attention matrix is materialized
for `extract_attention_mass.py`. `n=400 samples` per cell.

**Resumable:** a marker file
`<ATT_ROOT_FRESH>/<model>/_done_<dataset>.marker` is touched after a
successful extract. If the pod restarts, re-running the launcher
skips completed cells via the marker check.
"""),
    code(r"""
MECH_PANEL = [
    # (label, internal name, HF id, attn_impl)
    ("Gemma3-4b",         "gemma3-4b-it",                "google/gemma-3-4b-it",                    "eager"),
    ("Qwen2.5-VL-7b",     "qwen2.5-vl-7b-instruct",      "Qwen/Qwen2.5-VL-7B-Instruct",             "eager"),
    ("Interleave-7b",     "llava-next-interleaved-7b",   "llava-hf/llava-interleave-qwen-7b-hf",    "eager"),
    ("OneVision-7b",      "llava-onevision-qwen2-7b-ov", "llava-hf/llava-onevision-qwen2-7b-ov-hf", "eager"),
    ("Gemma3-27b",        "gemma3-27b-it",               "google/gemma-3-27b-it",                   "eager"),
    ("Qwen2.5-VL-32b",    "qwen2.5-vl-32b-instruct",     "Qwen/Qwen2.5-VL-32B-Instruct",            "eager"),
]

# (dataset_tag, config_slug, susceptibility_csv) — per-dataset susceptibility CSV
# tells `extract_attention_mass.py` which question_ids to sample (top-decile
# susceptible + bottom-decile resistant); without an explicit path it would
# fall back to the VQAv2 default. Executed in this order per model.
PEAK_DATASETS = [
    ("plotqa",     "experiment_e7_plotqa_full",         "susceptibility_plotqa_onevision.csv"),
    ("infovqa",    "experiment_e7_infographicvqa_full", "susceptibility_infovqa_onevision.csv"),
    ("chartqa",    "experiment_e5e_chartqa_full",       "susceptibility_chartqa_onevision.csv"),
    ("mathvista",  "experiment_e5e_mathvista_full",     "susceptibility_mathvista_onevision.csv"),
    ("tallyqa",    "experiment_e5e_tallyqa_full",       "susceptibility_tallyqa_onevision.csv"),
]

N_PER_CELL = 1000   # top-decile 500 + bottom-decile 500 (n=400 extended)
ONEVISION  = "llava-onevision-qwen2-7b-ov"

for label, name, hf, attn in MECH_PANEL:
    print(f"  {label:<18} → {name:<32}  (attn={attn})")
print()
print(f"Datasets: {', '.join(d for d, _, _ in PEAK_DATASETS)}")
print(f"n_per_cell = {N_PER_CELL}")
print(f"5 models × 5 datasets = 25 cells; resumable via marker files.")
"""),
    md(r"""
## 3 · Extract attention mass — resumable 5-GPU launcher

The 25 cells (5 models × 5 datasets) run as:
- **5 parallel processes**, one per model, each pinned to a single GPU
  via `CUDA_VISIBLE_DEVICES`.
- **Within a model**, the 5 datasets run sequentially in PlotQA →
  InfoVQA → ChartQA → MathVista → TallyQA order.
- **Marker-based resume**: each cell touches
  `<ATT_ROOT_FRESH>/<model>/_done_<dataset>.marker` on success. The
  launcher always re-checks the marker before running, so restarting
  after a pod crash picks up exactly where it left off.

The next cell **writes** the launch script and prints the invocation
commands. Run the script under tmux/screen (or as a nohup background
job) so the inference survives an interactive disconnect.
"""),
    code(r"""
LAUNCH_DIR    = ATT_ROOT_FRESH
LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
LAUNCH_SCRIPT = LAUNCH_DIR / "_launch_section_5_1.sh"
LAUNCH_LOG    = LAUNCH_DIR / "_launch_section_5_1.log"


def _build_launch_script() -> str:
    gpus = [g.strip() for g in GPUS.split(",") if g.strip()]
    # Round-robin GPU assignment when the panel grows past the GPU count
    # (e.g., 6-model panel on a 5-GPU cluster). The script still runs each
    # model loop in the background; the rotated GPU just shares with another
    # model. Marker-based resume + per-model logs handle the overlap.

    header = [
        "#!/usr/bin/env bash",
        "# Auto-generated by paper_section_5_1_attention_peaks.ipynb",
        "# Each model runs in the background on its own GPU. Marker-based",
        "# resume: re-running this script after a crash skips finished cells.",
        "set -uo pipefail",
        f"cd {MAIN}",
        "",
        "PIDS=()",
    ]

    bodies = []
    for i, (label, name, hf, attn) in enumerate(MECH_PANEL):
        gpu = gpus[i % len(gpus)]
        marker_dir = ATT_ROOT_FRESH / name
        log = LAUNCH_DIR / f"_log_{name}.txt"
        per_cell = []
        for ds_tag, cfg_slug, susc_csv in PEAK_DATASETS:
            marker = marker_dir / f"_done_{ds_tag}.marker"
            susc_path = DATA_DIR / susc_csv
            # extract_attention_mass.py auto-loads eager via EagerAttentionRunner
            # (no --attn-implementation flag). --susceptibility-csv selects which
            # question_ids enter the top/bottom-decile sample pool.
            cell = (
                f'  if [ -f "{marker}" ]; then\n'
                f'    echo "[{name}/{ds_tag}] skip (marker present)"\n'
                f'  else\n'
                f'    echo "[{name}/{ds_tag}] starting on GPU {gpu} at $(date)"\n'
                f'    CUDA_VISIBLE_DEVICES={gpu} uv run python '
                f'{SCRIPTS}/extract_attention_mass.py '
                f'--model {name} --hf-model {hf} '
                f'--config {CONFIGS}/{cfg_slug}.yaml '
                f'--susceptibility-csv {susc_path} '
                f'--top-decile-n {N_PER_CELL // 2} '
                f'--bottom-decile-n {N_PER_CELL // 2} '
                f'--max-samples {N_PER_CELL} '
                f'--bbox-file {BBOX_FILE} '
                f'--output-root {ATT_ROOT_FRESH} '
                f'  && mkdir -p "{marker_dir}" && touch "{marker}" '
                f'  && echo "[{name}/{ds_tag}] done at $(date)" '
                f'  || echo "[{name}/{ds_tag}] FAILED at $(date)"\n'
                f'  fi'
            )
            per_cell.append(cell)
        body = "\n".join([
            f"(",
            f"  echo '=== {label} ({name}) on GPU {gpu} ===';",
            *per_cell,
            f"  echo '=== {label} complete ===';",
            f") > '{log}' 2>&1 &",
            f"PIDS+=($!)",
        ])
        bodies.append(body)

    footer = [
        "",
        "echo \"launched ${#PIDS[@]} model jobs: ${PIDS[@]}\"",
        "echo \"per-model logs under " + str(LAUNCH_DIR) + "/_log_*.txt\"",
        "wait \"${PIDS[@]}\"",
        "echo \"all done at $(date)\"",
    ]
    return "\n".join(header + bodies + footer) + "\n"


LAUNCH_SCRIPT.write_text(_build_launch_script())
LAUNCH_SCRIPT.chmod(0o755)
print(f"Launch script written: {LAUNCH_SCRIPT}")
print()
print("Run options (in this priority):")
print(f"  # tmux (recommended) — survives terminal disconnect:")
print(f"  tmux new -s sec5_1 'bash {LAUNCH_SCRIPT} 2>&1 | tee {LAUNCH_LOG}'")
print()
print(f"  # nohup background (also survives disconnect):")
print(f"  nohup bash {LAUNCH_SCRIPT} > {LAUNCH_LOG} 2>&1 &")
print()
print(f"  # foreground (notebook will block):")
print(f"  bash {LAUNCH_SCRIPT}")
print()
print(f"Per-model live logs under {LAUNCH_DIR}/_log_*.txt")
print(f"Resumable: re-running the script skips any cell whose marker exists.")
"""),

    md(r"""
### 3.1 · Status — completed-cell count

Tally the per-(model, dataset) marker files to track progress across
crashes / pod restarts. 25 cells total when complete.
"""),
    code(r"""
def status_table() -> pd.DataFrame:
    rows = []
    for (label, name, hf, attn) in MECH_PANEL:
        for ds_tag, _cfg, _susc in PEAK_DATASETS:
            marker = ATT_ROOT_FRESH / name / f"_done_{ds_tag}.marker"
            rows.append({
                "model": name,
                "dataset": ds_tag,
                "done": marker.exists(),
                "marker": str(marker.relative_to(MAIN)) if marker.exists() else "",
            })
    return pd.DataFrame(rows)


S = status_table()
n_done  = int(S["done"].sum())
n_total = len(S)
print(f"completed cells: {n_done}/{n_total} ({100*n_done/n_total:.0f}%)")
S.pivot(index="model", columns="dataset", values="done").reindex(
    index=[m for _, m, _, _ in MECH_PANEL],
    columns=[d for d, _c, _s in PEAK_DATASETS],
).map(lambda v: "✓" if v else "·")
"""),
    md(r"""
## 4 · Aggregate → `cross_dataset_peaks.csv`

Two sequential stages: (a) reduce each per-cell JSONL into per-layer mean
attention mass, (b) cross-reference against per-dataset susceptibility
CSVs to identify the peak layer and its 95 % CI per `(model, dataset)`.
"""),
    code(r"""
def aggregate_attention() -> None:
    # `ATT_ROOT` resolves at notebook runtime to either the fresh isolated
    # tree (when RUN_INFERENCE=True) or the legacy `outputs/attention_analysis/`
    # tree (which already contains bbox-with runs for every mechanism model).
    rc = run_cmd(
        ["uv", "run", "python", str(SCRIPTS / "analyze_cross_dataset_peaks.py"),
         "--input-root", str(ATT_ROOT),
         "--output-csv", str(PEAKS_CSV),
         "--region",     "image_anchor_digit",
         # Susceptibility CSVs live under the main worktree (gitignored under
         # docs/insights/_data/), so point the analyzer there explicitly.
         "--susc-dir",   str(DATA_DIR)],
        # The peak analyzer is light + fast (no GPU); always run so the
        # `image_anchor_digit` peak table reflects the chosen input root.
        dry=False,
    )
    if rc:
        raise RuntimeError(f"analyze_cross_dataset_peaks.py exited {rc}")


aggregate_attention()
"""),
    md("## 5 · Inspect `cross_dataset_peaks.csv`"),
    code(r"""
peaks_path = PEAKS_CSV
if not peaks_path.exists():
    # Reproducer notebooks normally write the fresh CSV under the isolated tree;
    # fall back to the legacy canonical for smoke-only viewing.
    fallback = DATA_DIR / "cross_dataset_peaks.csv"
    if fallback.exists():
        print(f"  (fresh CSV missing; reading legacy {fallback} for smoke-only preview)")
        peaks_path = fallback

if peaks_path.exists():
    peaks = pd.read_csv(peaks_path)
    print(f"  source: {peaks_path}")
else:
    print(f"missing: {peaks_path}")
    peaks = pd.DataFrame()
peaks
"""),
    md("## 6 · §5.1 figure — per-model peak depth"),
    code(r"""
def fig_attention_peaks(peaks: pd.DataFrame) -> plt.Figure:
    PRETTY = {
        "gemma3-4b-it":                 "Gemma3-4b",
        "qwen2.5-vl-7b-instruct":       "Qwen2.5-VL-7b",
        "llava-next-interleaved-7b":    "Interleave-7b",
        "llava-onevision-qwen2-7b-ov":  "OneVision-7b",
        "gemma3-27b-it":                "Gemma3-27b",
        "qwen2.5-vl-32b-instruct":      "Qwen2.5-VL-32b",
    }
    df = peaks[peaks["model"].isin(PRETTY)].copy()
    # cross_dataset_peaks.csv has multiple rows per (model, dataset) — one per
    # generation step. The §5.1 figure reports the peak under the final answer
    # token alignment ("step" == "answer").
    if "step" in df.columns:
        df = df[df["step"] == "answer"].copy()
    df["depth_norm"] = df["peak_layer"] / df["n_layers"]

    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150)
    ds_order = ["plotqa", "infovqa", "chartqa", "mathvista", "tallyqa"]
    ds_color = {"plotqa": "#1F4FA8", "infovqa": "#C8102E",
                "chartqa": "#F2A900", "mathvista": "#6C7280", "tallyqa": "#1A7F3F"}
    ds_label = {"plotqa": "PlotQA", "infovqa": "InfoVQA",
                "chartqa": "ChartQA", "mathvista": "MathVista", "tallyqa": "TallyQA"}
    model_order = list(PRETTY)
    x = np.arange(len(model_order))
    width = 0.16

    for i, ds in enumerate(ds_order):
        sub = df[df["dataset"] == ds].set_index("model").reindex(model_order)
        ys = sub["depth_norm"].values.astype(float)
        ax.bar(x + (i - 2) * width, ys, width,
               color=ds_color[ds], edgecolor="black", linewidth=0.4,
               label=ds_label[ds])

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in model_order], rotation=15, ha="right")
    ax.set_ylabel("peak layer depth (peak_L / n_layers)")
    ax.set_ylim(0, 1.05)
    ax.set_title("§5.1 — Per-model attention peak depth (digit-pixel region, 5-model panel × 5 datasets)")
    ax.legend(loc="upper left", frameon=False, ncol=5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


if not peaks.empty:
    fig = fig_attention_peaks(peaks)
    save_figure(fig, "paper_5_1_attention_peak_depth")
    fig
"""),
    md(r"""
## Summary

Pipeline: 5 model × 3 dataset attention extraction (sharded one GPU per
model) → 2-stage aggregator → `docs/insights/_data/cross_dataset_peaks.csv`
→ §5.1 figure (`paper_5_1_attention_peak_depth.{pdf,png}`).

To rerun from scratch set `RUN_INFERENCE = True` in §1 and ensure
`VLM_ANCHOR_GPUS` covers the GPUs you want pinned. The §5.3 notebook
loads the peak table from disk so it can run figure-only after this.
"""),
]


# =============================================================================
# Notebook 2 — §5.2 K-subspace sweep
# =============================================================================

cells_52: list[dict] = _setup_cells(
    title="Paper §5.2 — K-subspace sweep (multi-direction within a layer)",
    subtitle=(
        "Reproduces the §5.2 pilot grid (PlotQA × OneVision, L × α × K) and "
        "the 5-dataset layer sweep at L=26 (K=8 vs K=1 fallback). Both rest on "
        "an (a − m) calibration of the OneVision Main residual stream."
    ),
    scope_doc="§5.2 (K-subspace sweep)",
)

cells_52 += [
    md(r"""
## 2 · Configuration

OneVision Main is `llava-onevision-qwen2-7b-ov`. The calibration scope
follows outline §6.1 — PlotQA + InfoVQA pooled, wrong-base + numeric
`(a, m)` pairs only.

The pilot grid is 3 layers × 3 α × 3 K = 27 cells (PlotQA, n=250).
The 5-dataset sweep adds 5 datasets × 5 layers at K=8 + 5 datasets ×
L=26 at K=1, all at α=1.0.
"""),
    code(r"""
ONEVISION    = "llava-onevision-qwen2-7b-ov"
ONEVISION_HF = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

PILOT_LAYERS = [25, 26, 27]
PILOT_ALPHAS = [0.5, 1.0, 2.0]
PILOT_KS     = [2, 4, 8]

CALIB_SCOPE       = "plotqa_infovqa_pooled_n5k"
CALIB_MAX_PAIRS   = 2500

SWEEP_DATASETS_5D = [
    ("plotqa",    "experiment_e7_plotqa_full"),
    ("infovqa",   "experiment_e7_infographicvqa_full"),
    ("tallyqa",   "experiment_e5e_tallyqa_full"),
    ("chartqa",   "experiment_e5e_chartqa_full"),
    ("mathvista", "experiment_e5e_mathvista_full"),
]
SWEEP_LAYERS_5D = [22, 24, 26, 28, 30]
SWEEP_KS_5D     = [8, 1]   # K=8 sweep + K=1 fallback at L=26
SWEEP_ALPHA_5D  = 1.0
"""),
    md(r"""
## 3 · Calibrate (a − m) subspace — 8-GPU sharded SVD

`run_calibrate_subspace_sharded.py` shards the wrong-base + 4-condition
sids round-robin across the 8 GPUs, collects residual diffs per shard,
concatenates D_wrong + D_all, then runs SVD once on the merged matrix.

Outline §6.1 calls for one calibration over the PlotQA + InfoVQA pool;
we calibrate per dataset and merge to keep each shard within VRAM.
"""),
    code(r"""
def calibrate_subspace(dataset_tag: str, config_slug: str):
    pred = PRED_ROOT / dataset_tag / ONEVISION / "predictions.jsonl"
    if not pred.exists() and RUN_INFERENCE:
        raise FileNotFoundError(
            f"missing predictions: {pred} — run paper_cross_model_cross_dataset.ipynb first."
        )
    out_dir = E6_ROOT / ONEVISION / f"calibration_{dataset_tag}"
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_calibrate_subspace_sharded.py"),
        "--config", str(CONFIGS / f"{config_slug}.yaml"),
        "--model", ONEVISION, "--hf-model", ONEVISION_HF,
        "--predictions-path", str(pred),
        "--dataset-tag", dataset_tag,
        "--max-calibrate-pairs", str(CALIB_MAX_PAIRS // 2),
        "--gpus", GPUS,
        "--out-dir", str(out_dir),  # isolate to E6_ROOT
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


calibrate_subspace("plotqa",  "experiment_e7_plotqa_full")
calibrate_subspace("infovqa", "experiment_e7_infographicvqa_full")

# Merge per-dataset subspaces into the pooled basis used by all downstream sweeps.
# `merge_calibrate_subspaces.py` reads from `outputs/e6_steering/<model>/`; the
# call here is informational — when running with the isolated E6_ROOT_FRESH
# tree, replicate or symlink the calibration_<tag> dirs first.
run_cmd(
    ["uv", "run", "python", str(SCRIPTS / "merge_calibrate_subspaces.py"),
     "--model", ONEVISION, "--scope", CALIB_SCOPE,
     "--inputs", "plotqa", "infovqa"],
    dry=not RUN_INFERENCE,
)
"""),
    md(r"""
## 4 · Pilot grid sweep — PlotQA × OneVision (8-GPU sharded)

27 cells of `(layer, α, K)`. Each cell shards its `(sid × cond)`
inference across all 8 GPUs; cells run sequentially. Expected wall
time on 8 × H200 ≈ 4–6 GPU-hours.
"""),
    code(r"""
def sweep_pilot():
    subspace_path = E6_ROOT / ONEVISION / "_subspace" / f"subspace_{CALIB_SCOPE}.pt"
    out_dir = E6_ROOT / ONEVISION / "pilot_grid_plotqa_n250"
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_sweep_subspace_sharded.py"),
        "--config", str(CONFIGS / "experiment_e7_plotqa_full.yaml"),
        "--model", ONEVISION, "--hf-model", ONEVISION_HF,
        "--predictions-path", str(PRED_ROOT / "plotqa" / ONEVISION / "predictions.jsonl"),
        "--dataset-tag", "plotqa",
        "--subspace-path", str(subspace_path),
        "--subspace-scope", CALIB_SCOPE,
        "--sweep-layers", ",".join(str(L) for L in PILOT_LAYERS),
        "--sweep-alphas", ",".join(str(a) for a in PILOT_ALPHAS),
        "--sweep-ks",     ",".join(str(k) for k in PILOT_KS),
        "--max-samples", "250",
        "--gpus", GPUS,
        "--out-dir", str(out_dir),  # isolate to E6_ROOT
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


sweep_pilot()
run_cmd(
    ["uv", "run", "python", str(SCRIPTS / "aggregate_e6_pilot_grid.py"),
     "--e6-root", str(E6_ROOT),
     "--out-csv", str(MAIN / "outputs" / "paper2" / "section_5_e6_steering" / "_data" / "E6_pilot_grid_27cells.csv"),
     "--fig-dir", str(MAIN / "outputs" / "paper2" / "section_5_figures")],
    dry=not RUN_INFERENCE,
)
"""),
    md("## 5 · Figure §5.2a — pilot grid heatmap (4 metrics × 3 K × 3 L × 3 α)"),
    code(r"""
def fig_pilot_grid() -> plt.Figure | None:
    src = DATA_DIR / "e6_pilot_grid_plotqa.csv"
    if not src.exists():
        print(f"  (skipped — {src.name} missing; run §4 with RUN_INFERENCE=True first)")
        return None
    grid = pd.read_csv(src)

    metrics = [
        ("delta_adopt_a", "Δ adopt(a) pp"),
        ("delta_df_a",    "Δ df(a) pp"),
        ("delta_em_a",    "Δ em(a) pp"),
        ("delta_em_b",    "Δ em(b) pp"),
    ]
    fig, axes = plt.subplots(len(metrics), len(PILOT_KS),
                             figsize=(11, 9.0), dpi=150,
                             sharex=True, sharey=True)
    for col, K in enumerate(PILOT_KS):
        for row, (col_metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            sub = grid[grid["K"] == K]
            piv = sub.pivot_table(index="L", columns="alpha", values=col_metric)
            piv = piv.reindex(index=PILOT_LAYERS, columns=PILOT_ALPHAS)
            vmax = float(piv.abs().values.max()) if piv.size else 1.0
            ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            for i, L in enumerate(PILOT_LAYERS):
                for j, alpha in enumerate(PILOT_ALPHAS):
                    v = piv.loc[L, alpha]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                                fontsize=8, color="black")
            if row == 0:
                ax.set_title(f"K={K}", fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(range(len(PILOT_ALPHAS))); ax.set_xticklabels(PILOT_ALPHAS, fontsize=8)
            ax.set_yticks(range(len(PILOT_LAYERS))); ax.set_yticklabels(PILOT_LAYERS, fontsize=8)
            if row == len(metrics) - 1:
                ax.set_xlabel("α", fontsize=9)
    fig.suptitle("§5.2a — E6 pilot grid (PlotQA × OneVision, n=250)\n"
                 "Δ vs base condition; ★ chosen cell = L=26, α=1.0, K=8",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


fig = fig_pilot_grid()
if fig is not None:
    save_figure(fig, "paper_5_2a_E6_pilot_grid_plotqa")
fig
"""),
    md(r"""
## 6 · 5-dataset layer sweep — K=8 main + K=1 fallback at L=26

Cross-dataset cross-check of the K choice. Each `(dataset, L)` cell is
one sharded sweep run; cells run sequentially. Expected ≈ 6–10 GPU-hours
on 8 × H200.
"""),
    code(r"""
def sweep_5dataset_layer():
    subspace_path = E6_ROOT / ONEVISION / "_subspace" / f"subspace_{CALIB_SCOPE}.pt"
    for ds_tag, cfg_slug in SWEEP_DATASETS_5D:
        pred = PRED_ROOT / ds_tag / ONEVISION / "predictions.jsonl"
        out_dir = E6_ROOT / ONEVISION / f"sweep_subspace_{ds_tag}_{CALIB_SCOPE}_p4_layer_sweep_K1_layers_K8"
        cmd = [
            "uv", "run", "python", str(SCRIPTS / "run_sweep_subspace_sharded.py"),
            "--config", str(CONFIGS / f"{cfg_slug}.yaml"),
            "--model", ONEVISION, "--hf-model", ONEVISION_HF,
            "--predictions-path", str(pred),
            "--dataset-tag", ds_tag,
            "--subspace-path", str(subspace_path),
            "--subspace-scope", CALIB_SCOPE,
            "--sweep-layers", ",".join(str(L) for L in SWEEP_LAYERS_5D),
            "--sweep-alphas", str(SWEEP_ALPHA_5D),
            "--sweep-ks", ",".join(str(k) for k in SWEEP_KS_5D),
            "--gpus", GPUS,
            "--out-dir", str(out_dir),  # isolate to E6_ROOT
        ]
        run_cmd(cmd, dry=not RUN_INFERENCE)

    run_cmd(
        ["uv", "run", "python", str(SCRIPTS / "aggregate_e6_layer_sweep_p4.py"),
         "--e6-root", str(E6_ROOT),
         "--out-data", str(MAIN / "outputs" / "paper2" / "section_5_e6_steering" / "_data")],
        dry=not RUN_INFERENCE,
    )


sweep_5dataset_layer()
"""),
    md("## 7 · Figure §5.2b — 5-dataset Δdf(a) sweep + K=1 fallback"),
    code(r"""
def fig_layer_sweep() -> plt.Figure | None:
    src = DATA_DIR / "p4_layer_sweep_per_cell_ci.csv"
    if not src.exists():
        print(f"  (skipped — {src.name} missing; run §6 with RUN_INFERENCE=True first)")
        return None
    sweep = pd.read_csv(src)
    # Canonical CSV uses `layer`, `delta_df`, and `ds_tag` for the dataset key.

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    color = {"plotqa": "#1F4FA8", "infovqa": "#C8102E",
             "tallyqa": "#1A7F3F", "chartqa": "#F2A900", "mathvista": "#6C7280"}
    for ds_tag, c in color.items():
        head = sweep[(sweep["ds_tag"] == ds_tag) & (sweep["K"] == SWEEP_KS_5D[0])]
        if len(head):
            ax.plot(head["layer"], head["delta_df"] * 100,
                    color=c, marker="o", label=f"{ds_tag} K=8")
        fb = sweep[(sweep["ds_tag"] == ds_tag) & (sweep["K"] == 1) & (sweep["layer"] == 26)]
        if len(fb):
            ax.scatter([26], fb["delta_df"].values * 100,
                       color=c, marker="s", s=110, edgecolor="black",
                       linewidth=0.8, label=f"{ds_tag} K=1 fallback @ L=26", zorder=5)
    ax.axhline(0, color="#888", linewidth=0.7, linestyle=":")
    ax.set_xlabel("layer (L)")
    ax.set_ylabel("Δ df(a)  pp  (negative ⇒ anchoring reduced)")
    ax.set_title("§5.2b — 5-dataset Δdf(a) at α=1.0 — K=8 sweep (lines) vs K=1 fallback at L=26 (squares)\n"
                 "K=1 weaker than K=8 ⇒ multi-direction required")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


fig = fig_layer_sweep()
if fig is not None:
    save_figure(fig, "paper_5_2b_layer_sweep_delta_df")
fig
"""),
    md(r"""
## Summary

Pipeline: calibrate (a−m) subspace on PlotQA+InfoVQA pooled → pilot
grid sweep on PlotQA × OneVision → 5-dataset layer sweep at K∈{8,1}
→ two figures (`paper_5_2a_E6_pilot_grid_plotqa`,
`paper_5_2b_layer_sweep_delta_df`).

Full reproduction on 8 × H200 ≈ 12–16 GPU-hours; calibration alone is
≈ 2 hours. The §5.3 narrative notebook synthesizes the §5.1 + §5.2
outputs.
"""),
]


# =============================================================================
# Notebook 3 — §5.3 Routing-and-integration (narrative)
# =============================================================================

cells_53: list[dict] = _setup_cells(
    title="Paper §5.3 — Routing-and-integration synthesis",
    subtitle=(
        "Narrative-only notebook that loads the §5.1 + §5.2 outputs and "
        "lays out the routing-and-integration framework the paper uses to "
        "synthesize them. No new compute."
    ),
    scope_doc="§5.3 (Routing-and-integration)",
)

cells_53 += [
    md(r"""
## 2 · Recap — what §5.1 + §5.2 established

**§5.1 (per-model peak heterogeneity).** Across the 5-model mechanism
panel the attention peak layer varies from early (Gemma3-4b ≈ L=5 / 42)
to mid (ConvLLaVA, LLaVA-1.5, FastVLM around mid-stack) to late
(OneVision ≈ L=27 / 28 on PlotQA + TallyQA). OneVision additionally
shows a *dataset-dependent* peak shift: InfoVQA pushes the peak back
to L≈14. No uniform causal site.

**§5.2 (within-layer multi-direction).** The K-subspace sweep on
OneVision Main shows a monotonic improvement K=1 → K=2 → K=4 → K=8 in
Δadopt(a) and Δdf(a) at the chosen integration site (L=26, α=1.0). At
L=26 the K=1 fallback fails to clear the anchoring effect across 5
datasets — single direction is insufficient (Figure §5.2b).
"""),
    code(r"""
peaks_path = PEAKS_CSV  # isolated reproducer artifact, falls back to legacy below
if not peaks_path.exists():
    legacy = DATA_DIR / "cross_dataset_peaks.csv"
    if legacy.exists():
        print(f"  (fresh CSV missing; reading legacy {legacy} for smoke-only preview)")
        peaks_path = legacy
sweep_path = DATA_DIR / "p4_layer_sweep_per_cell_ci.csv"
pilot_path = DATA_DIR / "e6_pilot_grid_plotqa.csv"

if peaks_path.exists():
    peaks = pd.read_csv(peaks_path)
    print(f"§5.1 peaks rows: {len(peaks)} (source: {peaks_path})")
else:
    print(f"missing: {peaks_path} — run paper_section_5_1_attention_peaks.ipynb first.")
if pilot_path.exists():
    pilot = pd.read_csv(pilot_path)
    print(f"§5.2a pilot cells: {len(pilot)}")
else:
    print(f"missing: {pilot_path} — run paper_section_5_2_subspace_sweep.ipynb first.")
if sweep_path.exists():
    sweep = pd.read_csv(sweep_path)
    print(f"§5.2b sweep cells: {len(sweep)}")
else:
    print(f"missing: {sweep_path} — run paper_section_5_2_subspace_sweep.ipynb first.")
"""),
    md(r"""
## 3 · Routing-and-integration framework

The two §5 findings co-jointly characterize the anchoring representation
as having two structural properties:

- **Routed across multiple attention layers** — model-specific peak
  layers indicate that different architectures handle anchoring at
  different stages of their forward pass. Uniformity would be the
  exception, not the rule, under this account; the §5.1 panel observes
  exactly that heterogeneity.
- **Integrated into a residual-stream subspace of dimension ≥ 2** —
  the K-monotonic improvement in §5.2 and the K=1 fallback failure
  rule out a single-direction representation. The integration site for
  OneVision Main is `L=26` of 28 layers (≈ 93 % depth), i.e., late in
  the residual stream.

**Mitigation design implications (§6).** The two structural properties
map directly onto the two §6 design choices:
- Choose the *integration site* per architecture (no shared causal site).
- Project out a *multi-direction subspace* (K ≥ 2), not a single
  direction. K=8 captures > 95 % of the explained variance on
  OneVision Main and is the chosen K in the paper's §6.2 mitigation.

**Limitations of §5 evidence.** The peak panel covers 5 architectures
on 3 datasets; further architectures (e.g., Gemma3-27b, Qwen2.5-VL-32b)
would extend the heterogeneity claim. The K-subspace sweep is run
on OneVision Main only; the K-monotonic property is *expected* to be
architecture-general but is *verified* on one model. Cross-architecture
directional verification (γ-β residual-stream bridge on Qwen3-VL) is in
Appendix E and confirms the *direction* of the mitigation but not its
magnitude.
"""),
    md(r"""
## Summary

§5.3 is interpretation only; it depends on §5.1 + §5.2 outputs being
present on disk. The two §5 findings (peak heterogeneity + within-layer
multi-direction) form the routing-and-integration framework used by §6.

Re-run order:
1. `notebooks/paper_section_5_1_attention_peaks.ipynb` — produces `cross_dataset_peaks.csv`.
2. `notebooks/paper_section_5_2_subspace_sweep.ipynb` — produces `e6_pilot_grid_plotqa.csv` + `p4_layer_sweep_per_cell_ci.csv`.
3. This notebook — reads them back, restates the framework.
"""),
]


# =============================================================================
# Write all three
# =============================================================================

def _write(filename: str, cells: list[dict], scope: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "vlm_anchor": {"purpose": "paper-reproduction", "scope": scope},
    }
    out = NB_DIR / filename
    with open(out, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {out}  ({len(cells)} cells)")


_write("paper_section_5_1_attention_peaks.ipynb", cells_51, "section_5_1_attention_peaks")
_write("paper_section_5_2_subspace_sweep.ipynb",  cells_52, "section_5_2_subspace_sweep")
_write("paper_section_5_3_routing_integration.ipynb", cells_53, "section_5_3_routing_integration")
