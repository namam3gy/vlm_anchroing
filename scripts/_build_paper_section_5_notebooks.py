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
SCRIPTS    = MAIN / "scripts"
CONFIGS    = MAIN / "configs"
DATA_DIR   = MAIN / "docs" / "insights" / "_data"
ATT_ROOT   = MAIN / "outputs" / "attention_analysis"
E6_ROOT    = MAIN / "outputs" / "e6_steering"
PRED_ROOT  = MAIN / "outputs" / "paper" / "cross_model_cross_dataset" / "predictions"

PDF_OUT = MAIN     / "outputs" / "paper" / "section_5_figures"
PNG_OUT = WORKTREE / "docs"    / "figures"
PDF_OUT.mkdir(parents=True, exist_ok=True)
PNG_OUT.mkdir(parents=True, exist_ok=True)

GPUS = os.environ.get("VLM_ANCHOR_GPUS", "0,1,2,3,4,5,6,7")  # 8 GPUs by default
RUN_INFERENCE = False  # set True to invoke the heavy sharded drivers.

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

Outline §5.1 expects the *5-model peak panel* (qwen2.5-vl-7b 's
attention probe is paper-deferred; the figure reports 5 models).

| Model | HF id | Encoder family | Expected peak depth |
|---|---|---|---|
| ConvLLaVA-7b      | `ConvLLaVA/ConvLLaVA-sft-1536`            | ConvNeXt encoder + LLaMA  | mid (L≈14 / 32) |
| LLaVA-1.5-7b      | `llava-hf/llava-1.5-7b-hf`                | CLIP-ViT + LLaMA          | mid (L≈14 / 32) |
| FastVLM-7b        | `apple/FastVLM-7B`                        | FastViT + Qwen2           | mid (L≈17 / 28) |
| Gemma3-4b         | `google/gemma-3-4b-it`                    | SigLIP2 + Gemma           | early (L≈5 / 42) |
| LLaVA-OneVision   | `llava-hf/llava-onevision-qwen2-7b-ov-hf` | AnyRes-CLIP + Qwen2       | late (L≈27 / 28 on PlotQA / TallyQA; L≈14 on InfoVQA) |

Datasets for §5.1:
- **PlotQA** (primary, used for headline peak on every model)
- **TallyQA** (OneVision-only, confirms late peak generalizes)
- **InfoVQA** (OneVision-only, exposes dataset-dependent peak shift)

Hyperparameters identical to §4: `seed=42`, greedy decoding, `max_new_tokens=16` (TallyQA 8); §5.1 additionally forces `attn_implementation=eager` so the attention matrix is materialized for `extract_attention_mass.py`.
"""),
    code(r"""
MECH_PANEL = [
    # (label, internal name, HF id, attn_impl)
    ("ConvLLaVA-7b",   "convllava-7b",                "ConvLLaVA/ConvLLaVA-sft-1536",            "eager"),
    ("LLaVA-1.5-7b",   "llava-1.5-7b",                "llava-hf/llava-1.5-7b-hf",                "eager"),
    ("FastVLM-7b",     "fastvlm-7b",                  "apple/FastVLM-7B",                        "eager"),
    ("Gemma3-4b",      "gemma3-4b-it",                "google/gemma-3-4b-it",                    "eager"),
    ("OneVision-7b",   "llava-onevision-qwen2-7b-ov", "llava-hf/llava-onevision-qwen2-7b-ov-hf", "eager"),
]

# (dataset_tag, config, max_samples_for_attention_pass, applies_to_all_models)
PEAK_DATASETS = [
    ("plotqa",   "experiment_e7_plotqa_full",         400, True),
    ("tallyqa",  "experiment_e5e_tallyqa_full",       400, False),  # OneVision only by default
    ("infovqa",  "experiment_e7_infographicvqa_full", 400, False),  # OneVision only by default
]

ONEVISION = "llava-onevision-qwen2-7b-ov"
for label, name, hf, attn in MECH_PANEL:
    print(f"  {label:<16} → {name:<32} (attn={attn})")
"""),
    md(r"""
## 3 · Extract attention mass — `extract_attention_mass.py`

For each `(model, dataset)` cell we launch `extract_attention_mass.py`
pinned to one GPU via `CUDA_VISIBLE_DEVICES`. Output JSONL records carry
per-layer attention mass to the four input regions (target image, anchor
image, prompt text, generated). With 5 models on 5 GPUs the per-model
serial sweep across 3 datasets finishes in ≈ 3 GPU-hours of wall time.
"""),
    code(r"""
import itertools

def extract_attention_for_cell(name: str, hf: str, attn: str,
                               dataset_tag: str, config_slug: str,
                               max_samples: int, gpu: int) -> int:
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "extract_attention_mass.py"),
        "--model", name, "--hf-model", hf,
        "--config", str(CONFIGS / f"{config_slug}.yaml"),
        "--attn-implementation", attn,
        "--max-samples", str(max_samples),
        "--dataset-tag", dataset_tag,
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE,
                   env={"CUDA_VISIBLE_DEVICES": str(gpu)})


gpu_iter = itertools.cycle([int(g) for g in GPUS.split(",")])
for (_label, name, hf, attn) in MECH_PANEL:
    gpu = next(gpu_iter)
    for (ds_tag, cfg, n, applies_all) in PEAK_DATASETS:
        if not applies_all and name != ONEVISION:
            continue
        extract_attention_for_cell(name, hf, attn, ds_tag, cfg, n, gpu)
"""),
    md(r"""
## 4 · Aggregate → `cross_dataset_peaks.csv`

Two sequential stages: (a) reduce each per-cell JSONL into per-layer mean
attention mass, (b) cross-reference against per-dataset susceptibility
CSVs to identify the peak layer and its 95 % CI per `(model, dataset)`.
"""),
    code(r"""
def aggregate_attention() -> None:
    for script in ("analyze_attention_per_layer.py", "analyze_cross_dataset_peaks.py"):
        rc = run_cmd(["uv", "run", "python", str(SCRIPTS / script)],
                     dry=not RUN_INFERENCE)
        if rc and RUN_INFERENCE:
            raise RuntimeError(f"{script} exited {rc}")

aggregate_attention()
"""),
    md("## 5 · Inspect `cross_dataset_peaks.csv`"),
    code(r"""
peaks_path = DATA_DIR / "cross_dataset_peaks.csv"
if peaks_path.exists():
    peaks = pd.read_csv(peaks_path)
else:
    print(f"missing: {peaks_path}")
    peaks = pd.DataFrame()
peaks
"""),
    md("## 6 · §5.1 figure — per-model peak depth"),
    code(r"""
def fig_attention_peaks(peaks: pd.DataFrame) -> plt.Figure:
    PRETTY = {
        "convllava-7b":                "ConvLLaVA-7b",
        "llava-1.5-7b":                "LLaVA-1.5-7b",
        "fastvlm-7b":                  "FastVLM-7b",
        "gemma3-4b-it":                "Gemma3-4b",
        "llava-onevision-qwen2-7b-ov": "OneVision-7b",
    }
    df = peaks[peaks["model"].isin(PRETTY)].copy()
    # cross_dataset_peaks.csv has multiple rows per (model, dataset) — one per
    # generation step. The §5.1 figure reports the peak under the final answer
    # token alignment ("step" == "answer").
    if "step" in df.columns:
        df = df[df["step"] == "answer"].copy()
    df["depth_norm"] = df["peak_layer"] / df["n_layers"]

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
    ds_order = ["plotqa", "tallyqa", "infovqa"]
    ds_color = {"plotqa": "#1F4FA8", "tallyqa": "#1A7F3F", "infovqa": "#C8102E"}
    ds_label = {"plotqa": "PlotQA", "tallyqa": "TallyQA", "infovqa": "InfoVQA"}
    model_order = list(PRETTY)
    x = np.arange(len(model_order))
    width = 0.27

    for i, ds in enumerate(ds_order):
        sub = df[df["dataset"] == ds].set_index("model").reindex(model_order)
        ys = sub["depth_norm"].values.astype(float)
        ax.bar(x + (i - 1) * width, ys, width,
               color=ds_color[ds], edgecolor="black", linewidth=0.4,
               label=ds_label[ds])

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[m] for m in model_order], rotation=15, ha="right")
    ax.set_ylabel("peak layer depth (peak_L / n_layers)")
    ax.set_ylim(0, 1.05)
    ax.set_title("§5.1 — Per-model attention peak depth\n"
                 "Heterogeneous integration sites: early (Gemma) → mid (ConvLLaVA, LLaVA-1.5, FastVLM) → late (OneVision)")
    ax.legend(loc="upper left", frameon=False, ncol=3)
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
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_calibrate_subspace_sharded.py"),
        "--config", str(CONFIGS / f"{config_slug}.yaml"),
        "--model", ONEVISION, "--hf-model", ONEVISION_HF,
        "--predictions-path", str(pred),
        "--dataset-tag", dataset_tag,
        "--max-calibrate-pairs", str(CALIB_MAX_PAIRS // 2),
        "--gpus", GPUS,
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


calibrate_subspace("plotqa",  "experiment_e7_plotqa_full")
calibrate_subspace("infovqa", "experiment_e7_infographicvqa_full")

# Merge per-dataset subspaces into the pooled basis used by all downstream sweeps.
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
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_sweep_subspace_sharded.py"),
        "--config", str(CONFIGS / "experiment_e7_plotqa_full.yaml"),
        "--model", ONEVISION, "--hf-model", ONEVISION_HF,
        "--predictions-path", str(PRED_ROOT / "plotqa" / ONEVISION / "predictions.jsonl"),
        "--dataset-tag", "plotqa",
        "--subspace-path", str(subspace_path),
        "--subspace-scope", CALIB_SCOPE,
        "--sweep-layers", *[str(L) for L in PILOT_LAYERS],
        "--sweep-alphas", *[str(a) for a in PILOT_ALPHAS],
        "--sweep-ks",     *[str(k) for k in PILOT_KS],
        "--max-samples", "250",
        "--gpus", GPUS,
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


sweep_pilot()
run_cmd(
    ["uv", "run", "python", str(SCRIPTS / "aggregate_e6_pilot_grid.py"),
     "--model", ONEVISION, "--dataset", "plotqa", "--scope", CALIB_SCOPE],
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
        cmd = [
            "uv", "run", "python", str(SCRIPTS / "run_sweep_subspace_sharded.py"),
            "--config", str(CONFIGS / f"{cfg_slug}.yaml"),
            "--model", ONEVISION, "--hf-model", ONEVISION_HF,
            "--predictions-path", str(pred),
            "--dataset-tag", ds_tag,
            "--subspace-path", str(subspace_path),
            "--subspace-scope", CALIB_SCOPE,
            "--sweep-layers", *[str(L) for L in SWEEP_LAYERS_5D],
            "--sweep-alphas", str(SWEEP_ALPHA_5D),
            "--sweep-ks", *[str(k) for k in SWEEP_KS_5D],
            "--gpus", GPUS,
        ]
        run_cmd(cmd, dry=not RUN_INFERENCE)

    run_cmd(["uv", "run", "python", str(SCRIPTS / "aggregate_e6_layer_sweep_p4.py")],
            dry=not RUN_INFERENCE)


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
peaks_path = DATA_DIR / "cross_dataset_peaks.csv"
sweep_path = DATA_DIR / "p4_layer_sweep_per_cell_ci.csv"
pilot_path = DATA_DIR / "e6_pilot_grid_plotqa.csv"

if peaks_path.exists():
    peaks = pd.read_csv(peaks_path)
    print(f"§5.1 peaks rows: {len(peaks)}")
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
