"""Build the three §6 reproducibility notebooks.

Outputs:
  notebooks/paper_section_6_1_method.ipynb
  notebooks/paper_section_6_2_anchoring_reduction.ipynb
  notebooks/paper_section_6_3_capability_preservation.ipynb

§6.1 — Algorithm (calibrated subspace projection) + the (a − m) calibration
recipe on the PlotQA + InfoVQA pooled wrong-base+numeric (a, m) pair pool.
The pooled SVD basis seeds the rest of §6 (the chosen-cell K=8 application
in §6.2) and §5.2 (the K-subspace sweep). Calibration step is *upstream* —
all §6.2 / §6.3 numbers depend on the same subspace tensor.

§6.2 — Cross-dataset paired-bootstrap CI on the chosen cell
`L=26 K=8 α=1.0` across PlotQA / InfoVQA / TallyQA / ChartQA / MathVista.
Re-runs inference at the chosen cell (one cell × 5 datasets) on top of
the §5.2 reproduction's predictions if those are present, or fresh
otherwise. Aggregates stage4 (95 % + Bonferroni-20 CI) and renders the
§6.2.3 Figure (`paper_6_2_3_stage4_5dataset_paired_ci.png`) + Table 6.2.

§6.3 — Capability preservation: 6-benchmark VLMEvalKit eval
(RealWorldQA + OCRBench + HallusionBench + MMStar + MMBench_DEV_EN +
POPE, n_total ≈ 10,507) under STRICT_FREE_LUNCH. Heavy wall (~13 h on
1 GPU); RUN_INFERENCE=False reads the canonical CSV instead. Renders
Table 6.3.

All three notebooks share the same setup cells (paths + subprocess helper)
as the §5 reproducers so the layout is symmetric.
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

This notebook drives heavy inference stages by `subprocess`-invoking the
existing drivers in `scripts/`. The `RUN_INFERENCE = False` toggle below
lets a reviewer read the full pipeline without GPU access — canonical
CSVs are read straight from disk and figures get re-rendered from them.
"""),
        md("## 1 · Setup — paths + subprocess helper"),
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
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], cwd=Path.cwd(), text=True
    ).strip()
    return Path(common).resolve().parent


def find_worktree_root() -> Path:
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), text=True
    ).strip()).resolve()


MAIN     = find_main_worktree()
WORKTREE = find_worktree_root()

# Scripts + configs from the active branch (worktree); gitignored artifacts
# (inputs/, outputs/, docs/insights/_data/) from MAIN.
SCRIPTS    = WORKTREE / "scripts"
CONFIGS    = WORKTREE / "configs"
DATA_DIR   = MAIN / "docs" / "insights" / "_data"
FIGURES    = WORKTREE / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

GPUS = os.environ.get("VLM_ANCHOR_GPUS", "0,1,2,3,4")
RUN_INFERENCE = False  # set True to invoke the heavy driver(s)

print(f"MAIN     = {MAIN}")
print(f"WORKTREE = {WORKTREE}")
print(f"GPUS     = {GPUS}")
print(f"RUN_INFERENCE = {RUN_INFERENCE}")
"""),
        code(r"""
def run_cmd(cmd: list[str] | str, *, dry: bool = False, env: dict | None = None) -> int:
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


def save_figure(fig, stem: str, png_dir: Path = FIGURES,
                pdf_dir: Path | None = None):
    png = png_dir / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {png}")
    if pdf_dir is not None:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf = pdf_dir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print(f"wrote {pdf}")
"""),
    ]


# =============================================================================
# Notebook 1 — §6.1 Method
# =============================================================================

cells_61: list[dict] = _setup_cells(
    title="Paper §6.1 — Calibrated subspace projection: method + calibration",
    subtitle=(
        "Defines the calibrated subspace projection mitigation and reproduces "
        "the (a − m) calibration on PlotQA + InfoVQA pooled (n ≈ 5,000 "
        "wrong-base) for OneVision Main. The K=8 SVD top-direction subspace "
        "produced here is the upstream artifact that §6.2 (chosen cell "
        "application) and §5.2 (K-subspace sweep) both read."
    ),
    scope_doc="§6.1 (Method + calibration recipe)",
)

cells_61 += [
    md(r"""
## 2 · Algorithm

**Calibrated subspace projection (E6).** Given an open-weight VLM, a
target intervention layer `L` (mid-to-late residual), and a calibration
pool of `(a, m)` paired-stimuli sids (anchored + masked-anchor pairs)
where the model is base-wrong, compute:

```
# --- Calibration (one-time, off-line) ---
for sid in calibration_pool:
    h_a = vlm.residual(layer=L, input=a_sid)        # anchored arm
    h_m = vlm.residual(layer=L, input=m_sid)        # digit-masked arm
    d_sid = h_a - h_m
D = stack(d_sid)                                    # (n, d_model)
U, S, V^T = SVD(D, full_matrices=False)
V_K = V_:K                                          # (K, d_model) basis

# --- Inference (any input, no anchor label needed) ---
def forward_with_mitigation(x):
    install_pre_hook(layer=L+1):
        h := residual at layer L+1 input
        proj = (h @ V_K.T) @ V_K                    # (B, T, d_model)
        h' = h − α * proj                           # remove K directions
        return h'
    return vlm.forward(x)
```

§6 chosen cell: **`L=26, K=8, α=1.0`** on OneVision Main
(`llava-onevision-qwen2-7b-ov`, 28-layer Qwen2 backbone — L=26 = 93 % of
the stack, late residual). Design choice anchored by §5.1 layer probes
(per-model peak heterogeneity around late residual) + §5.2 K-subspace
sweep (K=8 dominates K=1 by ~10× on PlotQA Δdf).
"""),
    md(r"""
## 3 · Calibration — pooled (a − m) SVD on PlotQA + InfoVQA

`scripts/run_calibrate_subspace_sharded.py` shards eligible sids
round-robin across K GPUs, collects per-shard `(D_wrong, D_all)` tensors,
concatenates, then runs SVD once on the merged matrix.

Outline §6.1 calls for one calibration over the PlotQA + InfoVQA pool;
in practice we calibrate per dataset then merge. The pooled basis tensor
ships in the canonical legacy E6 root at:

```
outputs/e6_steering/<model>/_subspace/subspace_plotqa_infovqa_pooled_n5k_K16.pt
```

The K16 suffix means the SVD retained 16 singular vectors — any K ≤ 16
can be sliced at use time (`V_K = V_all[L, :K, :]`).
"""),
    code(r"""
ONEVISION = "llava-onevision-qwen2-7b-ov"
SUBSPACE_PATH = MAIN / "outputs" / "e6_steering" / ONEVISION / "_subspace" / "subspace_plotqa_infovqa_pooled_n5k_K16.pt"
CALIB_DIR_PLOTQA  = MAIN / "outputs" / "e6_steering" / ONEVISION / "calibration_plotqa"
CALIB_DIR_INFOVQA = MAIN / "outputs" / "e6_steering" / ONEVISION / "calibration_infographicvqa"
CALIB_DIR_POOLED  = MAIN / "outputs" / "e6_steering" / ONEVISION / "calibration_plotqa_infovqa_pooled"


def calibrate(dataset_tag: str, config_slug: str, predictions_path: str):
    out_dir = MAIN / "outputs" / "e6_steering" / ONEVISION / f"calibration_{dataset_tag}"
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_calibrate_subspace_sharded.py"),
        "--config", str(CONFIGS / f"{config_slug}.yaml"),
        "--model", ONEVISION,
        "--hf-model", "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "--predictions-path", str(MAIN / predictions_path),
        "--dataset-tag", dataset_tag,
        "--max-calibrate-pairs", "2500",
        "--gpus", GPUS,
        "--out-dir", str(out_dir),
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


# Per-dataset calibrations
calibrate("plotqa", "experiment_e7_plotqa_full",
          "outputs/experiment_e7_plotqa_full/llava-onevision-qwen2-7b-ov/20260502-132624/predictions.jsonl")
calibrate("infographicvqa", "experiment_e7_infographicvqa_full",
          "outputs/experiment_e7_infographicvqa_full/llava-onevision-qwen2-7b-ov/20260502-152105/predictions.jsonl")

# Merge per-dataset calibration outputs into the pooled basis. The merge
# script is not in-repo (legacy artifact produced the canonical
# `calibration_plotqa_infovqa_pooled/` and the corresponding subspace .pt);
# below we just point at the existing canonical pool. RUN_INFERENCE=True
# above only re-creates the per-dataset shards; the pooled SVD remains the
# audited artifact that all §6 and §5.2 numbers rest on.
print()
print("=== canonical pooled calibration (read-only) ===")
print(f"  D matrix:   {CALIB_DIR_POOLED / 'D_all.pt'}")
print(f"  SVD basis:  {SUBSPACE_PATH}")
if SUBSPACE_PATH.exists():
    import torch
    V = torch.load(SUBSPACE_PATH, weights_only=True)
    print(f"  shape:      {tuple(V.shape)}   (layers, K_max, d_model)")
    print(f"  L=26 row:   K_max={V.shape[1]}; slicing V[26, :8, :] gives the §6 K=8 basis")
"""),
    md(r"""
## 4 · Pooled v_meta sidecar — calibration provenance
"""),
    code(r"""
v_meta = CALIB_DIR_POOLED / "v_meta.json"
if v_meta.exists():
    meta = json.loads(v_meta.read_text())
    print(f"  pooled calibration provenance:")
    for k in ("dataset_tag", "source_tags", "n_wrong", "n_all",
              "n_wrong_per_source", "n_all_per_source",
              "n_layers", "d_model", "D_wrong_shape", "D_all_shape"):
        if k in meta:
            print(f"    {k}: {meta[k]}")
else:
    print(f"  (sidecar absent: {v_meta})")
"""),
    md(r"""
## Summary

Pipeline: per-dataset calibration shards (PlotQA + InfoVQA, ~2,500 pairs
each) → pooled D matrix concat → SVD K=16 → seal as
`_subspace/subspace_plotqa_infovqa_pooled_n5k_K16.pt`.

Downstream uses:
- §5.2 K-subspace sweep slices the same basis at K ∈ {1, 2, 4, 8} and at
  multiple layers + alpha for the pilot grid.
- §6.2 chosen-cell paired-bootstrap CI applies V_K=8 at L=26, α=1.0 across
  5 datasets — see `paper_section_6_2_anchoring_reduction.ipynb`.
- §6.3 capability preservation applies the same projection at inference
  on 6 VLMEvalKit benchmarks — see `paper_section_6_3_capability_preservation.ipynb`.
"""),
]


# =============================================================================
# Notebook 2 — §6.2 Cross-dataset anchoring reduction
# =============================================================================

cells_62: list[dict] = _setup_cells(
    title="Paper §6.2 — Cross-dataset anchoring reduction (chosen cell paired-bootstrap CI)",
    subtitle=(
        "Reproduces the §6.2 headline: at the chosen cell `L=26 K=8 α=1.0` on "
        "OneVision Main, the calibrated subspace projection reduces anchoring "
        "across 5 datasets — **5/5 Δem(b) sign-clean under both 95 % and "
        "Bonferroni-20** corrected paired-bootstrap CIs. Inference is one "
        "cell × 5 datasets (PlotQA / InfoVQA / TallyQA / ChartQA / "
        "MathVista). Aggregator = `build_e6_stage4_bootstrap_ci.py`; figure "
        "= `build_paper_stage4_paired_ci_figure.py`."
    ),
    scope_doc="§6.2 (Cross-dataset anchoring reduction)",
)

cells_62 += [
    md(r"""
## 2 · Chosen cell — single-cell sweep across 5 datasets

For each dataset we run a one-cell sweep with `L=26, K=8, α=1.0` on top
of the canonical pooled subspace. Output dir is
`sweep_subspace_<ds_tag>_<scope>_chosen/`, which is what
`build_e6_stage4_bootstrap_ci.py` reads.

Note that the §5.2 reproduction already swept the same chosen cell as
part of its `..._p4_layer_sweep_K1_layers_K8` dir (one cell among many).
If you already ran §5.2, the same predictions can be reused — see the
`reuse_section_5_2` switch below.
"""),
    code(r"""
ONEVISION    = "llava-onevision-qwen2-7b-ov"
ONEVISION_HF = "llava-hf/llava-onevision-qwen2-7b-ov-hf"

SUBSPACE_PATH = MAIN / "outputs" / "e6_steering" / ONEVISION / "_subspace" / "subspace_plotqa_infovqa_pooled_n5k_K16.pt"
SCOPE         = "plotqa_infovqa_pooled_n5k"

CHOSEN_DATASETS = [
    ("plotqa",         "experiment_e7_plotqa_full",
     "outputs/experiment_e7_plotqa_full/llava-onevision-qwen2-7b-ov/20260502-132624/predictions.jsonl"),
    ("infographicvqa", "experiment_e7_infographicvqa_full",
     "outputs/experiment_e7_infographicvqa_full/llava-onevision-qwen2-7b-ov/20260502-152105/predictions.jsonl"),
    ("tallyqa",        "experiment_e5e_tallyqa_full",
     "outputs/experiment_e5e_tallyqa_full/llava-onevision-qwen2-7b-ov/20260502-083926/predictions.jsonl"),
    ("chartqa",        "experiment_e5e_chartqa_full",
     "outputs/experiment_e5e_chartqa_full/llava-onevision-qwen2-7b-ov/20260502-211028/predictions.jsonl"),
    ("mathvista",      "experiment_e5e_mathvista_full",
     "outputs/experiment_e5e_mathvista_full/llava-onevision-qwen2-7b-ov/20260502-212440/predictions.jsonl"),
]


def sweep_chosen(ds_tag: str, cfg: str, pred: str):
    out_dir = MAIN / "outputs" / "e6_steering" / ONEVISION / f"sweep_subspace_{ds_tag}_{SCOPE}_chosen"
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_sweep_subspace_sharded.py"),
        "--config", str(CONFIGS / f"{cfg}.yaml"),
        "--model", ONEVISION, "--hf-model", ONEVISION_HF,
        "--predictions-path", str(MAIN / pred),
        "--dataset-tag", ds_tag,
        "--subspace-path", str(SUBSPACE_PATH),
        "--subspace-scope", SCOPE,
        "--sweep-layers", "26",
        "--sweep-alphas", "1.0",
        "--sweep-ks", "8",
        "--batch-size", "1",
        "--prefetch-workers", "16",
        "--gpus", GPUS,
        "--out-dir", str(out_dir),
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


for ds_tag, cfg, pred in CHOSEN_DATASETS:
    sweep_chosen(ds_tag, cfg, pred)
"""),
    md(r"""
## 3 · Stage-4 paired-bootstrap CI aggregation

`build_e6_stage4_bootstrap_ci.py` reads each
`sweep_subspace_<ds>_<scope>_chosen/predictions.jsonl`, paired-resamples
sids B=10,000 times, and writes both 95 % and Bonferroni-20 corrected CIs
on Δadopt, Δdf, Δem(a), Δem(b) per dataset.
"""),
    code(r"""
def aggregate_stage4(bootstrap: int = 10_000, seed: int = 20260510):
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "build_e6_stage4_bootstrap_ci.py"),
        "--bootstrap", str(bootstrap),
        "--seed", str(seed),
        "--out-dir", str(DATA_DIR),
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


aggregate_stage4()
"""),
    md(r"""
## 4 · Table 6.2 — Per-dataset Δ at chosen cell

Reads the canonical stage4 CSV and emits Table 6.2 in markdown form.
"""),
    code(r"""
stage4_csv = DATA_DIR / "stage4_final_per_dataset_ci.csv"
if stage4_csv.exists():
    df = pd.read_csv(stage4_csv)
    rows = []
    for ds, label, scope in [
        ("TallyQA", "TallyQA", "held-out"),
        ("PlotQA", "PlotQA", "within"),
        ("InfoVQA", "InfoVQA", "within"),
        ("ChartQA", "ChartQA", "held-out"),
        ("MathVista", "MathVista", "held-out"),
    ]:
        sub = df[df["dataset"] == ds]
        if not len(sub):
            continue
        r = sub.iloc[0]
        rows.append({
            "Dataset": label,
            "n": int(r["n_paired"]),
            "Δ adopt(a)": f"{r['delta_adopt']*100:+.1f} [{r['delta_adopt_ci95_lo']*100:+.1f}, {r['delta_adopt_ci95_hi']*100:+.1f}]",
            "Δ df(a)":    f"{r['delta_df']*100:+.1f} [{r['delta_df_ci95_lo']*100:+.1f}, {r['delta_df_ci95_hi']*100:+.1f}]",
            "Δ em(a)":    f"{r['delta_em_a']*100:+.1f} [{r['delta_em_a_ci95_lo']*100:+.1f}, {r['delta_em_a_ci95_hi']*100:+.1f}]",
            "Δ em(b)":    f"{r['delta_em_b']*100:+.1f} [{r['delta_em_b_ci95_lo']*100:+.1f}, {r['delta_em_b_ci95_hi']*100:+.1f}]",
            "scope":      scope,
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
else:
    print(f"missing: {stage4_csv}")
"""),
    md(r"""
## 5 · Figure §6.2.3 — paired-bootstrap forest panels
"""),
    code(r"""
def render_stage4_figure():
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "build_paper_stage4_paired_ci_figure.py"),
    ]
    return run_cmd(cmd, dry=False)


render_stage4_figure()

# Display the figure
fig_path = WORKTREE / "docs" / "figures" / "paper_6_2_3_stage4_5dataset_paired_ci.png"
if fig_path.exists():
    from IPython.display import Image, display
    display(Image(str(fig_path)))
else:
    print(f"missing: {fig_path}")
"""),
    md(r"""
## Summary

Pipeline:
1. 5-dataset chosen-cell sweep (`L=26 K=8 α=1.0`, OneVision Main) →
   `sweep_subspace_<ds>_<scope>_chosen/predictions.jsonl`
2. `build_e6_stage4_bootstrap_ci.py` paired-bootstrap (B=10,000) →
   `docs/insights/_data/stage4_final_per_dataset_ci.{csv,md}`
3. `build_paper_stage4_paired_ci_figure.py` →
   `docs/figures/paper_6_2_3_stage4_5dataset_paired_ci.png`

Headline numbers (paper §6.2.2, Δem(b) sign-clean column):
- TallyQA  +13.8 [+12.9, +14.8]  (held-out)
- PlotQA    +4.7 [+3.8, +5.7]    (within calibration distribution)
- InfoVQA   +9.0 [+6.3, +11.7]   (within)
- ChartQA   +7.1 [+3.6, +10.7]   (held-out)
- MathVista +9.4 [+4.7, +14.7]   (held-out)

5/5 sign-clean under both 95 % and Bonferroni-20 corrected CIs.

§6.2 numbers are also reproducible from the §5.2 reproduction's
`outputs/paper/section_5_e6_steering/_data/p4_layer_sweep_per_cell_ci.csv`
at the chosen-cell row (L=26 K=8 α=1.0) for each dataset — match within
bf16 precision.
"""),
]


# =============================================================================
# Notebook 3 — §6.3 Capability preservation
# =============================================================================

cells_63: list[dict] = _setup_cells(
    title="Paper §6.3 — Capability preservation (6-benchmark VLMEvalKit)",
    subtitle=(
        "Reproduces the §6.3 STRICT_FREE_LUNCH headline: at the same chosen "
        "cell `L=26 K=8 α=1.0`, the mitigation applied at inference does "
        "not degrade general capability across 6 held-out VLMEvalKit "
        "benchmarks (n_total ≈ 10,507). Macro Δ = +0.41 pp; HallusionBench "
        "+2.21 [+1.14, +3.28] is the lone CI-clean positive (hallucination "
        "axis transfer)."
    ),
    scope_doc="§6.3 (Capability preservation)",
)

cells_63 += [
    md(r"""
## 2 · Capability eval — 6 benchmarks via VLMEvalKit

`scripts/run_capability_eval.py` is a per-benchmark interleaving driver
that runs baseline (vanilla `LLaVA-OneVision-HF`) and `+mit`
(`LLaVAOneVisionMitigated` wrapper that hooks the K=8 subspace projection
at L=26) on each benchmark sequentially. Live progress lands in
`outputs/capability_eval/progress.log` and a partial CSV is refreshed
after every benchmark.

**Wall time.** ~13 h on 1 GPU for the full 6-benchmark sweep
(POPE n=5127 is the long pole; HallusionBench n=951 is the first
CI-clean signal at ~2 h). Smoke-mode (`--max-questions 50`) finishes
in ~30 min.
"""),
    code(r"""
def run_capability_eval(config: str = "capability_eval.yaml",
                         max_questions: int | None = None):
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_capability_eval.py"),
        "--config", str(CONFIGS / config),
    ]
    if max_questions is not None:
        cmd += ["--max-questions", str(max_questions)]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


# Default: full 6-benchmark eval (~13 h). Toggle smoke=True for a
# 50-question subset (~30 min) to validate the pipeline before commit.
smoke = False
run_capability_eval(max_questions=50 if smoke else None)
"""),
    md(r"""
## 3 · Aggregate partial → final
"""),
    code(r"""
def aggregate_capability():
    partial_csv = MAIN / "docs" / "insights" / "_data" / "capability_eval_partial.csv"
    partial_md  = MAIN / "docs" / "insights" / "_data" / "capability_eval_partial.md"
    final_csv   = DATA_DIR / "capability_eval_per_benchmark.csv"
    final_md    = DATA_DIR / "capability_eval_per_benchmark.md"
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "aggregate_capability_eval.py"),
        "finalize",
        "--partial-csv", str(partial_csv),
        "--partial-md",  str(partial_md),
        "--final-csv",   str(final_csv),
        "--final-md",    str(final_md),
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


aggregate_capability()
"""),
    md(r"""
## 4 · Table 6.3 — Per-benchmark capability deltas
"""),
    code(r"""
cap_csv = DATA_DIR / "capability_eval_per_benchmark.csv"
if cap_csv.exists():
    df = pd.read_csv(cap_csv)
    # Render in the same column order as paper Table 6.3.
    df = df.assign(
        baseline=df["acc_baseline"] * 100,
        mit=df["acc_mit"] * 100,
        delta_pp=df["delta"] * 100,
        ci_low_pp=df["ci_low"] * 100,
        ci_high_pp=df["ci_high"] * 100,
    )
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Benchmark": r["benchmark"],
            "n":         int(r["n"]),
            "baseline":  f"{r['baseline']:.2f}",
            "+mit":      f"{r['mit']:.2f}",
            "Δ (pp)":    f"{r['delta_pp']:+.2f}",
            "95% CI":    f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]",
            "status":    r["status"],
        })
    macro_delta = df["delta"].mean() * 100
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(f"Macro Δ  = {macro_delta:+.2f} pp   (STRICT_FREE_LUNCH protocol)")
else:
    print(f"missing: {cap_csv}")
"""),
    md(r"""
## 5 · Figure §6.3 — capability bar chart (optional)

A bar chart of per-benchmark Δ with 95 % CI whiskers — paper publishes
Table 6.3 only, so this figure is for the appendix / supplementary slot.
"""),
    code(r"""
def fig_capability() -> plt.Figure | None:
    src = DATA_DIR / "capability_eval_per_benchmark.csv"
    if not src.exists():
        print(f"  (skipped — {src} missing)")
        return None
    df = pd.read_csv(src)
    # Order by absolute Δ descending so signal benchmarks lead.
    df = df.assign(absd=df["delta"].abs()).sort_values("absd", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    y = np.arange(len(df))
    deltas = df["delta"].values * 100
    yerr_lo = deltas - df["ci_low"].values * 100
    yerr_hi = df["ci_high"].values * 100 - deltas
    colors = ["#2ca25f" if (lo > 0 and hi > 0) else "#9e9e9e"
              for lo, hi in zip(df["ci_low"]*100, df["ci_high"]*100)]
    ax.barh(y, deltas, color=colors, alpha=0.7, edgecolor="black", linewidth=0.4)
    ax.errorbar(deltas, y, xerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="black", linewidth=0.8, capsize=3)
    ax.axvline(0, color="#888", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(df["benchmark"].values)
    ax.set_xlabel("Δ accuracy (pp)  — mitigation vs baseline")
    ax.set_title("§6.3 — Capability preservation across 6 VLMEvalKit benchmarks\n"
                 "STRICT_FREE_LUNCH; green = 95 % CI excludes 0 (positive)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    return fig


fig = fig_capability()
if fig is not None:
    save_figure(fig, "paper_6_3_capability_eval")
fig
"""),
    md(r"""
## Summary

Pipeline:
1. `run_capability_eval.py` — per-benchmark baseline + `+mit` runs via
   VLMEvalKit programmatic API. ~13 h wall on 1 GPU for the full 6-bench
   set; live partial CSV at every checkpoint.
2. `aggregate_capability_eval.py finalize` →
   `docs/insights/_data/capability_eval_per_benchmark.{csv,md}`.
3. Table 6.3 rendered from the canonical CSV.

Headline (paper §6.3):
- Macro Δ = **+0.41 pp** across 6 benchmarks (n_total = 10,507).
- HallusionBench **+2.21 pp [+1.14, +3.28]** — only CI-clean positive
  (hallucination-axis transfer; mitigation incidentally improves
  visual-distraction handling).
- 5 other benchmarks within ±1 pp band (POPE −0.06, MMStar +0.13,
  RealWorldQA +1.31, MMBench-DEV-EN −0.34, OCRBench −0.80).

Verdict: STRICT_FREE_LUNCH — no benchmark regresses with a CI-clean
negative. Mitigation preserves general capability while reducing
anchoring (§6.2).
"""),
]


# =============================================================================
# Write all three notebooks
# =============================================================================

def _nb(cells: list[dict]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


for stem, cells in [
    ("paper_section_6_1_method", cells_61),
    ("paper_section_6_2_anchoring_reduction", cells_62),
    ("paper_section_6_3_capability_preservation", cells_63),
]:
    out = NB_DIR / f"{stem}.ipynb"
    nbf.write(_nb(cells), out)
    print(f"Wrote {out}  ({len(cells)} cells)")
