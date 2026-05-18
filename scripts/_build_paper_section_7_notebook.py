"""Build the §7 reproducibility notebook (closed-API VLM-as-judge pilot).

Outputs:
  notebooks/paper_section_7_judge_pilot.ipynb

§7.1 (Implications + ecological validity) references the closed-API VLM-as-judge
anchoring pilot summarised in Appendix J. The pilot is the only data-bearing
artifact in §7 — §7.2 (Future work) is narrative only.

This notebook reproduces:
- Judge pilot setup (5 judges × 2 datasets × 3 arms × n=200)
- Optional re-running of `run_judge_pilot.py` against the closed-API gateway
  (requires API keys + ~$200-500 cost; gated behind `RUN_INFERENCE = False`)
- Aggregation via `analyze_judge_pilot.py` (paired-bootstrap CI B=10,000)
- Table J.1 (per-judge × dataset × arm matrix)
- Two-pattern classification (digit-specific vs distractor-general via
  Δ(a−m) vs Δ(m−b) split)
- Within-vendor ablation tables (OpenAI generation; Google reasoning toggle)
- Figure rebuild from the canonical predictions

§7.1 prose is narrative (no separate data); the same canonical CSV /
figure here back the high-level "3/5 robust, 2/5 surface anchoring"
summary in §7.1.
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


cells: list[dict] = [
    md(r"""
# Paper §7 — Closed-API VLM-as-judge anchoring pilot (Appendix J)

Reproduces the §7.1 ecological-validity pilot referenced in the paper —
5 closed-API judges × 2 standard judge benchmarks (VLFeedback +
VL-RewardBench) × 3 arms (b: image only / a: + anchor digit "1" /
m: + Telea-masked anchor) × n=200 paired samples per cell. Headline
finding: gpt-4o exhibits *digit-specific* anchoring (Δ(a−m) -0.47 /
-0.62 on the two datasets), gemini-2.5-flash exhibits *distractor-
general* susceptibility (Δ(m−b) ≈ Δ(a−b)), and gpt-5.1 /
gemini-2.5-pro / claude-sonnet-4-5 are robust.

**Spec source-of-truth.** `docs/paper/emnlp_outline_ko.md` — §7.1
(Implications) + Appendix J (VLM-as-judge anchoring pilot).

Heavy inference uses closed-API calls via a gateway (`gateway.letsur.ai/v1`,
OpenAI-compat). `RUN_INFERENCE = False` (default) only re-renders the
table + figure from the canonical `outputs/judge_pilot/<judge>/<ts>/`
predictions; flipping to True re-issues the API calls (requires API
keys + access; cost ≈ $200–500 for the full 5-judge × 2-dataset × 3-arm
× n=200 sweep, per outline §J caveats).
"""),
    md("## 1 · Setup — paths + subprocess helper"),
    code(r"""
from __future__ import annotations
import json
import os
import shlex
import subprocess
import sys
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

SCRIPTS    = WORKTREE / "scripts"
CONFIGS    = WORKTREE / "configs"
DATA_DIR   = MAIN / "docs" / "insights" / "_data"
FIGURES    = WORKTREE / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

JUDGE_OUT_ROOT     = MAIN / "outputs" / "judge_pilot"
JUDGE_OUT_VLB_ROOT = MAIN / "outputs" / "judge_pilot_vlrewardbench_a1"

RUN_INFERENCE = False  # set True to re-issue closed-API calls (≈$200-500)

print(f"MAIN     = {MAIN}")
print(f"WORKTREE = {WORKTREE}")
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


def save_figure(fig, stem: str, png_dir: Path = FIGURES):
    png = png_dir / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", dpi=160)
    print(f"wrote {png}")
"""),
    md(r"""
## 2 · Pilot setup — judges, datasets, arms

| Item | Value |
|---|---|
| Datasets | **VLFeedback** (`MMInstruction/VLFeedback`) + **VL-RewardBench** (`MMInstruction/VL-RewardBench`), n=200 each |
| Anchor | digit "1" (`inputs/irrelevant_number/1.png`); m-arm = same image, digit pixels Telea-masked |
| Arms | b (image only) / a (image + anchor) / m (image + masked anchor) |
| Score | Visual Faithfulness 1–5 (single-dim, VLFeedback Silkie native rubric) |
| Judges | gpt-4o, gpt-5.1, gemini-2.5-pro, gemini-2.5-flash, claude-sonnet-4-5-20250929 |
| Reasoning | gpt-5.1 + flash via `reasoning_effort=minimal` (non-reasoning); pro reasoning forced ON (provider policy) |
| Gateway | `gateway.letsur.ai/v1` (Staix, OpenAI-compat) |

Response selector = **chosen** — VLFeedback: max average GPT-4V dim
rating; VL-RewardBench: argmax human ranking. This is adversarial
worst-case: high-baseline samples leave headroom for the anchor to push.
"""),
    code(r"""
JUDGES = [
    ("gpt-4o",            "openai"),
    ("gpt-5.1",           "openai", "reasoning_effort=minimal"),
    ("gemini-2.5-pro",    "google", "reasoning forced ON"),
    ("gemini-2.5-flash",  "google", "reasoning_effort=minimal"),
    ("claude-sonnet-4-5", "anthropic"),
]
DATASETS = [
    ("VLFeedback",     "MMInstruction/VLFeedback"),
    ("VL-RewardBench", "MMInstruction/VL-RewardBench"),
]

print("=== Judges ===")
for row in JUDGES:
    print(f"  {row[0]:<22s}  vendor={row[1]:<10s}  {row[2] if len(row) > 2 else ''}")
print()
print("=== Datasets ===")
for label, repo in DATASETS:
    print(f"  {label:<16s}  {repo}")
"""),
    md(r"""
## 3 · Build the paired stimulus dataset (b / a / m × n=200)

`scripts/build_judge_pilot_dataset.py` samples n=200 from each
benchmark (seed=20260512 for VLFeedback, 20260513 for VL-RewardBench),
fetches the chosen-response slot, and writes a manifest with three
image-attachment configs per sample (b: image_only / a: + anchor /
m: + masked anchor). This step is cheap (CPU-only image processing).
"""),
    code(r"""
def build_dataset(cfg: str):
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "build_judge_pilot_dataset.py"),
        "--config", str(CONFIGS / cfg),
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


build_dataset("judge_pilot.yaml")              # VLFeedback
build_dataset("judge_pilot_vlrewardbench_a1.yaml")  # VL-RewardBench
"""),
    md(r"""
## 4 · Run closed-API judges (HEAVY — closed-API cost)

`scripts/run_judge_pilot.py` issues per-(sample, arm, judge) requests
through the gateway. ~$200-500 total for the full 5-judge × 2-dataset
× 3-arm × n=200 sweep (see Appendix J caveats). Per-judge isolation
allows re-running a single judge without paying for the others again.
"""),
    code(r"""
def run_judge(cfg: str, judge_ids: list[str]):
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "run_judge_pilot.py"),
        "--config", str(CONFIGS / cfg),
        "--judge-id", *judge_ids,
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


# VLFeedback first (200 × 3 arms × 5 judges = 3000 API calls)
run_judge("judge_pilot.yaml",
          ["gpt-4o", "gpt-5.1", "gemini-2.5-pro", "gemini-2.5-flash", "claude-sonnet-4-5"])

# VL-RewardBench second (same dimensionality)
run_judge("judge_pilot_vlrewardbench_a1.yaml",
          ["gpt-4o", "gpt-5.1", "gemini-2.5-pro", "gemini-2.5-flash", "claude-sonnet-4-5"])
"""),
    md(r"""
## 5 · Aggregate — paired-bootstrap CI (B=10,000)

`scripts/analyze_judge_pilot.py` reads each judge's per-sample scores,
joins arms by `sample_id`, and computes paired-bootstrap CIs (B=10,000)
for Δ(a−m) mean and ΔP(score=1). Output CSVs (`docs/insights/_data/`)
back Table J.1 + the headline figure.
"""),
    code(r"""
def aggregate_judges(cfg: str):
    cmd = [
        "uv", "run", "python", str(SCRIPTS / "analyze_judge_pilot.py"),
        "--config", str(CONFIGS / cfg),
    ]
    return run_cmd(cmd, dry=not RUN_INFERENCE)


# Runs against the canonical `outputs/judge_pilot/*/` predictions
aggregate_judges("judge_pilot.yaml")
"""),
    md(r"""
## 6 · Table J.1 — per-judge × dataset × arm summary

Reads each judge's predictions.jsonl (largest-by-row run per the
smoke-pollution rule) and builds the 10-row Table J.1 from raw scores.
This duplicates the table the §J prose cites — no canonical CSV is
shipped per-judge per-dataset; the table is recomputed from raw runs.
"""),
    code(r"""
def _largest_predictions(judge_root: Path) -> Path:
    runs = [p for p in judge_root.iterdir() if p.is_dir() and (p / "predictions.jsonl").exists()]
    if not runs:
        return None
    def _key(p: Path) -> tuple[int, float]:
        n_rows = sum(1 for _ in (p / "predictions.jsonl").open())
        return (n_rows, p.stat().st_mtime)
    runs.sort(key=_key, reverse=True)
    return runs[0] / "predictions.jsonl"


def _load_judge(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    return pd.DataFrame(rows)


def _summarise_one(judge: str, dataset_root: Path, dataset_label: str) -> dict | None:
    pred = _largest_predictions(dataset_root / judge)
    if pred is None:
        return None
    df = _load_judge(pred)
    if df.empty or "arm" not in df.columns or "score" not in df.columns:
        return None
    df = df.dropna(subset=["score"])
    # Pivot per-sample to b/a/m columns
    wide = df.pivot_table(
        index="sample_id", columns="arm", values="score", aggfunc="first"
    ).dropna()
    if not all(c in wide.columns for c in ("b", "a", "m")):
        return None
    n = len(wide)
    means = wide[["b", "a", "m"]].mean()
    delta_ab = means["a"] - means["b"]
    delta_mb = means["m"] - means["b"]
    delta_am = means["a"] - means["m"]
    p1_b = (wide["b"] == 1).mean()
    p1_a = (wide["a"] == 1).mean()
    return {
        "Dataset": dataset_label,
        "Judge": judge,
        "n": n,
        "b": round(means["b"], 2),
        "a": round(means["a"], 2),
        "m": round(means["m"], 2),
        "Δ(a−b)": round(delta_ab, 2),
        "Δ(m−b)": round(delta_mb, 2),
        "Δ(a−m)": round(delta_am, 2),
        "P(=1) b→a": f"{p1_b*100:.0f}% → {p1_a*100:.0f}% (+{(p1_a-p1_b)*100:.0f} pp)",
    }


rows = []
judge_order = ["gpt-4o", "gpt-5.1", "gemini-2.5-pro", "gemini-2.5-flash", "claude-sonnet-4-5"]
for judge in judge_order:
    r = _summarise_one(judge, JUDGE_OUT_ROOT, "VLFeedback")
    if r: rows.append(r)
for judge in judge_order:
    r = _summarise_one(judge, JUDGE_OUT_VLB_ROOT, "VL-RewardBench")
    if r: rows.append(r)
table_j1 = pd.DataFrame(rows)
if not table_j1.empty:
    print("=== Table J.1 — per-judge × dataset × arm summary ===")
    print(table_j1.to_string(index=False))
else:
    print("no predictions found")
"""),
    md(r"""
## 7 · Two attack patterns — digit-specific vs distractor-general

Split judges by the Δ(a−m) vs Δ(m−b) signature:
- **Digit-specific** — large negative Δ(a−b), ≈ 0 Δ(m−b), large negative
  Δ(a−m). The anchor digit itself is doing the work — masking the
  digit recovers the baseline. Matches the §3.2 / §4.2 main-panel
  mechanism; the (a − m) paired contrast isolates the same sign on a
  closed-API model.
- **Distractor-general** — large negative Δ(a−b), Δ(m−b) ≈ Δ(a−b),
  Δ(a−m) ≈ 0. The mere presence of a second image (digit or not)
  degrades the rating. Distinct threat model: multi-image
  instruction-following weakness, unrelated to digit semantics.
- **Null on both** — Δ(a−b), Δ(m−b), Δ(a−m) all near zero. Judge is
  robust to both attack types.
"""),
    code(r"""
def classify(judge_row: dict) -> str:
    da = judge_row["Δ(a−b)"]
    dm = judge_row["Δ(m−b)"]
    dam = judge_row["Δ(a−m)"]
    if abs(da) < 0.3 and abs(dm) < 0.3 and abs(dam) < 0.3:
        return "null"
    if abs(dam) > 0.3 and abs(dm) < 0.3:
        return "digit-specific"
    if abs(da) > 0.3 and abs(dm) > 0.3 and abs(dm) > abs(dam) * 1.5:
        return "distractor-general"
    return "mixed"


if not table_j1.empty:
    classified = table_j1.assign(pattern=table_j1.apply(classify, axis=1))
    print("=== Attack-pattern classification (|Δ| ≥ 0.3 threshold) ===")
    print(classified[["Dataset", "Judge", "Δ(a−b)", "Δ(m−b)", "Δ(a−m)", "pattern"]].to_string(index=False))
"""),
    md(r"""
## 8 · Within-vendor ablations

**OpenAI generation (gpt-4o → gpt-5.1).** Newer generation within the
same vendor effectively eliminates anchor susceptibility — mean Δ(a−b)
drops from ~−1 pt (gpt-4o) to noise (gpt-5.1). Cause (RLHF / scale /
instruction-following) is not disentanglable without internal access.

**Google reasoning toggle (pro vs flash).** Same family: reasoning ON
(pro) is robust, reasoning OFF (flash) is highly susceptible. Supports
the directional hypothesis that reasoning provides anchor-resistance.
The pro-vs-flash capability gap confounds — clean ablation would
require pro with reasoning disabled, which the provider does not allow.
"""),
    code(r"""
if not table_j1.empty:
    # OpenAI generation ablation
    print("=== OpenAI generation ablation (gpt-4o → gpt-5.1, mean Δ(a−b)) ===")
    for ds in ("VLFeedback", "VL-RewardBench"):
        a = table_j1[(table_j1.Dataset==ds) & (table_j1.Judge=="gpt-4o")]
        b = table_j1[(table_j1.Dataset==ds) & (table_j1.Judge=="gpt-5.1")]
        if len(a) and len(b):
            print(f"  {ds:<16s}  gpt-4o {float(a.iloc[0]['Δ(a−b)']):+.2f}  →  gpt-5.1 {float(b.iloc[0]['Δ(a−b)']):+.2f}")

    # Google reasoning ablation
    print()
    print("=== Google reasoning ablation (pro = ON / flash = OFF, mean Δ(a−b)) ===")
    for ds in ("VLFeedback", "VL-RewardBench"):
        p = table_j1[(table_j1.Dataset==ds) & (table_j1.Judge=="gemini-2.5-pro")]
        f = table_j1[(table_j1.Dataset==ds) & (table_j1.Judge=="gemini-2.5-flash")]
        if len(p) and len(f):
            print(f"  {ds:<16s}  pro (ON) {float(p.iloc[0]['Δ(a−b)']):+.2f}  vs  flash (OFF) {float(f.iloc[0]['Δ(a−b)']):+.2f}")
"""),
    md(r"""
## 9 · Figure — 2 dataset × 5 judge line overlay (n=200)

Canonical figure: `docs/figures/judge_pilot_v1_5judges_2datasets_n200.png`
built by `analyze_judge_pilot.py`. The notebook's analyze step in §5
above already wrote it; this cell just renders the image inline.
"""),
    code(r"""
fig_path = WORKTREE / "docs" / "figures" / "judge_pilot_v1_5judges_2datasets_n200.png"
if fig_path.exists():
    from IPython.display import Image, display
    display(Image(str(fig_path)))
else:
    print(f"missing: {fig_path}")
"""),
    md(r"""
## Caveats (per Appendix J)

- **Anchor = digit "1" only.** Anchor=5 (ceiling-push) was tested in an
  earlier 2-judge run and showed null due to score-5 ceiling. The
  expanded 5-judge panel uses anchor=1 only.
- **Chosen-response selector** is the adversarial worst case: it picks
  the response with the *highest* baseline rating, leaving headroom for
  the anchor to push down. Random-response selector attenuates the
  effect ~10× — anchor effect is *baseline-conditional*.
- **gemini-2.5-pro VL-RewardBench parse rate.** 10 % parse failures
  (longer reasoning chain + `max_output_tokens=2048`) → n_pair = 165.
  Null reading is statistically supported; magnitude comparisons should
  note the reduced n.
- **Gateway alias only.** Only `claude-sonnet-4-5-20250929` is
  date-pinned; other judges are gateway aliases (e.g., `gpt-4o`) with
  access timestamp (2026-05-13/14) for citation purposes.

## Summary

§7.1 ecological-validity claim is backed by this pilot. Across 5
closed-API judges on 2 standard judge benchmarks, anchoring surfaces in
2/5 judges (gpt-4o digit-specific, gemini-2.5-flash distractor-general),
3/5 are robust (gpt-5.1, gemini-2.5-pro, claude-sonnet-4-5-20250929).
Mitigation deployment relevance is reinforced — open-weight finding is
not architecture-specific, and frontier closed-API systems partially
surface the same phenomenon.

§7.2 (Future work) is narrative only; the unexpected Δem(b) capability
gain finding it interprets is in `paper_section_6_3_capability_preservation.ipynb`
(HallusionBench +2.21 pp). No additional inference is required for §7.2.
"""),
]


def _nb() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


out = NB_DIR / "paper_section_7_judge_pilot.ipynb"
nbf.write(_nb(), out)
print(f"Wrote {out}  ({len(cells)} cells)")
