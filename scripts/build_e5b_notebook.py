"""Construct notebooks/E5b_anchor_distance.ipynb via nbformat.

Run once to (re)generate the notebook source; then execute the notebook
itself (`jupyter nbconvert --to notebook --execute --inplace`) to embed
output cells. Same builder + execute pattern as other E*-notebooks.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(nbf.v4.new_markdown_cell(
        "# E5b — Anchor-distance robustness sweep (reproducer)\n"
        "\n"
        "Top-to-bottom reproducer for `docs/experiments/E5b-anchor-distance-design.md`.\n"
        "Reads the latest VQAv2 + TallyQA stratified runs under "
        "`outputs/experiment_distance_*/llava-next-interleaved-7b/<timestamp>/predictions.jsonl` "
        "and regenerates:\n"
        "\n"
        "1. Per-stratum summary table (direction-follow / adoption / EM / mean distance, with 95% bootstrap CI).\n"
        "2. Distance curve per dataset.\n"
        "3. Cross-dataset overlay.\n"
        "\n"
        "All heavy lifting lives in `scripts/analyze_e5b_distance.py` — this notebook just invokes it and displays the outputs."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT / 'scripts'))\n"
        "from analyze_e5b_distance import run\n"
        "out = run()\n"
        "summary = out['summary']\n"
        "print(f\"loaded {out['n_records']} records, wrote {out['out_csv']}\")"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Per-stratum summary\n"
        "\n"
        "Each row = one (dataset, stratum) cell. `direction_follow_rate` is the headline number; "
        "`df_minus_baseline` subtracts the per-dataset target_only direction-follow (≈0 by construction)."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "pd.set_option('display.float_format', '{:0.3f}'.format)\n"
        "summary"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Distance curve per dataset"
    ))

    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5b_distance_curve.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Cross-dataset overlay\n"
        "\n"
        "Two lines on one axis. The cutoff `d*` for paper-headline subset is the largest stratum where "
        "the effect remains > 50% of S1's effect."
    ))

    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5b_cross_dataset_overlay.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Sanity check vs main run\n"
        "\n"
        "VQAv2 main run (random anchor 0–9, n=17,730) had `direction_follow_rate = 0.348` on llava-interleave-7b. "
        "If E5b's anchor-distance distribution were uniform random over 0–9, the pooled effect would land near 0.348. "
        "The stratified sweep oversamples far strata (S4/S5), so we expect the pooled E5b direction-follow to be **lower** than 0.348."
    ))

    cells.append(nbf.v4.new_code_cell(
        "vqa_pool = summary[summary['dataset'] == 'VQAv2']\n"
        "n_total = vqa_pool['n'].sum()\n"
        "weighted_df = float((vqa_pool['direction_follow_rate'] * vqa_pool['n']).sum() / n_total)\n"
        "print(f'E5b VQAv2 pooled direction-follow (n={n_total}): {weighted_df:.3f}')\n"
        "print(f'reference — main run random-anchor 0..9 (n=17,730):  0.348')"
    ))

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).resolve().parents[1] / "notebooks" / "E5b_anchor_distance.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_path)
    print(f"wrote {out_path.relative_to(out_path.parents[1])}")
