"""Construct notebooks/E5b_anchor_distance.ipynb via nbformat."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# E5b - Anchor-distance robustness sweep (reproducer)\n"
        "\n"
        "Top-to-bottom reproducer for `docs/experiments/E5b-anchor-distance-design.md`.\n"
        "\n"
        "Headline metric: **paired conditional adoption** (M1 paired definition, denominator excludes case 4 `base=a=pred`).\n"
        "\n"
        "Key finding: adoption is **uncertainty-modulated AND plausibility-windowed**.\n"
        "- Correct-base records show essentially no anchor effect (~0.01-0.10 across all distances).\n"
        "- Wrong-base records show sharp plausibility-windowed pattern: adoption peaks at S2 [2,5] for VQAv2 and decays to ~0 at S4/S5 in both datasets.\n"
        "\n"
        "All heavy lifting in `scripts/analyze_e5b_distance.py` - this notebook just invokes it and displays."
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
        "## Per-cell summary\n"
        "\n"
        "Rows: (dataset, stratum, base) where base is 'correct' or 'wrong'. "
        "`adopt_cond = case2 / (case1+case2+case3)` — fraction of eligible records where the model moved to the anchor."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "pd.set_option('display.float_format', '{:0.4f}'.format)\n"
        "summary"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Headline figure - adoption vs distance, base correctness split\n"
        "\n"
        "Two panels (VQAv2, TallyQA). Each panel has two lines: correct-base (flat) and wrong-base (peaked at S2)."
    ))

    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5b_adopt_cond_curve.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Cross-dataset overlay (wrong-base only)\n"
        "\n"
        "Both datasets show the same plausibility-windowed adoption pattern when the model is uncertain."
    ))

    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5b_adopt_cond_overlay.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Sanity check - wrong-base S1 vs S5 ratio\n"
        "\n"
        "If plausibility-windowed adoption is real, the wrong-base S1/S2 cells should be much higher than S4/S5."
    ))

    cells.append(nbf.v4.new_code_cell(
        "for ds in ['VQAv2', 'TallyQA']:\n"
        "    cell = summary[(summary['dataset']==ds) & (summary['base']=='wrong')].set_index('stratum')\n"
        "    s1 = cell.loc['S1', 'adopt_cond']\n"
        "    s5 = cell.loc['S5', 'adopt_cond']\n"
        "    ratio = s1 / s5 if s5 > 0 else float('inf')\n"
        "    print(f'{ds:<8s} wrong-base adopt_cond: S1={s1:.4f}, S5={s5:.4f}, S1/S5 ratio={ratio:.1f}')"
    ))

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).resolve().parents[1] / "notebooks" / "E5b_anchor_distance.ipynb"
    nbf.write(nb, out_path)
    print(f"wrote {out_path.relative_to(out_path.parents[1])}")
