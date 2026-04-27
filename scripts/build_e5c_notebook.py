"""Construct notebooks/E5c_anchor_mask_control.ipynb via nbformat."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# E5c - Anchor-mask control (reproducer)\n"
        "\n"
        "Top-to-bottom reproducer for the E5c experiment, an extension of E5b\n"
        "with two additional irrelevant-image arms:\n"
        "\n"
        "- **anchor**: original digit image at distance stratum S1..S5 (E5b condition).\n"
        "- **masked**: pixel-masked variant of the same digit image, S1..S5.\n"
        "- **neutral**: digit-free distractor image, single across-distances bucket.\n"
        "\n"
        "12 conditions total: target_only + 5 anchor + 5 masked + 1 neutral.\n"
        "\n"
        "Headline metric: **paired conditional adoption** (case 4 `base=a=pred` excluded\n"
        "from the denominator). Stratified by (dataset, stratum, base_correct, condition_type).\n"
        "\n"
        "The two questions this answers:\n"
        "1. **Digit-pixel causality** — does masking the digit kill the anchoring effect?\n"
        "   Comparison: anchor vs masked at the same stratum.\n"
        "2. **Generic 2-image distraction** — is masked > neutral, i.e. does merely having a\n"
        "   second image (with structure but no digits) move the model? Comparison:\n"
        "   masked vs neutral.\n"
        "\n"
        "All heavy lifting in `scripts/analyze_e5c_distance.py` - this notebook just invokes it and displays."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT / 'scripts'))\n"
        "from analyze_e5c_distance import run\n"
        "out = run()\n"
        "summary = out['summary']\n"
        "print(f\"loaded {out['n_records']} records, wrote {out['out_csv']}\")"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Per-cell summary\n"
        "\n"
        "Rows: (dataset, stratum, base, condition_type). For anchor and masked, `stratum`\n"
        "iterates S1..S5; for neutral the row collapses to a single `stratum=all` bucket.\n"
        "\n"
        "`adopt_cond = case2 / (case1+case2+case3)` — fraction of eligible records where the\n"
        "model moved to the anchor digit (paired definition, case 4 excluded). For neutral\n"
        "rows there is no anchor digit, so `adopt_cond` is structurally 0 and serves as the\n"
        "baseline floor."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "pd.set_option('display.float_format', '{:0.4f}'.format)\n"
        "summary"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Anchor vs masked-anchor (wrong-base only)\n"
        "\n"
        "If digit-pixel content is what causes anchoring, masked should sit substantially\n"
        "below anchor at every stratum (especially S1 where the anchor is closest to GT)."
    ))

    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5c_anchor_vs_masked.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Three-way comparison (wrong-base only)\n"
        "\n"
        "Anchor and masked across S1..S5, with neutral drawn as a horizontal reference\n"
        "(stratum-collapsed). The vertical separation between the three lines decomposes the\n"
        "anchoring effect into (a) digit-pixel content (anchor − masked) and (b) generic\n"
        "2-image distraction (masked − neutral)."
    ))

    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5c_three_way_comparison.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Anchor adoption: correct-base vs wrong-base (sanity)\n"
        "\n"
        "E5b found that anchor adoption is gated on the model being uncertain (wrong-base).\n"
        "This panel re-checks the gating in the E5c re-run."
    ))

    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5c_correct_vs_wrong.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Sanity check - wrong-base S1 anchor vs masked vs neutral\n"
        "\n"
        "Headline numbers in one place. Expectation: anchor S1 stays near E5b's ~0.13 (VQAv2)\n"
        "/ ~0.092 (TallyQA); masked S1 falls substantially; neutral is 0 by construction."
    ))

    cells.append(nbf.v4.new_code_cell(
        "for ds in ['VQAv2', 'TallyQA']:\n"
        "    cell = summary[(summary['dataset']==ds) & (summary['base']=='wrong')]\n"
        "    a = cell[(cell['condition_type']=='anchor') & (cell['stratum']=='S1')]['adopt_cond'].iloc[0]\n"
        "    m = cell[(cell['condition_type']=='masked') & (cell['stratum']=='S1')]['adopt_cond'].iloc[0]\n"
        "    n = cell[(cell['condition_type']=='neutral')]['adopt_cond'].iloc[0]\n"
        "    print(f'{ds:<8s} wrong-base S1: anchor={a:.4f}, masked={m:.4f}, neutral(all)={n:.4f}, anchor/masked={a/m if m>0 else float(\"inf\"):.2f}x')\n"
        "print()\n"
        "print(out['headline'])"
    ))

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).resolve().parents[1] / "notebooks" / "E5c_anchor_mask_control.ipynb"
    nbf.write(nb, out_path)
    print(f"wrote {out_path.relative_to(out_path.parents[1])}")
