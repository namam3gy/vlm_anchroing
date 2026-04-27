"""Construct notebooks/E5c_anchor_mask_control.ipynb via nbformat."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# E5c - Anchor mask control (reproducer)\n"
        "\n"
        "Top-to-bottom reproducer for `docs/experiments/E5c-anchor-mask-control.md`.\n"
        "\n"
        "## Design\n"
        "\n"
        "Four conditions per question:\n"
        "1. `target_only` (base image)\n"
        "2. `target_plus_irrelevant_number_S{1..5}` (base + anchor image, digit visible)\n"
        "3. `target_plus_irrelevant_number_masked_S{1..5}` (base + same anchor scene with the digit pixel region inpainted out)\n"
        "4. `target_plus_irrelevant_neutral` (base + a generic neutral image; no digit, unrelated content)\n"
        "\n"
        "## Comparisons of interest\n"
        "\n"
        "- `(1, 2, 3)` - digit pixel causality. (1->2) total anchor effect; (1->3) effect with digit removed; (2 - 3) = pure digit-pixel contribution.\n"
        "- `(1, 3, 4)` - anchor-image-background contribution beyond generic 2-image distraction. (1->3) and (1->4) should match if the anchor scene's background offers no extra info.\n"
        "\n"
        "## Headline metric\n"
        "\n"
        "`adopt_cond` (paired conditional anchor adoption, M1; denominator = case 1 + 2 + 3 = excludes case 4 `base==a==pred`). Undefined for neutral (no anchor value).\n"
        "\n"
        "Direction-follow (`df_uncond`, `df_cond`) and accuracy drop (`acc_drop`) are also reported so the user can see the (1,3,4) `acc_drop` comparison and the `df` quirk that both anchor and masked produce nearly the same `df_cond` on wrong-base records (model uncertainty, not digit-specific signal).\n"
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT / 'scripts'))\n"
        "from analyze_e5c_distance import run\n"
        "out = run()\n"
        "summary = out['summary']\n"
        "print(f\"loaded {out['n_records']} records, wrote {out['out_csv']}\")\n"
        "import pandas as pd\n"
        "pd.set_option('display.float_format', '{:0.4f}'.format)"
    ))

    cells.append(nbf.v4.new_markdown_cell("## base = all"))
    cells.append(nbf.v4.new_code_cell(
        "summary[summary['base']=='all'][['dataset','condition_type','stratum','n','adopt_cond','df_uncond','df_cond','acc_drop']]"
    ))

    cells.append(nbf.v4.new_markdown_cell("## base = correct"))
    cells.append(nbf.v4.new_code_cell(
        "summary[summary['base']=='correct'][['dataset','condition_type','stratum','n','adopt_cond','df_uncond','df_cond','acc_drop']]"
    ))

    cells.append(nbf.v4.new_markdown_cell("## base = wrong"))
    cells.append(nbf.v4.new_code_cell(
        "summary[summary['base']=='wrong'][['dataset','condition_type','stratum','n','adopt_cond','df_uncond','df_cond','acc_drop']]"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Figure 1 - anchor vs masked x base correctness, adopt_cond\n"
        "\n"
        "Anchor > masked at S1 on both datasets and both base subsets. The gap (anchor - masked) is the pure digit-pixel contribution to paired adoption."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5c_anchor_vs_masked_adopt.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Figure 2 - direction-follow does NOT isolate the digit\n"
        "\n"
        "On wrong-base records, `df_cond` is essentially identical for anchor and masked at S2 (~0.52 VQAv2, ~0.46 TallyQA) and within a few pp at S1. The model's prediction shifts toward the anchor side regardless of whether the digit is visible - the second image's *presence* is enough to perturb predictions for uncertain models. Use `adopt_cond` (where pred lands exactly on anchor) to get the digit-specific signal."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5c_anchor_vs_masked_df.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Figure 3 - `acc_drop`: masked ~ neutral << anchor (on the meaningful base subsets)\n"
        "\n"
        "Comparing target_only's accuracy to each treatment's accuracy on the SAME base subset:\n"
        "- `correct-base` (model knows the answer baseline-side): all three distractors hurt accuracy, but masked and neutral hurt about the same (~7-9 pp on VQAv2, ~2-5 pp on TallyQA). Anchor hurts measurably more.\n"
        "- `wrong-base`: `acc_drop` is negative (regression-to-mean: distraction nudges some wrong predictions back to correct). Not informative.\n"
        "- The masked image acts like a generic neutral distractor on accuracy. The anchor image's *background* offers no extra information beyond generic 2-image distraction."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5c_acc_drop_3way.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Figure 4 - sanity check vs E5b: uncertainty gate still holds\n"
        "\n"
        "Reproducing the E5b finding on the E5c run: anchor adopt_cond decays with distance, with wrong-base higher than correct-base."
    ))
    cells.append(nbf.v4.new_code_cell(
        "display(Image(filename=str(ROOT / 'docs' / 'figures' / 'E5c_correct_vs_wrong_adopt.png')))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Distilled digit-pixel gap\n"
        "\n"
        "Anchor - masked on wrong-base, paired conditional adoption."
    ))
    cells.append(nbf.v4.new_code_cell(
        "for ds in ['VQAv2', 'TallyQA']:\n"
        "    rows = summary[(summary['dataset']==ds) & (summary['base']=='wrong')]\n"
        "    anchor = rows[rows['condition_type']=='anchor'].set_index('stratum')['adopt_cond']\n"
        "    masked = rows[rows['condition_type']=='masked'].set_index('stratum')['adopt_cond']\n"
        "    print(f'\\n{ds}  wrong-base anchor - masked:')\n"
        "    for s in ['S1','S2','S3','S4','S5']:\n"
        "        print(f'  {s}: anchor={anchor.get(s, float(\"nan\")):.4f}  masked={masked.get(s, float(\"nan\")):.4f}  gap={anchor.get(s,0) - masked.get(s,0):+.4f}')"
    ))

    nb["cells"] = cells
    return nb


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).resolve().parents[1] / "notebooks" / "E5c_anchor_mask_control.ipynb"
    nbf.write(nb, out_path)
    print(f"wrote {out_path.relative_to(out_path.parents[1])}")
