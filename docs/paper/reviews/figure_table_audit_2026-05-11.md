# Figure / Table consistency audit — `docs/paper/emnlp_draft_ko.md`

Audit date: 2026-05-11
Branch: `paper/p0-1-bridge-prose-update` (commit ec10bcf)
Auditor: claude-opus-4-7

Scope: every figure (.png) and table referenced in the paper draft is mapped
to (i) its generator script, (ii) its source data file, (iii) the local mtime
of both. "Stale" means the figure/data was last touched before
**2026-05-04** (Main = OneVision flip) or **2026-05-10** (InternVL3 removal +
6-bin headline) and the underlying panel composition changed.

## TL;DR — three classes of issue

1. **Hard mismatch (must fix)** — Figure 6 caption describes γ-β bridge
   bars, but the actual file `paper_E5e_mathvista_bars.png` shows E5e
   MathVista 6-model digit-pixel bars (gemma3-27b / llava-interleave /
   qwen2.5-vl). Caption ↔ figure is unrelated. Two correctly-typed γ-β
   figures already exist on disk and are unused: `gamma_beta_bridge_paired_delta.png`
   and `gamma_beta_bridge_per_token_trajectory.png` (mtime 2026-05-10).
2. **Source-data missing in this branch (regen needed)** — paper Table 5
   (γ-β L×K sweep), Table 6 (E6 stage4 bootstrap raw draws), §A.5 (E6
   pilot grid 27-cell aggregate) all cite CSV/NPZ paths under
   `docs/insights/_data/` that exist only in worktree
   `phase5+p0-1-gamma-beta-bridge`. `_data/` is gitignored, so PR merges
   carried prose but not artefacts. Generators are present in `scripts/`;
   single regen pass needed.
3. **Stale generation (low risk, but should be regen'd for paper polish)** —
   `paper_M2_variant_comparison.png` (Fig B2) was generated 2026-04-29 from
   2026-04-29 CSV; M2 was settled on 2026-04-28 and panel composition has
   not changed, so output is correct but the build_paper_figures.py
   followup (commit 361c71a) regen'd 2 of 4 figures and skipped this one
   along with `paper_E5e_mathvista_bars.png`. Re-running the script would
   regen both deterministically.

Plus a **Table 5 numbering bug** — both §4.6.1 and §6.1 use the label
"Table 5".

---

## Figure consistency table

| ID | Figure | File | Generator | Source CSV | Fig mtime | CSV mtime | Status |
|---|---|---|---|---|---|---|---|
| Fig 1 | 4-condition acc-drop | `E5c_acc_drop_3way.png` | `analyze_e5c_distance.py` | `E5c_per_cell.csv` | 04-29 | 04-29 | OK (panel unchanged) |
| Fig 2 | PlotQA correct vs wrong df | `paper_4_1_PlotQA_correct_vs_wrong_df.png` | `_build_figure_4_1_plotqa.py` | `main_panel_5dataset_per_cell.csv` | **05-10** | **05-10** | ✅ fresh (post InternVL3) |
| Fig 3 | E5c anchor vs masked adopt | `E5c_anchor_vs_masked_adopt.png` | `analyze_e5c_distance.py` | `E5c_per_cell.csv` | 04-29 | 04-29 | OK (legacy 3-model panel) |
| Fig 4 | 5-dataset cross-dataset summary | `paper_cross_dataset_summary.png` | `build_paper_figures.py::fig_cross_dataset_summary` | `main_panel_5dataset_per_cell.csv` | **05-10** | **05-10** | ✅ fresh |
| Fig 5 | L1 6-bin confidence quartile | `paper_L1_confidence_quartile.png` | `build_paper_figures.py::fig_L1_confidence_quartile` | `L1_confidence_quartile_long_6bin.csv` | **05-10** | **05-10** | ✅ fresh |
| **Fig 6** | γ-β MathVista pair | `paper_E5e_mathvista_bars.png` | `build_paper_figures.py::fig_E5e_mathvista_bars` | `experiment_e5e_mathvista_full_per_cell.csv` | 04-29 | 05-10 | ⛔ **CAPTION ↔ FILE MISMATCH** — caption is γ-β Qwen3-VL-Instruct/Thinking, file is E5e digit-pixel 3-model bars |
| Fig A1 | E5d ChartQA decay | `E5d_chartqa_decay.png` | `analyze_e5d_chartqa_validation.py` | `E5d_chartqa_per_stratum.csv` | 04-28 | 04-28 | OK (γ distance validation, single dataset) |
| Fig B1 | C-form migration main panel | `C_form_migration_main_panel.png` | `build_C_form_migration_report.py` | `C_form_migration_cells.csv` | 04-29 | 04-29 | OK (historical migration record, no panel change) |
| Fig B2 | M2 18-variant comparison | `paper_M2_variant_comparison.png` | `build_paper_figures.py::fig_M2_variant_comparison` | `M2_metric_variants_long.csv` | 04-29 | 04-29 | ⚠️ data is 04-29 (M2 settled 04-28), regen recommended for consistency w/ Fig 4/5 |
| Fig C1 | gemma3-27b digit-pixel causality | `E5c_anchor_vs_masked_adopt_gemma3-27b-it.png` | `analyze_e5c_distance.py` | `E5c_per_cell.csv` | 04-29 | 04-29 | OK |
| Fig C2 | qwen2.5-vl-7b digit-pixel causality | `E5c_anchor_vs_masked_adopt_qwen2.5-vl-7b-instruct.png` | `analyze_e5c_distance.py` | `E5c_per_cell.csv` | 04-29 | 04-29 | OK |
| Fig C3 | E5c anchor vs masked df | `E5c_anchor_vs_masked_df.png` | `analyze_e5c_distance.py` | `E5c_per_cell.csv` | 04-29 | 04-29 | OK |
| Fig C4 | (re-uses Fig 1) | `E5c_acc_drop_3way.png` | same | same | 04-29 | 04-29 | OK |
| Fig F1 | E5b adopt cond curve | `E5b_adopt_cond_curve.png` | `analyze_e5b_distance.py` | `E5b_per_stratum.csv` | 04-29 | 04-29 | OK (legacy 2-model × 2-dataset, paper §E.2) |
| Fig F2 | E5b adopt cond overlay | `E5b_adopt_cond_overlay.png` | `analyze_e5b_distance.py` | `E5b_per_stratum.csv` | 04-29 | 04-29 | OK |
| Fig G1 | E5d MathVista decay | `E5d_mathvista_decay.png` | `analyze_e5d_mathvista_validation.py` | `E5d_mathvista_per_stratum.csv` | 04-28 | 04-28 | OK |
| §A.5 | E6 pilot grid heatmap (PlotQA) | `E6_pilot_grid_plotqa_heatmap.png` | `aggregate_e6_pilot_grid.py` | **`E6_pilot_grid_27cells.csv` ABSENT** in this branch (only in worktree `phase5+p0-1-gamma-beta-bridge`) | 05-10 | — | ⚠️ source CSV missing locally; figure was generated in worktree, must regen here for reproducibility |
| §A.5 | E6 pilot grid heatmap (InfoVQA) | `E6_pilot_grid_infographicvqa_heatmap.png` | same | same | 05-10 | — | ⚠️ same |

### Figures present on disk but not referenced in the paper draft

| File | mtime | Note |
|---|---|---|
| `gamma_beta_bridge_paired_delta.png` | 05-10 | **Likely the intended Fig 6** — paired delta on γ-β (Qwen3-VL Instruct vs Thinking). Caption text in §4.5 already matches this figure. |
| `gamma_beta_bridge_per_token_trajectory.png` | 05-10 | γ-β per-token trajectory; could land as supplementary in §4.6.1. |
| `paper_4_4_binning_comparison.png` | 05-10 | §4.4 binning resolution comparison (4 vs 6 vs 10). Untracked (`??` in git status), generated 05-10 by ad-hoc `_compare_l1_binning.py`. Could support §4.4 6-bin headline as a justification figure. |
| `H6_2axis_scatter_5dataset.png` | 05-10 | H6 2-axis 6-model scatter. Memory note "H6 cluster re-anchored on 6-model panel" suggests it's prepared for an insights doc, not necessarily the paper. |
| `C_form_migration_scatter.png` | 04-29 | C-form audit; not cited. Fig B1 is the chosen panel. |
| `E5c_acc_drop_3way_*.png` per-model | 04-29 | Per-model variants of Fig 1; only the panel version is cited. |
| `E5c_correct_vs_wrong_adopt*.png` | 04-29 | Older E5c variant; superseded by Fig 2. |

---

## Table consistency table

| Tbl | Section | Source data | Status |
|---|---|---|---|
| Table 1 | §3.1 4-condition definitions | text only (no CSV) | ✅ static |
| Table 2 | §4.1 6-model PlotQA panel | `main_panel_5dataset_per_cell.csv` (subset PlotQA) | ✅ data fresh 05-10 |
| Table 3 | §4.2 wrong-base × S1 paired adoption + (a−m) gap | `experiment_e5e_*_full_per_cell.csv` + `experiment_e5b_5strat_*_onevision_per_cell.csv` | ✅ data fresh 05-10; OneVision TallyQA backfilled in commit ec10bcf |
| Table 4 | §4.6 γ-β H2 decomposition | `experiment_e5e_mathvista_reasoning_per_cell.csv` | ⚠️ verify cell-level numbers (×12.7 etc.) against the CSV; data fresh 05-10 but values were authored against the worktree state |
| **Table 5 (a)** | §4.6.1 L×K sweep Bonferroni-survivors | `gamma_beta_bridge_lk_sweep.csv` | ⛔ **CSV absent in current branch**, lives only in worktree `phase5+p0-1-gamma-beta-bridge`. Generator: `analyze_gamma_beta_bridge_lk_sweep.py` |
| **Table 5 (b)** | §6.1 E4 Phase 2 mitigation | `docs/insights/E4-mitigation-evidence.md` | ⚠️ **numbering bug** (two tables share label "Table 5"); also need to confirm the −14.6 % / −9.6 % cell-level figures still match the evidence doc |
| Table 6 | §6.2.3 E6 Stage 4-final paired-bootstrap CI | `stage4_final_per_dataset.csv` (point estimates) + **`stage4_final_bootstrap_draws.npz` ABSENT** locally | ⚠️ raw draws cited but absent in `_data/`; CI numbers in the table were authored in worktree. Generator: `build_e6_stage4_bootstrap_ci.py` |
| Table 7 | §6.5 multi-method comparison | qualitative text + `E6-tally-only-rerun-tracker.md`, `E6-steering-vector.md` | ✅ qualitative; no per-cell CSV |
| Table 8 | §7 E8 capability eval | `capability_eval_per_benchmark.csv` | ✅ data fresh 05-08 |
| Table C.1 | §C.1 7-model VQAv2 panel | `A1_asymmetric_*.csv` (Phase A) | ✅ stable; Phase A unchanged since 04-29 |
| Table C.2 | §C.2 cross-dataset wrong > correct | `section41_swap_analysis.csv` | ✅ data fresh 05-10 |
| Table C.3 | §C.3 Insight 1 replication | `section41_swap_analysis.csv` | ✅ same |

---

## Action checklist — recommended order

### Hard fixes (caption / file mismatch)

- [x] **Fig 6 caption-vs-file mismatch.** Resolved 2026-05-11: paper
      §4.5 Figure 6 path swapped from `paper_E5e_mathvista_bars.png` to
      `gamma_beta_bridge_paired_delta.png` (which actually contains the
      γ-β paired delta bars described in the caption). The
      `paper_E5e_mathvista_bars.png` file remains tracked but is no
      longer cited.

- [x] **Table 5 duplicate numbering.** Resolved 2026-05-11: §6.1 E4
      Phase 2 → Table 6; §6.2.3 E6 Stage 4 → Table 7; §6.5 multi-method
      → Table 8; §7 E8 → Table 9. All cross-references updated (lines
      55, 286, 293, 335, 337, 387, 424, 436, 462, 463, 484, 575) plus
      changelog v4 / v5 / v8 historical refs.

### Regen passes (single uv-run each, deterministic) — **all completed 2026-05-11**

- [x] `aggregate_e6_pilot_grid.py` — wrote
      `_data/E6_pilot_grid_27cells.csv` (54 rows) +
      `_selection_replay.md` + heatmap PNGs.
- [x] `build_e6_stage4_bootstrap_ci.py` — wrote
      `stage4_final_per_dataset_ci.csv/md` +
      `stage4_final_bootstrap_draws.npz` (B=10,000).
- [x] `analyze_gamma_beta_bridge_lk_sweep.py` — wrote
      `gamma_beta_bridge_lk_sweep.csv` (84 cells, n=522 paired sids).
- [x] `build_paper_figures.py` — regen'd all 4 figures uniformly
      (timestamp 2026-05-11).

### Cell-level verification — **all passed**

- [x] Table 4 (γ-β H2 decomposition): 4/4 cells match
      `experiment_e5e_mathvista_reasoning_per_cell.csv`. ×12.7 ratio
      verified against pooled wrong+correct df rates.
- [x] Table 5 (γ-β L×K sweep): 8/8 spot-checked cells match
      newly-regen'd `gamma_beta_bridge_lk_sweep.csv`.
- [x] Table 6 (E4 Phase 2): −14.6 % / −9.6 % / +0.77 pp / +1.30 pp all
      match `docs/insights/E4-mitigation-evidence.md` lines 37–38.
- [x] Table 7 (E6 Stage 4): all 25 cells (5 datasets × 5 metrics) match
      newly-regen'd `stage4_final_per_dataset_ci.md`.

### Layout decisions

- [x] `paper_4_4_binning_comparison.png` integrated as **Figure B3** in
      a new §B.1 "L1 confidence-bin resolution" subsection — justifies
      §4.4 6-bin headline against 4-bin (under-resolution) and 10-bin
      (over-resolution) on the same high-n PlotQA × OneVision cell.

### Resolved (2026-05-11)

- [x] **Fig 3 swap.** `paper_4_2_digit_pixel_causality.png` replaced
      legacy `E5c_anchor_vs_masked_adopt.png`. §4.2 Table 3 also
      restructured into two orthogonal slices (Slice A: PlotQA 6-model
      E7 panel; Slice B: OneVision Main × 5-dataset). VQAv2 rows
      dropped. Cross-references in §4.1 / §4.3 / §C.1 / §E.4 updated.
      roadmap §10 changelog records the reorg. (Folded into commit
      `adae6df`.)
- [x] **Worktree-to-main sync workflow.** Option A (rsync helper)
      shipped as `scripts/_sync_from_worktree.sh`. Excludes
      `docs/paper/reviews/` (already git-tracked via the
      `!docs/paper/reviews/*.md` rule). Use:
      `bash scripts/_sync_from_worktree.sh <branch> [--dry-run]`.
      (Folded into commit `adae6df`.)

---

## Source-of-truth pointers

- Figure index: `git ls-files docs/figures/`
- Data canonical CSVs: `docs/insights/_data/_artifact_index.json`
- Recent figure-related commits:
  - `361c71a` — Followup: regen 6-model figures + delete onevision queue script
  - `4d94870` — fig_cross_dataset_summary: 5-dataset main matrix
  - `9630f30` — Paper-summary deck: PPTX builder + 4 figures + speaker notes
- Memory `[InternVL3 fully removed 2026-05-10 (PR #21 + #22 merged)]` confirms
  that follow-up `paper_4_1` + `paper_cross_dataset_summary` were regen'd
  on the 6-model panel; the same pass did **not** touch
  `paper_E5e_mathvista_bars.png` (already-stale γ-β-mismatched filename) or
  `paper_M2_variant_comparison.png`.
