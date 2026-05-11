# Canonical paper figure — §6.2.3 Stage-4 5-dataset × paired CI

**Date**: 2026-05-11
**Scope**: One canonical figure carrying paper §6.2.3 headline — 5 evaluation datasets × {Δdf(a), Δem(b)} × paired-bootstrap 95% CI — to accompany Table 7. Figure does not replace Table 7; it adds a visual second pass of the multiplicity-robust headline and the sample-size confound.

## Why this figure exists

§6.2.3 Table 7 ships all four wrong-base metrics (Δadopt, Δdf, Δem(a), Δem(b)) with 95% CI text. The paper thesis (PR #27) treats E6 as a *worked example* of the (a − m) calibration-contrast design pattern, with two headline clauses:

1. **Anchoring effect** — Δdf(a) < 0, sample-size-bound: only PlotQA n=2,306 95% CI excludes 0; four small-n cells point-estimate-consistent but CI-borderline.
2. **Multiplicity-robust capability headline** — Δem(b) > 0 on 5/5 cells under both 95% AND Bonferroni-20 corrected CIs.

A canonical figure that places (Δdf, Δem(b)) side-by-side with sample-size-descending row order makes both clauses visually inspectable in one glance, and surfaces the sample-size dependency of the Δdf clause without prose.

## Data source

Canonical: `docs/insights/_data/stage4_final_per_dataset_ci.csv` (5 rows × `delta_df`, `delta_df_ci95_{lo,hi}`, `delta_em_b`, `delta_em_b_ci95_{lo,hi}`, `n_paired`).

Frozen 2026-05-10. Bonferroni-20 columns also present but not rendered (see §A.5 prose for multiplicity disclosure).

## Layout

Two sub-panels, side-by-side (1 row × 2 col), shared figure title.

- **Left panel**: Δdf(a) — headline direction is negative (anchoring-effect reduction). x-axis: `Δ direction-follow rate (pp)`.
- **Right panel**: Δem(b) — headline direction is positive (capability-side, non-anchored arm). x-axis: `Δ exact-match on non-anchored arm (pp)`.
- Each panel: 5 horizontal rows (one per dataset), **sample-size descending**: TallyQA → PlotQA → InfoVQA → ChartQA → MathVista. Same row order both panels.
- Each row: point-estimate filled dot + 95% CI horizontal whisker.
- x=0 dashed reference line. CI bars excluding zero in headline direction = strong color (steel-blue for Δdf, sea-green for Δem(b)); CI bars including zero = neutral gray.
- Right margin annotation per row: `n=<n_paired>`.
- Figure title: "E6 Stage-4 (L=26, K=8, α=1.0): 5-dataset paired-bootstrap deltas".
- Subtitle / caption-line under title: "B = 10,000 paired sids; 95% CI shown. Bonferroni-20 corrected CIs in §A.5."

Style baseline: match `docs/figures/gamma_beta_bridge_paired_delta.png` (matplotlib default + serif title, no heavy decoration).

## Files

- Script: `scripts/build_paper_stage4_paired_ci_figure.py` (uv run python entry; reads CSV, writes PNG; no CLI flags needed — single canonical output).
- Output: `docs/figures/paper_6_2_3_stage4_5dataset_paired_ci.png` (300 dpi; ~10 × 5 in).

## Paper wiring

Insert one figure reference in `docs/paper/sections/07_mechanism_mitigation.md` (and mirror in `docs/paper/emnlp_draft_ko.md`) immediately before Table 7 with a single-paragraph caption tying the two panels to the two headline clauses. No prose deletion; Table 7 stays as authoritative numeric source.

## Out of scope

- Cross-cell comparison (chosen #17 vs others) — that lives in `docs/figures/E6_pilot_grid_*_heatmap.png` already.
- Bonferroni-20 overlay — per user 2026-05-11, kept prose-only in §A.5.
- Δadopt / Δem(a) panels — Table 7 covers these; figure intentionally narrows to the two headline clauses.

## Acceptance criteria

1. Figure renders from canonical CSV in one `uv run python` invocation.
2. Numeric values reading off the figure match Table 7 (visual sanity check: PlotQA Δdf ≈ −5.2 pp, Δem(b) ≈ +4.7 pp; MathVista Δem(b) ≈ +9.4 pp; etc.).
3. PlotQA Δdf 95% CI visibly excludes 0; remaining 4 Δdf CIs cross 0. All 5 Δem(b) CIs visibly exclude 0.
4. Figure file committed; paper sections reference it; PR opened to master.
