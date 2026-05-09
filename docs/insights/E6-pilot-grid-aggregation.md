# E6 27-cell pilot grid — 4-metric heatmap aggregation (P1-6)

**Generated:** 2026-05-10
**Closes:** R4 CRIT-2 (cherry-pick concern on 27-cell pilot grid).

## Inputs

- Pilot runs (calibration phase, n=250 wrong-base sids per calib dataset, all 28 cells = baseline + 27 grid):
  - `outputs/e6_steering/llava-onevision-qwen2-7b-ov/pilot_grid_plotqa_n250/predictions.jsonl`
  - `outputs/e6_steering/llava-onevision-qwen2-7b-ov/pilot_grid_infographicvqa_n250/predictions.jsonl`
- Grid: L ∈ {25, 26, 27} × K ∈ {2, 4, 8} × α ∈ {0.5, 1.0, 2.0} → 27 cells.
- Selection rule (ex ante, see paper §6.2.2): reject any cell with
  **Δem(a) ≤ −6 pp on either calibration dataset**; rank surviving cells
  by combined |Δdf(a)| (mean across PlotQA + InfoVQA pilots).

## Outputs

- Canonical CSV: `_data/E6_pilot_grid_27cells.csv` (54 rows = 27 cells × 2 calib pilots).
- Selection-rule replay markdown: `_data/E6_pilot_grid_27cells_selection_replay.md`.
- Heatmap figures (4 metrics × 3 layers per dataset):
  - `docs/figures/E6_pilot_grid_plotqa_heatmap.png`
  - `docs/figures/E6_pilot_grid_infographicvqa_heatmap.png`
  - Each figure: 4 rows × 3 cols (rows = Δadopt(a) / Δdf(a) / Δem(a) / Δem(b); cols = L=25, 26, 27). Each panel = K (rows: 2, 4, 8) × α (cols: 0.5, 1.0, 2.0). Blue = mitigation-direction (negative for Δadopt/Δdf, positive for Δem). Chosen cell #17 (L=26, K=8, α=1.0) starred + outlined.
- Generator: `scripts/aggregate_e6_pilot_grid.py`.

## Selection-rule replay — key findings

- **No cell rejected by em-deal-breaker.** Of 27 cells, the most negative Δem(a) on either calib dataset is **−1.2 pp** (cell #19, L=27 K=2 α=0.5 on PlotQA pilot). The −6 pp threshold is therefore *non-binding* on the actual pilot — it served as a precommitted safety rail rather than a filter that pruned a cherry-pick window.
- **Chosen cell #17 (L=26, K=8, α=1.0) is the |Δdf(a)|-rank 1 cell after applying the rule.** Sorting all 27 cells by mean Δdf(a) (most negative = best mitigation):
  1. **#17 L26_K8_α1.0:** mean Δdf = −4.4 pp ← **chosen**
  2. #8 L25_K8_α1.0: mean Δdf = −3.2 pp
  3. #16 L26_K8_α0.5: mean Δdf = −2.8 pp
  4. (#9 / #2 / others tied around −2.2)
- **Cherry-pick concern resolution.** The chosen cell wins under the precommitted rule on the same pilot data the paper's selection used, with the rule's filter clause empty. Surfacing the full 27-cell × 4-metric heatmap is the standard transparency artifact a sweep-based selection requires; that artifact is now in the appendix (§A.5).

## Cross-cell pattern (qualitative)

- **K = 8 row dominates the strong-Δdf zone on PlotQA.** Across all three layers, K=8 cells have the most-negative Δdf cells (PlotQA: K=8 row min = −8.0, K=4 row min = −2.4, K=2 row min = −5.6). On InfoVQA the zone is more diffuse, consistent with InfoVQA's smaller df baseline (~0.10 vs PlotQA's ~0.20).
- **L = 27 weakens both Δdf and Δadopt across most cells** — consistent with the paper's §5.2 multi-layer-redundancy and §5.3 OneVision peak-layer narrative (peak before L=27 on this calibration data).
- **Δem(b) is monotone-positive on most cells** — the b-arm em recovery is broad across the grid, reinforcing the §6.3 "wrong-base error mode 제거" interpretation as not chosen-cell-specific.

## Cross-references

- Paper appendix: §A.5 (`docs/paper/emnlp_draft_ko.md`, `docs/paper/sections/` if split)
- Insight cousin: `E6-stage4-paired-bootstrap-ci.md` (P1-3 — the 5-dataset main-table CIs)
- Notebook: `notebooks/E6_pilot_grid_27cells_demo.ipynb`
