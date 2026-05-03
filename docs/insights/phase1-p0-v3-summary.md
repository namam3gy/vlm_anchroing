# Phase 1 P0 v3 — final summary (2026-05-04)

Umbrella for the Phase 1 P0 v3 push (2026-05-02 → 2026-05-04). Cross-references `references/project.md §0.0+§0.7` and `references/roadmap.md §3.0a+§10`.

## What landed

### Architecture
- **Main model**: switched to `llava-onevision-qwen2-7b-ov` (was llava-interleave-7b in v2)
- **6-model main panel**: llava-onevision-7b, qwen2.5-vl-7b, internvl3-8b, gemma3-4b, qwen2.5-vl-32b, gemma3-27b — `llava-next-interleaved-7b dropped` 2026-05-04 (low native resolution)
- **5-dataset main matrix**: TallyQA, ChartQA, MathVista, PlotQA, InfoVQA (VQAv2 → appendix)
- **5-model mech panel** (was 4): added `internvl3-8b` (perfect-square 27×27 routing). OneVision Main added via AnyRes per-image bbox routing (lite_eager monkey-patch)

### Infrastructure (this push)
- **Sharded inference** (`run_experiment.py` + `e6_steering_vector.py` + `causal_anchor_ablation.py`) — `--shard-idx`/`--num-shards`/`--output-dir` flags, fan-out drivers `run_*_sharded.py`
- **DataLoader prefetch** in `run_experiment.py` — opt-in via `VLM_ENABLE_PREFETCH=1`. Smoke-tested byte-equal modulo 10⁻⁵ tail-token logits (CUDA non-determinism)
- **lite_eager attention monkey-patch** (`extract_attention_mass.py`) — solves OneVision AnyRes OOM. Defensive `is_causal` mirror of HF SDPA logic
- **Per-process output dir** (`extract_attention_mass.py`) — microseconds + PID suffix prevents same-second timestamp collisions in parallel runs
- **gitignore expansion** — `docs/insights/_data/`, `docs/paper/`, `docs/ppt/`, `docs/superpowers/plans/`, `docs/CHANGELOG.md` untracked + history-rewritten via `git filter-repo`. Removed 113MB CSV from history that was blocking pushes.

## Results

See `docs/insights/headline-numbers.md §A` for full numbers. Headlines:

### Behavioural (§3 + §5 + §6)
6-model × 5-dataset main matrix (29/30 cells; internvl3-8b/TallyQA rerun in flight). Universal anchoring: every (model, dataset) cell has df(a) > 0 + df(a) > df(m) > df(d) ≈ 0.

**Susceptibility ranking** (avg df across 5 datasets):
- gemma3-4b ≫ gemma3-27b > llava-onevision/interleave > qwen2.5-vl-32b ≈ qwen2.5-vl-7b > internvl3-8b
- qwen family + internvl3 are most robust
- **Anti-scaling within Gemma**: 4B more pulled than 27B

**Dataset susceptibility ordering** (mean df across panel):
PlotQA ≈ MathVista > InfoVQA > ChartQA ≫ TallyQA — chart/figure ~2× counting.

### Mechanism (§7.1-7.3)

Phase D shipped 24/24 cells (5 panel × 4 datasets + OneVision × 4 datasets). New cross-dataset peak comparison via `scripts/analyze_cross_dataset_peaks.py`.

**Key new finding (2026-05-04)**: OneVision peak layer is **dataset-dependent**:
- L=27 (last layer) on PlotQA + TallyQA
- L=14 (mid-stack) on InfoVQA + VQAv2

This refines the earlier "OneVision = late-layer model-specific" reading. Model architecture sets the layer band; dataset content modulates which sub-band activates. Other models (gemma4-e4b especially) show stable peaks across datasets.

### Causal ablation (Phase E, OneVision × 4 datasets)

OneVision E1d on TallyQA + InfoVQA + ChartQA + MathVista. 6-mode ablation table at `outputs/causal_ablation/_summary/per_model_per_mode.csv`.

**Note**: Master queue's original ChartQA + MathVista runs were corrupted (PlotQA-CSV-reuse bug). Fixed 2026-05-04 by building per-dataset susceptibility CSVs from existing OneVision baselines + re-running. Commit `2d11876`.

### Mitigation (§7.4.5) — chosen cell

**L=26 K=8 α=1.0 subspace projection** selected from 27-cell pilot grid (L∈{25,26,27} × K∈{2,4,8} × α∈{0.5,1.0,2.0}) on PlotQA+InfoVQA pooled n5k.

Stage 4-final eval on 5 datasets (n=5000 wrong-base subset per dataset):
- avg Δ adopt(a) = -0.036
- avg Δ df(a) = **-0.025** (df reduction works)
- avg Δ em(a) = -0.024 (within em-drop dealbreaker)
- avg **Δ em(b) = +0.092** (unintended recovery on wrong-base sids)

The em(b) +9.2pp finding is paper-novel — needs §7.4.5 prose update (task #38).

## What did NOT land in this push

- **§7.4 E4 mitigation Main extension** — OneVision E4 sweep deferred. AnyRes encoder doesn't fit the mid-stack archetype that E4 was tuned for (LLaVA-1.5 / ConvLLaVA / InternVL3). Phase 3 may retry with archetype assignment from cross-dataset peak data.
- **6-cond × 7-model expansion (Priority 5)** — initially launched, killed when discovered 4-cond matrix already covered 30/35 cells. Reverted to 4-cond.
- **Phase 1.5 cross-dataset §7.1-7.3 validation (option C)** — superseded by Phase D which shipped the equivalent 24-cell matrix as part of master queue.

## Known data caveats

1. **internvl3-8b TallyQA cell**: rerun in flight (2026-05-04 02:42 → ~07:00 ETA) due to fast-tail watcher prematurely killing the original 3-shard run before merge step. Watcher bug fixed (`scripts/_phase1_recover_internvl3_fast_tail.sh`); rerun uses 2-shard + DataLoader prefetch.
2. **OneVision E1d direction-follow rates show 0.000 baseline**: aggregator stratification logic doesn't match OneVision's susceptibility CSV. Raw predictions are correct (`outputs/causal_ablation/llava-onevision-qwen2-7b-ov/<run>/predictions.jsonl`); refining the analyzer is a Phase 3 follow-up.
3. **llava-next-interleaved-7b**: Phase D §7.1-7.3 attention data exists for plot/tally/info (not VQAv2). Model dropped from main panel 2026-05-04. Data preserved as supplementary.

## Reproduction

```bash
# Reproduce 6-model × 5-dataset summary
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing
for exp in experiment_e5e_tallyqa_full experiment_e5e_chartqa_full \
           experiment_e5e_mathvista_full experiment_e7_plotqa_full \
           experiment_e7_infographicvqa_full; do
  uv run python scripts/analyze_e5e_wrong_correct.py --exp-dir "$exp"
done
uv run python scripts/build_e5e_e7_5dataset_summary.py --print

# Reproduce cross-dataset peak comparison
uv run python scripts/analyze_cross_dataset_peaks.py

# Reproduce Phase E causal ablation summary
uv run python scripts/analyze_causal_ablation.py
```

All output files land in `docs/insights/_data/` (gitignored).

## Commits (chronological, this push)

| commit | content |
|---|---|
| `9f9dfa0` | Phase B Stage 4-final L=26 K=8 α=1.0 × 5 datasets |
| `dd17457` | analyze_e6_pilot_cells: filter baseline to wrong-base only |
| `1b9c479` | Phase 1.5 (#27) cross_dataset_peaks analyzer |
| `7a27750` | Phase E E1d × 4 datasets (chart/math broken; recovered 2d11876) |
| `8fe1d81` | Phase G partial: qwen2.5-vl-32b-instruct × 5 datasets |
| `8ab1113` | Phase G partial: gemma3-4b-it × 5 datasets |
| `b2f24c1` | Phase G full: 7-model 5-dataset baselines + summary |
| `d4cc765` | Phase H: qwen2.5-vl-7b §7.1-7.3 × 5 datasets |
| `8c7cc43` | Phase I: 7-model × 5-dataset summary |
| `c556fb6` | Phase D: cross-dataset §7.1-7.3 + OneVision × VQAv2 |
| `666fea7` | 5-dataset summary script extended to 7 models |
| `0e7998e` | Drop llava-next-interleaved-7b from main panel |
| `2d11876` | Phase E recovery: ChartQA + MathVista E1d (proper susceptibility) |
| `6c7d99a` | fix: fast-tail watcher merged predictions detection |
| `92bba10` | Untrack docs/* paths + history rewrite (filter-repo) |
| `6d8dac4` | roadmap §3 + §10 update for Phase 1 P0 v3 final |
| `d036a2f` | Consolidate project.md + roadmap.md to 2026-05-04 final state |
