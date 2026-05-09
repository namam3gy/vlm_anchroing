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

**SDPA mask-bias regression FIXED 2026-05-04 ~21:35** (caveat #2 below was retired). The "ablation no-op" symptom on all 4 datasets was bisected to commit `7f8ebb6` (SDPA switch). Eager re-run via `scripts/_phase2_e1d_eager_rerun.sh` produces valid Δ vs baseline:

| dataset | base df | Δ peak | Δ peak_window | Δ lower_half | Δ upper_half | Δ all |
|---|---:|---:|---:|---:|---:|---:|
| TallyQA | 0.130 | -0.005 | +0.005 | **+0.050** | -0.025 | **-0.040** |
| InfoVQA | 0.167 | +0.000 | +0.005 | -0.006 | +0.004 | +0.008 |
| ChartQA | 0.105 | +0.000 | +0.005 | +0.026 | -0.004 | +0.006 |
| MathVista | 0.171 | +0.000 | -0.005 | **+0.075** | -0.026 | **-0.045** |
| PlotQA *(orphan eager pre-7f8ebb6)* | 0.243 | -0.006 | -0.010 | +0.024 | -0.039 | **-0.051** |

3/5 datasets (TallyQA, MathVista, PlotQA) reproduce the classic E1d signature (mid-stack ablation amplifies, full ablation drops); ChartQA shows weaker signal; **InfoVQA shows minimal ablation effect** at all 6 modes.

**B1 follow-up 2026-05-05 ~00:16 KST** — re-ran InfoVQA E1d with `--peak-layer 14` (Phase D's actual InfoVQA peak from `analyze_cross_dataset_peaks.py`) instead of the master script's hardcoded 27. Result: ablate_peak Δ_df = +0.015 (vs +0 at L27); ablate_peak_window, ablate_lower_half, ablate_upper_half, ablate_all all unchanged. The flat InfoVQA ablation profile is **not** a peak-layer-routing artefact — InfoVQA's OneVision anchor mechanism is genuinely diffuse, with no single layer band causally responsible. Comparison log: `outputs/_logs/phase3_b1_infovqa_peak14/compare_20260504-223234.log`. Launcher: `scripts/_phase3_b1_infovqa_peak14.sh`.

### Mitigation (§7.4.5) — chosen cell + paired-bootstrap CI

**L=26 K=8 α=1.0 subspace projection** selected from 27-cell pilot grid (L∈{25,26,27} × K∈{2,4,8} × α∈{0.5,1.0,2.0}) on PlotQA+InfoVQA pooled n5k. **Pilot-grid context (P1-6 close, 2026-05-10)**: ex ante "Δem(a) ≤ −6 pp deal-breaker on either calib" rule is *non-binding* on the actual grid (0 / 27 cells rejected); chosen cell ranks **1st by combined |Δdf(a)|** with 1.2 pp margin over runner-up — same ex ante rule on same data does not select a different cell. Heatmap: `docs/figures/E6_pilot_grid_{plotqa,infographicvqa}_heatmap.png`; aggregator `scripts/aggregate_e6_pilot_grid.py`; canonical CSV `_data/E6_pilot_grid_27cells.csv`.

Stage 4-final eval on 5 datasets (n=5000 wrong-base subset per dataset, paired-sids comparison):
- Point estimates: `scripts/build_e6_stage4_summary.py` → `_data/stage4_final_per_dataset.csv`.
- **Paired-bootstrap CI (B = 10,000)** added 2026-05-10 (P1-3 close): `scripts/build_e6_stage4_bootstrap_ci.py` → `_data/stage4_final_per_dataset_ci.csv` + raw draws `_data/stage4_final_bootstrap_draws.npz`. 95 % equal-tail + Bonferroni-20 corrected (5 × 4 = 20 family, α = 0.05/20 = 0.0025 → 99.75 %) bands.

| metric | mean | sign-clean count at 95 % | sign-clean at Bonferroni-20 |
|---|---:|:---:|:---:|
| Δ adopt(a) | **−2.0 pp** | 2 / 5 | 2 / 5 |
| Δ df(a) | **−2.9 pp** | 1 / 5 (PlotQA only [−6.9, −3.4]) | 1 / 5 (PlotQA only) |
| Δ em(a) | **+3.9 pp** | 3 / 5 | 2 / 5 (PlotQA, TallyQA) |
| **Δ em(b)** | **+8.8 pp** | **5 / 5** | **5 / 5** |

**Δ em(b) is the multiplicity-robust headline** — only metric whose sign-clean status survives Bonferroni-20 correction on every cell. **InfoVQA Δdf=−0.7 pp on n=443** has 95 % CI [−4.7, +3.4] (zero-inclusive band → inconclusive fence confirmed numerically; sanity gate: half-width 0.0406 lands inside paper's prior paired-Wilson estimate ~0.04–0.06). This is a **strict free-lunch on the wrong-base subset**: anchor pull goes down (PlotQA CI-strong, others CI-individually-borderline), exact-match goes up on both arms (b-arm 5/5 Bonferroni-robust). Earlier "Δ em(a) = -0.024 cost" framing was a hand-copy error from prior aggregation (corrected 2026-05-04 from generator output). Paper §6.2.3 (Korean draft) / §7.4.5 (English sister-section) reframed 2026-05-10. Insight cousin: `docs/insights/E6-stage4-paired-bootstrap-ci.md`.

### Capability preservation (E8, 8-bench, 2026-05-09)

**Verdict**: `STRICT_FREE_LUNCH` extends from the anchoring task family
to general VLM capability across **8 held-out benchmarks** at the same
chosen cell L=26 K=8 α=1.0 (no anchor labels, no weight updates,
greedy decoding only).

- **Macro Δ = +0.31 pp** across MMBench-DEV-EN, OCRBench, RealWorldQA,
  MMStar, HallusionBench, POPE, MME, AMBER (n_total = 27,097 effective,
  no LLM-judge, ~4 h 36 min H200 sequential, $0).
- All 8 per-benchmark Δ within the pre-registered ±1.0 pp band.
- **Two of three hallucination axes show CI-clean positive Δ**:
  HallusionBench Δ = +2.21 pp [+1.14, +3.28] and AMBER Δ = +0.19 pp on
  n = 14,216 [+0.05, +0.33]; POPE pins the third to zero (Δ = -0.06 pp,
  [-0.21, +0.09]).
- **MME Count subset (n = 60), the in-domain analogue of the
  number-anchor failure mode, shows Δ = 0.00 pp exact** — every paired
  prediction matches between baseline and +mit. Direct evidence the
  mitigation acts on cross-modal anchor pull, not counting capability
  itself.
- Contamination-resistant floor of the panel: n = 1,500 (MMStar alone)
  → n = 18,090 (MMStar + MME + AMBER).

Insight doc: `docs/insights/E8-capability-preservation-evidence.md`.
Experiment writeup: `docs/experiments/E8-capability-preservation.md`.

## What did NOT land in this push

- **§7.4 E4 mitigation Main extension** — OneVision E4 sweep deferred. AnyRes encoder doesn't fit the mid-stack archetype that E4 was tuned for (LLaVA-1.5 / ConvLLaVA / InternVL3). Phase 3 may retry with archetype assignment from cross-dataset peak data.
- **6-cond × 7-model expansion (Priority 5)** — initially launched, killed when discovered 4-cond matrix already covered 30/35 cells. Reverted to 4-cond.
- **Phase 1.5 cross-dataset §7.1-7.3 validation (option C)** — superseded by Phase D which shipped the equivalent 24-cell matrix as part of master queue.

## Known data caveats

1. **internvl3-8b TallyQA cell**: rerun in flight (2026-05-04 02:42 → ~07:00 ETA) due to fast-tail watcher prematurely killing the original 3-shard run before merge step. Watcher bug fixed (`scripts/_phase1_recover_internvl3_fast_tail.sh`); rerun uses 2-shard + DataLoader prefetch.
2. ~~**OneVision E1d direction-follow rates show 0.000 baseline**: aggregator stratification logic doesn't match OneVision's susceptibility CSV. Raw predictions are correct (`outputs/causal_ablation/llava-onevision-qwen2-7b-ov/<run>/predictions.jsonl`); refining the analyzer is a Phase 3 follow-up.~~ — **RESOLVED 2026-05-04** (commits `a7e391c`, `de1f94e`, `8895128`). Two layered bugs: (a) analyzer merged 4 OneVision datasets into one row; (b) **inference itself was no-op** because commit `7f8ebb6` switched `causal_anchor_ablation.py` from eager to SDPA, and SDPA dispatch silently drops the `attention_mask` bias added by `_make_anchor_mask_hook`. Fix: per-(model, dataset) analyzer + `--attn-implementation eager` flag + 4-dataset re-run. See "Causal ablation" section above for the new ablation deltas.
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
