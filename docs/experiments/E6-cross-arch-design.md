# E6 — Cross-architecture replication on Qwen2.5-VL-7B-Instruct: design

> **Status:** Pre-implementation design (2026-05-17). Branch:
> `worktree-e6-cross-arch-qwen25vl`. Closes outline `§8 Limitations`
> "Mitigation scope — cross-architecture *magnitude* transfer 는 후속 작업"
> + roadmap §7 P2-7 entry. Bar-raiser conditional-Main accept hinges on
> this instantiation landing positive
> (see [`docs/paper/reviews/_final_summary.md`](../paper/reviews/_final_summary.md)
> "Main-tier conditional on cross-architecture E6 instantiation").

## Goal

Instantiate the E6 *calibrated subspace projection* mitigation
(outline §6.1) on a **second architecture** — `Qwen2.5-VL-7B-Instruct`
— using the same `(a − m)` paired-inpaint calibration recipe, the
same 27-cell pilot grid topology, and the same 4-clause free-lunch
acceptance criterion that OneVision Main passed.

A positive result (≥1 cell satisfying 4-clause free-lunch on
Qwen2.5-VL) closes outline §8 "Mitigation scope" magnitude-transfer
gap and lifts the worked example from `N=1 architecture` to
`N=2 architecture` (partial close of R4 CRIT-1 per roadmap §7 P2-7).

## Non-goals (deferred to §8.4 follow-ups)

- **Gemma / FastVLM panel expansion.** Single-model first; if
  Qwen2.5-VL clears 4-clause, then Gemma3-27B-it (SigLIP-shared, positive
  control on encoder-family transfer) is the natural next step.
- **K-floor expansion (K∈{1, 16}).** Outline §4.6 K=1 finding on
  Qwen3-VL γ-β bridge is qualitatively independent; this experiment
  is *exact-replica of OneVision §6.1 recipe*, K∈{2,4,8} only.
  Outline §8 already discloses universal-K=8 partial falsification;
  no additional disclosure required.
- **Hyperparameter re-tuning beyond the 27-cell grid.** If 0/27 cells
  pass 4-clause, the negative result itself is the §6.4 contribution
  (cross-arch recipe-portability bound).

## Why Qwen2.5-VL-7B-Instruct first

| Property | Value | Why it matters |
|---|---|---|
| LM backbone | Qwen2.5-7B (28-layer) | Same depth as OneVision (28-layer); L bin proxy direct |
| Vision encoder | Qwen2 ViT (NaViT-style dynamic resolution) | **Different from OneVision (SigLIP)** — encoder-family transfer test |
| Already in main panel | Yes (`outline §3.4`) | Anchoring measurement already characterized — only mitigation is new |
| §5.1 attention probe | **Missing** (outline §G note) | Need quick probe to set L bin center (see Phase 0 below) |
| Inference cost | ~same as OneVision (7B class) | Roadmap §7 P2-7 estimate ~10 H200-day standalone |

Gemma3-27B-it deferred — same SigLIP encoder as OneVision makes it a
*positive control* (less informative on encoder-family transfer) and
~3× costlier than Qwen2.5-VL.

## Recipe — exact OneVision §6.1 replica

### Phase 0 — E6 calibration + peak-layer pick (`~4 H200-hour`, **✅ 완료 2026-05-17**)

> **Result**: L\*_qwen = 26 (top-1 by ‖v_wrong[L]‖, single-peak monotonic ramp).
> Top-5 = {23, 24, 25, 26, 27}. **27-cell pilot L bin = {25, 26, 27} —
> identical to OneVision.** Pooled n_wrong = 1,148 (PlotQA 926 + InfoVQA 222,
> OneVision 2,757 대비 42 %). Full evidence:
> [`docs/insights/E6-cross-arch-qwen25vl-phase0.md`](../insights/E6-cross-arch-qwen25vl-phase0.md).
> **Wall-time observation**: calibrate-subspace 30 min on Qwen2.5-VL vs ~4 min
> on OneVision (~7.5× slower — Qwen2-ViT NaViT dynamic-resolution longer
> visual sequences). Phase 1+2+3 total budget revised ~7 → ~10–12 H200-day.


**Recipe**: identical to OneVision's pooled calibration chain
(`scripts/run_p0_1_resume_b3.sh` Steps B3a/B3b/B4 used by P0-1
γ-β bridge for Qwen3-VL; same pipeline on Qwen2.5-VL substitutes
the model id + HF id + predictions paths). Three sub-steps:

**Step 0a — calibrate-subspace on PlotQA + InfoVQA (in parallel)**:
```
uv run python scripts/e6_steering_vector.py \
    --phase calibrate-subspace \
    --model qwen2.5-vl-7b-instruct \
    --hf-model Qwen/Qwen2.5-VL-7B-Instruct \
    --e5c-run-dir outputs/experiment_e7_plotqa_full/qwen2.5-vl-7b-instruct/<ts> \
    --predictions-path outputs/experiment_e7_plotqa_full/qwen2.5-vl-7b-instruct/<ts>/predictions.jsonl \
    --config configs/experiment_e7_plotqa_full.yaml \
    --dataset-tag plotqa \
    --max-calibrate-pairs 5000
```
Same for InfoVQA with `--dataset-tag infovqa` and the matching
predictions path.

Reads from existing E7 (b/a-S1/m-S1/d) predictions
(`outputs/experiment_e7_plotqa_full/qwen2.5-vl-7b-instruct/20260502-022631/`
+ `outputs/experiment_e7_infographicvqa_full/qwen2.5-vl-7b-instruct/20260502-071849/`)
— no new prediction inference required. Outputs per-source
D matrices to `outputs/e6_steering/qwen2.5-vl-7b-instruct/calibration_{plotqa,infovqa}/D_*.pt`.

**Step 0b — SVD pool**:
```
uv run python scripts/e6_compute_subspace.py \
    --model qwen2.5-vl-7b-instruct \
    --scope plotqa_infovqa_pooled \
    --tags plotqa,infovqa \
    --K-max 16
```
Pools per-source D matrices and emits the canonical
`outputs/e6_steering/qwen2.5-vl-7b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt`
+ a `calibration_plotqa_infovqa_pooled/v.pt + v_meta.json + norms_per_layer.csv`
identical-schema to OneVision's
`outputs/e6_steering/llava-onevision-qwen2-7b-ov/calibration_plotqa_infovqa_pooled/`.

**Step 0c — peak-layer pick**:
```
uv run python scripts/e6_pick_peak_layers.py \
    --model qwen2.5-vl-7b-instruct \
    --tag plotqa_infovqa_pooled \
    --top-k 5 \
    --out outputs/e6_steering/qwen2.5-vl-7b-instruct/_subspace/peak_layers_plotqa_infovqa_pooled.json
```
Picks top-K layers ranked by `‖v_wrong[L]‖` (per-layer (a − m) residual
diff norm). Output JSON `[L*_1, ..., L*_5]` ordered by norm.

**Output**: `L*_qwen` = top-1 layer; the 27-cell pilot L bin =
`{L*_qwen − 1, L*_qwen, L*_qwen + 1}`. If peak is at boundary
(L=0 or L=27), shift the bin inward.

> **No depth-norm proxy fallback.** Peak is determined empirically
> from the calibration-norm — per user direction (2026-05-17), Phase 0
> must actually find the layer, not assume L=26 by depth match.

**Cross-checks against OneVision norm profile**: outline §5.1 reports
OneVision integration site at L=27/28 (depth-norm 96%) on PlotQA/TallyQA
and L=14/28 (50%) on InfoVQA — dataset-dependent. If Qwen2.5-VL
returns a peak that disagrees with OneVision's depth-norm, that's a
genuine cross-arch finding (and may appear in outline §5.1 peak-heterogeneity
discussion as a Qwen2.5-VL row).

**Eligibility filter alignment**: calibrate-subspace internally tightens
to "wrong-base + all 4 conditions present" per memory
[[feedback_qao_q_d_alignment]] — matches OneVision's calibration eligibility.

### Phase 1 — 27-cell pilot grid (`~3 H200-day`)

Grid: `L ∈ {L*_qwen − 1, L*_qwen, L*_qwen + 1}` × `K ∈ {2, 4, 8}` ×
`α ∈ {0.5, 1.0, 2.0}` (27 cells), exact topology of OneVision §A.5
pilot grid (`docs/insights/_data/E6_pilot_grid_27cells.csv` for
OneVision reference).

**Calibration pool**: PlotQA + InfoVQA pooled wrong-base + 4-cond
eligible. n is *not* guaranteed equal to OneVision's ~5,000 —
outline §D.1 shows Qwen2.5-VL-7B is more anchor-resistant than
OneVision (df 0.146 vs 0.178 on broad cohort, lower base-wrong rate
correlates with higher base accuracy → potentially smaller wrong-base
pool). Actual `n_calibration` measured after Phase 0 prediction
inference + wrong-base + 4-cond eligibility filter; if `n < 2,000`,
flag as a Phase 1 caveat and consider expanding calibration pool to
include ChartQA or TallyQA before stopping.

**Pilot eval distributions**: PlotQA + InfoVQA (within-distribution
only at pilot stage, matching OneVision §A.5 protocol). 5-dataset
held-out eval reserved for Stage 4 on the chosen cell.

**Cell selection rule** (verbatim from outline §A.5):
- Δem(a) ≥ −6pp deal-breaker (memory [[feedback_em_drop_dealbreaker]])
- Rank surviving cells by combined `|Δdf(a)|` across PlotQA + InfoVQA
- Tie-break by Δem(b) (capability)

**Output**: `outputs/e6_steering/qwen2.5-vl-7b-instruct/_pilot_27cells/`
with per-cell summary + `docs/insights/_data/E6_pilot_grid_27cells_qwen.csv`.

### Phase 2 — Stage-4 5-dataset paired-bootstrap eval (`~2 H200-day`)

At the chosen cell from Phase 1, run full 5-dataset eval
(TallyQA + PlotQA + InfoVQA + ChartQA + MathVista) with paired-bootstrap
B=10,000 CI on Δdf(a), Δadopt(a), Δem(a), Δem(b). Same script as
OneVision: `scripts/build_e6_stage4_summary.py` +
`scripts/build_e6_stage4_bootstrap_ci.py` extended with `--model
qwen2.5-vl-7b-instruct` switch.

**Output**: `docs/insights/_data/stage4_final_per_dataset_ci_qwen.{csv,md}`
+ insight `docs/insights/E6-cross-arch-qwen-stage4.md`.

### Phase 3 — Capability preservation eval (`~2 H200-day`)

6-benchmark held-out eval (HallusionBench / RealWorldQA / MMStar /
POPE / MMBench-DEV-EN / OCRBench) under STRICT_FREE_LUNCH protocol,
identical to outline §6.3 + memory [[project_e8_capability_eval]] for
OneVision. VLMEvalKit operational quirks per
[[feedback_vlmevalkit_quirks]].

**Output**: `docs/insights/_data/capability_eval_per_benchmark_qwen.{csv,md}`.

## Success criteria (4-clause free-lunch, verbatim from outline §6.2.3)

1. **Anchoring reduction**: Δdf(a) < 0 with at least 1 dataset
   95% CI excludes 0 (matching OneVision's PlotQA-only single-dataset
   CI-clean; sample-size-bound on InfoVQA / TallyQA / ChartQA /
   MathVista is acceptable per outline §8 "Statistical scope").
2. **Both arms ≥ 0 on Em**: Δem(a) ≥ 0 mean *and* Δem(b) ≥ 0 mean
   across 5 datasets. Multiplicity-robust headline (matching OneVision):
   Δem(b) 5/5 cells × 95 % AND Bonferroni-20 corrected CIs exclude 0.
3. **Capability held**: 6-bench macro Δ ≥ −0.5pp (STRICT_FREE_LUNCH).
4. **At least 1 cell** in the 27-cell pilot grid passes (1)+(2)+(3).

Partial outcomes:
- **Best-case** (clauses 1+2+3 all pass): close §8 "Mitigation scope"
  magnitude-transfer gap. Outline §6.4 (new) reports Qwen2.5-VL Stage-4
  table. Lifts paper from Findings-conditional to Main-conditional met.
- **Clauses 1+3 pass, clause 2 fails on Em(b)**: report as "anchoring
  reduction transfers, capability gain is OneVision-specific". Still
  positive for §6 cross-arch but downgrades the *bonus capability*
  finding to single-arch.
- **Clauses 1 fails (0/27 cells reduce anchoring)**: report as bounded
  *recipe portability* result. Outline §6.4 reports null, §8
  "Mitigation scope" stays. Paper stays Findings.

## Risks + decision points

- **Risk: Qwen2.5-VL peak layer ≠ late residual.** If Phase 0 probe
  finds peak in mid-stack (e.g. L=14 like OneVision-on-InfoVQA per
  §5.3), grid center shifts — still 27-cell topology, but L bins
  reflect actual peak. **Mitigation**: depth-norm fallback rule in
  Phase 0.
- **Risk: SDPA precision drift** (memory [[feedback_sdpa_mask_hook_bug]],
  [[feedback_eager_sdpa_drift_check]]). E6 calibration uses
  `register_forward_hook` on residual, not attention_mask hooks, so
  SDPA-vs-eager drift is bounded — but cross-check the first 5
  calibration samples eager-vs-SDPA before launching Phase 1 sweep.
- **Risk: Qwen2 multi-token answer parsing**
  ([[project_phase1_resume_state]] Phase 1.5 fix). E6 pipeline uses
  `extract_first_number` on prediction text, not single-token
  `answer_token_logit` — should be robust on Qwen2.5-VL.
- **Decision point after Phase 1**: if 0/27 cells survive Δem(a)
  ≥ −6pp deal-breaker, stop and report null. Don't widen grid
  (per [[feedback_r4_overcorrection_filter]] — original recipe
  fail is a valid §6.4 result).

## Budget (revised total)

| Phase | Cost | Cumulative |
|---|---|---|
| Phase 0 peak probe | ~2 H100-hour | ~0.1 day |
| Phase 1 27-cell pilot | ~3 H200-day | ~3 day |
| Phase 2 Stage-4 paired CI | ~2 H200-day | ~5 day |
| Phase 3 6-bench capability | ~2 H200-day | ~7 day |
| **Total** | **~7 H200-day** | (Roadmap §7 P2-7 budgeted ~10 H200-day; this is tighter) |

## Paper alignment (outline_ko.md — canonical per [[feedback_paper_outline_canonical]])

This experiment closes / informs:

- **§6.4 (new)**: "Cross-architecture replication on Qwen2.5-VL-7B"
  subsection. Placeholder inserted in this PR; Stage 4 table populated
  after Phase 2.
- **§8 Limitations "Mitigation scope"**: edit to reflect Qwen2.5-VL
  replication *in progress* (was: "후속 작업"). After Phase 2 result
  lands, edit to reflect outcome (positive / partial / null).
- **§F Reproducibility Checklist**: add Qwen2.5-VL hyperparameter row.
- **Appendix slot (TBD)**: per-cell pilot grid table for Qwen2.5-VL
  (mirror of OneVision §A.5).

## Out-of-scope for this PR

This PR (`worktree-e6-cross-arch-qwen25vl`) lands **scaffolding only**:
- This design doc
- Outline §6.4 + §8 placeholder edits
- Roadmap §3 / §7 / §10 entries promoting P2-7 to active

Experiment execution (Phase 0 → 3) lands in subsequent PRs under the
same branch family (e.g., `e6-cross-arch-qwen25vl/phase1-pilot`,
`e6-cross-arch-qwen25vl/phase2-stage4`, etc.). User runs experiments
between PRs; agent stages results + writes evidence docs.

## References

- OneVision E6 worked example: outline §6.1–§6.3, insights
  `docs/insights/E6-pilot-grid-aggregation.md`,
  `docs/insights/E6-stage4-paired-bootstrap-ci.md`,
  `docs/insights/E6-leace-recalibration-evidence.md`.
- §5.1 peak-layer probe protocol: outline §5.1 + §G + memory
  [[project_paper_section4_plotqa_6bin_2026-05-10]].
- 4-clause free-lunch definition: outline §6.2.3.
- Roadmap P2-7 entry: `references/roadmap.md` §7.
- Bar-raiser conditional accept condition:
  `docs/paper/reviews/_final_summary.md` lines 36 + 77.
