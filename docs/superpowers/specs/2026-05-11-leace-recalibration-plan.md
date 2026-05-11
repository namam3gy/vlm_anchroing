# LEACE re-calibration on PlotQA + InfoVQA — plan

**Date:** 2026-05-11
**Status:** plan-only, not yet executed
**Motivation source:** author noticed §6.4 LEACE baseline result ("rank-1 closed-form ChartQA +56 % reversal") rests on calibration data with non-general label distribution (TallyQA-heavy / pooled VQA+Tally+ChartQA per `docs/experiments/E6-steering-vector.md` line 783). When §6.4 carries weight as the *single confirmatory anchor* of the §5.4 routing-vs-integration framework (see §5.4 prose), the calibration confound risk propagates upward to framework strength.

## 1. Goal

Run a controlled LEACE re-calibration on **the same (a − m) paired-inpaint contrast that E6 uses** (PlotQA + InfoVQA pooled, OneVision Main `llava-onevision-qwen2-7b-ov`, L=26), then re-evaluate cross-dataset df/em to determine whether §6.4's "single-direction cross-dataset failure" finding survives a calibration-confound control.

Outcome of this experiment is *informative in either direction* (see §4 below). Result drives the §5.4 framework prose and §6.4 baseline-comparison table.

## 2. Setup

### 2.1 Calibration (mirrors E6 §6.2 calibration substrate)

| step | detail |
|---|---|
| Model | `llava-onevision-qwen2-7b-ov` (Main; mirrors E6 §6.2) |
| Calibration data | PlotQA + InfoVQA pooled wrong-base (a − m) pairs; reuse E6 calibration substrate if available, else regenerate with same stratification |
| Class 0 (no anchor) | h^m at L=26 — masked-anchor representation (mirrors E6 (a − m) convention; differs from current LEACE class 0 = h^b in `e6_leace.py` which uses target_only baseline) |
| Class 1 (with anchor) | h^a at L=26 — anchor-present representation |
| Method | LEACE rank-1 closed-form (`LeaceEraser.fit`, idempotent affine projection) |
| Apply layer | L = 26 (OneVision integration site, mirrors E6 K=8 cell #17) |
| α sweep | α ∈ {0.5, 1.0, 2.0} per existing LEACE sweep convention; primary cell α = 1.0 (full erasure) |

**Key distinction from existing LEACE result in `E6-steering-vector.md` Method 4c**:
- Current LEACE Method 4c: class 0 = b-arm (target_only), class 1 ≈ b-arm + (a−m) diff; calibration pool = VQAv2 + TallyQA + ChartQA (1145 paired)
- This re-calibration: class 0 = m-arm (digit-pixel-inpainted), class 1 = a-arm directly; calibration pool = PlotQA + InfoVQA (no TallyQA / VQAv2)

This isolates the LEACE method from the calibration-data axis — same algorithm, same (a − m) contrast as E6, only the pool changes.

### 2.2 Evaluation (5-dataset cross-validation)

Same 5-dataset evaluation pipeline as E6 §6.2 / §6.5:

- PlotQA (in-calibration, expected — sanity check)
- InfoVQA (in-calibration, expected — sanity check)
- TallyQA (out-of-calibration — primary cross-dataset test 1)
- **ChartQA (out-of-calibration — primary cross-dataset test 2; locus of current §6.4 +56 % backfire claim)**
- MathVista (out-of-calibration — secondary cross-dataset test)

**Primary metric**: `Δdf(a)` cross-dataset, focus on ChartQA.

**Secondary metrics**: `Δem(a)`, `Δem(b)`, paired-bootstrap 95 % CI per dataset (B = 10,000).

**Calibration cells reused from E6**: same stratification + (b/a/m/d) 4-condition `predictions.jsonl` infrastructure; reuse n_pair counts where available.

## 3. Pre-registered outcomes (decided before running)

The result determines §6.4 + §5.4 framing per the following decision tree, *registered before inference*:

### Outcome A — ChartQA still backfires under PlotQA+InfoVQA calibration

LEACE rank-1 (a − m) calibration on PlotQA + InfoVQA → ChartQA Δdf > 0 with 95 % CI excluding 0.

**Interpretation**: TallyQA / VQAv2 inclusion was *not* the confound. Single-direction LEACE rank-1 fails on ChartQA *regardless of calibration pool*. §6.4 confirmatory anchor **strengthens** — "single-direction failure" finding holds across two independent calibration choices.

**Paper updates (Outcome A)**:
- §6.4 prose: add one paragraph reporting the re-calibration result and explicitly addressing the label-distribution confound — "ChartQA backfire reproduces with PlotQA+InfoVQA calibration on the same (a−m) contrast used by E6, ruling out TallyQA-distribution as confound".
- §5.4 framework prose: §6.4 confirmatory anchor strength preserved.
- §8.4 follow-up: this item can close.

### Outcome B — ChartQA no longer backfires under PlotQA+InfoVQA calibration

LEACE rank-1 (a − m) calibration on PlotQA + InfoVQA → ChartQA Δdf ≤ 0 (or 95 % CI overlaps 0).

**Interpretation**: TallyQA / VQAv2 inclusion *was* a confound — the current §6.4 +56 % backfire result is calibration-data artifact, not generic single-direction failure. §6.4 confirmatory anchor **weakens**.

**Paper updates (Outcome B)**:
- §6.4 prose: re-cast LEACE result with honest disclosure — "+56 % backfire under TallyQA-heavy calibration; with E6-matched PlotQA+InfoVQA (a−m) calibration the result attenuates (or reverses)". Move some of the "single-direction cross-dataset failure" weight to the ActAdd row + the cos(v_tally, v_chartqa) ≈ 0.47-0.62 direction-mismatch evidence (which is calibration-independent).
- §5.4 framework prose: §6.4 confirmatory anchor downgrades. Either §6.2.4 attention-pathway result (in parallel session) or §4.6 14/84 directional pattern takes more of the framework weight.
- §8.4 follow-up: register "stronger single-direction baseline panel — random-direction baseline + multiple calibration pools" as next-round work.

### Outcome C — Mixed (single-cell pass at α < 1 but α=1 still backfires)

LEACE at α ∈ {0.5} passes cross-dataset but α = 1.0 (full erasure, primary cell) still backfires.

**Interpretation**: Method is sensitive to erasure strength; the "rank-1 closed-form" claim survives at *full* erasure but partial erasure recovers. Mixed message.

**Paper updates (Outcome C)**: Report the α-sweep explicitly. §6.4 LEACE row in Table 7 split into α=1.0 (backfire) and α=0.5 (pass). §5.4 framework prose adds a hedge on "rank-1 erasure strength as additional axis".

## 4. Cost estimate

| step | cost | notes |
|---|---|---|
| LEACE re-calibration (PlotQA + InfoVQA pooled, L=26) | CPU-only, ~3-5 min | closed-form SVD on existing residual capture |
| 5-dataset eval inference (α=1.0, primary) | ~1-2 H100-hour | OneVision Main, reuse E6 stratified `sample_instance` lists; expected n_pair PlotQA ~2,306, TallyQA ~6,934, ChartQA ~517, MathVista ~127, InfoVQA ~865 (E6 §3.3 ranges) |
| α-sweep (α ∈ {0.5, 2.0}) | ~2-3 H100-hour | only if Outcome C is detected at α=1.0 |
| Aggregation + paired bootstrap CI | ~30 min CPU | reuse `scripts/build_e6_stage4_bootstrap_ci.py` |
| **Total** | **~1.5-5 H100-hour** | upper bound includes α-sweep |

If E6 calibration residual capture is gitignored-out and needs regeneration: add ~2-3 H100-hour for PlotQA + InfoVQA inference pass on OneVision Main. Most likely path: residual capture exists from E6 §6.2 work, reuse.

## 5. Reproducibility

```bash
# (1) Identify or regenerate E6 (a − m) residual capture on PlotQA + InfoVQA pooled
ls outputs/e6_steering/llava-onevision-qwen2-7b-ov/_subspace/  # should contain pooled D matrix or per-dataset captures
# If missing, regenerate via:
# uv run python scripts/e6_steering_vector.py --phase capture-am-diff \
#     --model llava-onevision-qwen2-7b-ov --datasets plotqa,infovqa \
#     --layer 26

# (2) Fit LEACE on (h^a, h^m) at L=26, PlotQA + InfoVQA pooled
uv run python scripts/e6_leace.py --phase calibrate-leace-am \
    --model llava-onevision-qwen2-7b-ov --layer 26 \
    --calibration-datasets plotqa,infovqa \
    --class-0-rep masked --class-1-rep anchor \
    --output outputs/e6_leace_recalibration/<ts>/

# (3) Apply at α=1.0 on 5 evaluation datasets
uv run python scripts/e6_leace.py --phase sweep-leace \
    --model llava-onevision-qwen2-7b-ov --layer 26 \
    --eval-datasets plotqa,infovqa,tallyqa,chartqa,mathvista \
    --alpha 1.0 \
    --leace-projection outputs/e6_leace_recalibration/<ts>/P_L26.pt

# (4) Aggregate + bootstrap CI
uv run python scripts/build_e6_stage4_bootstrap_ci.py \
    --predictions outputs/e6_leace_recalibration/<ts>/predictions/ \
    --bootstrap 10000

# (5) Evidence doc + paper update
# docs/insights/E6-leace-recalibration-evidence.md
# docs/paper/emnlp_draft_ko.md §6.4 + §5.4 per outcome class
```

**Note**: `scripts/e6_leace.py` may need a `--class-0-rep masked` flag added if the current implementation hardcodes class 0 = b-arm. ~1 hour of script work before inference.

## 6. Out-of-scope for this experiment

- Multi-direction LEACE (rank > 1): the current §6.4 "single-direction failure" framing uses rank-1 default; multi-direction LEACE conflates with E6 multi-direction subspace projection and is a separate baseline (see §6.5 Table 8 Note 1 CAA structural reduction discussion).
- Different layers (L ≠ 26): use E6 integration-site cell to keep the apples-to-apples comparison with E6's own L=26 cell.
- Cross-architecture LEACE re-calibration: separate work (§8.4 item 3 cross-architecture E6).
- Random-direction baseline at same (rank, calibration): separate work (§8.4 item 2 random-K=8 baseline).

## 7. Cross-references

- Source paper §: `docs/paper/emnlp_draft_ko.md` §6.4 (lines 383-385 + Table 7 LEACE row line 396), §5.4 (line 254 + line 266)
- Existing LEACE experiment: `docs/experiments/E6-steering-vector.md` Method 4c (lines 768-841) — Class 0 = b-arm, pool = VQA + Tally + ChartQA
- E6 (a − m) calibration substrate: §6.2.1 Insight ((a − m) contrast 핵심) + design rationale for PlotQA + InfoVQA pool
- Decision driver: author flagged the TallyQA-heavy LEACE calibration confound 2026-05-11 while reviewing PR #31 (worktree-paper-section4-6-prereg-cell, closed as over-correction)

## 8. Pre-flight check before execution

Before kicking off inference:

1. ✅ Outcome A / B / C decisions registered in this spec (above §3); paper update branches pre-specified per outcome
2. □ Verify E6 (a − m) residual capture availability for OneVision Main on PlotQA + InfoVQA
3. □ Verify `scripts/e6_leace.py` supports class-0-rep = masked (or add flag, ~1 hour script work)
4. □ Confirm n_pair counts for evaluation datasets match E6 §6.2.2 baseline numbers
5. □ Commit spec freeze before running inference

Execution starts after items 2-5 are checked.
