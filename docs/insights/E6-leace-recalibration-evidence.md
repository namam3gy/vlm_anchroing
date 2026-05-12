# E6 LEACE re-calibration evidence (PR #35 plan execution)

**Date:** 2026-05-12
**Status:** complete; **Outcome B confirmed** per pre-registered decision tree.
**Plan:** `docs/superpowers/specs/2026-05-11-leace-recalibration-plan.md` (PR #35).
**Eraser tag:** `plotqa_infovqa_recal` at `outputs/e6_steering/llava-onevision-qwen2-7b-ov/leace_erasers_plotqa_infovqa_recal/`.

## 1. Headline

**§6.4's "LEACE rank-1 ChartQA +56 % reversal" claim does not reproduce on OneVision Main under PlotQA+InfoVQA pooled calibration.** ChartQA at α=0.5 shows mild Δdf = **−0.027 [−0.054, −0.005]** — a CI-clean *mitigation*, opposite direction from the claimed +56 % backfire. At α=1.0 and α=2.0, ChartQA Δdf is null (CI overlaps 0). The original §6.4 finding was produced on `llava-next-interleaved-7b` (old Main, before 2026-05-04 architecture flip) with calibration pool VQAv2 + TallyQA + ChartQA; the current paper-wide-consistency rule treats §6.4 as an OneVision claim, but no OneVision LEACE artifact existed prior to this re-calibration. The re-calibration therefore *changes two axes simultaneously* (model + pool), and the prior +56 % is shown to be at least one of: (a) Interleave-specific architectural quirk, (b) calibration-pool artifact (TallyQA-heavy VQA+Tally+Chart distribution), or (c) both.

**Conclusion:** §6.4 LEACE row is *not* a robust single-direction failure anchor for the §5.4 routing-vs-integration framework on the current paper-wide Main model. The §5.4 framework's confirmatory weight redistributes to the §4.6 γ-β layer-routing partial-prospective verification (the K=1 sign-reversal cell).

## 2. Setup (axis-pinned C-prime path, per user decision)

| Axis | Original §6.4 LEACE | Re-calibration (this run) |
|---|---|---|
| Model | `llava-next-interleaved-7b` | `llava-onevision-qwen2-7b-ov` (current Main) |
| Calibration pool | VQAv2 + TallyQA + ChartQA (n=1145) | **PlotQA + InfoVQA** (n=2,757 pooled wrong-base) |
| Class 0 (no anchor) | `h^b` (target_only) | `h^b` (target_only) — **same algorithmic substrate** |
| Class 1 (with anchor) | `h^b + D` where D = h^a − h^m | `h^b + D` — **same algorithmic substrate** |
| Apply layer | L=28-30 (sweep) | **L=26** (mirrors E6 §6.2 cell #17 integration site) |
| Erasure strength α | 0.5 / 1.0 / 2.0 | **0.5 / 1.0 / 2.0** (3α sweep from start) |

**Decision (vs spec literal):** the spec literal asks for raw `h^m`/`h^a` class definitions; user chose **C-prime** (`h^b` / `h^b+D` — same algorithmic substrate as original §6.4) so that *only the calibration pool axis changes* between original §6.4 and this re-calibration. This isolates the pool-confound check from the substrate-axis change.

### 2.1 Capture (new artifacts on OneVision Main)

- `outputs/e6_steering/llava-onevision-qwen2-7b-ov/calibration_plotqa/Q_wrong.pt` — shape (2502, 28, 3584), n_wrong=2502 captured, 21.3 min wall on H200 (SDPA + inference_mode patch).
- `outputs/e6_steering/llava-onevision-qwen2-7b-ov/calibration_infographicvqa/Q_wrong.pt` — shape (480, 28, 3584), n_wrong=480 captured, 9.2 min wall.
- `outputs/e6_steering/llava-onevision-qwen2-7b-ov/leace_erasers_plotqa_infovqa_recal/P_stack.pt` — shape (28, 3584, 3584), pooled N=2757 (Q+D matched per-dataset to D shape, PlotQA N=2314 + InfoVQA N=443).
- Rank-1 verification at L=26: `‖I − P‖_F = 1.82`, top SV of (I−P) = 1.82, rest ≤ 2×10⁻⁷ — single dominant erasure direction confirmed.

### 2.2 Evaluation (5-dataset sweep)

- `outputs/e6_steering/llava-onevision-qwen2-7b-ov/sweep_leace_<ds>_recal_pooled/predictions.jsonl` for each of `chartqa, mathvista, infographicvqa, plotqa, tallyqa`.
- 4 cells per dataset: baseline + L26_a{0.5, 1.0, 2.0}.
- Eval predictions reused from existing OneVision E5b/E5e runs (same sample_instance lists as §6.2 chosen sweep).

## 3. Headline table (Δdf(a), paired-bootstrap 95 % CI, B = 10,000)

Source: [`docs/insights/_data/leace_recal_per_dataset_ci.csv`](_data/leace_recal_per_dataset_ci.csv).

| Dataset | n_paired | α=0.5 Δdf [95% CI] | α=1.0 Δdf [95% CI] | α=2.0 Δdf [95% CI] |
|---|---:|---:|---:|---:|
| **ChartQA** | 224 | **−0.027 [−0.054, −0.005]** ✓ mit | −0.022 [−0.045, +0.000] ~ | −0.018 [−0.058, +0.022] null |
| PlotQA | 2308 | −0.006 [−0.013, +0.001] ~ | −0.005 [−0.013, +0.003] null | −0.002 [−0.011, +0.007] null |
| InfoVQA | 443 | −0.009 [−0.025, +0.007] null | −0.007 [−0.025, +0.014] null | −0.018 [−0.043, +0.005] null |
| TallyQA | 2493 | +0.001 [−0.004, +0.006] null | **+0.006 [+0.000, +0.012]** ~bf | +0.006 [−0.003, +0.014] null |
| MathVista | 170 | −0.006 [−0.041, +0.029] null | +0.000 [−0.035, +0.035] null | +0.000 [−0.041, +0.041] null |

**Marker key**: ✓ = CI-clean (95 % excludes 0 in mitigation direction); ~bf = borderline backfire (CI lower at exact 0 — discretization floor on small bootstrap tails); ~ = CI close to but overlaps 0; null = CI clearly overlaps 0.

### 3.1 ChartQA — direct test of §6.4's +56 % claim

| α | Δdf (rate) | Δdf (%-relative to baseline 0.214) | 95% CI |
|---|---:|---:|---:|
| 0.5 | −0.027 | **−12.5 %** | [−25.0 %, −2.1 %] |
| 1.0 | −0.022 | **−10.4 %** | [−20.8 %, +0.0 %] |
| 2.0 | −0.018 | **−8.3 %** | [−27.1 %, +10.4 %] |

The §6.4 claim of "+56 %" reversal on ChartQA: **does not reproduce in any of the three α cells**. The closest signed value is α=2.0 with point estimate ‑8.3 % relative; CI upper bound on α=2.0 only reaches +10.4 % (less than one-fifth of the +56 % claim). At α=0.5, the result is **CI-clean mild mitigation** (−12.5 %).

### 3.2 Δem(b) capability preservation (non-anchored arm)

| Dataset | α=0.5 [CI] | α=1.0 [CI] | α=2.0 [CI] |
|---|---|---|---|
| PlotQA | +0.001 [−0.000, +0.004] | **+0.004 [+0.002, +0.007]** ✓ | **+0.009 [+0.005, +0.013]** ✓ |
| InfoVQA | **+0.011 [+0.002, +0.023]** ✓ | **+0.011 [+0.002, +0.023]** ✓ | **+0.023 [+0.009, +0.038]** ✓ |
| TallyQA | **+0.016 [+0.012, +0.022]** ✓ | **+0.027 [+0.021, +0.034]** ✓ | **+0.052 [+0.044, +0.061]** ✓ |
| ChartQA | +0.013 [+0.000, +0.031] ~bf | +0.005 [+0.000, +0.013] ~bf | **+0.027 [+0.009, +0.049]** ✓ |
| MathVista | +0.018 [+0.000, +0.041] ~bf | +0.012 [+0.000, +0.029] ~bf | **+0.035 [+0.012, +0.065]** ✓ |

**At α=1.0:** PlotQA/InfoVQA/TallyQA all CI-clean positive; ChartQA/MathVista borderline (lower bound at exact 0). At α=2.0: all 5 datasets CI-clean positive. **Capability preservation holds even at α=2.0 over-erasure**, which is itself an informative finding — the single direction LEACE removes IS not load-bearing for the non-anchored arm task.

## 4. Mapping to spec §3 pre-registered outcomes

Spec §3 outcome tree, evaluated against ChartQA Δdf at α=1.0 (primary cell):

- **A (ChartQA still backfires)**: requires Δdf > 0 with 95 % CI excluding 0. Observed Δdf = **−0.022 [−0.045, +0.000]**. **Outcome A is rejected.**
- **B (ChartQA no longer backfires)**: requires Δdf ≤ 0 or 95 % CI overlapping 0 from the backfire side. Observed satisfies both — point estimate negative, CI upper bound at exact 0. **Outcome B holds.**
- **C (mixed across α)**: would require α=0.5 backfire + α=1.0 not, or similar inconsistency. Observed: all three α cells have negative or null point estimates on ChartQA, **monotonically** approaching 0 as α decreases. No inconsistency. **Outcome C is rejected.**

**Pre-registered §3 Outcome B paper-update branch applies.**

## 5. Paper-update plan (per Outcome B branch)

### 5.1 §6.4 (Table 7 LEACE row + surrounding prose)

**Current text (line 396, paraphrase):**
> "LEACE closed-form (rank-1) | ❌ ChartQA backfire +56 % (gt ∈ [0,8]) | invariant | invariant | single-direction redundancy"

**Proposed replacement (Outcome B branch, per spec §3):**
> "LEACE closed-form (rank-1) | ChartQA Δdf = −0.022 [−0.045, +0.000] @ α=1.0 (mild mitigation, CI borderline); +56 % backfire on `llava-next-interleaved-7b` does NOT reproduce on Main model with PlotQA+InfoVQA (a−m) recalibration | unchanged em(a) | +0.5–1.1 pp em(b) | direction-mismatch baseline weaker than originally reported"

§6.4 prose should add a paragraph reporting the re-calibration result and explicitly disclosing:
- Original +56 % was on Interleave + VQA/Tally/Chart pool;
- Re-calibration on OneVision + PlotQA/InfoVQA (a−m) pool eliminates the backfire;
- Confound axes: model swap and pool change covaried, so the result alone does not resolve which axis matters.

The "single-direction ActAdd cross-dataset failure" finding (separate row in Table 7) remains independent — it lives on the cos(v_tally, v_chartqa) ≈ 0.47–0.62 direction-mismatch measurement, which is calibration-independent.

### 5.2 §5.4 (routing-vs-integration framework — confirmatory anchor list)

**Current text (line 266, paraphrase):**
> "§5.2 multi-layer redundancy + §5.3 OneVision dataset-dependent peak + **§6.4 LEACE rank-1 ChartQA +56 % reversal** — these three mechanism findings were all observed *before* this framework was written, and this section is *post-hoc* synthesis of those three results into a single mechanism narrative."

**Proposed replacement:**
> "§5.2 multi-layer redundancy + §5.3 OneVision dataset-dependent peak — these two mechanism findings were observed before this framework was written, and this section is *post-hoc* synthesis of those two results into a single mechanism narrative. §6.4 LEACE rank-1 cross-dataset behavior was *originally interpreted* as confirmatory anchor on Interleave (PR-archive); re-calibration on OneVision Main with PlotQA+InfoVQA (a−m) pool removes the backfire signal (mild mitigation or null, depending on α), so it no longer serves as confirmatory anchor for this framework on the current Main model."

### 5.3 §8.4 disclosure addendum

Add follow-up item:
- **§6.4 single-direction failure baseline robustness**: re-calibration on OneVision + PlotQA/InfoVQA (a−m) pool weakens the original Interleave + VQA/Tally/Chart result. Robustness to (model × pool) full factorial design not yet performed. Next-round: (i) random-direction baseline at same (rank=1, pool), (ii) cross-architecture LEACE re-calibration on the perfect-square mechanism panel.

### 5.4 §1.5 / Abstract — no change required

The thesis sentence (PR #27) does not depend on the §6.4 row holding as confirmatory anchor. The "design pattern" framing is supported by multi-direction E6 (the actual mitigation), not by single-direction baselines. No edit needed.

## 6. Limitations / caveats

- **Two axes changed simultaneously** (model + pool) vs original §6.4. The re-calibration alone does not isolate which axis drives the result — only confirms the original claim is brittle under at least one of the two changes.
- **ChartQA n=224 is small**; the CI upper bound at α=1.0 sits at exact 0 (the discretization floor). The mild mitigation at α=0.5 (CI-clean) is the most informative cell. At α=2.0 the CI [−0.058, +0.022] does include +0.022 (≈ +10 %), but this is one-fifth of the +56 % claim and the point estimate remains negative.
- **TallyQA α=1.0 shows +0.6 pp Δdf with CI lower at 0.0** — borderline backfire. Magnitude is ~10× smaller than the original §6.4 ChartQA +56 % claim, and α=0.5/α=2.0 cells do not show it CI-clean. Not informative for the §6.4 claim test but worth flagging in §6.5.
- **Spec literal (Path A, raw h^m / h^a class definition)** was not run — user chose C-prime to keep the algorithmic substrate matched to original §6.4. Path A could change the result if (h^m, h^a) substrate yields a more aggressive eraser than (h^b, h^b+D). Future work.

## 7. Reproducibility

```bash
# (1) Capture Q_wrong on PlotQA + InfoVQA (OneVision Main, SDPA, ~30 min total)
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  uv run python scripts/e6_query_adaptive_offset.py --phase calibrate-qao \
  --model llava-onevision-qwen2-7b-ov \
  --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --config configs/experiment_e5b_5strat_plotqa_onevision.yaml \
  --predictions-path outputs/experiment_e5b_5strat_plotqa_onevision/.../predictions.jsonl \
  --dataset-tag plotqa --max-calibrate-pairs 2502
# (then again with --dataset-tag infographicvqa, --max-calibrate-pairs 1076)

# (2) Fit LEACE pooled (CPU, ~100 s)
uv run python scripts/e6_leace.py --phase calibrate-leace \
  --model llava-onevision-qwen2-7b-ov \
  --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --calib-tags plotqa,infographicvqa \
  --eraser-tag plotqa_infovqa_recal

# (3) 5-dataset sweep at L=26 × α∈{0.5, 1.0, 2.0} (~6 h total)
bash scripts/_leace_recal_sweep.sh

# (4) Aggregate + paired bootstrap CI
uv run python scripts/build_e6_leace_recal_bootstrap_ci.py \
  --bootstrap 10000 --seed 20260511
```

## 8. Patches applied to scripts (during execution)

- `scripts/e6_query_adaptive_offset.py`: `_capture_last_token_residuals` wrapped in `@torch.inference_mode()`; added `del out, hidden, inputs; gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()` per iteration to fix activation-leak OOM on OneVision + AnyRes images. `_build_runner` switched from `build_eager_runner` to `build_runner(..., attn_implementation="sdpa")` for ~10× memory reduction without affecting hidden-state outputs.
- `scripts/e6_leace.py`: same SDPA + inference_mode patches applied to its `_capture_last_token_residuals` and `_build_runner`. Smoke phase hardcoded `L=28` (out of bounds for OneVision n_layers=28) replaced with `L=26 if n_layers > 26 else n_layers-1`.
- Both patches are diagnostic-class fixes (memory hygiene, not algorithmic change) — same hidden-state outputs as eager attention, validated via standalone debug script (`scripts/_debug_onevision_memory.py`).
