# γ-β Residual-Stream Bridge — Evidence

**Date:** 2026-05-10
**Branch:** `worktree-phase5+p0-1-gamma-beta-bridge`
**Spec:** [`docs/superpowers/specs/2026-05-10-p0-1-gamma-beta-bridge-design.md`](../superpowers/specs/2026-05-10-p0-1-gamma-beta-bridge-design.md)
**Source plan:** [`plan_post_review_2026-05-09.md`](plan_post_review_2026-05-09.md) §P0-1 (R5 bar-raiser signature ask)

## TL;DR

**Verdict — Alt-1 falsified** (per spec §4 acceptance criteria). Qwen3-VL-Thinking trace residual amplitude on the K=8 anchor subspace at L=33 *does* exceed Qwen3-VL-Instruct baseline by 1.24× with paired-bootstrap CI excluding zero — but **the same magnitude effect appears on the neutral `d` arm with zero anchor present**. The uniform Thinking-mode amplitude growth is a generic length / reasoning-trace effect, not an anchor-specific subspace activation. The mechanism-level bridge between paper §4.6 (γ-β behavioral amplification) and §6 (K=8 anchor subspace mitigation) is **not established by this experiment** at K=8 / L=33. §4.6 stays as behavioral existence-proof; §8.2 limitation extended.

## Setup

### Calibration (Phase B, 2026-05-10)
Path B Qwen3-VL self-calibration (no cross-arch hand-wave on OneVision V_K) per spec §2.1.

| Step | Detail |
|---|---|
| Calibration model | `Qwen/Qwen3-VL-8B-Instruct` (eager-attention smoke verified, SDPA used for production for memory) |
| Stimuli | PlotQA test V1 (n=5000 stratified) + InfographicVQA val (n=1147 numeric); 4-condition (`b/a-S1/m-S1/d`) |
| Wrong-base diffs | PlotQA 1017 + InfoVQA 220 = **1237 paired (h_a − h_m) at L∈{0..35}** |
| SVD output | `outputs/e6_steering/qwen3-vl-8b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt` shape `(36, 16, 4096)` |
| Spectrum at L=33 | sv[0..7] = `[577, 270, 246, 185, 177, 156, 142, 135]`; **sv7/sv8 elbow ratio = 1.026** (weak — K=8 is paper §6 prior, not spectrum-predicted at this calibration n) |

### Bridge inference (Phase C)

| Step | Detail |
|---|---|
| Stimuli | γ-β MathVista S1 (existing inputs/mathvista_testmini, integer-GT, gt ≤ 1000); 2 arms `a-S1` + `d` |
| Models | `Qwen/Qwen3-VL-8B-Instruct` + `Qwen/Qwen3-VL-8B-Thinking` |
| Capture | per-generated-token L∈{29, 30, 33, 34} residual; primary L=33 (= 33/36 ≈ OneVision 26/28 fraction) |
| Sampling | temperature=0, max_new_tokens=512 |
| Output | 1091 paired (sid, condition) records each model |
| Median trace length | Instruct ~7 tokens; Thinking ~480 tokens (after `</think>` close) |

### Bridge statistic (Phase D)

| Step | Detail |
|---|---|
| Per-trace aggregation | `mean_t ‖V_K^T h(x_t, L=33)‖_2` (paper convention) |
| Pairing | by `sample_instance_id × condition` between Instruct + Thinking |
| Sub-group | `base_correct = (γ-α MathVista Instruct's pred_b == ground_truth)` from `outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-instruct/20260428-114421/predictions.jsonl` |
| Test | paired bootstrap B=10,000, α=0.05 on per-item Δ = thinking_mean − instruct_mean |

## Bridge results

Generator: [`scripts/build_gamma_beta_bridge_summary.py`](../../scripts/build_gamma_beta_bridge_summary.py).
Canonical CSV: [`docs/insights/_data/gamma_beta_bridge_amplitude_per_trace.csv`](_data/gamma_beta_bridge_amplitude_per_trace.csv).

| arm | base | n | mean Δ | 95 % CI | mean ratio | CI excludes 0 |
|-----|------|---:|------:|---------|----------:|:-:|
| **a-S1 (anchor)** | all | 522 | **+45.115** | **[+44.134, +46.088]** | **1.24×** | ✓ |
| a-S1 | correct | 238 | +45.983 | [+44.551, +47.344] | 1.25× | ✓ |
| a-S1 | wrong | 125 | +40.870 | [+38.564, +43.070] | 1.22× | ✓ |
| **d (control)** | all | 569 | **+44.675** | **[+43.735, +45.603]** | **1.24×** | ✓ |
| d | correct | 249 | +45.519 | [+44.103, +46.889] | 1.24× | ✓ |
| d | wrong | 134 | +41.259 | [+38.980, +43.438] | 1.22× | ✓ |

### Difference-in-differences

The decisive measurement: paired DiD per `sample_instance_id` (n=522 with both arms) of `(Δ_a − Δ_d) = (thinking_a − instruct_a) − (thinking_d − instruct_d)`:

| sub-group | n | DiD mean | DiD std |
|---|---:|---:|---:|
| all | 522 | **+0.471** | 5.585 |
| correct-base | 238 | +0.395 | 4.65 |
| wrong-base | 125 | +0.115 | 7.42 |

DiD effectively zero with std ~5–7. The `a-S1` vs `d` Δ difference is **noise floor at every base sub-group** — Thinking-mode amplitude growth is uniform across anchor presence/absence.

## Per-token trajectory

Median amplitude per generated-token position over Thinking traces (primary L=33, sample n_a=522, n_d=569; capped at 300 tokens):

| token position | a-S1 median (IQR) | d median (IQR) | gap (a − d) |
|---:|---:|---:|---:|
| t=0 | 143.4 (141.4, 145.7) | 143.0 (141.1, 145.2) | +0.4 |
| t=10 | 256.0 (241.1, 267.9) | 257.2 (242.2, 267.0) | −1.2 |
| t=50 | 245.0 (225.0, 261.7) | 246.4 (226.9, 263.2) | −1.4 |
| t=100 | 242.3 (216.9, 256.6) | 238.6 (211.7, 256.0) | +3.7 |
| t=150 | 236.1 (213.7, 254.2) | 236.5 (211.5, 253.8) | −0.5 |
| t=200 | 233.8 (209.3, 248.8) | 235.3 (211.2, 251.9) | −1.5 |
| t=250 | 235.6 (215.0, 252.8) | 234.9 (215.8, 248.9) | +0.7 |

**Pattern**: Thinking trace amplitude ramps from ~143 (first generated token) to ~250 plateau within ~10 tokens, then is stable through the rest of the trace. The ramp-and-plateau shape is identical between anchor (a-S1) and neutral (d) arms — IQR overlap on every token, gaps within ±4. Visualization: [`notebooks/gamma_beta_bridge_amplitude.ipynb`](../../notebooks/gamma_beta_bridge_amplitude.ipynb).

## Verdict against spec acceptance

| spec § acceptance | result |
|---|---|
| Primary positive (a-S1 CI excludes 0 with positive Δ) | ✓ formal pass — Δ=+45.1 [+44.1, +46.1] |
| Quantitative confirm (correct-base ratio ≥ 2×) | ✗ ratio 1.25× (target ≥ 2×) |
| Stronger quantitative (ratio near ×12.7) | ✗ |
| Alt-1 falsification (d arm same Δ as a-S1) | **✓ falsified** — d Δ=+44.7, identical magnitude |

**Outcome class — spec §1.2 Adverse (Alt-1 falsified)**: bridge falsified by length/reasoning-trace confound. The 1.24× amplitude growth observed in Thinking traces is *real* (CI excludes 0) but it is NOT anchor-specific — neutral-arm exhibits the same effect.

## Why the bridge fails (interpretation)

Two non-mutually-exclusive interpretations consistent with the data:

1. **Length / amplitude-plateau effect (most parsimonious)**. Thinking traces generate 313–480 tokens at a higher residual-stream amplitude plateau (~234) than Instruct's brief (~7 token) terminal answer phase (~180). Mean amplitude over Thinking trace is dominated by reasoning-content tokens; the K=8 calibration subspace happens to contain dimensions that align with this generic Thinking-mode distribution. The fact that this happens on neutral `d` arm too — where there is *no anchor digit pixel anywhere in input* — confirms the activation is reasoning-trace-driven, not anchor-driven.

2. **K=8 / L=33 calibration subspace is not anchor-specific enough**. Self-calibration on PlotQA + InfoVQA `(a − m)` may have captured (i) digit-pixel anchor signal (intended), but also (ii) generic chart/infographic structural variations between paired anchor and masked images that are unrelated to anchoring. The subspace's anchor-specificity at K=8 is weak (sv7/sv8 elbow = 1.026 — gradual decay, not sharp dimensionality). A larger calibration set, sharper digit-bbox-only `(a − m)` design, or layer sweep might isolate cleaner anchor dimensions.

The §6 strict-free-lunch result for the OneVision Main model on the *same* L=26, K=8 subspace remains valid — that result is verified by held-out eval datasets and 6-bench capability preservation, not by per-token amplitude. The bridge experiment's null tells us only that *the K=8 subspace's anchor-specificity is too weak to dominate Thinking-mode reasoning-trace activation* — not that anchor mechanism doesn't exist or that §6 mitigation is questionable.

## Paper updates

Per spec §1.2 *Adverse (Alt-1 falsified)*:

- **§4.6.1 sub-section is NOT authored.** Bridge claim does not survive falsification.
- **§4.6** (existing γ-β behavioral) **stays unchanged** — the ×1.6 adopt / ×2.9 df behavioral amplification is observation; this experiment did not validate or invalidate it.
- **§1.5 (5) hedge stack** stays at current strength (single-model E6 deployable case study).
- **§8.2 limitation extended** with the bridge null finding.
- **§8.4 item 1** struck-through-and-annotated with the null result, not removed (so future readers see the experiment was attempted).

Suggested §8.2 paragraph (English working draft):

> γ-β residual-stream bridge experiment (Phase 5 P0-1, 2026-05-10): we attempted to interlock the γ-β behavioral amplification of §4.6 with the K=8 anchor subspace of §6 by projecting Qwen3-VL-Thinking trace residuals onto a self-calibrated V_K[L=33] (Qwen3-VL-Instruct, PlotQA + InfoVQA pooled n_wrong=1237). Thinking traces show statistically larger mean amplitude than Instruct (Δ=+45.1, paired bootstrap 95 % CI [+44.1, +46.1]), but the same effect appears on the neutral d arm (Δ=+44.7, [+43.7, +45.6]) — Thinking-mode reasoning-trace dynamics broadly activate the K=8 subspace rather than amplifying anchor-specifically. The mechanism-level bridge between behavioral γ-β amplification and the §6 mitigation subspace is therefore not established at this calibration scope; reframing the bridge with a digit-bbox-restricted (a−m) calibration, or testing finer-grained subspace dimensions, is left to future work.

(Korean translation pending paper-revise pass.)

## Reproducibility

```bash
# Phase B — calibration (~3.3 H200-hour total)
uv run python scripts/run_experiment.py \
    --config configs/p0_1_calibration_qwen3vl_plotqa.yaml --max-samples 5000
uv run python scripts/run_experiment.py \
    --config configs/p0_1_calibration_qwen3vl_infovqa.yaml --max-samples 5000
uv run python scripts/e6_steering_vector.py --phase calibrate-subspace \
    --model qwen3-vl-8b-instruct --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --e5c-run-dir outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_plotqa/qwen3-vl-8b-instruct/<ts> \
    --predictions-path outputs/p0_1_calibration_qwen3vl/p0_1_calibration_qwen3vl_plotqa/qwen3-vl-8b-instruct/<ts>/predictions.jsonl \
    --config configs/p0_1_calibration_qwen3vl_plotqa.yaml \
    --dataset-tag plotqa --max-calibrate-pairs 5000
# (repeat for infovqa)
uv run python scripts/e6_compute_subspace.py --model qwen3-vl-8b-instruct \
    --scope plotqa_infovqa_pooled --tags plotqa,infovqa --K-max 16

# Phase C — bridge inference (~3 H200-hour total)
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml --models qwen3-vl-8b-instruct
uv run python scripts/run_gamma_beta_bridge.py \
    --config configs/p0_1_gamma_beta_bridge.yaml --models qwen3-vl-8b-thinking

# Phase D — aggregation (CPU)
uv run python scripts/build_gamma_beta_bridge_summary.py \
    --instruct-amp <instruct jsonl> \
    --thinking-amp <thinking jsonl> \
    --instruct-preds outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-instruct/20260428-114421/predictions.jsonl
```

Total wall-clock: ~10 H200-hour (vs spec ~26h estimate — Qwen3-VL inference faster than projected).

## Cross-references

- Paper sections affected: §1.5 / §4.6 / §8.2 / §8.4 in `docs/paper/emnlp_draft_ko.md`
- Subspace artifact (gitignored): `outputs/e6_steering/qwen3-vl-8b-instruct/_subspace/subspace_plotqa_infovqa_pooled_K16.pt`
- Per-trace amplitude (gitignored): `docs/insights/_data/gamma_beta_bridge_amplitude_per_trace.csv` (1091 rows)
- Singular value spectrum (gitignored): `docs/insights/_data/gamma_beta_bridge_qwen3vl_singular_values.csv` (36 layers × 16 sv)
- Spec: `docs/superpowers/specs/2026-05-10-p0-1-gamma-beta-bridge-design.md` (commit `e173618`)
- Source plan: `docs/insights/plan_post_review_2026-05-09.md` §P0-1
