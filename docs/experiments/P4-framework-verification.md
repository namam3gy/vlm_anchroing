# P4 — §5.4 framework verification (layer sweep + K=1 falsification on OneVision Main)

> **Closed 2026-05-12 AM.** Single-experiment follow-up to PR #26 (§5.4 framework
> relocation from §6.6) verifying §5.4 framework Predictions 2 and 3 directly on
> OneVision Main. Lands paper §6.2.4 (PR #39). Full evidence summary:
> [`docs/insights/P4-framework-verification-evidence.md`](../insights/P4-framework-verification-evidence.md).

## Goal

PR #26 moved the routing-vs-integration framework from §6.6 to §5.4 so that
§6.1 / §6.2 mitigations read as *predict-then-verify* of the framework's two
mitigation sites (P1). However, the framework's other two predictions:

- **P2 (single-direction failure, multi-direction success)** — only verified
  via §6.4 LEACE rank-1 ChartQA +56 % backfire on the 5-mech panel
  (cross-architecture, not OneVision Main).
- **P3 (late residual layer as anchor integration site)** — only "verified"
  by §6.2.3 chosen-cell L=26 K=8 performing well (single point, no
  layer-sweep or K-sweep within OneVision Main).

User correctly flagged that the framework was *positioned* better in paper but
not *data-grounded*. P4 runs the direct verification.

## Hypotheses

- **H-P3 (Late-layer specificity)** — On OneVision Main, K=8 α=1.0 subspace
  projection should reduce direction-follow rate (Δdf) only at *late* residual
  layers, with early layers null (no integrated anchor structure to remove) and
  very-late layers null again (answer token already decided, projection cannot
  redirect).
  - Predicted shape: null at L=5/10/15, sig negative at L=20-26 plateau,
    null again at L=27.
- **H-P2 (Single-direction failure on OneVision-internal)** — On OneVision Main,
  K=1 (rank-1 subspace) projection at L=26 α=1.0 should fail to reproduce the
  K=8 chosen-cell Δdf effect cross-dataset, and may *backfire* (sig positive Δdf)
  on at least one dataset where the per-dataset anchor direction misaligns with
  the calibrated K=1 direction. This mirrors §6.4's LEACE rank-1 finding but on
  OneVision-internal (not cross-architecture).

## Design

**Calibration target (no change)**: PlotQA + InfoVQA pooled wrong-base set
N=5,000 on `llava-onevision-qwen2-7b-ov`. Subspace `V_K[L]` precomputed via
SVD of `D[i, L, :] = h(x_i^a, L) − h(x_i^m, L)` and saved at all 28 layers ×
K_max=16. File: `outputs/e6_steering/llava-onevision-qwen2-7b-ov/_subspace/
subspace_plotqa_infovqa_pooled_n5k_K16.pt`. **Identical to §6.2.3 chosen-cell
calibration**.

**Cells run** (7 new cells per dataset):

| Cell | Layer | K | α | Purpose |
|---|---:|---:|---:|---|
| `L05_K08_a1.0` | 5 | 8 | 1.0 | P3 early null |
| `L10_K08_a1.0` | 10 | 8 | 1.0 | P3 early null |
| `L15_K08_a1.0` | 15 | 8 | 1.0 | P3 early null |
| `L20_K08_a1.0` | 20 | 8 | 1.0 | P3 mid integration onset |
| `L25_K08_a1.0` | 25 | 8 | 1.0 | P3 late plateau |
| `L27_K08_a1.0` | 27 | 8 | 1.0 | P3 "too late" |
| `L26_K01_a1.0` | 26 | 1 | 1.0 | P2 single-direction falsification |

L=26 K=8 α=1.0 is the §6.2.3 chosen cell — not re-run here.

**Evaluation datasets** (same as §6.2.3 Stage 4-final):

| Dataset | Wrong-base sids (cap) | Note |
|---|---:|---|
| TallyQA | 5,000 | floor-level baseline anchor pull (§6.2.3 -0.3 ns) |
| PlotQA | 2,306 | calibration dataset, highest power |
| InfoVQA | 443 | calibration dataset, small n |
| ChartQA | 224 | small n |
| MathVista | 170 | smallest n |

**Statistics**: Paired-bootstrap CI, B=10,000 resamples, sid-paired (per-arm num
and den re-computed each draw), seed=20260511.

## Pipeline

### Sweep launcher

`scripts/_p4_layer_sweep_K1_followup.sh` — bash driver, idempotent per-dir skip:

```bash
# For each dataset, two separate sweep calls:
#  - layers_K8 dir: --sweep-layers 5,10,15,20,25,27 --sweep-ks 8 --sweep-alphas 1.0
#  - L26_K1 dir:    --sweep-layers 26              --sweep-ks 1 --sweep-alphas 1.0
#
# Two separate calls avoid cartesian-product waste of `--sweep-ks 1,8` (which
# would run 14 cells per dataset).
#
# Order: small datasets first (mathvista → chartqa → infographicvqa → plotqa → tallyqa)
# K=1 cell before layer sweep within each dataset (K=1 finishes faster).
```

Each cell × dataset call writes `predictions.jsonl` to a separate output dir,
which the python script's `_load_completed_keys_subspace` reads on restart for
resume support.

### Inference per cell

`scripts/e6_steering_vector.py --phase sweep-subspace` iterates
`for cell in cells: for sid in eligible_sids: for cond in target_conds: ...`.
Each forward pass:

1. Build prompt + image inputs as per §6.2.3 chosen-cell path
2. Install `_install_projection_hook(layers, L, V_K[L, :K, :], alpha)` —
   forward post-hook on LLM decoder layer L that subtracts
   `alpha * V_K V_K^T h` from residual on prefill (`seq_len > 1`)
3. Run greedy decoding via `HFAttentionRunner.generate_number(...)`
4. Append record to `predictions.jsonl` with `cell_label`, `cell_layer`,
   `subspace_K`, `cell_alpha` keyed by sid + condition

Baseline cell (`cell_label="baseline"`, `cell_alpha=0`, `subspace_K=0`) runs
once per call (no hook), shared across all cells in the same call.

### Aggregation

`scripts/aggregate_e6_layer_sweep_p4.py` — reuses bootstrap helpers from
`scripts/build_e6_stage4_bootstrap_ci.py`:

1. Load predictions per (dataset, sweep_subtag) dir
2. Group by `cell_label`; for each non-baseline cell:
   - Find paired sids (b + a conditions present in both baseline and mit, all
     `parsed_number` parseable as finite float — defensive `math.isfinite` guard
     added 2026-05-12 in `_per_sid_indicators` to handle early-layer inf
     predictions)
   - Compute per-sid `(num, den)` indicators for metrics {adopt, df, em_a, em_b}
   - Bootstrap-resample sids (B=10,000) and recompute per-arm rates each draw,
     accumulate Δ = mit_rate − base_rate
3. Output per (dataset, cell) row with point estimate + 95 % CI
4. Generate 5-panel layer-sweep figure (one subplot per dataset; x=layer,
   y=Δdf with K=8 line shaded by CI, K=1 marker at L=26)

## Results

### P3 verification — Δdf per layer (K=8 α=1.0)

| Layer | TallyQA n=4,978 | PlotQA n=2,306 | InfoVQA n=443 | ChartQA n=224 | MathVista n=170 |
|---|---:|---:|---:|---:|---:|
| L=5 | +0.4 [-3.8, +5.1] *(n=235)* | -1.6 [-4.5, +1.3] *(n=751)* | 0.0 [-5.2, +5.2] *(n=307)* | +2.7 [-4.8, +10.9] *(n=147)* | +7.1 [-5.4, +19.6] *(n=56)* |
| L=10 | -0.2 [-1.0, +0.6] | -0.7 [-2.2, +0.8] | -2.5 [-5.9, +0.9] | -3.1 [-7.6, +0.9] | +3.5 [-1.8, +8.8] |
| L=15 | +0.3 [-0.6, +1.2] | +1.2 [-0.4, +2.7] | +1.8 [-1.4, +5.0] | -0.5 [-5.4, +4.5] | +2.9 [-4.1, +10.0] |
| **L=20** | **-1.1 [-2.0, -0.2]** | **-4.7 [-6.4, -3.0]** | -0.5 [-4.1, +3.2] | -3.6 [-8.9, +1.8] | **-7.7 [-14.1, -0.6]** |
| L=25 | -0.5 [-1.5, +0.5] | **-3.0 [-4.8, -1.2]** | +0.5 [-3.6, +4.3] | -4.9 [-10.7, +0.5] | -2.4 [-9.4, +4.7] |
| L=26 *(§6.2.3 ref, eager)* | -0.3 [-1.3, +0.6] | **-5.2 [-6.9, -3.4]** | -0.7 [-4.7, +3.4] | -4.0 [-9.8, +1.8] | -4.1 [-11.8, +3.5] |
| L=27 | +0.4 [-0.6, +1.4] | -0.7 [-2.2, +0.8] | +0.9 [-2.7, +4.5] | 0.0 [-5.4, +5.4] | -0.5 [-7.6, +6.6] |

→ **PlotQA**: clean L=5/10/15 null → L=20-26 sig plateau → L=27 ns. Verifies
P3 in full including "very late null" (decision-immediate-not-too-late) framing.
→ **TallyQA**: L=10/15/25/27 null, only L=20 sig. Single-peak rather than
plateau because TallyQA baseline anchor pull is floor-level (§6.2.3 chosen-cell
-0.3 ns) — small headroom for mitigation.
→ **MathVista**: L=20 sig despite n=170. Wide CI but point estimate strong.
→ **InfoVQA / ChartQA**: small n, individual cells ns. Trend direction supports
P3 but underpowered.

### P2 verification — K=1 vs K=8 at L=26

| Dataset | K=1 Δdf (this work, n_paired) | K=8 chosen Δdf (§6.2.3 ref) | Gap |
|---|---:|---:|---:|
| **TallyQA** | **+1.4 [+0.5, +2.2]** sig BACKFIRE *(n=4,975)* | -0.3 [-1.3, +0.6] ns | +1.7 pp sign flip |
| PlotQA | -0.4 [-1.7, +0.9] ns *(n=2,306)* | **-5.2 [-6.9, -3.4]** sig | -4.85 pp (K=1 fails) |
| InfoVQA | -1.4 [-4.5, +1.8] ns *(n=443)* | -0.7 [-4.7, +3.4] ns | both ns |
| ChartQA | -2.7 [-6.7, +1.3] ns *(n=224)* | -4.0 [-9.8, +1.8] ns | both ns |
| MathVista | +5.3 [-2.4, +12.9] ns *(n=170)* | -4.1 [-11.8, +3.5] ns | sign-flip (ns) |

→ **TallyQA K=1 +1.4 pp sig BACKFIRE** mirrors §6.4's LEACE rank-1 ChartQA +56 %
reversal *on OneVision-internal*. Single-direction projection actively amplifies
anchor pull on at least one dataset.
→ **PlotQA gap 4.85 pp ≈ 4.4σ** between K=1 ns and K=8 sig. Robust to
methodology drift (~1 pp upper bound, 5× smaller than gap).

### Δem(b) all-layer positive (unexpected)

Every cell on every dataset shows sig positive Δem(b) (target-only arm em rise),
including L=5/L=10 K=8 cells where Δdf is null:

| Layer (PlotQA) | Δem(b) [95 % CI] |
|---|---:|
| L=5 K=8 | +2.3 [+1.2, +3.5] sig |
| L=10 K=8 | +1.2 [+0.7, +1.7] sig |
| L=15 K=8 | +1.9 [+1.3, +2.5] sig |
| L=20 K=8 | +1.2 [+0.6, +1.7] sig |
| L=25 K=8 | +3.6 [+2.8, +4.5] sig |
| L=27 K=8 | +3.1 [+2.3, +3.9] sig |
| L=26 K=1 | +1.1 [+0.7, +1.6] sig |

On TallyQA the b-arm em gain is even larger at every layer (L=25 K=8 = +17.8 pp).

→ K=8 subspace projection does *two things*: anchor pull reduction (late-layer
specific, Δdf) + generic regularization (all-layer, Δem(b)). §6.3 Insight 1.5
Alt-1 hypothesis (general regularization) is **reaffirmed but unfalsified**. The
random-K=8 baseline (Phase 5 P1-5, deferred) remains the definitive falsifier.

### L=5 inference destabilization (negative-direction P3 evidence)

| Dataset | L=5 K=8 paired n | Baseline n | Drop |
|---|---:|---:|---:|
| PlotQA | 751 | 2,306 | 67 % |
| TallyQA | 235 | 4,978 | 95 % |
| MathVista | 56 | 170 | 67 % |
| ChartQA | 147 | 224 | 34 % |
| InfoVQA | 307 | 443 | 31 % |

→ Early-layer K=8 projection produces inf / non-numeric predictions on a large
fraction of samples → filtered out of paired analysis. Self-consistent negative-
direction P3 evidence: early layer has no K-dim anchor structure to remove +
projection actively *disrupts* forward computation. The drop heals at L=10+
(paired n recovers to ~100 % of baseline).

## Caveats

1. **Methodology drift (eager → SDPA)**. Commit `5c2f52b` (2026-05-03) switched
   `e6_steering_vector.py` from eager to SDPA. §6.2.3 chosen-cell (2026-05-02
   run) used eager; P4 sweep (2026-05-11 run) uses SDPA. Cross-validation: 15/684
   = 2.19 % baseline pairs differ on mathvista. Boundary-sample bf16 precision
   flips; not systematic. **P4 sweep is internally self-consistent**; §6.2.3
   numbers are cited as separate reference, not mixed in P4 figure. Drift impact
   bounded at ~1 pp upper, 5× smaller than the P2 gap (4.85 pp).

2. **Δem(b) all-layer Alt-1 hypothesis unfalsified**. Doesn't affect §6.2.3
   deployability claim. Affects §6.3 mechanism attribution. Random-K=8 baseline
   needed for resolution (deferred).

3. **Power asymmetry**. PlotQA n=2,306 + TallyQA n=4,978 carry verification load.
   Smaller datasets (InfoVQA 443, ChartQA 224, MathVista 170) contribute trend
   support but individual cells often ns.

4. **L=26 K=8 not re-run on SDPA**. Optional follow-up (~3-4 H100-hour) for
   byte-clean within-figure comparison. Not load-bearing.

## What this doesn't say

- **The Δem(b) gain is non-anchor-specific.** It might still be anchor-specific
  (P4 doesn't rule out anchor-mechanism interpretation, just doesn't *confirm*
  it). Random-K=8 baseline (deferred) is what would resolve this.
- **L=20 is the universally-best layer for E6 mitigation.** TallyQA L=20
  -1.1 pp sig is interesting (vs §6.2.3 chosen-cell L=26 -0.3 ns on TallyQA),
  but this is *one dataset* and we deliberately do not re-tune per-dataset (would
  be cherry-picking the eval data). The §6.2.3 ex-ante calibration rule still
  governs the deployable mitigation; P4 just verifies the *framework prediction*,
  not optimal cell selection.
- **Cross-architecture generalises.** P4 is OneVision-internal. Phase 5 P2-7
  (Qwen2.5-VL-7B replication) is the cross-architecture follow-up.

## Pipeline notes / lessons

1. **Cartesian-product gotcha** in `--sweep-ks 1,8 --sweep-layers 5,...,27` →
   14 cells per dataset (we wanted 7). Two separate sweep calls (`layers_K8` and
   `L26_K1`) avoided 2× GPU cost. Each call shares its own baseline computation
   (~5k records overhead per call).
2. **Pod-death idempotency**. Python `_load_completed_keys_subspace` reads
   completed (sid, condition, layer, K, alpha) tuples from existing
   predictions.jsonl on restart. Pod died twice during the run; both times
   resumed cleanly. The bash launcher's `exists_or_force` skip-on-existing logic
   needed bypass for the in-progress dirs — handled by calling
   `e6_steering_vector.py` directly with the same `--output-dir` for the resume.
3. **`math.isfinite` defensive guard** in `_per_sid_indicators`. Early-layer
   (L=5) K=8 projection occasionally produces inf predictions; the bootstrap
   helper called `int(float('inf'))` and overflowed. Added before the `int(pa)
   == gt_a` / `int(pb) == gt_b` checks.
4. **Methodology drift discovery**. Advisor-flagged check (compare new P4
   baseline to §6.2.3 chosen-cell baseline on shared mathvista sids) caught the
   eager → SDPA transition. Worth running this kind of cross-check whenever a
   long-running pipeline is touched between phases.

## Repro

```bash
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

# 1. Verify subspace tensor exists (already built by §6.2.2 calibration step)
ls outputs/e6_steering/llava-onevision-qwen2-7b-ov/_subspace/subspace_plotqa_infovqa_pooled_n5k_K16.pt

# 2. Launch sweep (sequential across 5 datasets, ~12.5 H200-hour wall)
nohup bash scripts/_p4_layer_sweep_K1_followup.sh > logs/p4_layer_sweep_K1/launcher.out 2>&1 &
disown

# 3. Aggregate (CPU-only, ~1 minute)
uv run python scripts/aggregate_e6_layer_sweep_p4.py

# Outputs:
#   docs/insights/_data/p4_layer_sweep_per_cell_ci.csv   (per-cell × per-dataset rows)
#   docs/insights/_data/p4_layer_sweep_bootstrap_draws.npz (raw bootstrap draws)
#   docs/insights/_data/p4_layer_sweep_summary.md        (human-readable summary)
#   docs/figures/p4_layer_sweep_delta_df.png             (5-panel layer-sweep figure)
```

For mid-run resume (if pod death or interruption), call
`scripts/e6_steering_vector.py --phase sweep-subspace` directly with the same
`--output-dir` pointing at the existing partial `predictions.jsonl` — resume is
automatic via completed-key cache.
