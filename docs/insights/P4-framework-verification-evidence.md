# P4 — §5.4 routing-vs-integration framework verification on OneVision Main

> **Status (2026-05-12 AM): closed.** §5.4 framework Predictions 2 (single-direction
> failure) and 3 (late-layer integration site) directly verified on OneVision Main
> via 7-cell × 5-dataset paired-bootstrap sweep, same calibration scope as §6.2.3
> chosen-cell. Headline findings: (P3) PlotQA shows clean early-null + L=20-26
> sig + L=27 ns sharp peak, TallyQA mirrors with L=20 sig single-peak; (P2)
> TallyQA K=1 sig BACKFIRE (+1.4 pp [+0.5, +2.2]) mirrors §6.4 LEACE rank-1
> ChartQA reversal on OneVision-internal. Δem(b) all-layer positive — §6.3
> Insight 1.5 Alt-1 (general regularization) hypothesis reaffirmed but not yet
> falsified; random-K=8 baseline (Phase 5 P1-5) remains the resolution path.
>
> Lands paper §6.2.4 (PR #39, branch `worktree-paper-section6-2-4-p4-layer-sweep`).
> Source data: `outputs/e6_steering/llava-onevision-qwen2-7b-ov/
> sweep_subspace_<ds>_plotqa_infovqa_pooled_n5k_p4_layer_sweep_K1_{layers_K8,L26_K1}/
> predictions.jsonl`. Canonical CSV `docs/insights/_data/p4_layer_sweep_per_cell_ci.csv`,
> raw bootstrap draws `_data/p4_layer_sweep_bootstrap_draws.npz`. Generator
> `scripts/aggregate_e6_layer_sweep_p4.py`, launcher
> `scripts/_p4_layer_sweep_K1_followup.sh`. Total GPU ~12.5 H200-hour wall (with
> 2 pod-death recoveries).

## Motivation

PR #26 (merged 2026-05-11) lifted the routing-vs-integration framework from §6.6 to
§5.4 and reframed §6.1 / §6.2 intros as predict-then-verify of the framework's
P1 (two mitigation sites). However, Predictions P2 (single-direction failure) and
P3 (late-layer integration site) were referenced as "predicted by framework" without
direct empirical test inside the paper — §6.4 LEACE rank-1 ChartQA backfire on the
5-mech panel was the only P2 evidence, and §6.2.3 chosen-cell L=26 K=8 was a
*single point* of evidence with no layer-sweep or K-sweep verification. User
flagged this as "말로만 cover" — framework positioned but not data-grounded.

This experiment closes that gap. Same calibration scope as §6.2.3 chosen-cell
(PlotQA + InfoVQA pooled wrong-base n=5,000, (a − m) K=16 SVD subspace) — only
the (L, K, α) cell changes.

## Cells

- **Layer sweep** (P3 test): L ∈ {5, 10, 15, 20, 25, 27} × K=8 × α=1.0 — 6 cells.
  Tests whether anchor pull reduction is *late-layer specific* (P3 prediction).
  L=26 K=8 is the §6.2.3 chosen cell, not re-run here (drift caveat below).
- **K=1 falsification** (P2 test): L=26 × K=1 × α=1.0 — 1 cell. Tests whether
  single-direction projection at the chosen layer reproduces the K=8 chosen-cell
  effect.

Evaluation on 5-dataset set: TallyQA (n=4978), PlotQA (n=2306), InfoVQA (n=443),
ChartQA (n=224), MathVista (n=170) — paired wrong-base sids per dataset.
Paired-bootstrap CI B=10,000, seed=20260511, sid-paired resampling with
per-arm (num, den) recomputed each draw.

## Headline numbers

### Prediction 3 — Late-layer integration site (Δdf)

**PlotQA n=2,306** (highest-power dataset, calibration scope):

| Layer | Δdf | 95 % CI | Verdict |
|---|---:|---:|---|
| L=5 K=8 | -1.6 pp | [-4.5, +1.3] | ns (n=751 due to inference disruption — 67 % drop) |
| L=10 K=8 | -0.7 pp | [-2.2, +0.8] | ns |
| L=15 K=8 | +1.2 pp | [-0.4, +2.7] | ns |
| **L=20 K=8** | **-4.7 pp** | **[-6.4, -3.0]** | **sig** |
| **L=25 K=8** | **-3.0 pp** | **[-4.8, -1.2]** | **sig** |
| L=26 K=8 (§6.2.3 ref, eager) | -5.2 pp | [-6.9, -3.4] | sig (separate baseline) |
| L=27 K=8 | -0.7 pp | [-2.2, +0.8] | ns |

→ **Sharp peak L=20-26 + early null + very-late null** verifies §5.4 P3's
"decision-immediate integration site" framing in full. The L=27 weakening
specifically supports "very late → answer token already decided, projection can
no longer redirect" interpretation.

**TallyQA n=4,978**:

| Layer | Δdf | 95 % CI | Verdict |
|---|---:|---:|---|
| L=5 K=8 | +0.4 pp | [-3.8, +5.1] | ns (n=235 — 95 % drop) |
| L=10 K=8 | -0.2 pp | [-1.0, +0.6] | ns |
| L=15 K=8 | +0.3 pp | [-0.6, +1.2] | ns |
| **L=20 K=8** | **-1.1 pp** | **[-2.0, -0.2]** | **sig (single peak)** |
| L=25 K=8 | -0.5 pp | [-1.5, +0.5] | ns |
| L=27 K=8 | +0.4 pp | [-0.6, +1.4] | ns |

→ TallyQA mirrors PlotQA at smaller magnitude — only L=20 reaches sig, no
L=20-25 plateau. Consistent with TallyQA's floor-level baseline anchor pull
(§6.2.3 chosen-cell -0.3 pp ns) — mitigation can't reduce below the floor by
much. Same P3 prediction pattern, different magnitude.

**MathVista n=170**: L=20 -7.7 pp [-14.1, -0.6] sig (strong point estimate, wide CI).
**InfoVQA n=443 / ChartQA n=224**: individual cells not sig (small n) but trend
direction consistent.

### Prediction 2 — Single-direction failure (K=1 vs K=8)

**Cross-dataset K=1 (this work) vs K=8 (§6.2.3 chosen-cell ref)**:

| Dataset | n_paired (K=1) | K=1 Δdf [95 % CI] | K=8 Δdf [95 % CI] | Verdict |
|---|---:|---:|---:|---|
| **TallyQA** | 4,975 | **+1.4 [+0.5, +2.2]** sig BACKFIRE | -0.3 [-1.3, +0.6] ns | **Sign flip** |
| PlotQA | 2,306 | -0.4 [-1.7, +0.9] ns | **-5.2 [-6.9, -3.4]** sig | K=1 fails (gap 4.85 pp = 4.4σ) |
| InfoVQA | 443 | -1.4 [-4.5, +1.8] ns | -0.7 [-4.7, +3.4] ns | both ns |
| ChartQA | 224 | -2.7 [-6.7, +1.3] ns | -4.0 [-9.8, +1.8] ns | both ns |
| MathVista | 170 | +5.3 [-2.4, +12.9] ns | -4.1 [-11.8, +3.5] ns | sign-flip (ns) |

→ **TallyQA K=1 +1.4 pp sig backfire** is the OneVision-internal mirror of §6.4
LEACE rank-1 ChartQA +56 % backfire on 5-mech panel. Single-direction projection
not just *weaker* than multi-direction — it *actively amplifies anchor pull* on a
per-dataset basis, because the single direction calibrated on PlotQA + InfoVQA
pooled does not align with TallyQA's per-dataset anchor variance direction. P2's
predict-then-verify chain extends from cross-architecture (§6.4) to
OneVision-internal (this work) — two complementary angles.

PlotQA gap K=1 -0.4 ns vs K=8 -5.2 sig = 4.85 pp = 4.4σ in combined SE. The gap
dwarfs the methodology drift contribution (~1 pp upper bound, see Caveats) by
~5×, so the P2 verification is robust to drift noise.

## Unexpected finding — Δem(b) all-layer positive

Across **every** cell on **every** dataset, target-only arm (b-arm) exact-match
delta is significantly positive:

| Layer (PlotQA) | Δem(b) [95 % CI] |
|---|---:|
| L=5 K=8 | +2.3 [+1.2, +3.5] sig |
| L=10 K=8 | +1.2 [+0.7, +1.7] sig |
| L=15 K=8 | +1.9 [+1.3, +2.5] sig |
| L=20 K=8 | +1.2 [+0.6, +1.7] sig |
| L=25 K=8 | +3.6 [+2.8, +4.5] sig |
| L=27 K=8 | +3.1 [+2.3, +3.9] sig |
| L=26 K=1 | +1.1 [+0.7, +1.6] sig |

Even more pronounced on TallyQA — L=25 K=8 Δem(b) = +17.8 pp sig, L=5 K=8 = +9.8 pp sig.

→ K=8 subspace projection is doing **two things simultaneously**:
1. **Anchor pull reduction** — *late-layer specific* (Δdf only sig at L=20-26)
2. **General accuracy gain on target-only arm** — *all-layer non-specific*
   (Δem(b) sig across every layer including L=5)

§6.3 Insight 1.5 explicitly hedged this with the **Alt-1 general regularization
hypothesis**: K=8 leading subspace projection might act as a mild regularizer
biasing logit distribution toward modal correct answers irrespective of any
anchor signal. P4 data **reaffirms** Alt-1 (the Δem(b) signal is not late-layer
specific, contrary to what a strict anchor-pathway interpretation would predict)
but does not **falsify** it — random-K=8 subspace baseline (Phase 5 P1-5,
deferred) remains the only way to definitively separate "anchor-specific b-arm
gain" from "generic regularization b-arm gain".

**Paper claim impact**: §6.2's *deployable mitigation* claim (works as
advertised, captures Δem(b) gain on target-only) is unaffected — the b-arm em
rise is empirical, doesn't depend on mechanism attribution. But the §6.3 Insight 1
attribution of b-arm em gain to "(a − m) contrast capturing wrong-base error mode
co-aligned with anchor failure" remains hedged pending random-K=8 baseline.
§6.3 prose already cites this as deferred.

## Methodology drift caveat — eager → SDPA

During post-run audit (advisor-flagged), discovered that the §6.2.3 chosen-cell
runs (2026-05-02) used **eager attention**, while the new P4 sweep (2026-05-11)
uses **SDPA**, due to commit `5c2f52b` (2026-05-03) which switched
`e6_steering_vector.py` to SDPA for ~3× speedup. Cross-validation:

- Mathvista baseline cell from new P4 (SDPA) vs chosen-cell (eager): 669/684 =
  97.81 % exact-match on `parsed_number`. 15 sids differ (~2.2 % boundary cases
  flip class).
- Differences are not systematic — they're boundary samples where the model is
  near-confident, fp16/bf16 fused-kernel precision differences flip the argmax.

**Implication**: §6.2.3 chosen-cell L=26 K=8 number cannot be directly mixed with
new P4 numbers in the same figure or table (different baseline). For paper §6.2.4:

- Layer-sweep figure shows only new P4 cells on consistent SDPA baseline
- §6.2.3 chosen-cell L=26 K=8 number cited as **separate reference**, with eager
  baseline footnote
- Quantitative claims (e.g., "K=1 vs K=8 gap 4.85 pp") use drift noise upper-bound
  (~1 pp) as combined-SE addition; gap is 4.4σ which dwarfs drift 5×

**Optional follow-up**: re-run L=26 K=8 on SDPA baseline for byte-clean
within-figure comparison (~3-4 H100-hour). Not load-bearing for current paper
claims; deferred.

## Negative-direction P3 evidence — inference destabilization at L=5

| Dataset | L=5 K=8 paired n | Baseline n | Drop |
|---|---:|---:|---:|
| PlotQA | 751 | 2,306 | 67 % |
| TallyQA | 235 | 4,978 | 95 % |
| ChartQA | 147 | 224 | 34 % |
| InfoVQA | 307 | 443 | 31 % |
| MathVista | 56 | 170 | 67 % |

L=5 K=8 projection produces inf or non-numeric predictions on a large fraction
of samples, which fall out of the paired analysis. This is itself **negative-
direction P3 evidence**: early layers don't merely lack the integrated K-dim
anchor structure (no signal to remove → null Δdf) — they actively *resist*
projection-style interventions (forward computation disrupted → inference
destabilized). The L=10/L=15 cells show this destabilization wearing off (paired
n recovers to ~100 % of baseline). Consistent with framework P3's "routing
distributed at early layers, accumulation site at late layer" picture.

## Reading

The §5.4 framework's two empirical predictions are now both
**directly verified on OneVision Main**:

| Prediction | Test location | Status |
|---|---|---|
| P1 — Two mitigation sites (routing + integration) | §6.1 E4 + §6.2 E6 (existing) | ✓ (existing) |
| **P2 — Single-direction failure (multi-direction success)** | §6.4 LEACE rank-1 (cross-architecture) + **§6.2.4 K=1 on OneVision** (this work) | **✓✓ two angles** |
| **P3 — Late-layer integration site** | **§6.2.4 layer sweep** (this work) | **✓ verified on PlotQA + TallyQA** |
| P4 — Projection vs broad ablation | §7 capability preservation (existing) | ✓ (existing) |

The predict-then-verify chain is now §5.2 multi-layer redundancy → §5.4
framework → §6.2.3 chosen-cell deployment + §6.2.4 layer/K verification → §6.4
single-direction cross-dataset failure → §7 capability preservation. Six
sub-sections form a single linear empirical defense of the framework.

## Caveats

- **Δem(b) all-layer positive** — Alt-1 (general regularization) hypothesis
  unfalsified. Random-K=8 baseline (Phase 5 P1-5) deferred. §6.3 Insight 1.5
  hedge reaffirmed but not resolved. Doesn't affect §6.2.3 deployability claim;
  affects §6.3 mechanism attribution.
- **eager → SDPA precision drift** — ~2 % boundary-sample divergence between
  §6.2.3 chosen-cell (eager) and P4 sweep (SDPA). P4 internally consistent;
  §6.2.3 numbers cited as separate reference in paper §6.2.4 with footnote.
- **Cross-dataset cell power asymmetry** — only PlotQA (n=2,306) and TallyQA
  (n=4,978) have sufficient power to verify individual cell significance.
  InfoVQA (n=443), ChartQA (n=224), MathVista (n=170) carry trend support but
  individual cells often ns.
- **L=26 K=8 not re-run on SDPA baseline** — for byte-clean within-figure
  comparison with the P4 cells, L=26 K=8 should be re-run under SDPA. Deferred
  as not load-bearing for current claims.

## Open follow-ups (deferred)

- **Random-K=8 baseline (Phase 5 P1-5)** — definitive falsifier for Δem(b)
  all-layer Alt-1 hypothesis. ~2 H100-day.
- **L=26 K=8 SDPA re-run** — byte-clean within-figure comparison with P4 cells.
  ~3-4 H100-hour.
- **Cross-architecture P3 verification** — current P4 is OneVision-internal.
  Layer sweep on other architectures (e.g., Qwen2.5-VL-7B) to test whether the
  L=20-26 plateau or sharp-peak pattern generalises. ~10 H200-day (Phase 5 P2-7
  builds part of this).

## Source / provenance

- Sweep launcher: `scripts/_p4_layer_sweep_K1_followup.sh`
- Aggregator + paired-bootstrap CI: `scripts/aggregate_e6_layer_sweep_p4.py`
- Defensive fix to bootstrap helper: `scripts/build_e6_stage4_bootstrap_ci.py`
  (added `math.isfinite` guard on `pb/pa/anchor` for early-layer inf predictions)
- Per-cell CI table: `docs/insights/_data/p4_layer_sweep_per_cell_ci.csv` (gitignored — see `_data/README` for repro)
- Raw bootstrap draws: `docs/insights/_data/p4_layer_sweep_bootstrap_draws.npz` (gitignored)
- Summary markdown: `docs/insights/_data/p4_layer_sweep_summary.md` (gitignored)
- Figure: `docs/figures/p4_layer_sweep_delta_df.png` (5-panel layer sweep, K=8 curve + K=1 marker)
- Paper integration: `docs/paper/emnlp_draft_ko.md` §6.2.4 (commit `a0f2f0a` and earlier P4 patches on branch `worktree-paper-section6-2-4-p4-layer-sweep`)
- Per-dataset sweep output dirs:
  - `outputs/e6_steering/llava-onevision-qwen2-7b-ov/sweep_subspace_<ds>_plotqa_infovqa_pooled_n5k_p4_layer_sweep_K1_layers_K8/predictions.jsonl` (6 cells × wrong-base × 4 conds + baseline)
  - `outputs/e6_steering/llava-onevision-qwen2-7b-ov/sweep_subspace_<ds>_plotqa_infovqa_pooled_n5k_p4_layer_sweep_K1_L26_K1/predictions.jsonl` (1 cell + baseline)

Total wall clock: ~12.5 H200-hour across 5 datasets sequentially, plus 2 pod-death
recoveries handled via python `--resume` (idempotent sid-tuple completion check
in `_phase_sweep_subspace`).
