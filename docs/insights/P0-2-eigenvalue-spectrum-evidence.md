# P0-2 — Per-layer eigenvalue spectrum of the (a − m) anchor calibration matrix

**Status (2026-05-09).** Both pre-registered acceptance criteria FAIL in the predicted direction. Outcome corresponds to the plan's "(a only / b only / neither) graceful degradation" branch — the paper's *theoretical* contribution framing softens from "spectrum-predicted dimensionality" to "consistent with multi-layer redundancy + high-rank residual integration" with the spectrum reported honestly as a transparency item.

**Source.** `scripts/analyze_e6_eigenvalue_spectrum.py` on pooled wrong-base D_wrong from PlotQA + InfoVQA calibrations of OneVision Main (`outputs/e6_steering/llava-onevision-qwen2-7b-ov/calibration_{plotqa,infographicvqa}/D_wrong.pt`). N_total = 2,757 wrong-base records (PlotQA n=2,314 + InfoVQA n=443; the "n5k" label in upstream calibration dirs is the *cap* on wrong-base sweeps, not the actual count). Per-layer thin SVD on D[:, L, :] with L ∈ {0, ..., 27} for OneVision Qwen2-7B LM. CPU-only; ~60 s wall-clock.

**Pre-registered acceptance criteria** (locked in `scripts/analyze_e6_eigenvalue_spectrum.py` before any data inspection beyond the existing top-16 SV CSV peek; both thresholds chosen for falsifiability rather than ease):

| Criterion | Test | Threshold |
|---|---|---|
| (a) Rank-K elbow at L=26 | `sv_7 / sv_8 ≥ 1.5` OR `explained_var_K8 ≥ 0.70` | sharp localized gap or ≥70% top-8 variance |
| (b) Effective rank decreases monotonically across L=10 → final | Shannon effective rank trajectory | each step `eff_rank[L+1] ≤ eff_rank[L]` |

**Pre-registered measures.** All scale-invariant, since residual-stream norms grow ~50× with depth in OneVision and would otherwise confound "norm grew" with "integration":
- **Shannon effective rank** (Roy & Vetterli 2007): `exp(H(p))` where `p_i = σ_i / Σ σ_j` — primary measure.
- **Participation ratio**: `(Σ σ²)² / Σ σ⁴`.
- **Stable rank**: `Σ σ² / σ_0²`.
- **Explained variance at K=8**: `Σ_{i<8} σ_i² / Σ σ_i²`.
- **sv_7/sv_8 ratio**: local elbow probe (pre-registered threshold ≥ 1.5).

---

## Headline numbers

| Layer | sv_0 | sv_7 | sv_8 | sv_7/sv_8 | EV@K8 | eff_rank | part_R | stable_R |
|---|---|---|---|---|---|---|---|---|
| 0 | 7.70 | 1.83 | 1.80 | 1.017 | 0.6970 | 843.3 | 6.75 | 3.10 |
| 10 | 29.41 | 8.92 | 7.97 | 1.120 | 0.4977 | 1398.8 | 17.66 | 5.15 |
| 14 | 62.23 | 14.89 | 14.44 | 1.031 | 0.4996 | 1445.0 | 12.03 | 3.73 |
| 18 | 75.64 | 22.08 | 21.20 | 1.042 | 0.3882 | 1582.0 | 24.50 | 5.72 |
| 22 | 169.80 | 61.43 | 57.00 | 1.078 | 0.2842 | 1646.6 | 53.19 | 9.89 |
| **26** | **319.51** | **158.00** | **155.07** | **1.019** | **0.2126** | **1712.6** | **105.83** | **17.79** |
| 27 | 396.78 | 172.12 | 163.01 | 1.056 | 0.3840 | 1498.4 | 35.97 | 9.07 |

Full per-layer table at `docs/insights/_data/p0_2_per_layer_spectrum.{csv,md}`. Top-50 σ_k per layer cached at `docs/insights/_data/p0_2_full_svs_top.pt` for figure reuse.

---

## Verdict

### Criterion (a) — rank-K elbow at L=26 — FAIL

| Metric | Value at L=26 | Threshold | Pass? |
|---|---|---|---|
| sv_7 / sv_8 | **1.019** | ≥ 1.5 | ❌ |
| explained_var_K8 | **0.2126** | ≥ 0.70 | ❌ |

Across all 28 layers, `sv_7 / sv_8 ∈ [1.016, 1.120]` — comparable to `sv_8 / sv_9 ∈ [1.004, 1.126]`. There is **no rank-8-localized gap** in the spectrum at any layer. The decay is smooth around K=8.

**Implication.** The chosen E6 cell **K=8 is selected by the em-deal-breaker rule on the (Δdf, Δem(a), Δem(b)) Pareto front**, not by a spectrum-predicted optimum. §6.4 Insight 2 ("K=8 sweet spot") **stays empirical** — must not be promoted to "spectrum-predicted dimensionality."

### Criterion (b) — effective rank decreases L=10 → final — FAIL (in fact, INVERTED)

Shannon effective rank trajectory: **monotonic INCREASE** L=10 → L=26 (1399 → 1713), then sharp drop at L=27 (1713 → 1498) where the LM head compresses for next-token prediction. All three scale-invariant rank measures agree on direction:

| Measure | L=10 | L=26 | L=27 | direction L=10→26 |
|---|---|---|---|---|
| Shannon effective rank | 1399 | **1713 (max)** | 1498 | ↑ +22% |
| Participation ratio | 17.7 | **105.8 (max)** | 36.0 | ↑ +498% |
| Stable rank | 5.15 | **17.79 (max)** | 9.07 | ↑ +245% |
| Explained variance @ K=8 | 0.498 | **0.213 (min)** | 0.384 | ↓ −57% |

(Lower EV@K8 = top-8 captures less of total variance = anchor variance is **more** distributed, not more concentrated.)

**Implication.** The "routing-vs-integration framework's empirical signature" predicted in plan §P0-2.b — "anchor variance redistributes from broad rank at early layers to compact low-dim subspace at late layers" — is **not what the data shows**. Late layers are the **maximum-dispersion** site for anchor variance, not the minimum-dispersion site. L=26 is where the (a − m) difference is *most* spread across the residual stream's directions, and L=27 begins to compress for LM-head logit projection.

This does **not** invalidate L=26 as the chosen E6 hook site; it does require the §5.2 Insight 4 + §6.6 framework rephrasing to drop "compact low-dim subspace" language and lead with what actually holds:

1. **L=26 is integration-complete-but-pre-final**: late enough for anchor information to have been routed in from attention pathways into the residual stream (consistent with §5.2 multi-layer-redundancy attention-ablation null), but pre-final so logit-head compression hasn't happened yet (L=27 spectrum re-compacts).

2. **High residual-stream effective rank at L=26 is consistent with single-direction interventions under-performing K=8.** The paper's K=8 cell already captures only ~21 % of total (a − m) Frobenius variance and yet achieves free-lunch — this is itself evidence that **explained variance fraction ≠ causal sufficiency**. The spectrum therefore does *not* prove that K=1 cannot work; it predicts that K=1 will under-perform K=8 by a margin we can only measure empirically. P1-4 (CAA at K=1 + ITI at attention-head) is the empirical test. Soft prediction for §6.5 Table 7: CAA K=1 fails free-lunch on ≥ 1 / 5 datasets, consistent with the §6.5 structural-reduction Note. Strong claim ("cannot capture all anchor variance") is **not** supported by this spectrum alone.

3. **K=8 captures only 21% of L=26 anchor variance.** The remaining 79% is distributed across ranks 9..2,757. This is consistent with the §6.2.3 "free-lunch" finding (Δem(a) ≥ 0): the K=8 projection removes a small but well-chosen slice of anchor variance without disturbing the bulk of the residual stream.

---

## Paper updates required

### §6.4 Insight 2 ("K=8 sweet spot")

- **Drop**: any phrasing suggesting K=8 is a spectrum-predicted dimensionality, "rank-8 elbow," or "intrinsic anchor dimensionality."
- **Keep**: K=8 is the (Δdf, Δem(a), Δem(b)) Pareto-optimal cell under the em-deal-breaker rule on the 27-cell pilot grid (§A.5 + P1-6 deliverable).
- **Add**: Spectrum measurement (this evidence doc) reports `sv_7/sv_8 = 1.019` at L=26 and EV@K8 = 0.213. The K=8 cell removes a small slice of anchor variance precisely *because* the residual stream encodes anchor information at high effective rank — the free-lunch property is not sensitive to rank choice in the way a low-rank-elbow theory would predict.

### §5.2 Insight 4 + §6.6 routing-vs-integration framework

- **Drop**: "compact low-dim subspace at late layers", "anchor variance concentrates at L=26", "K=8 captures the integration site's principal directions."
- **Keep**: Categorical "residual ≠ attention pathway" distinction (residual stream is where multi-layer-redundant attention pathway accumulates).
- **Replace late-layer prediction with measurement**: L=26 carries **maximum** anchor-variance dispersion (Shannon effective rank 1713; EV@K8 = 0.213). The integration interpretation now reads: late residual stream is where anchor information accumulates **as a high-rank manifold**, not as a low-dim direction. Single-direction interventions (CAA K=1, ITI) are predicted to under-perform K=8 subspace projection by the residual's effective-rank gap. (P1-4 will test this empirically.)

### §1.5 (4a) framing

The "predict-then-verify chain" for the *이론적* contribution must restate the prediction:

- **Old (now falsified) prediction**: "Multi-layer redundancy at attention layers + low-rank concentration at late residual = K=8 subspace at L=26 captures the integration site."
- **New empirical-grounded chain**: "Multi-layer redundancy in attention pathway (§5.2 ablation null) + high-rank residual encoding at L=26 (P0-2 spectrum) ⇒ single-direction interventions cannot suppress anchor variance ⇒ K=8 subspace projection at L=26 is the smallest effective intervention rank that achieves free-lunch (§6.2.3 + §A.5 27-cell pilot grid)."

This is **strictly weaker** than the original framing but still load-bearing for the paper: it predicts (P1-4) CAA K=1 and ITI head-level should fail free-lunch on ≥1 dataset (consistent with the §6.5 structural-reduction Note).

### Don't-touch (R5 bar-raiser protect-list, unchanged)

The 7 protect-list items in `references/roadmap.md` Phase 5 sprint-ordering are not affected by this outcome:
- (a − m) calibration contrast — still load-bearing.
- Single-model 6-callsite hedge — still appropriate.
- §6.2.3 5/5 + InfoVQA fence — still correct.
- Δem(non-anchored) ≥ 0 clause — still substantive.
- §1.5 (1) hedge stack — still appropriate.
- §5.3 dataset-dependent peak self-disclosure — still a strength.
- §4.7 InternVL3 boundary case — still correctly framed.

---

## Acceptance criteria — final

| ID | Criterion | Outcome | Plan branch |
|---|---|---|---|
| (a) | rank-8 elbow at L=26 (sv_7/sv_8 ≥ 1.5 OR EV@K8 ≥ 0.70) | **FAIL** (1.019, 0.213) | "Insight 2 stays empirical" |
| (b) | eff_rank monotonic decrease L=10 → final | **FAIL — INVERTED** (1399 → 1713, +22%) | "framework framing softened to 'consistent with' rather than 'verified by'" |

Plan §P0-2 explicit graceful-degradation language: *"(a only / b only / neither) graceful degradation: Insight 2 stays empirical; per-layer figure becomes transparency item; framework framing softened to 'consistent with' rather than 'verified by.'"* This evidence doc implements that branch. The corresponding §6.4 / §5.2 / §1.5 prose updates in `docs/paper/emnlp_draft_ko.md` are **queued as a P0-2 follow-on** — the paper still over-claims "spectrum-predicted dimensionality" / "compact low-dim subspace at late layers" until that pass lands.

The figures (`docs/figures/P0-2_L26_spectrum.png`, `docs/figures/P0-2_per_layer_rank_trajectory.png`) are the transparency items — they ship as appendix figures (proposed: §A.5.1 spectrum + §A.5.2 trajectory) so the spectrum is honestly disclosed even though the original prediction did not land.

## Open questions / queued follow-ons

1. **Anchor-specificity control (queued P0-2-control).** The (b) finding — Shannon `eff_rank` increases L=10 → L=26 — is reported on the (a − m) anchor-difference matrix only. Without a non-anchor baseline (e.g., (d − b) neutral-image-difference at the same N samples and same layers, or a random pair contrast on calibration set residuals), we cannot strongly distinguish "L=26 is the **anchor-information** maximum-dispersion site" from "L=26 is the residual-stream maximum-dispersion site for any contrast" (generic transformer behavior — early layers haven't filled the residual stream with computation yet). The doc therefore stops at "the (a − m) residuals are encoded at high effective rank at L=26" and does **not** make the stronger anchor-specific claim. Cheap to run (~30 min CPU on existing D_all.pt files if neutral-condition residuals are extractable, or ~1 H200-hour to extract h(d) − h(b) per sample). Tracking as a follow-up before the §5.2 / §6.6 paper rewrite.

2. **Paper prose update (queued P0-2-prose).** §6.4 Insight 2 / §5.2 Insight 4 / §6.6 reconciliation paragraph / §1.5 (4a) restated chain — none of these are yet edited in `docs/paper/emnlp_draft_ko.md`. The evidence doc + roadmap document the required edits; the actual prose pass is the next P0-2 work item. Should land before the next /paper-review-loop run so reviewers see the updated framing.

---

## Reproducibility

```
uv run python scripts/analyze_e6_eigenvalue_spectrum.py \
    --model llava-onevision-qwen2-7b-ov \
    --tags plotqa,infographicvqa \
    --K-probe 8 \
    --peak-layer 26
```

CPU-only, ~60 s wall-clock. Notebook at `notebooks/P0-2_eigenvalue_spectrum.ipynb` runs the same analysis cell-by-cell with figure rendering inline.

**Inputs (existing on disk):** `outputs/e6_steering/llava-onevision-qwen2-7b-ov/calibration_{plotqa,infographicvqa}/D_wrong.pt` from the 2026-05-04 Phase B Stage 4-final calibration.

**Outputs (canonical):** `docs/insights/_data/p0_2_per_layer_spectrum.{csv,md}`, `docs/insights/_data/p0_2_acceptance_verdict.json`, `docs/insights/_data/p0_2_full_svs_top.pt`, `docs/figures/P0-2_*.png`.
