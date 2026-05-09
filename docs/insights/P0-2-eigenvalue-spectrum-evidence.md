# P0-2 — Per-layer eigenvalue spectrum of the (a − m) anchor calibration matrix

**Status (2026-05-09).** Both pre-registered spectrum acceptance criteria FAIL in the predicted direction (criteria (a) and (b), see verdict table below). Subsequent **H-A test (multi-layer redundancy hypothesis)** is **WEAK partial confirmation only** (3 of 11 early layers above threshold; pre-registered ≥4 = CONFIRMED, ≥2 = WEAK, ≤1 = FALSIFIED). Decisive test of the "late-cluster integration" interpretation is **P0-2-followup** (cross-layer mitigation at L=22/24/25); the paper-prose update **P0-2-prose** is *blocked* on (i) P0-2-followup landing in the predicted shape AND (ii) P0-2-control showing the late-cluster pattern is anchor-specific (i.e., does not appear on `(d − b)` neutral baseline). Until both land, the appropriate framing is "consistent with multi-layer accumulation localized to L=22-26, decisive test pending" — NOT a tier-shift upgrade.

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

## H-A test — does multi-layer redundancy explain the high-rank residual?

**Hypothesis.** §5.2 multi-layer redundancy (anchor signal redundantly written by multiple attention layers) PREDICTS the §6.4 high-effective-rank residual at L=26. Each attention layer writes anchor information into the residual along a slightly different direction; these directions accumulate at L=26 → naturally high effective rank, K=8 captures the multi-layer-redundant direction stack.

**Method — three measures, two are non-tautological.** Per-layer thin SVD of the **incremental** anchor matrix `ΔD[:, L, :] = D[:, L, :] − D[:, L−1, :]` (anchor-specific contribution of layer L's attn+FFN write) and of the **cumulative** matrix `D[:, L, :]`. Top-1 right singular vectors `u_L^inc` and `u_L^cum` per layer. Three alignment measures:

```
sub_align_inc(L) = ‖V_K[L=26]^T · u_L^inc‖₂              # NON-tautological at all L
sub_align_cum(L) = ‖V_K[L=26]^T · u_L^cum‖₂              # TAUTOLOGICAL: = 1.0 at L=26 by construction
EV_frac_VK(L)    = ‖V_K V_K^T D[:, L, :]‖_F² / ‖D[:, L, :]‖_F²    # NON-tautological at all L
                                                          # at L=26 it equals EV@K8(L=26) ≈ 0.21, NOT 1.0
```

`V_K[L=26]` is the 8 × d_model matrix of top-8 right singular vectors of the cumulative `D[:, L=26, :]` (= the chosen E6 cell's subspace). For `sub_align_*`, random baseline is `√(K/d) = √(8/3584) = 0.0472`; threshold 0.30 = ~6.4× sqrt-chance level. For `EV_frac_VK`, random isotropic baseline is `K/d = 0.00223`; the multiplier `EV_frac_VK / (K/d)` reports excess concentration over random.

**Caveat (advisor flag).** `sub_align_cum` is tautological at L=peak (= 1.0 by construction since u_L^cum at L=26 = top-1 right singular vector of D[:, L=26, :] = v_0 of V_K). The trajectory leading up to L=26 (cumulative top-1 direction "rotating into" v_0) is therefore partly an artifact of the test, not an independent observation. **Interpret only the non-tautological measures (`sub_align_inc` and `EV_frac_VK`)** for inference about late-cluster localization.

**Pre-registered thresholds.**

| Verdict | Criterion |
|---|---|
| H-A CONFIRMED | ≥ 4 distinct early layers L (excluding L=26) with `sub_align_inc(L) > 0.30` |
| H-A WEAK | ≥ 2 distinct early layers above threshold |
| H-A FALSIFIED | ≤ 1 layer (V_K aligns only with L=26 itself — pure cumulative) |

Early window: `L ∈ {1, 5, 10, 12, 14, 16, 18, 20, 22, 24, 25}` (11 layers).

**Result: H-A WEAK partial confirmation.**

| Layer L | sigma_inc (top-1 σ of ΔD) | sub_align_inc (non-taut.) | sub_align_cum (taut. at L=26) | EV_frac_V_K (× random K/d) | top v_k |
|---|---|---|---|---|---|
| 0 | 7.70 | 0.045 | 0.045 | 0.0027 (1.2×) | v_3 (0.027) |
| 5 | 14.57 | 0.037 | 0.045 | 0.0060 (2.7×) | v_1 (0.023) |
| 10 | 19.17 | 0.078 | 0.080 | 0.0070 (3.1×) | v_0 (0.046) |
| 14 | 45.53 | 0.074 | 0.068 | 0.0067 (3.0×) | v_0 (0.051) |
| 18 | 50.48 | 0.071 | 0.145 | 0.0101 (4.5×) | v_1 (0.053) |
| 20 | 67.39 | 0.184 | 0.296 | 0.0212 (9.5×) | v_0 (0.164) |
| **22** | **106.74** | **0.413** ✅ | 0.577 | **0.0681 (30.5×)** | v_0 (0.354) |
| **24** | **102.80** | **0.394** ✅ | 0.790 | **0.1349 (60.4×)** | v_0 (0.380) |
| **25** | **130.83** | **0.494** ✅ | 0.870 | **0.1623 (72.7×)** | v_0 (0.312) |
| 26 (own) | 177.80 | 0.683 | 1.000 (taut.) | 0.2126 (95.2×) | v_0 (0.483) |
| 27 | 381.40 | 0.156 | 0.278 | 0.0859 (38.5×) | v_1 (0.092) |

**Pre-registered verdict label: WEAK** (3 of 11 early layers above `sub_align_inc` threshold 0.30; CONFIRMED required ≥ 4).

### What the data does and does not say

**Does say** (defended by both non-tautological measures):

1. The non-tautological `EV_frac_V_K` shows a **sharp transition between L=20 and L=22** (9.5× → 30.5× isotropic random baseline) and a continued rise through L=25 (72.7×) before L=26's local maximum (95.2×). Earlier layers (L=0..18) sit between 1.2× and 5.8× random — close to noise floor. This *is* consistent with V_K-alignable variance emerging within a 4-5 layer window approaching L=26. (Whether this emerging variance is anchor-specific or reflects generic late-residual rank growth is gated on P0-2-control — see below.)

2. `sub_align_inc` agrees on the same window (L=22, L=24, L=25 above 0.30; L=20 at 0.184, L=18 and earlier near sqrt-baseline ~0.04-0.10).

3. L=27 transition: `EV_frac_V_K` drops 95.2× → 38.5× and `sub_align_inc` drops 0.683 → 0.156. The LM-head logit-projection layer reorganizes structure away from V_K.

**Does NOT say** (claims that would over-extend the data):

1. **Cumulative trajectory is partly tautological.** `sub_align_cum` at L=26 = 1.0 by construction (u_L=26^cum = v_0 ∈ V_K). The trajectory 0.577 → 0.790 → 0.870 → 1.000 over L=22..26 is therefore not an independent confirmation that "anchor signal is being built up" — it is partly the artifact of u_L^cum rotating toward its own definitional limit. Use only `sub_align_inc` (non-tautological) and `EV_frac_V_K` (non-tautological at all L) for inference.

2. **"v_0 consensus + v_1..v_7 distributed" is unsupported.** v_0 dominates the late-cluster heatmap because it is the top singular vector — by definition it is the largest direction. The other v_k not showing specific-layer alignment in the heatmap does NOT positively imply "distributed within-block contributions" — it is equally consistent with v_1..v_7 being noise, generic-residual structure, or artifacts. The structural claim "block-coding consensus + 7 distributed" has no direct measurement support and is dropped from the doc.

3. **Anchor-specificity is not yet established.** `EV_frac_V_K(L)` rising 30-95× isotropic-random at L=22..26 is non-trivial relative to *isotropic random*, but the relevant comparison is `EV_frac_V_K` measured on a *non-anchor contrast* (e.g. (d − b) neutral-image-difference) at the same N samples and same V_K. If a non-anchor contrast also shows L=22..26 cluster, then "late-developed feature alignment" is the explanation rather than "anchor-information localization." **P0-2-control is the deciding test** and is now a *blocker* (not just a queued sibling) for the §5.2 / §6.6 paper-prose update P0-2-prose.

### Decisive test (queued)

The strong falsifiable test of the late-cluster interpretation is **P0-2-followup**: run E6 mitigation hook at single-layer ∈ {L=22, L=24, L=25} (instead of L=26). If the late-cluster claim is correct, free-lunch effect at L=22/24/25 should be *partially* recovered with magnitude *roughly proportional* to the layer's `sub_align_inc` or `EV_frac_V_K`. If the L=22/24/25 effects are flat or unpatterned, the late-cluster interpretation collapses.

Predictions are intentionally qualitative ("partial recovery, scaling with alignment") rather than quantitative. Earlier draft of this doc gave specific percentages (~58/79/87 %); these are dropped because (i) they were derived from the tautological `sub_align_cum` measure, and (ii) the linear-proportionality assumption between subspace alignment and mitigation effect size is itself unjustified — the relationship between V_K projection geometry and downstream behavioral effect on anchor adoption is not necessarily linear and may saturate or be threshold-dependent.

Estimated cost: ~6 H200-hour (3 layer choices × ~2 H200-hour × 5 datasets evaluation reuse).

### Acceptance verdict (final)

| Test | Pre-registered criterion | Outcome | Verdict label |
|---|---|---|---|
| H-A | ≥ 4 distinct early layers with `sub_align_inc > 0.30` | 3 / 11 (L=22, L=24, L=25) | **WEAK partial confirmation** |

Verdict written to `docs/insights/_data/p0_2_HA_verdict.json`. Per-layer alignment data at `docs/insights/_data/p0_2_HA_subspace_alignment.csv`.

Figures (transparency items, proposed §A.5.3 / §A.5.4):
- `docs/figures/P0-2_HA_subspace_alignment.png` — two-panel: top = `sub_align_inc/cum` trajectory (with explicit caveat that `sub_align_cum` at L=peak is tautological); bottom = non-tautological `EV_frac_V_K(L)` log-scale, showing L=20 → L=22 sharp transition (9.5× → 30.5× random isotropic baseline).
- `docs/figures/P0-2_HA_per_vk_heatmap.png` — `|v_k^T · u_L^inc|` heatmap (8 v_k × 28 layers).

### What this means for the paper

**Empirical findings shipped (unchanged):** P0-2 spectrum graceful-degradation (criteria (a), (b) FAIL); H-A WEAK partial confirmation of multi-layer alignment localized to L=22-26 cluster; K=8 explains 21% of L=26 anchor variance and yet achieves free-lunch (variance fraction ≠ causal sufficiency).

**Paper prose update P0-2-prose is BLOCKED until both land:**

1. **P0-2-control** (~30 min CPU - 1 H200-hour) — non-anchor baseline `(d − b)` at same N + same layers. Required to establish that the L=22-26 alignment cluster is anchor-specific, not generic late-residual rank growth.

2. **P0-2-followup** (~6 H200-hour) — cross-layer mitigation at L=22/24/25. Required to convert the H-A "consistent with multi-layer accumulation in late cluster" interpretation from a description of measurement geometry into a behavioral mechanism claim.

If both land in the predicted direction, the paper-prose update can claim concrete late-cluster localization. If P0-2-control shows the cluster on `(d − b)` baseline, OR if P0-2-followup shows flat/un-patterned mitigation effects across L=22/24/25, the late-cluster interpretation collapses and the prose update is limited to the original graceful-degradation framing (criteria (a), (b) FAIL; K=8 is em-deal-breaker selected).

**Until both land**, paper prose should NOT claim:
- "anchor integration localizes to L=22-26"
- "block-coding within late cluster"
- "v_0 consensus direction"
- specific quantitative cross-layer mitigation predictions (~58/79/87 %)

The only safe paper-prose statement currently supported is:
> "K=8 explains a small fraction (~21 %) of L=26 anchor variance and achieves free-lunch; per-layer subspace-alignment measurements are reported in §A.5 as transparency."

---

## Open questions / queued follow-ons

1. **Anchor-specificity control (queued P0-2-control).** The (b) effective-rank inversion finding — Shannon `eff_rank` increases L=10 → L=26 — is reported on the (a − m) anchor-difference matrix only. Without a non-anchor baseline (e.g., (d − b) neutral-image-difference at the same N samples and same layers, or a random pair contrast on calibration set residuals), we cannot strongly distinguish "L=26 is the **anchor-information** maximum-dispersion site" from "L=26 is the residual-stream maximum-dispersion site for any contrast" (generic transformer behavior — early layers haven't filled the residual stream with computation yet). The doc therefore stops at "the (a − m) residuals are encoded at high effective rank at L=26" and does **not** make the stronger anchor-specific claim. Cheap to run (~30 min CPU on existing D_all.pt files if neutral-condition residuals are extractable, or ~1 H200-hour to extract h(d) − h(b) per sample). Tracking as a follow-up before the §5.2 / §6.6 paper rewrite. **H-A weakly mitigates this concern**: late-cluster localization (L=22-26) of *anchor* alignment with V_K is itself non-trivial (the same control on (d − b) would not show this clustering if anchor processing genuinely happens at L=22-26 specifically).

2. **Cross-layer mitigation test (queued P0-2-followup).** Run E6 mitigation hook at single-layer ∈ {L=22, L=24, L=25} (instead of L=26). Predicted free-lunch effect: ~58 %, ~79 %, ~87 % of the L=26 magnitude (proportional to `sub_align_cum`). ~6 H200-hour total (3 × ~2 H200-hour each, eval on 5 datasets). Confirms or falsifies the late-cluster integration interpretation directly.

3. **Paper prose update (queued P0-2-prose).** §6.4 Insight 2 / §5.2 Insight 4 / §6.6 reconciliation paragraph / §1.5 (4a) restated chain — none of these are yet edited in `docs/paper/emnlp_draft_ko.md`. The evidence doc + roadmap document the required edits; the actual prose pass is the next P0-2 work item. **H-A WEAK-confirmed + late-cluster sharpening provides the prose with concrete claims** (5-layer block, v_0 consensus, sub_align trajectory). Should land before the next /paper-review-loop run so reviewers see the updated framing.

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
