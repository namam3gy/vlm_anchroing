# E1d — peak-layer attention is correlational; the anchor's causal pathway is multi-layer

**Status:** Causal follow-up to E1b across the 6-model panel. Source data: `outputs/causal_ablation/<model>/<run>/predictions.jsonl`. Aggregate tables: `outputs/causal_ablation/_summary/{per_model_per_mode.csv, by_stratum.csv}`. Full writeup: `docs/experiments/E1d-causal-ablation.md`.

> **2026-05-10 update — Phase E OneVision analyzer fix landed (P4-12 closed).**
> The earlier "0.000 baseline df on all 4 datasets" symptom was traced to two issues, both fixed in `scripts/analyze_causal_ablation.py`: (i) `_build_triplets` was joining base/anchor on `sample_instance_id` only and so collapsed across datasets — fixed by adding `dataset` to the join key (commit `a7e391c`); (ii) the dataset key for each OneVision run dir was unknown to the analyzer — fixed by hardcoding the timestamp → dataset map for the canonical Phase E runs and adding a susceptibility-CSV qid-intersection auto-detect for re-runs (commit `de1f94e`). Per-dataset OneVision susceptibility CSVs (`docs/insights/_data/susceptibility_<ds>_onevision.csv`) are loaded for stratum lookup; the panel-wide `susceptibility_strata.csv` is used only for the legacy 6-mech-panel models.
>
> **OneVision Phase E results (n=200 stratified per dataset, B=2,000 bootstrap CI; canonical CSV: `outputs/causal_ablation/_summary/per_model_per_mode.csv`):**
>
> | Mode | TallyQA | InfoVQA | ChartQA | MathVista | PlotQA |
> |---|---:|---:|---:|---:|---:|
> | baseline df | 0.130 | 0.167 | 0.105 | 0.171 | 0.243 |
> | Δ `ablate_peak` (pp) | −0.5 [−5.0, +4.0] | +1.5 [−3.9, +7.0] | 0.0 [−4.0, +4.5] | 0.0 [−5.1, +5.5] | −0.6 [−6.2, +5.5] |
> | Δ `ablate_peak_window` | +0.5 [−4.0, +5.5] | +0.4 [−4.6, +5.6] | +0.5 [−3.5, +5.0] | −0.5 [−5.5, +4.9] | −1.0 [−6.6, +5.2] |
> | Δ `ablate_lower_half` | +5.0 [+0.0, +10.5] | −0.6 [−5.5, +4.7] | +2.6 [−2.0, +7.3] | **+7.5 [+1.6, +13.6]** | +2.4 [−3.7, +8.7] |
> | Δ `ablate_upper_half` | −2.5 [−6.5, +2.0] | +0.4 [−4.7, +6.0] | −0.5 [−4.4, +4.0] | −2.6 [−7.1, +2.4] | −3.9 [−9.4, +1.9] |
> | Δ `ablate_all` | −4.0 [−7.5, +0.0] | +0.8 [−4.2, +6.3] | +0.6 [−3.5, +5.1] | −4.5 [−9.0, +0.4] | −5.1 [−10.6, +0.5] |
>
> **Reading on OneVision (n=200 per dataset):**
> - **Single-layer ablation 5/5 null** (`ablate_peak` and `ablate_peak_window` 모두 5 dataset 전부 95 % CI overlap 0; max |Δ| = 1.5 pp on InfoVQA peak). 6-mech panel 6/6 null과 일관 — multi-layer redundancy claim이 OneVision Main으로 *확장* 검증.
> - **Upper-half ablation은 6-mech panel에서 −4.0 ~ −10.5 pp 균일 significant인 것과 달리, OneVision은 5/5 null at n=200** (point estimates [−3.9, +0.4] pp, 모든 CI overlap 0). PlotQA −3.9 pp가 가장 가깝지만 95 % CI [−9.4, +1.9]로 0 포함. 이는 §5.3 OneVision dataset-dependent peak (Plot/Tally L=27, Info/VQAv2 L=14)와 일관 — encoder-family-fixed upper-half locus가 OneVision에서는 *uniform 효과를 산출하지 않음*. 이 qualification은 §6.2 subspace projection이 attention re-weighting보다 OneVision에 적합한 mechanism-level 이유를 제공.
> - **Lower-half BACKFIRE는 OneVision에서도 1/5 significant** (MathVista +7.5 pp [+1.6, +13.6]) + TallyQA boundary (+5.0 pp [+0.0, +10.5]) — 6-mech panel의 3/6 backfire와 같은 heterogeneity pattern.
> - **Full ablation 0/5 significant on OneVision at n=200** (6-mech panel의 −5 ~ −12 pp uniform과 대비) — OneVision의 anchor effect dynamic range가 legacy panel 대비 좁다 (likely dataset-distribution effect, GT range가 넓어 anchor pull이 분산).
>
> **`mean_distance_to_anchor` caveat for OneVision.** OneVision은 다이어그램 응답에서 매우 큰 hallucinated 숫자 (e.g. 1e6 단위)를 산출하는 경우가 있어 OneVision row의 `mean_distance_to_anchor` (3000–8000 range)와 그 CI는 *fluency 비교 metric으로 사용 불가*. C-form direction-follow는 sign-only `(pa−pb)·(anchor−pb) > 0` 정의이므로 magnitude outlier에 영향받지 않는다 — 따라서 본 표의 Δdf 결과는 신뢰 가능.
>
> ---

> **2026-04-28 update (B안 — full C-form propagation).**
> `analyze_causal_ablation.py` was refactored to read the canonical M2
> `_moved` flag instead of computing a Phase-A pull-form
> `(|num_pred − anchor| < |base_pred − anchor|)`. The qualitative
> findings (single-layer ablation null on 6/6; upper-half ablation
> reduces df on 6/6; lower-half is heterogeneous; ablate_all has the
> largest reduction with fluency cost on 3/6) all survive. Quantitative
> ranges shift by 1-2 pp:
>
> - `ablate_upper_half`: pull-form −5.5 to −11.5 pp → C-form **−4.0 to −10.5 pp**
> - `ablate_all`: pull-form −10 to −22 pp → C-form **−9.6 to −24.5 pp**
> - `ablate_peak`: pull-form |Δ df| ≤ 3.2 pp → C-form |Δ df| ≤ 2.0 pp
> - `ablate_layer0`: pull-form Δ ∈ [−2.7, +0.5] pp → C-form same range
>
> Per-model deltas under C-form are documented in the refreshed
> `outputs/causal_ablation/_summary/per_model_per_mode.csv`. Pre-refactor
> pull-form results archived at `outputs/before_C_form/causal_ablation/`.
> The `_moved` flag is the M2 canonical numerator under C-form
> `(pa-pb)·(anchor-pb) > 0 AND pa != pb`, matching the §3.3 / §5
> headline metric exactly.

## The claim and the test

E1b reported that anchor attention concentrates at a single LLM layer per encoder family — Gemma L5, mid-stack cluster L14–16, Qwen and FastVLM L22. The natural follow-up: if we *remove the model's ability to attend to the anchor at that layer*, does the model's anchor-pull behaviour drop?

We ablate the anchor span at six different layer sets (single peak, peak ± 2, lower half, upper half, all layers, layer 0 as control) for each of the 6 models, n=200 stratified, three conditions, and compare `direction_follow_rate` against baseline. See `scripts/causal_anchor_ablation.py` and `scripts/analyze_causal_ablation.py`.

## What we found

| Mode | What it tests | Result (C-form) |
|---|---|---|
| `ablate_peak` | E1b's headline | **Null on 6/6 models** (|Δ df| ≤ 2.0 pp, all CIs overlap baseline) |
| `ablate_layer0` | Non-peak control | **Null on 6/6 models** (Δ df ∈ [−2.7, +0.5] pp, all CIs overlap baseline) — including Gemma, whose E1b reported anchor↔target swaps at L0–4 |
| `ablate_all` | Upper bound on causal effect | **−9.6 to −24.5 pp** on direction_follow, but **fluency degrades on 3/6 models** (mean-distance balloons 4–6× or 1000×) |
| `ablate_upper_half` | Mitigation candidate | **−4.0 to −10.5 pp** on **6/6 models**, fluency clean on 4/6 (mid-stack cluster + Qwen); Gemma marginal; FastVLM broken |
| `ablate_lower_half` | Diagnostic | **Heterogeneous: 3/6 BACKFIRE, 1/6 reduce, 2/6 flat** |

The single-layer ablation null — at the E1b peak *and* at layer 0 — is the most surprising result. E1b's per-layer peak (with up to 5–10× the layer-averaged anchor mass) was the obvious E4 intervention site, and Gemma's L0–4 anchor↔target swaps were the most plausible "early-layer alternative". Neither is load-bearing on its own.

## Reading

Reading **(A) Multi-layer redundancy** is confirmed by the layer-0 control. The anchor's effect on the answer is *encoded redundantly* across the LLM stack — removing one layer's view of the anchor at any single site (peak, layer 0, or anywhere in between, on any of 6 models) leaves the rest of the stack to reconstruct it. The peak layer is where the signal is most visible, not where it is uniquely produced. Reading **(B) "peak is correlational and a different single layer matters"** is unsupported even on Gemma, whose E1b L0–4 anchor↔target swaps made it the most plausible "early-layer alternative" candidate (Gemma layer-0 Δ = +0.005, indistinguishable from its peak Δ = +0.020).

The pragmatic positive result: **`ablate_upper_half` is the single mitigation locus that works on the entire 6-model panel** without exploding fluency on the mid-stack cluster. For LLaVA-1.5, ConvLLaVA, InternVL3, upper-half ablation reduces `direction_follow_rate` by 4.0–10.5 pp (C-form) while keeping `mean_distance_to_anchor` essentially flat. That is the strongest "architecture-blind" candidate the panel has yielded; the three encoders (CLIP-ViT, InternViT, ConvNeXt) at the same depth respond similarly.

## Why this matters

Two things change for the experiment plan and one stays the same.

**Changes for E4 design.** The per-family-peak attention re-weighting prototype previously planned by E1b is shelved — single-layer null at the peak *and* at layer 0 across all 6 models. Multi-layer intervention is required for any architecture in the panel. The simplest E4 prototype that works is *upper-half anchor attention re-weighting on the mid-stack cluster*.

**Changes for the paper claim.** The "where in the LLM stack the anchor gets read in" framing from E1b/E1c is *correlationally* sharp but does not imply *causal* per-family intervention sites. The paper text should distinguish:

- E1b finding: per-family peak layer is where the anchor signal concentrates in attention space (correlational).
- E1d finding: that single layer is not causally load-bearing on its own (single-layer attention-mask ablation null). The layer-0 control rules out the alternative "the E1b peak is the wrong single layer" — single-layer ablation is insufficient regardless of which layer.
- E1d finding: upper-half is the single architecture-blind mitigation locus that works on the panel — fluency-clean on the mid-stack cluster + Qwen, marginal on Gemma, broken on FastVLM.

**What stays.** The four-archetype encoder-family taxonomy from E1b is still the cleanest description of the *attention* signature. We are not retracting it. We are saying it's correlationally informative, not a recipe for single-layer causal interventions.

## Sub-finding flagged as "do not generalise to single architecture cluster"

ConvLLaVA and LLaVA-1.5 share the E1b answer-step peak (L16), the same text-stealing budget source, and almost-identical magnitudes. **They respond *opposite* to lower-half ablation.** ConvLLaVA `delta_df = −0.150`, LLaVA-1.5 `delta_df = +0.135` (C-form). Same-attention-signature does not imply same-causal-structure. We treat this as a caveat against treating "mid-stack cluster" as a *causally* uniform group; it remains a useful *attention-signature* group.

## What this doesn't say

- **The anchor isn't causal.** It is. `ablate_all` reduces `direction_follow` by 11–22 pp universally. The anchor is causally load-bearing in the aggregate; we just can't pin the entire effect on one layer.
- **The peak layer isn't useful.** It is the strongest correlational marker we have for where the anchor signature is in each model. It just isn't a single-layer intervention site.
- **All multi-layer ablations are equivalent.** They are not. `ablate_lower_half` BACKFIRES on 3/6 models (Gemma +27 pp, LLaVA-1.5 +16.5 pp). Multi-layer ablation needs the right *layers*, not just multiple layers.

## What we did NOT test

- Per-head sparsity. A single attention head at the peak layer might still be load-bearing where the layer-aggregate is not. This is a strict generalisation of the present test and is a clean follow-up.
- Vision-token re-projection (zero the anchor's embeddings before they enter the LLM). E1d only manipulates attention masks; the anchor still occupies the residual stream. The "deny the anchor input entirely" intervention class is not tested here.
- Combination ablations (peak + a complementary layer). The single-layer null suggests the right combination might recover the `ablate_all` effect more cleanly than `ablate_all` itself does.

## Implications for the experiment plan

- **E4 intervention class — open question 1 closed.** Single-layer attention re-weighting (at the E1b peak or layer 0) is ruled out as a candidate. Upper-half re-weighting is the prototype.
- **E4 architecture coverage.** The mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3) is the single highest-leverage prototype target — three encoders, one shared upper-half-clean response.
- **A new open question: multi-layer combinatorial ablation.** Does the union of (anchor span ablated at peak) + (anchor span ablated at one complementary layer) recover an `ablate_all`-magnitude reduction at lower fluency cost? Cheap to run; informs E4 directly.
- **Roadmap §6 E1 row:** open question "causal test" → **closed (null on single layer; layer-0 control confirms multi-layer redundancy across 6/6 models)**. Open new question: head-level sparsity. Open new question: multi-layer combinatorial ablation.

## Caveats

- Bootstrap CIs are over triplets, not over pairs. Slightly conservative.
- **InternVL3 and FastVLM both lose triplets** under several ablation modes (no-digit emissions). The "BACKFIRE" we report on InternVL3 lower-half is on n=83 vs baseline n=137 — survivor-bias caveat applies. FastVLM under `ablate_all` and `ablate_upper_half` similarly drops triplets. Read those entries as suggestive, not paired.
- The mask is `−1e4` not `−inf` (bf16-safe). Spot-checked on LLaVA-1.5: post-hook anchor-column attention weights are < 1e-4 across all heads at targeted layers. The intervention is empirically a strong suppression, not a guaranteed zero.
- Susceptibility strata loaded from `docs/insights/_data/susceptibility_strata.csv`; same as E1b.
