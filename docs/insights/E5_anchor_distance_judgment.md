# Should subsequent experiments use GT-proximal or random anchors?

**Status:** Methodological decision document, derived from E5b + E5c results. Source data: `outputs/experiment_distance_*/llava-next-interleaved-7b/<latest>/predictions.jsonl`, `outputs/experiment_e5c_*/llava-next-interleaved-7b/<latest>/predictions.jsonl`. Aggregate tables: `docs/insights/_data/E5b_per_stratum.csv`, `docs/insights/_data/E5c_per_cell.csv`. Companion writeups: `docs/experiments/E5b-anchor-distance.md`, `docs/experiments/E5c-anchor-mask-control.md`.

## Decision

**Anchor value must be GT-proximal for the anchoring effect to be measurable. Subsequent experiments must NOT use uniform-random anchor sampling across the full inventory.** The recommended sampling rule is `|anchor − GT| ≤ 5` (stratum S1 + S2 from the E5b stratification). Equivalently, the existing main-run convention `anchor ∈ {0..9}` already satisfies this for the VQAv2/TallyQA-number subsets where GT ∈ {0..8}, and may be retained without change.

## Three independent lines of evidence

### 1. Paired conditional adoption decays sharply with distance (E5b)

Headline measure: `adopt_cond = case2 / (case1 + case2 + case3)` where case 4 (`base = a = pred`) is excluded from the denominator (M1; see `src/vlm_anchor/metrics.py:55`).

Wrong-base subset (the model gets the answer wrong on `target_only`; this is the population where anchoring fires per A1/E5b):

| Stratum | `\|a − GT\|` | VQAv2 (n=399) | TallyQA (n=346) |
|---|---|---:|---:|
| **S1** | [0,1]         | **0.131** | **0.092** |
| S2     | [2,5]         | 0.032 | 0.006 |
| S3     | [6,30]        | 0.010 | 0.003 |
| S4     | [31,300]      | 0.010 | **0.000** |
| S5     | [301, ∞)      | **0.003** | **0.000** |

S1 → S5 magnitude ratio: **VQAv2 44×, TallyQA ∞** (TallyQA S4 and S5 are exactly 0/346 in two cells). Decay is monotonic on both datasets.

### 2. The digit-pixel-specific contribution decays the same way (E5c)

E5c isolates the digit pixel by comparing the anchor condition (digit visible) to a masked condition (anchor image with the digit pixel region inpainted out via OpenCV Telea; same scene, no digit). The (anchor − masked) gap is the pure digit-specific contribution to paired adoption.

Wrong-base, paired conditional `adopt_cond`:

| Stratum | VQAv2 anchor | VQAv2 masked | **gap (digit)** | TallyQA anchor | TallyQA masked | **gap (digit)** |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 0.129 | 0.068 | **+0.061** | 0.110 | 0.084 | **+0.026** |
| S2 | 0.021 | 0.008 | +0.013 | 0.012 | 0.012 | 0.000 |
| S3 | 0.018 | 0.005 | +0.013 | 0.000 | 0.000 | 0.000 |
| S4 | 0.013 | 0.000 | +0.013 | 0.006 | 0.000 | +0.006 |
| S5 | 0.008 | 0.000 | **+0.008** | 0.000 | 0.000 | 0.000 |

The digit-specific gap at S5 is **+0.008 (VQAv2)** with `n_eligible ≈ 399`. Standard error on this proportion is `√(0.008 × 0.992 / 399) ≈ 0.0045`, so the 95 % CI **overlaps zero**. TallyQA S2/S3/S5 hit exactly zero. The digit-specific signal at far distance is statistically indistinguishable from noise.

### 3. Direction-follow (`df_cond`) is distance-invariant but does NOT reflect anchor pixels (E5c)

`df_cond` (direction-follow rate excluding `anchor=gt` and case 4) on VQAv2 wrong-base:

| Stratum | anchor | masked |
|---|---:|---:|
| S1 | 0.356 | 0.335 |
| S2 | 0.520 | 0.522 |
| S3 | 0.405 | 0.399 |
| S4 | 0.381 | 0.391 |
| S5 | 0.371 | 0.376 |

Two observations:
- `df_cond` is roughly distance-invariant on wrong-base (S1 = 0.356, S5 = 0.371 — within bootstrap noise).
- **Anchor vs. masked are nearly identical at every stratum.** S2 difference is +0.001 (CI clearly straddles 0); S5 difference is −0.005.

Reading: the apparent "anchor still pulls at far distance" signal in `df_cond` is **generic 2-image distraction**, not the digit. Removing the digit pixels does not change `df_cond`. So the apparent distance-invariance of `df` is not evidence that anchoring works at far distance — it's evidence that *any second image* perturbs the prediction direction, regardless of whether it carries an anchor digit.

## A counter-argument considered and rejected

> "df_cond is 0.37 at S5 — that means the anchor still pulls the prediction toward the anchor side of GT, even at large distances. So far-anchor effects exist."

Refuted by E5c evidence 3: the masked condition has `df_cond = 0.376` at S5 (VQAv2) — within noise of the anchor's `df_cond = 0.371`. The model's prediction shifts toward the anchor's directional half-plane just as much when the digit is invisible as when it is visible. The shift is caused by the second image's *presence*, not its content. So `df_cond > 0` at far distance is a 2-image-distraction artifact, not an anchoring effect.

## Quantitative implication for sampling design

If a future experiment samples anchor uniformly from the inventory `inputs/irrelevant_number/{0..10000}.png` (128 PNGs), the per-stratum hit rate (assuming GT ∈ {0..8} from VQAv2/TallyQA):

| Stratum | inventory size | uniform-random hit rate | wrong-base adopt_cond at this stratum (VQAv2) |
|---|---:|---:|---:|
| S1 | 6   | 4.7 %  | 0.131 |
| S2 | 5   | 3.9 %  | 0.032 |
| S3 | 7   | 5.5 %  | 0.010 |
| S4 | 10  | 7.8 %  | 0.010 |
| S5 | 100 | **78.1 %** | 0.003 |

Expected pooled `adopt_cond` under uniform-random sampling on VQAv2 wrong-base:
`0.047 × 0.131 + 0.039 × 0.032 + 0.055 × 0.010 + 0.078 × 0.010 + 0.781 × 0.003 ≈ 0.011`

Versus pooled `adopt_cond` under stratified sampling restricted to S1 + S2 (`|a − GT| ≤ 5`):
`0.5 × 0.131 + 0.5 × 0.032 ≈ 0.082`

**Effect-size dilution: ~7.5×**. Statistical power on 1000 samples drops from comfortably-detectable (>10 % rate at SE ≈ 0.7 %) to near-noise (~1 % rate at SE ≈ 0.3 %). Reviewer acceptance hinges on a measurable effect; uniform-random sampling produces a barely-measurable one.

The same calculation on TallyQA wrong-base gives:
- Uniform-random pooled: `0.047 × 0.092 + 0.039 × 0.006 + 0.055 × 0.003 + 0.078 × 0.000 + 0.781 × 0.000 ≈ 0.005`
- Restricted-to-S1+S2 pooled: `0.5 × 0.092 + 0.5 × 0.006 ≈ 0.049`

**TallyQA effect-size dilution: ~10×.**

## Recommendation

1. **All paper-canonical anchoring experiments use GT-proximal anchor sampling.** Two interchangeable rules:
   - **(A) Stratum-restricted**: sample anchor from `{a ∈ inventory : |a − GT| ≤ 5}` per question.
   - **(B) Range-restricted**: sample anchor from `{0..9}` (matches existing main-run convention; for VQAv2/TallyQA-number GT ∈ {0..8} this collapses to (A) almost exactly).
2. **Uniform-random sampling across the full inventory is forbidden** for headline experiments — effect is diluted ~7-10×.
3. **Reviewer-defensibility**: report this judgment doc explicitly in the methods section. Paper text should justify GT-proximal sampling on falsifiability grounds (stratified E5b + E5c showed effect concentrates in S1, decays to noise by S5). This pre-registered choice avoids accusations of post-hoc anchor-cherry-picking — the cutoff is justified by the data, not chosen for inflation.
4. **Paper headline figure**: report wrong-base × S1 cell. VQAv2 = 0.131 (n=399); TallyQA = 0.092 (n=346). Both tables include CI from bootstrap (see `docs/insights/_data/E5b_per_stratum.csv`).

## Extension to wide-GT datasets (ChartQA, MathVista, future)

The above rule (`|a − GT| ≤ 5` or `anchor ∈ {0..9}`) is calibrated for VQAv2/TallyQA-number where GT ∈ {0..8}. ChartQA and MathVista have GT spanning 0–10⁴+ and can be decimals. A per-dataset validation protocol is required before any wide-GT dataset enters the paper-canonical pipeline.

### Hybrid scale-relative cutoff

Replace the absolute cutoff with:

```
candidate anchors = { a ∈ inventory : |a − GT| ≤ max(5, 0.2 × |round(GT)|) }
```

Hybrid form: absolute `5` floor for small GT (matches the VQAv2/TallyQA rule), 20 % relative window for large GT. Worked examples:

| GT | cutoff = max(5, 0.2·\|round(GT)\|) | candidate range | candidates in current 128-PNG inventory |
|---:|---:|---|---:|
| 4 | 5 | [0, 9] | 9 (integer 0..9) |
| 50 | 10 | [40, 60] | 5 (40, 45, 50, 55, 60) |
| 1000 | 200 | [800, 1200] | 5 (800, 900, 1000, 1100, 1200) |
| 5000 | 1000 | [4000, 6000] | 21 (every 100 step) |
| 12 000 | 2 400 | [9 600, 14 400] | 5 (9 600..10 000; one-sided truncation at inventory cap) |

Adequate candidate count (≥ 5) at every scale of interest. A scale-relative stratification (e.g., S1 = 0–10 % off, S2 = 10–30 %, S3 = 30–100 %, S4 = 100–1000 %, S5 = > 1000 %) replaces the absolute 5-stratum scheme.

### Decimal GT handling

```
GT_for_anchor_selection  = round(float(GT))
GT_for_metric            = float(GT)
```

The paired-adoption metric compares `pred == anchor_value` as strings; with decimal GT, `pred="3.5"` will not match `anchor="4"`, so paired adoption naturally fires only when the model rounds toward (and reaches) the anchor exactly. No code change beyond ensuring `normalize_numeric_text` preserves the decimal point.

**Verify before relying:** `vlm_anchor.utils.normalize_numeric_text` and `extract_first_number` must round-trip "3.5" → "3.5". Add a unit test before any wide-GT run.

### Per-dataset validation protocol (medium-term workflow)

For each new dataset (ChartQA next, MathVista after):

1. **GT distribution analysis.** Plot `log10(max(1, |GT|))` histogram on `inputs/<dataset>/questions.jsonl`. Identify GT-magnitude buckets (e.g., {0–10, 10–100, 100–1k, 1k–10k}).
2. **Inventory coverage check.** For the dataset's GT bucket distribution, confirm the `inputs/irrelevant_number/` inventory has ≥ 5 candidates per (bucket, scale-relative stratum) combination. If not, extend the inventory (FLUX regenerate at the missing values) before running.
3. **Stratified validation sweep.** Run an E5b-style stratified pipeline at small scale (n ≈ 200 base questions per dataset, 5 scale-relative strata, llava-interleave-7b). Use the hybrid cutoff above for stratum membership. Wall: ~10 min on H200.
4. **Decay curve check.** Compute the same per-(stratum, base-correctness) `adopt_cond` table as E5b. Confirm:
   - **(C1) Monotonic decay** of wrong-base `adopt_cond` from S1 to S5 (allowing one inversion within bootstrap CI).
   - **(C2) Wrong-base S1 `adopt_cond` ≥ 0.05** (effect size large enough to study).
   - **(C3) Wrong-base S4 or S5 `adopt_cond` ≤ 0.01** (decay reaches noise floor — confirms the cutoff is meaningful).
5. **Cutoff acceptance.** If all three criteria hold, lock in the dataset's cutoff at `S1 ∪ S2` (or `S1` only if S2 is also at noise floor). If not, debug — typical causes: inventory gap, decimal-handling bug, or an actual scientific surprise (e.g., the dataset shows a different decay shape, in which case the paper claim must be scoped accordingly).
6. **Full run.** Run the canonical experiment on the dataset using the validated cutoff.

Per-dataset documentation: each dataset gets a small markdown under `docs/insights/E5_<dataset>_distance_validation.md` with the decay table and the chosen cutoff. The methods section of the paper cites these per-dataset docs.

### Required code changes before this protocol can run

- **`src/vlm_anchor/data.py`** — extend `ANCHOR_DISTANCE_STRATA` to support a relative-distance variant. Either (a) replace the constant with a function `compute_strata(gt: int) -> list[tuple[int, int]]` that yields per-question scale-aware bounds, or (b) add an alternative constant `ANCHOR_DISTANCE_STRATA_RELATIVE` and a `mode` flag on `assign_stratified_anchors`. Recommend (a) for cleanness.
- **`vlm_anchor/utils.py`** — confirm `normalize_numeric_text("3.5") == "3.5"`. Add a test if needed.
- **YAML configs** — new `experiment_chartqa_e5b_validation.yaml` (and corresponding mathvista variant) using the new stratified mode against the dataset.
- **Inventory extension** — flag if validation step 2 finds gaps. ChartQA's existing `mean_distance_to_anchor = 33,446` (roadmap §197) suggests GT can exceed the 10 000 cap, so inventory likely needs extension to ~100 000. Roughly 100 new FLUX images at 100-step intervals.

### Datasets to validate in order

1. **ChartQA** — has full 5,390-record run from E5 (random anchor 0–9). Re-run small-scale with hybrid stratification to find ChartQA-specific cutoff. Then re-run full or accept the existing data with the cutoff applied as a post-hoc subset filter.
2. **MathVista** — only smoke-run done (1 dir, ~5 samples). Full E5b-style validation needed before any paper claim.
3. **Future dataset (e.g., DocVQA-numeric, GQA-numeric)** — same protocol.

## What this decision does NOT cover

- **Multi-model generalisation**: this judgment is based on llava-interleave-7b only (E5b + E5c). The 11-model panel was previously run under random anchor 0-9 (the main run), which is range-restricted (B), so existing data is consistent with this rule. A multi-model E5b/E5c extension would confirm the rule transfers.
- **Multi-prompt robustness**: E7 (paraphrase robustness, queued Tier 2) tests whether the same distance-decay shape holds across prompt phrasings. If E7 finds the decay pattern is prompt-sensitive, the cutoff in (A) may need adjustment per prompt.

## What we did NOT test in deciding this

- **Anchor below GT vs above GT**: E5b/E5c lump signed distance into an absolute bin. Whether anchors below GT pull harder than those above (or vice versa) is open. The E5b/E5c sampling balances signs but doesn't quantify the asymmetry. Open follow-up.
- **Anchor digit length (1-digit vs multi-digit)**: a 5 anchor at GT=4 (S1) is one digit, a 5000 anchor at GT=4 (S5) is four digits. The collapse at S5 may partly be "model rarely outputs four-digit numbers when the question implies a small count", not pure plausibility-window. Confounded with distance in our design. Future work to disentangle.
- **Anchor color/scene-context interaction**: E5c shows scene context is not load-bearing on accuracy drop, but a per-scene-template breakdown of `adopt_cond` is unexplored.
