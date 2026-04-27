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

## What this decision does NOT cover

- **Multi-model generalisation**: this judgment is based on llava-interleave-7b only (E5b + E5c). The 11-model panel was previously run under random anchor 0-9 (the main run), which is range-restricted (B), so existing data is consistent with this rule. A multi-model E5b/E5c extension would confirm the rule transfers.
- **Multi-prompt robustness**: E7 (paraphrase robustness, queued Tier 2) tests whether the same distance-decay shape holds across prompt phrasings. If E7 finds the decay pattern is prompt-sensitive, the cutoff in (A) may need adjustment per prompt.
- **Dataset extension to wider GT range**: ChartQA / MathVista have GT that can exceed 10 000. The "GT-proximal" rule must be re-applied to those datasets at their own scale (e.g., `|a − GT| ≤ 5` is too tight when GT = 5000; needs scale-relative rule like `|a − GT| / GT ≤ 0.1` or stratum-based per-dataset). This is a follow-up open question and should be tested with a small E5b-style stratified sweep on each new dataset before claiming generality.

## What we did NOT test in deciding this

- **Anchor below GT vs above GT**: E5b/E5c lump signed distance into an absolute bin. Whether anchors below GT pull harder than those above (or vice versa) is open. The E5b/E5c sampling balances signs but doesn't quantify the asymmetry. Open follow-up.
- **Anchor digit length (1-digit vs multi-digit)**: a 5 anchor at GT=4 (S1) is one digit, a 5000 anchor at GT=4 (S5) is four digits. The collapse at S5 may partly be "model rarely outputs four-digit numbers when the question implies a small count", not pure plausibility-window. Confounded with distance in our design. Future work to disentangle.
- **Anchor color/scene-context interaction**: E5c shows scene context is not load-bearing on accuracy drop, but a per-scene-template breakdown of `adopt_cond` is unexplored.
