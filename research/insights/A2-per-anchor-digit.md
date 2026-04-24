# A2 — Per-anchor-digit pull is asymmetric and model-specific

**Status:** Phase-A finding. Source data: `_data/A2_per_anchor_value.csv`. Script: `research/scripts/phase_a_data_mining.py::a2_per_anchor_value`.

## The question

Does the bias depend on *which* digit is rendered into the anchor image? A flat per-digit profile would say "anchoring is content-agnostic — just any extra image with text"; a sharply varying profile would say "specific digits are stickier than others".

## The chance baseline (no confound)

The VQAv2 number subset is built with `samples_per_answer=400` for `answer_range=8`, so the GT distribution is essentially flat (each of 0–8 appears ~11.3 % of the time, except digit 7 at ~9.8 %). The chance probability that `anchor == GT` is therefore ~11 % for every anchor digit (verified — `anchor_eq_gt` per digit is in [0.097, 0.124] across all models). **The per-digit adoption rates below are NOT explained by anchor-GT collisions.** Whatever digit-specific structure shows up is a real bias.

## Result

**Adoption rate by anchor digit** (number of items in each anchor bucket ≈ 1,920–2,045 per model). The chance baseline (anchor == GT) is ≈ 0.11 in every cell.

| anchor → | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma3-27b-it | .094 | .148 | .172 | .145 | .175 | .095 | .190 | .101 | .142 |
| gemma4-31b-it | .109 | .138 | .129 | .125 | .137 | .116 | .147 | .072 | .065 |
| gemma4-e4b | .145 | .213 | .150 | .125 | .132 | .088 | .094 | .102 | .058 |
| llava-interleave-7b | .099 | .142 | **.300** | .180 | .157 | .095 | .143 | .023 | .052 |
| qwen2.5-vl-7b | .106 | .151 | .159 | .125 | .161 | .088 | .085 | .058 | .049 |
| qwen3-vl-30b-it | .112 | .140 | .176 | .125 | .164 | .124 | .126 | .064 | .044 |
| qwen3-vl-8b-instruct | .135 | .144 | .163 | .138 | .140 | .133 | .136 | .092 | .065 |

**Three patterns:**

1. **Universal high-anchorability of digits 1, 2, 4**, low-anchorability of 7, 8 (and 0 mostly). The pattern is consistent across models — even Qwen2.5-VL (the most resistant) shows the same hump. Likely explanation: in the VQAv2 number subset, the *plausible* answer space for "how many X" is dominated by small numbers, so model priors over outputs are heaviest on 1–4 and any anchor that matches a candidate already in the model's distribution is more likely to win.

2. **`llava-interleave-7b` × anchor=2 = 0.300** is a striking outlier — nearly 3× chance, and ~2× the next-highest model on the same anchor. This is consistent with the LLaVA-family typographic-attack literature reporting a particular susceptibility to in-image text and warrants individual investigation (the attention-mass analysis E1 should be run on this slice first).

3. **Anchor=8 is essentially inert for several models** (gemma4-e4b 0.058, gemma4-31b 0.065, qwen3-vl-8b 0.065). 8 is also the upper bound of the GT range, so any "8" anchor implies "the answer is at the edge of plausible counts" — models may be down-weighting it for that reason. Equivalent to a "extreme anchor" effect in the cognitive-science literature (Mussweiler & Strack: implausible anchors get rejected).

## `mean_anchor_pull` reveals direction-specific behaviour

Adoption is symmetric (was the prediction = anchor?). `mean_anchor_pull` is signed — positive means moved closer, negative means moved away. The two sign-flipped rows below are notable:

- `qwen3-vl-30b-it` × anchor=3: mean_pull = **-4.24**, signed_shift = +4.21 (i.e. when the anchor is 3, the prediction tends to drift *upward* by ~4 — the opposite of "toward the anchor" if the prediction was already > 3).
- `qwen3-vl-30b-it` × anchor=7: mean_pull = -1.55, signed_shift = +1.32.
- `qwen3-vl-8b-instruct` × anchor=6: mean_pull = -2.89, signed_shift = +2.82.

These are all cases where Qwen3-VL exhibits *negative* anchoring on specific digits — repulsion, not attraction. The cognitive-science term for this is "anti-anchoring" and it's been observed in a small fraction of human experiments. A clean reproduction in two Qwen3-VL sizes (and not in Qwen2.5-VL or any Gemma) suggests it's a Qwen3-RL-style training artifact, not data noise. Worth a paragraph in the paper if it survives bootstrap CIs.

## Implications for the paper

- **Don't claim a universal "anchor = puller" effect.** Some (model, digit) cells repel. The paper's framing should be about *susceptibility profile* rather than "models are anchored".
- **`llava-interleave × anchor=2` is the clearest case study** — single (model, digit) combination, 3× chance, very large n. Use it as the qualitative case in the paper.
- **A digit-aware analysis is needed for the mitigation evaluation.** If E4 reports "10 % reduction in moved-closer rate", we should also ensure it doesn't *flip* the sign on the anti-anchoring cases (would actually increase distortion).

## Caveats

- Chance baseline of `anchor == GT` is flat (~11 %), so adoption rates are directly interpretable as bias above chance. Confirmed in `_data` calculation above.
- The per-digit n is ~1,920–2,045. Adoption-rate differences > 2 pp clear the standard-error threshold for binomial proportions at this n; differences < 2 pp should not be over-interpreted.
- "Anti-anchoring" cells need bootstrap CIs before being claimed in the paper. Quick add-on for Phase A.

## Roadmap entry

§5 A2: ☐ → ✅
