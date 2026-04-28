# §8. Discussion, limitations, and future work

## 8.1 Reasoning amplifies anchoring on a VLM (γ-β)

Single-pair single-dataset, but striking: Qwen3-VL-8B-Thinking on
MathVista shows adopt(a) ×1.6 and direction-follow C-form ×2.9
relative to its non-reasoning Instruct counterpart, with acc(b)
*lower* on Thinking (0.196 vs 0.216). The Thinking checkpoint is a
separately-trained model — same architecture, same chat template,
same 4-condition stimuli. The only delta is the trained reasoning
behaviour.

This aligns with recent text-only LRM literature [Wang et al.,
"Assessing Judging Bias in Large Reasoning Models",
arXiv:2504.09946, 2025; Nguyen et al., VLMBias on reasoning
models, arXiv:2505.23941, 2025]: reasoning trace can *amplify*
biases by giving the model more elaboration steps over which the
biased anchor information accumulates, without the trace itself
catching the bias.

We treat γ-β as a paper-tier hook (§1, abstract) but as an
**existence-proof at single pair × single dataset** for the
broader claim. Generalising it to a multi-model multi-dataset
reasoning panel is future work. Specifically:

- Add a second Qwen3-VL pair at 30B-A3B (instruct vs. thinking) to
  test scale-invariance.
- Add a different family's reasoning checkpoint (e.g.
  Gemma3-Thinking, GPT-4o reasoning-mode) on the same panel.
- Cross-dataset generalisation: γ-β-style on ChartQA / TallyQA, not
  just MathVista.

## 8.2 LLM/VLM architectural diff (preferred future work)

The text-only LLM-anchoring literature has measured anchoring on
the same numerical question, framed as text. The VLM here delivers
the anchor as an image. A clean methodological contribution would
be a paired comparison: same numerical question, same anchor value,
delivered (a) as text in a prompt-style template versus (b) as a
rendered-digit image. Layer-wise integration profiles (E1-style
attention mass + per-layer logit-lens) would tell us where the
two modalities of anchor delivery converge or diverge inside the
LLM stack.

This is the cleanest single follow-up paper — it sits in the gap
between this work (image anchor only) and the LLM-anchoring
literature (text anchor only).

## 8.3 Image-vs-text anchor on the same VLM

A weaker version of §8.2 that does not need text-only LLMs: on the
same VLM, present the anchor as a rendered image *or* as text
within the question prompt. Compare effect-size deltas. This
controls for the LLM and isolates the anchor-modality channel. We
release both stimulus inventories (rendered-digit images and
prompt-text equivalents) so others can pick this up.

## 8.4 Limitations

**Single-prompt runs.** All experiments use one JSON-strict prompt;
paraphrase robustness (3-5 prompt variants × bootstrap CIs) is
the next obvious hardening pass. Reviewers on prior cognitive-bias
LLM papers consistently flag this; we acknowledge it explicitly.

**Open weights only.** The seven main-panel models are all
open-weight. A closed-model defuse run (GPT-4o or Gemini 2.5 on
~500 stratified samples) would address the "only open models"
critique cheaply if we can secure access during revision.

**No human baseline.** The cognitive-science framing in §1 / §6
relies on prior literature (Tversky-Kahneman, Mussweiler-Strack,
Jacowitz-Kahneman). A 50-participant Prolific study replicating one
or two conditions would ground the analogy more directly. We did
not run this on the current ARR clock.

**Distance window depends on dataset.** The plausibility-window
reading in §5 is robust on VQAv2 and TallyQA but uses a relative
cutoff `|a − gt| ≤ max(1, 0.10·gt)` on ChartQA / MathVista
(validated empirically in E5d). The relative form is an inductive
choice that may need re-validation on datasets with very different
GT distributions.

**Mid-stack mitigation is single-cluster.** E4's free-lunch result
is established on three mid-stack-cluster models (LLaVA-1.5,
ConvLLaVA, InternVL3). Whether the same locus generalises to the
SigLIP-Gemma early peak or the Qwen-ViT late peak is open (E4
generalisation, P3 in our roadmap).

**γ-β single pair.** As noted in §8.1, the
reasoning-amplifies-anchoring result is one pair on one dataset.
Treat as existence proof, not as a quantitative law.

**Driver schema audit.** During this work we discovered a driver
schema gap that silently zeroed `direction_follow_rate` on
directly-driven `summary.json` files between the M1 and M2 metric
refactors. The audit + remediation (C-form refactor + reaggregate
sweep + before/after migration report) is documented in
`references/roadmap.md` §10 and
`docs/insights/C-form-migration-report.md`. We disclose this in
the spirit of full reproducibility; all numerical claims in this
paper trace to post-audit C-form re-aggregated outputs.

## 8.5 Conclusion

Cross-modal numerical anchoring is real on VLMs, gated by a
three-factor conjunction (uncertainty, plausibility, digit-pixel
visibility), mechanistically multi-layer-redundant, and mitigable
at the upper-half attention locus on a mid-stack-cluster of
encoder families with a measurable accuracy improvement on the
anchor condition. The Phase-A binary wrong/correct projection is a
coarse rendering of a continuous confidence gradient (§6), and a
reasoning-mode VLM amplifies rather than suppresses the effect on
MathVista (§7.5 / §8.1) — pointing at a research agenda where
reasoning trace becomes a place for biases to *accumulate*, not
a place for them to be caught.

The full code, configs, anchor inventories, per-sample predictions,
and aggregate CSVs (post-audit, C-form-aligned) are released for
replication.
