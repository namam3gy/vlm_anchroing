# §2. Related work

## 2.1 LLM anchoring lineage

The cognitive-anchoring effect is one of the most-replicated findings
in heuristics-and-biases research [Tversky-Kahneman 1974;
Mussweiler-Strack's selective accessibility model, 1997;
Jacowitz-Kahneman's plausibility-window account, 1995]. Its first
LLM-side investigation came from Jones-Steinhardt [NeurIPS 2022], who
documented anchoring along with availability and representativeness in
GPT-3. Echterhoff et al. [EMNLP Findings 2024] expanded the panel to
13,465 prompts × 5 biases × 4+ models and proposed a debiasing method.
Lou-Sun [2024] partitioned by model strength and reported that stronger
LLMs anchor *more*, not less. "Understanding the Anchoring Effect of
LLM" [arXiv:2505.15392, 2025] introduced the A-Index and R-Error
metrics that we adopt as comparison points. Recent work on reasoning
models [Wang et al., "Judging Bias in Large Reasoning Models", 2025]
shows that reasoning-mode LRMs can be *more* susceptible to several
biases than their non-reasoning counterparts — a result we confirm
on a VLM in §7-§8. All of these are text-only.

## 2.2 Multimodal cognitive-bias evaluation

VLMBias [Vo, Nguyen et al., arXiv:2505.23941, 2025] is the most
comprehensive multimodal-bias benchmark to date. They probe whether
memorized-subject labels (e.g. "Aston Martin DB5") cause counterfactual
counting failures on images containing the named subject. AIpsych
[Liu et al., arXiv:2507.03123, 2025] is a psychology-grounded VLM
cognitive-bias benchmark; it focuses on sycophancy, appeal-to-authority,
and logical-consistency biases. CIVET [EMNLP Findings 2025]
systematises position-understanding evaluation. Tinted Frames
[Fan et al., arXiv:2603.19203, 2026] varies question-form framing in
VLMs.

**None of the above measures cross-modal numerical anchoring.** VLMBias
is the closest neighbour: both inject visual content that biases the
answer. The differences are load-bearing — VLMBias uses memorized-subject
*labels* (semantic cues with known referents in training data), and
measures *classification flips* on counterfactual scenes. We use a
*single rendered digit with no semantic label* and measure
*regression-style numerical shift* on open-numeric VQA.

## 2.3 Typographic attacks and FigStep

Typographic-attack work [Goh et al., "Multimodal Neurons", 2021;
Wang-Zhao-Larson, NAACL 2025, arXiv:2502.08193; Hufe et al.,
"Dyslexify", arXiv:2508.20570, 2025] shares one mechanism with our
setup: in-image rendered text activates concept neurons in the vision
encoder. The key differences:
(a) typographic attacks paste *class-label text* on the target image
to flip a classifier; we paste a *single numerical value* on a *separate*
irrelevant image alongside the target.
(b) typographic attacks measure ASR (attack success rate, classification
flip); we measure regression-style numeric shift with cognitive-science
grounding.
(c) typographic attacks have no analogue of digit-pixel inpainting as a
causal control; we add the masked arm specifically to isolate the digit
pixels from the anchor scene's background distraction.

FigStep [Gong et al., AAAI 2025] renders harmful instructions as an
image to bypass safety-tuned LLMs. It shares the rendered-text-as-image
mechanism but targets jailbreaking, not numeric estimation.

## 2.4 Mechanism-localising VLM work

Goh et al.'s "Multimodal Neurons" [2021] is the canonical "in-image
text → concept-neuron activation" reference. Hufe et al.'s
"Dyslexify" [arXiv:2508.20570, 2025] localises typographic-attack
susceptibility to a circuit of attention heads in the latter half of
CLIP's vision encoder, and proposes head ablation as a mitigation.
Weng et al.'s "Images Speak Louder than Words" [EMNLP 2024 Main] is
the gold-standard "behavioural finding + causal-mediation analysis +
22 % mitigation" template — what an EMNLP-Main-tier paper looks like
for a multimodal behavioural finding. We follow that template here
(§5 behavioural / §7 attention-locus / §7 mitigation). Our §7 result
that **upper-half** attention re-weighting on the LLM stack reduces
direction-follow on 6/6 archetypes is consistent in spirit with
Dyslexify's "later-half" CLIP finding, suggesting that the encoder's
late-stack typographic circuit propagates into the LLM stack's upper
half along the same depth axis.

## 2.5 What this paper adds, in one paragraph

The cross-modal numerical anchoring effect — anchor as a stand-alone
rendered-number image, target as a semantically-unrelated VQA image,
metric as gt-free baseline-relative regression-style shift toward the
anchor — is, to the best of our knowledge, unmeasured prior to this
work. Beyond establishing the effect on seven VLMs, the paper
contributes (i) a digit-pixel-causality control via OpenCV-inpainted
mask anchors, (ii) attention-mass localisation across four
encoder-family archetypes followed by causal-ablation null at any
single layer, (iii) a "free-lunch" mid-stack-cluster mitigation that
reduces direction-follow without hurting accuracy, and (iv) an empirical
demonstration on Qwen3-VL that reasoning-mode *amplifies* rather than
suppresses anchoring on a VLM, mirroring recent text-only
LRM-judging findings.
