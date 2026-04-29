# E4 — attention re-weighting mitigation prototype: design

**Status:** **Superseded by results writeup as of 2026-04-29.** Pre-implementation design (approved 2026-04-25) — preserved as the design record. Phase 1 + 2 complete; C-form re-aggregation landed 2026-04-28 (commit `fe33a9d`). Results: `docs/experiments/E4-mitigation.md` (writeup) + `docs/insights/E4-mitigation-evidence.md` (evidence) + `docs/insights/paper-section-7-4-mitigation-free-lunch.md` (paper §7.4 framing).

## Goal

Implement a working inference-time mitigation for cross-modal anchoring on the mid-stack-cluster VLMs (LLaVA-1.5, ConvLLaVA, InternVL3), guided by the E1d finding that `ablate_upper_half` is the single architecture-blind locus that reduces `direction_follow_rate` without breaking fluency.

**Target** (per roadmap §6): ≥ 10 % reduction in `direction_follow_rate` with ≤ 2 pp drop in standard VQA accuracy on the VQAv2 number subset.

## Why this design

E1d ruled out single-layer interventions (peak ablation, layer-0 control both null on 6/6). The remaining viable class is multi-layer attention re-weighting. E1d further ruled out lower-half ablation (3/6 BACKFIRES). Upper-half ablation at hard mask (−1e4) is the only mode that worked across the panel, and on the mid-stack cluster it was fluency-clean (mean_distance_to_anchor stayed within ~1 unit of baseline). E4 extends that result with two changes:

1. **Strength knob** — soft re-weighting at variable strengths instead of hard masking. The hard mask collapses anchor attention to ≈ 0; a softer mask down-scales it by `exp(strength)`. Probing the strength axis lets us trade direction-follow reduction for accuracy preservation, and find the operating point that meets the roadmap target.
2. **Accuracy metric** — E1d only tracked `direction_follow_rate` and `mean_distance_to_anchor`. E4's target is stated in standard-VQA-accuracy terms, so the script must record `exact_match` against ground truth.

## Phase 1 — Strength sweep (n=200, mid-stack 3 models)

### What

For each model in {LLaVA-1.5, ConvLLaVA, InternVL3}, run the E1b-stratified n=200 question set (top-decile-susceptible × 100 + bottom-decile-resistant × 100) under `target_only` / `target_plus_irrelevant_number` / `target_plus_irrelevant_neutral` × **7 strength values** of upper-half attention re-weighting:

| strength | post-softmax anchor-attention multiplier `exp(strength)` | reading |
|---:|---:|---|
| 0 | 1.000× | baseline (no mask) |
| −0.5 | 0.607× | gentle |
| −1.0 | 0.368× | moderate |
| −2.0 | 0.135× | strong |
| −3.0 | 0.050× | very strong |
| −5.0 | 0.0067× | near-zero |
| −1e4 | ≈ 0 | hard mask (E1d control) |

The mask intervention is implemented as a forward pre-hook on each LLM decoder layer in `[n_layers/2, n_layers)`, adding `strength` to the `attention_mask` columns at the anchor-image-token span. The math: `attention_mask` is added before softmax, so adding `s` to anchor columns multiplies post-softmax anchor-attention weight by `exp(s)`. Strength values are logarithmically spaced through the meaningful range — beyond `−5` the attention weight is already < 1 % of baseline.

### Metrics (per (model, strength, condition) cell, n=200)

- `direction_follow_rate` (carried over from E1d)
- `adoption_rate`
- `mean_distance_to_anchor` (fluency monitor)
- **`exact_match` rate** — `# pred_number == ground_truth ÷ valid triplets`. The standard-VQA-accuracy proxy used as the roadmap-target denominator. (Full VQAv2 10-annotator soft accuracy is not available for our subset; exact_match is the closest interpretable proxy.)
- All four with bootstrap 95 % CIs (2,000 iter).

### Output

Pareto plot (strength on x-axis, `direction_follow_rate` and `exact_match(target_plus_irrelevant_number)` on y-axis with separate scales). Pick the **smallest |strength|** that satisfies:

```
direction_follow_rate(target_plus_irrelevant_number, strength)
  ≤ 0.9 × direction_follow_rate(target_plus_irrelevant_number, strength=0)
AND
exact_match(target_plus_irrelevant_number, strength)
  ≥ exact_match(target_plus_irrelevant_number, strength=0) − 0.02
```

The accuracy criterion is evaluated on `target_plus_irrelevant_number` because that's the condition where the mitigation actually fires: the upper-half hook only modifies attention to the anchor span, and the anchor span is empty (=(0, 0)) on `target_only`. As a sanity check we also verify `exact_match(target_only, strength) ≈ exact_match(target_only, strength=0)` for every strength — any drift there would indicate the hook is leaking into single-image inference, which would be a bug.

If no strength meets both criteria, the design escalates: try (a) a denser strength grid, (b) `ablate_upper_quarter` (`[3n/4, n)`) — narrower band of layers, (c) per-model strength selection.

### Compute

Per model: 200 samples × 3 conditions × 7 strengths = 4,200 generations × ~0.5–1 s = 35–70 min. Three models sequential on GPU 0 = **~2–4 hours total**.

## Phase 2 — Full-scale validation (n=17,730, mid-stack 3 models, single optimal strength)

### What

Once Phase 1 picks an optimal strength `s*` for each model (or one shared strength if the per-model optima coincide), run the full VQAv2 number subset (n=17,730 sample-instances, 5 irrelevant sets per question) under three conditions × two modes (baseline vs. upper-half re-weighting at `s*`).

### Compute

Per model: 17,730 sample-instances × 3 conditions × 2 modes ≈ **106 k generations** × ~0.5–1 s ≈ **15–30 hours/model**. Three models sequential = **45–90 hours**.

### Resumability requirement

Phase 2 cannot reasonably complete in one uninterrupted session. The script must be resumable across kills/crashes/restarts:

- **Output structure**: `outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl` — one canonical file per (model, phase). All records appended.
- **Resume protocol**: on startup, the script reads the existing JSONL (if any), builds a set of completed `(sample_instance_id, condition, mode_strength)` keys, and skips those during iteration. It appends new records to the same file.
- **Crash safety**: each record is `fh.write(json.dumps(...) + "\n")` followed by `fh.flush()`. On kill, at most the last (still-being-written) line is partial. The reader uses a try/except around `json.loads(line)` and silently skips malformed last lines.
- **Run-once semantics**: re-running the same command is a no-op once the file is complete. Append-only writes mean a previously-completed run can be re-loaded without recomputation.
- **Verification at end**: script prints expected vs. actual record count at the end (e.g., "completed 106,380 / 106,380"). If short by ≥ 1 record at exit, prints the missing keys.

### Metrics (per (model, mode, condition))

Same four metrics as Phase 1, with bootstrap 95 % CIs over the full-scale n.

## Code structure

Two new scripts + one writeup pair.

- **`scripts/e4_attention_reweighting.py`** — based on `scripts/causal_anchor_ablation.py`'s hook/anchor-span/runner-build plumbing (imported, not duplicated). Adds:
  - `--strength FLOAT` (default `-1e4`); when not `0`, install upper-half hooks
  - `--phase {sweep,full}` controls sample-set source (n=200 stratified for `sweep`, full VQAv2 number subset for `full`) and output directory layout
  - Resumability (Phase 2 hard requirement; harmless overhead in Phase 1)
  - Records `parsed_number`, `ground_truth`, `exact_match` in JSONL alongside existing E1d fields
- **`scripts/analyze_e4_mitigation.py`** — aggregates per `(model, strength, condition)` cell with bootstrap CIs; produces Pareto plot, per-strength summary CSV, full-validation CSV; encodes the strength-selection rule from Phase 1 above and prints the chosen `s*`.
- **`docs/experiments/E4-mitigation.md`** (+ `_ko.md`) — written once Phase 1 yields the strength choice; updated with Phase 2 numbers when full validation lands. **This design doc (`E4-mitigation-design.md` + `_ko.md`) is superseded by the results writeup at that point**, but is kept in git history.

### Output directory

```
outputs/e4_mitigation/
  <model>/
    sweep_n200/
      predictions.jsonl     # Phase 1: 4,200 records (3 conditions × 7 strengths × 200)
    full_n17730/
      predictions.jsonl     # Phase 2: ~106k records (3 conditions × 2 modes × 17,730)
  _summary/
    sweep_pareto.csv
    sweep_pareto.png
    chosen_strength.json    # {model: chosen_strength_s*}
    full_validation.csv
    full_validation_summary.md
```

## Open questions to revisit at the writeup stage

- **ConvLLaVA inclusion in the paper-grade result.** ConvLLaVA was kept in Phase 1 and Phase 2 here, but its E1d sub-finding ("lower-half ablation behaviour is the *opposite* of LLaVA-1.5's, despite identical E1b peak/mechanism") flags a causal-structure caveat. If the Phase 2 numbers show ConvLLaVA's E4 response is also unstable or substantially different from LLaVA-1.5/InternVL3, **decide at writeup time whether to drop it** from the paper's headline mid-stack-cluster claim and demote it to a discussion caveat. Document the decision in `E4-mitigation.md`.
- **Per-model vs. shared optimal strength.** If Phase 1 picks similar `s*` across the 3 models (e.g., all in {−1, −2}), report a single shared strength as the architecture-blind prototype. If `s*` diverges meaningfully (e.g., LLaVA-1.5 at −0.5, InternVL3 at −3), per-model strength becomes part of the prototype spec — weakens the "one mitigation across encoders" claim, but doesn't kill it.
- **Failure escalation path.** If no strength in [−5, 0] meets the target on any model, Phase 1 results inform whether to (a) try `ablate_upper_quarter` instead of `ablate_upper_half`, (b) move to a different intervention class (contrastive decoding, vision-token re-projection — flagged by E1d as untested but plausible), or (c) accept the looser "≥ 5 pp" target and document.

## Roadmap update on completion

- §6 Tier 1 E4 row: status changes from `☐` to `✅` when Phase 2 completes.
- §10 changelog entry dated at completion time.
