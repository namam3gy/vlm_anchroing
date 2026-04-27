# E5b — anchor-distance robustness sweep: design

**Status:** Pre-implementation design. Approved by user 2026-04-27. Sub-experiment of E5 (multi-dataset full runs); parallels the E1b/c/d sub-experiment naming under E1. Will be replaced by `docs/experiments/E5b-anchor-distance.md` (results writeup) once the run completes.

## Goal

Quantify how cross-modal anchoring strength varies with the absolute distance between the anchor digit and the question's ground-truth answer, and confirm the relationship is consistent across two datasets that share question form but differ in image domain (TallyQA — natural-photo counting; VQAv2 number subset — open-domain numeric VQA). The output is one figure (anchor effect vs distance bin) per dataset and a cross-dataset overlay.

**Decision impact.** The distance curve justifies (or blocks) a stratified anchor-sampling rule for paper headline figures: if effect decays sharply with distance, headline cells should report on the near-anchor subset. If effect is flat, anchor selection is not load-bearing and a single random-anchor methodology suffices for all paper sections.

## Why this design

A reviewer will challenge any anchor-effect claim with two questions: *(i) is the effect a feature of the specific anchor digits chosen, and (ii) does it survive scale changes (TallyQA goes higher than VQAv2)?* Both reduce to "does anchor distance relative to the plausible answer matter". E5b answers the first directly with a stratified distance sweep; the cross-dataset comparison answers the second.

Three candidate designs were considered:

1. **Uniform random sampling from the 0..10000 anchor inventory.** Rejected — 78 % of inventory items are at d > 30 from typical TallyQA/VQAv2 GT (≈ 0–3), starving the near-distance bins that carry the strongest signal.
2. **Per-question 5-nearest-to-prediction selection.** Rejected — concentrates all data in the d ≤ 5 regime, prevents the curve from being drawn, and biases toward the dependent variable. Reviewer-hostile.
3. **Per-question 5-stratum anchor sampling, 1 random anchor per stratum.** Selected — preserves random sampling within each stratum (defensible against "selection" attack) while guaranteeing equal coverage of every distance regime (efficient curve estimation).

## Experimental design

### Datasets and N

| Dataset | Subset rule | N (base questions) |
|---|---|---:|
| VQAv2 number val | `answer_range = 8`, `samples_per_answer = 100`, `max_samples = 500`, `require_single_numeric_gt = True`, GT ≤ 10000 | 500 |
| TallyQA test | `answer_type_filter = ["number"]`, `answer_range = 8`, `samples_per_answer = 100`, `max_samples = 500`, `require_single_numeric_gt = True`, GT ≤ 10000 | 500 |

`answer_range=8` retains GT classes 0..8 (9 classes total per `data.load_number_vqa_samples` filter at line 77). `samples_per_answer = 100` caps each class at 100 base questions; `max_samples = 500` is the hard total cap that terminates iteration once 500 base questions are admitted. Combined behaviour: classes are filled in dataset order until either the per-class cap (100) or total cap (500) fires — typically the total fires first so the actual class distribution reflects the dataset's natural ordering. N = exactly 500 base questions per dataset. N is reported in **base questions**, not sample-instances: in stratified mode each base question contributes one set of 5 anchors (one per stratum) plus one `target_only` baseline = 6 generations. Subset rule α (GT ≤ 10000) per user decision is a no-op at this filter (TallyQA/VQAv2 cap at GT ≤ 8) but is recorded explicitly so the same config applies cleanly when E5b is later extended to ChartQA/MathVista where GT can exceed 10000.

### Anchor sampling — 5 strata, GT-based reference

For each sample-instance with ground-truth answer `g`, draw 5 anchor values `(a₁..a₅)` independently from the inventory `inputs/irrelevant_number/{0..10000}.png` (128 PNGs), one from each stratum:

| Stratum | `|a − g|` range | role |
|---|---|---|
| **S1** | [0, 1] | near-peak (effect strongest if literature is right) |
| **S2** | [2, 5] | adjacent mid |
| **S3** | [6, 30] | intermediate decay |
| **S4** | [31, 300] | far |
| **S5** | [301, ∞) | very far / saturation tail |

Implementation: a new pure function in `vlm_anchor.data` (working name `sample_stratified_anchors`) takes `(gt: int, inventory: list[int], rng: random.Random)` and returns 5 stratified anchor values. With GT ∈ 0..8 and inventory `{0..10, 15, 20, ..., 100, 200, ..., 10000}` (128 values), every stratum has ≥ 4 inventory matches for every GT in this experiment, so the empty-stratum branch is logged-and-skipped but never fires. The branch is implemented anyway because future ChartQA/MathVista extensions may hit it.

### Conditions per sample

| Condition | Inputs | Anchor value |
|---|---|---|
| `target_only` | target image only | — |
| `target_plus_irrelevant_number_S1` | target + anchor PNG | `a₁` |
| `target_plus_irrelevant_number_S2` | target + anchor PNG | `a₂` |
| `target_plus_irrelevant_number_S3` | target + anchor PNG | `a₃` |
| `target_plus_irrelevant_number_S4` | target + anchor PNG | `a₄` |
| `target_plus_irrelevant_number_S5` | target + anchor PNG | `a₅` |

**Neutral arm dropped** for E5b (per user decision): the S5 condition (very far anchor) provides the same "anchor information ≈ 0" reference, and target_only already provides the "no second image" baseline.

Total: **6 generations per base question** (1 target_only + 5 anchor strata; no redundant target_only across strata).

In the existing pipeline, `assign_irrelevant_images` produces one sample-instance per `irrelevant_sets_per_sample` and `build_conditions` yields 3 conditions per sample-instance — leading to redundant `target_only` runs across replicas of the same base question. In `stratified=True` mode, each base question becomes a single sample-instance with a list-valued `anchor_strata` field (5 anchors); `build_conditions` then yields 6 conditions for that one sample-instance, and `irrelevant_sets_per_sample` from the YAML is ignored.

### Model

Single model: **`llava-interleave-7b`** (`llava-hf/llava-interleave-qwen-7b-hf`).

Rationale:
- Full 17,730-record VQAv2 main-run baseline already exists → distance curve can be sanity-checked against random-anchor adoption (0.134) and direction-follow (0.348) numbers.
- Highest 7B-class direction-follow in panel (0.348) → largest dynamic range for the distance curve.
- Multi-image native (interleave training) → the second-image anchor pathway is the model's design intent, not a stress mode.
- Already in the existing TallyQA smoke run config.

### Sampling

- `temperature = 0.0`, `top_p = 1.0`, `max_new_tokens = 8`. Greedy, JSON-only system prompt — identical to `experiment.yaml`.
- `seed = 42`.

## Driver and config changes

### `src/vlm_anchor/data.py`

Add:

```python
def sample_stratified_anchors(
    gt: int,
    inventory: list[int],
    rng: random.Random,
    strata: list[tuple[int, int]] = ANCHOR_DISTANCE_STRATA,
) -> list[int | None]:
    """Return one random anchor per stratum, or None if no inventory match."""
```

`ANCHOR_DISTANCE_STRATA` constant in the same module: `[(0, 1), (2, 5), (6, 30), (31, 300), (301, 10**9)]`.

Extend `assign_irrelevant_images` to accept a `stratified=True` flag. When set, anchor PNG paths are resolved from `sample_stratified_anchors` output (5 anchors per sample, one per stratum) instead of the existing uniform-random-from-pool path. Number-image filename convention: `inputs/irrelevant_number/{anchor_value}.png`.

### `src/vlm_anchor/data.py` — `build_conditions`

Extend to emit one condition per stratum when 5 anchor values are present, with condition names `target_plus_irrelevant_number_S{1..5}`. The neutral arm is omitted when `stratified=True`. Existing 3-condition behaviour (target_only / neutral / number) is preserved when `stratified=False` for backward compatibility with the main run.

### Configs

Two new files, mirroring `configs/experiment_tallyqa.yaml` style:

- `configs/experiment_distance_vqa.yaml`
- `configs/experiment_distance_tally.yaml`

Both set `inputs.anchor_sampling: stratified`, `inputs.distance_strata: [[0,1],[2,5],[6,30],[31,300],[301,1000000000]]`, `inputs.irrelevant_neutral_dir: null` (skip neutral), `models: [llava-next-interleaved-7b]`, identical sampling block. They differ only in `vqa_dataset.local_path` (`inputs/vqav2_number_val` vs `inputs/tallyqa_test`) and `samples_per_answer = 63`.

### Driver script — no new file

`scripts/run_experiment.py` is reused. The stratified flag is read from the YAML; the per-condition emission and per-record `anchor_stratum_id` field flow through the existing pipeline. `metrics.summarize_condition` is generic over condition name so it produces a per-stratum summary block automatically.

## Analysis and deliverables

### Notebook

`notebooks/E5b_anchor_distance.ipynb` — pandas analysis on the merged 6,000-record dataframe (500 samples × 6 conditions × 2 datasets):

1. **Per-stratum summary table**: `direction_follow_rate`, `adoption_rate`, `exact_match`, `mean_distance_to_anchor` per (dataset, stratum). 95 % CI via per-record bootstrap.
2. **Distance curve figure**: x = stratum midpoint (1, 3.5, 18, 165, 650), y = direction-follow gap vs target_only baseline. Two lines (TallyQA, VQAv2) with shaded CIs.
3. **Cross-dataset overlay**: same figure with both lines on one axis. Annotate the d* point — the largest stratum where effect remains > 50 % of S1 effect — as the candidate cutoff for paper headline subset.
4. **Sanity check**: VQAv2 S1+S2 pooled direction-follow vs the existing 17,730-record main-run direction-follow (0.348). The pooled ≈ main only if anchor-distance distribution matches; otherwise note the mismatch as expected.

### Writeups

- `docs/experiments/E5b-anchor-distance.md` (+ `_ko.md` mirror) — full experimental writeup once results land. Replaces this design doc as the canonical E5b reference.
- `docs/insights/E5b-anchor-distance-evidence.md` (+ `_ko.md` mirror) — distilled one-claim insight.

### Roadmap

- Update `references/roadmap.md` §3 (status), §6 Tier 2 (add E5b row), §10 changelog upon completion.
- Korean mirror updated in lockstep.

## Time estimate

- Per dataset: 500 samples × 6 generations = 3,000 records.
- Two datasets: 6,000 records total.
- llava-interleave-7b on 2-image inference: empirical wall ≈ 0.5 sec/record (mid-band of past panel; ChartQA was 0.12, VQAv2 main was ≈ 1.6 — interleave's two-image inference sits between).
- Expected wall: **~50 minutes total** (one model, both datasets sequential).

## Risks and caveats

- **TallyQA stratum coverage** — TallyQA's `answer_range=8` truncates GT to 0..8, identical to VQAv2 number. The cross-dataset comparison therefore tests **image-domain effects on anchoring**, not GT-scale effects (a true scale comparison needs ChartQA/MathVista, deferred). E5b headline must frame as "image-domain robustness", not "scale robustness".
- **Anchor inventory mid-range gap** — anchors in [10, 100] are spaced every 5 (not every 1), so for GT in 10..100 the d=1,2 resolution within S1 is lost. With TallyQA/VQAv2 GT mostly in 0..3, this affects no sample in this experiment but is logged for future ChartQA/MathVista extension.
- **S5 floor not zero** — S5 captures "anchor very far" but is not a true anchor-information-zero condition. If the paper later needs a strict zero-information control, the neutral arm has to be re-added in a follow-up. For this sanity check, target_only is the true zero-information baseline and is recorded.
- **Single-model claim scope** — results from llava-interleave-7b alone do not establish that the distance dependence generalises. The roadmap update will explicitly list "extension to mid-stack cluster (LLaVA-1.5 / ConvLLaVA / InternVL3)" as the natural follow-up; cost ≈ 3 × 50 min = 2.5 h sequential.
- **Driver back-compat** — the `stratified=True` flag must not change behaviour for any existing config. The smoke test (`uv run python -m pytest`) plus a 5-sample run on `configs/experiment.yaml` are the regression checks before launching the full sweep.

## Out of scope (deferred)

- **Phase B post-hoc bin on existing data** — user decision 2026-04-27: the analysis-side time saving is negligible compared to writing two methodologies into the paper, so no post-hoc bin tables are produced. All headline figures continue to use random-anchor 0..9 sampling from the existing main runs.
- **ChartQA/MathVista distance sweep** — anchor inventory cap=10000 already covers ChartQA GT ≤ 10000. Extension is a follow-up E5c if E5b confirms the distance dependence is real.
- **Pred-based reference point** — GT-based selected per user decision. Pred-based introduces (i) two-pass execution, (ii) per-model anchor divergence breaking item-paired analysis, (iii) reproducibility instability. Cognitive-framing motivation (uncertainty-modulated) is already covered by Phase A's A1 finding via a different stratification axis (target_only correctness).
