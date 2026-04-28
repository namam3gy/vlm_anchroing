# Repository Guide (humans & AI agents)

Single source of truth for contributors and coding agents working in this repo.
`CLAUDE.md` imports this file; no need to duplicate content there.

## What this project does

Evaluates **cross-modal anchoring bias** in VLMs: does showing an irrelevant
image containing a number bias a model's numeric answer? Up to four
conditions per `sample_instance`, with canonical short names used in code,
docs, and analysis:

| short | code label | second image |
|---|---|---|
| `b` | `target_only` | none (baseline) |
| `a` | `target_plus_irrelevant_number(_S?)` | image with one digit (the anchor) |
| `m` | `target_plus_irrelevant_number_masked(_S?)` | same anchor image, digit pixel inpainted out |
| `d` | `target_plus_irrelevant_neutral` | digit-free FLUX render (2-image-distraction control) |

Predictions are written `pred_b / pred_a / pred_m / pred_d`; ground truth is
`gt`; anchor value is `anchor`. Boolean flags use the prefix:
`pb_eq_a = (pred_b == anchor)`, `pa_ne_pb`, `gt_eq_a`, etc.

Canonical metrics (M2, settled 2026-04-28; see
`docs/insights/M2-metric-definition-evidence.md`):

```
adopt_rate            = #(pa == anchor AND pb != anchor) / #(pb != anchor)
direction_follow_rate = #( (pa-pb)·(anchor-pb) > 0  AND  pa != pb )
                        / #(numeric pair AND anchor present)
exact_match           = #(pa == gt) / #(numeric pair)
anchor_effect_M       = M(a-arm) - M(d-arm)
```

Same form on the m-arm (substitute `pred_m`).

## Layout

```
configs/             experiment YAMLs (entry point for runs)
docs/
  experiments/       experiment plans + result writeups (E*-*.md)
  insights/          Phase-A data-mining findings (A*-*.md) + _data/ numeric artifacts
  figures/           exported figures for the paper / writeups
inputs/              (gitignored) local VQAv2 snapshot + irrelevant number/neutral images
notebooks/           post-run analysis (imports from vlm_anchor.analysis)
outputs/             (gitignored) per-run predictions, summaries, attention maps
references/
  project.md         canonical research plan
  roadmap.md         active research roadmap built from project.md
  AGENTS.md          this file — repo guide for humans and coding agents
scripts/             CLI entrypoints (experiment driver, data fetchers, image generators, phase-A analysis)
src/vlm_anchor/      library code
tests/               unittest-based tests, run via pytest
```

`references/` is the reading list: start every task by consulting
`project.md §0` (current paper outline as of 2026-04-28; the 2026-04-23
candid feasibility review is preserved below the divider in the same file)
and `roadmap.md` (paper-section-anchored operational plan). Update
`roadmap.md` §3 (status snapshot) / §6 (per-section experiment matrix) /
§7 (priority queue) / §10 (changelog) as work progresses.

## Setup

```bash
uv sync          # preferred
pip install -e . # alternative
```

All Python entrypoints run through `uv run python ...`.

## Commands

| Task | Command |
|---|---|
| Full experiment | `uv run python scripts/run_experiment.py --config configs/experiment.yaml` |
| Smoke run | `uv run python scripts/run_experiment.py --config configs/experiment.yaml --max-samples 5 --models qwen2.5-vl-7b-instruct` |
| Tests | `uv run python -m pytest` |
| Refresh VQAv2 snapshot | `uv run python scripts/fetch_vqav2_number_val.py --output-dir inputs/vqav2_number_val --max-samples 300` |
| Generate number distractors | `uv run python scripts/generate_irrelevant_number_images.py --steps 1 --guidance-scale 0 --width 1024 --height 1024` |
| Generate neutral distractors | `uv run python scripts/generate_irrelevant_neutral_images.py --overwrite` |

The smoke run is the minimum validation after changes to the inference or data
pipeline.

## Architecture

### Core package `src/vlm_anchor/`

| Module | Responsibility |
|---|---|
| `data.py` | Load the VQAv2 JSONL snapshot, assign irrelevant images, emit per-condition dicts via `build_conditions()` |
| `models.py` | `HFAttentionRunner` wrapping `AutoModelForImageTextToText` with a `generate_number()` helper |
| `metrics.py` | Per-sample evaluation (`evaluate_sample`) and condition/experiment summarization. M2 canonical adopt + direction_follow definitions live here (refactor pending — `docs/insights/M2-metric-definition-evidence.md`) |
| `visualization.py` | Experiment-result chart rendering (`save_experiment_analysis_figures`) |
| `analysis.py` | Post-experiment pandas analysis + plotting (matplotlib/plotly/seaborn), used by notebooks |
| `utils.py` | File I/O, number parsing (`extract_first_number`, `normalize_numeric_text`), YAML/seed helpers |

### Data flow

```
configs/experiment.yaml
  → load_number_vqa_samples()   # reads inputs/<snapshot>/questions.jsonl + images
  → assign_irrelevant_images()  # adds irrelevant_number_image / irrelevant_neutral_image fields
  → build_conditions()          # yields up to 4 condition dicts per sample-instance (b/a/m/d)
  → HFAttentionRunner.generate_number()
  → evaluate_sample()           # → VQASampleEval
  → outputs/<model>/predictions.{jsonl,csv} + summary.json
  → save_experiment_analysis_figures() → outputs/analysis/*.png
```

### Config (`configs/*.yaml`)

All paths are resolved relative to the project root.
- `vqa_dataset.local_path` — local JSONL snapshot (no remote fetch at run time).
- `vqa_dataset.answer_range` / `samples_per_answer` — stratification filters applied at load.
- `inputs.irrelevant_sets_per_sample` — (number, neutral) pairs per question.
- `models[].hf_model` — HF model id; loaded by `HFAttentionRunner`.
- `sampling.{temperature, top_p, max_new_tokens}` — passed through to `model.generate(...)`.

### Post-run analysis

`analysis.py` is the notebook-facing API. Main entry points:

- `load_experiment_records(output_root)` — DataFrame built from per-model `predictions.csv`.
- `build_paired_dataframe(records_df)` — wide format with base/number/neutral side-by-side.
- `make_root_aggregate_summary(output_root)` — one-line summary per model.
- `summarize_compare_roots([root1, root2])` — compare across experiment configs.

## Coding conventions

- Target Python 3.10+, 4-space indent, `snake_case` for modules and functions, `PascalCase` for classes and dataclasses.
- Explicit type hints on public functions. Prefer `pathlib.Path` over raw strings.
- Import order: stdlib → third-party → `vlm_anchor.*`.
- No formatter/linter is configured — match surrounding style and avoid reformatting untouched code.

## Testing

Tests live in `tests/` and use `unittest`. Run them with `uv run python -m pytest`.
The package is pip-installable, so tests import from `vlm_anchor.*` directly (no
`sys.path` manipulation needed). For pipeline-level changes, follow a green test
run with a `--max-samples 5` smoke run and verify `outputs/<model>/summary.json`.

## Data & artifact hygiene

- `inputs/` and `outputs/` are git-ignored. Do not commit snapshots, generated
  distractor images, predictions, or attention maps.
- Large generated artifacts should land under `outputs/`, never back into `src/`.
- Keep `configs/*.yaml` aligned with available assets under `inputs/`.

## Commits & PRs

- Use short imperative subjects (e.g., `Add smoke-run limit for visualization`).
- PRs: describe experiment impact, list commands run, note config/dataset changes,
  and include sample output paths or plot screenshots when visualization changes.
