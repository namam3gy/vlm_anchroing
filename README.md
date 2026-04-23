# VLM Cross-Modal Anchoring

Does an irrelevant image containing a number bias a VLM's numeric answer on
VQAv2 number questions? This project runs three conditions per sample across
several HuggingFace VLMs and reports anchor-adoption, direction-follow, and
numeric-distance metrics.

**Conditions**

- `target_only`
- `target_plus_irrelevant_number`
- `target_plus_irrelevant_neutral`

**Default models** (see `configs/experiment.yaml`)

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `google/gemma-3-27b-it`
- `google/gemma-4-E4B-it`
- `google/gemma-4-31B-it`
- `llava-hf/llava-interleave-qwen-7b-hf`

## Install

```bash
uv sync          # preferred
pip install -e . # alternative
```

## Quick start

```bash
# Full sweep
uv run python scripts/run_experiment.py --config configs/experiment.yaml

# Local smoke run
uv run python scripts/run_experiment.py \
  --config configs/experiment.yaml \
  --max-samples 5 \
  --models qwen2.5-vl-7b-instruct
```

## Inputs

- `inputs/vqav2_number_val/` — local VQAv2 snapshot (JSONL + images).
  Rebuild via `scripts/fetch_vqav2_number_val.py`.
- `inputs/irrelevant_number/{1..N}.png` — number-bearing distractor images.
  Generate via `scripts/generate_irrelevant_number_images.py` (sdxl-turbo).
- `inputs/irrelevant_neutral/{1..N}.png` — neutral distractor images.
  Generate via `scripts/generate_irrelevant_neutral_images.py`.

## Outputs

Each run writes under `outputs/<model>/`:

- `predictions.jsonl`, `predictions.csv`
- `summary.json`

Aggregate figures land in `outputs/analysis/`.

## Metrics

- Standard VQA accuracy and exact-match accuracy
- Accuracy drop vs `target_only`
- Anchor adoption rate
- Anchor-direction follow rate
- Mean numeric distance to anchor

## More

See [AGENTS.md](AGENTS.md) for architecture, data flow, conventions, and the
commands contributors and coding agents should use.
