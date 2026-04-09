# VLM Cross-Modal Anchoring Project

This project evaluates whether irrelevant images bias VLM answers on VQAv2 number questions.

## Experimental conditions
- `target_only`
- `target_plus_irrelevant_number`
- `target_plus_irrelevant_neutral`

## Models
- Default config includes `google/gemma-4-E4B-it`
- Default config includes `google/gemma-3-27b-it`
- Default config includes `google/gemma-4-31B-it`
- Default config includes `Qwen/Qwen2.5-VL-7B-Instruct`
- Default config includes `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Default config includes `llava-hf/llava-interleave-qwen-7b-hf`

## Dataset
Default config uses the checked-in local snapshot at `inputs/vqav2_number_val_300`.
This snapshot was exported from `lmms-lab/VQAv2` validation data and filtered to `answer_type == "number"`.

## Input folders
Place your irrelevant images here:
- `inputs/irrelevant_number/1.png ... 100.png`
- `inputs/irrelevant_neutral/1.png ... 10.png`

To build another local VQAv2 snapshot under `inputs/`:
```bash
uv run python scripts/fetch_vqav2_number_val.py --output-dir inputs/vqav2_number_val_300 --max-samples 300
```

## Install
```bash
uv sync
# or
pip install -e .
```

## Pull models
```bash
ollama pull qwen3-vl:8b-instruct
ollama pull gemma4:e4b
```

## Generate irrelevant number images
The generator now defaults to a local Hugging Face `diffusers` backend instead of the hosted inference API.

Default local model:
- `stabilityai/sdxl-turbo`

Generate `1.png ... 100.png` under `inputs/irrelevant_number/`:
```bash
uv run python scripts/generate_irrelevant_number_images.py \
  --steps 1 \
  --guidance-scale 0 \
  --width 1024 \
  --height 1024
```

Notes:
- Use `uv run python ...` for Python entrypoints in this repository.
- If you want a FLUX-family model instead, pass `--model black-forest-labs/FLUX.1-schnell`. That model is gated on Hugging Face, so you need to accept the model access terms once and have `HF_TOKEN` or an active Hugging Face login available.
- If you want the old hosted API path, keep using `--backend hf-api`.

## Generate irrelevant neutral images
Generate `1.png ... 30.png` under `inputs/irrelevant_neutral/`:
```bash
uv run python scripts/generate_irrelevant_neutral_images.py --overwrite
```

The neutral-image prompts are designed to avoid numbers, letters, logos, and other readable text.

## Run
```bash
uv run python scripts/run_experiment.py --config configs/experiment.yaml --visualize
```

## Outputs
Each model writes:
- `outputs/<model>/predictions.jsonl`
- `outputs/<model>/summary.json`
- `outputs/<model>/attention_maps/*.png`

## Metrics
- standard VQA accuracy
- exact-match accuracy
- accuracy drop vs `target_only`
- anchor adoption rate
- anchor susceptibility gap vs `target_only`
- anchor-direction follow rate
- mean numeric distance to anchor

## Attention visualization note
For decoder-style VLMs, the default visualization backend is now **TAM-style answer-token activation mapping**:
- keep only the output token(s) that correspond to the numeric answer span
- project each target token class back onto prompt and image positions using generation-step hidden states
- reduce repeated-text interference with an Estimated Causal Inference style subtraction over earlier prompt and generated tokens
- respect pooled Qwen2.5-VL image-token layouts derived from `image_grid_thw` plus the processor merge size
- reshape the per-image token map, apply the TAM rank-Gaussian filter, and upsample it onto the original image for overlay

The legacy attention-head aggregation path is still available as an internal fallback when the TAM backend is unavailable for a given HF model.
