# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/vlm_anchor/`. Keep data loading in `data.py`, model backends in `models.py`, metrics in `metrics.py`, visualization helpers in `visualization.py`, and shared parsing or file utilities in `utils.py`. Use `scripts/run_experiment.py` as the main entrypoint for experiments, and keep reusable logic out of scripts when possible. Runtime settings belong in `configs/experiment.yaml`. The `inputs/` tree contains checked-in VQAv2 subsets plus user-supplied irrelevant images; generated artifacts should go to `outputs/`, not back into `src/`.

## Build, Test, and Development Commands
Install dependencies with `uv sync` or `pip install -e .`.
Run Python entrypoints with `uv run python ...` in this repository.

Run the full experiment with:
```bash
uv run python scripts/run_experiment.py --config configs/experiment.yaml --visualize
```

Use a smaller local check while developing:
```bash
uv run python scripts/run_experiment.py --config configs/experiment.yaml --max-samples 5 --models qwen3-vl-8b-instruct
```

Refresh a local VQAv2 number-only snapshot with:
```bash
uv run python scripts/fetch_vqav2_number_val.py --output-dir inputs/vqav2_number_val_300 --max-samples 300
```

Pull required Ollama models before running inference:
```bash
ollama pull qwen3-vl:8b-instruct
ollama pull gemma4:e4b
```

## Coding Style & Naming Conventions
Target Python 3.10+ and follow the existing style: 4-space indentation, module-level `snake_case`, class and dataclass names in `PascalCase`, and explicit type hints on public functions. Prefer `pathlib.Path` over raw path strings in new code. Group imports as standard library, third-party, then local modules. No formatter or linter is configured in this snapshot, so keep edits consistent with the surrounding file instead of reformatting unrelated code.

## Testing Guidelines
There is no committed `tests/` directory or coverage gate yet. Treat a smoke run as the minimum check: execute `uv run python scripts/run_experiment.py` with a small `--max-samples` value and verify `outputs/<model>/summary.json` plus any attention maps you changed. For logic-heavy additions, add focused unit tests in a new `tests/` package using `test_<module>.py` naming and run them with `uv run pytest`.

## Commit & Pull Request Guidelines
This workspace does not include `.git` history, so no repository-specific commit convention can be verified here. Use short imperative commit subjects such as `Add smoke-run limit for visualization`. PRs should describe the experiment impact, list commands run, note config or dataset changes, and include sample output paths or screenshots when plots or attention panels change.

## Data & Configuration Notes
The default experiment reads from the local snapshot at `inputs/vqav2_number_val_300`, not a remote Hugging Face dataset at runtime. Keep `configs/experiment.yaml` aligned with available local assets, especially `inputs/irrelevant_number` and `inputs/irrelevant_neutral`. Do not commit large generated outputs or unreviewed image assets by default.
