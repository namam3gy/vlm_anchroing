# E4 attention re-weighting mitigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate an inference-time anchor-attention re-weighting mitigation on the mid-stack-cluster VLMs (LLaVA-1.5, ConvLLaVA, InternVL3), per design `docs/experiments/E4-mitigation-design.md`.

**Architecture:** Two-phase. Phase 1 sweeps mask strength ∈ {0, −0.5, −1, −2, −3, −5, −1e4} at n=200 stratified samples on the upper-half LLM layers, picks an optimal strength per model satisfying ≥10 % direction-follow drop with ≤2 pp accuracy drop. Phase 2 validates the chosen strength on the full VQAv2 number subset (n=17,730) with hard resumability across kills/restarts. Reuses the forward-pre-hook attention-mask machinery from E1d's `scripts/causal_anchor_ablation.py`; does NOT modify E1d's script except to thread a `strength` parameter through the hook factory.

**Tech Stack:** Python 3.10+, PyTorch (HuggingFace transformers eager-attention path), pandas + matplotlib for analysis, `unittest` for tests, `uv` for the runtime, GPU 0 only (per user constraint).

---

## File Structure

| File | Purpose | Status |
|---|---|---|
| `scripts/causal_anchor_ablation.py` | E1d driver — refactor `_make_anchor_mask_hook` and `_install_hooks` to accept a `strength` kwarg, default `-1e4` (E1d behaviour preserved). | Modify (Task 1) |
| `scripts/e4_attention_reweighting.py` | E4 driver — argparse, sample loading per phase, strength loop, resumability, JSONL writer with `exact_match`. Imports plumbing from `causal_anchor_ablation`. | Create (Tasks 2–6) |
| `scripts/analyze_e4_mitigation.py` | E4 analysis — per-(model, strength, condition) bootstrap CIs, Pareto plot, strength-selection rule, full-validation summary. | Create (Task 7) |
| `tests/test_e4_helpers.py` | Unit tests for the strength-parametrised hook, the resumed-keys loader, and the strength-selection rule. | Create (Tasks 1, 4, 7) |
| `outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl` | Phase 1 raw output. Same shape as E1d JSONL plus `mask_strength` and `exact_match` columns. | Create at runtime (Task 8) |
| `outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl` | Phase 2 raw output. | Create at runtime (Task 10) |
| `outputs/e4_mitigation/_summary/{sweep_pareto.csv,sweep_pareto.png,chosen_strength.json,full_validation.csv,full_validation_summary.md}` | Analysis aggregates. | Create at runtime (Tasks 9, 11) |
| `docs/experiments/E4-mitigation.md` (+ `_ko.md`) | Results writeup. Replaces the design doc once Phase 2 lands. | Create (Task 11) |
| `references/roadmap.md` (+ `_ko.md`) | §6 E4 row → ✅, §10 changelog entry. | Modify (Task 12) |

---

## Task 1: Refactor `_make_anchor_mask_hook` to accept `strength` kwarg

**Files:**
- Modify: `scripts/causal_anchor_ablation.py:94-132` (`_make_anchor_mask_hook`)
- Modify: `scripts/causal_anchor_ablation.py:135-147` (`_install_hooks`)
- Test: `tests/test_e4_helpers.py`

**Why this task first:** E4's strength knob requires the hook to take a `strength` parameter instead of a hardcoded `-1e4`. Doing this in the existing E1d script (with a `strength=-1e4` default) keeps E1d behaviour byte-identical and lets E4 import the same machinery.

- [ ] **Step 1: Write the failing test**

Create `tests/test_e4_helpers.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from causal_anchor_ablation import _make_anchor_mask_hook  # noqa: E402


class AnchorMaskHookStrengthTests(unittest.TestCase):
    def _apply_hook(self, mask: torch.Tensor, anchor_span, strength: float) -> torch.Tensor:
        hook = _make_anchor_mask_hook(anchor_span, strength=strength)
        self.assertIsNotNone(hook)
        # Mock a forward call: pass the mask via kwargs (LLaMA-family signature)
        _, kwargs = hook(None, (), {"attention_mask": mask})
        return kwargs["attention_mask"]

    def test_default_strength_matches_e1d_negative_1e4(self) -> None:
        mask = torch.zeros(1, 1, 4, 8)  # (B, H, Q, K), 4D
        out = self._apply_hook(mask, (2, 5), strength=-1e4)
        # Anchor span columns 2..4 should be -1e4; others 0
        self.assertTrue(torch.allclose(out[..., 2:5], torch.full((1, 1, 4, 3), -1e4)))
        self.assertTrue(torch.allclose(out[..., :2], torch.zeros((1, 1, 4, 2))))
        self.assertTrue(torch.allclose(out[..., 5:], torch.zeros((1, 1, 4, 3))))

    def test_soft_strength_minus_one_adds_minus_one(self) -> None:
        mask = torch.zeros(1, 1, 4, 8)
        out = self._apply_hook(mask, (2, 5), strength=-1.0)
        self.assertTrue(torch.allclose(out[..., 2:5], torch.full((1, 1, 4, 3), -1.0)))
        # exp(-1) ≈ 0.368 — verify by simulating softmax effect
        scores = torch.zeros(1, 1, 4, 8)
        post = torch.softmax(scores + out, dim=-1)  # add hook-modified mask
        anchor_share = post[..., 2:5].sum(dim=-1)
        # Without mask, each column = 1/8; anchor (3 cols) = 3/8 = 0.375
        # With mask=-1, anchor weight scales by exp(-1) before renormalise
        # Closed form: 3*exp(-1) / (5 + 3*exp(-1)) ≈ 0.181
        expected = 3 * torch.exp(torch.tensor(-1.0)) / (5 + 3 * torch.exp(torch.tensor(-1.0)))
        self.assertTrue(torch.allclose(anchor_share, expected.expand_as(anchor_share), atol=1e-5))

    def test_zero_strength_returns_none_hook(self) -> None:
        # strength=0 means "no mitigation" — production code skips installation; hook factory may return a no-op
        # We accept either: hook returns the input unchanged, OR the factory returns None.
        hook = _make_anchor_mask_hook((2, 5), strength=0.0)
        if hook is None:
            return  # acceptable
        mask = torch.zeros(1, 1, 4, 8)
        _, kwargs = hook(None, (), {"attention_mask": mask})
        self.assertTrue(torch.allclose(kwargs["attention_mask"], mask))

    def test_empty_anchor_span_returns_none(self) -> None:
        self.assertIsNone(_make_anchor_mask_hook((0, 0), strength=-1.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_e4_helpers.py -v`

Expected: FAIL — current `_make_anchor_mask_hook` has signature `(anchor_span)` only and uses a hardcoded `NEG = -1e4`. Tests with `strength=-1.0` and `strength=0.0` will error (unexpected kwarg) or produce wrong output.

- [ ] **Step 3: Refactor `_make_anchor_mask_hook` to accept `strength`**

Edit `scripts/causal_anchor_ablation.py`. Replace the existing `_make_anchor_mask_hook` (lines ~94–132):

```python
def _make_anchor_mask_hook(anchor_span: tuple[int, int], strength: float = -1e4):
    """Return a forward pre-hook that adds `strength` to `attention_mask` columns
    inside the anchor span, so post-softmax attention to those keys is multiplied
    by exp(strength). strength=-1e4 (default) is hard masking (E1d behaviour).
    Smaller |strength| values give graded down-weighting.

    Returns None if the anchor span is empty (single-image conditions) or if
    strength == 0 (no-op intervention — caller should skip installation).
    """
    s, e = anchor_span
    if e <= s:
        return None
    if strength == 0:
        return None  # no-op: don't install a hook

    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor) and a.dim() == 4:
                    args = list(args)
                    a = a.clone()
                    a[..., s:e] = a[..., s:e] + strength
                    args[i] = a
                    return tuple(args), kwargs
            return None
        mask = mask.clone()
        if mask.dim() == 4:
            mask[..., s:e] = mask[..., s:e] + strength
        elif mask.dim() == 2:
            mask[..., s:e] = mask[..., s:e].masked_fill(mask[..., s:e] >= 0, strength)
        else:
            return None
        kwargs["attention_mask"] = mask
        return args, kwargs

    return hook
```

Also update `_install_hooks` signature (around line 135):

```python
def _install_hooks(layers: nn.ModuleList, layer_indices: list[int],
                   anchor_span: tuple[int, int], strength: float = -1e4):
    handles = []
    if anchor_span is None or anchor_span[1] <= anchor_span[0]:
        return handles
    if strength == 0:
        return handles
    hook = _make_anchor_mask_hook(anchor_span, strength=strength)
    if hook is None:
        return handles
    for i in layer_indices:
        if i < 0 or i >= len(layers):
            continue
        h = layers[i].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(h)
    return handles
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_e4_helpers.py::AnchorMaskHookStrengthTests -v`

Expected: 4 PASSED.

- [ ] **Step 5: Verify E1d behaviour is preserved with a smoke run**

Run: `CUDA_VISIBLE_DEVICES=0 uv run python scripts/causal_anchor_ablation.py --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf --peak-layer 16 --modes baseline,ablate_peak --max-samples 3 2>&1 | tail -10`

Expected: setup messages, "[done] 3 samples processed", no errors. Output dir has a `predictions.jsonl` with 3 × 3 × 2 = 18 records.

- [ ] **Step 6: Commit**

```bash
git add scripts/causal_anchor_ablation.py tests/test_e4_helpers.py
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "Parametrise causal_anchor_ablation hook with mask strength

E4 needs soft re-weighting (variable mask strength) where E1d only
needed hard masking. Default strength=-1e4 preserves E1d behaviour
byte-for-byte; smaller |strength| values down-scale anchor attention
by exp(strength) instead of zeroing it. Adds tests/test_e4_helpers.py
covering hard-mask equivalence and soft-mask softmax behaviour.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Skeleton of `scripts/e4_attention_reweighting.py`

**Files:**
- Create: `scripts/e4_attention_reweighting.py`

- [ ] **Step 1: Create the skeleton with argparse and imports**

```python
"""E4 — attention re-weighting mitigation.

Two phases share the same script:
  --phase sweep  : 7 strengths × n=200 stratified samples × 3 conditions
  --phase full   : 1 strength × full VQAv2 number subset × 3 conditions × {baseline, mode}

Resumable: re-running the same command picks up where the previous one stopped
by reading the existing predictions.jsonl and skipping completed
(sample_instance_id, condition, mask_strength) keys.

Usage (Phase 1, single model):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \\
        --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf \\
        --phase sweep

Usage (Phase 2, after Phase 1 picks s*):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \\
        --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf \\
        --phase full --strength -1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from vlm_anchor.data import (
    assign_irrelevant_images,
    build_conditions,
    load_number_vqa_samples,
)
from vlm_anchor.models import InferenceConfig
from vlm_anchor.utils import set_seed

# Re-use plumbing from E1 + E1d
sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_anchor_ablation import (  # noqa: E402
    _get_llm_layers,
    _install_hooks,
    _resolve_anchor_span,
)
from extract_attention_mass import (  # noqa: E402
    EagerAttentionRunner,
    _resolve_image_token_id,
    _select_susceptibility_strata,
    build_eager_runner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SWEEP_STRENGTHS: list[float] = [0.0, -0.5, -1.0, -2.0, -3.0, -5.0, -1e4]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--phase", choices=("sweep", "full"), required=True)
    parser.add_argument("--strength", type=float, default=None,
                        help="Required for --phase full; ignored for --phase sweep "
                             "(which iterates the canonical SWEEP_STRENGTHS list).")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--susceptibility-csv",
                        default="docs/insights/_data/susceptibility_strata.csv")
    parser.add_argument("--top-decile-n", type=int, default=100)
    parser.add_argument("--bottom-decile-n", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples (smoke testing). Phase 1 already capped at n=200.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.phase == "full" and args.strength is None:
        parser.error("--phase full requires --strength")
    return args


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    print(f"[setup] phase={args.phase} model={args.model} strength={args.strength}")
    # Concrete logic added in Tasks 3–6.


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

Run: `uv run python scripts/e4_attention_reweighting.py --model llava-1.5-7b --hf-model x --phase sweep` (will print `[setup]` and exit cleanly)

Expected: prints `[setup] phase=sweep model=llava-1.5-7b strength=None` and exits 0.

- [ ] **Step 3: Verify error path**

Run: `uv run python scripts/e4_attention_reweighting.py --model llava-1.5-7b --hf-model x --phase full`

Expected: argparse error "--phase full requires --strength", exit 2.

- [ ] **Step 4: Commit**

```bash
git add scripts/e4_attention_reweighting.py
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "Add E4 mitigation driver skeleton (argparse + imports)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Sample loading per phase

**Files:**
- Modify: `scripts/e4_attention_reweighting.py`

- [ ] **Step 1: Add `_load_samples` function**

Insert above `main()` in `scripts/e4_attention_reweighting.py`:

```python
def _load_samples(args: argparse.Namespace, config: dict) -> list[dict]:
    """Load and enrich samples per phase.

    sweep: same n=200 stratified set as E1b/E1d (top decile susceptible × 100 +
           bottom decile resistant × 100), 1 irrelevant variant per sample.
    full:  full VQAv2 number subset per configs/experiment.yaml; uses the
           configured `irrelevant_sets_per_sample`.
    """
    vqa_cfg = config["vqa_dataset"]
    inputs_cfg = config["inputs"]
    samples = load_number_vqa_samples(
        dataset_path=PROJECT_ROOT / vqa_cfg["local_path"],
        max_samples=None,
        require_single_numeric_gt=vqa_cfg.get("require_single_numeric_gt", True),
        answer_range=vqa_cfg.get("answer_range"),
        samples_per_answer=vqa_cfg.get("samples_per_answer"),
        answer_type_filter=vqa_cfg.get("answer_type_filter"),
    )
    if args.phase == "sweep":
        susc_path = PROJECT_ROOT / args.susceptibility_csv
        target_qids = _select_susceptibility_strata(
            susc_path, args.top_decile_n, args.bottom_decile_n, args.seed
        )
        samples = [s for s in samples if int(s["question_id"]) in target_qids]
        variants = 1
    else:  # full
        variants = inputs_cfg.get("irrelevant_sets_per_sample", 5)

    enriched = assign_irrelevant_images(
        samples,
        irrelevant_number_dir=PROJECT_ROOT / inputs_cfg["irrelevant_number_dir"],
        irrelevant_neutral_dir=PROJECT_ROOT / inputs_cfg["irrelevant_neutral_dir"],
        seed=args.seed,
        variants_per_sample=variants,
    )
    if args.max_samples:
        enriched = enriched[: args.max_samples]
    return enriched
```

- [ ] **Step 2: Wire into main**

Replace the `main()` body:

```python
def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    enriched = _load_samples(args, config)
    print(f"[setup] phase={args.phase} model={args.model} "
          f"sample_instances={len(enriched)} strength={args.strength}")
```

- [ ] **Step 3: Smoke test phase=sweep loads ~200 samples**

Run: `uv run python scripts/e4_attention_reweighting.py --model llava-1.5-7b --hf-model x --phase sweep --max-samples 200`

Expected: `[setup] phase=sweep model=llava-1.5-7b sample_instances=200 strength=None`. Exit 0.

- [ ] **Step 4: Smoke test phase=full loads many more**

Run: `uv run python scripts/e4_attention_reweighting.py --model llava-1.5-7b --hf-model x --phase full --strength -1 --max-samples 50`

Expected: `[setup] phase=full ... sample_instances=50 ...`. Exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/e4_attention_reweighting.py
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "E4: load susceptibility-stratified n=200 (sweep) or full subset (full)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Resumability — completed-key loader and append-mode writer

**Files:**
- Modify: `scripts/e4_attention_reweighting.py`
- Modify: `tests/test_e4_helpers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_e4_helpers.py`:

```python
import json
import tempfile

# (already imported at top: Path)


class ResumeKeyLoaderTests(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    def test_returns_empty_set_for_missing_file(self) -> None:
        from e4_attention_reweighting import _load_completed_keys  # local import

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nope.jsonl"
            self.assertEqual(_load_completed_keys(path), set())

    def test_loads_complete_records(self) -> None:
        from e4_attention_reweighting import _load_completed_keys

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.jsonl"
            self._write_jsonl(path, [
                {"sample_instance_id": "a-0", "condition": "target_only", "mask_strength": -1.0},
                {"sample_instance_id": "a-0", "condition": "target_plus_irrelevant_number",
                 "mask_strength": -1.0},
                {"sample_instance_id": "b-1", "condition": "target_only", "mask_strength": 0.0},
            ])
            self.assertEqual(_load_completed_keys(path), {
                ("a-0", "target_only", -1.0),
                ("a-0", "target_plus_irrelevant_number", -1.0),
                ("b-1", "target_only", 0.0),
            })

    def test_skips_malformed_trailing_line(self) -> None:
        from e4_attention_reweighting import _load_completed_keys

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.jsonl"
            with path.open("w") as fh:
                fh.write(json.dumps({
                    "sample_instance_id": "a", "condition": "target_only", "mask_strength": 0.0
                }) + "\n")
                fh.write('{"sample_instance_id": "b", "condition":')  # truncated
            self.assertEqual(_load_completed_keys(path), {("a", "target_only", 0.0)})
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `uv run python -m pytest tests/test_e4_helpers.py::ResumeKeyLoaderTests -v`

Expected: FAIL — `_load_completed_keys` not defined yet.

- [ ] **Step 3: Implement `_load_completed_keys` and the writer helpers**

Insert into `scripts/e4_attention_reweighting.py` above `main()`:

```python
def _output_path(args: argparse.Namespace) -> Path:
    """Canonical output path per (model, phase). Single file accumulates across resumes."""
    sub = "sweep_n200" if args.phase == "sweep" else "full_n17730"
    return PROJECT_ROOT / "outputs" / "e4_mitigation" / args.model / sub / "predictions.jsonl"


def _load_completed_keys(path: Path) -> set[tuple[str, str, float]]:
    """Read existing JSONL, return set of completed (sample_instance_id, condition, strength)
    tuples. Robust to missing file, empty file, and a truncated trailing line."""
    if not path.exists():
        return set()
    completed: set[tuple[str, str, float]] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated/partial trailing line — skip
            try:
                completed.add((
                    str(r["sample_instance_id"]),
                    str(r["condition"]),
                    float(r["mask_strength"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    return completed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_e4_helpers.py::ResumeKeyLoaderTests -v`

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/e4_attention_reweighting.py tests/test_e4_helpers.py
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "E4: resumable JSONL writer — load completed keys, skip on rerun

Tests cover missing file, valid records, and a truncated trailing line.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Generation loop with strength sweep + accuracy metric

**Files:**
- Modify: `scripts/e4_attention_reweighting.py`

- [ ] **Step 1: Implement `_generate_one`, `_resolve_upper_half_layers`, and the main loop**

Insert into `scripts/e4_attention_reweighting.py` above `main()`:

```python
def _resolve_upper_half_layers(n_layers: int) -> list[int]:
    return list(range(n_layers // 2, n_layers))


def _exact_match(parsed: int | None, ground_truth: str | None) -> int:
    if parsed is None or ground_truth is None:
        return 0
    try:
        return int(int(parsed) == int(str(ground_truth).strip()))
    except (ValueError, TypeError):
        return 0


def _generate_one(runner, sample: dict, image_token_id: int | None,
                  layers, layer_indices: list[int], strength: float,
                  max_new_tokens: int) -> dict[str, Any]:
    anchor_span = _resolve_anchor_span(runner, sample, image_token_id)
    install_indices = layer_indices if strength != 0 else []
    handles = _install_hooks(layers, install_indices, anchor_span, strength=strength) \
        if install_indices else []
    try:
        out = runner.generate_number(
            question=sample["question"],
            images=sample["input_images"],
            max_new_tokens=max_new_tokens,
        )
    finally:
        for h in handles:
            h.remove()
    return {
        "anchor_span": list(anchor_span),
        "decoded": out["raw_text"],
        "parsed_number": out["parsed_number"],
    }
```

- [ ] **Step 2: Replace `main()` body with the full pipeline**

Replace the `main()` body in `scripts/e4_attention_reweighting.py`:

```python
def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    enriched = _load_samples(args, config)

    out_path = _output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(out_path)
    print(f"[setup] phase={args.phase} model={args.model} "
          f"sample_instances={len(enriched)} resuming_from={len(completed)}")

    sampling = config["sampling"]
    prompt = config["prompt"]
    inference_cfg = InferenceConfig(
        system_prompt=prompt["system"],
        user_template=prompt["user_template"],
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        max_new_tokens=sampling["max_new_tokens"],
    )
    print(f"[setup] loading {args.hf_model} (eager attention)")
    runner = build_eager_runner(args.hf_model, inference_config=inference_cfg)
    image_token_id: int | None = None
    if isinstance(runner, EagerAttentionRunner):
        image_token_id = _resolve_image_token_id(runner.processor)
    layers = _get_llm_layers(runner.model)
    n_layers = len(layers)
    upper_half = _resolve_upper_half_layers(n_layers)
    print(f"[setup] LLM layers={n_layers}; upper_half={upper_half[0]}..{upper_half[-1]}")

    if args.phase == "sweep":
        strengths = SWEEP_STRENGTHS
    else:
        strengths = [0.0, args.strength]  # baseline + chosen
    print(f"[setup] strengths={strengths}")
    print(f"[setup] writing to {out_path}")

    n_done = 0
    n_skipped = 0
    t0 = time.time()
    with out_path.open("a") as fh:
        for sample in enriched:
            for cond in build_conditions(sample):
                for strength in strengths:
                    key = (str(cond["sample_instance_id"]), cond["condition"], float(strength))
                    if key in completed:
                        n_skipped += 1
                        continue
                    try:
                        gen = _generate_one(
                            runner, cond, image_token_id, layers,
                            upper_half, strength, args.max_new_tokens,
                        )
                    except Exception as exc:
                        gen = {"error": str(exc), "decoded": None, "parsed_number": None,
                               "anchor_span": None}
                    record = {
                        "model": args.model,
                        "sample_instance_id": cond["sample_instance_id"],
                        "question_id": cond["question_id"],
                        "image_id": cond["image_id"],
                        "ground_truth": cond["ground_truth"],
                        "condition": cond["condition"],
                        "irrelevant_type": cond["irrelevant_type"],
                        "anchor_value": cond.get("anchor_value_for_metrics"),
                        "mask_strength": float(strength),
                        "n_llm_layers": n_layers,
                        "exact_match": _exact_match(gen.get("parsed_number"),
                                                    cond["ground_truth"]),
                        **gen,
                    }
                    fh.write(json.dumps(record, default=str) + "\n")
                    fh.flush()
            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (len(enriched) - n_done) / rate if rate > 0 else 0
                print(f"[progress] {n_done}/{len(enriched)} "
                      f"({rate:.2f}/s, ~{remaining:.0f}s left, skipped={n_skipped})")

    print(f"[done] {n_done} samples processed in {time.time() - t0:.1f}s; "
          f"skipped {n_skipped} pre-completed; total records "
          f"≈ {n_done * 3 * len(strengths) - n_skipped}")
```

- [ ] **Step 3: Smoke run with --max-samples 3, --phase sweep**

Run:
```
CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
  --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf \
  --phase sweep --max-samples 3 2>&1 | tail -20
```

Expected: setup messages, [progress] hidden because n_done<10, [done] line. Output file `outputs/e4_mitigation/llava-1.5-7b/sweep_n200/predictions.jsonl` exists with `3 × 3 × 7 = 63` records.

Verify: `wc -l outputs/e4_mitigation/llava-1.5-7b/sweep_n200/predictions.jsonl` should equal 63.

- [ ] **Step 4: Smoke test resume — re-run same command**

Run again the same command. Expected: `[setup] resuming_from=63`, `[done] 0 samples processed; skipped 63 pre-completed; total records ≈ -63` (negative is fine — diagnostic only).

Verify: `wc -l` still 63 (no duplicate appends).

- [ ] **Step 5: Verify exact_match column populated**

Run: `head -1 outputs/e4_mitigation/llava-1.5-7b/sweep_n200/predictions.jsonl | uv run python -c "import json,sys; r=json.loads(sys.stdin.read()); print('exact_match' in r, r.get('mask_strength'), r.get('exact_match'))"`

Expected: `True 0.0 0` or `True 0.0 1` (match/no-match).

- [ ] **Step 6: Clean up smoke output and commit**

```bash
rm -rf outputs/e4_mitigation/llava-1.5-7b/sweep_n200/
git add scripts/e4_attention_reweighting.py
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "E4: generation loop with strength sweep + exact_match per record

Each record carries mask_strength (float) and exact_match (int) on top
of the E1d schema. Re-running the same command resumes from the existing
JSONL by skipping completed (sample_instance_id, condition, mask_strength)
keys. Smoke-tested at n=3 sweep on LLaVA-1.5: 63 records on first run,
63 skipped + 0 written on rerun.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: (Removed — folded into Task 5)

(Strength loop and accuracy column landed together in Task 5.)

---

## Task 7: Analysis script with bootstrap CIs, Pareto plot, strength selection

**Files:**
- Create: `scripts/analyze_e4_mitigation.py`
- Modify: `tests/test_e4_helpers.py`

- [ ] **Step 1: Write a failing test for the strength-selection rule**

Append to `tests/test_e4_helpers.py`:

```python
class StrengthSelectionTests(unittest.TestCase):
    def test_picks_smallest_magnitude_meeting_both_criteria(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from analyze_e4_mitigation import select_optimal_strength

        # Format: dict[strength] -> {"df_num": float, "em_num": float}
        baseline_df, baseline_em = 0.30, 0.55
        per_strength = {
            0.0:   {"df_num": 0.30, "em_num": 0.55},  # baseline
            -0.5:  {"df_num": 0.29, "em_num": 0.55},  # df drop only 0.01 — fails df criterion
            -1.0:  {"df_num": 0.26, "em_num": 0.54},  # df = 0.30 - 0.04 = -13.3% — passes df; em drop 1pp passes
            -2.0:  {"df_num": 0.20, "em_num": 0.50},  # passes df easily; em drop 5pp — fails em criterion
            -1e4:  {"df_num": 0.18, "em_num": 0.42},  # fails em criterion
        }
        chosen = select_optimal_strength(per_strength, baseline_df, baseline_em,
                                         df_drop_target=0.10, em_drop_max=0.02)
        self.assertEqual(chosen, -1.0)

    def test_returns_none_when_no_strength_qualifies(self) -> None:
        from analyze_e4_mitigation import select_optimal_strength

        baseline_df, baseline_em = 0.30, 0.55
        per_strength = {
            0.0:   {"df_num": 0.30, "em_num": 0.55},
            -1.0:  {"df_num": 0.29, "em_num": 0.45},  # huge em drop — fails
            -1e4:  {"df_num": 0.10, "em_num": 0.30},  # massive em drop
        }
        self.assertIsNone(select_optimal_strength(per_strength, baseline_df, baseline_em,
                                                  df_drop_target=0.10, em_drop_max=0.02))
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `uv run python -m pytest tests/test_e4_helpers.py::StrengthSelectionTests -v`

Expected: FAIL — module not yet created.

- [ ] **Step 3: Create `scripts/analyze_e4_mitigation.py`**

```python
"""Analyse E4 mitigation runs — sweep Pareto, full validation summary.

Reads outputs/e4_mitigation/<model>/sweep_n200/predictions.jsonl (Phase 1) and
outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl (Phase 2) for each
mid-stack-cluster model. Produces per-strength bootstrap CIs, a Pareto plot,
and a chosen-strength selection per model.

Usage:
    uv run python scripts/analyze_e4_mitigation.py --phase sweep
    uv run python scripts/analyze_e4_mitigation.py --phase full

Strength-selection rule (per design doc):
  Among strengths where
    direction_follow_rate(target_plus_irrelevant_number, s)
      <= 0.9 * direction_follow_rate(target_plus_irrelevant_number, 0)
    AND exact_match(target_plus_irrelevant_number, s)
      >= exact_match(target_plus_irrelevant_number, 0) - 0.02,
  pick the smallest |s|.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E4_ROOT = PROJECT_ROOT / "outputs" / "e4_mitigation"
SUMMARY_DIR = E4_ROOT / "_summary"

PANEL_MODELS = ["llava-1.5-7b", "convllava-7b", "internvl3-8b"]


def _load_phase(model: str, phase: str) -> pd.DataFrame | None:
    sub = "sweep_n200" if phase == "sweep" else "full_n17730"
    path = E4_ROOT / model / sub / "predictions.jsonl"
    if not path.exists():
        return None
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df


def _build_triplets(df: pd.DataFrame) -> pd.DataFrame:
    """Per (sample_instance_id, mask_strength), join target_only baseline (strength=0)
    with target_plus_irrelevant_number under that strength. Adds columns
    base_pred (= target_only @ s=0 parsed_number) and num_pred (= number @ s)."""
    base = (
        df[(df["condition"] == "target_only") & (df["mask_strength"] == 0.0)]
        [["sample_instance_id", "parsed_number", "ground_truth"]]
        .rename(columns={"parsed_number": "base_pred"})
        .drop_duplicates("sample_instance_id")
    )
    num = df[df["condition"] == "target_plus_irrelevant_number"].copy()
    num = num.rename(columns={"parsed_number": "num_pred"})
    num = num.merge(base, on="sample_instance_id", how="inner",
                    suffixes=("", "_base"))
    return num


def _to_int(x):
    try:
        return int(str(x).strip())
    except (ValueError, TypeError):
        return None


def _metrics(triplets: pd.DataFrame) -> dict[str, float | int]:
    base = triplets["base_pred"].map(_to_int)
    num = triplets["num_pred"].map(_to_int)
    anchor = triplets["anchor_value"].map(_to_int)
    gt = triplets["ground_truth"].map(_to_int)
    valid = base.notna() & num.notna() & anchor.notna()
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "df_num": np.nan, "adopt_num": np.nan, "em_num": np.nan,
                "mean_dist": np.nan}
    db = (base[valid] - anchor[valid]).abs()
    dn = (num[valid] - anchor[valid]).abs()
    df_num = float((dn < db).mean())
    adopt = float((num[valid] == anchor[valid]).mean())
    mean_dist = float(dn.mean())
    em_valid = num.notna() & gt.notna()
    em = float((num[em_valid] == gt[em_valid]).mean()) if em_valid.any() else np.nan
    return {"n": n, "df_num": df_num, "adopt_num": adopt, "em_num": em,
            "mean_dist": mean_dist}


def _bootstrap(triplets: pd.DataFrame, metric: str, n_boot: int = 2000,
               seed: int = 42) -> tuple[float, float]:
    base = triplets["base_pred"].map(_to_int)
    num = triplets["num_pred"].map(_to_int)
    anchor = triplets["anchor_value"].map(_to_int)
    gt = triplets["ground_truth"].map(_to_int)
    valid = base.notna() & num.notna() & anchor.notna()
    arr_base = base[valid].to_numpy()
    arr_num = num[valid].to_numpy()
    arr_anc = anchor[valid].to_numpy()
    arr_gt = gt[valid].to_numpy()
    n = len(arr_num)
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric == "df_num":
            db = np.abs(arr_base[idx] - arr_anc[idx])
            dn = np.abs(arr_num[idx] - arr_anc[idx])
            draws[i] = (dn < db).mean()
        elif metric == "adopt_num":
            draws[i] = (arr_num[idx] == arr_anc[idx]).mean()
        elif metric == "em_num":
            mask_gt = arr_gt[idx] != None
            if mask_gt.sum() == 0:
                draws[i] = np.nan
            else:
                draws[i] = (arr_num[idx][mask_gt] == arr_gt[idx][mask_gt]).mean()
        elif metric == "mean_dist":
            draws[i] = float(np.abs(arr_num[idx] - arr_anc[idx]).mean())
        else:
            raise ValueError(metric)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def select_optimal_strength(per_strength: dict[float, dict[str, float]],
                            baseline_df: float, baseline_em: float,
                            df_drop_target: float = 0.10,
                            em_drop_max: float = 0.02) -> float | None:
    """Pick smallest |strength| satisfying both target criteria. Returns None if none qualify."""
    df_threshold = baseline_df * (1.0 - df_drop_target)
    em_threshold = baseline_em - em_drop_max
    candidates = []
    for s, m in per_strength.items():
        if s == 0.0:
            continue
        df_num = m["df_num"]
        em_num = m["em_num"]
        if np.isnan(df_num) or np.isnan(em_num):
            continue
        if df_num <= df_threshold and em_num >= em_threshold:
            candidates.append(s)
    if not candidates:
        return None
    return min(candidates, key=abs)


def _summarise_phase(phase: str) -> pd.DataFrame:
    rows = []
    for model in PANEL_MODELS:
        df = _load_phase(model, phase)
        if df is None:
            print(f"[{model}] skipping — no {phase} run")
            continue
        triplets = _build_triplets(df)
        for s in sorted(triplets["mask_strength"].unique()):
            sub = triplets[triplets["mask_strength"] == s]
            stats = _metrics(sub)
            ci_df_lo, ci_df_hi = _bootstrap(sub, "df_num")
            ci_em_lo, ci_em_hi = _bootstrap(sub, "em_num")
            rows.append({
                "model": model, "mask_strength": float(s), **stats,
                "df_ci_low": ci_df_lo, "df_ci_high": ci_df_hi,
                "em_ci_low": ci_em_lo, "em_ci_high": ci_em_hi,
            })
    return pd.DataFrame(rows)


def _pareto_plot(df_summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(PANEL_MODELS), figsize=(5 * len(PANEL_MODELS), 4),
                             sharey=False)
    if len(PANEL_MODELS) == 1:
        axes = [axes]
    for ax, model in zip(axes, PANEL_MODELS):
        sub = df_summary[df_summary["model"] == model].sort_values("mask_strength")
        if sub.empty:
            ax.set_title(f"{model} (no data)")
            continue
        # Plot direction-follow on left axis, exact-match on right axis
        x = sub["mask_strength"]
        ax.plot(x, sub["df_num"], "o-", label="direction_follow", color="tab:red")
        ax.fill_between(x, sub["df_ci_low"], sub["df_ci_high"], color="tab:red", alpha=0.2)
        ax.set_xlabel("mask_strength")
        ax.set_ylabel("direction_follow_rate", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax.twinx()
        ax2.plot(x, sub["em_num"], "s-", label="exact_match", color="tab:blue")
        ax2.fill_between(x, sub["em_ci_low"], sub["em_ci_high"], color="tab:blue", alpha=0.2)
        ax2.set_ylabel("exact_match (target_plus_irrelevant_number)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_title(model)
        ax.set_xscale("symlog", linthresh=0.5)  # so -1e4 plots reasonably
    fig.suptitle("E4 strength sweep — direction-follow vs exact-match")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sweep", "full"), default="sweep")
    args = parser.parse_args()

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "sweep":
        df_summary = _summarise_phase("sweep")
        out_csv = SUMMARY_DIR / "sweep_pareto.csv"
        df_summary.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}")
        _pareto_plot(df_summary, SUMMARY_DIR / "sweep_pareto.png")
        print(f"[write] {SUMMARY_DIR / 'sweep_pareto.png'}")

        # Pick optimal per model
        chosen: dict[str, float | None] = {}
        for model in PANEL_MODELS:
            sub = df_summary[df_summary["model"] == model]
            if sub.empty:
                chosen[model] = None
                continue
            baseline = sub[sub["mask_strength"] == 0.0]
            if baseline.empty:
                chosen[model] = None
                continue
            base_df = float(baseline["df_num"].iloc[0])
            base_em = float(baseline["em_num"].iloc[0])
            per_strength = {
                float(r.mask_strength): {"df_num": float(r.df_num), "em_num": float(r.em_num)}
                for r in sub.itertuples(index=False)
            }
            chosen[model] = select_optimal_strength(per_strength, base_df, base_em)
        out_json = SUMMARY_DIR / "chosen_strength.json"
        out_json.write_text(json.dumps(chosen, indent=2))
        print(f"[write] {out_json}")
        print("=== chosen strengths ===")
        for k, v in chosen.items():
            print(f"  {k}: {v}")
    else:
        df_summary = _summarise_phase("full")
        out_csv = SUMMARY_DIR / "full_validation.csv"
        df_summary.to_csv(out_csv, index=False)
        print(f"[write] {out_csv}")
        print("=== full-scale validation ===")
        print(df_summary.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test to verify selection rule passes**

Run: `uv run python -m pytest tests/test_e4_helpers.py::StrengthSelectionTests -v`

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_e4_mitigation.py tests/test_e4_helpers.py
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "E4: analysis script — Pareto plot + strength-selection rule

Per-(model, strength, condition) bootstrap CIs (2000 iter, 95%) for
direction_follow_rate, adoption_rate, exact_match, mean_distance_to_anchor.
Picks smallest |strength| where direction_follow drops >=10% AND exact_match
drops <=2pp (target_plus_irrelevant_number condition). Writes
sweep_pareto.{csv,png} and chosen_strength.json (Phase 1) or
full_validation.csv (Phase 2). Selection rule covered by unit tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Run Phase 1 sweep on the 3 mid-stack models

**Files:**
- Read: `scripts/e4_attention_reweighting.py`

- [ ] **Step 1: Verify GPU and disk are clean**

Run: `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader && df -h /mnt/ddn`
Expected: GPU 0 free; ample disk.

- [ ] **Step 2: Launch LLaVA-1.5 sweep (background, nohup)**

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
  --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf --phase sweep \
  > /tmp/e4_sweep_llava.log 2>&1; echo "[exit] $?" >> /tmp/e4_sweep_llava.log' \
  </dev/null >/dev/null 2>&1 &
disown
```

Monitor: `tail -F /tmp/e4_sweep_llava.log` until `[exit] 0`.
Expected runtime: ~30–60 min.
Expected output: `outputs/e4_mitigation/llava-1.5-7b/sweep_n200/predictions.jsonl` with 200 × 3 × 7 = 4,200 records.

- [ ] **Step 3: Verify LLaVA-1.5 sweep completed**

Run: `wc -l outputs/e4_mitigation/llava-1.5-7b/sweep_n200/predictions.jsonl`
Expected: 4200.

- [ ] **Step 4: Launch ConvLLaVA sweep**

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
  --model convllava-7b --hf-model ConvLLaVA/ConvLLaVA-sft-1536 --phase sweep \
  > /tmp/e4_sweep_convllava.log 2>&1; echo "[exit] $?" >> /tmp/e4_sweep_convllava.log' \
  </dev/null >/dev/null 2>&1 &
disown
```

Monitor: `tail -F /tmp/e4_sweep_convllava.log` until `[exit] 0`.
Expected output: 4,200 records.

- [ ] **Step 5: Launch InternVL3 sweep**

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
  --model internvl3-8b --hf-model OpenGVLab/InternVL3-8B-hf --phase sweep \
  > /tmp/e4_sweep_internvl3.log 2>&1; echo "[exit] $?" >> /tmp/e4_sweep_internvl3.log' \
  </dev/null >/dev/null 2>&1 &
disown
```

Monitor until exit. Verify 4,200 records.

- [ ] **Step 6: No commit (data only — not in git)**

`outputs/` is gitignored. Just verify all three runs landed.

---

## Task 9: Aggregate Phase 1 results, pick `s*`

**Files:**
- Read: `scripts/analyze_e4_mitigation.py`

- [ ] **Step 1: Run analysis**

Run: `uv run python scripts/analyze_e4_mitigation.py --phase sweep 2>&1 | tail -30`
Expected: writes `outputs/e4_mitigation/_summary/sweep_pareto.{csv,png}` and `chosen_strength.json`. Prints chosen strengths per model.

- [ ] **Step 2: Inspect the Pareto plot**

Open `outputs/e4_mitigation/_summary/sweep_pareto.png`. Verify:
- Direction-follow trends down as `|strength|` grows (more negative = more reduction).
- Exact-match holds roughly flat for moderate strengths (-1, -2), drops at strong strengths.
- The chosen strength per model lies at a "knee" of the curve.

- [ ] **Step 3: Decide shared vs per-model strength**

If `chosen_strength.json` shows the same s* (or adjacent values) across all three models, treat the modal value as the **shared canonical strength** for Phase 2. If they diverge meaningfully (e.g., one model's s* is 3× another's), use per-model values for Phase 2 and flag in the writeup.

Record the decision (shared vs per-model, plus the value) for use in Task 10.

- [ ] **Step 4: No commit (analysis output not in git)**

Same reasoning — `outputs/` is gitignored. The chosen strength value is captured in `chosen_strength.json` and will be referenced by Task 10's command.

---

## Task 10: Run Phase 2 full-scale validation (resumable, ~45–90 hours)

**Files:**
- Read: `scripts/e4_attention_reweighting.py`

This task spans multiple sessions. The user can stop and resume at any time by re-running the same command.

- [ ] **Step 1: Read the chosen strength**

Run: `cat outputs/e4_mitigation/_summary/chosen_strength.json`
Note `<s*_llava>`, `<s*_conv>`, `<s*_intern>` for the three models.

- [ ] **Step 2: Launch LLaVA-1.5 full validation**

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
  --model llava-1.5-7b --hf-model llava-hf/llava-1.5-7b-hf \
  --phase full --strength <s*_llava> \
  > /tmp/e4_full_llava.log 2>&1; echo "[exit] $?" >> /tmp/e4_full_llava.log' \
  </dev/null >/dev/null 2>&1 &
disown
```

Substitute `<s*_llava>` with the actual chosen strength (e.g., `-1.0`).

Expected runtime: ~15–30 hours.
Expected output: `outputs/e4_mitigation/llava-1.5-7b/full_n17730/predictions.jsonl` with 17,730 × 3 × 2 ≈ 106,380 records.

If interrupted: re-run the same command. The `_load_completed_keys` path resumes from the last completed `(sample, condition, strength)` triple.

- [ ] **Step 3: Launch ConvLLaVA full validation**

After (or during, if disk and GPU allow — they do not, GPU 0 only) LLaVA-1.5 completes:

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
  --model convllava-7b --hf-model ConvLLaVA/ConvLLaVA-sft-1536 \
  --phase full --strength <s*_conv> \
  > /tmp/e4_full_convllava.log 2>&1; echo "[exit] $?" >> /tmp/e4_full_convllava.log' \
  </dev/null >/dev/null 2>&1 &
disown
```

Wait until `[exit] 0`. Verify line count.

- [ ] **Step 4: Launch InternVL3 full validation**

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run python scripts/e4_attention_reweighting.py \
  --model internvl3-8b --hf-model OpenGVLab/InternVL3-8B-hf \
  --phase full --strength <s*_intern> \
  > /tmp/e4_full_internvl3.log 2>&1; echo "[exit] $?" >> /tmp/e4_full_internvl3.log' \
  </dev/null >/dev/null 2>&1 &
disown
```

Verify line count when done.

- [ ] **Step 5: Verify completion of all three**

For each of llava-1.5-7b / convllava-7b / internvl3-8b:

```bash
wc -l outputs/e4_mitigation/<model>/full_n17730/predictions.jsonl
```

Each should report ≥ 106,380 records (exact count depends on seeds and any sample-instance loss; ≥ 99 % of expected is acceptable).

- [ ] **Step 6: No commit (data only)**

---

## Task 11: Aggregate Phase 2 + write `docs/experiments/E4-mitigation.md` + Korean mirror

**Files:**
- Read: `outputs/e4_mitigation/_summary/full_validation.csv`
- Create: `docs/experiments/E4-mitigation.md`
- Create: `docs/experiments/E4-mitigation_ko.md`

- [ ] **Step 1: Run full-phase analysis**

Run: `uv run python scripts/analyze_e4_mitigation.py --phase full 2>&1 | tail -30`
Expected: writes `full_validation.csv`; prints per-(model, mask_strength, condition) stats.

- [ ] **Step 2: Write `docs/experiments/E4-mitigation.md`**

Structure (model after `E1d-causal-ablation.md`):
1. **Status** line — points to outputs and the (now-superseded) design doc.
2. **TL;DR — 3 findings.** Whether the chosen strength meets the target on each model; whether it generalises across the 3 mid-stack models; ConvLLaVA inclusion decision.
3. **Setup.** 7-strength sweep (Phase 1) → chosen strength → full validation (Phase 2). Reference design doc for criterion math.
4. **Result 1 — Phase 1 sweep & strength choice.** Table per model: baseline df/em vs each strength, chosen s*, justification. Reference `outputs/e4_mitigation/_summary/sweep_pareto.png`.
5. **Result 2 — Phase 2 full-scale validation.** Table per model: baseline (s=0) df/em vs chosen-strength df/em on all three conditions, with bootstrap CIs. Compare against E1d's `ablate_upper_half` numbers.
6. **ConvLLaVA decision.** Whether numbers justify keeping ConvLLaVA in the headline mid-stack-cluster claim, or demoting to discussion caveat. Reference E1d's lower-half divergence.
7. **Caveats.** Resumability across runs (any cross-session sample drift?), exact_match as accuracy proxy, anchor distribution skew (anchors 7, 8 inert per A2), ConvLLaVA caveat.
8. **Roadmap update.** §6 E4 row → ✅; §10 changelog 2026-XX-XX entry.

- [ ] **Step 3: Write `docs/experiments/E4-mitigation_ko.md`**

Korean mirror of the English writeup. Tables and code blocks stay English; narrative in Korean.

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/E4-mitigation.md docs/experiments/E4-mitigation_ko.md
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "E4 mitigation results: <one-line headline of finding>

<Body summarising chosen strength per model, full-scale validation
delta vs baseline, ConvLLaVA inclusion decision, and any caveats.>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Roadmap update

**Files:**
- Modify: `references/roadmap.md`
- Modify: `references/roadmap_ko.md`

- [ ] **Step 1: Update §6 Tier 1 E4 row**

Edit `references/roadmap.md`. Find the E4 row in §6 (added by the E1d commit). Change `Status` from `☐` to `✅` and prefix the `Why it matters` cell with the actual outcome (e.g., "✅ Phase 1 + Phase 2 complete on the mid-stack cluster (LLaVA-1.5, ConvLLaVA, InternVL3). Optimal strength `s* = -X` reduces direction_follow_rate by Y pp on Z/3 models with ≤ W pp accuracy drop. ...").

Mirror the change in `references/roadmap_ko.md`.

- [ ] **Step 2: Append §10 changelog entry**

Add a new dated entry at the bottom of §10 in both files, stating the headline result and pointing to `docs/experiments/E4-mitigation.md`.

- [ ] **Step 3: Update §7 ordering and decision triggers**

If Phase C (Tier-2 hardening) is the new active phase, update the §7 sequence accordingly. Add a "After E4" decision-trigger entry stating whether the target was met and what the next step is.

- [ ] **Step 4: Commit**

```bash
git add references/roadmap.md references/roadmap_ko.md
GIT_AUTHOR_NAME="namam3gy" GIT_AUTHOR_EMAIL="namam3gy@gmail.com" \
GIT_COMMITTER_NAME="namam3gy" GIT_COMMITTER_EMAIL="namam3gy@gmail.com" \
git commit -m "Roadmap: close E4 — <headline outcome>

§6 Tier 1 E4 row marked ✅. §10 changelog entry. §7 decision trigger
updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes

- Spec coverage: Phase 1 sweep (Tasks 5, 8, 9), Phase 2 full validation (Task 10), strength-knob and resumability (Tasks 1, 4, 5), accuracy metric (Task 5), ConvLLaVA inclusion deferred to writeup (Task 11), failure escalation noted in design (handled at Task 9 decision step).
- No placeholders, no "TBD" / "similar to Task N" / "fill in details". Every code block is concrete.
- Type consistency: `strength: float`, `mask_strength: float`, `_load_completed_keys → set[tuple[str, str, float]]`, `select_optimal_strength → float | None` are consistent across tasks.
- Task 6 was folded into Task 5 (strength loop and accuracy column landed together) — left as a placeholder note rather than renumbering, to avoid breaking inbound references.
