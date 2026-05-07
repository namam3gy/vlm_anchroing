"""Mitigated LLaVA-OneVision-HF wrapper for VLMEvalKit-driven capability eval.

Installs the §7.4.5 chosen subspace projection hook (L=26, K=8, α=1.0
by default) on a single Qwen2 decoder layer at construction. The hook
fires at prefill on the last input token only — see
``vlm_anchor.hooks.make_subspace_projection_hook`` for the math.

Drop-in replacement for ``vlmeval.vlm.llava.llava.LLaVA_OneVision_HF``
from VLMEvalKit's perspective: the same ``generate`` / ``forward_short``
/ etc. methods are inherited unchanged.

We use the HF backend (transformers.LlavaOnevisionForConditionalGeneration)
rather than the LLaVA-NeXT backend because (a) the LLaVA-NeXT package is
not in our venv and adding it risks dep conflicts with our pinned
torch/transformers, and (b) the HF mirror weights `llava-hf/llava-onevision-qwen2-7b-ov-hf`
are the converted-format mirror of `lmms-lab/llava-onevision-qwen2-7b-ov`
(same training, identical weights). Capability eval results are therefore
on the HF format; subspace tensor was calibrated against the LLaVA-NeXT
loader, but the underlying Qwen2 LM is byte-identical, so layer-26
residuals are numerically equivalent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from vlmeval.vlm.llava.llava import LLaVA_OneVision_HF

from vlm_anchor.hooks import make_subspace_projection_hook


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_subspace(subspace_path: str | Path, layer: int, K: int) -> torch.Tensor:
    """Load (n_layers, K_max, d_model) tensor and return the (K, d_model) slice."""
    p = Path(subspace_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    full = torch.load(p, weights_only=True)
    if full.dim() != 3:
        raise ValueError(f"Expected 3D subspace tensor, got shape {tuple(full.shape)}")
    if full.shape[0] <= layer:
        raise ValueError(f"layer={layer} out of range for n_layers={full.shape[0]}")
    if full.shape[1] < K:
        raise ValueError(f"K={K} exceeds K_max={full.shape[1]}")
    return full[layer, :K, :].contiguous()


class LLaVAOneVisionMitigated(LLaVA_OneVision_HF):
    """LLaVA-OneVision-HF with the chosen subspace mitigation hook installed."""

    def __init__(self, *,
                 subspace_path: str,
                 layer: int = 26,
                 K: int = 8,
                 alpha: float = 1.0,
                 **vlmeval_kwargs: Any):
        super().__init__(**vlmeval_kwargs)

        V_K = _resolve_subspace(subspace_path, layer, K)
        # Move to model device + dtype for stable hook math.
        target = next(self.model.parameters())
        V_K = V_K.to(device=target.device, dtype=target.dtype)

        # HF backend layer path (not LLaVA-NeXT's path).
        layers = self.model.model.language_model.layers
        if layer >= len(layers):
            raise ValueError(f"layer={layer} >= len(layers)={len(layers)}")

        proj_hook = make_subspace_projection_hook(V_K, alpha)
        if proj_hook is None:
            raise ValueError("alpha=0 produces no hook — caller likely misconfigured.")

        # Wrap projection hook so we can count activations for sanity check.
        self._mit_calls = {"prefill": 0, "decode": 0, "other": 0}
        counter = self._mit_calls

        def counting_hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                if hidden.shape[1] > 1:
                    counter["prefill"] += 1
                else:
                    counter["decode"] += 1
            else:
                counter["other"] += 1
            return proj_hook(module, args, output)

        self._mit_handle = layers[layer].register_forward_hook(counting_hook)
        self._mit_meta = {"layer": layer, "K": K, "alpha": alpha,
                          "subspace_path": str(subspace_path)}

    def __del__(self):
        h = getattr(self, "_mit_handle", None)
        if h is not None:
            try:
                h.remove()
            except Exception:
                pass
