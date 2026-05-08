"""Forward-hook factories for inference-time interventions.

Hosted here (rather than under scripts/) so anchoring sweeps and capability
regression tests share one source of truth. Anchoring callers:
``scripts/e6_steering_vector.py``. Capability callers:
``src/vlm_anchor/capability_eval.py``.
"""

from __future__ import annotations

import torch


def make_subspace_projection_hook(V_K: torch.Tensor, alpha: float):
    """Return a forward-hook that subtracts α · V_K^T V_K · h from the
    last-token residual at prefill. Decode steps (seq_len == 1) are skipped;
    the projection's effect propagates to subsequent decode steps via the
    KV cache (same convention as residual-offset hooks elsewhere).

    Returns None if alpha == 0, signalling the caller to install nothing.

    V_K shape: (K, d_model). Rows should be (approximately) orthonormal —
    typically top-K right singular vectors of a pooled difference matrix.
    """
    if alpha == 0:
        return None

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return output
        if hidden.shape[1] <= 1:
            return output  # decode step — propagate via KV cache only
        V_cast = V_K.to(device=hidden.device, dtype=hidden.dtype)
        last = hidden[:, -1, :]
        proj = (last @ V_cast.T) @ V_cast
        hidden[:, -1, :] = last - alpha * proj
        return output

    return hook
