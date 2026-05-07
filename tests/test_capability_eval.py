"""Tests for E8 capability eval — hook factory parity, aggregator math, verdict logic."""

from __future__ import annotations

import torch

from vlm_anchor.hooks import make_subspace_projection_hook


def test_hook_projects_last_token_at_prefill():
    """At prefill (seq_len > 1), only the last token's residual is updated;
    the update equals h - α · V^T V h."""
    torch.manual_seed(0)
    d_model, K, seq_len = 8, 2, 5
    alpha = 1.0
    # Construct V_K with orthonormal rows so V^T V is a proper projector
    V_K = torch.linalg.qr(torch.randn(d_model, K)).Q.T  # (K, d_model)
    hidden = torch.randn(1, seq_len, d_model)
    expected = hidden.clone()
    last = expected[:, -1, :]
    proj = (last @ V_K.T) @ V_K
    expected[:, -1, :] = last - alpha * proj

    hook = make_subspace_projection_hook(V_K, alpha)
    output = (hidden.clone(),)  # mimic HF decoder layer output (tuple)
    new_output = hook(None, None, output)
    new_hidden = new_output[0] if isinstance(new_output, tuple) else new_output

    torch.testing.assert_close(new_hidden, expected)


def test_hook_decode_step_is_noop():
    """At decode (seq_len == 1), the hook returns output unchanged."""
    d_model, K = 8, 2
    V_K = torch.linalg.qr(torch.randn(d_model, K)).Q.T
    hook = make_subspace_projection_hook(V_K, alpha=1.0)
    hidden = torch.randn(1, 1, d_model)
    output = (hidden.clone(),)
    new_output = hook(None, None, output)
    new_hidden = new_output[0] if isinstance(new_output, tuple) else new_output
    torch.testing.assert_close(new_hidden, hidden)


def test_hook_alpha_zero_returns_none():
    """alpha=0 means caller installs no hook; factory returns None."""
    V_K = torch.randn(2, 8)
    assert make_subspace_projection_hook(V_K, alpha=0.0) is None


def test_hook_non_3d_output_unchanged():
    """If hook receives a non-3D tensor (e.g. tuple of None), it returns input."""
    V_K = torch.linalg.qr(torch.randn(8, 2)).Q.T
    hook = make_subspace_projection_hook(V_K, alpha=1.0)
    # Tuple where first element is not a 3D tensor
    weird_output = (torch.randn(8),)  # 1D
    result = hook(None, None, weird_output)
    assert result is weird_output  # unchanged


def test_hook_dtype_device_match():
    """Hook casts V_K to match hidden's device + dtype."""
    d_model, K = 8, 2
    V_K = torch.linalg.qr(torch.randn(d_model, K)).Q.T.to(torch.float32)
    hook = make_subspace_projection_hook(V_K, alpha=1.0)
    hidden = torch.randn(1, 3, d_model, dtype=torch.float16)
    output = (hidden.clone(),)
    new_output = hook(None, None, output)
    new_hidden = new_output[0] if isinstance(new_output, tuple) else new_output
    assert new_hidden.dtype == torch.float16
