from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from causal_anchor_ablation import _make_anchor_mask_hook  # noqa: E402


class AnchorMaskHookStrengthTests(unittest.TestCase):
    def _apply_hook(self, mask: torch.Tensor, anchor_span, strength: float) -> torch.Tensor:
        hook = _make_anchor_mask_hook(anchor_span, strength=strength)
        self.assertIsNotNone(hook)
        _, kwargs = hook(None, (), {"attention_mask": mask})
        return kwargs["attention_mask"]

    def test_default_strength_matches_e1d_negative_1e4(self) -> None:
        mask = torch.zeros(1, 1, 4, 8)
        out = self._apply_hook(mask, (2, 5), strength=-1e4)
        self.assertTrue(torch.allclose(out[..., 2:5], torch.full((1, 1, 4, 3), -1e4)))
        self.assertTrue(torch.allclose(out[..., :2], torch.zeros((1, 1, 4, 2))))
        self.assertTrue(torch.allclose(out[..., 5:], torch.zeros((1, 1, 4, 3))))

    def test_soft_strength_minus_one_adds_minus_one(self) -> None:
        mask = torch.zeros(1, 1, 4, 8)
        out = self._apply_hook(mask, (2, 5), strength=-1.0)
        self.assertTrue(torch.allclose(out[..., 2:5], torch.full((1, 1, 4, 3), -1.0)))
        scores = torch.zeros(1, 1, 4, 8)
        post = torch.softmax(scores + out, dim=-1)
        anchor_share = post[..., 2:5].sum(dim=-1)
        expected = 3 * torch.exp(torch.tensor(-1.0)) / (5 + 3 * torch.exp(torch.tensor(-1.0)))
        self.assertTrue(torch.allclose(anchor_share, expected.expand_as(anchor_share), atol=1e-5))

    def test_zero_strength_returns_none_or_noop(self) -> None:
        hook = _make_anchor_mask_hook((2, 5), strength=0.0)
        if hook is None:
            return
        mask = torch.zeros(1, 1, 4, 8)
        _, kwargs = hook(None, (), {"attention_mask": mask})
        self.assertTrue(torch.allclose(kwargs["attention_mask"], mask))

    def test_empty_anchor_span_returns_none(self) -> None:
        self.assertIsNone(_make_anchor_mask_hook((0, 0), strength=-1.0))


if __name__ == "__main__":
    unittest.main()
