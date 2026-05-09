"""P0-1 γ-β bridge — smoke test that Qwen3-VL forward hook on language_model.layers[33]
captures (n_tokens, 4096) hidden state during generation.

Prevents regression of memory:feedback_sdpa_mask_hook_bug — eager attention
required for hook-modified kwargs to be honored.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class Qwen3VLHookSmokeTest(unittest.TestCase):

    def test_layer_33_residual_capture_shape_during_generation(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from PIL import Image

        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",   # SDPA mask-hook bug guard
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()

        # Tiny dummy image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        captured: list[torch.Tensor] = []

        def make_hook(target_layer: int):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # hidden shape: (batch=1, seq_len, hidden_size)
                captured.append(hidden.detach().to("cpu", dtype=torch.float32))
            return hook

        # Find the LM layer stack — Qwen3-VL exposes via model.model.language_model.layers
        # (or model.language_model.layers depending on accessor).
        lm = getattr(model, "model", model)
        lm = getattr(lm, "language_model", lm)
        layers = lm.layers
        self.assertGreaterEqual(len(layers), 36, "Qwen3-VL-8B should have 36 LM layers")

        target_layer = 33
        h = layers[target_layer].register_forward_hook(make_hook(target_layer))

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What number is shown? Answer with one integer."},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True,
                )
        finally:
            h.remove()

        # We expect captured to have entries for: 1 prefill + N decode steps (N <= 5)
        self.assertGreaterEqual(len(captured), 2, "expected at least one prefill + one decode")
        # Last hidden state of last decode = (1, 1, 4096)
        last = captured[-1]
        self.assertEqual(last.shape[0], 1)
        # During decode steps seq_len is 1; during prefill seq_len > 1
        self.assertEqual(last.shape[2], 4096, f"hidden_size mismatch: got {last.shape[2]}")

        # Prefill tensor — first capture — should have seq_len > 1
        prefill = captured[0]
        self.assertGreater(prefill.shape[1], 1)


class ProjectionArithmeticTest(unittest.TestCase):
    """V_K^T @ h projection produces correct amplitude regardless of h variation.

    These tests validate the projection math used by
    `scripts/run_gamma_beta_bridge.py::project_amplitude` independently of GPU
    inference: a residual lying entirely in span(V_K) must have its
    amplitude equal to the L2 norm of the basis coefficients, and a residual
    in the orthogonal complement must project to ~0.
    """

    def test_projection_along_basis(self):
        torch.manual_seed(0)
        d_model = 4096
        K = 8
        # Random orthonormal basis via QR. Q's columns are orthonormal.
        V_full = torch.linalg.qr(torch.randn(d_model, d_model))[0]
        V_K = V_full[:, :K]   # shape (4096, 8)
        # Construct a residual that lies entirely in span(V_K).
        coeffs = torch.tensor([1.0, 2.0, 0.5, -1.5, 0.0, 3.0, -2.0, 1.0])
        h = V_K @ coeffs   # shape (4096,)
        # Projection = V_K^T @ h should recover coeffs (since V_K is orthonormal).
        proj = V_K.T @ h
        torch.testing.assert_close(proj, coeffs, rtol=1e-5, atol=1e-5)
        amp = proj.norm().item()
        self.assertAlmostEqual(amp, coeffs.norm().item(), places=4)

    def test_projection_orthogonal_zero(self):
        torch.manual_seed(1)
        d_model = 4096
        K = 8
        V_full = torch.linalg.qr(torch.randn(d_model, d_model))[0]
        V_K = V_full[:, :K]
        # Residual entirely in the orthogonal complement.
        coeffs_perp = torch.randn(d_model - K)
        h_perp = V_full[:, K:] @ coeffs_perp
        proj = V_K.T @ h_perp
        # Should be ~0 (numerical noise from QR orthonormalization in float32).
        self.assertLess(proj.norm().item(), 1e-3)

    def test_per_token_amplitude_shape(self):
        """project_amplitude takes (n_gen, d_model) and returns (n_gen,)."""
        import sys
        SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
        if str(SCRIPTS_ROOT) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_ROOT))
        from run_gamma_beta_bridge import project_amplitude
        torch.manual_seed(2)
        d_model = 4096
        K = 8
        V_K = torch.linalg.qr(torch.randn(d_model, d_model))[0][:, :K]
        residuals = torch.randn(7, d_model)   # 7 generated tokens
        amps = project_amplitude(residuals, V_K)
        self.assertEqual(amps.shape, (7,))
        self.assertTrue((amps >= 0).all(), "amplitude must be non-negative")


if __name__ == "__main__":
    unittest.main()
