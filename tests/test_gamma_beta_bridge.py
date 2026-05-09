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


if __name__ == "__main__":
    unittest.main()
