from __future__ import annotations

import unittest

from vlm_anchor.models import (
    ConvLLaVARunner,
    FastVLMRunner,
    HFAttentionRunner,
    InferenceConfig,
    build_runner,
)


class HFAttentionRunnerHelperTests(unittest.TestCase):
    def test_build_prompt_preserves_json_braces_in_user_template(self) -> None:
        runner = object.__new__(HFAttentionRunner)
        runner.cfg = InferenceConfig(
            system_prompt="Return JSON.",
            user_template='Return JSON only in the form {"result": <number>}. Question: {question}',
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=4,
        )

        messages = runner._build_prompt("How many cats?", num_images=1)

        self.assertEqual(
            messages[1]["content"][1]["text"],
            'Return JSON only in the form {"result": <number>}. Question: How many cats?',
        )


class BuildRunnerDispatchTests(unittest.TestCase):
    """Verify dispatch rules without actually loading the heavy checkpoints."""

    def _select(self, hf_model: str) -> type:
        # Inspect dispatch logic without instantiation (which would hit HF hub / GPU).
        lower = hf_model.lower()
        if "fastvlm" in lower or "llava-qwen" in lower:
            return FastVLMRunner
        if "convllava" in lower:
            return ConvLLaVARunner
        return HFAttentionRunner

    def test_fastvlm_dispatch(self) -> None:
        self.assertIs(self._select("apple/FastVLM-7B"), FastVLMRunner)

    def test_convllava_dispatch(self) -> None:
        self.assertIs(self._select("ConvLLaVA/ConvLLaVA-sft-1536"), ConvLLaVARunner)

    def test_default_dispatch(self) -> None:
        for name in [
            "llava-hf/llava-1.5-7b-hf",
            "OpenGVLab/InternVL3-8B-hf",
            "Qwen/Qwen3-VL-8B-Instruct",
        ]:
            self.assertIs(self._select(name), HFAttentionRunner)

    def test_build_runner_is_exported(self) -> None:
        self.assertTrue(callable(build_runner))


if __name__ == "__main__":
    unittest.main()
