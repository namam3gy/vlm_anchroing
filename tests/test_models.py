from __future__ import annotations

import unittest

from vlm_anchor.models import HFAttentionRunner, InferenceConfig


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

        self.assertEqual(messages[1]["content"][1]["text"], 'Return JSON only in the form {"result": <number>}. Question: How many cats?')


if __name__ == "__main__":
    unittest.main()
