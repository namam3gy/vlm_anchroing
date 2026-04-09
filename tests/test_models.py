from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_anchor.models import (
    AttentionVisualizationConfig,
    HFAttentionRunner,
    InferenceConfig,
    TokenActivationRecord,
    _supports_attention_visualization,
    build_model_runner,
)


class _FakeTokenizer:
    all_special_ids = [1, 106]
    TOKEN_MAP = {
        11: "11",
        12: "12",
        21: "21",
        22: "22",
        31: "1",
        32: "2",
        500: "{",
        501: "}",
        502: '"result"',
        503: ":",
        504: "7",
    }

    @staticmethod
    def convert_ids_to_tokens(token_ids: list[int]) -> list[str]:
        return [_FakeTokenizer.TOKEN_MAP.get(token_id, f"tok_{token_id}") for token_id in token_ids]

    @staticmethod
    def decode(token_ids: list[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False) -> str:
        return "".join(_FakeTokenizer.convert_ids_to_tokens(token_ids))


class HFAttentionRunnerHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = object.__new__(HFAttentionRunner)
        self.runner.processor = SimpleNamespace(tokenizer=_FakeTokenizer())
        self.runner._output_projection_cache = {}
        self.runner.attention_cfg = AttentionVisualizationConfig(
            span_source="target_or_prediction",
            head_top_k=1,
            head_min_image_mass=0.1,
            layer_start_ratio=0.5,
            layer_count=2,
            layer_strategy="mean",
        )

    @staticmethod
    def _step_layer(head_rows: list[list[float]]) -> torch.Tensor:
        return torch.tensor([[head_rows[0], head_rows[1]]], dtype=torch.float32).unsqueeze(2)

    def test_select_answer_token_span_uses_core_numeric_span_inside_json_output(self) -> None:
        span = self.runner._select_answer_token_span(
            generated_token_ids=[500, 502, 503, 31, 32, 501],
            target_answer_text="12",
            predicted_answer_text="12",
        )

        self.assertEqual(span["source"], "target_answer")
        self.assertEqual(span["relative_positions"], [3, 4])
        self.assertEqual(span["token_ids"], [31, 32])

    def test_select_answer_token_span_falls_back_to_predicted_span_when_target_is_absent(self) -> None:
        span = self.runner._select_answer_token_span(
            generated_token_ids=[500, 502, 503, 31, 32, 501],
            target_answer_text="7",
            predicted_answer_text="12",
        )

        self.assertEqual(span["source"], "predicted_answer")
        self.assertEqual(span["relative_positions"], [3, 4])
        self.assertEqual(span["token_ids"], [31, 32])

    def test_select_image_centric_heads_prefers_focused_image_head(self) -> None:
        head_maps = torch.tensor(
            [
                [0.05, 0.30, 0.30, 0.30, 0.05],
                [0.05, 0.85, 0.00, 0.00, 0.10],
                [0.45, 0.03, 0.02, 0.00, 0.50],
            ],
            dtype=torch.float32,
        )

        selected_heads, stats = self.runner._select_image_centric_heads(
            head_maps=head_maps,
            image_token_indices=np.array([1, 2, 3], dtype=np.int64),
        )

        self.assertEqual(selected_heads, [1])
        self.assertGreater(float(stats["scores"][1]), float(stats["scores"][0]))
        self.assertGreater(float(stats["image_mass"][1]), float(stats["image_mass"][2]))

    def test_aggregate_prompt_attention_from_generation_steps_uses_mid_to_late_layers(self) -> None:
        step_attentions = [
            (
                torch.zeros((1, 2, 3, 5), dtype=torch.float32),
                torch.zeros((1, 2, 3, 5), dtype=torch.float32),
                torch.zeros((1, 2, 3, 5), dtype=torch.float32),
                torch.zeros((1, 2, 3, 5), dtype=torch.float32),
            ),
            (
                self._step_layer([[0.20, 0.20, 0.20, 0.20, 0.20], [0.20, 0.20, 0.20, 0.20, 0.20]]),
                self._step_layer([[0.05, 0.85, 0.05, 0.05, 0.00], [0.45, 0.05, 0.05, 0.40, 0.05]]),
                self._step_layer([[0.20, 0.20, 0.20, 0.20, 0.20], [0.20, 0.20, 0.20, 0.20, 0.20]]),
                self._step_layer([[0.05, 0.10, 0.75, 0.10, 0.00], [0.45, 0.10, 0.10, 0.30, 0.05]]),
            ),
            (
                self._step_layer([[0.20, 0.20, 0.20, 0.20, 0.20], [0.20, 0.20, 0.20, 0.20, 0.20]]),
                self._step_layer([[0.05, 0.80, 0.10, 0.05, 0.00], [0.40, 0.10, 0.10, 0.35, 0.05]]),
                self._step_layer([[0.20, 0.20, 0.20, 0.20, 0.20], [0.20, 0.20, 0.20, 0.20, 0.20]]),
                self._step_layer([[0.05, 0.05, 0.80, 0.10, 0.00], [0.45, 0.10, 0.10, 0.25, 0.10]]),
            ),
        ]

        prompt_attention, metadata = self.runner._aggregate_prompt_attention_from_generation_steps(
            step_attentions=step_attentions,
            step_indices=[1, 2],
            prompt_seq_len=5,
            image_token_indices=np.array([1, 2, 3], dtype=np.int64),
        )

        self.assertEqual(metadata["layer_indices"], [1, 3])
        self.assertEqual(metadata["layer_strategy"], "mean")
        self.assertEqual(metadata["selected_layers"][0]["selected_heads"], [0])
        self.assertEqual(metadata["selected_layers"][1]["selected_heads"], [0])
        self.assertGreater(prompt_attention[1], prompt_attention[0])
        self.assertGreater(prompt_attention[2], prompt_attention[0])

    def test_aggregate_prompt_attention_from_layers_supports_rollout(self) -> None:
        self.runner.attention_cfg = AttentionVisualizationConfig(
            span_source="target_or_prediction",
            head_top_k=1,
            head_min_image_mass=0.1,
            layer_start_ratio=1.0,
            layer_count=1,
            layer_strategy="rollout",
        )
        layer0 = torch.eye(5, dtype=torch.float32).reshape(1, 1, 5, 5).repeat(1, 2, 1, 1)
        layer1 = torch.eye(5, dtype=torch.float32).reshape(1, 1, 5, 5).repeat(1, 2, 1, 1)
        layer1[0, 0, 4, :] = torch.tensor([0.05, 0.10, 0.75, 0.10, 0.00], dtype=torch.float32)
        layer1[0, 1, 4, :] = torch.tensor([0.45, 0.10, 0.10, 0.25, 0.10], dtype=torch.float32)

        prompt_attention, metadata = self.runner._aggregate_prompt_attention_from_layers(
            layer_attentions=[layer0, layer1],
            query_positions=[4],
            prompt_seq_len=4,
            image_token_indices=np.array([1, 2, 3], dtype=np.int64),
            allow_rollout=True,
        )

        self.assertEqual(metadata["layer_strategy"], "rollout")
        self.assertEqual(metadata["layer_indices"], [1])
        self.assertEqual(metadata["selected_layers"][0]["selected_heads"], [0])
        self.assertGreater(prompt_attention[2], prompt_attention[0])

    def test_compute_qwen_layout_specs_preserves_real_grid_shape(self) -> None:
        token_counts, grid_shapes = HFAttentionRunner._compute_qwen_layout_specs(
            image_grid_thw=torch.tensor([[1, 30, 40], [1, 64, 64]]),
            merge_size=2,
        )

        self.assertEqual(token_counts, [300, 1024])
        self.assertEqual(grid_shapes, [(15, 20), (32, 32)])

    def test_compute_gemma_layout_specs_uses_pooled_position_grid(self) -> None:
        image_position_ids = np.full((1, 2520, 2), -1, dtype=np.int64)
        valid_positions = []
        for y in range(14):
            for x in range(19):
                for dy in range(3):
                    for dx in range(3):
                        valid_positions.append((x * 3 + dx, y * 3 + dy))
        image_position_ids[0, : len(valid_positions), :] = np.array(valid_positions, dtype=np.int64)

        token_counts, grid_shapes = HFAttentionRunner._compute_gemma_layout_specs(
            num_soft_tokens_per_image=[266],
            image_position_ids=image_position_ids,
            pooling_kernel_size=3,
        )

        self.assertEqual(token_counts, [266])
        self.assertEqual(grid_shapes, [(14, 19)])

    def test_compute_generation_class_scores_concatenates_prompt_and_generation_steps(self) -> None:
        self.runner._output_projection_cache[31] = (torch.tensor([1.0, 0.0], dtype=torch.float32), 0.5)
        step_hidden_states = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        ]

        scores = self.runner._compute_generation_class_scores(
            step_hidden_states=step_hidden_states,
            class_token_id=31,
            max_step_index=1,
        )

        np.testing.assert_allclose(scores, np.array([1.5, 3.5, 5.5], dtype=np.float32))

    def test_apply_tam_eci_removes_weighted_interference(self) -> None:
        previous_records = [
            TokenActivationRecord(
                sequence_position=2,
                token_id=500,
                token_piece="{",
                image_scores=np.array([0.9, 0.1], dtype=np.float32),
            ),
            TokenActivationRecord(
                sequence_position=3,
                token_id=503,
                token_piece=":",
                image_scores=np.array([0.2, 0.8], dtype=np.float32),
            ),
        ]

        corrected = self.runner._apply_tam_eci(
            current_image_scores=np.array([0.8, 0.4], dtype=np.float32),
            current_token_piece="7",
            previous_records=previous_records,
            previous_text_scores=np.array([0.9, 0.1], dtype=np.float32),
        )

        self.assertLess(corrected[0], 0.8)
        self.assertGreaterEqual(corrected[1], 0.0)

    def test_select_prompt_text_positions_skips_special_and_image_tokens(self) -> None:
        positions = self.runner._select_prompt_text_positions(
            prompt_input_ids=[1, 11, 12, 13, 106],
            prompt_seq_len=5,
            image_token_indices=np.array([2], dtype=np.int64),
        )

        self.assertEqual(positions, [1, 3])

    def test_build_prompt_preserves_json_braces_in_user_template(self) -> None:
        self.runner.cfg = InferenceConfig(
            system_prompt="Return JSON.",
            user_template='Return JSON only in the form {"result": <number>}. Question: {question}',
            temperature=0.0,
            top_p=1.0,
            num_ctx=1024,
            max_new_tokens=4,
        )

        messages = self.runner._build_prompt("How many cats?", num_images=1)

        self.assertEqual(messages[1]["content"][1]["text"], 'Return JSON only in the form {"result": <number>}. Question: How many cats?')

    def test_build_model_runner_uses_default_hf_runner(self) -> None:
        sentinel = object()
        with patch("vlm_anchor.models.HFAttentionRunner", return_value=sentinel) as hf_ctor:
            runner = build_model_runner("Qwen/Qwen2.5-VL-7B-Instruct")

        self.assertIs(runner, sentinel)
        hf_ctor.assert_called_once()

    def test_supports_attention_visualization_for_qwen_and_gemma_only(self) -> None:
        self.assertTrue(_supports_attention_visualization("Qwen/Qwen2.5-VL-7B-Instruct"))
        self.assertTrue(_supports_attention_visualization("google/gemma-4-31B-it"))
        self.assertFalse(_supports_attention_visualization("llava-hf/llava-interleave-qwen-7b-hf"))


if __name__ == "__main__":
    unittest.main()
