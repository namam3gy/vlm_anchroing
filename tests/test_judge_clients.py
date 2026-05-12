import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from PIL import Image

from vlm_anchor.judge_clients import (
    JudgeResponse,
    OpenAIJudgeClient,
    GeminiJudgeClient,
    parse_score,
)


def _write_dummy_png(path: Path) -> None:
    Image.new("RGB", (16, 16), color=(0, 0, 0)).save(path)


class TestParseScore(unittest.TestCase):
    def test_parses_bare_integer(self) -> None:
        self.assertEqual(parse_score("4"), 4)

    def test_parses_score_with_padding(self) -> None:
        self.assertEqual(parse_score("Score: 5\n"), 5)

    def test_parses_score_in_sentence(self) -> None:
        self.assertEqual(parse_score("I rate this 3 out of 5."), 3)

    def test_clamps_out_of_range_to_none(self) -> None:
        self.assertIsNone(parse_score("9"))
        self.assertIsNone(parse_score("0"))

    def test_returns_none_when_no_digit(self) -> None:
        self.assertIsNone(parse_score("the response is faithful"))


class TestOpenAIJudgeClient(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.img1 = self.tmp_path / "i1.png"
        self.img2 = self.tmp_path / "i2.png"
        _write_dummy_png(self.img1)
        _write_dummy_png(self.img2)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @patch("vlm_anchor.judge_clients.OpenAI")
    def test_score_calls_openai_with_multi_image_payload(self, openai_cls: MagicMock) -> None:
        client_instance = MagicMock()
        client_instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="4"))]
        )
        openai_cls.return_value = client_instance

        judge = OpenAIJudgeClient(model_name="gpt-4o-2024-11-20", api_key="sk-fake")
        response = judge.score(images=[self.img1, self.img2], prompt="Rate it.")
        self.assertIsInstance(response, JudgeResponse)
        self.assertEqual(response.score, 4)
        self.assertEqual(response.raw, "4")

        kwargs = client_instance.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4o-2024-11-20")
        self.assertEqual(kwargs["temperature"], 0.0)
        content = kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[2]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/"))


class TestGeminiJudgeClient(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.img1 = self.tmp_path / "i1.png"
        self.img2 = self.tmp_path / "i2.png"
        _write_dummy_png(self.img1)
        _write_dummy_png(self.img2)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @patch("vlm_anchor.judge_clients.genai")
    def test_score_calls_gemini_with_multi_image_payload(self, genai_module: MagicMock) -> None:
        client_instance = MagicMock()
        client_instance.models.generate_content.return_value = MagicMock(text="3")
        genai_module.Client.return_value = client_instance

        judge = GeminiJudgeClient(model_name="gemini-2.5-pro", api_key="ak-fake")
        response = judge.score(images=[self.img1, self.img2], prompt="Rate it.")
        self.assertEqual(response.score, 3)

        kwargs = client_instance.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-pro")
        contents = kwargs["contents"]
        self.assertEqual(contents[0], "Rate it.")
        self.assertEqual(len(contents), 3)


if __name__ == "__main__":
    unittest.main()
