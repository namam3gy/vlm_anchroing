from __future__ import annotations

import unittest

from vlm_anchor.utils import extract_first_number, extract_last_number, normalize_numeric_text


class DecimalHandlingTest(unittest.TestCase):
    """Documents the existing decimal round-trip behaviour of the
    numeric-parse helpers. The E5d ChartQA validation filters to
    integer GT so this is informational; full ChartQA runs that admit
    decimal GT will need to revisit. The current quirk: `normalize_numeric_text`
    preserves trailing `.0` (returns "12.0"), but `extract_first_number`
    strips trailing `.0` (returns "12") — asymmetric.
    """

    def test_normalize_numeric_text_preserves_decimals(self) -> None:
        self.assertEqual(normalize_numeric_text("3.5"), "3.5")
        self.assertEqual(normalize_numeric_text("0.57"), "0.57")
        # Existing behaviour: normalize_numeric_text does NOT strip ".0".
        self.assertEqual(normalize_numeric_text("12.0"), "12.0")

    def test_extract_first_number_handles_decimals(self) -> None:
        self.assertEqual(extract_first_number("the answer is 3.5 dollars"), "3.5")
        self.assertEqual(extract_first_number("0.57"), "0.57")
        # Existing behaviour: extract_first_number strips ".0" suffix.
        self.assertEqual(extract_first_number("12.0"), "12")


class ExtractLastNumberTest(unittest.TestCase):
    """`extract_last_number` is the §6 (γ-β) reasoning-mode helper. The
    headline use case is parsing the *final* answer from a reasoning trace
    where the trace itself contains many irrelevant numerals."""

    def test_picks_last_numeric_span(self) -> None:
        self.assertEqual(extract_last_number("Let me count: 1, 2, 3. Final answer: 7"), "7")
        self.assertEqual(extract_last_number("first 5 then 12 then -3"), "-3")

    def test_strips_trailing_dot_zero(self) -> None:
        self.assertEqual(extract_last_number("12.0 then 7.0"), "7")

    def test_falls_back_to_word_form(self) -> None:
        self.assertEqual(extract_last_number("answer is twelve"), "12")

    def test_empty_input(self) -> None:
        self.assertEqual(extract_last_number(""), "")
        self.assertEqual(extract_last_number(None), "")


if __name__ == "__main__":
    unittest.main()
