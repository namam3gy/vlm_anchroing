from __future__ import annotations

import unittest

from vlm_anchor.utils import extract_first_number, normalize_numeric_text


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


if __name__ == "__main__":
    unittest.main()
