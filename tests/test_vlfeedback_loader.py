import unittest

from vlm_anchor.vlfeedback_loader import (
    derive_chosen_completion_index,
    parse_rating,
)


class TestParseRating(unittest.TestCase):
    def test_parses_string_int(self) -> None:
        self.assertEqual(parse_rating("3"), 3.0)

    def test_returns_none_for_non_integer(self) -> None:
        self.assertIsNone(parse_rating("N/A"))
        self.assertIsNone(parse_rating(""))
        self.assertIsNone(parse_rating(None))

    def test_clamps_out_of_range(self) -> None:
        self.assertIsNone(parse_rating("0"))
        self.assertIsNone(parse_rating("6"))


class TestDeriveChosenCompletionIndex(unittest.TestCase):
    def _annotations(self, ratings: tuple[str, str, str]) -> dict:
        h, vf, eth = ratings
        return {
            "Helpfulness": {"Rating": h, "Rationale": ""},
            "Visual Faithfulness": {"Rating": vf, "Rationale": ""},
            "Ethical Considerations": {"Rating": eth, "Rationale": ""},
        }

    def test_picks_highest_mean(self) -> None:
        completions = [
            {"annotations": self._annotations(("1", "1", "1"))},  # mean 1.0
            {"annotations": self._annotations(("5", "4", "5"))},  # mean 4.67
            {"annotations": self._annotations(("3", "3", "3"))},  # mean 3.0
            {"annotations": self._annotations(("4", "5", "4"))},  # mean 4.33
        ]
        self.assertEqual(derive_chosen_completion_index(completions), 1)

    def test_breaks_ties_by_lowest_index(self) -> None:
        completions = [
            {"annotations": self._annotations(("4", "4", "4"))},
            {"annotations": self._annotations(("4", "4", "4"))},
        ]
        self.assertEqual(derive_chosen_completion_index(completions), 0)

    def test_skips_completions_with_all_missing_ratings(self) -> None:
        completions = [
            {"annotations": self._annotations(("N/A", "", "N/A"))},  # all missing → skip
            {"annotations": self._annotations(("3", "3", "3"))},
        ]
        self.assertEqual(derive_chosen_completion_index(completions), 1)

    def test_returns_none_when_all_completions_unrateable(self) -> None:
        completions = [
            {"annotations": self._annotations(("", "", ""))},
            {"annotations": self._annotations(("N/A", "N/A", "N/A"))},
        ]
        self.assertIsNone(derive_chosen_completion_index(completions))


if __name__ == "__main__":
    unittest.main()
