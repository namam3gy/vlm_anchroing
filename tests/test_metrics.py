from __future__ import annotations

import unittest

from vlm_anchor.metrics import evaluate_sample


class EvaluateSamplePairedAdoptionTest(unittest.TestCase):
    """anchor_adopted = (base_pred != anchor) AND (pred == anchor)."""

    def _eval(self, prediction: str, anchor_value: str | None, base_prediction: str | None,
              gt_answer: str = "5", all_answers: list[str] | None = None):
        return evaluate_sample(
            prediction=prediction,
            gt_answer=gt_answer,
            all_answers=all_answers if all_answers is not None else [gt_answer] * 10,
            anchor_value=anchor_value,
            base_prediction=base_prediction,
        )

    def test_adopted_when_base_differs_from_anchor_and_pred_matches_anchor(self) -> None:
        ev = self._eval(prediction="3", anchor_value="3", base_prediction="5")
        self.assertEqual(ev.anchor_adopted, 1)

    def test_not_adopted_when_base_already_equals_anchor(self) -> None:
        # The classic confound: base_pred == anchor → cannot attribute to anchor exposure.
        ev = self._eval(prediction="3", anchor_value="3", base_prediction="3")
        self.assertEqual(ev.anchor_adopted, 0)

    def test_not_adopted_when_pred_differs_from_anchor(self) -> None:
        ev = self._eval(prediction="4", anchor_value="3", base_prediction="5")
        self.assertEqual(ev.anchor_adopted, 0)

    def test_zero_when_anchor_value_is_none(self) -> None:
        # target_only / neutral arm: no anchor → adoption is undefined → 0.
        ev = self._eval(prediction="3", anchor_value=None, base_prediction="5")
        self.assertEqual(ev.anchor_adopted, 0)

    def test_zero_when_base_prediction_missing_or_unparseable(self) -> None:
        # Conservative: cannot establish baseline → cannot count as adoption.
        ev_none = self._eval(prediction="3", anchor_value="3", base_prediction=None)
        ev_garbage = self._eval(prediction="3", anchor_value="3", base_prediction="not a number")
        self.assertEqual(ev_none.anchor_adopted, 0)
        self.assertEqual(ev_garbage.anchor_adopted, 0)

    def test_direction_follow_unchanged_by_paired_refactor(self) -> None:
        # Direction-follow is independent of base_prediction — must still work as before.
        ev = self._eval(prediction="4", anchor_value="3", base_prediction="5", gt_answer="5")
        # gt=5, anchor=3 (anchor < gt), pred=4 (between), so pred − gt = -1, anchor − gt = -2,
        # product = 2 > 0 → direction_followed = 1.
        self.assertEqual(ev.anchor_direction_followed, 1)

    def test_distance_unchanged_by_paired_refactor(self) -> None:
        ev = self._eval(prediction="4", anchor_value="3", base_prediction="5")
        self.assertEqual(ev.numeric_distance_to_anchor, 1)
