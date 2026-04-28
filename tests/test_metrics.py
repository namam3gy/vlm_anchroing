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


class EvaluateSampleM2FlagsTest(unittest.TestCase):
    """M2-specific per-row flags: pb_eq_anchor / pa_ne_pb / df_moved."""

    def _eval(self, prediction: str, anchor_value: str | None,
              base_prediction: str | None, gt_answer: str = "5"):
        return evaluate_sample(
            prediction=prediction,
            gt_answer=gt_answer,
            all_answers=[gt_answer] * 10,
            anchor_value=anchor_value,
            base_prediction=base_prediction,
        )

    def test_pred_b_equal_anchor_true_when_base_matches_anchor(self) -> None:
        ev = self._eval(prediction="3", anchor_value="3", base_prediction="3")
        self.assertEqual(ev.pred_b_equal_anchor, 1)

    def test_pred_b_equal_anchor_false_when_base_differs(self) -> None:
        ev = self._eval(prediction="3", anchor_value="3", base_prediction="5")
        self.assertEqual(ev.pred_b_equal_anchor, 0)

    def test_pred_b_equal_anchor_zero_when_anchor_or_base_missing(self) -> None:
        # No anchor → flag is 0.
        ev1 = self._eval(prediction="3", anchor_value=None, base_prediction="3")
        # Base unparseable → flag is 0 (conservative).
        ev2 = self._eval(prediction="3", anchor_value="3", base_prediction=None)
        self.assertEqual(ev1.pred_b_equal_anchor, 0)
        self.assertEqual(ev2.pred_b_equal_anchor, 0)

    def test_pred_diff_from_base_when_pa_changes(self) -> None:
        ev = self._eval(prediction="4", anchor_value="3", base_prediction="5")
        self.assertEqual(ev.pred_diff_from_base, 1)

    def test_pred_diff_from_base_zero_when_no_movement(self) -> None:
        # pa == pb → no movement.
        ev = self._eval(prediction="5", anchor_value="3", base_prediction="5")
        self.assertEqual(ev.pred_diff_from_base, 0)

    def test_direction_followed_moved_requires_both(self) -> None:
        # gt=5, anchor=3 (anchor<gt), base=5 (no df because pb=gt → product=0),
        # but here we simulate base=6 (pb−gt=1, pa−gt=−1 → product=−1 → df=0).
        # Take a clearer setup: gt=5, base=7, pred=4, anchor=3.
        # pb−gt = 2, pa−gt = −1, anchor−gt = −2.
        # df = (pa−gt)(anchor−gt) = (−1)(−2) = 2 > 0 → df_raw=1.
        # pa != pb (4 != 7) → df_moved=1.
        ev_moved = self._eval(prediction="4", anchor_value="3", base_prediction="7")
        self.assertEqual(ev_moved.anchor_direction_followed, 1)
        self.assertEqual(ev_moved.anchor_direction_followed_moved, 1)

        # df_raw fires but pa == pb → df_moved must be zero.
        # gt=5, base=4, pred=4, anchor=3: pa−gt = −1, anchor−gt = −2 → product 2 → df=1.
        # pa == pb (4 == 4) → df_moved=0.
        ev_no_move = self._eval(prediction="4", anchor_value="3", base_prediction="4")
        self.assertEqual(ev_no_move.anchor_direction_followed, 1)
        self.assertEqual(ev_no_move.anchor_direction_followed_moved, 0)


class SummarizeConditionM2DenominatorTest(unittest.TestCase):
    """Headline rates use M2 denominators; legacy rates remain reported."""

    @staticmethod
    def _record(condition: str, **fields) -> dict:
        defaults = {
            "standard_vqa_accuracy": 0.0,
            "exact_match": 0,
            "anchor_adopted": 0,
            "anchor_direction_followed": 0,
            "anchor_direction_followed_moved": 0,
            "pred_b_equal_anchor": 0,
            "pred_diff_from_base": 0,
            "numeric_distance_to_anchor": None,
            "anchor_value": "3",
            "normalized_prediction": "0",
            "normalized_ground_truth": "0",
            "condition": condition,
        }
        defaults.update(fields)
        return defaults

    def test_adoption_rate_uses_paired_denominator(self) -> None:
        # 10 records with anchor='3'; 2 of them have pred_b == anchor.
        # Among the 8 paired-eligible: 2 are adopted.
        # M2 rate = 2/8 = 0.25; marginal = 2/10 = 0.20.
        from vlm_anchor.metrics import summarize_condition

        records: list[dict] = []
        for i in range(2):
            records.append(self._record("a", pred_b_equal_anchor=1))  # excluded from M2 denominator
        for i in range(2):
            records.append(self._record("a", anchor_adopted=1))
        for i in range(6):
            records.append(self._record("a"))

        summary = summarize_condition(records, "a")
        self.assertAlmostEqual(summary["anchor_adoption_rate"], 2 / 8)
        self.assertAlmostEqual(summary["anchor_adoption_rate_marginal"], 2 / 10)
        self.assertEqual(summary["n_pb_ne_anchor_denominator"], 8)

    def test_direction_follow_rate_uses_moved_numerator(self) -> None:
        from vlm_anchor.metrics import summarize_condition

        # All 5 records numeric with anchor.
        # 3 of 5 are df_raw=1, 2 of those also df_moved=1.
        # df_raw rate = 3/5; df (M2) rate = 2/5.
        records: list[dict] = []
        for _ in range(2):
            records.append(self._record(
                "a",
                anchor_direction_followed=1,
                anchor_direction_followed_moved=1,
                anchor_value="3",
                numeric_distance_to_anchor=1.0,
            ))
        for _ in range(1):
            records.append(self._record(
                "a",
                anchor_direction_followed=1,
                anchor_direction_followed_moved=0,
                anchor_value="3",
                numeric_distance_to_anchor=1.0,
            ))
        for _ in range(2):
            records.append(self._record(
                "a",
                anchor_value="3",
                numeric_distance_to_anchor=3.0,
            ))

        summary = summarize_condition(records, "a")
        self.assertAlmostEqual(summary["anchor_direction_follow_rate_raw"], 3 / 5)
        self.assertAlmostEqual(summary["anchor_direction_follow_rate"], 2 / 5)
        self.assertEqual(summary["n_numeric_anchor_denominator"], 5)
