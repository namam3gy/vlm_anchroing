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

    def test_direction_follow_C_form(self) -> None:
        # C-form: (pa-pb) · (anchor-pb) > 0  AND  pa != pb.
        # base=5, anchor=3 (anchor below base), pa=4 (moved from 5 toward 3) →
        # (pa-pb) = -1, (anchor-pb) = -2, product = 2 > 0 → direction_followed=1.
        ev = self._eval(prediction="4", anchor_value="3", base_prediction="5", gt_answer="5")
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
        # C-form direction-follow: (pa-pb) · (anchor-pb) > 0  AND  pa != pb.
        # Setup: base=7, pred=4, anchor=3.
        #   (pa-pb) = -3, (anchor-pb) = -4, product = 12 > 0 → df_raw=1.
        #   pa != pb (4 != 7) → df_moved=1.
        ev_moved = self._eval(prediction="4", anchor_value="3", base_prediction="7")
        self.assertEqual(ev_moved.anchor_direction_followed, 1)
        self.assertEqual(ev_moved.anchor_direction_followed_moved, 1)

        # pa == pb under C-form → (pa-pb) = 0 → product = 0 → df_raw = 0
        # automatically. So df_moved is also 0. (Differs from the old
        # anchor·gt form, where df_raw could fire even when pa == pb; under
        # C-form that case is structurally impossible.)
        ev_no_move = self._eval(prediction="4", anchor_value="3", base_prediction="4")
        self.assertEqual(ev_no_move.anchor_direction_followed, 0)
        self.assertEqual(ev_no_move.anchor_direction_followed_moved, 0)

        # Anchor on the opposite side of pb from pa → df_raw = 0.
        # base=5, pred=7 (pa above pb), anchor=2 (anchor below pb) →
        # (pa-pb) = +2, (anchor-pb) = -3, product = -6 < 0 → df_raw=0.
        ev_opposite = self._eval(prediction="7", anchor_value="2", base_prediction="5")
        self.assertEqual(ev_opposite.anchor_direction_followed, 0)
        self.assertEqual(ev_opposite.anchor_direction_followed_moved, 0)


class DriverRowSchemaRegressionTest(unittest.TestCase):
    """Catch the driver-row schema gap that caused df_M2 = 0 silently.

    Every M2 flag declared on ``VQASampleEval`` MUST be threaded into the
    ``run_experiment.py`` row dict, else ``summarize_condition._flag()``
    will read None and report 0 on the live driver path. We previously
    shipped without ``anchor_direction_followed_moved`` /
    ``pred_b_equal_anchor`` / ``pred_diff_from_base`` for ~3 weeks.
    """

    def test_run_experiment_threads_every_M2_flag(self) -> None:
        import dataclasses
        from pathlib import Path

        from vlm_anchor.metrics import VQASampleEval

        src = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "run_experiment.py"
        ).read_text()

        # Fields that exist on VQASampleEval but are intentionally renamed or
        # mapped to other column names in the row dict.
        # `prediction` is captured as `raw_prediction` (full model text).
        # `normalized_prediction` is captured as `prediction` (the canonical
        # parsed string used downstream).
        renamed = {
            "prediction",
            "normalized_prediction",
            "ground_truth",
            "normalized_ground_truth",
            "anchor_value",
        }
        for field in dataclasses.fields(VQASampleEval):
            if field.name in renamed:
                continue
            self.assertIn(
                f"sample_eval.{field.name}",
                src,
                f"run_experiment.py row dict missing thread of M2 flag "
                f"`{field.name}` — summarize_condition will silently report 0.",
            )


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
