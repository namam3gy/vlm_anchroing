"""Pre-compute the canonical 5,000-sid TallyQA sub-sample for Option H1.

The list is shared across all 6 main-panel models so the H1 panel stays
paired at the sid level — comparing OneVision n=5000 vs Qwen2.5-VL n=5000
only makes sense if both ran on the SAME 5000 sids.

Approach: take the first 5,000 eligible samples in `questions.jsonl` order
(question_id sort). Filter matches `experiment_e5e_tallyqa_full.yaml`:
`answer_type_filter=["number"]`, `answer_range=8`,
`require_single_numeric_gt=True`. No stratification, no seeded shuffle —
deterministic-by-file-order is the cheapest audit-able sub-sampler.

Output: `inputs/tallyqa_5k_sids.json` (sorted sample_instance_id list).
"""
from __future__ import annotations
import json
from pathlib import Path

from vlm_anchor.data import load_number_vqa_samples

REPO = Path(__file__).resolve().parents[1]
LOCAL_PATH = REPO / "inputs" / "tallyqa_test"
OUT_PATH = REPO / "inputs" / "tallyqa_5k_sids.json"
CAP = 5000


def main() -> None:
    samples = load_number_vqa_samples(
        dataset_path=LOCAL_PATH,
        max_samples=CAP,
        require_single_numeric_gt=True,
        answer_range=8,
        samples_per_answer=None,
        answer_type_filter=["number"],
    )
    # The loader returns base rows keyed by (question_id, image_id) — those
    # are the H1-stable identifiers, persisted ahead of `build_conditions`
    # generating per-variant `sample_instance_id` strings.
    qids = sorted(int(s["question_id"]) for s in samples)
    if len(qids) != CAP:
        raise SystemExit(
            f"Expected {CAP} eligible samples, got {len(qids)}. "
            "Check loader filter vs config."
        )

    OUT_PATH.write_text(json.dumps({"question_ids": qids, "cap": CAP}, indent=2) + "\n")
    print(f"Wrote {OUT_PATH} ({len(qids)} question_ids)")
    print(f"First 5: {qids[:5]}")
    print(f"Last 5:  {qids[-5:]}")


if __name__ == "__main__":
    main()
