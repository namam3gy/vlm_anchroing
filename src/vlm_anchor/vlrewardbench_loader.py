"""VL-RewardBench loading helpers (HF: MMInstruction/VL-RewardBench).

Schema (1247 rows, single 'test' split):
    id              str
    query           str            # prompt text
    response        list[str]      # 2 model responses (pairwise)
    image           PIL.Image
    human_ranking   list[int]      # which response is preferred (50% human, 25% gpt-4o)
    models          list[str]
    judge           str
    rationale       str
    query_source    str            # COCO, GQA, VQAv2, POVID, ..., or empty (= MMMU-Pro/MathVerse)
    ground_truth    str

Random-response design: pick one of the 2 responses uniformly at random
(seed-controlled). Spreads baseline judge-score distribution.
"""

from __future__ import annotations

import random
from typing import Iterable


def random_response_index(responses: Iterable[str], rng: random.Random) -> int | None:
    """Pick a random response index, skipping empty strings."""
    eligible: list[int] = []
    for idx, resp in enumerate(responses):
        if (resp or "").strip():
            eligible.append(idx)
    if not eligible:
        return None
    return rng.choice(eligible)


def chosen_response_index(responses: Iterable[str], human_ranking: Iterable[int]) -> int | None:
    """Pick the preferred response per `human_ranking` (higher rank = better).

    Mirrors v1 chosen-response semantics from VLFeedback (max-avg-rated).
    Returns None if all candidate responses are empty.
    """
    responses = list(responses)
    rankings = list(human_ranking) if human_ranking is not None else []
    if not responses or len(rankings) != len(responses):
        return None
    best_idx: int | None = None
    best_rank: int = -10**9
    for idx, (resp, rank) in enumerate(zip(responses, rankings)):
        if not (resp or "").strip():
            continue
        if rank > best_rank:
            best_rank = int(rank)
            best_idx = idx
    return best_idx
