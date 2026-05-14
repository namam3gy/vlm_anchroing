"""VLFeedback loading helpers.

VLFeedback records have 4 model `completions`, each with `annotations` over
3 dimensions (Helpfulness, Visual Faithfulness, Ethical Considerations),
each with a "Rating" string in {"1", ..., "5"}. We derive the "chosen"
completion as the one with the highest mean rating across the 3 dimensions
(skipping completions where all 3 ratings are unparseable).

These ratings are NOT shown to our pilot judges — they are only an offline
selector for which of the 4 responses to feed into the b/a/m arms.
"""

from __future__ import annotations

import random
from typing import Iterable


def parse_rating(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    if v < 1 or v > 5:
        return None
    return float(v)


def _completion_mean(completion: dict) -> float | None:
    anns = completion.get("annotations") or {}
    parts: list[float] = []
    for dim in ("Helpfulness", "Visual Faithfulness", "Ethical Considerations"):
        rating = parse_rating((anns.get(dim) or {}).get("Rating"))
        if rating is not None:
            parts.append(rating)
    if not parts:
        return None
    return sum(parts) / len(parts)


def derive_chosen_completion_index(completions: Iterable[dict]) -> int | None:
    best_idx: int | None = None
    best_mean: float = float("-inf")
    for idx, comp in enumerate(completions):
        mean = _completion_mean(comp)
        if mean is None:
            continue
        if mean > best_mean:
            best_mean = mean
            best_idx = idx
    return best_idx


def random_completion_index(completions: Iterable[dict], rng: random.Random) -> int | None:
    """Pick a random completion index, skipping ones with non-empty response.

    Used by the v2 random-response design (vs derive_chosen_*) — random selector
    spreads baseline VF distribution across 1-5, giving both floor- and
    ceiling-push anchor variants room to move.
    """
    eligible: list[int] = []
    for idx, comp in enumerate(completions):
        resp = (comp.get("response") or "").strip()
        if resp:
            eligible.append(idx)
    if not eligible:
        return None
    return rng.choice(eligible)
