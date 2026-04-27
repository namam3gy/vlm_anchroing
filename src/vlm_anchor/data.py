from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, UnidentifiedImageError

from vlm_anchor.utils import extract_first_number


ALLOWED_NUMBER_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _image_is_decodable(path: Path) -> bool:
    """Return True if PIL can decode the file without raising."""
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return False
    return True


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    items = [p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_NUMBER_EXTS]
    return sorted(items, key=lambda p: p.name)


def _select_image_variants(images: list[Path], count: int, rng: random.Random) -> list[Path]:
    if count < 1:
        raise ValueError("variants_per_sample must be at least 1")

    selected: list[Path] = []
    pool = images.copy()
    while len(selected) < count:
        rng.shuffle(pool)
        remaining = count - len(selected)
        selected.extend(pool[:remaining])
    return selected


ANCHOR_DISTANCE_STRATA: list[tuple[int, int]] = [
    (0, 1),
    (2, 5),
    (6, 30),
    (31, 300),
    (301, 10**9),
]


def compute_strata(gt: float | int, scheme: str = "absolute") -> list[tuple[int, int]]:
    """Return per-GT (lo, hi) inclusive distance strata.

    scheme="absolute": fixed [(0,1), (2,5), (6,30), (31,300), (301,inf)] —
        VQAv2/TallyQA convention. Returns the same constant regardless of `gt`.

    scheme="relative": hybrid — max(absolute_floor, fractional_of_gt) on each
        upper boundary, with `lo` chained from the previous stratum's `hi`.
        Floors: hi ≥ [1, 5, 30, 300, inf]; fractions: [10%, 30%, 100%, 300%, inf]
        of |round(gt)|. At small GT (e.g., 4) the fractional floor binds
        nowhere and the scheme collapses to "absolute". At large GT
        (e.g., 1000) the strata open up so the wrong-base anchor cohort
        stays defined relative to the question's numeric scale.

    The intended use: ChartQA (and other wide-GT datasets) where a fixed
    301-distance "far" cutoff would lump nearly all anchors into one
    stratum. Use scheme="absolute" for VQAv2/TallyQA (GT 0–8) where the
    fixed convention matches the published E5b/E5c protocol.
    """
    if scheme == "absolute":
        return [(0, 1), (2, 5), (6, 30), (31, 300), (301, 10**9)]
    if scheme != "relative":
        raise ValueError(f"unknown scheme: {scheme!r}")
    g = abs(int(round(float(gt))))
    floors_lo = [0, 2, 6, 31, 301]
    floors_hi = [1, 5, 30, 300, 10**9]
    fractions_hi = [0.10, 0.30, 1.00, 3.00, float("inf")]
    out: list[tuple[int, int]] = []
    prev_hi = -1
    for f_lo, f_hi, frac in zip(floors_lo, floors_hi, fractions_hi):
        if frac == float("inf"):
            hi = 10**9
        else:
            hi = max(f_hi, int(frac * g))
        lo = max(f_lo, prev_hi + 1)
        out.append((lo, hi))
        prev_hi = hi
    return out


def sample_stratified_anchors(
    gt: int,
    inventory: list[int],
    rng: random.Random,
    strata: list[tuple[int, int]] = ANCHOR_DISTANCE_STRATA,
) -> list[int | None]:
    """Return one randomly-chosen anchor per stratum, keyed off |a - gt|.

    Returns None for strata with no inventory match. Strata are matched
    on absolute distance from `gt`; bounds are inclusive on both sides.
    """
    out: list[int | None] = []
    for lo, hi in strata:
        candidates = [a for a in inventory if lo <= abs(a - gt) <= hi]
        out.append(rng.choice(candidates) if candidates else None)
    return out


def load_number_vqa_samples(
    dataset_path: str | Path,
    max_samples: int | None,
    require_single_numeric_gt: bool = True,
    answer_range: int | None = None,
    samples_per_answer: int | None = None,
    answer_type_filter: Iterable[str] | None = None,
) -> list[dict]:
    if answer_range is not None and answer_range < 0:
        raise ValueError("answer_range must be >= 0")
    if samples_per_answer is not None and samples_per_answer <= 0:
        raise ValueError("samples_per_answer must be > 0")

    allowed_answer_types = {str(t).strip() for t in answer_type_filter} if answer_type_filter else None

    dataset_path = Path(dataset_path)
    questions_path = dataset_path / "questions.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"Could not find questions file at {questions_path}")

    samples: list[dict] = []
    answer_counts: dict[int, int] = {}
    with open(questions_path, "r", encoding="utf-8") as f:
        rows = (json.loads(line) for line in f if line.strip())
        for row in rows:
            if allowed_answer_types is not None and row.get("answer_type") not in allowed_answer_types:
                continue
            gt = extract_first_number(row.get("multiple_choice_answer", ""))
            if not gt or not gt.lstrip("-").isdigit():
                continue
            gt_value = int(gt)
            if answer_range is not None and not 0 <= gt_value <= answer_range:
                continue
            if samples_per_answer is not None and answer_counts.get(gt_value, 0) >= samples_per_answer:
                continue
            answers = [extract_first_number(a["answer"]) for a in row.get("answers", [])]
            answers = [a for a in answers if a]
            if require_single_numeric_gt and not all(a.lstrip("-").isdigit() for a in answers if a):
                continue

            image_rel = row.get("image_file")
            if not image_rel:
                raise KeyError("Each question row must include an image_file field")
            image_path = dataset_path / image_rel
            if not image_path.exists():
                raise FileNotFoundError(f"Missing local image for question {row.get('question_id')}: {image_path}")
            if not _image_is_decodable(image_path):
                # Skip silently — corrupt files are rare but a single one
                # crashes a multi-day run; quarantine in inputs/ if you want
                # to see how many were dropped.
                continue

            samples.append(
                {
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "question": row["question"],
                    "image": image_path,
                    "ground_truth": gt,
                    "answers": answers,
                    "question_type": row.get("question_type", ""),
                }
            )
            if samples_per_answer is not None:
                answer_counts[gt_value] = answer_counts.get(gt_value, 0) + 1
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples



def assign_irrelevant_images(
    samples: list[dict],
    irrelevant_number_dir: str | Path,
    irrelevant_neutral_dir: str | Path,
    seed: int = 42,
    variants_per_sample: int = 1,
) -> list[dict]:
    rng = random.Random(seed)
    number_images = list_images(irrelevant_number_dir)
    neutral_images = list_images(irrelevant_neutral_dir)
    if not number_images:
        raise FileNotFoundError(f"No number images found in {irrelevant_number_dir}")
    if not neutral_images:
        raise FileNotFoundError(f"No neutral images found in {irrelevant_neutral_dir}")

    enriched = []
    for sample in samples:
        number_variants = _select_image_variants(number_images, variants_per_sample, rng)
        neutral_variants = _select_image_variants(neutral_images, variants_per_sample, rng)
        for variant_index, (num_img, neu_img) in enumerate(zip(number_variants, neutral_variants)):
            anchor_value = extract_first_number(num_img.stem)
            enriched.append(
                {
                    **sample,
                    "sample_instance_id": f"{sample['question_id']}_{sample['image_id']}_set{variant_index:02d}",
                    "sample_instance_index": variant_index,
                    "irrelevant_number_image": str(num_img),
                    "irrelevant_number_image_name": num_img.name,
                    "irrelevant_neutral_image": str(neu_img),
                    "irrelevant_neutral_image_name": neu_img.name,
                    "anchor_value": anchor_value,
                }
            )
    return enriched



def assign_stratified_anchors(
    samples: list[dict],
    irrelevant_number_dir: str | Path,
    seed: int = 42,
    strata: list[tuple[int, int]] = ANCHOR_DISTANCE_STRATA,
    irrelevant_number_masked_dir: str | Path | None = None,
    irrelevant_neutral_dir: str | Path | None = None,
    scheme: str = "absolute",
) -> list[dict]:
    """Per-question stratified anchor sampling.

    Each output sample carries an `anchor_strata` field: a list of dicts
    in the order of `strata`, each containing `stratum_id` ("S1".."Sn"),
    `stratum_range` (the (lo, hi) tuple), `anchor_value` (int or None),
    and `irrelevant_number_image` (str path or None).

    Only integer-stem PNG files in `irrelevant_number_dir` are eligible.
    Decimal- or text-stem files are silently ignored.

    If `irrelevant_number_masked_dir` is provided, each per-stratum entry
    also gets an `irrelevant_number_masked_image` field — the path to
    `<masked_dir>/{anchor_value}.png` if it exists, else None. The
    `anchor_value` itself is unchanged.

    If `irrelevant_neutral_dir` is provided, each output sample gets a
    sample-level `irrelevant_neutral_image` field — a randomly-chosen
    neutral image path drawn with the same seeded RNG.

    `scheme` selects the cutoff regime. "absolute" (default) uses the
    `strata` argument verbatim — backward-compatible with E5b/E5c.
    "relative" computes per-question strata via
    `compute_strata(gt, "relative")`, ignoring the `strata` argument.
    Use "relative" for wide-GT datasets like ChartQA where a fixed
    301-distance "far" stratum would lump nearly all anchors together.
    """
    # Pre-pass: validate every ground_truth before constructing the RNG, so
    # a mid-dataset failure does not leave the RNG advanced past earlier
    # samples (which would break seed-determinism on a corrected re-run).
    for sample in samples:
        gt_str = sample["ground_truth"]
        if not gt_str.lstrip("-").isdigit():
            raise ValueError(
                f"sample {sample.get('question_id')} has non-integer ground_truth {gt_str!r}; "
                "stratified mode requires integer GT"
            )

    rng = random.Random(seed)
    number_images = list_images(irrelevant_number_dir)
    if not number_images:
        raise FileNotFoundError(f"No number images found in {irrelevant_number_dir}")

    inventory_by_value: dict[int, Path] = {}
    for img in number_images:
        v = extract_first_number(img.stem)
        if v and v.lstrip("-").isdigit():
            inventory_by_value[int(v)] = img

    inventory_values = sorted(inventory_by_value.keys())

    masked_dir_path: Path | None = (
        Path(irrelevant_number_masked_dir) if irrelevant_number_masked_dir is not None else None
    )

    neutral_images: list[Path] = []
    if irrelevant_neutral_dir is not None:
        neutral_images = list_images(irrelevant_neutral_dir)
        if not neutral_images:
            raise FileNotFoundError(f"No neutral images found in {irrelevant_neutral_dir}")

    enriched: list[dict] = []
    for sample in samples:
        gt = int(sample["ground_truth"])
        per_gt_strata = compute_strata(gt, scheme) if scheme != "absolute" else strata
        anchor_values = sample_stratified_anchors(gt, inventory_values, rng, strata=per_gt_strata)
        anchor_strata: list[dict] = []
        for idx, ((lo, hi), value) in enumerate(zip(per_gt_strata, anchor_values), start=1):
            entry = {
                "stratum_id": f"S{idx}",
                "stratum_range": (lo, hi),
                "anchor_value": value,
                "irrelevant_number_image": str(inventory_by_value[value]) if value is not None else None,
            }
            if masked_dir_path is not None:
                masked_path: str | None = None
                if value is not None:
                    candidate = masked_dir_path / f"{value}.png"
                    if candidate.exists():
                        masked_path = str(candidate)
                entry["irrelevant_number_masked_image"] = masked_path
            anchor_strata.append(entry)
        enriched_sample = {
            **sample,
            "sample_instance_id": f"{sample['question_id']}_{sample['image_id']}_stratified",
            "sample_instance_index": 0,
            "anchor_strata": anchor_strata,
        }
        if neutral_images:
            enriched_sample["irrelevant_neutral_image"] = str(rng.choice(neutral_images))
        enriched.append(enriched_sample)
    return enriched


def build_conditions(sample: dict) -> Iterator[dict]:
    """Yield per-condition dicts for one sample.

    If `sample` carries `anchor_strata` (output of `assign_stratified_anchors`),
    yields target_only + one condition per non-None stratum (E5b mode). Otherwise
    yields the legacy 3 conditions (target_only + irrelevant_number + irrelevant_neutral).
    """
    if "anchor_strata" in sample:
        yield {
            **sample,
            "condition": "target_only",
            "input_images": [sample["image"]],
            "anchor_value_for_metrics": None,
            "irrelevant_type": "none",
            "irrelevant_image": None,
            "anchor_stratum_id": None,
        }
        for entry in sample["anchor_strata"]:
            if entry["anchor_value"] is None:
                continue
            yield {
                **sample,
                "condition": f"target_plus_irrelevant_number_{entry['stratum_id']}",
                "input_images": [sample["image"], entry["irrelevant_number_image"]],
                "anchor_value_for_metrics": str(entry["anchor_value"]),
                "irrelevant_type": "number",
                "irrelevant_image": entry["irrelevant_number_image"],
                "anchor_stratum_id": entry["stratum_id"],
                "anchor_stratum_range": entry["stratum_range"],
            }
        for entry in sample["anchor_strata"]:
            masked_image = entry.get("irrelevant_number_masked_image")
            if not isinstance(masked_image, str):
                continue
            yield {
                **sample,
                "condition": f"target_plus_irrelevant_number_masked_{entry['stratum_id']}",
                "input_images": [sample["image"], masked_image],
                "anchor_value_for_metrics": str(entry["anchor_value"]),
                "irrelevant_type": "number_masked",
                "irrelevant_image": masked_image,
                "anchor_stratum_id": entry["stratum_id"],
                "anchor_stratum_range": entry["stratum_range"],
            }
        neutral_image = sample.get("irrelevant_neutral_image")
        if isinstance(neutral_image, str):
            yield {
                **sample,
                "condition": "target_plus_irrelevant_neutral",
                "input_images": [sample["image"], neutral_image],
                "anchor_value_for_metrics": None,
                "irrelevant_type": "neutral",
                "irrelevant_image": neutral_image,
                "anchor_stratum_id": None,
            }
        return

    yield {
        **sample,
        "condition": "target_only",
        "input_images": [sample["image"]],
        "anchor_value_for_metrics": None,
        "irrelevant_type": "none",
        "irrelevant_image": None,
    }
    yield {
        **sample,
        "condition": "target_plus_irrelevant_number",
        "input_images": [sample["image"], sample["irrelevant_number_image"]],
        "anchor_value_for_metrics": sample["anchor_value"],
        "irrelevant_type": "number",
        "irrelevant_image": sample["irrelevant_number_image"],
    }
    yield {
        **sample,
        "condition": "target_plus_irrelevant_neutral",
        "input_images": [sample["image"], sample["irrelevant_neutral_image"]],
        "anchor_value_for_metrics": None,
        "irrelevant_type": "neutral",
        "irrelevant_image": sample["irrelevant_neutral_image"],
    }
