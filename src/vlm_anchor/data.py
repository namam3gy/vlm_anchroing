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



def build_conditions(sample: dict) -> Iterator[dict]:
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
