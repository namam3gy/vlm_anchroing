"""Build site/data/demo.json + site/assets/img/ from outputs/ predictions."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Display id -> outputs/ subdirectory name
MAIN_PANEL: dict[str, str] = {
    "llava-onevision-7b":  "llava-onevision-qwen2-7b-ov",
    "qwen2.5-vl-7b":       "qwen2.5-vl-7b-instruct",
    "gemma-3-27b":         "gemma3-27b-it",
    "internvl3-8b":        "internvl3-8b",
    "gemma-3-4b":          "gemma3-4b-it",
}
MAIN_PANEL_LABELS: dict[str, str] = {
    "llava-onevision-7b":  "LLaVA-OneVision 7B (main)",
    "qwen2.5-vl-7b":       "Qwen2.5-VL 7B",
    "gemma-3-27b":         "Gemma-3 27B",
    "internvl3-8b":        "InternVL3 8B",
    "gemma-3-4b":          "Gemma-3 4B",
}
FORBIDDEN_OUTPUT_SUBTREE = "before_C_form"

# Reverse map: outputs/ dir name -> display id
DIR_TO_DISPLAY: dict[str, str] = {v: k for k, v in MAIN_PANEL.items()}

# Inputs subdirectory -> dataset display name
DATASET_LABELS: dict[str, str] = {
    "vqav2_number_val": "VQAv2",
    "chartqa_test": "ChartQA",
    "plotqa_test": "PlotQA",
    "mathvista_testmini": "MathVista",
    "infographicvqa_val": "InfographicsVQA",
    "tallyqa_test": "TallyQA",
}

CONDITION_BASE_TO_CODE: dict[str, str] = {
    "target_only": "b",
    "target_plus_irrelevant_number": "a",
    "target_plus_irrelevant_number_masked": "m",
    "target_plus_irrelevant_neutral": "d",
}


def _condition_code(label: str, anchor_stratum: str) -> str | None:
    """Map a CSV condition string to b/a/m/d, respecting anchor_stratum.

    For stratified labels (e.g. ``target_plus_irrelevant_number_S1``), accept
    only the configured stratum; for non-stratified labels accept directly.
    """
    if label in CONDITION_BASE_TO_CODE:
        return CONDITION_BASE_TO_CODE[label]
    m = re.match(r"^(target_plus_irrelevant_(?:number|number_masked|neutral))_S\d+$", label)
    if not m:
        return None
    base, suffix = m.group(1), label.rsplit("_", 1)[-1]
    if suffix != anchor_stratum:
        return None
    return CONDITION_BASE_TO_CODE.get(base)


def _infer_dataset(input_image_paths: str) -> str:
    """Extract dataset display name from a Python-list-style stringified path."""
    if not isinstance(input_image_paths, str):
        return "unknown"
    for key, label in DATASET_LABELS.items():
        if f"/inputs/{key}/" in input_image_paths or f"\\inputs\\{key}\\" in input_image_paths:
            return label
    return "unknown"


def _to_int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _first_path(input_image_paths: str) -> str | None:
    """Extract the first absolute path from a stringified list.

    Live CSVs serialise this column as a JSON-style array with double
    quotes (``["/abs/path", ...]``); some legacy / synthetic rows use
    Python's ``repr()`` form with single quotes. Parse JSON first, then
    fall back to a quote-agnostic regex.
    """
    if not isinstance(input_image_paths, str):
        return None
    text = input_image_paths.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            return first if isinstance(first, str) else None
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"['\"]([^'\"]+)['\"]", text)
    return m.group(1) if m else None


def _iter_prediction_csvs(outputs_root: Path) -> Iterable[Path]:
    if not outputs_root.exists():
        return
    for path in outputs_root.rglob("predictions.csv"):
        rel = path.relative_to(outputs_root)
        if rel.parts and rel.parts[0] == FORBIDDEN_OUTPUT_SUBTREE:
            continue
        yield path


def _latest_runs(csv_paths: Iterable[Path]) -> dict[tuple[str, str], Path]:
    """For each (experiment, model_dir), pick the run with the largest CSV."""
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in csv_paths:
        parts = path.parts
        try:
            exp_idx = parts.index("outputs") + 1
        except ValueError:
            continue
        if len(parts) < exp_idx + 4:
            continue
        experiment, model_dir = parts[exp_idx], parts[exp_idx + 1]
        if model_dir == "analysis":
            continue
        grouped[(experiment, model_dir)].append(path)

    chosen: dict[tuple[str, str], Path] = {}
    for key, paths in grouped.items():
        # Largest CSV wins — avoids smoke-run pollution (many tiny runs accumulate
        # under outputs/<exp>/<model>/, an alphabetical "latest" rule would shadow
        # the canonical full run).
        chosen[key] = max(paths, key=lambda p: p.stat().st_size)
    return chosen


def load_predictions(
    *, outputs_root: Path, anchor_stratum: str = "S1",
) -> dict[str, dict[str, dict]]:
    """Return {display_model_id: {sample_id: {b, a, m, d, meta}}}.

    ``meta`` carries question, gt, anchor, dataset, and the b-arm input image
    path so the caller can copy assets without re-reading the CSV.
    """
    runs = _latest_runs(_iter_prediction_csvs(outputs_root))
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)

    for (_, model_dir), csv_path in runs.items():
        display_id = DIR_TO_DISPLAY.get(model_dir)
        if display_id is None:
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        for sid, group in df.groupby("sample_instance_id"):
            sample: dict = {"meta": {}}
            for _, row in group.iterrows():
                code = _condition_code(str(row["condition"]), anchor_stratum)
                if code is None:
                    continue
                pred = _to_int(row["prediction"])
                if pred is None:
                    continue
                sample[code] = pred
                if code == "b":
                    sample["meta"] = {
                        "question": str(row.get("question", "")),
                        "gt": _to_int(row.get("ground_truth")),
                        "anchor": _to_int(row.get("anchor_value")),
                        "dataset": _infer_dataset(str(row.get("input_image_paths", ""))),
                        "target_image_path": _first_path(row.get("input_image_paths", "")),
                    }
                elif code == "a" and sample["meta"].get("anchor") is None and "question" in sample["meta"]:
                    sample["meta"]["anchor"] = _to_int(row.get("anchor_value"))
            if "b" not in sample or "question" not in sample["meta"]:
                continue
            by_model[display_id][str(sid)] = sample
    return dict(by_model)


REQUIRED_CONDITIONS = ("b", "a", "m", "d")


def eligible_samples(by_model: dict[str, dict[str, dict]]) -> list[str]:
    """Return sample ids that have every (main-panel model × b/a/m/d).

    Also drops samples whose anchor value equals the ground truth — those
    are degenerate for an anchoring demo because adopting the anchor and
    being correct are indistinguishable.
    """
    if not all(mid in by_model for mid in MAIN_PANEL):
        missing = [mid for mid in MAIN_PANEL if mid not in by_model]
        print(f"WARN: missing models in outputs/: {missing}", file=sys.stderr)
        return []
    sample_ids = set(by_model[next(iter(MAIN_PANEL))])
    for mid in MAIN_PANEL:
        sample_ids &= set(by_model[mid])
    eligible = []
    for sid in sorted(sample_ids):
        ok = True
        for mid in MAIN_PANEL:
            sample = by_model[mid][sid]
            if not all(cond in sample for cond in REQUIRED_CONDITIONS):
                ok = False
                break
        if not ok:
            continue
        ref_meta = by_model[next(iter(MAIN_PANEL))][sid]["meta"]
        gt = ref_meta.get("gt")
        anchor = ref_meta.get("anchor")
        if gt is None or anchor is None or gt == anchor:
            continue
        # Require at least one model to actually *move* toward the anchor on
        # the a-arm — sample is otherwise flat, which makes the demo
        # unconvincing.
        n_movers = sum(
            1
            for mid in MAIN_PANEL
            if by_model[mid][sid]["a"] == anchor and by_model[mid][sid]["b"] != anchor
        )
        if n_movers < 1:
            continue
        # Require at least one model to walk the full 4-arm textbook signature
        # b == gt → a == anchor → m == gt → d == gt. This is the only pattern
        # where both controls (digit-pixel mask and neutral 2-image
        # distractor) fire simultaneously, isolating the digit as the cause
        # rather than 2-image distraction or anchor-image background. Without
        # this filter the demo can show samples whose anchor effect is
        # ambiguous between the three explanations.
        n_full = sum(
            1
            for mid in MAIN_PANEL
            if by_model[mid][sid]["b"] == gt
            and by_model[mid][sid]["a"] == anchor
            and by_model[mid][sid]["m"] == gt
            and by_model[mid][sid]["d"] == gt
        )
        if n_full < 1:
            continue
        eligible.append(sid)
    return eligible


def score_sample(by_model: dict[str, dict[str, dict]], sample_id: str) -> float:
    """Higher score = stronger anchoring signature.

    The textbook 4-arm signature is

        b == gt   AND   a == anchor   AND   m == gt   AND   d == gt

    — the model is base-correct, gets pulled to the digit anchor on the
    a-arm, recovers when the digit pixels are masked (m), and recovers
    again under the neutral 2-image distractor (d). Both controls firing
    is what isolates the digit pixel as the causal feature. We weight
    each model by how much of that 4-arm trajectory it shows:

    +15 per model with the full 4-arm signature (b/a/m/d all match)
    +10 per model with the 3-arm subset b/a/m (d may differ)
     +4 per model with b == gt and a == anchor (no m recovery)
     +2 per model that adopts the anchor on a (b may already be wrong)
     +2 per model that is base-correct (b == gt) — tiebreaker
     +5 if ≥2 models show the full 4-arm signature (demo-gold)
     +2 if ≥1 model shows the full 4-arm signature
     +1 if ≥3 models show at least the 3-arm subset
    """
    score = 0.0
    full_4 = 0
    full_3 = 0
    for mid in MAIN_PANEL:
        s = by_model[mid][sample_id]
        gt = s["meta"]["gt"]
        anchor = s["meta"]["anchor"]
        if gt is None or anchor is None:
            continue
        b_correct = s["b"] == gt
        a_pulled = s["a"] == anchor
        m_recovers = s["m"] == gt
        d_recovers = s["d"] == gt
        if b_correct and a_pulled and m_recovers and d_recovers:
            score += 15
            full_4 += 1
        elif b_correct and a_pulled and m_recovers:
            score += 10
            full_3 += 1
        elif b_correct and a_pulled:
            score += 4
        elif a_pulled:
            score += 2
        if b_correct:
            score += 2
    if full_4 >= 2:
        score += 5
    elif full_4 >= 1:
        score += 2
    elif full_3 >= 3:
        score += 1
    return score


def pick_samples(scored: dict[str, tuple[str, float]], n: int) -> list[str]:
    """Round-robin top-N picker keyed on dataset, then score.

    ``scored`` maps sample_id -> (dataset, score). While unseen datasets
    are still represented in the remaining pool, only candidates from
    those unseen datasets are eligible for the next pick (highest score
    wins). Once every dataset has contributed at least one sample, the
    picker falls back to highest score regardless of dataset.

    A small score bonus is *not* used because the score range on real
    data spans several dozen points, far larger than any reasonable
    bonus, so a score-bonus rule reduces to "always pick the highest"
    on real data.
    """
    if not scored:
        return []
    remaining = dict(scored)
    all_datasets = {ds for ds, _ in scored.values()}
    chosen: list[str] = []
    chosen_datasets: set[str] = set()
    while remaining and len(chosen) < n:
        unseen_datasets = all_datasets - chosen_datasets
        if unseen_datasets:
            pool = {sid: v for sid, v in remaining.items() if v[0] in unseen_datasets}
        else:
            pool = remaining
        sid, (ds, _) = max(pool.items(), key=lambda item: item[1][1])
        chosen.append(sid)
        chosen_datasets.add(ds)
        remaining.pop(sid)
    return chosen


def _copy_image(src: Path, dst_stem: Path, max_edge: int) -> str:
    """Copy ``src`` to ``dst_stem.<ext>``, resizing if the long edge exceeds
    ``max_edge``. Preserves the source format: PNG → PNG (lossless), JPEG /
    everything-else → JPEG quality 92. Returns the destination filename so
    the caller can record the actual extension in demo.json.

    Lossless PNG path matters for the digit anchor / masked / neutral
    images: they have sharp glyph edges that JPEG q85 visibly blurred in
    the first iteration.
    """
    dst_stem.parent.mkdir(parents=True, exist_ok=True)
    is_png = src.suffix.lower() == ".png"
    with Image.open(src) as im:
        if is_png:
            im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "P") else "RGB")
        else:
            im = im.convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        if long_edge > max_edge:
            scale = max_edge / long_edge
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if is_png:
            dst = dst_stem.with_suffix(".png")
            im.save(dst, "PNG", optimize=True)
        else:
            dst = dst_stem.with_suffix(".jpg")
            im.save(dst, "JPEG", quality=92)
    return dst.name


def _resolve_anchor_image(inputs_root: Path, anchor_value: int) -> Path:
    candidate = inputs_root / "irrelevant_number" / f"{anchor_value}.png"
    if not candidate.exists():
        raise FileNotFoundError(f"anchor image not found: {candidate}")
    return candidate


def _resolve_masked_image(inputs_root: Path, anchor_value: int) -> Path:
    candidate = inputs_root / "irrelevant_number_masked" / f"{anchor_value}.png"
    if not candidate.exists():
        raise FileNotFoundError(f"masked image not found: {candidate}")
    return candidate


def _resolve_neutral_image(inputs_root: Path) -> Path:
    folder = inputs_root / "irrelevant_neutral"
    candidates = sorted(folder.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"no neutral images under {folder}")
    return candidates[0]


def build_site_artifacts(
    *,
    chosen: list[str],
    by_model: dict[str, dict[str, dict]],
    inputs_root: Path,
    site_root: Path,
    anchor_stratum: str,
    max_image_px: int,
) -> dict:
    data_dir = site_root / "data"
    img_root = site_root / "assets" / "img"
    if img_root.exists():
        shutil.rmtree(img_root)
    img_root.mkdir(parents=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    samples_out: list[dict] = []
    for sid in chosen:
        ref = by_model[next(iter(MAIN_PANEL))][sid]
        meta = ref["meta"]
        anchor_value = meta["anchor"]
        target_src = Path(meta["target_image_path"])
        anchor_src = _resolve_anchor_image(inputs_root, anchor_value)
        masked_src = _resolve_masked_image(inputs_root, anchor_value)
        neutral_src = _resolve_neutral_image(inputs_root)

        sample_img_dir = img_root / sid
        image_paths: dict[str, str] = {}
        for kind, src in (
            ("target", target_src), ("anchor", anchor_src),
            ("masked", masked_src), ("neutral", neutral_src),
        ):
            fname = _copy_image(src, sample_img_dir / kind, max_image_px)
            image_paths[kind] = f"assets/img/{sid}/{fname}"

        predictions = {
            mid: {c: by_model[mid][sid][c] for c in REQUIRED_CONDITIONS}
            for mid in MAIN_PANEL
        }
        samples_out.append({
            "id": sid,
            "dataset": meta["dataset"],
            "question": meta["question"],
            "gt": meta["gt"],
            "anchor": anchor_value,
            "images": image_paths,
            "predictions": predictions,
        })

    demo = {
        "models": [
            {"id": mid, "label": MAIN_PANEL_LABELS[mid]} for mid in MAIN_PANEL
        ],
        "samples": samples_out,
        "anchor_stratum": anchor_stratum,
    }
    (data_dir / "demo.json").write_text(json.dumps(demo, indent=2))
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--inputs-root", type=Path, default=PROJECT_ROOT / "inputs")
    p.add_argument("--site-root", type=Path, default=PROJECT_ROOT / "site")
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--anchor-stratum", default="S1",
                   help="Stratum suffix used for the anchor (a) and masked (m) conditions.")
    p.add_argument("--max-image-px", type=int, default=1280,
                   help="Long-edge pixel cap for copied images. Source PNGs "
                        "stay PNG (lossless); source JPEGs save as JPEG q92.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"build_demo_data: outputs={args.outputs_root} site={args.site_root}", file=sys.stderr)

    by_model = load_predictions(
        outputs_root=args.outputs_root, anchor_stratum=args.anchor_stratum,
    )
    eligible = eligible_samples(by_model)
    if not eligible:
        print("ERROR: no samples eligible across the full main panel.", file=sys.stderr)
        print("Check coverage with `find outputs -name predictions.csv`.", file=sys.stderr)
        return 1

    scored = {
        sid: (
            by_model[next(iter(MAIN_PANEL))][sid]["meta"]["dataset"],
            score_sample(by_model, sid),
        )
        for sid in eligible
    }
    chosen = pick_samples(scored, args.num_samples)
    if len(chosen) < args.num_samples:
        print(
            f"WARN: only {len(chosen)} samples eligible; spec requires "
            f"{args.num_samples}. Proceeding with the smaller set.",
            file=sys.stderr,
        )

    demo = build_site_artifacts(
        chosen=chosen, by_model=by_model,
        inputs_root=args.inputs_root, site_root=args.site_root,
        anchor_stratum=args.anchor_stratum, max_image_px=args.max_image_px,
    )
    print(
        f"OK: {len(demo['samples'])} samples × {len(demo['models'])} models written to "
        f"{(args.site_root / 'data' / 'demo.json')}",
        file=sys.stderr,
    )
    for s in demo["samples"]:
        print(f"  - {s['id']} [{s['dataset']}] anchor={s['anchor']} gt={s['gt']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
