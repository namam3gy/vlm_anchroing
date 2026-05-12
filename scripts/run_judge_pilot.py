"""Run the judge anchoring pilot end-to-end.

For each (judge, sample, arm in [b, a, m]):
  1. Build the prompt by formatting the YAML template with the sample's
     prompt + chosen_response.
  2. Call the judge with the per-arm image list.
  3. Append a record to predictions.jsonl.

Usage:
    uv run python scripts/run_judge_pilot.py \
        --config configs/judge_pilot.yaml \
        --judge-id gpt-4o-2024-11-20 gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml

from vlm_anchor.judge_clients import (
    GeminiJudgeClient,
    JudgeResponse,
    OpenAIJudgeClient,
)
from vlm_anchor.judge_pilot_data import iter_pilot_arms, load_manifest_jsonl


def _build_judge(judge_cfg: dict):
    vendor = judge_cfg["vendor"]
    if vendor == "openai":
        return OpenAIJudgeClient(
            model_name=judge_cfg["model_name"],
            max_output_tokens=int(judge_cfg.get("max_output_tokens", 8)),
            temperature=float(judge_cfg.get("temperature", 0.0)),
        )
    if vendor == "google":
        return GeminiJudgeClient(
            model_name=judge_cfg["model_name"],
            max_output_tokens=int(judge_cfg.get("max_output_tokens", 8)),
            temperature=float(judge_cfg.get("temperature", 0.0)),
        )
    raise ValueError(f"Unknown vendor: {vendor!r}")


def _existing_keys(predictions_path: Path) -> set[tuple[str, str]]:
    if not predictions_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with predictions_path.open() as f:
        for line in f:
            r = json.loads(line)
            keys.add((r["sample_id"], r["arm"]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--judge-id", nargs="+", required=True)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap for smoke runs")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Append to an existing predictions.jsonl rather than starting fresh")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    manifest = load_manifest_jsonl(Path(cfg["dataset"]["manifest_out"]))
    if args.max_samples is not None:
        manifest = manifest[: args.max_samples]
    print(f"Loaded {len(manifest)} pilot samples")

    judges_by_id = {j["id"]: j for j in cfg["judges"]}
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(cfg["output"]["root"])

    for judge_id in args.judge_id:
        if judge_id not in judges_by_id:
            raise KeyError(f"Judge id {judge_id!r} not in config")
        judge_cfg = judges_by_id[judge_id]
        client = _build_judge(judge_cfg)

        out_dir = (args.resume_from.parent if args.resume_from else out_root / judge_id / timestamp)
        out_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = (args.resume_from if args.resume_from else out_dir / "predictions.jsonl")
        seen = _existing_keys(predictions_path)
        print(f"[{judge_id}] writing to {predictions_path} (resuming over {len(seen)} prior records)")

        with predictions_path.open("a") as out:
            for sample in manifest:
                for arm in iter_pilot_arms(sample):
                    key = (sample.sample_id, arm["arm"])
                    if key in seen:
                        continue
                    prompt = cfg["prompt"]["template"].format(
                        prompt=sample.prompt,
                        response=sample.chosen_response,
                    )
                    t0 = time.perf_counter()
                    try:
                        response: JudgeResponse = client.score(images=arm["images"], prompt=prompt)
                        err = None
                    except Exception as exc:  # noqa: BLE001 — log and continue
                        response = JudgeResponse(score=None, raw="")
                        err = repr(exc)
                    rec = {
                        "judge_id": judge_id,
                        "sample_id": sample.sample_id,
                        "arm": arm["arm"],
                        "n_images": len(arm["images"]),
                        "score": response.score,
                        "raw": response.raw,
                        "elapsed_s": round(time.perf_counter() - t0, 3),
                        "error": err,
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"  {judge_id} {sample.sample_id} {arm['arm']:>1} -> score={response.score} ({rec['elapsed_s']}s)" + (f" err={err}" if err else ""))

        summary = {
            "judge_id": judge_id,
            "timestamp": timestamp,
            "manifest_n": len(manifest),
            "predictions_path": str(predictions_path),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
