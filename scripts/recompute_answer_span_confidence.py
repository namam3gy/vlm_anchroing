"""Post-hoc add answer-span aggregate confidence proxies to predictions.jsonl.

Walks `token_info` per record, identifies the answer span tokens by forward
digit accumulation that matches the parsed `prediction`, and writes 14 new
fields (logit/prob/log-prob/entropy variants). Also adds PlotQA-relaxed (5%)
and InfoVQA-ANLS (>=0.5) base-correctness columns on the matching exp dirs so
paper supplementary acc(b) numbers can be reported alongside strict
exact_match.

Why: Qwen2 / Llava-Interleave tokenizers split multi-digit numbers per digit,
so the canonical `answer_token_logit` and `answer_token_probability` fields
(captured by single-token text matching) are None on 75-89 % of multi-digit
predictions on PlotQA / InfoVQA / ChartQA. The aggregate proxies recover
sequence-level confidence semantics from token_info that is already saved.

Idempotent. Skips records where all new fields are already present unless
--force. Backs up the original predictions.jsonl as
predictions.pre_span_proxies.bak.jsonl on first rewrite.

Usage:
  uv run python scripts/recompute_answer_span_confidence.py
  uv run python scripts/recompute_answer_span_confidence.py --root outputs/experiment_e7_plotqa_full
  uv run python scripts/recompute_answer_span_confidence.py --force
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from vlm_anchor.utils import extract_first_number  # noqa: E402
EPS = 1e-12

NEW_FIELDS = (
    "answer_span_n_tokens",
    "answer_span_token_indices",
    "answer_span_first_logit",
    "answer_span_min_logit",
    "answer_span_mean_logit",
    "answer_span_first_prob",
    "answer_span_min_prob",
    "answer_span_mean_prob",
    "answer_span_geo_mean_prob",
    "answer_span_log_prob_sum",
    "answer_span_log_prob_mean",
    "answer_span_cross_entropy",
    "answer_span_entropy_sum",
    "answer_span_entropy_mean",
)


def _digits_only(text: str | None) -> str:
    if not text:
        return ""
    return "".join(c for c in str(text) if c.isdigit())


def find_answer_span(token_info: list[dict], prediction) -> tuple[int, int] | None:
    """Try every contiguous (start, end) span; return the first whose joined
    token text contains the prediction's digit-only string as a contiguous
    digit substring.

    Handles bare digit emission ('1','0','0' for prediction='100'), JSON
    wrapper tokens between digits, single multi-digit tokens ('12'), and
    decimals where non-digit tokens like '.' are part of the answer span
    ('75.5' → tokens ['7','5','.','5']).

    O(n²) but n ≤ max_new_tokens (≤ 16) so trivially cheap.
    """
    if not token_info:
        return None

    # Strategy A: digit-based forward span match.
    target_digits = _digits_only(prediction)
    if target_digits:
        n = len(token_info)
        for start in range(n):
            accum_text = ""
            for end in range(start + 1, n + 1):
                tok = token_info[end - 1]
                accum_text += (tok.get("token_text") or "") if isinstance(tok, dict) else ""
                accum_digits = _digits_only(accum_text)
                if accum_digits == target_digits:
                    return (start, end)
                if not target_digits.startswith(accum_digits):
                    break  # diverged

    # Strategy B: word-number fallback. extract_first_number turns "two"→"2",
    # so a token whose text decodes to the same parsed prediction is the span.
    # Scan all single-token candidates and adjacent pairs ("twenty one" = "21").
    target_str = str(prediction).strip() if prediction is not None else ""
    if not target_str:
        return None
    n = len(token_info)
    for i, tok in enumerate(token_info):
        text = (tok.get("token_text") or "") if isinstance(tok, dict) else ""
        if not text.strip():
            continue
        if extract_first_number(text) == target_str:
            return (i, i + 1)
    # Two-token combos for "twenty one" / "one hundred" style.
    for i in range(n - 1):
        a = (token_info[i].get("token_text") or "") if isinstance(token_info[i], dict) else ""
        b = (token_info[i + 1].get("token_text") or "") if isinstance(token_info[i + 1], dict) else ""
        if not a.strip() or not b.strip():
            continue
        joined = a + b
        if extract_first_number(joined) == target_str:
            return (i, i + 2)
    return None


def compute_span_proxies(token_info: list[dict], span: tuple[int, int] | None) -> dict:
    out: dict = {f: None for f in NEW_FIELDS}
    if span is None or not token_info:
        return out
    start, end = span
    sub = token_info[start:end]
    probs = [float(t.get("probability") or 0.0) for t in sub if isinstance(t, dict)]
    logits = [float(t.get("logit") or 0.0) for t in sub if isinstance(t, dict)]
    if not probs or not logits or len(probs) != len(logits):
        return out
    n = len(sub)
    log_probs = [math.log(p + EPS) for p in probs]
    neg_p_log_p = [-(p * math.log(p + EPS)) for p in probs]
    out["answer_span_n_tokens"] = n
    out["answer_span_token_indices"] = [start, end]
    out["answer_span_first_logit"] = logits[0]
    out["answer_span_min_logit"] = min(logits)
    out["answer_span_mean_logit"] = sum(logits) / n
    out["answer_span_first_prob"] = probs[0]
    out["answer_span_min_prob"] = min(probs)
    out["answer_span_mean_prob"] = sum(probs) / n
    out["answer_span_log_prob_sum"] = sum(log_probs)
    out["answer_span_log_prob_mean"] = sum(log_probs) / n
    out["answer_span_geo_mean_prob"] = math.exp(out["answer_span_log_prob_mean"])
    out["answer_span_cross_entropy"] = -out["answer_span_log_prob_mean"]
    out["answer_span_entropy_sum"] = sum(neg_p_log_p)
    out["answer_span_entropy_mean"] = sum(neg_p_log_p) / n
    return out


# ----------------------------------------------------------------------------
# Dataset-specific supplementary metrics (PlotQA-relaxed, InfoVQA-ANLS)
# ----------------------------------------------------------------------------

def _parse_int(s) -> int | None:
    if s is None:
        return None
    txt = str(s).strip()
    if not txt:
        return None
    m = re.search(r"-?\d+", txt)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def plotqa_relaxed_correct(pred, gt, tol: float = 0.05) -> int | None:
    """PlotQA official metric: |pred - gt| / |gt| < tol → correct.
    Returns 1/0 for numeric pairs, None when pred/gt not parseable."""
    p = _parse_int(pred)
    g = _parse_int(gt)
    if p is None or g is None:
        return None
    if g == 0:
        return int(p == 0)
    return int(abs(p - g) / abs(g) < tol)


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def infovqa_anls(pred, gt, threshold: float = 0.5) -> float | None:
    """InfoVQA official metric: 1 - NLD(pred, gt) / max(|pred|, |gt|),
    threshold-clipped at the given value."""
    if pred is None or gt is None:
        return None
    s1 = str(pred).strip().lower()
    s2 = str(gt).strip().lower()
    if not s1 or not s2:
        return None
    m = max(len(s1), len(s2))
    nld = _levenshtein(s1, s2) / max(m, 1)
    sim = 1.0 - nld
    return sim if sim >= threshold else 0.0


def supplementary_metrics(record: dict, exp_dir_name: str) -> dict:
    out: dict = {}
    pred = record.get("prediction")
    gt = record.get("ground_truth")
    name = exp_dir_name.lower()
    if "plotqa" in name or "chartqa" in name:
        out["plotqa_relaxed_correct"] = plotqa_relaxed_correct(pred, gt, tol=0.05)
    if "infographicvqa" in name or "infovqa" in name:
        out["infovqa_anls"] = infovqa_anls(pred, gt, threshold=0.5)
    return out


# ----------------------------------------------------------------------------
# File I/O
# ----------------------------------------------------------------------------

def _load_first_record(path: Path) -> dict | None:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def already_processed(path: Path) -> bool:
    rec = _load_first_record(path)
    if rec is None:
        return False
    return all(f in rec for f in NEW_FIELDS)


def discover_predictions(root: Path) -> Iterable[Path]:
    """Yield every predictions.jsonl under root that is not in an analysis,
    legacy archive, or steering output subtree.

    Skip rules:
      - parts named 'analysis' / '_subspace' / '_logs'
      - any part starting with '_legacy_' (archived runs preserved for
        evidence; not part of any active analysis pipeline)
      - parts starting with 'before_' (e.g. before_C_form pre-refactor backup)
      - the e6_steering tree (calibration / sweep predictions are processed
        by their own pipeline, not the behavioural confidence one)
      - .pre_span_proxies.bak.jsonl backups
    """
    skip_parts = {"analysis", "_subspace", "_logs", "e6_steering"}
    for p in root.rglob("predictions.jsonl"):
        parts = p.parts
        if any(part in skip_parts for part in parts):
            continue
        if any(part.startswith("_legacy_") or part.startswith("before_") for part in parts):
            continue
        if p.name.endswith(".pre_span_proxies.bak.jsonl"):
            continue
        yield p


def process_file(path: Path, force: bool) -> tuple[int, int, int]:
    """Returns (n_records, n_span_found, n_span_missing)."""
    if not force and already_processed(path):
        return (0, 0, 0)

    backup = path.with_suffix(".pre_span_proxies.bak.jsonl")
    if not backup.exists():
        shutil.copy2(path, backup)

    # Determine exp_dir name for supplementary metric routing.
    # Path layout: outputs/<exp_dir>/<model>/<ts>/predictions.jsonl
    try:
        exp_dir_name = path.parents[2].name
    except IndexError:
        exp_dir_name = ""

    tmp = path.with_suffix(".rewriting.tmp.jsonl")
    n = 0
    found = 0
    missing = 0
    with backup.open() as fh_in, tmp.open("w") as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fh_out.write(line + "\n")
                continue
            n += 1
            ti = rec.get("token_info") or []
            pred = rec.get("prediction")
            span = find_answer_span(ti, pred)
            if span is not None:
                found += 1
            else:
                missing += 1
            rec.update(compute_span_proxies(ti, span))
            rec.update(supplementary_metrics(rec, exp_dir_name))
            fh_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return (n, found, missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs",
                    help="Root dir to scan for predictions.jsonl. Default: outputs/")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if all new fields are already present.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files that would be processed; do not modify.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    paths = sorted(discover_predictions(root))
    if not paths:
        print(f"[discover] no predictions.jsonl under {root}")
        return

    print(f"[discover] {len(paths)} predictions.jsonl files under {root}")
    if args.dry_run:
        for p in paths:
            status = "skip (already processed)" if (not args.force and already_processed(p)) else "process"
            print(f"  {status}: {p.relative_to(PROJECT_ROOT)}")
        return

    total_n = total_found = total_missing = total_skipped = 0
    for p in paths:
        rel = p.relative_to(PROJECT_ROOT)
        if not args.force and already_processed(p):
            print(f"[skip] {rel}  (already has span proxies)")
            total_skipped += 1
            continue
        n, found, missing = process_file(p, force=args.force)
        pct = 100.0 * found / max(n, 1)
        print(f"[done] {rel}  n={n}  span_found={found} ({pct:.1f}%)  span_missing={missing}")
        total_n += n
        total_found += found
        total_missing += missing

    print(f"\n[summary] processed={len(paths) - total_skipped}  skipped={total_skipped}")
    if total_n:
        print(f"[summary] records={total_n}  span_found={total_found} ({100.0*total_found/total_n:.1f}%)  "
              f"span_missing={total_missing}")


if __name__ == "__main__":
    main()
