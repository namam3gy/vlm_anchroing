"""Bin the ground-truth label distribution for TallyQA / ChartQA / MathVista.

Bin scheme (per user spec, 2026-04-27):
  bin "x <= -101"        : v <= -101
  bin "-100 <= x <= -1"  : -100 <= v <= -1   (also catches -1 < v < 0 for floats)
  bins "0".."10"         : v in [k, k+1), k = 0..10
  bins "11-20".."91-100" : 10-wide buckets, integer floor
  bins "101-200"..       : 100-wide buckets, integer floor
  bin  "1001+"           : v >= 1001
  bin  "non-numeric"     : answer not parseable as a number
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from vlm_anchor.utils import extract_first_number

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "TallyQA (test)": ROOT / "inputs/tallyqa_test/questions.jsonl",
    "ChartQA (test)": ROOT / "inputs/chartqa_test/questions.jsonl",
    "MathVista (testmini)": ROOT / "inputs/mathvista_testmini/questions.jsonl",
}


def parse_numeric(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    tok = extract_first_number(s)
    if tok == "":
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def assign_bin(v: float) -> str:
    if math.isnan(v) or math.isinf(v):
        return "non-numeric"
    if v <= -101:
        return "x <= -101"
    if v < 0:
        return "[-100, -1]"
    if v < 11:
        return str(int(math.floor(v)))
    if v < 101:
        lo = ((int(math.floor(v)) - 1) // 10) * 10 + 1
        return f"{lo}-{lo + 9}"
    if v < 1001:
        lo = ((int(math.floor(v)) - 1) // 100) * 100 + 1
        return f"{lo}-{lo + 99}"
    return "1001+"


BIN_ORDER: list[str] = (
    ["x <= -101", "[-100, -1]"]
    + [str(i) for i in range(11)]
    + [f"{lo}-{lo + 9}" for lo in range(11, 101, 10)]
    + [f"{lo}-{lo + 99}" for lo in range(101, 1001, 100)]
    + ["1001+", "non-numeric"]
)


def count_dataset(path: Path) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = {b: 0 for b in BIN_ORDER}
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ans = rec.get("multiple_choice_answer")
            v = parse_numeric(ans)
            if v is None:
                counts["non-numeric"] += 1
            else:
                counts[assign_bin(v)] += 1
            total += 1
    return counts, total


def render_table(per_dataset: dict[str, tuple[dict[str, int], int]]) -> str:
    names = list(per_dataset.keys())
    header = "| Bin | " + " | ".join(f"{n} (n)" for n in names) + " | " + " | ".join(f"{n} (%)" for n in names) + " |"
    sep = "|---|" + "---:|" * (2 * len(names))

    def cell_n(c: dict[str, int], b: str) -> str:
        return f"{c[b]:,}" if c[b] else "—"

    def cell_p(c: dict[str, int], total: int, b: str) -> str:
        if total == 0 or c[b] == 0:
            return "—"
        pct = 100.0 * c[b] / total
        return f"{pct:.2f}"

    rows = [header, sep]
    for b in BIN_ORDER:
        if all(per_dataset[n][0][b] == 0 for n in names):
            continue
        ns = " | ".join(cell_n(per_dataset[n][0], b) for n in names)
        ps = " | ".join(cell_p(per_dataset[n][0], per_dataset[n][1], b) for n in names)
        rows.append(f"| {b} | {ns} | {ps} |")
    totals_n = " | ".join(f"{per_dataset[n][1]:,}" for n in names)
    totals_p = " | ".join("100.00" for _ in names)
    rows.append(f"| **total** | {totals_n} | {totals_p} |")
    return "\n".join(rows)


def main() -> None:
    per_dataset: dict[str, tuple[dict[str, int], int]] = {}
    for label, path in DATASETS.items():
        per_dataset[label] = count_dataset(path)

    for label, (counts, total) in per_dataset.items():
        numeric = total - counts["non-numeric"]
        print(f"\n=== {label} ===")
        print(f"  total: {total:,}    numeric: {numeric:,}    non-numeric: {counts['non-numeric']:,}")

    print()
    print(render_table(per_dataset))


if __name__ == "__main__":
    main()
