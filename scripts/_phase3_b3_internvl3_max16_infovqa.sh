#!/usr/bin/env bash
# B3: re-run InternVL3-8b on InfoVQA with --max-new-tokens 16 (vs the panel
# default 8) to test the format-locking hypothesis from B2.
#
# Hypothesis: --max-new-tokens 8 truncates "Based on..." prose preamble; the
# parsed first numeric token then varies with prose noise rather than
# anchor pull, which is what produces InternVL3's H7 reversal on chart-text
# datasets (Δ Q4 − Q1 = −0.154 wrong-base on InfoVQA, n=861).
# With --max-new-tokens 16, the post-prose answer should be captured and the
# normal Q4 > Q1 monotonicity should restore.
#
# Smoke-test scale: --max-samples 250 (vs the full 1147). With em(b)~0.19
# this gives ~200 wrong-base sids — enough to detect a 0.10+ reversal
# delta. Keeps the run under ~30 min on H200.

set -euo pipefail
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

LOG_DIR="outputs/_logs/phase3_b3_internvl3_max16"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
SUMMARY_LOG="$LOG_DIR/run_${TS}.log"
exec > >(tee -a "$SUMMARY_LOG") 2>&1

note() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

OUT_DIR="outputs/_diag/internvl3_infovqa_max16/${TS}"

note "=== InternVL3-8b InfoVQA max_new_tokens=16 starting ==="
uv run python scripts/run_experiment.py \
    --config configs/experiment_e7_infographicvqa_full.yaml \
    --models internvl3-8b \
    --max-samples 250 \
    --max-new-tokens 16 \
    --output-dir "$OUT_DIR" \
    >> "$LOG_DIR/inference_${TS}.log" 2>&1

note "=== Inference done. Reaggregating ==="
uv run python scripts/reaggregate_paired_adoption.py --apply \
    --root "$OUT_DIR" \
    >> "$LOG_DIR/reaggregate_${TS}.log" 2>&1

note "=== Recompute span proxies + comparison ==="
uv run python scripts/recompute_answer_span_confidence.py \
    --root "$OUT_DIR" \
    >> "$LOG_DIR/spans_${TS}.log" 2>&1

# Compare wrong-base Q4 df between max=8 (existing run) and max=16 (this run)
python3 << EOF >> "$LOG_DIR/compare_${TS}.log" 2>&1
import json
from collections import defaultdict
from pathlib import Path
from glob import glob

def load(run_dir):
    p = next(iter(glob(f"{run_dir}/predictions.jsonl")), None)
    if p is None:
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows

# Old run with max=8
old = []
for d in glob('outputs/experiment_e7_infographicvqa_full/internvl3-8b/*/predictions.jsonl'):
    rows = [json.loads(l) for l in open(d) if l.strip()]
    old.extend(rows)
print(f'old (max=8): {len(old)} records')

# New run with max=16
new = load('$OUT_DIR/internvl3-8b')
if new is None:
    new_dirs = glob('$OUT_DIR/internvl3-8b/*')
    if new_dirs:
        new_dirs = sorted(new_dirs)
        new_p = f'{new_dirs[-1]}/predictions.jsonl'
        new = [json.loads(l) for l in open(new_p) if l.strip()]
print(f'new (max=16): {len(new) if new else 0} records')

def summarize(rows, label):
    if not rows: return
    # Build wrong-base / correct-base S1 a-arm cells
    by_sid_b = {}
    for r in rows:
        if r['condition'] == 'target_only':
            by_sid_b[r['sample_instance_id']] = r
    a_recs = [r for r in rows if r['condition'].startswith('target_plus_irrelevant_number') and not r['condition'].endswith('masked_S1')]
    correct = []
    wrong = []
    for r in a_recs:
        b = by_sid_b.get(r['sample_instance_id'])
        if b is None: continue
        em_b = b.get('exact_match', 0)
        df_moved = r.get('anchor_direction_followed_moved', 0) or 0
        if em_b > 0.5:
            correct.append(df_moved)
        else:
            wrong.append(df_moved)
    print(f'  {label}: wrong-base n={len(wrong)} df={sum(wrong)/max(len(wrong),1):.3f}; correct-base n={len(correct)} df={sum(correct)/max(len(correct),1):.3f}')

summarize(old, 'max=8 (existing)')
summarize(new, 'max=16 (B3 rerun)')
print('\nNote: full quartile comparison requires re-running scripts/analyze_confidence_anchoring.py.')
EOF

note "=== ALL DONE ==="
