#!/usr/bin/env bash
# B1: re-run OneVision E1d on InfoVQA with --peak-layer 14 instead of 27.
#
# Phase D §7.1-7.3 found OneVision attention peak L=14 on InfoVQA / VQAv2 vs
# L=27 on PlotQA / TallyQA. Phase E master script hardcoded --peak-layer 27,
# so on InfoVQA the ablate_peak / ablate_peak_window modes hit the wrong
# band and the eager-rerun showed flat Δ vs baseline. This script re-runs
# InfoVQA at its actual peak L=14.
#
# Wall: ~30-45 min eager on H200.

set -euo pipefail
cd /mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing

LOG_DIR="outputs/_logs/phase3_b1_infovqa_peak14"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/run_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

note() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

note "=== InfoVQA --peak-layer 14 starting ==="
uv run python scripts/causal_anchor_ablation.py \
    --model llava-onevision-qwen2-7b-ov \
    --hf-model llava-hf/llava-onevision-qwen2-7b-ov-hf \
    --peak-layer 14 \
    --config configs/experiment_e7_infographicvqa_full.yaml \
    --susceptibility-csv docs/insights/_data/susceptibility_infovqa_onevision.csv \
    --top-decile-n 100 --bottom-decile-n 100 \
    --max-new-tokens 8 \
    --attn-implementation eager \
    --output-dir "outputs/causal_ablation/llava-onevision-qwen2-7b-ov/${TS}-infovqa-peak14" \
    >> "$LOG_DIR/inference_${TS}.log" 2>&1

note "=== InfoVQA peak-14 done. Reaggregating ==="
uv run python scripts/reaggregate_paired_adoption.py --apply \
    --root "outputs/causal_ablation/llava-onevision-qwen2-7b-ov/${TS}-infovqa-peak14" \
    >> "$LOG_DIR/reaggregate_${TS}.log" 2>&1

note "=== Comparing peak-27 (original eager) vs peak-14 (this rerun) on InfoVQA ==="
python3 << EOF >> "$LOG_DIR/compare_${TS}.log" 2>&1
import json, csv
from collections import defaultdict
from pathlib import Path

# Find latest peak-27 InfoVQA eager run (timestamp 20260504-181319 from master eager rerun)
peak27 = "outputs/causal_ablation/llava-onevision-qwen2-7b-ov/20260504-181319"
peak14 = "outputs/causal_ablation/llava-onevision-qwen2-7b-ov/${TS}-infovqa-peak14"

def summarize(run_dir):
    rows = []
    p = Path(run_dir) / "predictions.jsonl"
    if not p.exists():
        return None
    for line in open(p):
        if line.strip():
            rows.append(json.loads(line))
    by_mode = defaultdict(list)
    for r in rows:
        if r['condition'] != 'target_plus_irrelevant_number': continue
        by_mode[r['mode']].append(r)
    out = {}
    for mode, recs in by_mode.items():
        n = len(recs)
        df = sum(r.get('anchor_direction_followed_moved', 0) for r in recs) / max(n, 1)
        out[mode] = {'n': n, 'df': df}
    return out

p27 = summarize(peak27)
p14 = summarize(peak14)
print(f'InfoVQA OneVision: peak-27 (master eager) vs peak-14 (B1 retry)')
print(f"{'mode':25s} {'n27':>5s} {'df27':>8s} {'n14':>5s} {'df14':>8s} {'Δ_df':>8s}")
all_modes = sorted(set(p27.keys()) | set(p14.keys()))
for mode in all_modes:
    a = p27.get(mode, {'n':0,'df':None})
    b = p14.get(mode, {'n':0,'df':None})
    delta = (b['df'] - a['df']) if a['df'] is not None and b['df'] is not None else None
    delta_s = f"{delta:+.3f}" if delta is not None else "?"
    df27_s = f"{a['df']:.3f}" if a['df'] is not None else "?"
    df14_s = f"{b['df']:.3f}" if b['df'] is not None else "?"
    print(f"{mode:25s} {a['n']:>5d} {df27_s:>8s} {b['n']:>5d} {df14_s:>8s} {delta_s:>8s}")
EOF

note "=== ALL DONE. Compare log: $LOG_DIR/compare_${TS}.log ==="
