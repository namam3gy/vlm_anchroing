"""Pick top-K peak layers by ‖v_wrong[L]‖ from a calibration directory.

Usage:
  uv run python scripts/e6_pick_peak_layers.py \\
      --model llava-next-interleaved-7b \\
      --tag tally_e5e_n5k \\
      --top-k 5 \\
      --out outputs/e6_steering/llava-next-interleaved-7b/_subspace/peak_layers_tally_e5e_n5k.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True,
                    help="Calibration tag (corresponds to calibration_<tag>/).")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    calib_dir = (PROJECT_ROOT / "outputs" / "e6_steering"
                 / args.model / f"calibration_{args.tag}")
    v_path = calib_dir / "v.pt"
    if not v_path.exists():
        raise FileNotFoundError(f"v.pt not found: {v_path}")

    v = torch.load(v_path, map_location="cpu", weights_only=True)  # [2, n_layers, d]
    v_wrong = v[0].float()  # (n_layers, d)
    norms = v_wrong.norm(dim=-1).tolist()  # (n_layers,)

    ranked = sorted(enumerate(norms), key=lambda kv: -kv[1])
    top_layers = sorted(idx for idx, _ in ranked[:args.top_k])

    print(f"[setup] {args.tag} v_wrong norms by layer:")
    for L, n in enumerate(norms):
        marker = "★" if L in top_layers else " "
        print(f"  L{L:02d} {marker} ‖v‖={n:.4f}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "calibration_tag": args.tag,
        "top_k": args.top_k,
        "top_layers": top_layers,
        "all_norms": norms,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[done] top-{args.top_k} layers (sorted asc): {top_layers}")
    print(f"        saved {out_path}")


if __name__ == "__main__":
    main()
