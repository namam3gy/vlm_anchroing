"""Pool D_wrong / D_all from PlotQA + InfoVQA OneVision calibration into
calibration_plotqa_infovqa_pooled/v.pt for ActAdd recalibration.

Mirrors the substrate used by E6 K=8 subspace; produces a single mean-direction
v.pt readable by `e6_steering_vector.py --calibration-tag plotqa_infovqa_pooled`.

CPU-only. Safe to run anytime.

Usage:
    uv run python scripts/_pool_v_actadd_recal.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "llava-onevision-qwen2-7b-ov"
HF_MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
SOURCE_TAGS = ["plotqa", "infographicvqa"]
OUT_TAG = "plotqa_infovqa_pooled"


def main() -> None:
    base = PROJECT_ROOT / "outputs" / "e6_steering" / MODEL
    out_dir = base / f"calibration_{OUT_TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)

    D_wrong_parts: list[torch.Tensor] = []
    D_all_parts: list[torch.Tensor] = []
    n_wrong_per: dict[str, int] = {}
    n_all_per: dict[str, int] = {}
    n_layers = None
    d_model = None

    t0 = time.time()
    for tag in SOURCE_TAGS:
        src = base / f"calibration_{tag}"
        Dw = torch.load(src / "D_wrong.pt", map_location="cpu", weights_only=True)
        Da = torch.load(src / "D_all.pt", map_location="cpu", weights_only=True)
        assert Dw.dim() == 3 and Da.dim() == 3, f"{tag}: expected (N, L, D)"
        if n_layers is None:
            n_layers, d_model = Dw.shape[1], Dw.shape[2]
        else:
            assert (n_layers, d_model) == (Dw.shape[1], Dw.shape[2]), \
                f"{tag}: shape mismatch ({Dw.shape}) vs ({n_layers},{d_model})"
        D_wrong_parts.append(Dw.float())
        D_all_parts.append(Da.float())
        n_wrong_per[tag] = int(Dw.shape[0])
        n_all_per[tag] = int(Da.shape[0])
        print(f"  loaded {tag}: D_wrong {tuple(Dw.shape)}  D_all {tuple(Da.shape)}")

    D_wrong = torch.cat(D_wrong_parts, dim=0)
    D_all = torch.cat(D_all_parts, dim=0)

    v_wrong = D_wrong.mean(dim=0)  # (n_layers, d_model)
    v_all = D_all.mean(dim=0)
    v = torch.stack([v_wrong, v_all], dim=0)  # (2, n_layers, d_model)

    torch.save(v, out_dir / "v.pt")
    norms_w = v_wrong.norm(dim=1).tolist()
    norms_a = v_all.norm(dim=1).tolist()

    meta = {
        "model": MODEL,
        "hf_model": HF_MODEL,
        "dataset_tag": OUT_TAG,
        "n_wrong": int(D_wrong.shape[0]),
        "n_all": int(D_all.shape[0]),
        "n_wrong_per_source": n_wrong_per,
        "n_all_per_source": n_all_per,
        "source_tags": SOURCE_TAGS,
        "n_layers": n_layers,
        "d_model": d_model,
        "D_wrong_shape": list(D_wrong.shape),
        "D_all_shape": list(D_all.shape),
        "v_index_0": "v_wrong",
        "v_index_1": "v_all",
        "norms_v_wrong_per_layer": norms_w,
        "norms_v_all_per_layer": norms_a,
        "wall_seconds": time.time() - t0,
        "note": (
            "Pooled mean direction for ActAdd recalibration on the same (a-m) "
            "calibration substrate as E6 K=8 subspace (PlotQA + InfoVQA)."
        ),
    }
    with (out_dir / "v_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] saved {out_dir}/v.pt  shape={tuple(v.shape)}  "
          f"n_wrong={D_wrong.shape[0]}  n_all={D_all.shape[0]}")
    print(f"[done] v_wrong norms: L0={norms_w[0]:.3f}  L26={norms_w[26]:.3f}  "
          f"max@L{int(torch.tensor(norms_w).argmax())}={max(norms_w):.3f}")


if __name__ == "__main__":
    main()
