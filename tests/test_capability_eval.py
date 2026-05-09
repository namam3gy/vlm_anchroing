"""Tests for E8 capability eval — hook factory parity, aggregator math, verdict logic."""

from __future__ import annotations

import torch

from vlm_anchor.hooks import make_subspace_projection_hook


def test_hook_projects_last_token_at_prefill():
    """At prefill (seq_len > 1), only the last token's residual is updated;
    the update equals h - α · V^T V h."""
    torch.manual_seed(0)
    d_model, K, seq_len = 8, 2, 5
    alpha = 1.0
    # Construct V_K with orthonormal rows so V^T V is a proper projector
    V_K = torch.linalg.qr(torch.randn(d_model, K)).Q.T  # (K, d_model)
    hidden = torch.randn(1, seq_len, d_model)
    expected = hidden.clone()
    last = expected[:, -1, :]
    proj = (last @ V_K.T) @ V_K
    expected[:, -1, :] = last - alpha * proj

    hook = make_subspace_projection_hook(V_K, alpha)
    output = (hidden.clone(),)  # mimic HF decoder layer output (tuple)
    new_output = hook(None, None, output)
    new_hidden = new_output[0] if isinstance(new_output, tuple) else new_output

    torch.testing.assert_close(new_hidden, expected)


def test_hook_decode_step_is_noop():
    """At decode (seq_len == 1), the hook returns output unchanged."""
    d_model, K = 8, 2
    V_K = torch.linalg.qr(torch.randn(d_model, K)).Q.T
    hook = make_subspace_projection_hook(V_K, alpha=1.0)
    hidden = torch.randn(1, 1, d_model)
    output = (hidden.clone(),)
    new_output = hook(None, None, output)
    new_hidden = new_output[0] if isinstance(new_output, tuple) else new_output
    torch.testing.assert_close(new_hidden, hidden)


def test_hook_alpha_zero_returns_none():
    """alpha=0 means caller installs no hook; factory returns None."""
    V_K = torch.randn(2, 8)
    assert make_subspace_projection_hook(V_K, alpha=0.0) is None


def test_hook_non_3d_output_unchanged():
    """If hook receives a non-3D tensor (e.g. tuple of None), it returns input."""
    V_K = torch.linalg.qr(torch.randn(8, 2)).Q.T
    hook = make_subspace_projection_hook(V_K, alpha=1.0)
    # Tuple where first element is not a 3D tensor
    weird_output = (torch.randn(8),)  # 1D
    result = hook(None, None, weird_output)
    assert result is weird_output  # unchanged


def test_hook_dtype_device_match():
    """Hook casts V_K to match hidden's device + dtype."""
    d_model, K = 8, 2
    V_K = torch.linalg.qr(torch.randn(d_model, K)).Q.T.to(torch.float32)
    hook = make_subspace_projection_hook(V_K, alpha=1.0)
    hidden = torch.randn(1, 3, d_model, dtype=torch.float16)
    output = (hidden.clone(),)
    new_output = hook(None, None, output)
    new_hidden = new_output[0] if isinstance(new_output, tuple) else new_output
    assert new_hidden.dtype == torch.float16


# ── Aggregator + verdict tests ────────────────────────────────────────


def _import_agg():
    """Lazy import so this file is importable before Task 4 lands."""
    import importlib.util
    import sys
    from pathlib import Path
    p = Path(__file__).parent.parent / "scripts" / "aggregate_capability_eval.py"
    spec = importlib.util.spec_from_file_location("agg", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agg"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_thresholds_are_pre_registered():
    """Pin verdict thresholds — paper §7.4.5 strict free-lunch claim
    rests on these specific values being chosen ex-ante. Changing them
    requires also editing this assertion (auditable in the diff)."""
    agg = _import_agg()
    assert agg.PER_BENCH_THRESHOLD == -0.01, "per-bench threshold must remain -1.0pp"
    assert agg.MACRO_THRESHOLD == -0.005, "macro threshold must remain -0.5pp"


def test_mcnemar_paired_se_zero_when_identical():
    agg = _import_agg()
    n = 100
    correct_b = [True] * 60 + [False] * 40
    correct_m = list(correct_b)  # identical
    delta, se = agg.mcnemar_paired(correct_b, correct_m)
    assert delta == 0.0
    assert se == 0.0


def test_mcnemar_paired_se_nontrivial():
    agg = _import_agg()
    # 80 both correct, 5 only baseline correct, 10 only mit correct, 5 both wrong
    correct_b = [True] * 85 + [False] * 15
    correct_m = [True] * 80 + [False] * 5 + [True] * 10 + [False] * 5
    delta, se = agg.mcnemar_paired(correct_b, correct_m)
    # Δ in proportion: (80+10)/100 - (80+5)/100 = 0.05
    assert abs(delta - 0.05) < 1e-9
    # discordant pairs = 15 (5 + 10), McNemar SE on proportion ≈ sqrt(15)/100
    expected_se = (15 ** 0.5) / 100
    assert abs(se - expected_se) < 1e-9


def test_verdict_strict_free_lunch():
    agg = _import_agg()
    deltas = {"a": 0.001, "b": -0.002, "c": -0.005, "d": 0.01, "e": -0.003}
    macro = sum(deltas.values()) / len(deltas)
    assert agg.verdict(deltas, macro) == "STRICT_FREE_LUNCH"


def test_verdict_per_bench_degraded():
    agg = _import_agg()
    deltas = {"a": 0.0, "b": 0.0, "c": -0.012, "d": 0.0, "e": 0.0}  # c < -1pp
    macro = sum(deltas.values()) / len(deltas)
    v = agg.verdict(deltas, macro)
    assert v.startswith("DEGRADED:")
    assert "c" in v


def test_verdict_macro_degraded():
    agg = _import_agg()
    # All within per-bench bound but macro < -0.5pp
    deltas = {"a": -0.009, "b": -0.009, "c": -0.009, "d": -0.009, "e": -0.009}
    macro = sum(deltas.values()) / len(deltas)
    assert agg.verdict(deltas, macro) == "DEGRADED:macro"


def test_verdict_helped_does_not_flip():
    agg = _import_agg()
    deltas = {"a": 0.02, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0}
    macro = sum(deltas.values()) / len(deltas)
    assert agg.verdict(deltas, macro) == "STRICT_FREE_LUNCH"


# ── Merge subcommand tests ────────────────────────────────────────────


def _write_csv(path, rows):
    """Helper: write a final-format CSV with the given (name, delta) rows."""
    import csv as _csv
    with path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["benchmark", "n", "acc_baseline", "acc_mit",
                    "delta", "se", "ci_low", "ci_high", "status", "note"])
        for name, delta in rows:
            w.writerow([name, 100, 0.5, 0.5 + delta,
                        delta, 0.01,
                        delta - 0.0196, delta + 0.0196,
                        "OK", ""])


def test_merge_combines_disjoint_panels(tmp_path):
    agg = _import_agg()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "out.md"
    _write_csv(a, [("X", 0.001), ("Y", -0.002)])
    _write_csv(b, [("Z", 0.003), ("W", 0.0)])
    v = agg.merge_finals([a, b], out_csv, out_md)
    rows = list(out_csv.open())
    # header + 4 rows
    assert len(rows) == 5
    assert v == "STRICT_FREE_LUNCH"
    # Macro Δ = (0.001 - 0.002 + 0.003 + 0.0) / 4 = 0.0005
    md = out_md.read_text()
    assert "Macro Δ:" in md and "+0.05pp" in md


def test_merge_later_overrides_earlier(tmp_path):
    """Re-running a benchmark should supersede the older row."""
    agg = _import_agg()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "out.md"
    _write_csv(a, [("X", -0.05)])  # this would FAIL on its own
    _write_csv(b, [("X", 0.005)])  # later run; passes
    v = agg.merge_finals([a, b], out_csv, out_md)
    assert v == "STRICT_FREE_LUNCH"
    csv_text = out_csv.read_text()
    # Later value wins (0.005 not -0.05)
    assert "0.005" in csv_text
    assert "-0.05" not in csv_text


def test_merge_preserves_failure(tmp_path):
    """If merged panel has a per-bench DEGRADED row, verdict reflects it."""
    agg = _import_agg()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    out_csv = tmp_path / "out.csv"
    out_md = tmp_path / "out.md"
    _write_csv(a, [("X", 0.0), ("Y", 0.0)])
    _write_csv(b, [("Z", -0.05)])  # Δ < -1pp threshold
    v = agg.merge_finals([a, b], out_csv, out_md)
    assert v.startswith("DEGRADED:")
    assert "Z" in v
