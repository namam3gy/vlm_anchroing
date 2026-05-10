# P0-1 design — γ-β residual-stream bridge experiment

**날짜**: 2026-05-10
**Branch**: `worktree-phase5+p0-1-gamma-beta-bridge`
**출처**: [`docs/insights/plan_post_review_2026-05-09.md`](../../../docs/insights/plan_post_review_2026-05-09.md) §P0-1 (R5 bar-raiser signature ask)
**관련 §**: paper §4.6 (γ-β behavioral) / §5.2 (multi-layer redundancy) / §6.2 (K=8 subspace) / §8.4 item 1

---

## 1. Task 정의

### 1.1 Research question

Qwen3-VL-Thinking이 reasoning trace를 만들어내는 동안, anchor subspace 진폭이 Qwen3-VL-Instruct 단일-스텝 baseline 대비 *증폭*되는가? 그리고 그 증폭이 §4.5에서 관찰된 ×12.7 correct-base direction-follow ratio를 정량적으로 예측하는가?

### 1.2 Paper 측 leverage

현재 paper의 세 thread가 같은 페이지에 있지만 인과적으로 묶이지 않은 상태:

| § | 내용 | 측정 axis |
|---|---|---|
| §4.6 | γ-β behavioral: Thinking이 anchor pull을 증폭 (adopt ×1.6, df ×2.9, correct-base df ×12.7) | behavior |
| §5.2 | mechanism panel single-layer attention ablation 6/6 null — 신호는 multi-layer redundant | attention pathway |
| §6.2 | OneVision Main에서 L=26 K=8 subspace projection이 strict free-lunch | residual stream |

**Bridge가 positive로 land**:
- §4.6 (behavioral 증폭) ←→ §6.2 (residual subspace 진폭) 인과 사슬이 닫힘
- §1.5 contribution stack의 (4a) "routing vs integration framework"가 empirically interlocked
- §8.4 item 1 strikethrough; R4 CRIT-1 (E6 N=1 deployable) 부분 close
- bar-raiser convergent verdict의 Main 진입 조건 충족

**Bridge가 null로 land**:
- §4.6은 behavioral existence-proof로 fence — paper 본문 변동 없음
- §1.5 (5) hedge stack 강화, §8.4가 null bridge acknowledge
- methodology defensibility 유지 (failed experiment의 honest reporting은 reviewer-comfort 요소)
- paper tier 자체는 *Solid Findings, top of band*에 그대로 남음

**Bridge가 falsified (Alt-1, length confound)**:
- 진폭이 anchor arm뿐 아니라 neutral `d` arm에서도 같은 크기로 증폭
- §4.6.1 신규 section ship 안 함; §8.2 limitation에 명시
- paper tier 동일

⇒ upside가 크고 downside가 작은 비대칭 베팅 → P0 single-highest-leverage 지정 근거.

---

## 2. Approach 결정 사항

### 2.1 Cross-architecture 처리: Path B (Qwen3-VL self-calibration)

OneVision (Qwen2-7B 백본, hidden=3584, 28-layer)과 Qwen3-VL-8B (Qwen3-8B 백본, hidden=4096, 36-layer)는 hidden_dim mismatch가 있어 OneVision의 V_K[L=26]을 직접 4096-dim에 사영 불가. 세 옵션:

| Path | 설명 | 결정 |
|---|---|---|
| A | OneVision V_K (3584-dim)을 4096-dim에 truncate / learned linear mapping. ~2 H100-day. | **기각** — methodology hand-wave |
| **B** | **Qwen3-VL-Instruct의 자체 V_K[L=33]을 (a−m) calibration set으로 새로 보정.** | **선택** — architecture-native, defensible |
| C | B + augmented calibration (TallyQA mix). ~6-8 H100-day. | 기각 — 추가 비용 정당화 부족 |

**근거** (사용자 결정 2026-05-10): cross-arch가 bridge의 본질적 속성이므로 (Qwen2 backbone에 thinking variant 부재, paper §6 ↔ §4.6의 cross-architecture 합치 자체가 검증 대상), self-calibration이 reviewer-defensible 유일 path.

### 2.2 Calibration stimulus set: PlotQA + InfoVQA pooled

paper §6.2가 OneVision E6에서 PlotQA + InfoVQA pooled n=5000을 쓴 근거 (gt range 균등, row 수 충분, E5d cutoff validation 통과)는 Qwen3-VL self-calibration에서도 유효. MathVista는 calibration에 부적합:

| 기준 | PlotQA + InfoVQA | MathVista |
|---|---|---|
| Wrong-base paired sids | n≈5,000 | n≈125 (Instruct) |
| GT 분포 균등성 | 5 gt-bin 균등 + 정수 폭 [1, ~10000] | math/science 혼합 multi-modal |
| E5d cutoff validation | 통과 | **C3 FAIL** (roadmap §9 caveat) |
| Calibration leakage | γ-β eval(MathVista)와 disjoint | **leakage 위험** (eval = calibration) |

**γ-α MathVista를 calibration으로 쓰면 안 되는 이유**: P0-1의 bridge claim은 "Qwen3-VL의 anchor mechanism이 reasoning 모드에서 증폭된다"인데, calibration이 evaluation과 same distribution이면 reviewer가 "subspace가 stimulus set의 idiosyncratic noise를 잡았다"라고 즉시 공격 가능. paper §6 → §4.6 chain의 인과 강도가 무너짐.

⇒ Qwen3-VL-Instruct로 PlotQA + InfoVQA pooled n=5000에서 새 inference + (a, m) paired residual capture + SVD.

### 2.3 Layer / K / α 처리

P0-1은 **measurement-only** 실험 (mitigation 적용 아님, residual을 push하지 않음, 단순 사영 진폭 측정). 따라서:

| 파라미터 | OneVision E6 결정 방법 | Qwen3-VL P0-1 처리 |
|---|---|---|
| **L** | 27-cell pilot grid `L∈{25,26,27}` → cell #17 = L=26 | proportional default `L=33` (33/36 = 0.917 ≈ OneVision 26/28 = 0.929) + `L∈{29, 30, 33, 34}` single-pass robustness sweep (capture 추가 비용 거의 0) |
| **K** | grid `K∈{2,4,8}` → K=8 | paper §6 prior K=8 그대로; D 행렬 spectrum (singular value 1..16)을 evidence doc에 보고 — top-8 elbow 분리 sanity check. K-grid sweep은 P0-2 (eigenvalue spectrum)가 별도 처리 |
| **α** | grid `α∈{0.5,1.0,2.0}` → α=1.0 | **무관** — P0-1은 measurement only. residual을 V_K 방향으로 push하지 않음, 단순 inner product `‖V_K^T h_t‖_2`만 |

**핵심 logic**: P0-1은 paper §6 finding (K=8, late layer)이 cross-arch transfer하는지를 묻는 실험이므로, K=8 prior + proportional L을 그대로 쓰고 *failure to transfer*가 나면 그 자체가 informative result. 그 시점에 P2-7 (E6 cross-arch on Qwen2.5-VL) 등 별도 grid 작업으로 분기. P0-1 자체에서는 grid 안 함.

---

## 3. 데이터 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│ Phase 1 — Calibration                                            │
│                                                                   │
│ inputs/plotqa_test/ (n=5000 stratified)                          │
│ inputs/infographicvqa_val/ (n=1147 numeric)                      │
│         │                                                         │
│         │ Qwen3-VL-Instruct inference                            │
│         │   + per-token L∈{29,30,33,34} residual capture          │
│         │   + answer-step token only (last text-token before gen)│
│         ▼                                                         │
│ wrong-base paired (a, m): expected n≈3000-5000 across two ds      │
│         │                                                         │
│         │ D = stack[(h_a − h_m)(L=33)] for sid in wrong-base     │
│         ▼                                                         │
│ SVD top-K=8 → V_K[L=33] ∈ ℝ^{4096×8}                            │
│ (parallel: V_K[L=29], V_K[L=30], V_K[L=34] for robustness)       │
│                                                                   │
│ outputs/gamma_beta_bridge/_subspace/                             │
│   qwen3vl_instruct_VK_L{29,30,33,34}_K8.pt                       │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 2 — Bridge inference                                       │
│                                                                   │
│ existing γ-β MathVista S1 stimuli (4-cond × 365 sample_instances)│
│         │                                                         │
│         │ a-S1 (anchor) + d (neutral, length control) arms only  │
│         ▼                                                         │
│ Qwen3-VL-Instruct  per-token L=33 residual capture (median 7 tok)│
│ Qwen3-VL-Thinking  per-token L=33 residual capture (median 313)  │
│         │                                                         │
│         │ for each (item, mode, cond, layer) trace:               │
│         │   a_t = ‖V_K^T h(x_t, L)‖_2  ∀ generated token t       │
│         │   trace_mean = mean_t a_t                               │
│         │   trace_max  = max_t a_t                                │
│         ▼                                                         │
│ outputs/gamma_beta_bridge/{instruct,thinking}/<ts>/               │
│   amplitude_per_trace.jsonl                                      │
└──────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ Phase 3 — Bridge statistic                                       │
│                                                                   │
│ paired by sample_instance_id:                                    │
│   Δ_i = thinking_trace_mean(i) − instruct_trace_mean(i)          │
│                                                                   │
│ Sub-groups:                                                      │
│   - all-base (primary)                                           │
│   - correct-base (Instruct's pred_b == gt) — ×12.7 region       │
│   - wrong-base                                                   │
│                                                                   │
│ Test:                                                            │
│   1. paired bootstrap B=10,000 → 95% CI on mean(Δ_i)            │
│   2. amplitude ratio = mean(thinking) / mean(instruct) per group│
│   3. Alt-1 falsification: same Δ on d arm                        │
│                                                                   │
│ docs/insights/_data/gamma_beta_bridge_amplitude_per_trace.csv    │
│ docs/insights/gamma-beta-bridge-evidence.md                      │
│ notebooks/gamma_beta_bridge_amplitude.ipynb                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Acceptance criteria

| 결과 | 통계 조건 | 후속 처리 |
|---|---|---|
| **Primary positive** | `mean_i Δ_i > 0`, paired bootstrap 95 % CI 0 제외 (a-S1 arm) | §4.6.1 sub-section ship; §1.5 hedge 약화; §8.4 item 1 strikethrough |
| **Quantitative confirm (Main-tier)** | correct-base ratio `mean(thinking_mean)/mean(instruct_mean)` ≥ 2× | headline 강화 — "behavioral ×12.7 amplification은 residual subspace ≥2× amplitude growth로 mechanism-level 매핑" |
| **Stronger quantitative** | correct-base ratio가 ×12.7 근처 (±50%) | bridge가 정량적 예측까지 — paper의 가장 강한 chain |
| **Null** | CI 0 포함 | §4.6 behavioral existence-proof로 fence; §8.4 null bridge acknowledge; paper tier 무변 |
| **Adverse (Alt-1)** | d arm에서도 동일 크기 Δ | bridge falsified (length confound); §4.6.1 ship 안 함 |

---

## 5. 구성 요소

| 파일 | 역할 |
|---|---|
| `scripts/calibrate_qwen3vl_subspace.py` | 신규 — Qwen3-VL-Instruct PlotQA+InfoVQA inference + per-layer residual capture + SVD |
| `scripts/run_gamma_beta_bridge.py` | 신규 — γ-β inference + per-token L=33 residual capture + projection |
| `scripts/build_gamma_beta_bridge_summary.py` | 신규 — paired bootstrap CI aggregator + canonical CSV/MD emit |
| `tests/test_gamma_beta_bridge.py` | 신규 — subspace orthonormality 단위 test, projection 산술 unit test, smoke run on n=5 |
| `notebooks/gamma_beta_bridge_amplitude.ipynb` | 신규 — top-to-bottom 재현 (per-token trajectory, per-item paired histogram, bootstrap CI table) |
| `configs/gamma_beta_bridge.yaml` | 신규 — 모델 + capture 사양 |
| `docs/insights/gamma-beta-bridge-evidence.md` | 신규 — full evidence doc |
| `docs/insights/_data/gamma_beta_bridge_amplitude_per_trace.csv` | 신규 — canonical per-trace 표 |
| `docs/insights/_data/gamma_beta_bridge_qwen3vl_singular_values.csv` | 신규 — D 행렬 spectrum (K=8 elbow sanity check) |

`vlm_anchor/models.py` core는 손대지 않음. forward hook은 capture script 내부에서만 install. P3-11 / P4-12 같은 별도 파일 변경 없음.

---

## 6. 위험 + 완화

| 위험 | 완화 |
|---|---|
| Qwen3-VL hook silent fail (custom code path 가능성, OneVision SDPA-mask hook 버그 [`feedback_sdpa_mask_hook_bug`](memory) 재발 우려) | n=5 smoke run으로 captured tensor shape == `(n_gen_tokens, 4096)` assertion 검증을 *long inference 시작 전* 의무화. eager attention impl로 강제 (SDPA bypass 회피). |
| Thinking 512-tok trace + residual capture OOM | 36 layer × 4096 × 512 × bf16 activation = 144 MB/forward + KV cache ~25 GB; 단일 H200 (140 GB) 안전. capture는 selected layer (4개)만, 즉 16 MB/item × 720 items = ~12 GB disk. |
| Trace-length confound (Thinking 313 vs Instruct 7 tok) | mean과 max를 둘 다 보고; **d neutral arm을 length-only control로 같이 실행** — d arm에서 동일 amplitude 증폭이면 length-effect로 falsified |
| Calibration n SVD instability | spectrum singular value 1..16 보고; top-8과 9th 사이 분리가 약하면 evidence doc에 명시 + K=4 fallback row 추가 보고 |
| Calibration leakage | calibration = PlotQA+InfoVQA, eval = MathVista — 분포 disjoint |
| Layer choice arbitrariness | proportional L=33 + L∈{29,30,33,34} sweep 모두 보고; primary가 어느 layer인지 evidence doc 명시 |
| Cross-arch trivial 비교 (Qwen3-VL self-calibration이 OneVision 결과와 무관해 보임) | spectrum + L sweep + K sanity check를 통해 paper §6 finding의 falsifiability 명시; failure를 informative result로 보고 |

---

## 7. 비용 + 일정

| Phase | Wall-clock | GPU |
|---|---|---|
| Hook smoke 검증 (n=5) | 30 min | 1× H200 |
| Calibration inference (Qwen3-VL-Instruct on PlotQA n=5000 + InfoVQA n=1147) | ~5 h (n=6147 total stims, 2 cond capture passes) | 1× H200 |
| SVD (n≈3000-5000 wrong-base D 행렬, K=8) + L sweep | ~30 min | none |
| Bridge inference Qwen3-VL-Instruct (γ-β a-S1 + d, n=2 cond × 360) | ~2 h | 1× H200 |
| Bridge inference Qwen3-VL-Thinking (median 313 tok × 360 × 2 cond) | ~12 h | 1× H200 |
| Aggregator + paired bootstrap + notebook + evidence doc | ~6 h | none |
| **합계** | **~26 h ≈ 1.5일** | 1× H200 |

Conservative budget 2일. plan §P0-1 cheap form ~2 H100-day와 동일 envelope 안.

---

## 8. Dependency

- 입력 자원: **모두 disk에 존재**
  - `inputs/plotqa_test/` ✓
  - `inputs/infographicvqa_val/` ✓
  - γ-β MathVista S1 stimuli (`outputs/experiment_e5e_mathvista_reasoning/qwen3-vl-8b-{instruct,thinking}/20260428-114421/`) ✓
  - irrelevant_number / irrelevant_number_masked / irrelevant_neutral inventories ✓
- 모델: HuggingFace `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-8B-Thinking` (container에서 즉시 다운로드)
- 코드 자원:
  - `scripts/e6_compute_subspace.py` — SVD/projection 로직 참조
  - `scripts/extract_attention_mass.py` — forward hook pattern 참조
  - `vlm_anchor/data.py` — load_number_vqa_samples + assign_irrelevant_images 재활용
  - `vlm_anchor/models.py` — HFAttentionRunner 재활용 (capture는 외부 hook으로)
- P0-2 (eigenvalue spectrum)와는 partial overlap이지만 disjoint task — P0-1은 K=8 prior 사용, P0-2가 K-grid 별도 검증

---

## 9. PR 흐름 (memory rule [`feedback_auto_branch_merge_push`](Paper-completion phase: PR-only workflow))

1. branch `worktree-phase5+p0-1-gamma-beta-bridge` (현재) 위에서 모든 작업
2. 각 phase 완료 후 incremental commit
3. evidence doc + canonical CSV + notebook + paper §4.6.1 / §1.5 / §8.4 update까지 끝나면 push + `gh pr create`
4. 사용자 merge 대기 (master 직접 push 금지)

---

## 10. Open question — 명시적 user confirm 받아야 함

이 design 작성 시점에 unresolved되어 있는 항목 (구현 들어가기 전 user confirm 권장):

1. **Calibration stimulus 변경 (MathVista → PlotQA+InfoVQA)**: §2.2의 변경 근거가 reviewer-defensibility라 사용자 동의 필요. budget +4-5 H200-hour, total 1.5-2일 wall-clock.
2. **§4 acceptance threshold**: primary는 paired-bootstrap CI > 0이 합리적 (R4 MAJ-4와 일관). quantitative confirm threshold ≥2×는 plan §P0-1의 acceptance criterion. ×12.7 (강한 quantitative match)는 stretch goal.
3. **Length confound 처리 방식**: d neutral arm을 length-only control로 같이 실행하는 것이 합당. 추가 ~1.5 H200-hour 비용.

이 세 부분 confirm 후 writing-plans skill로 implementation plan 작성, 그리고 implementation 시작.

---

## 11. Out-of-scope

- P0-2 (eigenvalue spectrum) — 별도 phase 5 P0 항목
- P1-3 / P1-4 / P1-5 / P1-6 — 별도 sprint 항목
- P2-7 (E6 cross-arch on Qwen2.5-VL) — Qwen3-VL P0-1과 별개의 dedicated cross-arch replication 작업
- α grid sweep on Qwen3-VL — measurement-only experiment이므로 α 정의되지 않음. mitigation을 actually 적용하려면 P2-7-style separate grid 필요
- Qwen3-VL의 mitigation arm capability eval — P0-1은 mitigation 적용 안 함, capability eval 영향 없음

---

*End of design.*
