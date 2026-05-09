# Post-Review Improvement Plan — 2026-05-09

**Source.** Synthesizes the 5-round iterative review loop on `docs/paper/emnlp_draft_ko.md` (commit `8ffdc2d`, merged into master `86fb66a`). Round 1-5 review + response files at [`docs/paper/reviews/`](paper/reviews/) and final loop summary at [`docs/paper/reviews/_final_summary.md`](paper/reviews/_final_summary.md).

**Goal.** Tier-shift from *Solid Findings (top of band)* to *weak-accept Main*. Per the convergent verdict from R4 author / R5 bar-raiser / R5 author, the single highest-leverage move is the **γ-β residual-stream bridge experiment** (P0 / 항목 1 below). Items P1-P3 are Main-acceptance hardening; P4 is camera-ready hygiene; P5+ is future submissions.

**Update (2026-05-09 post-merge):** A user-driven review pass identified a separate paper-narrative gap — the §5.2 single-layer attention ablation null and the §6 single-layer L=26 subspace projection appeared to *contradict* unless the paper explicitly framed *attention as routing pathway* and *residual stream as integration site*. The paper now ships the **routing vs integration framework** (§5.2 Insight 4 + §6.6 reconciliation paragraph) which (i) reconciles single-layer null with single-layer projection success, (ii) justifies *late layer* selection (L=26 as integration-complete-but-pre-final), (iii) justifies *projection* tool over ablation. This framework is the paper's *이론적* contribution and is now propagated across abstract / §1.3 / §1.5 (4a) / §8.1 callsites. **P0-2 per-layer spectrum sweep is the framework's directly falsifiable empirical anchor** — see updated P0-2 below.

**Compute envelope.** Reasonable budget: ~2 H100-week + ~4 H200-week if all P0-P2 land. Clean Main path = P0 + P1 (2 items). Maximum hardening = P0 through P3 inclusive.

---

## P0 — Tier-shift experiments (ship for Main acceptance)

### P0-1 · γ-β residual-stream bridge experiment (cheap form)

**Bar-raiser signature ask.** Does the (a − m) K=8 subspace amplitude grow over Thinking-mode trace generation, and does that growth quantitatively predict the ×12.7 correct-base amplification?

**What.** Project the residual stream of Qwen3-VL-8B-Thinking γ-β trace tokens onto V_K[L=26] (the OneVision-calibrated K=8 anchor subspace). Measure subspace amplitude `‖V_K^T h(x_t, L=26)‖_2` per token across the trace. Test: does mean amplitude on `correct-base` Thinking traces *exceed* mean amplitude on `correct-base` Instruct (single-token) baseline by a factor that quantitatively tracks the ×12.7 df ratio?

**Why.** Currently §4.6 (γ-β behavioral) / §5.2 (multi-layer redundancy) / §6.2 (K=8 subspace) sit on the same page but never causally interlock. Positive → all three threads collapse to one mechanism-grounded chain, partially closing CRIT-1 (single-model deployable). Negative → properly fences §4.6 as behavioral existence-proof and weakens nothing already in the paper.

**How — cheap form.** Reuse OneVision V_K[L=26] as instrument (already on disk at `docs/insights/_data/...`). Run Qwen3-VL-Thinking on existing γ-β stimuli (`outputs/experiment_e5e_mathvista_reasoning/`), capture residual stream at corresponding Qwen3-VL layer, project, sum amplitude per trace, compare aggregate distribution against Instruct baseline.

**How — clean form (optional, stronger).** Calibrate Qwen3-VL-Instruct's *own* K=8 subspace from a small (a − m) calibration set on Qwen3-VL (no PoolQA + InfoVQA needed; can reuse a 200-300 wrong-base subset). Project Thinking traces onto Qwen3-VL's own subspace. Eliminates cross-model layer alignment confound.

**Estimate.**
- Cheap form: ~2 H100-day (inference + projection script + analysis notebook).
- Clean form: +~4 H100-day (calibration N=200-300 wrong-base on Qwen3-VL).

**Deliverable.**
- `outputs/gamma_beta_bridge/qwen3_vl_thinking_residual_amplitude_<ts>/`
- New §4.6.1 sub-section in paper draft: "γ-β residual-stream amplitude — bridge to §6 mechanism."
- New evidence doc `docs/insights/gamma-beta-bridge-evidence.md`.
- New canonical CSV `docs/insights/_data/gamma_beta_bridge_amplitude_per_trace.csv`.

**Acceptance criteria.**
- (positive outcome) Thinking-trace mean amplitude > Instruct baseline by factor ≥ 2× (interlock established) → write the §4.6.1 + abstract + §1.3 + §8.1 update; scope hedge in §1.5(5) softens.
- (null outcome) No significant difference (95 % CI overlaps zero) → fence §4.6 explicitly to behavioral existence-proof, §1.5(6) hedge stack reinforced, §8.4 closing note acknowledges null bridge.
- Either way: §8.4 item 1 strikethrough.

**Risk.** Cross-architecture residual stream alignment between OneVision (Qwen2-7B) and Qwen3-VL (different LM backbone) — clean form mitigates fully; cheap form requires explicit caveat noting V_K is "near-archetype" instrument, not exact.

**Owner.** thyun.park.

**Dependency.** None (all data + subspace already on disk).

---

### P0-2 · Eigenvalue spectrum of `D[:, L, :]` — per-layer integration check

**What.** Compute and plot the singular value spectrum of the (a − m) calibration difference matrix at L=26 (PlotQA + InfoVQA pooled N=5,000) AND at a sweep of layers `L ∈ {10, 14, 18, 22, 26, 28}`. Identify (i) rank-K elbow at L=26, and (ii) whether anchor variance becomes *more concentrated* (lower effective rank, sharper elbow) at later layers.

**Why.** Two claims to test together:
1. **K=8 is data-predicted, not grid-search artifact** (single-layer rank-8 elbow at L=26). Bar-raiser axis 5 (theoretical contribution) currently weakest.
2. **Late residual stream is the integration site** (per §5.2 Insight 4 + §6.6 routing-vs-integration framework). Empirical signature: anchor variance redistributes from broad rank at early layers (signal still distributed across attention routing) to compact low-dim subspace at late layers (signal integrated into residual). This is the framework's *direct falsifiable prediction*; per-layer spectrum is the cheapest test.

If both land positive, the paper's *이론적* contribution graduates from "we framed the mechanism" to "we predicted and verified the mechanism's empirical signature."

**How.** Re-load V_L Σ_L from existing E6 calibration (`docs/insights/_data/...`); plot Σ_L spectrum per layer, log-scale + relative-decay overlay; report eigengap at rank-8 per layer + effective rank trajectory across L.

**Estimate.** ~4 H100-hour (single-layer L=26) + ~2 H100-hour additional for the per-layer sweep (mostly plotting + the SVDs already exist from E6 pilot grid + can be re-extracted from L≠26 with one extra pass on calibration set).

**Deliverable.**
- New Figure 1 (paper main body, §6.4) — SVD spectrum at L=26 with rank-8 elbow annotated.
- New Figure 2 (paper main body, §5.2 OR §6.4) — per-layer effective rank trajectory; integration claim's empirical signature.
- New paragraph in §6.4 Insight 2 (currently "K=8 sweet spot") elevating from empirical to spectrum-predicted.
- Per-layer evidence cited in §5.2 Insight 4 (routing-vs-integration framework's empirical anchor).
- Update §1.5 (4a) framing if both lands clean — predict-then-verify chain anchored on per-layer spectrum.

**Acceptance criteria.**
- (a) **Clean elbow at rank-8 on L=26**: Insight 2 promoted from "trade-off sweet spot" to "spectrum-predicted dimensionality." Citable theoretical contribution.
- (b) **Effective rank decreases monotonically across L=10→28** (or at least drops sharply at the L=20-26 transition): integration framework empirically confirmed; §5.2 Insight 4 cited from data not just hypothesis.
- (a only / b only / neither) graceful degradation: Insight 2 stays empirical; per-layer figure becomes transparency item; framework framing softened to "consistent with" rather than "verified by."

**Risk.** Per-layer spectrum may show continuous decay (common in real residual streams). If integration is more gradual than predicted, §5.2 Insight 4 framing softens but doesn't break — routing-vs-integration is still defensible at categorical level (residual ≠ attention pathway), just without empirical sharpness.

**Owner.** thyun.park.

**Dependency.** P0-2 should run *before* P0-1 final write-up — if P0-1 lands positive but spectrum is continuous, the paper still benefits from P0-1 alone.

---

## P1 — Adversarial-defense rigor (ship for stronger Findings + Main hardening)

### P1-3 · Paired-bootstrap CI on §6.2.3 Table 6

**What.** Re-aggregate the 5-dataset E6 paired-sids deltas in `docs/insights/_data/stage4_final_per_dataset.md` with paired bootstrap (B=10,000) on (Δ adopt, Δ df, Δ em(a), Δ em(b)) per dataset. Report 95 % CI per cell.

**Why.** R4 MAJ-4 critical: InfoVQA n=443 Δdf=−0.7 pp is noise floor without CI. R4 reviser DEFERred this; aggressive reviewer named it as Main blocker. The paper currently fences this caveat in prose; adding the CI converts caveat to defense.

**How.** Extend `scripts/build_e6_stage4_summary.py` with `--bootstrap-n=10000` flag; per-cell paired bootstrap on parsed predictions; emit `*_per_dataset_with_ci.{csv,md}`. Update Table 6 in paper.

**Estimate.** ~1 day (wall-clock; mostly scripting; bootstrap is fast on existing data).

**Deliverable.**
- Updated `docs/insights/_data/stage4_final_per_dataset_with_ci.{csv,md}`.
- Updated Table 6 in `docs/paper/emnlp_draft_ko.md`.
- §6.2.3 신뢰구간 caveat upgraded from prose-fence to numeric-fence with explicit CI per cell.

**Acceptance criteria.**
- (4/5 datasets Δdf CI excludes zero) Headline "5/5" claim survives with 4/5 confirmed + InfoVQA fenced as inconclusive *with citable CI*.
- (Bonferroni-corrected CIs still hold) §7 Bonferroni-6 robustness story mirrors here.

**Owner.** thyun.park.

**Dependency.** None (existing prediction files).

---

### P1-4 · CAA at K=1 / ITI at attention-head — actual Table 7 rows

**What.** Run two empirical baselines for §6.5 Table 7 (currently structural-reduction Note only):
- **CAA at K=1** [Panickssery et al. 2024]: ActAdd direction = mean of (a − m) at L=26 (= our V_K[L=26] at K=1). Apply at inference α=1.0 on Qwen2-7B residual stream. 5-dataset eval.
- **ITI at attention-head** [Li et al. 2023]: Find the top-K=8 most "anchor-discriminative" attention heads (probe accuracy for anchor-present vs anchor-absent on calibration set), shift residual at those heads' output by mean-anchor direction. 5-dataset eval.

**Why.** R4 MAJ-5 named these as Main blockers — paper claims structural-reduction Note for both. Empirical rows convert claim to comparison.

**How.**
- CAA: trivial (K=1 is rank-1 case of existing E6 SVD).
- ITI: requires probe training (~200-300 train, ~50 val on calibration set per attention head, choose top-K=8) + inference-time hook on those heads.

**Estimate.**
- CAA K=1: ~4 H100-hour (1 day max).
- ITI: ~2 H200-day (probe training + 5-dataset eval).
- Combined: ~3 H200-day.

**Deliverable.**
- 2 new rows in §6.5 Table 7 (CAA K=1 + ITI top-K=8).
- New evidence doc `docs/insights/E6-extended-mitigation-baselines.md`.
- §6.5 Note replaced with empirical rows.

**Acceptance criteria.**
- CAA K=1 fails free-lunch on 5/5 (predicted by §5.2 multi-layer redundancy + §6.4 cos≈0.47-0.62 non-collinearity) → §5.2 → §6.4 predict-then-verify chain *empirically* validated.
- ITI fails free-lunch on at least 1/5 datasets → consistent with multi-layer redundancy applied to attention pathway.

**Owner.** thyun.park.

**Dependency.** None for CAA. ITI requires building the attention-head probe pipeline (small new script).

---

### P1-5 · Random-K=8 baseline for §6.3 b-arm em (Alt-1 falsification)

**What.** Sample a random K=8 subspace from the residual-stream distribution at L=26 (random orthonormal projection); apply during inference identically to E6. Compare Δem(b) on b-arm.

**Why.** R4 CRIT-3: §6.3 b-arm em +8.8 pp could be (Alt-1) general regularization. Random subspace is the cleanest Alt-1 falsification — if random K=8 also moves Δem(b) positive, then E6's effect is *not* anchor-specific.

**How.** Sample 5 independent random K=8 subspaces (different seeds); 5-dataset eval per random subspace; aggregate.

**Estimate.** ~2 H100-day (5 × 5-dataset evaluations; embarrassingly parallel).

**Deliverable.**
- New cell-block in §6.3 Insight 1.5 with random-K=8 baseline empirical column.
- Update §6.3 evidence: Alt-1 falsified (random ≈ 0 ± noise, anchor +8.8 pp) OR weakened (random > 0, smaller effect).

**Acceptance criteria.**
- (random-K=8 Δem(b) ≈ 0 ± 1 pp) Alt-1 falsified, b-arm em is anchor-mechanism specific.
- (random-K=8 Δem(b) > 0 by half-magnitude) E6 still has anchor-specific contribution but co-aligned regularization is real — paper softens claim accordingly.

**Owner.** thyun.park.

**Dependency.** None.

---

### P1-6 · §A.5 27-cell pilot grid 4-metric heatmap aggregation

**What.** Aggregate the 27 (L, K, α) pilot cells into a per-cell 4-metric (Δdf, Δadopt, Δem(a), Δem(b)) heatmap. Surface in §A.5.

**Why.** R4 CRIT-2: 27-cell pilot grid was DEFERred across R1 and R3. Currently §A.5 has cell labels + chosen #17 marked but no aggregated metric values. Surfacing closes the cherry-pick concern.

**How.** Aggregate from existing predictions in `outputs/e6_steering/llava-onevision-qwen2-7b-ov/sweep_subspace_*_pilot/`. Trivial — existing data, just needs aggregation script.

**Estimate.** ~1 day.

**Deliverable.**
- New canonical `docs/insights/_data/e6_pilot_grid_27cell_full.{csv,md}`.
- Updated §A.5 with 4-metric heatmap (or 4 sub-tables, one per metric).
- §6.2.2 deal-breaker rule prose updated to point to the heatmap with chosen-cell #17 highlighted.

**Acceptance criteria.**
- All 26 non-chosen cells visible. Em-deal-breaker rule (Δem(a) ≤ −6 pp on either calibration dataset → reject) verified visually.
- Chosen cell #17 (L=26, K=8, α=1.0) sits at clear non-cherry-picked optimum (best |Δdf| among em-deal-breaker survivors).

**Owner.** thyun.park.

**Dependency.** None.

---

## P2 — Cross-architecture verification (ship for full Main acceptance)

### P2-7 · Cross-architecture E6 replication on 1 perfect-square archetype

**What.** Re-calibrate the E6 procedure on a *different* encoder archetype on the same PlotQA + InfoVQA pooled N=5,000 calibration set. Recommended target: **Qwen2.5-VL-7B** (Qwen-ViT late peak archetype, completely different encoder from OneVision's SigLIP). Calibrate (L*, K, α) with same em-deal-breaker rule; eval on 5 datasets.

**Why.** R4 CRIT-1 kill shot: E6 mitigation chain is N=1 model. Cross-architecture replication on 1 different archetype converts "single-model PoC" to "transferable recipe" — partial close of CRIT-1 (full close requires 3 archetypes per R5 secondary).

**How.** Run the E6 calibration script on Qwen2.5-VL-7B (modify `scripts/calibrate_subspace.py` for Qwen-ViT residual stream extraction). Pilot 27-cell grid. Choose via em-deal-breaker. 5-dataset eval.

**Estimate.** ~10 H200-day (calibration n=5000 wrong-base + 27-cell pilot + 5-dataset eval).

**Deliverable.**
- New Section §6.7 "Cross-architecture replication on Qwen2.5-VL" or new paragraph in §6.6 "Two mitigations' position".
- Updated §1.4 framing — E6 from "case study on `llava-onevision-qwen2-7b-ov`" to "two-archetype generalization."
- §1.5 (5) hedge stack softened from N=1 to N=2.
- Bar-raiser signature ask sub-route (γ-β cross-architecture) opened.

**Acceptance criteria.**
- (Qwen2.5-VL E6 chosen cell clears 4/5 datasets free-lunch) Cross-arch generalization confirmed; CRIT-1 partial close.
- (Qwen2.5-VL chosen cell different (L*, K, α)) Per-archetype calibration interpretation; recipe still transferable.
- (Qwen2.5-VL E6 fails) Single-arch limit confirmed; paper retitled as "case study" or expanded discussion section.

**Owner.** thyun.park.

**Dependency.** P0-2 (eigenvalue spectrum) helpful but not blocking — if rank-8 elbow holds on OneVision, predicts similar on Qwen-ViT.

**Risk.** Highest cost item on this plan. If P0-1 lands positive (γ-β bridge), P2-7 becomes optional rather than blocking.

---

## P3 — Robustness hardening (ship for camera-ready / strong submission)

### P3-8 · Paraphrase robustness (3-5 prompt variants × bootstrap CI)

**What.** Re-run the headline E6 5-dataset evaluation under 3-5 paraphrased JSON-strict prompts; bootstrap CI on cell-level Δdf, Δem.

**Why.** R1-R4 carryover: every cognitive-bias LLM paper has been criticized for single-prompt fragility. Defuses the most common reviewer critique cheaply.

**How.** Author paraphrase variants (5 prompts: original + 4 rephrasings preserving JSON-strict requirement). Re-run with `--prompt-variant=N` flag; aggregate.

**Estimate.** ~3 H200-day (5× existing 5-dataset eval × 1 model = 25 evaluation sweeps).

**Deliverable.**
- §A.X new appendix "Paraphrase robustness": Table of Δdf cell × prompt variant; SD across paraphrases.
- §8.2 limitation softened from "단일 prompt" to "5-prompt average ± SD."

**Owner.** thyun.park.

---

### P3-9 · Closed-source defuse (~500 stratified samples on GPT-4o or Gemini 2.5)

**What.** Test main behavioral findings (Phase-A H2 + plausibility window + digit-pixel causality) on 1 closed-source model (GPT-4o-2024-11 or Gemini 2.5) at ~500 stratified samples drawn proportionally from the 5-dataset main matrix. Headline: does Phase-A H2 +6.9-19.6 pp wrong-correct gap hold on closed model?

**Why.** R4 MIN: "open-weight only" is a stable reviewer critique. ~$10-15 API budget; defuses cheaply.

**How.** API client + prompt re-use; subset sampler; aggregator.

**Estimate.** ~1-2 day (mostly engineering + API rate limits).

**Deliverable.**
- §3.6 new bullet (model panel) — closed-source variant.
- §4.1 / §4.2 / §4.3 each get 1 row: closed-source result on subset.
- §8.2 limitation softened.

**Owner.** thyun.park.

---

### P3-10 · Encoder-family promotion to top-line contribution (camera-ready)

**What.** R5 bar-raiser secondary ask: §1.5 (4b) encoder-family-determines-archetype is currently a single bullet. Promotion to top-line contribution = (i) §1 intro paragraph framing, (ii) §1.5 bullet refactor that lists encoder-family-archetype as a *separate* contribution category from "mechanism analysis", (iii) §5 architecture re-introduces the encoder-family axis as the §5 organizing principle.

**Why.** Bar-raiser identified this as the most-citable 5-year finding's natural foundation (§5.2 → §6.4 chain rests on encoder-family archetype map).

**How.** Prose-only edits in `docs/paper/emnlp_draft_ko.md` (camera-ready timing).

**Estimate.** ~half-day (text-only).

**Deliverable.**
- Updated §1 + §1.5 + §5 in paper draft.

**Owner.** thyun.park.

**Dependency.** Hold for camera-ready (after Findings vs Main decision lands).

---

### P3-11 · VQAv2 single-dataset depth panel — Main backfill + cross-panel consistency

**Decision context (2026-05-09 user pass).** VQAv2 drop vs keep 결정에서 *keep*으로 reposition: VQAv2가 main matrix의 *cross-dataset breadth*와 *상보적인 single-dataset depth* axis (n=17,730 per model, paper 내 최대 n) 를 운반하며, "legacy panel" framing은 의도적 *single-dataset depth panel*로 격상. Tables 2 + 3에 Main model (OneVision) 부재는 reader confusion + cross-panel inconsistency를 만들므로 backfill 필수.

**What.** 두 tier로 분할:

**Tier 1 (필수, ~1.5 H200-day):**
1. **OneVision × VQAv2 (b/a/d) full panel** — 17,730 base × 3 cond, ~30-45 min H200. 결과 → Table 2 8-model panel 완성.
2. **OneVision × VQAv2 E5c (b/a/m/d × S1)** — ~17k base × 4 cond, ~30 min H200. 결과 → Table 3 VQAv2 OneVision 행.
3. **OneVision × TallyQA E5c (b/a/m/d × S1)** — stratified ~5k base × 4 cond, ~30 min H200. 결과 → Table 3 TallyQA OneVision 행.

**Tier 2 (cross-panel consistency, ~3 H200-day, 선택):**
Main matrix 6 model 중 VQAv2 panel에 부재한 3 model을 VQAv2에 추가하여 *11-model panel* 구성:
4. Gemma3-4b × VQAv2 (b/a/d) — ~30 min H200
5. InternVL3-8b × VQAv2 (b/a/d) — ~30 min H200
6. Qwen2.5-VL-32B × VQAv2 (b/a/d) — ~1-2 h H200

→ VQAv2 panel이 legacy 7 + OneVision + main matrix 3 = **11-model breadth × n=17,730 depth**로 paper 내 *최대 single-dataset comprehensive panel*이 됨; main matrix와의 *모델 overlap* 완성으로 cross-panel 정합 의문 제거.

**Tier 3 (Out of scope — *do not run*):** VQAv2 strengthen-prompt / VQAv2 4-cond (b/a/m/d) 확장 등은 §A 부록의 caveat에 이미 처리되었고, 정성적 신규 발견 추가 없음.

**Why.** (i) Main model이 §4.1 / §4.2 첫 노출 panel에서 *행으로* 등장 — reader가 §4 처음부터 OneVision Main 추적 가능; (ii) E5c VQAv2 wrong-base S1 (a − m) gap +6.1 pp (llava-interleave) 는 paper의 *가장 clean한 digit-pixel causality* 측정 (S1 absolute cutoff, scene-level confound 직접 falsify) 인데 Main model 측정 부재 — §F.3 chart-stack 4 dataset cover하지만 VQAv2 absolute cutoff 측정의 *깊이*가 다름; (iii) "VQAv2 안 썼나" reviewer 의문 사전 차단 — VQA의 정전 benchmark 부재는 reviewer comfort 손상; (iv) 11-model panel은 *legacy + main matrix overlap* 으로 cross-panel 정합 강화 (gemma4-31b / qwen3-vl-30b 같은 legacy-only model의 narrative 연결 부족 약점 부분 close).

**Paper-side reframe (병행).** §4.1 (Korean main + 영문 §5.1) 의 "legacy 7-model VQAv2 panel" → "**VQAv2 single-dataset depth panel** (n=17,730 per model, Tier 1 후 8-model / Tier 2 후 11-model)" 로 *의도적 selection*임을 명시. 한 줄 정당화 추가: "VQAv2는 본 논문 panel 내 *최대 single-dataset n*을 운반하며, main matrix의 5-dataset breadth와 *상보적인 single-dataset depth* axis로 사용; Phase-A H2 binary projection / (a − m) digit-pixel gap / L1 quartile gradient의 *replication depth*가 가장 높은 panel."

**How.** 기존 driver `scripts/run_experiment.py` + 기존 stimulus inventory (a, m, d) + 기존 E5c config 재활용. 새 코드 0줄.

**Estimate.** Tier 1 ~1.5 H200-day; Tier 2 추가 ~3 H200-day; 합 ~4.5 H200-day if both. Tier 1만으로도 Tables 2 + 3 OneVision gap 완전 close; Tier 2는 cross-panel polish.

**Deliverable.**
- Tier 1: `outputs/experiment_vqav2_onevision/<ts>/`, `outputs/experiment_e5c_vqa_onevision/<ts>/`, `outputs/experiment_e5c_tally_onevision/<ts>/`. Tables 2 + 3 OneVision 행으로 placeholder *(plan)* 닫음.
- Tier 2: `outputs/experiment_vqav2_<gemma3_4b|internvl3_8b|qwen2.5vl_32b>/<ts>/`. Table 2를 8-model → 11-model panel로 확장.
- §4.1 prose reframe — "legacy" 표현 삭제, "single-dataset depth panel" framing.

**Acceptance criteria.**
- Tier 1: (a) OneVision × VQAv2 `df(a)` ∈ [0.10, 0.18] 범위 (main matrix susceptibility 4위 ranking에서 추론, Gemma3-27b 0.167 ~ Qwen3-VL-30b 0.170 사이 예상). (b) OneVision × VQAv2/TallyQA wrong-base S1 (a − m) gap > 0 (§F.3의 chart-stack +0.7 ~ +6.6 pp prior). (c) Tables 2 + 3 placeholder *(plan)* numeric 값으로 대체.
- Tier 2: 추가 3 model이 panel에 합리적 위치 (예: gemma3-4b가 main matrix에서 가장 susceptible — VQAv2에서도 상위 expected).

**Owner.** thyun.park.

**Dependency.** None. 기존 driver / stimulus / config.

**Priority.** Tier 1은 P3 sprint 시작 시 *가장 우선* 실행 — 기존 driver에 새 코드 없고 paper-readability 직접 영향. Tier 2는 Tier 1 완료 후 *최종 polish* 시 결정.

---

## P4 — Future submissions (out-of-scope for current paper)

### P4-11 · Bonferroni-20 correction on Table 6 paired-test family

Trivial post-P1-3 (family-wise correction once paired CIs exist). Owner: thyun.park. Holds for revision pass after P1-3 lands.

### ~~P4-12 · OneVision E1d analyzer fix (Phase E)~~ ✅ closed 2026-05-10

Two analyzer bugs fixed in `scripts/analyze_causal_ablation.py`: (i) `_build_triplets` joined base/anchor on `sample_instance_id` only — added `dataset` to join key (commit `a7e391c`); (ii) per-run dataset key was unknown to analyzer — added timestamp→dataset hardcode for canonical Phase E runs + susceptibility-CSV qid-intersection auto-detect for re-runs (commit `de1f94e`). OneVision Main 5-dataset Δdf table integrated into paper §5.2 / §5.3 / Appendix §E.2 + `docs/insights/E1d-causal-evidence.md` 2026-05-10 update block. **Headline:** single-layer 5/5 null on OneVision (multi-layer redundancy claim *확장 검증*); upper-half 5/5 null at n=200 on OneVision (heterogeneous, dataset-dependent — §5.3 dataset-dependent peak와 일관, §6.2 subspace-projection 도구 선택 강화). PR: `paper/p4-12-onevision-e1d-analyzer-fix`.

### P4-13 · Pre-registration registry (OSF / AsPredicted)

R4 MAJ-6: not retroactive for current paper. Reserved for future submissions.

### P4-14 · Human baseline (50 Prolific subjects on 1-2 conditions)

Stronger cognitive-science framing. ~longer-term IRB / Prolific cycle. Future submission.

### P4-15 · γ-β cross-architecture replication (other Thinking-mode VLM pairs)

P0-1 sub-route. Useful follow-up if P0-1 lands positive on Qwen3-VL. ~1 H200-week per pair.

### P4-16 · §5.2 multi-layer redundancy formal definition

R5 bar-raiser axis 3/5 longer-term theoretical project. Not a paper edit; new theoretical paper or appendix.

### P4-17 · E4 generalization to SigLIP-Gemma early / Qwen-ViT late archetypes

R1-R4 carryover P3 task; ~1 H200-week per archetype. Useful breadth strengthening for follow-up paper.

---

## Priority summary table

| Pri | ID | Task | Cost | Tier impact |
|---|---|---|---|---|
| **P0** | P0-1 | γ-β residual-stream bridge experiment (cheap form) | ~2 H100-day | **Tier-shifter** — bar-raiser signature ask. Single highest-leverage move. |
| **P0** | P0-2 | Eigenvalue spectrum at L=26 (rank-8 elbow check) | ~4 H100-hour | Theoretical contribution upgrade if elbow clean. |
| **P1** | P1-3 | §6.2.3 Table 6 paired-bootstrap CI | ~1 day | Closes R4 MAJ-4 (5/5 → 4/5 + InfoVQA fence). |
| **P1** | P1-4 | CAA K=1 + ITI Table 7 empirical rows | ~3 H200-day | Closes R4 MAJ-5 (structural Note → empirical). |
| **P1** | P1-5 | Random-K=8 baseline for §6.3 (Alt-1) | ~2 H100-day | Closes R4 CRIT-3 (b-arm em alternative). |
| **P1** | P1-6 | §A.5 27-cell pilot 4-metric heatmap | ~1 day | Closes R4 CRIT-2 (cherry-pick). |
| **P2** | P2-7 | E6 cross-arch replication on Qwen2.5-VL | ~10 H200-day | Partial close of R4 CRIT-1 (N=1 → N=2). |
| **P3** | P3-8 | Paraphrase robustness (5 prompts × 5 datasets) | ~3 H200-day | Defuses single-prompt critique. |
| **P3** | P3-9 | Closed-source defuse (~500 sample on GPT-4o / Gemini 2.5) | ~1-2 day + ~$15 API | Defuses open-only critique. |
| **P3** | P3-10 | Encoder-family promotion (prose) | ~half-day | Camera-ready polish. |
| **P4** | P4-11..17 | Future submissions / longer-term | varies | Out of scope for current paper. |

---

## Recommended execution sequence

**Week 1 (sprint 1, Findings hardening):** P0-1 (cheap) + P1-3 + P1-6 + P1-5 in parallel (different compute resources possible). Total ~5-7 day wall-clock with parallelism. Cost: ~3 H100-day + ~2 H100-day + ~1 day script work.

**Week 2 (sprint 2, Main shift):** P0-2 + P1-4 + (begin P2-7). P0-2 + P1-4 finish in <3 day; P2-7 spans into week 3.

**Week 3 (sprint 3, Main consolidation):** Complete P2-7. If P0-1 cheap form was positive, run clean form. Update paper (§4.6.1, §6.7, §1 framing, §8.4 strikethrough).

**Camera-ready (post-Findings/Main decision):** P3-8 + P3-9 + P3-10.

**Future submissions (independent track):** P4-13, P4-14, P4-16.

---

## What does NOT need doing

Per R5 bar-raiser's 7-item protect-list:

1. (a − m) calibration contrast — load-bearing as-is.
2. Single-model 6-callsite hedge — doesn't need further softening.
3. §6.2.3 reframing — current "5/5 with InfoVQA fence" is correct.
4. Δem(non-anchored) ≥ 0 clause — substantive, not promotional.
5. §1.5 (1) "first-evidence 평가 프레임워크" hedge stack — appropriate.
6. §5.3 dataset-dependent peak self-disclosure — strength, not weakness.
7. §4.7 InternVL3 boundary case — correctly framed.

Do NOT touch these in P1-P3 work.

---

*End of post-review improvement plan. Next iteration: execute P0 sprint, then re-run /paper-review-loop on the revised paper to verify the bridge experiment outcome lands the framing changes correctly.*
