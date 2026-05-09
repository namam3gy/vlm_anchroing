# Paper Review Loop — Final Summary

**Paper:** `docs/paper/emnlp_draft_ko.md` — "Cross-Modal Numerical Anchoring in Vision-Language Models: Uncertainty, Plausibility, and Digit-Pixel Gates with a Deployable Subspace-Projection Mitigation."
**Loop dates:** 2026-05-09 (5 rounds in single day, alternating reviewer + author response).
**Final paper state:** 604 lines, v7 changelog footer.

## Round-by-round headline

| # | Reviewer persona | Headline critique | Author decision | Net edit posture |
|---:|---|---|---|---|
| 1 | Methodology reviewer | C-form metric definition needed audit; M2 18-variant analysis required justification; canonical CSV consistency | Accept core methodology critiques; add §B variant analysis appendix; backfill canonical citations | EDIT-heavy (new §B, M2 evidence file referenced, metric appendix expanded) |
| 2 | Writing reviewer | Forced Korean translations of technical terms (거리감쇠 / 사상 / 0결과); Table robustness ordering inconsistency; sentence-fragment endings; abstract overclaim ("입증") | Accept all writing fixes; restore English technical terms; tighten abstract; add Figure 1 in-text reference | EDIT-heavy (13 prose-level edits across abstract / §1 / §4 / §5 / §7 / §8) |
| 3 | Novelty / positioning reviewer | VLMBias mischaracterization in §2; missing CAA / ITI / LEACE prior-work block; §1.5 (1) "first-evidence" framing under-hedged; LRM / Wang 2025a citation ambiguity; strict free-lunch needs formal definition | Accept all positioning critiques; add §2 *Activation steering and concept erasure* paragraph (CAA / ITI / LEACE); add Hufe 2025 Dyslexify; add Lou and Sun 2024; add Chand et al. 2025 negative-result precedent; add §6.2.3 strict free-lunch 4-clause definition; add 4 references | EDIT-heavy (+11 edits, 4 new references, §2 substantially expanded, §1.5 (1) 7-axis hedge stack landed) |
| 4 | Aggressive adversarial reviewer | CRIT-1 E6 mitigation chain is N=1 model; CRIT-2 27-cell pilot grid two rounds deferred; CRIT-3 §6.3 b-arm em +8.8 pp post-hoc with un-falsified alternatives; MAJ-4 paired-bootstrap CI absent on Table 6; MAJ-5 CAA/ITI as Note not rows; MAJ-6 multiple-comparisons; MIN-10 FLUX seed missing | Accept all CRIT framings; do not contest; propagate **single-model case study** scope hedge across 6 callsites; add §6.3 Insight 1.5 (Alt-1 / Alt-2 enumeration); §6.2.3 신뢰구간 caveat + InfoVQA inconclusive fence; §A.4 FLUX seed=1729; §A.5 27-cell cell-label enumeration with chosen #17; §7 Bonferroni-6 robustness; §8.2 deferred-list expansion | EDIT-heavy + DEFER (6-callsite hedge propagation, §6.3 Insight 1.5, §A.4, §A.5, §7 Bonferroni paragraph; 4 items honestly DEFERred with GPU-hour estimates) |
| 5 | Bar-raiser (forward-looking) | Three threads (§4.6 γ-β / §5.2 multi-layer redundancy / §6.2 K=8 subspace) sit on same page but never causally interlock; signature ask = does (a−m) K=8 subspace amplitude grow over Thinking-mode trace and predict ×12.7? Most-citable 5-year finding = §5.2 → §6.4 predict-then-verify chain | Bar-raiser is forward-looking — bridge experiment is future work. Sharpen predict-then-verify chain framing (4 callsites: abstract / §1.3 / §1.5 (4a) / §8.1) as paper's *이론적* contribution; add §8.4 후속 작업 with bridge experiment as lead item + 5 secondary asks ranked; honor 7-item protect-list (no edits to (a−m) contrast / 6-callsite hedge / §6.2.3 reframe / Δem(b)≥0 clause / §1.5(1) hedge / §5.3 disclosure / §4.7 boundary case); one-sentence (a−m) design-pattern append in §6.2.1 | EDIT-light (4 framing-sharpening edits + 1 new §8.4 section + 1 design-pattern append + v7 changelog); 0 REBUT; 4 DEFER (bridge / spectrum / random-K=8 / cross-arch as future-work items with GPU-hour estimates) |

## Major themes addressed across rounds

1. **Methodology audit (R1)** — C-form direction-follow definition, M2 18-variant preservation analysis, canonical CSV traceability. Settled in §3.2 + §B + canonical evidence files at `docs/insights/_data/`.
2. **Korean / English technical-term register (R2)** — Korean prose with English technical terms (no forced translations). Established in user MEMORY + applied across all subsequent rounds.
3. **Positioning relative to prior work (R3)** — VLMBias / typographic attack literature / activation steering literature now properly differentiated. §2 substantially expanded; 4 references added; §1.5 (1) 7-axis hedge stack delivered.
4. **Strict free-lunch as formal criterion (R3 + R4)** — 4-clause definition (Δdf < 0 ∧ Δem(anchored) ≥ 0 ∧ Δem(non-anchored) ≥ 0 ∧ Δ(held-out capability macro) ≥ −0.5 pp); contrasted with Chand et al. 2025 LM-debiasing negative result; defended against R4 framing as "celebration criterion" → load-bearing screening rule.
5. **Single-model case study scope hedge (R4)** — E6 mitigation chain is N=1 model. Propagated across 6 callsites (abstract / §1.3 / §1.5 (5) / §6.6 / §8.1 / §8.2). Multi-model behavioral + mechanism panels (§3-§5) vs single-model E6 (§6-§7) panel-scope split made explicit.
6. **§6.3 b-arm em +8.8 pp alternative explanations (R4)** — Insight 1.5 enumerates Alt-1 (general regularization) + Alt-2 (numeric mode-collapse); POPE pinned-to-zero as partial signal weakening Alt-1 only; random-K=8 baseline DEFER.
7. **Multiple-comparisons + bootstrap CI (R4)** — §6.2.3 InfoVQA Δdf=−0.7 inconclusive fence; §7 Bonferroni-6 post-hoc check showing HallusionBench / POPE conclusions robust; full bootstrap CI on Table 6 DEFER.
8. **Predict-then-verify chain elevation (R5)** — §5.2 → §6.4 chain framed as *theoretical contribution* across 4 callsites (abstract / §1.3 / §1.5 (4a) / §8.1).
9. **Forward-looking bridge experiment (R5)** — γ-β residual-stream amplitude on Qwen3-VL-Thinking traces named as the elevating experiment; new §8.4 후속 작업 section with cheap/clean form, positive/negative implications, GPU-hour estimates.

## Outstanding DEFERred items (consolidated R1-R5, mapped to §8.4 priority)

| # | Item | Origin | Estimate | §8.4 priority |
|---:|---|---|---|---|
| 1 | **γ-β residual-stream bridge experiment (Qwen3-VL-Thinking trace amplitude on K=8 subspace)** | R5 bar-raiser signature ask | cheap ~2 H100-day, clean ~1 H200-week | §8.4 item 1 (lead — elevating) |
| 2 | Eigenvalue spectrum of `D[:, L=26, :]` rank-8 elbow check | R5 bar-raiser secondary | ~4 H100-hour | §8.4 item 2 |
| 3 | Random-K=8 subspace baseline (§6.3 Alt-1 falsification) | R4 CRIT-3 + R5 secondary | ~2 H100-day | §8.4 item 3 |
| 4 | Cross-architecture E6 replication (3 archetypes) | R4 CRIT-1 + R5 carryover | ~30 H200-day | §8.4 item 4 |
| 5 | CAA / ITI empirical Table 7 rows | R4 MAJ-5 | ~4-8 H200-hour + 1-2 day | §8.4 item 5 |
| 6 | §6.2.3 paired-bootstrap CI on Table 6 (5 cells) | R4 MAJ-4 | ~1 day | §8.4 item 5 |
| 7 | §A.5 27-cell pilot grid 4-metric heatmap aggregation | R4 CRIT-2 | ~1 day (existing data) | §8.4 deferred (accepted limitation) |
| 8 | Bonferroni-20 correction on Table 6 paired-test family | R4 MAJ-6 | trivial post-CI | §8.4 item 5 |
| 9 | OneVision E1d analyzer fix (Phase E) | R4 carryover from earlier rounds | 1-2 day | §8.2 deferred |
| 10 | Pre-registration registry document on OSF / AsPredicted | R4 MAJ-6 | not retroactive | future submissions |
| 11 | Paraphrase robustness (3-5 prompt variants × bootstrap CI) | R1-R4 carryover | 1-2 H200-day | §8.4 item 6 |
| 12 | Closed-source defuse (GPT-4o / Gemini 2.5, ~500 sample) | R1-R4 carryover | 1-2 day | §8.4 item 6 |
| 13 | Human baseline (50 Prolific subjects on 1-2 conditions) | R1-R4 carryover | longer-term | §8.4 item 6 |
| 14 | E4 generalization to SigLIP-Gemma early / Qwen-ViT late archetypes | R1-R4 carryover | ~1 H200-week per archetype | §8.4 deferred |
| 15 | γ-β cross-architecture replication (other Thinking-mode VLM pairs) | R4 carryover | ~1 H200-week per pair | §8.4 item 1 sub-route |
| 16 | Encoder-family bridge promotion to top-line contribution | R5 bar-raiser secondary | prose-only (camera-ready) | camera-ready editor |
| 17 | §5.2 multi-layer redundancy formal definition | R5 bar-raiser axis 3/5 | longer-term theoretical project | future submissions |

## Bar-raiser signature ask (verbatim)

> *"Does the (a − m) K=8 subspace amplitude grow over Thinking-mode trace generation, and does that growth quantitatively predict the ×12.7 correct-base amplification — measured by projecting the residual stream of Qwen3-VL-Thinking γ-β traces onto V_K[L=26] (cheap form, reuse OneVision subspace as instrument) or onto Qwen3-VL-Instruct's own calibrated subspace (clean form)?"*

— `docs/paper/reviews/round5_bar_raiser.md`, line 84.

## Diff stat (initial → final)

- **Lines:** initial 516 (start of loop, post-Phase-4 commit `d8710b2`) → 581 (post-R4) → **604 (post-R5)** = **+88 net** across 5 rounds.
- **Sections substantially rewritten:**
  - §2 *Activation steering and concept erasure* paragraph **newly added** (R3).
  - §6.2.3 strict free-lunch 4-clause formal definition **newly added** (R3).
  - §6.3 Insight 1.5 (b-arm em alternative explanations) **newly added** (R4).
  - §A.4 FLUX seed reproducibility **newly added** (R4).
  - §A.5 27-cell pilot grid cell-label enumeration **newly added** (R4).
  - §7 *Multiple-comparisons 보정* paragraph **newly added** (R4).
  - §8.4 후속 작업 (load-bearing follow-up) **newly added** (R5, 6 ranked items).
- **Sections substantially modified:** Abstract (R1 + R2 + R3 + R4 + R5), §1.3 (R3 + R4 + R5), §1.5 (R3 split (4)→(4a)/(4b) + R4 hedges + R5 (4a) sharpening), §6.2.2 (R4 deal-breaker rule + R5 §A.5 pointer), §6.2.3 (R3 + R4 신뢰구간 caveat), §6.5 (R3 CAA/ITI Note + R4 panel-scope qualifier), §6.6 (R4 single-model statement), §8.1 (R5 predict-then-verify framing), §8.2 (R4 deferred-list expansion).
- **Tables modified:** Tables 1-8 + A.5 + E.1 + E.2 — most R1 / R2 audits + R4 number consistency checks confirmed canonical against `_data/`. No metric numbers changed in R5.
- **Figures embedded:** 16 inline figure embeds preserved across all rounds. None added or removed in R5.
- **References added across rounds:** Belrose et al. 2023 (LEACE), Li et al. 2023 (ITI), Panickssery et al. 2024 (CAA), Chand et al. 2025 (No Free Lunch), Hufe et al. 2025 (Dyslexify), Lou and Sun 2024, Wang et al. 2025a (LRM judging bias), Hagendorff 2023 — total 8 added across R3-R5.
- **Changelog entries:** v3 (initial), v4 (R2), v5 (R3), v6 (R4), v7 (R5).

## Final tier verdict

**Solid Findings, top of band — borderline / weak-accept Main contingent on §8.4 item 1 (bridge experiment) landing positive in next revision.**

This anchor converges across three independent assessments:
- R4 author honest verdict: *"closer to strong Findings than weak-accept Main."*
- R5 bar-raiser tier verdict: *"Solid Findings (top of band) — weak accept Findings without hesitation; borderline Main with a clear note that the bridge analysis is what separates this paper from the upper half of accepted Main."*
- R5 author honest verdict (final): same as bar-raiser's anchor.

The 5-year most-citable finding the loop converged on (per R5 bar-raiser): **"Cross-modal anchor signal is multi-layer redundant in VLM residual streams, predicting and verifying that single-direction mitigation methods fail across datasets — motivating multi-direction subspace projection."** This is the §5.2 → §6.4 predict-then-verify chain, now framed across 4 callsites as the paper's *이론적* contribution.

## Recommendation for next pass

**If the goal is Findings acceptance:** the paper after Round 5 is *publishable as-is* per the bar-raiser's "weak accept Findings without hesitation" verdict. The 4 DEFERred load-bearing items (bridge experiment / paired-bootstrap CI / CAA-ITI rows / cross-arch E6) are honest deferrals with explicit GPU-hour estimates; reviewers who accept transparency about deferred work should clear Findings.

**If the goal is Main acceptance:** the *single highest-leverage next action* is **§8.4 item 1 — the γ-β residual-stream bridge experiment in cheap form** (~2 H100-day). Bar-raiser explicitly named this as the experiment that would tip the verdict. Cheap form is sufficient (reuses existing OneVision K=8 subspace as measurement instrument); clean form is preferred but not required. Positive outcome interlocks §4.6 + §5.2 + §6.2 into one mechanism-grounded chain. Negative outcome properly fences §4.6 as behavioral existence proof. Either resolves the ambiguity the paper currently leaves open.

**Secondary actions, ranked by cost-effectiveness:** §8.4 item 2 (eigenvalue spectrum, ~4 H100-hour, converts K=8 from grid-search artifact to data-property prediction) and §8.4 item 3 (random-K=8 baseline, ~2 H100-day, head-to-head Alt-1 falsification for §6.3) are the cheapest available rigor improvements on the entire DEFER list.

**Tertiary action:** Cross-architecture E6 replication (§8.4 item 4, ~30 H200-day) is the most expensive but converts CRIT-1 from "single-model case study" to "transferable recipe." If §8.4 item 2 lands rank-8 elbow positive, item 4 becomes "spectrum-predicts-cell" rather than "brute-force re-tune" — substantially elevated as a theoretical contribution.

**Camera-ready / venue decisions:** R5 bar-raiser secondary (encoder-family promotion to top-line contribution) + Δem(non-anchored) ≥ 0 clause renaming (if any future reviewer pushes back) are camera-ready discretion items. R4 MAJ-6 pre-registration registry is non-retroactive; reserved for future submissions.

---

*End of paper review loop. 5 rounds × 10 steps complete. Next iteration is the bridge experiment, not another review pass.*
