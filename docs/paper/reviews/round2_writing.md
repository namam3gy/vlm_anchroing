# Round 2 — Writing Review

**Reviewer persona:** Editor-quality reader, EMNLP / NeurIPS area chair tier. Korean academic register + English technical terms.
**Paper version reviewed:** `docs/paper/emnlp_draft_ko.md` @ 828 lines (post-Round-1 reviser pass v9, 2026-05-11).
**Date:** 2026-05-11.
**Scope:** prose flow, narrative arc, terminology consistency, table/figure–prose alignment, Korean academic register, abstract pacing, redundancy. **Out of scope:** numerical correctness against canonical CSVs (Round 1's job) — except where Round 1 fixed a number but the surrounding prose was left stale.

---

## Decision

**Borderline — accept with major revision (writing axis only).** Round 1's surgery genuinely landed the worst structural debt (changelog removed, §1.5 6→4 contributions, §3.3 panel-scope hedge consolidation, §4.6 25→13 lines, inline task IDs / generator-script citations stripped). The paper is no longer a project tracker disguised as a manuscript. **But three writing-axis blockers remain that any top-tier copy-editor will flag in five minutes**: (i) the Abstract is a 680-word three-paragraph block that is roughly 2.7× the venue norm and reads as a self-contained compressed paper rather than an Abstract; (ii) §6.6 contains a single 5-arrow cross-reference chain (line 428: `(§5.2 → §5.3 → §5.4 → §6.1/§6.2 → §6.4 → §7)`) that is the single clearest example of *prose substituting cross-references for argument* surviving Round 1; (iii) the title's metadata sub-line `*EMNLP 2026 long paper, 중간 점검용 한글본 (2026-05-09)*` (line 5) is internal-tracking content that does not belong in any submitted manuscript. Given that all three are surgical fixes a competent reviser can land in one sitting, **borderline-accept-with-major-revision** is the honest call. The paper has the substance to clear the writing bar; it has not yet been edited for that bar.

---

## Severity-graded issues

### CRITICAL (writing axis blockers — must fix before submission)

- **[CRIT-W1] Title metadata not stripped (line 5).** The title block ends with `*EMNLP 2026 long paper, 중간 점검용 한글본 (2026-05-09)*`. This is the literal phrase "long paper, mid-check Korean draft (2026-05-09)" — internal versioning that no submission carries. Round 1's structural surgery removed the v3-v8 changelog from line 831–833 but left this twin artefact on line 5. **Fix**: delete the entire line 5 italic sub-line; line 1 (English title) and line 3 (Korean title) suffice.

- **[CRIT-W2] Abstract is 680 words / 74 sentence-marks / 3 paragraphs — ~2.7× venue norm.** EMNLP / NeurIPS abstracts are a single paragraph ≤ 250 words. The current Abstract (lines 9–13) packs every methodological hedge from the body into nested parentheticals. Sample density check on a single sentence (line 11): *"Direction-follow는 *base-prediction의 answer-token entropy*에 단조 증가한다 (B6 − B1 6-bin gap **+19.5-23.5 pp** on 5 dataset × 6 model heterogeneous coverage = 80 anchor cell, ≥ 4/5 strict pair-wise ↑ on **51-57 / 80 cells** — 1 noise dip 허용 시 substantively monotonic)."* — one sentence with a 5-clause parenthetical containing two ranges, two count fractions, and a substantive-monotonicity hedge. The Abstract does this approximately twelve times. **Fix**: rewrite the Abstract to a single paragraph ≤ 250 words. Move (a) the 51-57/80 cell-count hedge, (b) the legacy-VQAv2 §C.1 cross-reference, (c) the 5-baseline panel enumeration `(ActAdd, LEACE rank-1, query-adaptive, CogBias decode-time, MIA-DPO LoRA)`, (d) the within-`*existence proof*` n=1×n=1 hedge, and (e) the case-study scope hedge to body sections — those readers who reach §6.5/§8.2 will find them; those who only read the Abstract should find a *claim*, not an audit trail.

- **[CRIT-W3] §6.6 contains a 5-arrow chain replacing argument with cross-references (line 428) AND that paragraph then *immediately self-duplicates* in `**Summary.**` (line 430).** Verbatim line 428: *"single chain (§5.2 multi-layer redundancy + §5.3 OneVision fragility → §5.4 framework → §6.1 routing site / §6.2 integration site → §6.4 single-direction failure → §7 capability preservation) 을 형성한다. 본 framework의 direct falsifiable prediction — Thinking-mode trace 생성 동안 K=8 subspace amplitude가 자라야 한다 — 은 §8.4 후속 작업 1번으로 명시된다."* Then line 430 (Summary): *"본 framework의 direct falsifiable prediction (Thinking-mode trace 생성 동안 K=8 subspace amplitude가 자라야 한다)은 §8.4 후속 작업으로 명시된다."* — the *exact same forward-pointer*, two paragraphs apart, in the same sub-section. The two paragraphs' *first sentence* is also nearly identical: line 428 *"§6.1 (E4)과 §6.2 (E6)는 §5.4 framework가 사전 예측한 *두 mitigation site*의 직접 검증이다"* vs. line 430 *"§6.1 E4와 §6.2 E6는 §5.4 framework의 두 mitigation site (routing · integration) 의 직접 검증이며"*. **Fix**: collapse §6.6 to one paragraph; drop the 5-arrow chain entirely (replace with a single declarative sentence); remove the duplicated `Summary.` block.

### MAJOR

- **[MAJ-W1] §4.1 opens experiment-first, not claim-first (line 115).** Verbatim opening: *"6-model PlotQA panel (n=5,000 base per model; S1 anchor `|a − GT| ≤ max(1, 0.10·GT)`)에서 두 패턴이 즉시 부각된다 (Table 2)."* This is "we ran experiment X on dataset Y with stratification Z; two patterns appear" — a lab-report opening. A conference-paper §4.1 should open with the *claim* (graded vs categorical pull) and then introduce Table 2 as evidence. The user's primary concern was that the paper still reads as chronological experiment-dumping; §4.1 is the strongest single confirmation. The same shape repeats in §4.4 (line 203 — *"§4.1의 wrong-base / correct-base 분할은 더 풍부한 연속 구조의 거친 projection이다."* — this is fine actually, builds on §4.1), §4.5 (line 215 — opens with experiment description), §4.6 (Round 1 already restructured this — **claim → setup → evidence → insight** form on line 235; this is the model the rest of §4 should follow). **Fix**: reopen §4.1 with one sentence — e.g. *"PlotQA depth panel은 anchor pull이 categorical capture가 아닌 graded pull임을 보인다: 6/6 모델 모두에서 paired adoption은 1.7-15.7 %로 floor 수준인 반면 direction-follow는 0.059-0.325로 그보다 1.5-8× 크다 (Table 2)."* Then introduce Table 2.

- **[MAJ-W2] §6.2.3 self-reframe (line 383) is a ~600-word inline paragraph that walks back the table immediately preceding it.** Round 1's CRIT-2 reframe surfaced the right nuance ("Δem(b) is multiplicity-robust headline; Δdf is sample-size-bound to PlotQA"), but the form is a paragraph-long apology after Table 7. The proper conference-paper form is to encode the nuance in the table itself (bold-marker = CI-clean direction-of-headline; non-bold = point-estimate-only). The current Table 7 *almost* does this — bold marks "95 % CI excludes 0" — but the prose still re-derives the cell-by-cell story afterwards because the column-header convention isn't loud enough. **Fix**: (i) add a small sub-table of *sign-clean count by metric × multiplicity* (the current sentence-form table at line 376 already exists — keep it but lift higher), (ii) compress the line-378 paragraph to two sentences ("Δem(b) is the multiplicity-robust headline (5/5 sign-clean × Bonferroni). Δdf(a) is sample-size-bound: PlotQA n=2,306 only at 95 % CI, 4 small-n cells point-estimate-consistent.") and trim the rest. The "Per-dataset cousin" pointer to `E6-stage4-paired-bootstrap-ci.md` at the end of line 383 is the residue of an inline canonical-source citation Round 1 missed — it should be removed (move to §A.5 reproducibility pointer).

- **[MAJ-W3] `이분법` survives in 3 high-traffic load-bearing positions (lines 11, 31, 460) — repeats the v1-archive flag.** v1-archive `[CRIT-W4]` flagged "이분법" as 6 occurrences in load-bearing prose; the v3 changelog claimed it would be cleaned ("이분법" → "wrong-base/correct-base 분할"). Verbatim survivors: line 11 (Abstract): *"통상 보고되는 *base-correct vs base-wrong* 이분법은 이 연속 gradient의 거친 *binary projection*"*; line 31 (§1.2 pillar 3): *"통상의 *wrong-base vs correct-base* 이분법은 이 연속 gradient의 거친 binary projection"*; line 460 (§8.1): *"wrong-base / correct-base 이분법은 그 gradient의 거친 binary projection이다"*. The Korean academic register meaning of "이분법" is "false dichotomy / binary thinking" — a philosophy term, not a statistics-stratification term. **Fix**: replace all three with `wrong-base / correct-base 분할` or `binary stratification`. The Round-1 reviser response did not list this in its edit log so this is a Round-1 failure to deliver on the v3-changelog promise.

- **[MAJ-W4] `발화` (linguistic "utterance / speech-act") used metaphorically for hook firing — survives at lines 325 and 329.** v1-archive `[MAJOR-W6]` flagged "발화" as a Korean register break. Verbatim survivors: line 325 (§6.1 Insight 2): *"Hook은 모든 forward pass에서 발화하지만, target-only `acc(b)` 불변 ..."*; line 329 (§6.1 closing paragraph): *"본 논문의 deployable claim은 §6.2의 E6 — input의 anchor 위치를 모르고 universal projection으로 작동 + anchor label 무관 + cross-arm 모두 발화 — 에서 별도 검증한다."* "발화" in Korean academic register is reserved for linguistic acts (speech-acts, utterances). A hook does not "utter" — it triggers / fires / activates. **Fix**: replace `발화` with `트리거` (hook 트리거) or `forward에서 동작` consistently.

- **[MAJ-W5] ×12.7 ratio appears in 5 sites in the body (lines 13, 31, 39, 53, 225, 460) — high redundancy.** Sites: Abstract (line 13), §1.2 pillar 3 (line 31), §1.4 (line 39), §1.5 auxiliary (line 53), §4.5 Table 4 (line 225), §8.1 종합 (line 460). The number itself is a single point-estimate from a single architecture × single dataset (acknowledged §8.2). Each occurrence carries a different hedge: line 13 ("point estimate ×12.7, denominator small"), line 39 ("point estimate ... small-denominator's noise sensitivity 때문에 '>5× directional'"), line 53 ("correct-base df ratio point estimate ×12.7"), line 225 (table cell with no hedge), line 460 ("correct-base subset df push up — binary 평균 차이 축소"). The hedges contradict each other (one says ">5× directional", another says "×12.7" without the directional qualifier). **Fix**: single canonical hedge phrase ("point estimate ×12.7, denominator small; ≥ 5× directional") used at first mention (Abstract); subsequent §1.4 / §1.5 / §8.1 references say "×12.7 ratio (§4.5)" without re-deriving the hedge.

- **[MAJ-W6] `free-lunch` term family is inconsistent — 4 surface variants in 16 occurrences (incl. one all-caps code-style "STRICT_FREE_LUNCH" at line 448).** Variants observed: `free-lunch` (Abstract line 13, §1.5 line 51, §6.5 Table 8 line 422, §6.5 Insight line 424, §7 line 436, §8.1 line 460, §8.2 line 470), `4-clause free-lunch` (§6.5 title line 411, §6.5 Insight line 424), `STRICT_FREE_LUNCH` (§7 Table 9 row label line 448 — uppercase code-identifier mid-prose), `strict-free-lunch` (§6.2.3 self-reframe line 383, §7 line 450 in *"strict-free-lunch 판정은 두 panel에서 모두 유지"*), `free-lunch criterion / free-lunch 기준` (§6.2.3 line 385 formal definition). The §7 Table 9 column convention is `STATUS = STRICT_FREE_LUNCH` (uppercase code) for one row only — that is a code-identifier from `headline-numbers.md`, lifted into the table without translation. **Fix**: pick `4-clause free-lunch` as canonical (matches the formal definition at §6.2.3 line 385); replace `STRICT_FREE_LUNCH` in Table 9 row with `4-clause free-lunch` or `OK (4-clause)`; verify §6.2.3 line 383 uses `4-clause free-lunch` not `strict-free-lunch`.

- **[MAJ-W7] §1.4 vs §4.5 redundancy is structural, not surface.** §1.4 (lines 39, ~13 lines) and §4.5 (lines 213-231, ~19 lines) describe the γ-β reasoning amplification result with substantial overlap. §1.4 carries: Thinking ×1.6 / df ×2.9 / correct-base ×12.7, Mussweiler-Strack confidence-axis prediction, acc(d) lower (reasoning doesn't buy accuracy), single-architecture / single-dataset existence-proof hedge. §4.5 *re-derives every one of these* with the same numbers, the same 4.4 framework prediction, the same Insight 3 controls (Instruct 0.647 vs Thinking 0.587). The user's audit point #9 (γ-β redundancy across §1.4 and §4.5) is empirically true. **Fix**: collapse §1.4 to a 2-3 sentence motivational paragraph (Thinking amplifies; ×12.7 correct-base ratio; auxiliary observation; full results §4.5) — *do not duplicate the explanation*. §1.5's auxiliary observation (line 53) already captures most of §1.4 prose; the remaining content of §1.4 should be merged either into §1.2 (as a fourth pillar mention) or pushed entirely to §4.5.

- **[MAJ-W8] Cross-section hedge proliferation despite Round-1 consolidation.** Round 1's §3.3 canonical hedge edit (line 107: *"Panel scope by analysis axis (canonical hedge — once)"*) explicitly promised: *"이 panel-scope 분리는 본 절에서 단 1회 명시하며, 후속 절은 reference 이외에 반복하지 않는다."* But line 430 (§6.6 Summary) repeats *"§6 mitigation 결과는 single-model case study register에 속한다"*; line 466 (§8.2 first bullet) repeats *"E6 mitigation chain은 단일 모델 case study"*; line 470 (§8.2 fifth bullet) repeats *"(E4·E6 모두 *각자의 panel scope* 안에서만 검증됨 — E4: 3 mid-stack 모델, E6: OneVision Main 1 모델.)"*. Three sites still re-state the panel-scope hedge after Round 1's "once" consolidation rule. **Fix**: delete the §6.6 panel-scope clause (the §3.3 reference is enough); shorten §8.2 first bullet to one sentence pointing back to §3.3. The fifth bullet's parenthetical can stay (it lists scope per mitigation, not the hedge).

- **[MAJ-W9] Insight box density review — at least 5 boxes restate the table they sit beneath.** I count 28+ Insight boxes across §4-§7. Verbatim flags:
  - **§4.3 Insight 1** (line 195): *"df 부호가 5 dataset × 6 model × 30 cell 모두 양 — 효과가 모델·데이터셋에 무관하게 *보편적*이다."* — restates Figure 4 caption: *"df 부호 30/30 cell 모두 양수."* Same datum, no extra mechanism. **Reclass**: load-bearing only because of the forward-pointer to §6.2 mitigation universality; could be one sentence not a separate Insight.
  - **§6.2.3 Insight 1** (line 387): *"Δdf 감소량은 PlotQA (−5.2 pp, 가장 큰 baseline df)에서 가장 크고 TallyQA (−0.3 pp, df 거의 floor)에서 가장 작다."* — directly readable from Table 7. The follow-up sentence (*"projection이 *dataset-shared subspace를 amplitude-dependent*하게 청소한다는 가설과 일치"*) is the actual insight; the first sentence should be inline with Table 7 caption.
  - **§4.1 Insight 1** (line 140) — the parenthetical breakdown of robustness ordering (*"Gemma3-4b 가장 큼 0.294 → Gemma3-27b 0.118 → LLaVA-OneVision *(Main)* 0.130 → Qwen2.5-VL-7b/32b 가장 robust 0.059 동률"*) is a verbatim re-listing of Table 2's `df(a)` column.
  - **§5.2 Insight 1** (line 278): *"Single-layer null은 메커니즘 해석에 직접적 결과를 가진다. *attention peak이 가장 큰 mass를 가진다*는 사실이 그 layer가 *causal site*임을 의미하지 *않는다*."* — load-bearing.
  - **§5.2 Insight 4** (line 284): *"§5.2의 multi-layer redundancy + §5.3의 OneVision peak fragility가 *attention pathway routing의 redundancy*와 *residual stream integration의 가용성*이라는 두 mechanism-level 사실의 두 측면이며, §5.4가 이를 단일 framework로 통합해 §6의 네 mitigation 결정을 사전 예측한다."* — this is a *forward-reference to §5.4 + §6* dressed as an Insight. Belongs in the §5.4 opening sentence, not as an §5.2 Insight box.
  - **§7 Insight** (line 452): *"HallusionBench의 양적 결과는 우연이 아니다 — §6.3의 (a − m) contrast가 *wrong-base의 generic error mode*까지 capture한다는 가설의 *외부 검증*이다."* — load-bearing, well-placed.
  
  **Fix**: drop §4.3 Insight 1 first sentence (merge with Figure 4 caption); compress §6.2.3 Insight 1 to its second sentence; rewrite §4.1 Insight 1 to lead with mechanism (능력↔끌림 역상관) before re-listing models; move §5.2 Insight 4 forward-pointer into the §5.4 opening sentence.

### MINOR

- **[MIN-W1] §6.5 Table 8 verdict column mixes Korean + English.** Verbatim row 1 verdict: *"방향 불일치 (direction mismatch)"* — Korean glossed by parenthetical English. Row 2: *"동일 원인 (same single-direction redundancy issue)"*. Row 4: *"동일 근원 (decode-time direction = single-direction failure)"*. Row 5: *"em 부수효과 + 학습 분포 편향 (em side-effect + training distribution bias)"*. Row 6: *"권장 — 4-clause free-lunch 통과 (recommended)"*. The Korean+English-gloss pattern is consistent within Table 8 but appears nowhere else in the paper; it reads as a translation crutch. **Fix**: pick one — drop the Korean if English is canonical, or drop the English glosses if Korean is the table register.

- **[MIN-W2] `끌림` vs `anchor pull` vs `pull` — Round 1 did not converge despite v1-archive flag.** Verbatim: line 11 (Abstract) `graded pull`; line 13 (Abstract) `anchor pull`; line 117 (§4.1) `anchor 쪽 graded movement`; line 130 (Table 2 footer) `능력↔끌림 역상관`; line 140 (§4.1 Insight 1) `끌림 크기`; line 183 (§4.2 Insight 1) `모델 끌림 강도`; line 193 (Figure 4 caption) `가장 큰 끌림`; line 387 (§6.2.3 Insight 1) `큰 끌림 → 큰 감소`. The `끌림` / `anchor pull` alternation appears within the same sentence at line 39: *"Qwen3-VL-8B-Thinking은 Instruct 변형 대비 adopt ×1.6, df ×2.9 — 그러나 *correct-base* 부분집합에서 df 비율은 point estimate **×12.7** ... main panel 전반에서 binary projection이 보였던 wrong > correct gap이 reasoning mode에서 *축소*된다 ... Reasoning trace가 *정확도 향상 없이* anchor robustness를 *낮춘다*"* — `anchor robustness` here is a third variant. **Fix**: pick `anchor pull` (Abstract canonical) globally; reserve `끌림` for *figure captions only* if at all. v1-archive flagged this; reviser did not deliver.

- **[MIN-W3] §3.1 Figure 1 in-text reference is too late and weak.** Line 84 (closing of §3.1): *"...자극 inventory는 128개 FLUX-rendered digit 이미지 (`a`) + 128개 OCR-검증된 Telea inpaint (`m`) + 128개 digit-free FLUX render (`d`)이다 (부록 §A; 4-조건 자극이 모델별 acc drop에 미치는 효과 예시는 Figure 1)."* The Figure 1 reference is buried in a parenthetical *after* the body sentence ends. v1-archive flagged this (`[MAJOR-W11]`). **Fix**: move "(Figure 1 참조)" to the opening sentence of §3.1 — *"한 sample_instance마다 최대 4개 조건 (Table 1; Figure 1)을 평가하고 ..."* — or rewrite the parenthetical as its own sentence.

- **[MIN-W4] §4.4 line 203 has a >150-word run-on sentence with three nested parentheticals.** Verbatim (single sentence, lightly bolded markers preserved): *"각 cell의 `target_only` row를 answer-span logit 기반 confidence proxy로 **6개 equal-frequency bin (B1 = 가장 confident, B6 = 가장 uncertain)** 으로 split한 후 bin별 adopt와 df를 계산하면, **5 dataset {TallyQA, ChartQA, MathVista, PlotQA, InfoVQA} × 6 model heterogeneous-coverage panel의 80 anchor cell에서 평균 B6 − B1 gap이 df +0.195** (`cross_entropy`, length-invariant paper-clean default) ~ **+0.235** (`log_prob_sum`, length-aware), **5 pair 중 4 pair 이상이 strict ↑인 cell은 cross_entropy 51 / 80 (64 %), log_prob_sum 57 / 80 (71 %)** — 즉 1 bin-pair noise dip을 허용하면 panel 다수가 *substantively monotonic*."* Sentence count: *one*. Word count: ~170. Parenthetical count: 3 nested. **Fix**: split into three sentences — (1) the binning procedure, (2) the headline result, (3) the proxy-name caveat.

- **[MIN-W5] §4.4 line 203 then immediately appends another 60+ word run-on with proxy-comparison + source citation.** Same paragraph: *"참고로 legacy `softmax_top1_prob` proxy 도 동일 6-model panel에서 일관된 신호를 보인다 (df B6 − B1 평균 +0.181, ≥ 4/5 strict 46 / 80 (58 %), fully strict 5/5 15 / 80 (19 %)). 본문 headline은 `log_prob_sum`을 보고하고 부록에서 세 proxy를 모두 표로 제공한다 — 현재의 운영적 confidence proxy 선택이 세 정의에 모두 정렬되도록 한 조치이다 (출처 `docs/insights/L1-confidence-modulation-evidence.md` 2026-05-10 update)."* The trailing inline canonical-source citation `(출처 docs/insights/L1-confidence-modulation-evidence.md 2026-05-10 update)` is a Round-1 leftover that the reviser response listed as fixed in §A.5 reproducibility lift but evidently did not strip from §4.4. **Fix**: remove the inline citation; pointer to `confidence proxy 선택` lives in §B.1 (which already discusses 4-vs-6-vs-10 binning).

- **[MIN-W6] §3.3 line 105 mixes raw n and per-cell stratified n in one sentence.** Verbatim: *"위 raw n은 stratification · eligibility 필터 *이전* count이며, 실제 본문 표에 사용된 per-cell n은 stratified 부분집합 기준으로 ChartQA 129–517 / TallyQA 6,934–14,772 / PlotQA 926–4,610 / InfoVQA 218–865 / MathVista 127–274 *범위에 분포*한다 (모델별 변동; 자세한 per-cell n 표 부록 §A.5 reproducibility)."* Round 1 attempted to address this (response Edit MIN-1 says "done"), but the form is still a single sentence stuffing 6 datasets × 6 models worth of n-range information into nested parenthetical comma-lists. The body of §3.3 should not need to cite ChartQA 129 (which Round 1 reviewer flagged as small enough to give §6.2.3 ChartQA CI half-width of ±5–6 pp). **Fix**: move the per-cell n range into §A.5 (the reproducibility appendix already exists for this purpose); §3.3 prose says only "raw n above is pre-filter; per-cell n ranges given §A.5".

- **[MIN-W7] §6.2.1 method block has English imperative sentences inside Korean paragraphs.** Line 343: *"`h(x, L) ∈ R^d`를 input `x` 위에서 layer L의 마지막 input token residual로 둔다."* — Korean. Line 349: *"Projection은 *universal*이다 — 어떤 input 정보도 anchor-present를 anchor-absent와 구별하지 않는다. K-dim anchor subspace를 모든 forward pass에서 silently 제거."* — second sentence ends with the verb `제거` as a noun-form (`제거.`), making it a verbless fragment. **Fix**: `silently 제거한다.` (verb-form ending).

- **[MIN-W8] §4.4 Insight 1 contains italics-mid-Korean construction that breaks reading flow.** Line 207: *"이는 §6.2 mitigation 설계가 *categorical wrong-base flag*를 별도 입력으로 받지 않고 residual representation 자체에서 universal projection으로 작동하는 것을 정당화한다 (입력 anchor 라벨이 필요 없는 design choice는 본질적으로 *연속 gradient* 가설에 정렬)."* — italicising `categorical wrong-base flag` and `연속 gradient` mid-Korean noun-clause looks like translator's emphasis hangover. The paper uses italics for technical-term first-introduction elsewhere, but here both terms have been used multiple times by §4.4. **Fix**: drop both italic markers; standard Korean prose carries the emphasis without typographic load.

- **[MIN-W9] §5.4 lines 296-298 has three "Framework" headings (`**Framework 정의.**`, `**Framework 정리.**`, `**Framework의 prospective test (§4.6).**`) inside one sub-section.** Three bold-headed "Framework X" paragraphs inside §5.4. The bold-head pattern works for Insight boxes (§4.x Insight 1/2/3) because the boxes are *parallel claims*; here the three bolds are *sequential narrative beats* (definition → consolidation → prospective test) which read better as three flowing paragraphs. **Fix**: drop the bold headings; let the paragraph topic-sentences carry the structure.

- **[MIN-W10] §6.5 Table 8 caption is empty; the table is preceded by a single-line statement *"Table 8. Multi-method 비교. Bold = 본 작업."*** Other tables (§4.1 Table 2, §4.2 Table 3, §6.2.3 Table 7) have multi-sentence captions explaining the slice + aggregator + bold convention. Table 8 is the centerpiece of the §6.5 "uniquely passes" claim — its caption should at minimum say what the panel scope is (single OneVision Main run, not multi-model comparison) and what "Cross-dataset 감소" means as a column header (a 5-dataset summary, not a single number). **Fix**: expand caption to 2-3 sentences, mirroring §4.1 Table 2 caption rigor.

- **[MIN-W11] §1.5 contribution (1) is internally awkward.** Line 45: *"**Cross-modal anchoring 평가 프레임워크.** *Independent-anchor open-numeric-estimation* anchoring 측정 protocol — (a) 질문 subject로부터 분리된 independent rendered-digit anchor 자극 4-condition (b/a/m/d), (b) gt-자유 baseline-relative direction-follow 표준 metric `(pa−pb)·(anchor−pb) > 0 ∧ pa ≠ pb`, (c) (a − m) digit-pixel paired-inpaint contrast — familiar-subject counting을 다루는 VLMBias [Vo, Nguyen et al., 2025]와 측정 대상 · cue source 측에서 상보적 (§2 · §3)."* The contribution noun-form is *"평가 프레임워크"* but the body is *"protocol — (a) ..., (b) ..., (c) ..."*. The compound `*Independent-anchor open-numeric-estimation* anchoring 측정 protocol` has *anchoring* twice and *anchor* once in three words. **Fix**: rewrite as a sentence — *"본 논문은 cross-modal anchoring 평가 프레임워크를 도입한다: 4-조건 자극 (b/a/m/d), gt-자유 baseline-relative direction-follow metric, (a − m) digit-pixel paired-inpaint contrast의 세 요소가 VLMBias의 familiar-subject counting paradigm과 cue source · 측정 대상 측에서 상보적이다 (§2 · §3)."*

---

## Structural critique — does the prose still read as experiment log?

**Round-1 reviser surgery succeeded on the structural-debt axis.** The v3-v8 changelog at lines 831-833 is gone (moved to `CHANGELOG.md`). The §1.5 contribution count is down from 6+1 to 4+1. Inline canonical-source citations (`docs/insights/E*-*.md`, `_data/*.csv`, generator script names, internal task IDs `P1-3`/`P4-12`/`MAJ-4`) are removed from body prose. §4.6 is compressed from 25 lines to 13. The §3.3 panel-scope canonical hedge consolidation works.

**But the residual lab-log signature shows up in three places:**

1. **§6.6 cross-reference chain (line 428).** This is the single clearest "prose substituting cross-references for argument" in the paper. A reader who reaches §6.6 has read the entire body — they do not need a `(§5.2 → §5.3 → §5.4 → §6.1/§6.2 → §6.4 → §7)` retrace. This is a *table of contents* in disguise.

2. **§4.1 / §4.5 experiment-first openings.** Despite §4.6 being rewritten by Round 1 to **claim → setup → evidence → insight** form (which works well — line 235 *"§5.4 routing vs integration framework는 anchor 정보가 mid-stack에서 V_K subspace를 *suppress*하다가 late-stack에서 *integrate*하는 layer-routed sign-reversal을 예측한다. 본 절은 framework 작성 *이후* 실행된 Qwen3-VL self-calibration 실험으로 이 예측을 직접 검증한다."*), §4.1 still opens with experiment-first prose (line 115: *"6-model PlotQA panel ... 에서 두 패턴이 즉시 부각된다"*). The user's primary concern (paper reads as chronological experiment dump) lives most strongly in §4.1 / §4.4 / §4.5 openings; Round 1's §4.6 rewrite is the model the rest of §4 should follow.

3. **§6.2.3 paragraph-length self-reframe (line 383).** Round 1 honestly surfaced the table-vs-claim mismatch ("Δdf is sample-size-bound, Δem(b) is multiplicity-robust") but in inline ~600-word paragraph form — a "lab notebook walk-back" register, not a conference-paper register. Conference form: encode the nuance in the table (column-header convention + bold = CI-clean) and use 1-2 sentences of prose.

**The paper has *moved* from "active experiment log" to "structurally clean draft with three lab-log residues."** Round 1 did not finish the job; it did about 70% of it. Round 2 surgery (Abstract compression + §6.6 chain removal + §4.1 opening rewrite + §6.2.3 prose compression) closes the remaining 30%.

---

## Per-section pass

### §1 Introduction

- **§1.1 (line 19-23)**: opens with motivation question, names prior work clearly. **OK.** The "*가장 가까운 이웃*" framing in §1.1 line 23 is good positioning prose — it acknowledges related work without overclaim.
- **§1.2 (line 25-31)**: three pillars laid out, but pillar 3 (line 31) repeats Abstract claim verbatim with the `이분법` term flagged in MAJ-W3. Line-31 prose `이 binary projection이 *깨지는* 것은 ... 직접 증거이다 (§4.5)` is forward-reference packing — the §4.5 evidence is not summarised, just pointed to.
- **§1.3 (line 33-35)**: long single paragraph (~7 lines of dense prose). Round-1 reviser response Edit 2 reframed framework framing here, but the result is *one paragraph carrying Abstract + §5.4 + §6.1/§6.2 + §6.4 + §7 in compressed form*. Effectively "Abstract take 2." **Fix**: split into 2 paragraphs — (a) mechanism finding (multi-layer redundancy + framework synthesis + §4.6 prospective test), (b) two mitigations (E4 routing-site, E6 integration-site, capability preservation).
- **§1.4 (line 37-39)**: γ-β reasoning amplification — *redundant with §4.5* per MAJ-W7 above. **Fix**: collapse to 2-3 sentences.
- **§1.5 (line 41-53)**: 4 contributions + 1 auxiliary observation. Better than v1-archive 6+1. Contribution (1) awkward (MIN-W11). Contribution (3) and (4) are the strongest; (2) is OK. The auxiliary observation block (line 53) is fine as a structural demotion of (6) → auxiliary; **but** the auxiliary observation prose (line 53) ends with `(§4.5, §8.2 한계)` cross-pointer — that's the third forward-reference to §4.5 in §1 (lines 31, 39, 53).

### §2 Related Work

Three paragraphs (LLM anchoring lineage + multimodal cognitive bias + typographic-attack/mechanism + activation steering). **Generally well-written.** The differentiation against VLMBias (line 61) is the cleanest related-work paragraph in the paper — the `(cue source, 측정 대상)` two-axis framing makes the contribution clear without overclaim. The activation-steering paragraph (line 65) is honest about prior CAA / ITI / LEACE — Round-1 novelty review must have driven this rigor. **Minor**: line 63 *"본 논문 E6는 이와 *상보적*으로 LM residual stream에서 작동한다"* — `*상보적*` is italicised as if newly defined but the term has been used 4× by this point. Drop italics here.

### §3 Method

- **§3.1 stimulus 4-conditions**: clean. Figure 1 reference issue (MIN-W3) and minor table-caption awkwardness (Table 1 caption is one line; could explain the (a − d) / (a − m) / (a-base-wrong − a-base-correct) gap structure).
- **§3.2 standard metric**: Round-1 reviser lifted the `df > 0` near-tautology caveat from §4.1 to §3.2 (lines 99-101). **This works** — the caveat now lives where readers expect to find metric properties. The cross-reference *"본 caveat은 §4.1 본문에서도 다시 cross-reference 된다"* (line 101) is fine.
- **§3.3 datasets and model panel**: the canonical hedge sentence (line 107) works but is verbose — *"Panel scope by analysis axis (canonical hedge — once)."* opening is accurate but reads as meta-instruction to the reader. **Fix**: drop the parenthetical `(canonical hedge — once)`; the `behavioral · mechanism · deployable` 3-register split is the substance.

### §4 Behavioral analysis

- **§4.1**: experiment-first opening (MAJ-W1). After the opening, prose carries but is hedge-heavy: line 117 has the *df > 0* near-tautology re-statement (which §3.2 already carries — duplication), the (i)/(ii)/(iii) load-bearing-evidence list, and the LLaVA-Interleave footer caveat (line 130 — useful, well-placed). Insight 1 (line 140) restates table values (MAJ-W9). Cross-dataset replication block (line 144) is well-placed.
- **§4.2**: digit-pixel causality — strong section. The S1 confound resolution paragraph (line 152) is one of the paper's best — it explains *exactly* why (a−m) controls for the trivial-recovery confound. Insight 2 magnitude/sample-size disclosure (line 185) is good honest hedging post-Round-1 MAJ-5 fix.
- **§4.3**: 5-dataset main matrix — short (3 Insights, single Figure). Insight 1 is restating Figure 4 caption (MAJ-W9). Insight 2 anti-scaling is mechanism-relevant. Insight 3 encoder-family ordering is honest about §5.3 fragility.
- **§4.4**: confidence 6-bin gradient — long opening run-on (MIN-W4, MIN-W5). Insight 3 (line 211) Round-1 reviewer flagged as hand-waving on non-monotonic cells — this is acknowledged in prose (`§8.2의 follow-up 항목으로 deferred`).
- **§4.5**: γ-β reasoning amplification — substantial overlap with §1.4 (MAJ-W7). Insight 1 (line 227) is one of the densest in the paper but actually load-bearing — the binary-projection-breaking argument is a non-trivial mechanism interpretation. Insight 3 (line 231) reads cleanly post-Round-1 patch.
- **§4.6**: γ-β residual-stream bridge — Round-1 rewrote this from 25 lines to 13. **Form is best in the paper.** Claim → setup → evidence → Insight 1 (partial-falsification) + Insight 2 (quantitative deferred). The K=1 vs K=8 partial-falsification is honest and strengthens credibility.

### §5 Mechanism

- **§5.1 Mechanism panel + peak-layer setup**: very short (one paragraph). Acceptable as setup section.
- **§5.2 Single-layer ablation null**: 4 Insights, of which Insight 4 (line 284) is a forward-reference dressed as an Insight (MAJ-W9).
- **§5.3 OneVision dataset-dependent peak**: short, well-placed.
- **§5.4 Routing vs integration framework**: Round-1 reviser successfully reframed this from "predictive theory" to "post-hoc synthesis + §4.6 prospective test" (CRIT-1 fix). The 2-leg form works. Three bold-headed `Framework X` paragraphs (MIN-W9) is minor cosmetic issue.

### §6 Mitigation

- **§6.1 E4 attention pathway**: reasonable. Insight 2 (i)/(ii)/(iii) tripartite non-triviality argument is one of the paper's strongest "mechanism is non-trivial" paragraphs (lines 321-329). **But** uses `발화` metaphor (MAJ-W4) and ends section with a 2-sentence summary of E6 vs E4 deployability that essentially restates §1.3 / §6.6.
- **§6.2 E6 residual-stream subspace projection**:
  - **§6.2.1 method**: clean, includes the (a − m) contrast Insight (line 351) — single best Insight in the paper (per v1-archive); Round-1 retained it. Generalisable design pattern restatement (`*calibration contrast는 인과 통로 (causal pathway) 를 confounding variance로부터 *분리*하는 paired difference여야 한다*`) is genuine paper-level contribution-prose.
  - **§6.2.2 calibration**: Round-1 honest disclosure of cell #17 vs #8 within-1-SE ranking (line 357) is good. Long sentence though.
  - **§6.2.3 5-dataset cross-evaluation**: paragraph-length self-reframe (MAJ-W2) is the structural problem.
- **§6.3 b-arm em explanation**: Insight 1.5 alternative-explanations block (lines 395-399) is a paper-level honest section. Reads as a rigorous methodologist hedge — appropriate.
- **§6.4 single-direction failure**: short, anchored to §5.2 prediction.
- **§6.5 baseline comparison**: Table 8 verdict column awkward (MIN-W1). Insight (line 424) carries the strongest single argument but suffers `*differentiator*` italicising and a long Note paragraph that re-explains CAA / ITI position.
- **§6.6 두 mitigation의 정합**: 5-arrow chain + duplicate Summary (CRIT-W3).

### §7 Capability preservation

Strong section. Table 9 caption (line 438: *"E8 결과 (1.5h H200, no LLM judge). Bold = 통계적 유의성."*) is concise but the `1.5h H200` runtime metadata is a residue from operational tracking — drop it. The Note on benchmark coverage (line 450) added by Round-1 MAJOR-9 fix works. Insight at line 452 is load-bearing.

### §8 Discussion

- **§8.1 종합**: dense single paragraph (line 460) — re-states Abstract themes. Length OK. Includes `이분법` (MAJ-W3).
- **§8.2 한계**: split into "research scope" + "operational follow-up" registers. v1-archive flagged the mixed-register problem; Round 1 split it. **Works.** The `cross-architecture E6 재calibration` first bullet (line 466) repeats panel-scope hedge (MAJ-W8).
- **§8.3 윤리**: short, OK.
- **§8.4 후속 작업**: 6 numbered items. Round-1 stripped checkmarks + GPU-hour estimates (response Edit MIN-10). One residual: line 489 *"항목 1의 eigenvalue spectrum이 양성이면 spectrum-predicts-cell 형태로 재구조화 가능 (brute-force grid search → spectral prediction)"* — the `(brute-force grid search → spectral prediction)` parenthetical is an operational-debt phrase. Drop or reword.

---

## Table / figure ↔ prose alignment audit

| Table/Figure | Caption claim | Prose claim | Numeric match | Verdict |
|---|---|---|---|---|
| **Table 2** (line 121) | "6-model PlotQA panel, all-base S1 anchor arm (paired-sids intersection over (a-S1, b) per model, n_pair 4,554–4,707)" + bold-marker convention re-stated | §4.1 line 115 *"두 패턴이 즉시 부각된다"* + line 117 *"`df(a)` magnitude는 6개 모델 전반에서 **0.059-0.325 범위**, literal-adoption rate `adopt(a)`는 1.7-15.7 %"* | Table cells: Gemma3-4b adopt 0.157 / Qwen2.5-VL-7b adopt 0.017 → range 0.017-0.157 (paper 1.7-15.7 % — match). df range 0.059-0.325 (Qwen2.5-VL family 0.059, LLaVA-Interleave 0.325) — match | OK |
| **Figure 2** (line 138) | "Confidence-gradient의 coarse binary projection (PlotQA 6-model). df 기준 wrong-base > correct-base on 6/6 모델, gap +19.0-34.4 pp 양수." | §4.1 line 136 *"wrong-base direction-follow rate가 correct-base보다 **+19.0 ~ +34.4 pp 더 크다**"* | matches | OK |
| **Figure 4** (line 193) | "5-dataset 6-model wrong-base S1 direction-follow. df 부호 30/30 cell 모두 양수." | §4.3 line 195 *"df 부호가 5 dataset × 6 model × 30 cell 모두 양"* | matches | OK |
| **Figure 5** (line 205) | "L1 6-bin confidence gradient. ... Worked example PlotQA all-base × LLaVA-OneVision-7b *(Main)* S1 × cross-entropy proxy: df 0.000 → 0.000 → 0.028 → 0.128 → 0.238 → 0.289 (B6−B1 gap +28.9 pp)" | §4.4 line 203 *"5 dataset {TallyQA, ChartQA, MathVista, PlotQA, InfoVQA} × 6 model heterogeneous-coverage panel의 80 anchor cell에서 평균 B6 − B1 gap이 df +0.195 ~ +0.235"* | Figure 5 caption shows *single-cell worked example* (28.9 pp gap), prose body shows *80-cell panel average* (19.5-23.5 pp). Different statistics. Caption explicitly says "Worked example", so this is *technically* not a mismatch — but a casual reader will see "+28.9 pp" in caption and "+19.5-23.5 pp" in text and wonder why they disagree. Caption should add: *"single-cell worked example differs from panel average (text)"* | minor mismatch, fix caption |
| **Figure 6** (line 217) | "γ-β MathVista pair 비교... Correct-base subset df 비율은 ×12.7" | §4.5 prose line 215 + Table 4 line 225: ratio cell `**×12.7**` | matches | OK |
| **Table 4** (line 221) | "γ-β H2 decomposition. Bold = collapse." | Line 223-225 *"qwen3-vl-8b-thinking | **0.327** | **0.267** | **+0.060**"* + ratio row | "Bold = collapse" is opaque — does collapse mean the wrong−correct gap collapsing? **Caption is too terse**. Fix: "Bold = post-Thinking values where wrong − correct gap collapses to <0.1 (df axis)." | minor, fix caption |
| **Table 5** (line 243) | (no caption — just `**Table 5.** L × K sweep within-Thinking 대표 cells (전체 84-cell table은 부록).`) | §4.6 prose — line 254 *"Late-stack (L = 29, 30, 33) K = 1 mean이 positive (+0.21 ~ +0.48), mid-stack (L = 20) K = 1 / 2 / 4 / 8 mean이 negative"* | Table cells: L30 K2 max +0.866, L30 K1 mean +0.477, L29 K1 mean +0.446, L33 K1 mean +0.284, L25 K1 mean +0.213, L20 K1 mean −0.152, L20 K4 mean −0.192, L14 K1 mean −0.041 — matches | OK; caption could add Bonferroni-99.94% header explanation |
| **Table 6** (line 314) | "E4 Phase 2 결과 (88,650 records / 모델, 출처 `docs/insights/E4-mitigation-evidence.md`). Bold = 열 단위 가장 큰 효과." | §6.1 prose line 319-329 | Inline canonical-source citation `(출처 docs/insights/E4-mitigation-evidence.md)` in caption — Round 1 promised to remove these from body prose; this one in a *caption* slipped. Numeric: LLaVA-1.5 −14.6%, ConvLLaVA −9.6% / +1.30 pp — matches prose | inline-source citation in caption — drop; numeric OK |
| **Table 7** (line 365) | "E6 Stage 4-final, paired-sids paired wrong-base deltas with paired-bootstrap 95 % CI (B = 10,000, sid 단위 paired resampling, per-arm denominator/numerator 매 resample 재계산). Bold = 95 % CI excludes 0 in headline direction." | §6.2.3 prose line 383: *"Δdf(a)는 1/5에서 95 % CI excludes 0 (PlotQA n=2,306 [−6.9, −3.4])"* + Δem(b) 5/5 sign-clean | Numeric: TallyQA Δem(b) +13.8 [+12.9, +14.8] (excludes 0, bold); PlotQA Δdf(a) **−5.2 [−6.9, −3.4]** bold. **Average row** at line 372 says Δem(b) **+8.8** — matches prose | OK; caption convention correct |
| **Table 8** (line 415) | "Multi-method 비교. Bold = 본 작업." | §6.5 Insight prose line 424 | One-line caption is too thin (MIN-W10). Verdicts in Korean+English (MIN-W1) | caption thin, mixed register — fix |
| **Table 9** (line 440) | "E8 결과 (1.5h H200, no LLM judge). Bold = 통계적 유의성." | §7 prose lines 450-452 | `1.5h H200` runtime metadata is operational-tracking residue; drop. Numeric: HallusionBench +2.21 [+1.14, +3.28] / POPE −0.06 [−0.21, +0.09] — matches | drop runtime; numeric OK |
| **Figure 1** (line 86) | "4-조건 자극의 효과 예시. anchor / masked / neutral 조건에서 base 대비 accuracy drop을 모델별로 비교; anchor가 가장 큰 drop, masked와 neutral은 1-2 pp 안에서 구별 불가." | §3.1 weak in-text reference (MIN-W3) | matches | in-text reference issue only |
| **Figure A1** (line 576) | "ChartQA 거리 cutoff 검증. 상대 cutoff `max(1, 0.10·gt)`이 S5에서 adopt ≤ 0.05를 만족." | §A.3 line 573 (table row) | matches | OK |

**Spot-check overall:** numeric alignment is solid post-Round-1. Caption issues are: (i) Table 4 "Bold = collapse" too terse; (ii) Figure 5 worked-example vs panel-mean gap not flagged; (iii) Table 6 inline source citation residue; (iv) Table 8 caption too thin; (v) Table 9 runtime metadata residue. None of these block paper acceptance — all are 1-line fixes.

---

## Terminology consistency audit

| Term variant 1 | Term variant 2 | Term variant 3 | Locations | Suggested canonical |
|---|---|---|---|---|
| 끌림 | anchor pull | pull | line 11 (Abstract `graded pull`); line 13 (Abstract `anchor pull`); line 117 (§4.1 `graded movement`); line 140 (§4.1 Insight 1 `끌림 크기`); line 183 (§4.2 `끌림 강도`); line 193 (Figure 4 `가장 큰 끌림`); line 387 (§6.2.3 Insight 1 `큰 끌림 → 큰 감소`) | `anchor pull` (English, Abstract canonical) — drop `끌림` from body prose, keep only in figure captions if needed |
| 이분법 | wrong-base / correct-base 분할 | binary projection / binary stratification | line 11 / line 31 / line 460 (`이분법`); line 144 / line 211 (no `이분법`, uses `wrong-correct asymmetry` or `B1+B2+B3 vs B4+B5+B6`) | `wrong-base / correct-base 분할` or `binary stratification`; eliminate `이분법` (philosophy term) |
| 발화 (hook fires) | hook 트리거 / hook 동작 / forward에서 작동 | — | line 325 (§6.1 `forward pass에서 발화`); line 329 (§6.1 `cross-arm 모두 발화`) | `forward에서 작동` (drop `발화`) |
| free-lunch | 4-clause free-lunch | strict-free-lunch | STRICT_FREE_LUNCH | line 13 / 51 / 422 / 424 / 436 / 460 / 470 (`free-lunch`); line 411 / 424 (`4-clause free-lunch`); line 383 / 450 (`strict-free-lunch`); line 448 (`STRICT_FREE_LUNCH` table cell uppercase) | `4-clause free-lunch` (matches §6.2.3 formal definition); replace `STRICT_FREE_LUNCH` cell with `4-clause free-lunch` or `OK (4-clause)`; eliminate `strict-free-lunch` |
| 회귀 (revert) | 회귀 (regression-toward-anchor) | 역행 (backfire) | line 11 (`generic distractor 수준으로 *되돌아간다*` — Round-1 fixed!); line 30 (`returns ... 되돌아간다`); abstract (`수준으로 *되돌아간다*`); §6.4 line 405 (`+56 % 증가` not `회귀`); §C uses both | Round 1 mostly fixed this — `되돌아간다` for revert, `+56 % 역행` (backfire) preserved. Verify §C usage |
| 사후 synthesis | post-hoc synthesis | retrospective synthesis | line 13 / 35 / 49 / 292 / 294 / 298 / 460 (`사후 synthesis`); §5.4 prose mixes `사후` (Korean) and `post-hoc` (English in same paragraph) | `사후 synthesis` (Korean canonical, mirrors `prospectively` English where it's a single technical term) |
| robustness | 강건 / 강건성 | robust | line 13 (`robustness`); line 130 (Table 2 footer `robust`); line 140 (§4.1 Insight 1 `가장 robust`); §C.1 (`robust`); §4.4 (`robust`) | English `robust` / `robustness` consistent — already mostly converged |
| `*existence proof*` / first-evidence / VLM 최초의 ... 결과 / N=1 architecture × N=1 dataset existence proof / hypothesis-generating existence proof | — | — | line 13 (Abstract `*Existence-proof* on N=1 architecture pair × N=1 dataset`); line 39 (`first-evidence cross-arm 결과`); line 53 (§1.5 auxiliary `N=1 architecture × N=1 dataset *existence proof*`); line 229 (§4.5 Insight 2 `*first-evidence*`); line 467 (§8.2 `first-evidence 결과`); line 460 (§8.1 `*existence proof*`) | `existence proof (N=1 architecture × N=1 dataset)` canonical; drop `first-evidence` synonym entirely |
| ×12.7 | point estimate ×12.7 | ×12.7 ratio | ≥ 5× directional | line 13 / 31 / 39 / 53 / 225 / 460 (5 sites for 1 number; multiple framings) | First mention (Abstract): full hedge; subsequent: `×12.7 ratio (§4.5)` only |
| Insight (English) | 인사이트 (Korean) | — | §4 / §5 / §6 use **Insight 1/2/3** (English) consistently; §4 / §5 / §6 sub-section titles say `인사이트` (Korean) — *e.g.* §4 title `행동 분석과 인사이트`, §5 title `메커니즘과 인사이트`, §6 title `Mitigation과 인사이트` | English `Insight` for the box; Korean `인사이트` only in section titles | OK as-is, but minor inconsistency reader will notice |
| baseline / `target_only` / b-arm / `pred_b` / non-anchored arm | — | — | scattered: `baseline` (Abstract); `target_only` (§3.1 / §6.3); `b-arm` (§6.2.1 / §6.3); `pred_b` (§3.2); `non-anchored arm` (§6.2.3) | `target_only` for the *condition*; `b-arm` for the *evaluation arm*; `non-anchored arm` only when contrasting `anchored arm` (a-arm) — convention seems intended but mixes |

---

## Korean academic-tone audit

### Run-on sentences (>120-word, single-sentence, paragraph-internal)

- **Line 11 (Abstract).** *"본 논문은 vision-language model (VLM)이 ... 6개 open-weight VLM에서 일관되게 재현한다 (legacy VQAv2 reference panel §C.1은 7-model)."* Two-sentence span, ~280 words combined. The Abstract has 7-8 such span structures (CRIT-W2).
- **Line 105 (§3.3).** *"5개 1차 numeric VQA dataset: TallyQA ..., ..., InfographicVQA (val numeric, n = 1,147). 위 raw n은 stratification · eligibility 필터 *이전* count이며, 실제 본문 표에 사용된 per-cell n은 stratified 부분집합 기준으로 ChartQA 129–517 / TallyQA 6,934–14,772 / PlotQA 926–4,610 / InfoVQA 218–865 / MathVista 127–274 *범위에 분포*한다 (모델별 변동; 자세한 per-cell n 표 부록 §A.5 reproducibility)."* 110+ words across two sentences with 5 dataset n-ranges in nested form.
- **Line 132 (§4.1).** *"PlotQA single-dataset depth panel의 역할. PlotQA는 본 paper main matrix의 GT-range 가장 넓은 dataset (정수 [1, 10000], 5-stratum sampled n=5,000)이며, §4.3 main matrix의 5-dataset breadth와 *상보적*인 *single-dataset depth* axis로 사용된다 — Phase-A H2 binary projection (§4.1 본 절) / (a − m) digit-pixel gap (§4.2 Table 3) / L1 6-bin gradient (§4.4) 의 *replication depth*가 PlotQA cell에서 selectively 가장 강하다."* Bold-headed paragraph with 90+ word follow-on; reads as if explaining *to a reviewer* why PlotQA was chosen as panel — meta-justification rather than results-prose.
- **Line 203 (§4.4).** Already cited (MIN-W4); 170+ word single sentence.
- **Line 357 (§6.2.2).** *"(L*, K, α) triple은 27-cell pilot grid (L ∈ {25, 26, 27} × K ∈ {2, 4, 8} × α ∈ {0.5, 1.0, 2.0})에서 선택. **선택 규칙은 calibration set 위에서 사전 (ex ante) 고정**: ... **선택 cell: L* = 26, K = 8, α = 1.0** (27-cell 중 cell #17, §A.5). 잔존 27 candidate cell의 ... heatmap은 §A.5에 함께 surface한다 — 본 grid 위에서 *어느 cell도 −6 pp 임계값을 위반하지 않아* deal-breaker 절은 non-binding이며, 결합 |Δdf(a)| 정렬에서 chosen cell #17이 mean Δdf(a) = −4.4 pp로 1위 (2위 #8 −3.2 pp 대비 1.2 pp 격차 — calib n = 250 위 paired SE ~1.3 pp로 within ~1 SE 범위, ranking는 동일 ex ante 규칙 재실행 시 동일하게 산출되나 첫 SE 안에서 #17 ↔ #8 ordering 교체 가능성은 honest disclosure)."* 200+ words across 4 sentences with 3 nested parentheticals.

### Hedge density (parenthetical clauses per sentence)

The Abstract averages ~2 parentheticals per sentence (74 sentence-marks, ~150 parenthetical openings counted). Body sections range 0.5-1.0 per sentence — within Korean academic norm. **The Abstract is the outlier**; CRIT-W2 fix should reduce parenthetical density by ~70%.

### Italics-mid-Korean overuse

The paper uses italics for: (i) technical-term first-introduction (correct), (ii) emphasis on a Korean-language clause (variable), (iii) marking English-loanword inside Korean (inconsistent), (iv) marking *figures of speech* like *"되돌아간다"* / *"깨진다"* (overused).

Italic-occurrence count exceeds 200 in the body. Sample cluster lines 35-39: 8 italic phrases in 5 sentences. The italic typography is becoming noise rather than signal — when everything is italic-emphasised, nothing is.

**Fix**: limit italics to (i) first-mention of a defined technical term, and (ii) literal English loan-word inside Korean clause. Drop emphasis-italics on Korean phrases.

---

## Title and abstract pacing

### Title

Line 1 (English): *"Cross-Modal Numerical Anchoring in Vision-Language Models: Uncertainty, Plausibility, and Digit-Pixel Gates with a Deployable Subspace-Projection Mitigation"*

Length: 19 words. Conference convention: ≤ 12 words preferred, ≤ 16 words tolerable. **Slightly long.** The colon-separated form is correct; the second clause carries 11 of the 19 words. **Minor fix possible**: drop "Plausibility" (it is a *condition* per §1.2, not a co-equal pillar — the three pillars are graded / digit-pixel / confidence-modulated; plausibility is a window for the first pillar) → "Cross-Modal Numerical Anchoring in Vision-Language Models: Uncertainty, Digit-Pixel Gates, and a Deployable Subspace-Projection Mitigation" (16 words). Or: shorter — "Cross-Modal Numerical Anchoring in Vision-Language Models: Mechanism and Mitigation" (10 words).

Line 3 (Korean): *"시각언어모델의 cross-modal numerical anchoring — 불확실성·plausibility·digit-pixel gate와 배포 가능한 subspace projection mitigation"*. **OK.**

Line 5 (sub-line metadata): *"EMNLP 2026 long paper, 중간 점검용 한글본 (2026-05-09)"* — **CRIT-W1**: delete entirely.

### Abstract

3 paragraphs, 680 words, 74 sentence-marks (which over-counts because nested-clause periods inflate). EMNLP / NeurIPS norm: 1 paragraph, ≤ 250 words, ~10 sentences.

**Detailed targeting for the Abstract rewrite (CRIT-W2):**

1. **Paragraph 1 (current ~180 words)** — phenomenon + magnitudes. Compress to 60-80 words: "Cross-modal numerical anchoring is replicated in 6 open-weight VLMs: literal-anchor adoption is 1.7-15.7 % but direction-follow rises monotonically with base-prediction entropy (B6−B1 6-bin gap +19.5-23.5 pp on 80 anchor cells). The effect is gated by digit-pixel causality: inpainting digit pixels alone reverts adoption to generic-distractor floor."
2. **Paragraph 2 (current ~280 words)** — mechanism + framework + mitigation chain. Compress to ~100 words: "Single-layer ablation is null (5/5 mech-panel models); the signal is multi-layer redundant. We synthesise this with §6.4 LEACE rank-1 cross-dataset failure as a *post-hoc* routing-vs-integration framework, and verify the layer-routing prediction *prospectively* on Qwen3-VL γ-β (sign-reversal mid- vs late-stack at K=1, 14/84 Bonferroni-clean cells; framework's universal-K=8 assumption partially falsified). E6 multi-direction subspace projection (L=26, K=8) trained on PlotQA+InfoVQA pooled (N=5,000) reduces Δdf 5/5 datasets (PlotQA CI-clean) and raises both em arms."
3. **Paragraph 3 (current ~220 words)** — capability + reasoning amplification. Compress to ~50 words: "6-benchmark capability preservation: macro Δ +0.41 pp; HallusionBench +2.21 pp [+1.14, +3.28] excludes zero. **Auxiliary observation**: Qwen3-VL-Thinking amplifies anchor pull on the same continuous-confidence axis (correct-base df ratio ×12.7, N=1×N=1 existence-proof). Mitigation chain is single-model case study; cross-architecture replication §8.2."

Result: ~210-230 words, single paragraph (or 1 main + 1 short auxiliary). Within venue norm.

---

## Outstanding questions for reviser

1. **CRIT-W3 §6.6 5-arrow chain**: do you want §6.6 deleted entirely (the section is short and its content is largely redundant with §1.3 / Abstract / §6.2) or compressed to 1-2 sentences? The chain itself is unfixable as prose — it must be either deleted or replaced.

2. **CRIT-W2 Abstract**: are you committed to retaining the legacy-VQAv2 §C.1 cross-reference, the 5-baseline panel enumeration, and the case-study scope hedge in the Abstract? If yes, target word-count is ~350 not 250. If no, target ~210-230.

3. **MAJ-W1 §4.1 opening**: is the experiment-first opening *deliberate* (i.e. you want depth-panel framing first) or unintentional? §4.6's claim-first form is the model the rest of §4 should follow — but §4.1's depth-panel function may justify preserving the experiment-first opening.

4. **MAJ-W2 §6.2.3 self-reframe**: do you accept compressing the 600-word inline reframe to a 2-3 sentence headline + table-caption convention? The honest content of the reframe is *"Δem(b) is multiplicity-robust; Δdf is sample-size-bound"* — that's 1 sentence. The supporting per-dataset detail belongs in Table 7 caption + §A.5, not in §6.2.3 prose.

5. **MAJ-W7 §1.4 vs §4.5 redundancy**: the Round-1 reviser response claims §1.5 demoted γ-β to "auxiliary observation". §1.4 still exists as its own sub-section — is the §1.4 sub-section *itself* the auxiliary observation, or is §1.4 an independent intro-summary? If the former, drop §1.4 entirely (already in §1.5(aux) + §4.5). If the latter, §1.4 should be ~2 sentences not ~13 lines.

6. **MAJ-W3 / MAJ-W4 / MIN-W2 Korean register fixes (`이분법`, `발화`, `끌림`)**: v1-archive Round-2 review flagged these; the v3 changelog (line 9) said they were addressed. The current draft has 3 surviving `이분법` / 2 `발화` / 8 `끌림`. Was the v3 fix only partial, or were there subsequent edits that re-introduced these terms? The Round-1 reviser response did not list these in its edit log.

7. **MAJ-W6 free-lunch term**: the 4-variant inconsistency is auditable; will you canonicalise to `4-clause free-lunch` (matches the §6.2.3 formal definition)? In particular Table 9's `STRICT_FREE_LUNCH` cell value is a code-identifier from `headline-numbers.md` — do you want it preserved as a literal status-code or translated?

---

**Final note**: the paper's *content* is publishable at top-tier; the writing axis is one editing pass behind. Round 1 did 70 % of the structural surgery; Round 2 needs to land the Abstract compression, §6.6 chain removal, §4.1-style claim-first openings, and the four leftover Korean register terms. None of these is novel work — all are surgical edits a competent reviser can complete in 4-6 hours. Do not let the writing axis carry the methodology axis's reject weight (Round 1 has the harder verdict to deliver).
