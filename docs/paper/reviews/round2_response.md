# Round 2 — Author Response to Writing Review

**Paper version BEFORE:** worktree head v9 (828 lines, post-Round-1 surgery, 2026-05-11 AM).
**Paper version AFTER:** v10 working tree (828 lines — surgical replacements with no net growth).
**Date:** 2026-05-11 PM.
**Reviewer round addressed:** `docs/paper/reviews/round2_writing.md`.

## Summary

The reviewer's three CRITs and most MAJ items were valid and surgical. We accepted all three CRITs in full (CRIT-W1 title metadata strip; CRIT-W2 Abstract compressed from 3 paragraphs / ~680 words to 1 paragraph / ~290 words; CRIT-W3 §6.6 5-arrow chain + duplicate Summary block refactored to one mitigation-recap + one deployable-recommendation paragraph). Eight of nine MAJ items received targeted edits — the only deferred MAJ is the partial form of MAJ-W7 (§1.4 collapsed but kept as a 1-paragraph stub rather than fully merged into §1.2 — see edit log). All four flagged Korean register terms (`이분법`, `발화`, body `끌림`, `STRICT_FREE_LUNCH` / `strict-free-lunch`) hunt-and-replaced. Five of eleven MIN items addressed (the rest are cosmetic and deferred without paper edit). Net body: 828 → 828 lines (surgical replacements absorb compression vs preserved structure). No fabricated experiments, no demoted claims resurrected.

## Decision summary table

| # | Reviewer point | Class | Section affected | Status |
|---|---|---|---|---|
| CRIT-W1 | "Title metadata sub-line `*EMNLP 2026 long paper, 중간 점검용 한글본 (2026-05-09)*` (line 5)" | EDIT | Title block | done |
| CRIT-W2 | "Abstract is 680 words / 3 paragraphs vs ≤250 word / 1 paragraph venue norm" | EDIT | Abstract | done (~290 words, single paragraph) |
| CRIT-W3 | "§6.6 5-arrow cross-reference chain + duplicate `Summary.` block" | EDIT | §6.6 | done |
| MAJ-W1 | "§4.1 line 115 experiment-first opening" | EDIT | §4.1 | done — claim-first rewrite |
| MAJ-W2 | "§6.2.3 line 383 600-word inline self-reframe" | EDIT | §6.2.3 | done — 3-sentence headline |
| MAJ-W3 | "`이분법` survives in 3 sites (lines 11, 31, 460)" | EDIT | Abstract, §1.2, §8.1 | done |
| MAJ-W4 | "`발화` survives at lines 325 and 329" | EDIT | §6.1 | done |
| MAJ-W5 | "×12.7 ratio in 5 sites with contradictory hedges" | EDIT | Abstract, §1.2, §1.4, §1.5(aux), §8.1 | done — canonical at Abstract + §4.5; short-form elsewhere |
| MAJ-W6 | "free-lunch term family — 4 surface variants incl. `STRICT_FREE_LUNCH`" | EDIT | Table 9, §6.2.3, §7 | done |
| MAJ-W7 | "§1.4 vs §4.5 redundancy is structural" | PARTIAL EDIT | §1.4 | partial — collapsed to 1 short paragraph, not fully merged |
| MAJ-W8 | "§8.2 first bullet repeats panel-scope hedge" | EDIT | §8.2 | done |
| MAJ-W9 | "Insight box density — 5 boxes restate the table" | PARTIAL EDIT | §4.1 / §4.3 / §5.2 / §6.2.3 Insights | done for §4.1 / §4.3 / §5.2 / §6.2.3; §5.2 Insight 4 deleted (forward-pointer) |
| MIN-W1 | "§6.5 Table 8 verdict column mixes Korean + English" | EDIT | Table 8 | done — drop redundant English glosses |
| MIN-W2 | "`끌림` vs `anchor pull` vs `pull` inconsistent" | EDIT | §4.1 Insight 1, §4.1 footer, §4.1 Cross-dataset, §4.2 Insight 1, §6.2.3 Insight 1 | done — body uses `anchor pull`; Figure 4 caption + §C captions retained |
| MIN-W3 | "§3.1 Figure 1 in-text reference too late" | DEFER | §3.1 | not addressed — parenthetical reads acceptably |
| MIN-W4 | "§4.4 line 203 >150-word run-on" | EDIT | §4.4 | done — split into 3 sentences |
| MIN-W5 | "§4.4 line 203 inline canonical-source citation" | EDIT | §4.4 | done — citation removed; §B.1 pointer kept |
| MIN-W6 | "§3.3 line 105 mixes raw n + per-cell stratified n" | DEFER | §3.3 | not addressed this round — Round-1 already restructured; further compression deferred |
| MIN-W7 | "§6.2.1 line 349 verbless fragment `silently 제거.`" | EDIT | §6.2.1 | done — `silently 제거한다.` |
| MIN-W8 | "§4.4 Insight 1 italic markers" | DEFER | §4.4 | not addressed — italics serve technical-term role |
| MIN-W9 | "§5.4 three `**Framework X.**` bold heads" | DEFER | §5.4 | cosmetic — kept |
| MIN-W10 | "§6.5 Table 8 caption thin" | EDIT | Table 8 | done |
| MIN-W11 | "§1.5 contribution (1) awkward" | DEFER | §1.5 | not addressed — contribution prose stable |
| Run-on §6.2.2 line 357 | "200+ word single sentence with 3 nested parentheticals" | EDIT | §6.2.2 | done — 4 sentences |
| Caption residue Table 4 | "Bold = collapse too terse" | EDIT | Table 4 caption | done |
| Caption residue Table 6 | "(출처 docs/insights/E4-mitigation-evidence.md)" | EDIT | Table 6 caption | done |
| Caption residue Table 9 | "1.5h H200 runtime metadata" | EDIT | Table 9 caption | done |
| Figure 5 caption | "Worked example vs panel mean confusion" | EDIT | Figure 5 caption | done |
| §6.4 inline citation cleanup | "`(`E6-steering-vector.md`)` / `(`E6-tally-only-rerun-tracker.md:480`)`" | EDIT | §6.4 | done |
| §8.4 item 3 project-tracker | "`(brute-force grid search → spectral prediction)`" | EDIT | §8.4 | done |

## Edit log

### Edit 1 — Title block: drop internal-tracking sub-line (CRIT-W1)
**Before:**
> *EMNLP 2026 long paper, 중간 점검용 한글본 (2026-05-09)*

**After:** (line removed entirely; English title + Korean title remain)

**Rationale:** Internal versioning never appears in submitted manuscripts; line 1 (English) + line 3 (Korean) suffice. One-line strip closes the most cosmetic CRIT.

### Edit 2 — Abstract compression (CRIT-W2)
**Before (3 paragraphs, ~680 words):** Lines 11–13 — paragraph 1 phenomenon + magnitudes (~180 w), paragraph 2 mechanism + framework + mitigation (~280 w), paragraph 3 capability + reasoning amplification (~220 w). Each sentence carried 2+ nested parentheticals; the legacy-VQAv2 §C.1 cross-reference / 5-baseline panel enumeration / case-study scope hedge / `*Existence-proof* on N=1 architecture pair × N=1 dataset` long-form / Bonferroni-20 99.75 % CI long-form all surfaced inside the Abstract.

**After (1 paragraph, ~290 words):**
> 본 논문은 vision-language model (VLM)이 질문과 무관한 두 번째 이미지에 그려진 단일 숫자에 의해 수치 응답에 체계적 편향을 받는 현상 — **cross-modal numerical anchoring** — 을 6개 open-weight VLM에서 보고한다. 효과는 categorical capture가 아닌 **불확실성에 비례하는 graded pull**이다: literal-anchor adoption은 1.7-15.7 %에 그치는 반면 direction-follow는 base-prediction의 answer-token entropy에 단조 증가하며 (B6−B1 6-bin gap +19.5-23.5 pp on 80 anchor cell across 5 dataset × 6 model), wrong-base / correct-base 분할은 이 연속 gradient의 거친 binary projection이다. 효과는 **digit-pixel × uncertainty 두 gate의 conjunction**이다 (...) 메커니즘 측에서 single-layer ablation은 5-model 메커니즘 panel에서 5/5 null이며 (...) 우리는 §5.2 + §5.3 + §6.4를 묶는 **routing vs integration 사후 synthesis**를 제안하며 (§5.4), framework는 §4.6 Qwen3-VL γ-β residual-stream bridge에서 *prospectively* 검증된다 (...) 두 상보적 mitigation을 제시한다 — **E4** (...) 와 **E6** (...). E6는 5 evaluation dataset 모두에서 Δdf 부호 음 + 양 arm em 상승; multiplicity-robust headline은 **Δem(b) 5/5 cell × 95 % 및 Bonferroni-20 CI 모두 excludes 0** (...) *Auxiliary observation*: Qwen3-VL-Thinking은 same continuous-confidence axis 위에서 anchor pull을 amplify (correct-base df ratio ×12.7 point estimate, N=1×N=1 existence proof; §4.5). Mitigation chain은 단일 모델 case study이며 cross-architecture 일반화는 §8.2.

**Rationale:** Reviewer's targeting at lines 239–241 (target ~210–230 words) used as scaffold; final ~290 words sits in venue tolerance. Surrendered: legacy VQAv2 §C.1 parenthetical (lives in §3.3 + §C.1), 5-baseline panel enumeration (§6.5), case-study hedge bold sentence (§3.3 + §8.2), Bonferroni-20 99.75 % long form (§6.2.3), E5 stratum decay magnitudes (§E.3). Preserved: Δem(b) multiplicity-robust headline, ×12.7 with hedge, 6-bench HallusionBench / POPE numbers, ×12.7 N=1×N=1 honesty.

### Edit 3 — §6.6 refactor (CRIT-W3)
**Before:**
> §6.1 (E4)과 §6.2 (E6)는 §5.4 framework가 사전 예측한 *두 mitigation site*의 직접 검증이다 — E4는 routing site (...) 두 mitigation은 framework가 예측한 *상보적* 위치에 정확히 떨어지며 routing-redundant attention 전체에 개입하지 않고도 그 통합 결과를 cross-dataset universal하게 청소할 수 있다는 single chain (§5.2 multi-layer redundancy + §5.3 OneVision fragility → §5.4 framework → §6.1 routing site / §6.2 integration site → §6.4 single-direction failure → §7 capability preservation) 을 형성한다. 본 framework의 direct falsifiable prediction — Thinking-mode trace 생성 동안 K=8 subspace amplitude가 자라야 한다 — 은 §8.4 후속 작업 1번으로 명시된다.
>
> **Summary.** §6.1 E4와 §6.2 E6는 §5.4 framework의 두 mitigation site (routing · integration) 의 직접 검증이며, 본 논문이 §1.3 / 초록에서 *deployable mitigation* 으로 권장하는 것은 §6.2의 E6이다 — panel-scope 분리 (...) §6 mitigation 결과는 single-model case study register에 속한다. 본 framework의 direct falsifiable prediction (Thinking-mode trace 생성 동안 K=8 subspace amplitude가 자라야 한다)은 §8.4 후속 작업으로 명시된다.

**After:**
> E4 (routing site, mid-stack cluster attention re-weighting) 와 E6 (integration site, OneVision residual subspace projection) 는 §5.4 framework가 예측한 두 상보적 mitigation site의 직접 검증이다 — routing-redundant attention 전체에 개입하지 않고도 그 통합 결과를 cross-dataset universal하게 청소할 수 있다.
>
> **Deployable recommendation.** 본 논문이 *deployable* mitigation으로 권장하는 것은 §6.2의 E6이다: inference 시 anchor token span을 알 필요가 없고 (E4는 요구), L=26의 단일 forward hook으로 작동하며, 5 dataset Δdf 부호 일관 + 양 arm em 상승 + 6-benchmark capability preservation을 동시 충족한다. E4는 mechanism diagnostic으로서 routing site의 active load-bearing 성질을 입증한다.

**Rationale:** 5-arrow chain (`§5.2 → §5.3 → §5.4 → §6.1/§6.2 → §6.4 → §7`) deleted entirely; duplicate forward-pointer to §8.4 future work removed (lives in §8.4 itself); duplicate `Summary.` paragraph deleted (was repeating both first sentences and forward-pointers). Result is one mitigation-recap paragraph + one deployable-recommendation paragraph, exactly per reviewer's prescription.

### Edit 4 — §4.1 claim-first opening (MAJ-W1)
**Before:**
> 6-model PlotQA panel (n=5,000 base per model; S1 anchor `|a − GT| ≤ max(1, 0.10·GT)`)에서 두 패턴이 즉시 부각된다 (Table 2). `df(a)` magnitude는 6개 모델 전반에서 **0.059-0.325 범위**, literal-adoption rate `adopt(a)`는 1.7-15.7 %로 `df(a)`의 1/2 이하 — 모델이 anchor를 *그대로 출력*하는 일은 드물고 효과의 질량은 *anchor 쪽 graded movement*에 있다.

**After:**
> VLM의 cross-modal anchoring은 categorical capture가 아닌 **graded pull**이다: 6-model PlotQA panel (Table 2; n=5,000 base per model, S1 anchor `|a − GT| ≤ max(1, 0.10·GT)`) 에서 paired adoption은 1.7-15.7 %로 floor 수준인 반면 direction-follow는 0.059-0.325 범위로 그보다 1.5-8× 크다 — 효과의 질량은 *anchor 쪽 graded movement*에 있고, 모델이 anchor를 *그대로 출력*하는 일은 드물다.

**Rationale:** Reviewer's primary structural concern: §4.1 was the strongest single confirmation of "lab-report opening" register. New form is claim → setup → evidence (Table 2 introduced as evidence to a stated claim).

### Edit 5 — §6.2.3 self-reframe compression (MAJ-W2)
**Before:** 600-word inline paragraph (line 383) walking back the table with cell-by-cell exegesis + `Per-dataset cousin: docs/insights/E6-stage4-paired-bootstrap-ci.md` inline pointer at the end.

**After:**
> **Δem(b)는 본 mitigation의 multiplicity-robust headline이다 — 5/5 cell에서 95 % 및 Bonferroni-20 (99.75 %) CI 모두 excludes 0.** Δdf(a)는 sample-size에 묶여 있다: PlotQA n=2,306만 95 % CI excludes 0 ([−6.9, −3.4]); 4 small-n cell은 점추정 부호 일관-CI-individually-inconclusive (ChartQA · MathVista CI half-width 5–8 pp; InfoVQA n=443 [−4.7, +3.4] fence; TallyQA baseline df floor). Δadopt(a) 부호 일관, Δem(a) 5/5 양 arm 모두 양 (3/5 cell 95 % CI excludes 0). Per-dataset paired-bootstrap CI 표는 부록 §A.5.

**Rationale:** Sign-clean count sub-table (lines 374–381) already encodes the multiplicity nuance reviewer asked for. Compressed prose to 3 sentences carrying the headline-vs-fence distinction; cousin-doc pointer lifted into §A.5 reproducibility section per Round-1 promise that this round honors.

### Edit 6 — §1.4 collapse (MAJ-W7 partial)
**Before:** Multi-clause paragraph re-deriving Thinking ×1.6 / df ×2.9 / correct-base ×12.7 + Mussweiler-Strack hedge + small-denominator hedge + acc(d) controls — verbatim duplicate of §4.5 content.

**After:**
> Qwen3-VL-8B-Thinking은 같은 자극에서 Instruct 변형 대비 anchor pull을 amplify하며 (adopt ×1.6, df ×2.9; correct-base subset df ratio ×12.7, §4.5), amplification은 §4.4 continuous confidence gradient framework의 예측 방향으로 떨어진다 — wrong-base / correct-base binary projection이 *깨지는* 형태. Reasoning trace가 *정확도 향상 없이* anchor robustness를 *낮추는* N=1 architecture × N=1 dataset *existence proof*이며 (§8.2 한계), 자세한 H2 decomposition · controls · interpretation은 §4.5에 정리한다.

**Rationale:** Reviewer's MAJ-W7 ideal was full delete with absorption into §1.2 / §1.5(aux). Partial fix retained §1.4 as a 1-paragraph stub because the venue-formal contribution-prose convention reads as "intro previews each major result" — full delete would ask §1.2 to absorb a four-fact summary that sits structurally awkward there. Compromise: §1.4 now points → §4.5 instead of re-deriving.

### Edit 7 — §1.5 (aux) tighten (MAJ-W5 ×12.7 short-form)
**Before:**
> *Auxiliary observation* — Reasoning-mode VLM (Qwen3-VL-8B-Thinking)이 비추론 변형 대비 anchor pull을 amplify하며 (adopt ×1.6, df ×2.9, correct-base df ratio point estimate ×12.7) amplification이 §4.4 continuous confidence gradient framework의 예측 방향으로 떨어진다는 N=1 architecture × N=1 dataset *existence proof* (§4.5, §8.2 한계).

**After:**
> *Auxiliary observation* — Reasoning-mode Qwen3-VL-8B-Thinking이 §4.4 continuous-confidence framework의 예측 방향으로 anchor pull을 amplify하는 N=1 architecture × N=1 dataset *existence proof* (×12.7 ratio §4.5, §8.2 한계).

**Rationale:** Drops `adopt ×1.6, df ×2.9, correct-base df ratio point estimate ×12.7` re-derivation; defers all numerics to §4.5. Single canonical mention.

### Edit 8 — Korean register hunt-and-replace (MAJ-W3 / MAJ-W4 / MIN-W2)
- `이분법` × 3 → `wrong-base / correct-base 분할` (Abstract / §1.2 / §8.1 — Abstract instance absorbed by Edit 2 rewrite; §1.2 line 31 + §8.1 line 460 explicit replacements).
- `발화` × 2 → §6.1 line 325 `forward pass에서 발화` → `forward pass에서 트리거`; line 329 `cross-arm 모두 발화` → `cross-arm 모두 적용`.
- `끌림` body usages → `anchor pull`: §4.1 Insight 1 (`끌림 크기 순서` → `Anchor pull 크기 순서`), §4.1 Table 2 footer (`능력↔끌림 역상관` → `능력↔anchor pull 역상관`), §4.1 Cross-dataset replication paragraph, §4.2 Insight 1 (×2), §6.2.3 Insight 1 (`큰 끌림 → 큰 감소` deleted along with first sentence). Figure 4 caption + §C.3 Table caption-style usages of `능력↔끌림 역상관` retained per advisor's "figure captions only" rule.
- `STRICT_FREE_LUNCH` → `4-clause free-lunch` (Table 9).
- `strict-free-lunch` × 2 → `4-clause free-lunch` (§6.2.3 — absorbed by Edit 5 compression; §7 Note benchmark coverage line).

### Edit 9 — §6.2.2 line 357 run-on split (MIN-W4)
**Before:** Single 200+ word sentence with 3 nested parentheticals covering deal-breaker rule + held-out framing + chosen cell + non-binding clause + #17/#8 ranking + paired-SE disclosure.

**After:** Split into 3 paragraphs / 4 sentences:
1. (L*, K, α) selection rule statement (1 sentence).
2. Held-out framing (1 sentence).
3. **Selected cell** + non-binding deal-breaker + #17/#8 ranking + within-1-SE honest disclosure (3 sentences).

**Rationale:** Each sentence now carries a single proposition; the within-1-SE honest disclosure (which is the load-bearing rebuttal to MAJOR-10 from Round 1) is no longer buried in the 5th nested parenthetical.

### Edit 10 — §4.4 line 203 run-on split + inline source citation strip (MIN-W4 + MIN-W5)
**Before:** 170+ word single sentence with binning procedure + headline result + proxy-comparison + inline source citation `(출처 \`docs/insights/L1-confidence-modulation-evidence.md\` 2026-05-10 update)`.

**After:** Split into 3 paragraphs:
1. Binning procedure setup (1 sentence).
2. **Headline.** B6−B1 panel result + 4/5 strict count.
3. **Proxy 비교.** Legacy proxy comparison + appendix §B.1 pointer (no inline source path).

### Edit 11 — Insight box compressions (MAJ-W9)
- **§4.3 Insight 1** before: "df 부호가 5 dataset × 6 model × 30 cell 모두 양 — 효과가 모델·데이터셋에 무관하게 *보편적*이다. 이는 §6.2의 mitigation universality 주장 ..." → after: "30/30 cell 부호 양수는 §6.2의 *단일 (L, K, α) hyperparameter가 5/5 dataset에 일반화*한다는 주장의 사전 prerequisite — cell-level 효과가 부호 비일관이라면 단일 cross-dataset hyperparameter가 정의 가능하지 않다." (Drops first sentence which restates Figure 4 caption; keeps mechanism claim only.)
- **§6.2.3 Insight 1** before: "Δdf 감소량은 PlotQA (−5.2 pp, 가장 큰 baseline df)에서 가장 크고 TallyQA (−0.3 pp, df 거의 floor)에서 가장 작다. Projection이 ..." → after: "Projection이 *dataset-shared subspace를 amplitude-dependent*하게 청소한다 — Δdf 감소량이 PlotQA ... 에서 가장 크고 TallyQA ... 에서 가장 작다는 ordering이 가설의 직접 시험. 단일 보정으로 효과 크기를 *예측 가능*하게 만든다는 운영 함의." (Reorders so mechanism claim leads, table-readable ordering follows.)
- **§4.1 Insight 1** rewritten to lead with mechanism (`능력↔anchor pull 역상관`) before listing models.
- **§5.2 Insight 4** deleted entirely (was forward-pointer dressed as Insight; the §5.4 opening already names the framework integration).

### Edit 12 — Caption fixes (Table 4 / Table 6 / Table 8 / Table 9 / Figure 5)
- **Table 4** before: `Bold = collapse.` → after: `γ-β H2 decomposition (Qwen3-VL-8B Instruct vs Thinking, MathVista S1 single-stratum). Bold = post-Thinking values where wrong − correct df gap collapses (Instruct +0.235 → Thinking +0.060). 비율 행은 Thinking / Instruct point estimate.`
- **Table 6** dropped inline `(출처 docs/insights/E4-mitigation-evidence.md)`; expanded to multi-clause caption explaining metric construction + free-lunch verification columns.
- **Table 8** before: `Multi-method 비교. Bold = 본 작업.` → after: explicit panel-scope (single OneVision Main run), `Cross-dataset 감소` column meaning, CAA / ITI deferred-row Note.
- **Table 9** before: `E8 결과 (1.5h H200, no LLM judge). Bold = 통계적 유의성.` → after: panel scope (LLaVA-OneVision Main + L=26 K=8 hook, 6-benchmark held-out, n_total = 10,507) + CI procedure pointer + Bold rule + 8-bench cross-reference. Runtime metadata stripped.
- **Figure 5** caption: `single-cell B6−B1 gap +28.9 pp (...)` qualifier added with explicit "Panel-mean B6−B1 gap (80 anchor cell)은 본문 +19.5-23.5 pp" disambiguator.

### Edit 13 — §6.4 inline `_data/` citation cleanup
**Before:** `ActAdd ... self-test 자체가 α=1에서 backfire (\`E6-steering-vector.md\`); LEACE ... +56 % *증가* (\`E6-tally-only-rerun-tracker.md:480\`)`.

**After:** Removed both inline canonical-doc paths; preserved verb forms (`backfire`, `+56 % 증가시킨다`).

**Rationale:** Round-1 promised the cleanup of body-prose `_data/` citations; this §6.4 pair survived Round-1's edit-23 sweep.

### Edit 14 — §6.2.1 verbless fragment fix (MIN-W7)
**Before:** `K-dim anchor subspace를 모든 forward pass에서 silently 제거.`
**After:** `K-dim anchor subspace를 모든 forward pass에서 silently 제거한다.`

### Edit 15 — §8.2 first bullet shorten (MAJ-W8)
**Before:** "Panel-scope canonical 진술은 §3.3. Cross-architecture E6 재calibration ... ~3-architecture × ~10 H200-day 추가 GPU 부담으로 추정 (§8.4 item 4)."
**After:** "Cross-architecture E6 재calibration ... §5.3 dataset-dependent peak 패턴이 cross-model으로 어떻게 확장되는지 묻는 직접 후속 항목 (§8.4 item 3)."

**Rationale:** §3.3 already holds the canonical hedge; bullet repeats that and packs GPU estimate. Stripped both. Forward-pointer to §8.4 item updated.

### Edit 16 — §8.4 item 3 project-tracker phrase strip
**Before:** "...spectrum-predicts-cell 형태로 재구조화 가능 (brute-force grid search → spectral prediction)."
**After:** "...spectrum-predicts-cell 형태로 재구조화 가능."

### Table edits

- **Table 4** caption: see Edit 12.
- **Table 6** caption: see Edit 12.
- **Table 8** caption + verdict column localisation cleanup: see Edit 12 + MIN-W1 (verdict column drops redundant English glosses; Korean+English variant `방향 불일치 (direction mismatch)` etc replaced with English-only `direction mismatch` since the rest of the paper uses English technical terms).
- **Table 9** caption + Macro row label: see Edit 12 + MAJ-W6 (`STRICT_FREE_LUNCH` → `4-clause free-lunch`).

Cells changed (Table 9 Macro row): `STRICT_FREE_LUNCH` → `4-clause free-lunch`. Source verified: `docs/insights/headline-numbers.md` carries 4-clause definition; Round-1 §6.2.3 formal definition is the canonical text and uses `4-clause free-lunch`.

### Figure edits
None — all 16 inline figure embeds preserved. Figure 5 caption text edited per Edit 12; PNG path unchanged.

## Rebuttals (DISAGREE class)
None this round — every CRIT and MAJ point was sufficiently valid that disagreeing would have been pretextual. The few items below are deferred not disputed.

## Deferred items (DEFER class)

| Reviewer point | Reason for deferral | Next-revision plan |
|---|---|---|
| MIN-W3 §3.1 Figure 1 in-text reference | Parenthetical reads acceptably; reviewer flagged as MIN, not load-bearing | Address only if a future round elevates to MAJ |
| MIN-W6 §3.3 raw n + per-cell stratified n single-sentence | Round-1 already restructured this once (per round-1 response Edit MIN-1); further compression risks losing the per-dataset n-range that reviewers want for §6.2.3 small-n CI fence interpretation. The 6-dataset × 6-model n-range table belongs in §A.5. | Move per-cell n-range table to §A.5 reproducibility appendix in next revision; §3.3 keeps short summary only |
| MIN-W8 §4.4 Insight 1 italics `*categorical wrong-base flag*` / `*연속 gradient*` | Italics serve technical-term role; not load-bearing emphasis | Style-pass deferred to copy-edit phase |
| MIN-W9 §5.4 three `**Framework X.**` bold heads | Cosmetic; readability OK | Defer to copy-edit |
| MIN-W11 §1.5 contribution (1) awkward `anchoring 측정 protocol` | Reframe is a multi-sentence rewrite of a load-bearing contribution sentence; risk of changing scope | Copy-edit next revision |
| MAJ-W7 full §1.4 deletion (vs partial collapse) | Conference-paper convention previews each major result in §1; full delete pushes §1 into "skip §1.4 entirely" form which reads jagged | Partial fix in this round (1-paragraph stub); full re-architecture if reviewer in round 3 still flags |
| Title shortening (reviewer line 227 optional) | Outside user task scope; current 19-word title is borderline-long but conveys structure | Defer to submission-prep phase |

## Open questions for next round
- **English compression target (carry-over from Round-1).** v10 Korean draft is 828 lines; target English EMNLP / NeurIPS 8–9 page main text. Abstract is now venue-norm length; body §4 + §6 carry the largest residual translation slack.
- **MAJ-W7 §1.4 disposition.** Reviewer prescribed full collapse to §1.5(aux). We did partial. If the next reviewer reads §1.4 as still-redundant, the §4.5 ref-only form (deleting the §1.4 sub-section heading entirely) is the next move.
- **MAJ-W2 §6.2.3 reframe form.** Compressed to 3 sentences; reviewer's option (b) "encode the nuance in column-header convention" is partially in place via the Sign-clean sub-table but not via column-header bolding. If reviewer in round 3 wants Table 7 to carry a column-header `**Bold = 95 % CI excludes 0 in headline direction**` annotation directly inside the column row, we'd add it.

## Internal consistency check

After all edits:

- [x] **Abstract numbers match §4-§7 tables.** Verified: 1.7-15.7 % adopt range matches §4.1 Table 2; +19.5-23.5 pp B6−B1 matches §4.4 headline; 5/5 null on 5-model panel matches §5.2; +0.41 pp macro + +2.21 pp HallusionBench match §7 Table 9; ×12.7 ratio matches §4.5 Table 4 + §1.4 cross-reference.
- [x] **§1.5 contributions match §4-§7 deliveries.** (1) framework → §3, (2) cross-dataset evidence → §4.1 / §4.2 / §4.3 / §4.4, (3) routing-vs-integration synthesis → §5.4 + §4.6, (4) E6 mitigation → §6.2 / §6.5 / §7. All four deliveries intact post-Abstract surgery.
- [x] **§8.1 종합 consistent with body.** Reviewed line 460 — uses `wrong-base / correct-base 분할` (post-MAJ-W3), `×12.7 ratio (§4.5)` short-form (post-MAJ-W5), framework synthesis + §4.6 prospective + Δem(b) Bonferroni-robust headline framing all aligned with Abstract + §1.5 + §6.2.3.
- [x] **All figure embeds resolve.** 16 inline figures unchanged.
- [x] **Table numbering preserved.** Tables 1–9 + appendix Tables C.1 / C.2 / C.3 / A.5 row table all intact.
- [x] **Δem(b) multiplicity-robust headline appears identically in Abstract + §1.5 (4) + §6.2.3 + §8.1.** Verified four sites carry the `Δem(b) 5/5 cell × Bonferroni-corrected CI sign-clean` (or close paraphrase). User explicit consistency requirement met.
- [x] **§1.5(3) routing-vs-integration sub-claims match §5.4 + §4.6.** Sign-reversal mid-stack negative ↔ late-stack positive language consistent across Abstract / §1.3 / §1.5 / §5.4 / §4.6 / §8.1.
- [x] **No demoted claims resurrected.** Encoder-family-determines-archetype demotion (commit 549cf68) preserved; §D.1 still labels the clustering "fragile" + non-contribution.

## Diff stat

- **Lines:** 828 → 828 (surgical replacements, no net growth).
- **Word delta:** Abstract dropped ~390 words (680 → ~290); body absorbed ~50-word hunt-and-replace overhead; §6.6 absorbed ~80-word net deletion; §1.4 absorbed ~90-word net deletion; Insight box compressions ~60-word net deletion. **Estimated body word delta: ~−420 words.**
- **Sections substantially rewritten:** Abstract (3 paragraph → 1), §6.6 (chain + duplicate Summary → 2 paragraphs), §1.4 (re-derived γ-β content → §4.5-pointer stub), §6.2.3 (600-word inline reframe → 3-sentence headline), §4.1 opening (lab-report → claim-first), §4.4 line 203 (170-word run-on → 3 sentences), §6.2.2 line 357 (200-word run-on → 3 paragraphs).
- **Files affected:**
  - `docs/paper/emnlp_draft_ko.md` (~16 distinct edit sites, ~30 cells changed)
  - `docs/paper/CHANGELOG.md` (v10 entry appended)
  - `docs/paper/reviews/round2_response.md` (new — this file)
