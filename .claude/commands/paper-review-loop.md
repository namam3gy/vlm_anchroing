---
description: Run a 5-round (10-step) iterative review + revision loop on docs/paper/emnlp_draft_ko.md. Methodology → Writing → Novelty → Aggressive → Bar Raiser, each followed by author revision.
---

You are running the iterative review-revision loop on the paper at `docs/paper/emnlp_draft_ko.md`.

## Pipeline (10 sequential steps — strict order)

| Step | Agent | Output file |
|---|---|---|
| 1 | `paper-reviewer-methodology` | `docs/paper/reviews/round1_methodology.md` |
| 2 | `paper-reviser` (responds to round 1) | `docs/paper/reviews/round1_response.md` + paper edits |
| 3 | `paper-reviewer-writing` | `docs/paper/reviews/round2_writing.md` |
| 4 | `paper-reviser` (responds to round 2) | `docs/paper/reviews/round2_response.md` + paper edits |
| 5 | `paper-reviewer-novelty` | `docs/paper/reviews/round3_novelty.md` |
| 6 | `paper-reviser` (responds to round 3) | `docs/paper/reviews/round3_response.md` + paper edits |
| 7 | `paper-reviewer-aggressive` | `docs/paper/reviews/round4_aggressive.md` |
| 8 | `paper-reviser` (responds to round 4) | `docs/paper/reviews/round4_response.md` + paper edits |
| 9 | `paper-reviewer-bar-raiser` | `docs/paper/reviews/round5_bar_raiser.md` |
| 10 | `paper-reviser` (responds to round 5) | `docs/paper/reviews/round5_response.md` + paper edits |

## Arguments

- `$ARGUMENTS` may specify a starting step number (default `1`) and/or a target paper path. Examples:
  - no argument → start from step 1, paper = default path.
  - `5` → resume from step 5 (skip steps 1-4).
  - `--from=3 --paper=docs/paper/emnlp_draft_ko.md` → explicit.
  - `--single-round=3` → run only one round (steps 5-6 in this case).

## Pre-flight checks

Before starting:

1. Verify `docs/paper/emnlp_draft_ko.md` exists. If not, abort with explanation.
2. Verify `docs/paper/reviews/` exists (create if not).
3. Verify the six agent files exist under `.claude/agents/`:
   - `paper-reviewer-methodology.md`
   - `paper-reviewer-writing.md`
   - `paper-reviewer-novelty.md`
   - `paper-reviewer-aggressive.md`
   - `paper-reviewer-bar-raiser.md`
   - `paper-reviser.md`
4. Capture starting paper state: `git rev-parse HEAD` if in repo, else `stat -c '%Y' <paper>`.

## Execution rules

For **each step**, do exactly this:

### Reviewer step (steps 1, 3, 5, 7, 9)

1. **Re-read the paper** before dispatching — it may have been edited by the prior reviser step.
2. **Dispatch the agent** via Agent tool with:
   - `subagent_type` = the agent name (e.g. `paper-reviewer-methodology`)
   - `prompt` = a self-contained brief that includes:
     - "Round N. Paper at docs/paper/emnlp_draft_ko.md."
     - "Prior reviews to read first: <list paths of round_*_<persona>.md and round_*_response.md that exist>."
     - "Save your review to docs/paper/reviews/round<N>_<persona>.md."
     - "Be thorough. Verify numerical claims against canonical CSV/MD in docs/insights/_data/."
     - "Cite section + line + canonical source for every weakness."
3. **Verify the review file exists and is non-trivial** (>2 KB at minimum). If the agent returned a stub or skipped sections, dispatch again with a corrective prompt.
4. **Mark this step complete** before moving on.

### Reviser step (steps 2, 4, 6, 8, 10)

1. **Re-read the just-written review file.**
2. **Dispatch `paper-reviser`** with:
   - `subagent_type` = `paper-reviser`
   - `prompt` = self-contained brief:
     - "Round N. Address the review at docs/paper/reviews/round<N>_<persona>.md."
     - "Read the paper at docs/paper/emnlp_draft_ko.md and all prior round files in docs/paper/reviews/."
     - "Edit the paper directly using Edit tool. Save the response document to docs/paper/reviews/round<N>_response.md."
     - "Do not fabricate experimental results. If reviewer asks for an experiment that isn't done, mark as DEFER honestly."
3. **After dispatch, verify**:
   - The response file exists and has the structured sections (decision table, edit log, rebuttals, deferred, internal-consistency check).
   - The paper file mtime is newer than the review file mtime (i.e. some edit actually happened — unless every reviewer point was DISAGREE/DEFER, in which case note this).
4. **Spot-check internal consistency**: re-read abstract + §1.5 + §8.1, verify they still cohere with §4-§7 tables.

## Inter-round bookkeeping

Between rounds:

1. Append a one-line entry to a running ledger at `docs/paper/reviews/_ledger.md`:
   ```
   - YYYY-MM-DD HH:MM | Round N | <persona> | review N words / response N words / edits N
   ```
2. If any reviewer step OR reviser step fails (file not written / file too short / agent error), STOP the loop and report to user. Do not silently skip.
3. If the reviser DEFERs a critical-level issue, log it but continue.

## After all 10 steps

Write a final summary at `docs/paper/reviews/_final_summary.md` with:

```markdown
# Paper Review Loop — Final Summary

**Paper:** docs/paper/emnlp_draft_ko.md
**Started:** <timestamp>
**Finished:** <timestamp>
**Initial paper version:** <git rev / mtime>
**Final paper version:** <git rev / mtime>

## Round-by-round headline

| Round | Persona | Decision | Major themes |
|---|---|---|---|
| 1 | Methodology | <accept/borderline/reject> | … |
| 2 | Writing | … | … |
| 3 | Novelty | … | … |
| 4 | Aggressive | … | … |
| 5 | Bar Raiser | <tier verdict> | … |

## Major themes addressed across rounds
- …

## Outstanding (DEFERred) items requiring future work
- …

## Bar-raiser signature ask (the one question)
<verbatim from round 5 review>

## Diff stat
- Lines changed: N
- Words delta: ±N
- Sections substantially rewritten: …
- Tables modified: …
- Figures unchanged / added / removed: …

## Final tier verdict
<from round 5>

## Recommendation for next pass
- …
```

## Discipline rules

- **Strict sequential.** Never run two reviewer agents in parallel; never skip ahead. The reviser must run between reviewers because the next reviewer must see the paper *after* prior revisions.
- **Re-read between steps.** The paper changes; agents that don't re-read will work from stale assumptions. Always pass the latest path.
- **Don't summarize away the reviews.** Each round file is the artifact. Don't paraphrase reviews into a "rolled-up" file — agents need the full prior review text.
- **Halt on failure.** If a reviewer file is too short or a reviser doesn't actually edit the paper, halt and ask the user before continuing.
- **The user can interrupt.** Auto mode is on, but the user may want to inspect after a specific round. If the user says "stop after round 3," respect it and save state cleanly.
- **Time budget.** Each reviewer run might take 5-15 minutes; reviser 5-15 minutes. Budget ~2-3 hours wall-clock for a full 10-step loop. Use background dispatch (run_in_background) only if explicitly requested — by default keep foreground so progress is visible.

## Resumability

If the loop is interrupted:
- The user can resume by invoking `/paper-review-loop <starting-step>`.
- Before resuming, verify the previous step's output file exists and is non-trivial. If incomplete, restart from the failed step.
- Do not redo completed steps unless the user asks.

## Argument: $ARGUMENTS

If `$ARGUMENTS` is non-empty, parse it as a starting step number (1-10). Otherwise default to step 1.
