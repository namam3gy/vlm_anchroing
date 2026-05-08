# Citation audit — 2026-05-04

Per `feedback_literature_citation_rigor.md`: every 2025+ arXiv ID in
`references/project.md` and `docs/paper/sections/` must resolve to a
real paper with title + first author matching the citation.

Source scan (9 IDs found):

| ID | Cited in | Cited as | Title (verified) | First author | Status |
|---|---|---|---|---|---|
| 2504.09946 | §1, §2, §8 | "Wang LRM-judging" | Assessing Judging Bias in Large Reasoning Models: An Empirical Study | Wang | ✅ matches |
| 2505.23941 | §2, §8 | "VLMBias [Vo, Nguyen et al.]" | Vision Language Models are Biased | Vo | ✅ matches |
| 2505.15392 | §2 | "A-Index / R-Error" anchoring paper | Understanding the Anchoring Effect of LLM with Synthetic Data: Existence, Mechanism, and Potential Mitigations | Huang | ✅ matches; venue = HCAIR workshop @ ICLR 2026 (verified 2026-05-08 via arXiv abs page) |
| 2507.03123 | §2 | "AIpsych [Liu et al.] psychology-grounded VLM cognitive-bias benchmark — sycophancy/authority/consistency" | Investigating VLM Hallucination from a Cognitive Psychology Perspective | Liu | ✅ matches — paper title is about hallucination but defines the AIpsych benchmark internally; abstract confirms "we design AIpsych, a scalable benchmark that reveals psychological tendencies" |
| 2506.05146 | §2 | "CIVET — VLM understanding [Rizzoli et al., 2025]" | CIVET: Systematic Evaluation of Understanding in VLMs | Rizzoli | ✅ matches; **EMNLP Findings 2025 tag NOT verified** — paper is arXiv preprint only as of 2026-05-08 (no named venue on arXiv abs page). §2 venue tag removed; project.md "went to Findings" softened to "no venue confirmed at audit time" |
| 2502.08193 | §2 | "Wang-Zhao-Larson, NAACL 2025" typographic-attacks | Typographic Attacks in a Multi-Image Setting | Wang, Zhao, Larson | ✅ matches; **NAACL 2025 confirmed** via arXiv abs page comments field (verified 2026-05-08) |
| 2508.20570 | §2 | "Dyslexify [Hufe et al.]" mechanistic typographic-attack defense | Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP | Hufe | ✅ matches |
| 2511.21397 | project.md "red flags" | "Idis" distractor benchmark | Do Reasoning Vision-Language Models Inversely Scale in Test-Time Compute? A Distractor-centric Empirical Analysis | Bae | ✅ matches — paper introduces the Idis (Images with distractors) VQA dataset internally |
| 2603.19203 | project.md "red flags" | "Tinted Frames" question-form framing | Tinted Frames: Question Framing Blinds Vision-Language Models | Fan | ✅ matches |

## Action items

All venue tags resolved on 2026-05-08:

- ✅ **2502.08193 (Wang-Zhao-Larson) — NAACL 2025 confirmed.**
- ❌ **2506.05146 (CIVET) — EMNLP Findings 2025 NOT confirmed.**
  Paper is arXiv preprint only as of 2026-05-08. §2 paper tag
  removed (now reads "Rizzoli et al., arXiv:2506.05146, 2025");
  project.md strategic argument softened.
- ✅ **2505.15392 (Huang anchoring) — HCAIR workshop @ ICLR 2026
  confirmed.** Final-paper proceedings link still pending until
  ICLR 2026 publishes workshop list.

Remaining hygiene before camera-ready:

- Verify all non-arXiv references (Jones & Steinhardt NeurIPS 2022,
  Echterhoff EMNLP Findings 2024, Goh Multimodal Neurons 2021,
  Hagendorff/Fabi/Kosinski Nature Computational Science 2023,
  Mussweiler & Strack 1999, Tversky & Kahneman 1974,
  Jacowitz & Kahneman 1995) against publication records.

## What's clean

9/9 arXiv citations have verified title + author match. 8/9 have a
fully-verified venue (or are clearly preprint-only); the remaining
1 (HCAIR @ ICLR 2026) has a venue claim that needs the final
proceedings URL when ICLR 2026 publishes.

## What is NOT in scope here

- Non-arXiv references (Jones & Steinhardt NeurIPS 2022, Echterhoff
  EMNLP Findings 2024, Goh Multimodal Neurons 2021,
  Hagendorff/Fabi/Kosinski Nature Computational Science 2023,
  Mussweiler & Strack 1999, etc.) — verify against publication
  records before camera-ready, but they are not in the 2025/2026
  arXiv-ID risk class that this audit targets.
- Internal `_data/*.csv` numeric-citation audit — covered separately
  by `feedback_paper_table_audit.md`.
