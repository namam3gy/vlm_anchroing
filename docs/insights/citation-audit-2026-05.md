# Citation audit — 2026-05-04

Per `feedback_literature_citation_rigor.md`: every 2025+ arXiv ID in
`references/project.md` and `docs/paper/sections/` must resolve to a
real paper with title + first author matching the citation.

Source scan (9 IDs found):

| ID | Cited in | Cited as | Title (verified) | First author | Status |
|---|---|---|---|---|---|
| 2504.09946 | §1, §2, §8 | "Wang LRM-judging" | Assessing Judging Bias in Large Reasoning Models: An Empirical Study | Wang | ✅ matches |
| 2505.23941 | §2, §8 | "VLMBias [Vo, Nguyen et al.]" | Vision Language Models are Biased | Vo | ✅ matches |
| 2505.15392 | §2 | "A-Index / R-Error" anchoring paper | Understanding the Anchoring Effect of LLM with Synthetic Data: Existence, Mechanism, and Potential Mitigations | Huang | ✅ matches; venue = ICLR 2026 HCAIR workshop |
| 2507.03123 | §2 | "AIpsych [Liu et al.] psychology-grounded VLM cognitive-bias benchmark — sycophancy/authority/consistency" | Investigating VLM Hallucination from a Cognitive Psychology Perspective | Liu | ✅ matches — paper title is about hallucination but defines the AIpsych benchmark internally; abstract confirms "we design AIpsych, a scalable benchmark that reveals psychological tendencies" |
| 2506.05146 | §2 | "CIVET — VLM understanding [Rizzoli et al., EMNLP Findings 2025]" | CIVET: Systematic Evaluation of Understanding in VLMs | Rizzoli | ✅ matches; EMNLP Findings 2025 venue claim unverified by arXiv metadata alone |
| 2502.08193 | §2 | "Wang-Zhao-Larson, NAACL 2025" typographic-attacks | Typographic Attacks in a Multi-Image Setting | Wang, Zhao, Larson (verified via arXiv citation_author meta) | ⚠ author + title match; **NAACL 2025 venue unverified**. Submitted to arXiv 2025-02-12 — possible Findings/workshop track, not confirmed via aclanthology fetch. |
| 2508.20570 | §2 | "Dyslexify [Hufe et al.]" mechanistic typographic-attack defense | Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP | Hufe | ✅ matches |
| 2511.21397 | project.md "red flags" | "Idis" distractor benchmark | Do Reasoning Vision-Language Models Inversely Scale in Test-Time Compute? A Distractor-centric Empirical Analysis | Bae | ✅ matches — paper introduces the Idis (Images with distractors) VQA dataset internally |
| 2603.19203 | project.md "red flags" | "Tinted Frames" question-form framing | Tinted Frames: Question Framing Blinds Vision-Language Models | Fan | ✅ matches |

## Action items

- **Confirm or remove "NAACL 2025" venue tag on 2502.08193.** Search
  ACL Anthology for "Typographic Attacks in a Multi-Image Setting" or
  "Wang Zhao Larson typographic" before final paper submission. If
  not found, replace with "arXiv 2502.08193, 2025-02" only.
- **Confirm or remove "EMNLP Findings 2025" venue tag on 2506.05146**
  (CIVET).
- **Confirm or remove "ICLR 2026 HCAIR workshop" venue tag on
  2505.15392** (Huang anchoring) — surfaced via WebFetch on the abs
  page; would still need final-paper proceedings link.

## What's clean

7/9 citations have title + author resolving exactly to the cited
paper. The remaining 2 have minor description-vs-title gaps that are
nonetheless internally consistent (the cited *content* — AIpsych
benchmark, Idis dataset — is genuinely defined in the respective
papers, just not in the title).

## What is NOT in scope here

- Non-arXiv references (Jones & Steinhardt NeurIPS 2022, Echterhoff
  EMNLP Findings 2024, Goh Multimodal Neurons 2021,
  Hagendorff/Fabi/Kosinski Nature Computational Science 2023,
  Mussweiler & Strack 1999, etc.) — verify against publication
  records before camera-ready, but they are not in the 2025/2026
  arXiv-ID risk class that this audit targets.
- Internal `_data/*.csv` numeric-citation audit — covered separately
  by `feedback_paper_table_audit.md`.
