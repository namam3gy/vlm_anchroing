// Korean paper-summary deck for VLM cross-modal anchoring paper (expanded).
// Output: paper_summary_kr.pptx (28 slides, LAYOUT_WIDE 13.3" × 7.5")
//
// Sourced from docs/paper/sections/0[1-8]_*.md + references/roadmap.md §3.3
// + docs/insights/_data/*.csv + docs/insights/E1-patch-evidence.md
// Numbers are pinned to the C-form M2 metrics (§3.4 of the paper draft).

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const REPO = "/mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing";

// ----------------------------------------------------------------------------
// Palette + typography
// ----------------------------------------------------------------------------
const NAVY = "1E2761";
const TERRA = "B85042";
const GOLD = "D4AF37";
const CREAM = "F8F6F2";
const CHARCOAL = "333333";
const MUTED = "6B7280";
const GREEN = "2C5F2D";
const PALE_NAVY = "E8ECF5";
const PALE_TERRA = "FAEDEA";

const TITLE_FONT = "Calibri";
const BODY_FONT = "Calibri";

const W = 13.333, H = 7.5;
const TOTAL_PAGES = 28;

// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------
function addPageHeader(slide, title, subtitle) {
  slide.addShape("rect", {
    x: 0.4, y: 0.55, w: 0.10, h: 0.7,
    fill: { color: TERRA }, line: { color: TERRA, width: 0 },
  });
  slide.addText(title, {
    x: 0.65, y: 0.40, w: W - 1.5, h: 0.55,
    fontFace: TITLE_FONT, fontSize: 24, bold: true,
    color: NAVY, valign: "middle", margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.65, y: 0.95, w: W - 1.5, h: 0.35,
      fontFace: BODY_FONT, fontSize: 11, italic: true,
      color: MUTED, valign: "middle", margin: 0,
    });
  }
}

function addPageFooter(slide, pageNum, sectionLabel) {
  slide.addShape("rect", {
    x: 0, y: H - 0.32, w: W, h: 0.04,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  slide.addText(sectionLabel || "", {
    x: 0.4, y: H - 0.32, w: 6, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, color: MUTED,
    valign: "middle", margin: 0,
  });
  slide.addText(`${pageNum} / ${TOTAL_PAGES}`, {
    x: W - 1.2, y: H - 0.32, w: 0.8, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, color: MUTED,
    align: "right", valign: "middle", margin: 0,
  });
}

function tH(text, opts = {}) {
  return { text, options: {
    bold: true, color: "FFFFFF", fill: { color: NAVY },
    fontSize: 10, fontFace: BODY_FONT, align: "center", valign: "middle",
    margin: 0.04, ...opts,
  }};
}
function tB(text, opts = {}) {
  return { text: String(text), options: {
    color: CHARCOAL, fontSize: 10, fontFace: BODY_FONT,
    align: "center", valign: "middle",
    fill: { color: "FFFFFF" }, margin: 0.04, ...opts,
  }};
}
function tBL(text, opts = {}) { return tB(text, { align: "left", margin: 0.06, ...opts }); }
function tHL(text, opts = {}) {
  return tB(text, { bold: true, color: TERRA, fill: { color: CREAM }, ...opts });
}

function note(slide, text, opts = {}) {
  slide.addText(text, {
    x: 0.6, y: H - 0.85, w: W - 1.2, h: 0.45,
    fontFace: BODY_FONT, fontSize: 9.5, italic: true,
    color: MUTED, valign: "top", margin: 0, ...opts,
  });
}

// ============================================================================
// Build deck
// ============================================================================
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "VLM 교차모달 숫자 anchoring — paper §1-§8 요약 (한국어, 확장판)";
pres.author = "vlm_anchoring authors";

// ---------------- Slide 1 — Title ----------------
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape("rect", { x: 0, y: H - 0.6, w: W, h: 0.6, fill: { color: TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("VLM의 교차모달 숫자 anchoring", {
    x: 0.8, y: 1.7, w: W - 1.6, h: 1.0,
    fontFace: TITLE_FONT, fontSize: 44, bold: true, color: "FFFFFF", valign: "middle",
  });
  s.addText("Cross-modal numerical anchoring in vision-language models", {
    x: 0.8, y: 2.7, w: W - 1.6, h: 0.5,
    fontFace: BODY_FONT, fontSize: 20, italic: true, color: GOLD, valign: "middle",
  });
  s.addText([
    { text: "uncertainty-modulated graded pull · digit-pixel 인과성 · upper-half free-lunch mitigation · reasoning이 anchoring을 강화", options: { fontSize: 14, color: "FFFFFF" } },
  ], { x: 0.8, y: 3.5, w: W - 1.6, h: 0.5, valign: "middle" });
  s.addText([
    { text: "EMNLP 2026 Main · ARR May 25 target", options: { fontSize: 13, color: CREAM, bold: true } },
    { text: "  ·  ", options: { fontSize: 13, color: GOLD } },
    { text: "12 open-weight VLM · 4 numeric VQA dataset · ~1.6M generations", options: { fontSize: 13, color: CREAM } },
  ], { x: 0.8, y: 4.1, w: W - 1.6, h: 0.4, valign: "middle" });
  s.addText("paper §1-§8 종합 요약 · 한국어판 (확장) · 2026-04-29", {
    x: 0.8, y: H - 0.55, w: W - 1.6, h: 0.5,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: "FFFFFF", valign: "middle",
  });
}

// ---------------- Slide 2 — TL;DR ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "한 줄 요약 (TL;DR)", "§1.3 Headline · 3-pillar claim + free-lunch mitigation + γ-β reasoning hook");
  s.addText("VLM에 무관한 숫자 이미지를 함께 주면 예측이 그 숫자 쪽으로 끌립니다. 이 효과는 categorical capture가 아닌 graded pull, base-prediction 불확실성에 비례, anchor 이미지의 digit pixel이 인과적입니다.", {
    x: 0.6, y: 1.45, w: W - 1.2, h: 0.85,
    fontFace: BODY_FONT, fontSize: 14, color: CHARCOAL, valign: "middle", margin: 0,
    fill: { color: PALE_NAVY }, charSpacing: -0.5,
  });
  const cardY = 2.45, cardH = 2.6;
  const cards = [
    { n: "1", title: "Graded vs. categorical", body: "wrong-base direction-follow이 correct-base보다 +6.9~+19.6 pp 큼 (7/7 모델). 그러나 paired adopt(literal copy)는 2-7%에 불과 — 모델은 anchor 숫자를 그대로 출력하지 않고 자기 baseline 예측을 anchor 쪽으로 ‘기울일’ 뿐." },
    { n: "2", title: "Digit-pixel 인과성", body: "anchor 이미지에서 숫자 픽셀만 OpenCV Telea inpaint로 지우면 효과가 generic 2-image distractor 수준으로 떨어짐. 배경 scene이 아니라 digit pixel이 원인. 4-model E1-patch에서 fair share 대비 +24~+40 pp 집중." },
    { n: "3", title: "Confidence-modulated", body: "answer-token entropy로 4분위 분할하면 direction-follow가 단조 증가 (mean Q4-Q1 = +15.2 pp, 23/35 cell에서 fully monotone). Phase-A wrong/correct 이분법은 이 연속 구조의 거친 투영." },
  ];
  const cw = (W - 1.2 - 0.4) / 3;
  cards.forEach((c, i) => {
    const cx = 0.6 + i * (cw + 0.2);
    s.addShape("rect", { x: cx, y: cardY, w: cw, h: cardH, fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 } });
    s.addShape("ellipse", { x: cx + 0.25, y: cardY + 0.25, w: 0.55, h: 0.55, fill: { color: TERRA }, line: { color: TERRA, width: 0 } });
    s.addText(c.n, { x: cx + 0.25, y: cardY + 0.25, w: 0.55, h: 0.55, fontFace: TITLE_FONT, fontSize: 22, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(c.title, { x: cx + 0.95, y: cardY + 0.25, w: cw - 1.1, h: 0.55, fontFace: TITLE_FONT, fontSize: 14, bold: true, color: NAVY, valign: "middle" });
    s.addText(c.body, { x: cx + 0.25, y: cardY + 0.95, w: cw - 0.5, h: cardH - 1.15, fontFace: BODY_FONT, fontSize: 10.5, color: CHARCOAL, valign: "top", paraSpaceAfter: 4 });
  });
  // 두 추가 hook
  s.addShape("rect", { x: 0.6, y: 5.2, w: (W - 1.4) / 2, h: 1.5, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0.75 } });
  s.addText("Free-lunch mitigation (§7.4)", { x: 0.8, y: 5.3, w: (W - 1.4) / 2 - 0.4, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA });
  s.addText("LLM stack의 upper-half attention re-weighting (E4 Phase 2, n=88,650/모델, mid-stack-cluster 3 모델). df −5.8~−14.6% relative ↓ · exact_match +0.49~+1.30 pp ↑ · target_only accuracy 불변 — anchor pathway에만 작용.", {
    x: 0.8, y: 5.7, w: (W - 1.4) / 2 - 0.4, h: 0.95, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 3,
  });
  s.addShape("rect", { x: 0.6 + (W - 1.4) / 2 + 0.2, y: 5.2, w: (W - 1.4) / 2, h: 1.5, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0.75 } });
  s.addText("γ-β: reasoning은 anchor를 더 강화 (§8.1)", { x: 0.8 + (W - 1.4) / 2 + 0.2, y: 5.3, w: (W - 1.4) / 2 - 0.4, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  s.addText("Qwen3-VL-8B-Instruct vs Thinking, MathVista — Thinking이 adopt ×1.6, df ×2.9. text-only LRM-judging 결과 (Wang 2025)와 일치하는 multimodal counterpart. existence proof.", {
    x: 0.8 + (W - 1.4) / 2 + 0.2, y: 5.7, w: (W - 1.4) / 2 - 0.4, h: 0.95, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 3,
  });
  addPageFooter(s, 2, "§1 · TL;DR");
}

// ---------------- Slide 3 — Motivation + 인지과학 ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "동기 — 인지과학에서 LLM, 그리고 VLM으로", "§1.1 motivation: humans anchor on irrelevant numerical cues, LLM도 그렇다 — VLM은?");
  s.addText("실세계 멀티이미지 prompt — 화면 캡처, 앨범, 문서 스캔. 그중 하나가 무관한 숫자 이미지라면 모델은 흔들리는가?", {
    x: 0.6, y: 1.55, w: W - 1.2, h: 0.5,
    fontFace: BODY_FONT, fontSize: 13, italic: true, color: NAVY, valign: "middle", margin: 0,
  });
  // Three columns: Cognitive | LLM | This work (VLM)
  const colY = 2.2, colH = 4.5;
  const cw = (W - 1.4) / 3;
  const cols = [
    { color: NAVY, head: "1) Cognitive science prior", body: [
      { b: "Tversky-Kahneman 1974", t: "anchoring & adjustment heuristic — 무관한 숫자에도 인간 추정이 끌림" },
      { b: "Mussweiler-Strack 1997", t: "selective accessibility model — anchor가 search-space의 candidate로 들어가서 답에 ‘blend’" },
      { b: "Jacowitz-Kahneman 1995", t: "plausibility-window account — 그럴듯한 거리 안에서만 anchor가 작동" },
    ]},
    { color: TERRA, head: "2) LLM-only literature", body: [
      { b: "Jones-Steinhardt 2022", t: "GPT-3/Codex/CodeGen에서 anchoring + framing + availability 6 biases 측정" },
      { b: "Echterhoff 2024", t: "EMNLP Findings — 13,465 prompt × 5 bias × 4+ model + BiasBuster debias" },
      { b: "Lou-Sun 2024", t: "stronger LLM이 더 anchor — capability ≠ robustness" },
      { b: "Huang 2025", t: "A-Index, R-Error metric (본 논문 metric의 비교 대상)" },
      { b: "Wang 2025 (LRM)", t: "reasoning models이 several biases에 더 취약 — 본 논문 §8.1 γ-β의 motivation" },
    ]},
    { color: GREEN, head: "3) 본 논문이 다루는 gap", body: [
      { b: "VLM에서의 cross-modal numerical anchoring", t: "stand-alone rendered-digit image (semantic label 없음) × 무관 target 이미지" },
      { b: "regression-style numeric shift 측정", t: "ASR/classification flip 아닌 ‘pa−pb’ 회귀형 — Jacowitz-Kahneman과 직접 호환" },
      { b: "인지과학 grounding 유지", t: "uncertainty (§5.2/§6) + plausibility (§5.3) + digit-pixel (§5.4) — 3-gate가 모두 동작" },
    ]},
  ];
  cols.forEach((c, i) => {
    const cx = 0.6 + i * (cw + 0.2);
    s.addShape("rect", { x: cx, y: colY, w: cw, h: colH, fill: { color: "FFFFFF" }, line: { color: c.color, width: 1.5 } });
    s.addShape("rect", { x: cx, y: colY, w: cw, h: 0.45, fill: { color: c.color }, line: { color: c.color, width: 0 } });
    s.addText(c.head, { x: cx + 0.1, y: colY, w: cw - 0.2, h: 0.45, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: "FFFFFF", valign: "middle" });
    let yy = colY + 0.55;
    c.body.forEach((bb) => {
      s.addText(bb.b, { x: cx + 0.15, y: yy, w: cw - 0.3, h: 0.3, fontFace: TITLE_FONT, fontSize: 10.5, bold: true, color: c.color, valign: "top" });
      s.addText(bb.t, { x: cx + 0.15, y: yy + 0.3, w: cw - 0.3, h: 0.7, fontFace: BODY_FONT, fontSize: 9.5, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
      yy += 0.95;
    });
  });
  note(s, "Ask: 이 anchoring 효과가 image modality로 전이되는가? answer label 없이 숫자만 그려진 이미지 하나가, 별개의 target 이미지에 함께 주어졌을 때 예측을 어떻게 흔드는가?");
  addPageFooter(s, 3, "§1.1 · Motivation");
}

// ---------------- Slide 4 — Novelty matrix vs prior art ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "Novelty 매트릭스 — 기존 멀티모달 bias 연구와 무엇이 다른가", "§1.2 + §2 — 본 논문이 unmeasured했던 정확한 위치");
  const rows = [
    [tH("연구"), tH("자극 (cue 형태)"), tH("측정 metric"), tH("본 논문과의 차이")],
    [tBL("VLMBias (Vo, Nguyen 2025)"), tBL("memorized-subject label\n(예: ‘Aston Martin DB5’)"), tBL("counterfactual counting flip"), tBL("semantic label 있음 vs. 우리는 단일 숫자 (label 없음)\nclassification flip vs. 우리는 regression-style numeric shift")],
    [tBL("Typographic attacks\n(Goh 2021 / Wang 2025 NAACL / Hufe 2025 Dyslexify)"), tBL("class-label text를 target 위에 paste"), tBL("ASR (classification flip)"), tBL("target에 paste vs. 우리는 별개의 anchor image\nASR vs. 우리는 anchor pull regression\n인과 control(masked arm) 없음 vs. 우리는 inpaint mask로 분리")],
    [tBL("FigStep (Gong AAAI 2025)"), tBL("harmful instruction을 image로 렌더"), tBL("jailbreak 우회"), tBL("rendered-text 메커니즘 공유, 그러나 numeric estimation이 아닌 jailbreak 표적")],
    [tBL("AIpsych (Liu 2025)"), tBL("psychology-grounded VLM bias 평가"), tBL("sycophancy / authority / consistency"), tBL("anchoring은 covered되지 않음 — 본 논문의 cross-modal anchoring과 직접 겹치지 않음")],
    [tBL("CIVET (Rizzoli EMNLP F 2025)"), tBL("VLM understanding 평가"), tBL("position bias 등 robustness"), tBL("numeric anchoring을 별도로 측정하지 않음")],
    [tBL("Tinted Frames (Fan 2026)"), tBL("question-form framing 변형"), tBL("framing-effect 응답 변화"), tBL("framing이지 cross-modal numerical anchoring이 아님")],
    [tBL("Weng EMNLP 2024 Main\n(causal mediation + 22% mitigation)"), tBL("multimodal hallucination causal study"), tBL("hallucination behavioural + 22% mitigation"), tBL("연구 template 동일 (behavioural+mechanism+mitigation) — 본 논문이 해당 template를 cross-modal anchoring에 적용")],
    [tBL("Huang 2025 A-Index, R-Error"), tBL("text anchor (LLM)"), tBL("text-only LLM anchoring"), tBL("metric 비교 대상으로 reuse, modality는 image로 확장")],
  ];
  s.addTable(rows, {
    x: 0.55, y: 1.55, w: W - 1.1, colW: [2.4, 3.0, 2.6, W - 1.1 - 2.4 - 3.0 - 2.6],
    rowH: [0.3, 0.6, 0.8, 0.55, 0.55, 0.5, 0.55, 0.7, 0.55],
    fontFace: BODY_FONT, fontSize: 9.5,
  });
  note(s, "TL;DR — VLMBias가 가장 가까운 이웃이지만 (a) cue 형태 (semantic label vs digit-only), (b) metric (flip vs regression shift), (c) 인과 control (없음 vs masked arm)에서 load-bearing한 차이.   인지과학 framing 유지가 본 논문의 분석 깊이를 결정한다.");
  addPageFooter(s, 4, "§1.2 / §2 · Novelty");
}

// ---------------- Slide 5 — 4-condition setup with images ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "실험 셋업 — 4-condition 자극 (per sample_instance)", "§3.1 · 동일한 (target, question) 위에 두 번째 이미지를 b/a/m/d로 바꿔 출력 비교");
  const tilesY = 1.55;
  const tileW = (W - 1.6) / 4;
  const tileH = 2.6;
  const tiles = [
    { tag: "b · target_only", color: NAVY, img: path.join(REPO, "inputs/vqav2_number_val/images/000000000139.jpg"),
      cap: "두 번째 이미지 없음 (baseline)", pred: "pred_b" },
    { tag: "a · target + anchor", color: TERRA, img: path.join(REPO, "inputs/irrelevant_number/3.png"),
      cap: "FLUX-쉬넬 1024² 1-step rendering — Arabic numeral 1개 (예: 3)", pred: "pred_a" },
    { tag: "m · target + masked", color: GOLD, img: path.join(REPO, "inputs/irrelevant_number_masked/3.png"),
      cap: "동일 anchor scene을 PaddleOCR-bbox + Telea inpaint로 digit pixel만 제거", pred: "pred_m" },
    { tag: "d · target + neutral", color: GREEN, img: path.join(REPO, "inputs/irrelevant_neutral/13.png"),
      cap: "scene-stylistic distribution은 anchor와 매칭, 숫자 없음 (generic 2-image distractor)", pred: "pred_d" },
  ];
  tiles.forEach((t, i) => {
    const tx = 0.6 + i * (tileW + 0.2);
    s.addShape("rect", { x: tx, y: tilesY, w: tileW, h: 0.4, fill: { color: t.color }, line: { color: t.color, width: 0 } });
    s.addText(t.tag, { x: tx, y: tilesY, w: tileW, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    if (fs.existsSync(t.img)) {
      s.addImage({ path: t.img, x: tx + 0.1, y: tilesY + 0.5, w: tileW - 0.2, h: 1.6 });
    } else {
      s.addShape("rect", { x: tx + 0.1, y: tilesY + 0.5, w: tileW - 0.2, h: 1.6, fill: { color: CREAM }, line: { color: MUTED, width: 0.5 } });
    }
    s.addText(t.cap, { x: tx + 0.1, y: tilesY + 2.2, w: tileW - 0.2, h: 0.45, fontFace: BODY_FONT, fontSize: 9, color: CHARCOAL, align: "center", valign: "top", paraSpaceAfter: 1 });
    s.addText(t.pred, { x: tx, y: tilesY + tileH + 0.1, w: tileW, h: 0.3, fontFace: TITLE_FONT, fontSize: 12, bold: true, italic: true, color: t.color, align: "center", valign: "middle" });
  });
  // Three condition gaps card
  s.addShape("rect", { x: 0.6, y: 4.85, w: W - 1.2, h: 1.85, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0.75 } });
  s.addText("3가지 핵심 비교 (condition gap)", { x: 0.8, y: 4.95, w: W - 1.6, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  s.addText([
    { text: "①  (a − d)\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "anchoring vs. generic 2-image distraction 분리 — digit 자체가 핵심인지 검증.\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "②  (a − m)\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "digit pixel만의 기여 — 같은 scene을 inpaint한 m을 빼면 픽셀 효과 분리.\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "③  ((a, base-wrong) − (a, base-correct))\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "uncertainty modulation (Phase A 이분법; §6에서 entropy 4분위로 연속화).", options: { fontSize: 10, color: CHARCOAL } },
  ], { x: 0.8, y: 5.35, w: W - 1.6, h: 1.3, valign: "top", paraSpaceAfter: 0 });
  addPageFooter(s, 5, "§3.1 · Setup");
}

// ---------------- Slide 6 — Stimulus inventory + prompt + decoding ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "자극 인벤토리 + Prompt + Decoding", "§3.2 + §3.3 — 재현 가능한 stimuli/prompt/decoding 명세");
  // Left: inventories
  s.addShape("rect", { x: 0.55, y: 1.5, w: 6.0, h: 5.4, fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 } });
  s.addShape("rect", { x: 0.55, y: 1.5, w: 6.0, h: 0.45, fill: { color: NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("3가지 stimulus 인벤토리 (각 128 PNG)", { x: 0.65, y: 1.5, w: 5.8, h: 0.45, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: "FFFFFF", valign: "middle" });
  s.addText([
    { text: "▸ Anchor 인벤토리 (`a`)\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "FLUX.1-schnell · 1024×1024 · 1 inference step · guidance scale 0.\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "filename = digit value (예: `5.png`). 각 PNG는 단일 Arabic numeral 한 개를 그린 random scene.\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "▸ Mask 인벤토리 (`m`)\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "각 anchor의 ‘digit pixel region만 inpaint’한 counterpart.\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "PaddleOCR로 digit bbox 추출 → dilated bbox 영역에 OpenCV Telea inpaint 적용.\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "Synthetic-bbox fallback. OCR-validated post-inpaint (digit 미검출 보장).\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "▸ Neutral 인벤토리 (`d`)\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "FLUX-rendered digit-free image, scene-stylistic distribution은 anchor와 matched.\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "generic 2-image distractor 역할 — “두 번째 이미지가 있어서 산만한가”의 baseline.\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "▸ Release\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "3 인벤토리 모두 코드 + FLUX seed + OpenCV inpaint 파라미터와 함께 공개 (§1.7).", options: { fontSize: 10, color: CHARCOAL } },
  ], { x: 0.75, y: 2.05, w: 5.6, h: 4.7, valign: "top" });

  // Right: prompt + decoding
  s.addShape("rect", { x: 6.85, y: 1.5, w: 5.95, h: 5.4, fill: { color: "FFFFFF" }, line: { color: TERRA, width: 1 } });
  s.addShape("rect", { x: 6.85, y: 1.5, w: 5.95, h: 0.45, fill: { color: TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("Prompt template + Decoding (§3.3)", { x: 6.95, y: 1.5, w: 5.75, h: 0.45, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: "FFFFFF", valign: "middle" });
  s.addText([
    { text: "system\n", options: { bold: true, fontSize: 10, color: NAVY } },
    { text: "You are a visual question answering system. Return valid JSON only in the form ", options: { fontSize: 9.5, color: CHARCOAL, fontFace: "Consolas" } },
    { text: "{\"result\": <number>}", options: { fontSize: 9.5, color: TERRA, fontFace: "Consolas", bold: true } },
    { text: ". Use a numeric JSON value for <number>, not a string. Do not output any other keys, words, explanation, or markdown. If uncertain, still output the single most likely number in that JSON format.\n\n", options: { fontSize: 9.5, color: CHARCOAL, fontFace: "Consolas" } },
    { text: "user\n", options: { bold: true, fontSize: 10, color: NAVY } },
    { text: "Answer the question using the provided image(s). Return JSON only in the form ", options: { fontSize: 9.5, color: CHARCOAL, fontFace: "Consolas" } },
    { text: "{\"result\": <number>}", options: { fontSize: 9.5, color: TERRA, fontFace: "Consolas", bold: true } },
    { text: ". Question: {question}", options: { fontSize: 9.5, color: CHARCOAL, fontFace: "Consolas" } },
  ], { x: 7.0, y: 2.05, w: 5.7, h: 3.0, valign: "top", fill: { color: CREAM } });
  s.addText([
    { text: "Decoding\n", options: { bold: true, fontSize: 11, color: NAVY } },
    { text: "• temperature 0 · top_p 1 · greedy\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "• max_new_tokens=8 (non-thinking 모델)\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "• max_new_tokens=512 (Qwen3-VL Thinking γ-β만, ", options: { fontSize: 10, color: CHARCOAL } },
    { text: "</think>", options: { fontSize: 10, fontFace: "Consolas", color: TERRA } },
    { text: " post-trace 파싱)\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "Single-prompt run", options: { bold: true, fontSize: 10, color: NAVY } },
    { text: " — paraphrase robustness (3-5 prompt × bootstrap CI)는 §8.4 limitations에서 disclose.", options: { fontSize: 10, color: CHARCOAL } },
  ], { x: 7.0, y: 5.2, w: 5.7, h: 1.7, valign: "top" });
  addPageFooter(s, 6, "§3.2 / §3.3 · Stimuli + Prompt");
}

// ---------------- Slide 7 — M2 metrics + C-form decision rationale ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "Canonical M2 metrics (C-form, 2026-04-28 확정)", "§3.4 + §3.6 · 모든 §5/§6/§7 표가 이 정의 + 폐기된 두 후보(anchor·gt / pb·gt)와의 비교");
  // Left: 3 metric cards
  const metricsX = 0.55, metricsY = 1.5, mw = 7.0, mh = 5.4;
  s.addShape("rect", { x: metricsX, y: metricsY, w: mw, h: mh, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("3 핵심 지표 (per (model, dataset, condition) cell)", { x: metricsX + 0.15, y: metricsY + 0.1, w: mw - 0.3, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY, valign: "middle" });
  const metrics = [
    { name: "adopt_rate (M2 paired)", formula: "#(pa = anchor AND pb ≠ anchor) / #(pb ≠ anchor)", reading: "기저 예측이 anchor와 다른 샘플 중, anchor 추가 후 anchor 값을 그대로 출력하는 비율 (literal copy).\n분모에서 base-prediction confound 제거 — 우연히 base가 anchor와 같았던 케이스 빼고 계산." },
    { name: "direction_follow_rate (C-form)", formula: "#( (pa − pb) · (anchor − pb) > 0  AND  pa ≠ pb ) / #(numeric pair AND anchor present)", reading: "예측이 baseline에서 anchor 방향으로 움직였는지의 sign-based 측정 (gt 미사용).\npa가 pb 기준으로 anchor 쪽 부호로 이동했으면 1, 아니면 0. stimulus draw에 robust." },
    { name: "exact_match", formula: "#(pa = gt) / #(numeric pair)", reading: "anchor 조건에서의 정답 일치율 (per-arm accuracy).\nanchoring으로 떨어진 정확도를 직접 측정 — §7.4 mitigation 결과 검증에 사용." },
  ];
  let yy = metricsY + 0.6;
  metrics.forEach((m) => {
    s.addText(m.name, { x: metricsX + 0.15, y: yy, w: mw - 0.3, h: 0.3, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
    s.addText(m.formula, { x: metricsX + 0.3, y: yy + 0.3, w: mw - 0.5, h: 0.35, fontFace: "Consolas", fontSize: 10.5, color: NAVY, fill: { color: "FFFFFF" }, valign: "middle", margin: 0.06 });
    s.addText(m.reading, { x: metricsX + 0.3, y: yy + 0.7, w: mw - 0.5, h: 0.85, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
    yy += 1.6;
  });

  // Right: C-form rationale — 2 discarded forms
  s.addShape("rect", { x: 7.65, y: 1.5, w: 5.15, h: 5.4, fill: { color: "FFFFFF" }, line: { color: TERRA, width: 1 } });
  s.addShape("rect", { x: 7.65, y: 1.5, w: 5.15, h: 0.45, fill: { color: TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("왜 C-form? — 폐기된 2 후보", { x: 7.75, y: 1.5, w: 4.95, h: 0.45, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: "FFFFFF", valign: "middle" });
  s.addText([
    { text: "✗ anchor·gt form\n", options: { bold: true, fontSize: 11, color: NAVY } },
    { text: "(pa−gt)·(anchor−gt) > 0\n", options: { fontFace: "Consolas", fontSize: 10, color: CHARCOAL } },
    { text: "→ pa와 anchor가 gt 같은 쪽인지. anchor draw에 따라 indicator 흔들림 (특히 VQAv2처럼 gt와 anchor 분포가 겹치는 dataset에서). M1 era 결과 폐기, 2026-04-28 reaggregate.\n\n", options: { fontSize: 9.5, color: CHARCOAL } },
    { text: "✗ pb·gt form\n", options: { bold: true, fontSize: 11, color: NAVY } },
    { text: "(pb−gt)·(pa−gt) > 0\n", options: { fontFace: "Consolas", fontSize: 10, color: CHARCOAL } },
    { text: "→ anchor가 formula에 등장하지 않음. pb-stickiness만 측정. metric 이름과 conflict, 폐기.\n\n", options: { fontSize: 9.5, color: CHARCOAL } },
    { text: "✓ C-form 선택\n", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: "(pa−pb)·(anchor−pb) > 0\n", options: { fontFace: "Consolas", fontSize: 10, color: CHARCOAL } },
    { text: "→ 모델 출력 + anchor draw에만 의존 (gt 미사용). dataset 간 비교 robust. 207 cell migration audit 후 모든 paper-tier 결과 보존 또는 강화 — 가장 큰 변화는 L1 confidence quartile mean Q4-Q1 (0.128 → 0.152).", options: { fontSize: 9.5, color: CHARCOAL } },
  ], { x: 7.8, y: 2.05, w: 4.85, h: 4.7, valign: "top" });

  note(s, "6 per-row flag (anchor_adopted / anchor_direction_followed / pred_b_equal_anchor / pred_diff_from_base / numeric_distance_to_anchor / *_moved)을 predictions.jsonl에 persist + driver schema parity 회귀 테스트 — silent zeroing은 재발 불가 (§3.5).");
  addPageFooter(s, 7, "§3.4 / §3.6 · Metrics");
}

// ---------------- Slide 8 — Datasets table + cutoff rules ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "Datasets — 4종 numeric VQA + per-dataset cutoff rule", "§4.1 — VQAv2 / TallyQA / ChartQA / MathVista, integer-GT 부분집합 (require_single_numeric_gt)");
  const rows = [
    [tH("Dataset"), tH("내용"), tH("GT 범위"), tH("필터된 샘플"), tH("역할 / 비고")],
    [tBL("VQAv2 number"), tBL("자연 이미지 counting / numeric VQA"), tBL("0–8"), tBL("17,730"), tBL("가장 큰 counting VQA — primary 메인 패널")],
    [tBL("TallyQA"), tBL("counting under 가림/모호 (자연 이미지)"), tBL("0–15 (≤8 cap)"), tBL("38,245 (test number-type)"), tBL("counting 난이도가 더 높음")],
    [tBL("ChartQA"), tBL("chart QA, numeric answer"), tBL("1–1000+ (integer subset)"), tBL("5,390"), tBL("타겟 안에 정답 숫자가 ‘legible’하게 보이는 케이스")],
    [tBL("MathVista"), tBL("math reasoning (diagram + chart)"), tBL("1–1000 (integer)"), tBL("385 (testmini integer)"), tBL("math-reasoning prompt → 본 논문 panel-largest cell")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: W - 1.1, colW: [2.0, 3.0, 1.5, 2.2, W - 1.1 - 2.0 - 3.0 - 1.5 - 2.2], rowH: [0.3, 0.45, 0.5, 0.45, 0.45], fontFace: BODY_FONT, fontSize: 10 });

  s.addShape("rect", { x: 0.55, y: 4.0, w: W - 1.1, h: 2.7, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("Per-dataset distance cutoff (E5d 단일 reference 모델로 검증 → cross-model 패널 적용)", { x: 0.7, y: 4.1, w: W - 1.4, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY, valign: "middle" });
  const rows2 = [
    [tH("Dataset"), tH("Cutoff rule"), tH("형태"), tH("Status")],
    [tBL("VQAv2"), tBL("|a − gt| ≤ 5  (anchor ∈ {0..9} → range overlap)"), tBL("absolute"), tBL("✅ 적용")],
    [tBL("TallyQA"), tBL("|a − gt| ≤ 5  (cap된 GT range 활용)"), tBL("absolute"), tBL("✅ 적용")],
    [tBL("ChartQA"), tBL("|a − gt| ≤ max(1, 0.10·gt)"), tBL("relative"), tBL("✅ E5d C3-validated")],
    [tBL("MathVista"), tBL("relative_s1 — ChartQA와 같은 룰"), tBL("relative"), tBL("✅ 단일-stratum 설계 (γ-α / γ-β)")],
  ];
  s.addTable(rows2, { x: 0.7, y: 4.5, w: W - 1.4, colW: [1.7, 4.4, 1.5, W - 1.4 - 1.7 - 4.4 - 1.5], rowH: [0.3, 0.34, 0.34, 0.34, 0.34], fontFace: BODY_FONT, fontSize: 10 });
  note(s, "absolute vs relative — VQAv2/TallyQA는 GT가 좁은 range (0-15)에 갇혀 있어 absolute가 자연스럽고, ChartQA/MathVista는 GT가 1-1000 wide range라 relative cutoff이 필요. 이 inductive choice가 §8.4 limitations에 disclose.");
  addPageFooter(s, 8, "§4.1 · Datasets");
}

// ---------------- Slide 9 — Model panel + sample sizes + compute ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "Model 패널 (12) · Sample size · Compute 예산", "§4.3 + §4.4 + §4.5 — 7+ open-weight 메인 + 메커니즘/mitigation/γ-β 서브패널");
  // Left: model panel
  const modelRows = [
    [tH("Model"), tH("Params"), tH("Encoder"), tH("Role")],
    [tBL("Gemma4-e4b"), tBL("4B"), tBL("SigLIP"), tBL("Main + 메커니즘 (SigLIP archetype)")],
    [tBL("LLaVA-1.5-7b"), tBL("7B"), tBL("CLIP-ViT-L/14"), tBL("메커니즘 + E4 mid-stack-cluster + Phase-A pilot")],
    [tBL("LLaVA-Next-Interleaved-7b"), tBL("7B"), tBL("CLIP-ViT-L/14"), tBL("Main + E5b/E5c reference")],
    [tBL("InternVL3-8b"), tBL("8B"), tBL("InternViT-300M"), tBL("메커니즘 + E4 mid-stack-cluster")],
    [tBL("ConvLLaVA-7b"), tBL("7B"), tBL("ConvNeXt"), tBL("메커니즘 + E4 mid-stack-cluster")],
    [tBL("FastVLM-7b"), tBL("7B"), tBL("FastViT"), tBL("메커니즘 (4번째 archetype)")],
    [tBL("Gemma3-27b-it"), tBL("27B"), tBL("SigLIP-So-400m"), tBL("Main + E5e (panel-leading)")],
    [tBL("Gemma4-31b-it"), tBL("31B"), tBL("Gemma4 multimodal"), tBL("Main")],
    [tBL("Qwen2.5-VL-7b"), tBL("7B"), tBL("Qwen-ViT"), tBL("Main + E5e (가장 anchor 저항적)")],
    [tBL("Qwen3-VL-8b Instruct"), tBL("8B"), tBL("Qwen3-VL"), tBL("Main + γ-β instruct arm")],
    [tBL("Qwen3-VL-30b-A3B"), tBL("30B (3B active)"), tBL("Qwen3-VL MoE"), tBL("Main")],
    [tBL("Qwen3-VL-8b Thinking"), tBL("8B"), tBL("Qwen3-VL"), tBL("γ-β reasoning-mode arm", { bold: true, color: TERRA })],
  ];
  s.addTable(modelRows, { x: 0.55, y: 1.5, w: 7.7, colW: [2.0, 1.0, 1.8, 2.9], rowH: [0.3, ...Array(12).fill(0.32)], fontFace: BODY_FONT, fontSize: 9.5 });
  // Right: sample size + compute
  s.addShape("rect", { x: 8.5, y: 1.5, w: 4.3, h: 2.6, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("Sample size (paper §4.4)", { x: 8.65, y: 1.55, w: 4.0, h: 0.35, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY });
  s.addText([
    { text: "메인 패널 (b/a/d):  ", options: { bold: true, fontSize: 10 } }, { text: "VQAv2 17,730 × 3 cond × 7 model = 372,330\n", options: { fontSize: 10 } },
    { text: "Strengthen-prompt:  ", options: { bold: true, fontSize: 10 } }, { text: "+372,330 (paraphrase robustness)\n", options: { fontSize: 10 } },
    { text: "E5b distance / E5c digit-mask:  ", options: { bold: true, fontSize: 10 } }, { text: "VQAv2+TallyQA × 1k × 12-18 cond = 36k\n", options: { fontSize: 10 } },
    { text: "E5e ChartQA + TallyQA + MathVista:  ", options: { bold: true, fontSize: 10 } }, { text: "~ 470k\n", options: { fontSize: 10 } },
    { text: "E1 / E1d 메커니즘:  ", options: { bold: true, fontSize: 10 } }, { text: "200 × 6 model × 7 mode × 3 cond ≈ 28,800\n", options: { fontSize: 10 } },
    { text: "E4 Phase 2 mitigation:  ", options: { bold: true, fontSize: 10 } }, { text: "17,730 × 5 cond × 3 model = 265,950\n\n", options: { fontSize: 10 } },
    { text: "Total:  ", options: { bold: true, fontSize: 11, color: TERRA } }, { text: "~ 1.6M model generations", options: { bold: true, fontSize: 11, color: TERRA } },
  ], { x: 8.65, y: 1.95, w: 4.0, h: 2.1, valign: "top", color: CHARCOAL });
  s.addShape("rect", { x: 8.5, y: 4.25, w: 4.3, h: 2.45, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("Compute envelope (§4.5)", { x: 8.65, y: 4.3, w: 4.0, h: 0.35, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
  s.addText([
    { text: "Hardware:  ", options: { bold: true, fontSize: 10 } }, { text: "8 × NVIDIA H200 (1 GPU shared with vLLM Qwen2.5-32B reserve ~55% VRAM, ~60GB usable)\n\n", options: { fontSize: 10 } },
    { text: "메모리 hygiene:  ", options: { bold: true, fontSize: 10 } }, { text: "del runner / gc.collect() / torch.cuda.empty_cache() — 8B BF16 모델 sequential load OOM 방지\n\n", options: { fontSize: 10 } },
    { text: "메인 패널 wall-clock:  ", options: { bold: true, fontSize: 10 } }, { text: "~ 6h (7 model × VQAv2)\n", options: { fontSize: 10 } },
    { text: "E4 Phase 2 wall-clock:  ", options: { bold: true, fontSize: 10 } }, { text: "~ 16h/모델 (mid-stack-cluster 3 모델)\n", options: { fontSize: 10 } },
    { text: "γ-β Thinking:  ", options: { bold: true, fontSize: 10 } }, { text: "~ 45 min (1,540 generation, 20.8 tok/s)\n\n", options: { fontSize: 10 } },
    { text: "Total project envelope:  ", options: { bold: true, fontSize: 11, color: TERRA } }, { text: "~ 5,760 GPU-hour", options: { bold: true, fontSize: 11, color: TERRA } },
  ], { x: 8.65, y: 4.65, w: 4.0, h: 2.0, valign: "top", color: CHARCOAL });
  addPageFooter(s, 9, "§4.3 / §4.4 / §4.5 · Models + Compute");
}

// ---------------- Slide 10 — §5.1 Main panel ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§5.1 메인 패널 — 7 모델 × VQAv2 number (n=17,730 each)", "C-form direction-follow는 모든 모델에서 양수 — anchoring은 패널 전반에서 검출");
  const rows = [
    [tH("Model"), tH("acc(b)"), tH("acc(d)"), tH("acc(a)"), tH("adopt(a)"), tH("df(a) C-form")],
    [tBL("Gemma4-e4b"), tB("0.553"), tB("0.505"), tB("0.541"), tHL("0.066"), tHL("0.274")],
    [tBL("LLaVA-Interleave-7b"), tB("0.619"), tB("0.577"), tB("0.576"), tHL("0.053"), tHL("0.172")],
    [tBL("Gemma3-27b-it"), tB("0.628"), tB("0.623"), tB("0.633"), tHL("0.053"), tHL("0.167")],
    [tBL("Qwen3-VL-30b-A3B"), tB("0.759"), tB("0.709"), tB("0.707"), tHL("0.039"), tHL("0.170")],
    [tBL("Qwen3-VL-8b"), tB("0.751"), tB("0.709"), tB("0.715"), tHL("0.033"), tHL("0.104")],
    [tBL("Qwen2.5-VL-7b"), tB("0.736"), tB("0.708"), tB("0.711"), tHL("0.021"), tHL("0.094")],
    [tBL("Gemma4-31b-it"), tB("0.749"), tB("0.723"), tB("0.741"), tHL("0.024"), tHL("0.085")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.5, w: 7.5, colW: [2.4, 1.0, 1.0, 1.0, 1.05, 1.05], rowH: [0.3, ...Array(7).fill(0.36)], fontFace: BODY_FONT, fontSize: 11 });

  // 2 patterns
  s.addShape("rect", { x: 8.3, y: 1.5, w: 4.5, h: 2.65, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("패턴 1 — df(a) > 0 on every model", { x: 8.45, y: 1.55, w: 4.2, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY });
  s.addText("모든 모델에서 0.085~0.274 범위. C-form metric은 baseline-relative shift을 panel 전반에서 검출. 가장 강한 모델(gemma4-e4b)이 ×3.2배 큰 effect.", { x: 8.45, y: 1.95, w: 4.2, h: 2.1, fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, valign: "top", paraSpaceAfter: 3 });

  s.addShape("rect", { x: 8.3, y: 4.25, w: 4.5, h: 2.45, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("패턴 2 — adopt(a) << df(a)", { x: 8.45, y: 4.3, w: 4.2, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
  s.addText("literal copy는 2-7%에 불과. 효과의 mass는 graded shift에 있고, 모델은 anchor 숫자를 그대로 출력하지 않음. → §1.3 ‘categorical capture가 아닌 graded pull’의 직접 증거.", { x: 8.45, y: 4.7, w: 4.2, h: 1.95, fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, valign: "top", paraSpaceAfter: 3 });

  // bottom strip
  s.addText("두 패턴은 (a − d) condition gap 수준에서도 유지 — anchor가 generic 2-image distractor 위에 ‘추가’ anchor pull을 더한다. (a − d)는 모든 모델에서 양수.", {
    x: 0.55, y: 6.0, w: W - 1.1, h: 0.6,
    fontFace: BODY_FONT, fontSize: 11.5, italic: true, color: CHARCOAL, fill: { color: CREAM }, margin: 0.1, valign: "middle",
  });
  addPageFooter(s, 10, "§5.1 · Main panel");
}

// ---------------- Slide 11 — §5.2 wrong/correct asymmetry ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§5.2 wrong-base vs correct-base 비대칭 (Phase A)", "기저 예측이 틀렸던 샘플에서 anchor 효과가 훨씬 큼 — 7/7 모델 모두");
  const rows = [
    [tH("Model"), tH("wrong − correct\n(moved-closer rate)"), tH("부호")],
    [tBL("Gemma4-e4b"), tHL("+19.6 pp"), tB("+", { bold: true, color: GREEN })],
    [tBL("Gemma3-27b-it"), tHL("+15.9 pp"), tB("+", { bold: true, color: GREEN })],
    [tBL("Qwen3-VL-30b"), tHL("+12.2 pp"), tB("+", { bold: true, color: GREEN })],
    [tBL("Gemma4-31b-it"), tHL("+8.4 pp"), tB("+", { bold: true, color: GREEN })],
    [tBL("Qwen3-VL-8b"), tHL("+8.0 pp"), tB("+", { bold: true, color: GREEN })],
    [tBL("LLaVA-Interleave"), tHL("+7.2 pp"), tB("+", { bold: true, color: GREEN })],
    [tBL("Qwen2.5-VL-7b"), tHL("+6.9 pp"), tB("+", { bold: true, color: GREEN })],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: 6.5, colW: [3.0, 2.7, 0.8], rowH: [0.5, ...Array(7).fill(0.4)], fontFace: BODY_FONT, fontSize: 11.5 });
  // Reading
  s.addShape("rect", { x: 7.3, y: 1.55, w: 5.5, h: 5.15, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("Reading", { x: 7.45, y: 1.6, w: 5.2, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  s.addText([
    { text: "1) Anchoring concentrates on items where the model would have been wrong without the anchor", options: { bold: true, fontSize: 11, color: CHARCOAL } },
    { text: " — base-prediction entropy가 가장 높은 cohort.\n\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "2) ", options: { fontSize: 11 } }, { text: "이 binary decomposition은 §6에서 answer-token entropy 4분위라는 ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "연속 confidence proxy", options: { bold: true, fontSize: 11, color: TERRA } }, { text: "로 정밀화 (mean Q4-Q1 = +15.2 pp on direction-follow).\n\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "3) 인지과학 대응:", options: { bold: true, fontSize: 11, color: NAVY } }, { text: " Mussweiler-Strack의 selective accessibility model — anchor가 search-space의 candidate로 들어가서, 모델이 자기 prior에 의존하는 정도에 비례해 답에 ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "blend", options: { italic: true, fontSize: 11, color: TERRA } }, { text: ".\n\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "4) Novelty:", options: { bold: true, fontSize: 11, color: NAVY } }, { text: " 선행 anchoring 문헌(LLM, VLM 모두) 중 ‘base correctness별 분리’를 한 사례 없음 — A1이 본 논문의 가장 강한 intellectual hook.", options: { fontSize: 11, color: CHARCOAL } },
  ], { x: 7.45, y: 2.0, w: 5.2, h: 4.65, valign: "top", paraSpaceAfter: 0 });
  addPageFooter(s, 11, "§5.2 · Asymmetry");
}

// ---------------- Slide 12 — §5.3 distance × plausibility ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§5.3 거리별 anchor 효과 (E5b) — plausibility window", "|a − gt|을 5단계 stratum으로 분할 · llava-interleave-7b + qwen2.5-vl-7b × VQAv2 + TallyQA");
  const rows = [
    [tH("Stratum"), tH("|a − gt|"), tH("VQAv2 llava"), tH("VQAv2 qwen2.5"), tH("TallyQA llava"), tH("TallyQA qwen2.5")],
    [tBL("S1"), tBL("[0,1]"), tHL("0.130"), tHL("0.070"), tHL("0.092"), tHL("0.033")],
    [tBL("S2"), tBL("[2,5]"), tB("0.032"), tB("0.014"), tB("0.006"), tB("0.015")],
    [tBL("S3"), tBL("[6,30]"), tB("0.010"), tB("0.003"), tB("0.003"), tB("0.000")],
    [tBL("S4"), tBL("[31,300]"), tB("0.010"), tB("0.003"), tB("0.000"), tB("0.000")],
    [tBL("S5"), tBL("[301,∞)"), tB("0.003"), tB("0.003"), tB("0.000"), tB("0.000")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: 7.4, colW: [0.85, 1.3, 1.3, 1.6, 1.2, 1.15], rowH: [0.55, ...Array(5).fill(0.36)], fontFace: BODY_FONT, fontSize: 10.5 });

  // figure
  const figPath = path.join(REPO, "docs/figures/E5b_adopt_cond_curve.png");
  if (fs.existsSync(figPath)) {
    s.addImage({ path: figPath, x: 8.05, y: 1.55, w: 4.75, h: 3.0 });
  } else {
    s.addShape("rect", { x: 8.05, y: 1.55, w: 4.75, h: 3.0, fill: { color: CREAM }, line: { color: MUTED, width: 0.5 } });
  }
  s.addText("E5b S1 peak / S5 floor (wrong-base 빨강 vs correct-base 파랑)", { x: 8.05, y: 4.55, w: 4.75, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, italic: true, color: MUTED, align: "center" });

  // Reading
  s.addShape("rect", { x: 0.55, y: 4.05, w: 7.4, h: 2.65, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("두 gate가 동시에 작동", { x: 0.7, y: 4.1, w: 7.1, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY });
  s.addText([
    { text: "①  Uncertainty gate:", options: { bold: true, fontSize: 11, color: TERRA } }, { text: " correct-base에서는 anchor가 끌어오지 못함 (S1에서도 ≤ 0.10).\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "②  Plausibility gate:", options: { bold: true, fontSize: 11, color: TERRA } }, { text: " wrong-base여도 |a-gt| > 5 (S3+) 이면 anchor 거부.\n\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "Cross-dataset robustness:", options: { bold: true, fontSize: 11, color: NAVY } }, { text: " VQAv2 base accuracy 0.62 / 0.81 vs TallyQA 0.21 / 0.24 — base-accuracy 차이가 패턴을 바꾸지 않음. 두 architecturally distinct 모델 (llava-CLIP-ViT vs qwen2.5-Qwen-ViT) 모두에서 S1 peak / S3-S5 floor 형태가 동일. ", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "TallyQA S4/S5는 정확히 0.000 — implausible anchor는 완전 거부됨.", options: { italic: true, fontSize: 10.5, color: TERRA } },
  ], { x: 0.7, y: 4.5, w: 7.1, h: 2.1, valign: "top", paraSpaceAfter: 0 });

  s.addText("두 gate의 product가 signature — 두 모델 family / 두 dataset에 걸쳐 재현됨", { x: 8.05, y: 5.4, w: 4.75, h: 1.3, fontFace: BODY_FONT, fontSize: 11, italic: true, color: NAVY, fill: { color: PALE_TERRA }, margin: 0.15, valign: "middle" });
  addPageFooter(s, 12, "§5.3 · E5b distance");
}

// ---------------- Slide 13 — §5.4 digit-pixel causality (E5c) ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§5.4 digit-pixel 인과성 (E5c) — anchor vs masked + (1,3,4) control", "wrong-base × S1 paired adopt: 같은 scene을 inpaint한 m을 빼면 digit pixels의 순수 기여");
  const rows = [
    [tH("Dataset"), tH("Model"), tH("anchor adopt_cond"), tH("masked adopt_cond"), tH("a − m gap (digit-pixel 기여)")],
    [tBL("VQAv2"), tBL("llava-interleave-7b"), tB("0.129"), tB("0.068"), tHL("+6.1 pp")],
    [tBL("VQAv2"), tBL("qwen2.5-vl-7b"), tB("0.070"), tB("0.066"), tB("+0.4 pp")],
    [tBL("TallyQA"), tBL("llava-interleave-7b"), tB("0.110"), tB("0.084"), tHL("+2.5 pp")],
    [tBL("TallyQA"), tBL("qwen2.5-vl-7b"), tB("0.033"), tB("0.037"), tB("−0.5 pp")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: 7.5, colW: [1.3, 2.4, 1.6, 1.6, 0.6 + 1.1], rowH: [0.5, ...Array(4).fill(0.4)], fontFace: BODY_FONT, fontSize: 10.5 });

  // figure
  const figPath = path.join(REPO, "docs/figures/E5c_anchor_vs_masked_adopt.png");
  if (fs.existsSync(figPath)) {
    s.addImage({ path: figPath, x: 8.2, y: 1.55, w: 4.6, h: 2.7 });
  }
  s.addText("anchor (●) vs masked (■) × wrong/correct base — llava-interleave", { x: 8.2, y: 4.25, w: 4.6, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, italic: true, color: MUTED, align: "center" });

  // Reading
  s.addShape("rect", { x: 0.55, y: 3.85, w: 7.5, h: 2.85, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("Reading — direction-consistent across models", { x: 0.7, y: 3.9, w: 7.2, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY });
  s.addText([
    { text: "•  llava-interleave: ", options: { bold: true, fontSize: 11, color: TERRA } }, { text: "anchor > masked 양수 (+6.1 / +2.5 pp). digit pixel이 paired adoption의 인과 경로.\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "•  qwen2.5-vl: ", options: { bold: true, fontSize: 11, color: TERRA } }, { text: "양 arm 모두 noise floor — §5.1 main-panel ranking에서 가장 anchor-resistant 모델 (df(a)=0.094)이라 E5c도 floor.\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "•  Reading:", options: { bold: true, fontSize: 11, color: TERRA } }, { text: " ‘pull이 검출 가능한 곳에서는 gap이 양수, pull이 floor면 gap도 floor’ — direction-consistent. gemma3-27b-it E5c VQAv2 cell pending (~5-6h H200) → mid-panel 모델이 llava-style인지 qwen-style인지 결정.\n\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "(1,3,4) comparison:", options: { bold: true, fontSize: 11, color: NAVY } }, { text: " masked vs neutral의 acc_drop 차이는 1-2 pp뿐 → ", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "anchor scene 배경은 generic 2-image distractor 이상의 일을 하지 않음.", options: { italic: true, fontSize: 11, color: TERRA } },
  ], { x: 0.7, y: 4.3, w: 7.2, h: 2.3, valign: "top", paraSpaceAfter: 0 });

  s.addText("⇒ 위치-가능한 곳에서는 digit pixels이 인과 pathway. plausibility window 내 wrong-base S1 cell에서 검증.", {
    x: 8.2, y: 5.05, w: 4.6, h: 1.6, fontFace: BODY_FONT, fontSize: 11, italic: true, color: NAVY, fill: { color: PALE_TERRA }, margin: 0.15, valign: "middle"
  });
  addPageFooter(s, 13, "§5.4 · Digit-pixel");
}

// ---------------- Slide 14 — §5.5 cross-dataset E5e ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§5.5 Cross-dataset 확장 (E5e)", "S1 단일-stratum × b/a/m/d × 3 모델 × ChartQA + TallyQA + MathVista — a > m 패턴이 dataset 전반에서 재현");
  const rows = [
    [tH("Dataset"), tH("Model"), tH("adopt(a)"), tH("adopt(m)"), tH("df(a) C-form"), tH("df(m) C-form")],
    [tBL("ChartQA"), tBL("gemma3-27b-it"), tB("0.037"), tB("0.022"), tB("0.096"), tB("0.079")],
    [tBL("ChartQA"), tBL("llava-interleave"), tB("0.028"), tB("0.009"), tB("0.152"), tB("0.115")],
    [tBL("ChartQA"), tBL("qwen2.5-vl-7b"), tB("0.017"), tB("0.013"), tB("0.051"), tB("0.046")],
    [tBL("TallyQA"), tBL("gemma3-27b-it"), tHL("0.027"), tB("0.016"), tHL("0.073"), tB("0.060")],
    [tBL("TallyQA"), tBL("llava-interleave"), tB("0.026"), tB("0.014"), tB("0.066"), tB("0.056")],
    [tBL("TallyQA"), tBL("qwen2.5-vl-7b"), tB("0.011"), tB("0.011"), tB("0.029"), tB("0.030")],
    [tBL("MathVista"), tBL("gemma3-27b-it"), tHL("0.176"), tB("0.047"), tHL("0.216"), tB("0.134")],
    [tBL("MathVista"), tBL("llava-interleave"), tB("0.066"), tB("0.030"), tB("0.205"), tB("0.125")],
    [tBL("MathVista"), tBL("qwen2.5-vl-7b"), tB("0.020"), tB("0.008"), tB("0.072"), tB("0.041")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.5, w: 8.0, colW: [1.2, 2.4, 1.1, 1.1, 1.1, 1.1], rowH: [0.3, ...Array(9).fill(0.32)], fontFace: BODY_FONT, fontSize: 10 });
  // figure
  const figPath = path.join(REPO, "docs/figures/paper_E5e_mathvista_bars.png");
  if (fs.existsSync(figPath)) {
    s.addImage({ path: figPath, x: 8.7, y: 1.5, w: 4.1, h: 3.0 });
  }
  // 3 observations
  s.addShape("rect", { x: 8.7, y: 4.6, w: 4.1, h: 2.1, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("3 관찰", { x: 8.85, y: 4.65, w: 3.8, h: 0.3, fontFace: TITLE_FONT, fontSize: 11, bold: true, color: NAVY });
  s.addText([
    { text: "1) 3/3 모델이 모든 dataset에서 a > m 보존 — digit-pixel causality 일반화.\n\n", options: { fontSize: 9.5, color: CHARCOAL } },
    { text: "2) ", options: { fontSize: 9.5 } }, { text: "MathVista가 panel-largest cell.", options: { bold: true, fontSize: 9.5, color: TERRA } }, { text: " gemma3-27b-it adopt 0.176 / df 0.216 (전체-base S1).\n\n", options: { fontSize: 9.5, color: CHARCOAL } },
    { text: "3) plausibility-window 패턴은 per-dataset cutoff (E5d)로 양적 검증.", options: { fontSize: 9.5, color: CHARCOAL } },
  ], { x: 8.85, y: 4.95, w: 3.8, h: 1.7, valign: "top" });
  // bottom: TallyQA gemma3 highlight
  s.addText([
    { text: "TallyQA × gemma3-27b-it (n=38,245, full integer subset, 2026-04-29 landed):", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: " adopt(a)=0.027, df(a)=0.073 — 동시 panel-leading. inference 2026-04-28 23:28 완료 (예상 30-35h budget 대비 단축), C-form re-aggregate 2026-04-29.", options: { fontSize: 11, color: CHARCOAL } },
  ], { x: 0.55, y: 5.6, w: 8.0, h: 1.1, fill: { color: CREAM }, margin: 0.15, valign: "middle" });
  addPageFooter(s, 14, "§5.5 · E5e cross-dataset");
}

// ---------------- Slide 15 — MathVista deep dive + §5.6 three-gate conjunction ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§5.5 MathVista — panel-largest cell + §5.6 three-gate signature", "MathVista가 본 논문에서 가장 anchor pull이 큰 dataset · 3-gate conjunction이 cross-dataset signature");
  // MathVista deep dive
  s.addShape("rect", { x: 0.55, y: 1.55, w: 6.1, h: 5.15, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 1 } });
  s.addText("MathVista wrong-base × S1 (panel-largest cell)", { x: 0.7, y: 1.65, w: 5.8, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA });
  const rows = [
    [tH("Model"), tH("adopt(a)"), tH("adopt(m)"), tH("df(a)"), tH("a − m gap")],
    [tBL("gemma3-27b-it"), tHL("0.230"), tB("0.051"), tHL("0.332"), tHL("+17.9 pp")],
    [tBL("llava-interleave"), tB("0.119"), tB("0.051"), tB("0.299"), tB("+6.8 pp")],
    [tBL("qwen2.5-vl-7b"), tB("0.040"), tB("0.014"), tB("0.105"), tB("+2.6 pp")],
  ];
  s.addTable(rows, { x: 0.7, y: 2.1, w: 5.8, colW: [1.7, 1.0, 1.0, 1.05, 1.05], rowH: [0.3, ...Array(3).fill(0.36)], fontFace: BODY_FONT, fontSize: 10 });
  s.addText([
    { text: "Why MathVista이 가장 큰가? — 추정 원인:\n\n", options: { bold: true, fontSize: 11, color: NAVY } },
    { text: "(a) ", options: { fontSize: 10.5 } }, { text: "math-reasoning prompt이 high model capability에서도 wrong-base entropy를 높임", options: { fontSize: 10.5, color: CHARCOAL } }, { text: "\n(b) ", options: { fontSize: 10.5 } },
    { text: "gemma3-27b의 SigLIP-So-400m encoder는 typographic susceptibility로 알려져 있음 (§7.1 SigLIP-Gemma early peak).\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "\n— 둘 다 follow-up에서 testable.", options: { italic: true, fontSize: 10.5, color: TERRA } },
  ], { x: 0.7, y: 3.5, w: 5.8, h: 3.1, valign: "top", paraSpaceAfter: 0 });

  // 3-gate signature
  s.addShape("rect", { x: 6.85, y: 1.55, w: 5.95, h: 5.15, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 1 } });
  s.addText("§5.6 — 3-gate conjunction이 signature", { x: 7.0, y: 1.65, w: 5.65, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });

  const gates = [
    { tag: "①", title: "Uncertainty gate", body: "anchor pull이 ‘base가 wrong이었을’ cohort에 집중 (Phase A +6.9~+19.6 pp wrong-correct on direction-follow)." },
    { tag: "②", title: "Plausibility gate", body: "anchor pull이 approximate-magnitude window 안에서만 작동 (S1 peak / S5 floor on 두 dataset E5b, all 3 dataset E5d cutoff 검증)." },
    { tag: "③", title: "Digit-pixel gate", body: "plausibility window 내 operative cause는 digit pixels 자체 (E5c paired adopt gap +6.1 pp on VQAv2 wrong-base S1, +17.9 pp on MathVista wrong-base S1)." },
  ];
  let yy = 2.15;
  gates.forEach((g) => {
    s.addShape("ellipse", { x: 7.0, y: yy + 0.05, w: 0.5, h: 0.5, fill: { color: NAVY }, line: { color: NAVY, width: 0 } });
    s.addText(g.tag, { x: 7.0, y: yy + 0.05, w: 0.5, h: 0.5, fontFace: TITLE_FONT, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(g.title, { x: 7.6, y: yy, w: 5.05, h: 0.35, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY, valign: "middle" });
    s.addText(g.body, { x: 7.6, y: yy + 0.35, w: 5.05, h: 1.1, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
    yy += 1.55;
  });
  addPageFooter(s, 15, "§5.5–§5.6 · 3-gate signature");
}

// ---------------- Slide 16 — §6 confidence proxies + monotone gradient ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§6 confidence-modulated anchoring (L1 분석) — 3 proxy 비교", "wrong/correct 이분법은 ‘answer-token entropy 4분위 monotone’의 거친 투영");
  // proxy table
  const rows = [
    [tH("Confidence proxy"), tH("mean(adopt Q4 − Q1)"), tH("mean(df Q4 − Q1)"), tH("fully monotone Q1<Q4 cells")],
    [tHL("entropy_top_k", { fill: { color: PALE_TERRA } }), tHL("+0.044"), tHL("+0.152"), tHL("23 / 35 (df), 10 / 35 (adopt)")],
    [tBL("softmax_top1_prob"), tB("+0.036"), tB("+0.108"), tB("15 / 34 (df), 5 / 34 (adopt)")],
    [tBL("top1_minus_top2_margin"), tB("+0.017"), tB("+0.013"), tB("8 / 34 (df), 7 / 34 (adopt)")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: W - 1.1, colW: [3.2, 2.6, 2.6, W - 1.1 - 3.2 - 2.6 - 2.6], rowH: [0.4, ...Array(3).fill(0.42)], fontFace: BODY_FONT, fontSize: 11 });
  s.addText("Q4 − Q1: 가장-uncertain 4분위 − 가장-confident 4분위. positive면 anchor pull이 less-confident base prediction에 더 큼.", { x: 0.55, y: 3.4, w: W - 1.1, h: 0.4, fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED, valign: "middle" });

  // worked example table
  const ex = [
    [tH("quartile"), tH("base correctness mean(em_b)"), tH("anchor adopt"), tH("direction-follow C-form")],
    [tBL("Q1 (most confident)"), tB("0.92"), tB("0.043"), tB("0.032")],
    [tBL("Q2"), tB("0.72"), tB("0.084"), tB("0.080")],
    [tBL("Q3"), tB("0.42"), tB("0.149"), tB("0.137")],
    [tBL("Q4 (least confident)"), tHL("0.34"), tHL("0.172"), tHL("0.210")],
    [tBL("Δ (Q4 − Q1)"), tB("−0.58", { bold: true }), tHL("+0.130"), tHL("+0.178")],
  ];
  s.addTable(ex, { x: 0.55, y: 3.85, w: 7.5, colW: [2.6, 2.0, 1.4, 1.5], rowH: [0.3, ...Array(5).fill(0.34)], fontFace: BODY_FONT, fontSize: 10 });
  s.addText("Worked example — E5c VQAv2 wrong-base S1 on llava-interleave-7b (Phase-A의 +7.2 pp binary gap에 대응되는 cell)", { x: 0.55, y: 5.85, w: 7.5, h: 0.3, fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED, align: "center" });

  // figure
  const figPath = path.join(REPO, "docs/figures/paper_L1_confidence_quartile.png");
  if (fs.existsSync(figPath)) {
    s.addImage({ path: figPath, x: 8.3, y: 3.85, w: 4.5, h: 2.6 });
  }
  s.addText("entropy_top_k Q1→Q4 monotone increase", { x: 8.3, y: 6.45, w: 4.5, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, italic: true, color: MUTED, align: "center" });
  addPageFooter(s, 16, "§6 · Confidence proxies");
}

// ---------------- Slide 17 — §6.3 selective accessibility reading + γ-β confirmation ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§6.3 Reading — graded blending (selective accessibility) + γ-β confirms", "‘categorical capture under threshold’ 가설을 reject하고, Mussweiler-Strack의 graded-blending 가설과 일치");
  // Left: rejected vs accepted reading
  s.addShape("rect", { x: 0.55, y: 1.55, w: 6.0, h: 5.15, fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 } });
  s.addText("두 가설의 비교", { x: 0.7, y: 1.65, w: 5.7, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  s.addShape("rect", { x: 0.7, y: 2.1, w: 5.7, h: 1.45, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("✗ Categorical capture (rejected)", { x: 0.85, y: 2.15, w: 5.4, h: 0.3, fontFace: TITLE_FONT, fontSize: 11, bold: true, color: TERRA });
  s.addText("Anchor가 단순히 prediction을 ‘교체’할 뿐이라면 adopt(Q4) >> adopt(Q3) ~ adopt(Q2) ~ adopt(Q1) 형태의 step이 보여야 함. → 데이터에서 step 미관측.", {
    x: 0.85, y: 2.45, w: 5.4, h: 1.1, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top",
  });
  s.addShape("rect", { x: 0.7, y: 3.7, w: 5.7, h: 2.95, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("✓ Graded blending (accepted) — Mussweiler-Strack", { x: 0.85, y: 3.75, w: 5.4, h: 0.3, fontFace: TITLE_FONT, fontSize: 11, bold: true, color: NAVY });
  s.addText([
    { text: "Q1→Q4 adopt 단조 증가 0.043 → 0.084 → 0.149 → 0.172 (gradient).\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "Q1→Q4 df 0.032 → 0.080 → 0.137 → 0.210 (더 깨끗한 gradient).\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "Reading: ", options: { bold: true, fontSize: 10.5, color: NAVY } }, { text: "anchor가 search-space의 candidate로 들어가서, 모델이 자기 prior에 의존하는 정도(=base entropy)에 비례해 답에 ", options: { fontSize: 10, color: CHARCOAL } }, { text: "blended", options: { italic: true, fontSize: 10, color: TERRA } }, { text: "됨.\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "23 / 35 (model × dataset × stratum) cell이 fully monotone — 비-monotone cell은 (a) 작은 denominator이거나 (b) noise floor에 있는 cell.", options: { fontSize: 10, color: CHARCOAL } },
  ], { x: 0.85, y: 4.05, w: 5.4, h: 2.6, valign: "top" });

  // Right: γ-β cross-cell confirmation
  s.addShape("rect", { x: 6.75, y: 1.55, w: 6.05, h: 5.15, fill: { color: "FFFFFF" }, line: { color: TERRA, width: 1 } });
  s.addText("§6.5 γ-β confirmation — gradient가 cell 사이에서도 같은 방향", { x: 6.9, y: 1.65, w: 5.75, h: 0.45, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
  s.addText("Qwen3-VL-8b-Thinking on MathVista vs Instruct: Thinking이 더 entropy 높고 더 anchor에 끌림 (acc(b) ↓, adopt(a) ↑, df(a) ↑).", { x: 6.9, y: 2.15, w: 5.75, h: 0.6, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top" });
  const rows = [
    [tH("Quantity"), tH("Thinking"), tH("Instruct"), tH("Δ")],
    [tBL("acc(b)"), tB("0.196"), tB("0.216"), tB("−0.020", { color: GREEN })],
    [tBL("adopt(a)"), tHL("0.117"), tB("0.074"), tHL("×1.6")],
    [tBL("df(a) C-form"), tHL("0.291"), tB("0.102"), tHL("×2.9")],
  ];
  s.addTable(rows, { x: 6.9, y: 2.85, w: 5.75, colW: [2.0, 1.4, 1.4, 0.95], rowH: [0.32, ...Array(3).fill(0.4)], fontFace: BODY_FONT, fontSize: 10 });
  s.addText([
    { text: "Quartile gradient ", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "within ", options: { italic: true, fontSize: 10.5, color: NAVY } },
    { text: "each cell이 보여준 것과 같은 방향이 ", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "across ", options: { italic: true, fontSize: 10.5, color: NAVY } },
    { text: "model variants에서도 재현됨 — 기저 entropy가 더 높은 모델이 더 큰 anchor pull. 단일 dataset/단일 pair existence proof.", options: { fontSize: 10.5, color: CHARCOAL } },
  ], { x: 6.9, y: 4.45, w: 5.75, h: 1.4, valign: "top" });
  s.addText("§7-§8: 왜 reasoning이 entropy를 ‘낮추지 않고 오히려 높이는가’의 후속 질문 → γ-β scale-up이 future work.", { x: 6.9, y: 5.95, w: 5.75, h: 0.7, fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED, valign: "middle", fill: { color: CREAM }, margin: 0.1 });
  addPageFooter(s, 17, "§6.3 / §6.5 · Reading + γ-β");
}

// ---------------- Slide 18 — §7.1 anchor attention mass + sub-findings ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§7.1 anchor attention mass — base finding + 2 sub-findings (E1)", "n=200 stratified · 4 base-experiment 모델에서 mean +0.004~+0.007, CI excludes 0 — anchor attention은 baseline-level artifact 아님");
  // Big finding card
  s.addShape("rect", { x: 0.55, y: 1.55, w: W - 1.1, h: 1.5, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 1 } });
  s.addText("Base finding — anchor attention mass > 0", { x: 0.7, y: 1.65, w: W - 1.4, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  s.addText("각 generation step에서 anchor image-token span으로 가는 attention weight를 layer-averaged 측정. 6 panel 모델 (gemma4-e4b, qwen2.5-vl-7b, llava-1.5-7b, internvl3-8b, convllava-7b, fastvlm-7b) × n=200 stratified. 4/4 base-experiment 모델에서 mean +0.004~+0.007 (CI excludes 0). → §5/§6의 행동 효과는 generation 시 실제 attention-mass shift을 동반.", {
    x: 0.7, y: 2.05, w: W - 1.4, h: 0.95, fontFace: BODY_FONT, fontSize: 10.5, color: CHARCOAL, valign: "top",
  });
  // 2 sub-findings
  s.addShape("rect", { x: 0.55, y: 3.25, w: (W - 1.4) / 2, h: 3.45, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 1 } });
  s.addText("Sub-finding 1 — wrong>correct attention asymmetry FALSIFIED", { x: 0.7, y: 3.35, w: (W - 1.4) / 2 - 0.3, h: 0.55, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
  s.addText([
    { text: "Pre-spec hypothesis: Phase-A의 wrong/correct 비대칭이 attention level에도 보일 것 (anchor mass가 uncertain item에 shift up)\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "관측: 4/4 (later 6/6) 모델에서 wrong vs correct 사이에 anchor attention mass 차이 무 — 행동 차이는 attention magnitude가 아니라 ", options: { fontSize: 10, color: CHARCOAL } },
    { text: "downstream blending의 차이", options: { bold: true, fontSize: 10, color: TERRA } }, { text: "에서 옴.\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "→ ", options: { fontSize: 10 } }, { text: "uncertainty modulation은 ‘얼마나 attention을 주는가’가 아니라 ‘어떻게 prediction에 weight되는가’로 작동", options: { italic: true, bold: true, fontSize: 10.5, color: NAVY } }, { text: ". §6 blended-into-prior 가설과 일치.", options: { fontSize: 10, color: CHARCOAL } },
  ], { x: 0.7, y: 3.95, w: (W - 1.4) / 2 - 0.3, h: 2.65, valign: "top" });

  s.addShape("rect", { x: 0.55 + (W - 1.4) / 2 + 0.2, y: 3.25, w: (W - 1.4) / 2, h: 3.45, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 1 } });
  s.addText("Sub-finding 2 — susceptibility tracks attention 3/4, INVERTS on Gemma-SigLIP", { x: 0.7 + (W - 1.4) / 2 + 0.2, y: 3.35, w: (W - 1.4) / 2 - 0.3, h: 0.55, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY });
  s.addText([
    { text: "A7의 cross-model item-susceptibility (top-decile vs bottom-decile)이 ", options: { fontSize: 10, color: CHARCOAL } },
    { text: "answer-step anchor attention 3/4 모델에서 일치 ", options: { bold: true, fontSize: 10, color: GREEN } },
    { text: "(highly-susceptible item에서 attention mass↑).\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "Gemma-SigLIP에서는 inverts ", options: { bold: true, fontSize: 10, color: TERRA } },
    { text: "— anchor attention이 step 0 (image processing)에 집중되고 answer-step에서는 미미. SigLIP의 typographic-attack inheritance 문헌과 일치.\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "→ depth-axis가 architecture-axis를 대체하는 framing의 동기 (§7.2 H3 retired).", options: { italic: true, fontSize: 10.5, color: NAVY } },
  ], { x: 0.7 + (W - 1.4) / 2 + 0.2, y: 3.95, w: (W - 1.4) / 2 - 0.3, h: 2.65, valign: "top" });

  addPageFooter(s, 18, "§7.1 · Attention mass");
}

// ---------------- Slide 19 — §7.2 per-layer 4 archetypes ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§7.2 per-layer 4 archetypes (E1b) — anchor mass concentrates at one peak per family", "anchor mass가 layer-averaged 대비 최대 ×3배 — encoder family별 ‘peak layer’ + (H3) encoder architecture is NOT predictive");
  const rows = [
    [tH("Archetype"), tH("Reference model"), tH("Peak layer"), tH("δ at peak"), tH("Budget source")],
    [tBL("SigLIP-Gemma early"), tBL("gemma4-e4b"), tHL("L5 / 42 (12% depth)"), tB("+0.050"), tB("text-stealing")],
    [tBL("Mid-stack cluster"), tBL("llava-1.5-7b (CLIP-ViT)"), tHL("L16 / 32"), tB("+0.019"), tB("text-stealing")],
    [tBL("Mid-stack cluster"), tBL("convllava-7b (ConvNeXt)"), tHL("L16 / 32"), tB("+0.022"), tB("text-stealing")],
    [tBL("Mid-stack cluster"), tBL("internvl3-8b (InternViT)"), tHL("L14 / 28"), tB("+0.019"), tB("text-stealing")],
    [tBL("Qwen-ViT late"), tBL("qwen2.5-vl-7b"), tHL("L22 / 28 (82%)"), tB("+0.015"), tB("target-stealing")],
    [tBL("FastVLM late"), tBL("fastvlm-7b"), tHL("L22"), tB("+0.047"), tB("text-stealing")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: W - 1.1, colW: [2.6, 3.0, 2.4, 1.5, W - 1.1 - 2.6 - 3.0 - 2.4 - 1.5], rowH: [0.35, ...Array(6).fill(0.4)], fontFace: BODY_FONT, fontSize: 11 });

  s.addShape("rect", { x: 0.55, y: 4.45, w: 6.0, h: 2.25, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("Mid-stack cluster — 가장 leverage 큰 mitigation target", { x: 0.7, y: 4.55, w: 5.7, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
  s.addText("아키텍처적으로 다른 3 encoder (CLIP-ViT, ConvNeXt, InternViT)가 같은 mid-stack layer-depth signature로 수렴. 이 cluster가 §7.4 free-lunch mitigation의 panel.", { x: 0.7, y: 4.95, w: 5.7, h: 1.65, fontFace: BODY_FONT, fontSize: 10.5, color: CHARCOAL, valign: "top", paraSpaceAfter: 3 });

  s.addShape("rect", { x: 6.85, y: 4.45, w: 5.95, h: 2.25, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("(H3 retired) encoder architecture per se ≠ predictive", { x: 7.0, y: 4.55, w: 5.65, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY });
  s.addText("3 architecturally different encoder가 같은 mid-stack signature로 수렴 → encoder-architecture per-se는 anchoring susceptibility를 예측하지 못함. depth-axis로 framing 교체.", { x: 7.0, y: 4.95, w: 5.65, h: 1.65, fontFace: BODY_FONT, fontSize: 10.5, color: CHARCOAL, valign: "top", paraSpaceAfter: 3 });

  addPageFooter(s, 19, "§7.2 · Per-layer 4 archetypes");
}

// ---------------- Slide 20 — §7.2.1 E1-patch digit-pixel concentration (NEW!) ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§7.2.1 E1-patch — anchor 안 ‘digit pixels’에 attention 집중 (4-model panel)", "where in the anchor: digit bbox(JSON, anchor−mask diff)로 layer별 digit-fraction-of-anchor-mass 측정");
  const rows = [
    [tH("Model"), tH("encoder archetype"), tH("digit/anchor at peak"), tH("peak L"), tH("concentration above fair share")],
    [tBL("gemma4-e4b"), tBL("SigLIP-Gemma early"), tHL("0.631"), tBL("L9 / 42"), tHL("+0.404 pp")],
    [tBL("convllava-7b"), tBL("CLIP-ConvNeXt mid-stack"), tHL("0.552"), tBL("L7 / 32"), tHL("+0.325 pp")],
    [tBL("fastvlm-7b"), tBL("FastViT late"), tHL("0.531"), tBL("L4 / 28"), tHL("+0.304 pp")],
    [tBL("llava-1.5-7b"), tBL("CLIP-ViT mid-stack"), tHL("0.468"), tBL("L7 / 32"), tHL("+0.241 pp")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.5, w: W - 1.1, colW: [1.8, 3.4, 2.4, 1.6, W - 1.1 - 1.8 - 3.4 - 2.4 - 1.6], rowH: [0.42, ...Array(4).fill(0.4)], fontFace: BODY_FONT, fontSize: 11 });
  s.addText("fair-share baseline ~ 0.227 — 128 anchor에서 digit bbox의 mean area 비율. 균일 attention이라면 digit-fraction이 0.227 정도여야 함.", { x: 0.55, y: 3.65, w: W - 1.1, h: 0.4, fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED, valign: "middle" });

  // 3 profile shapes
  s.addText("3가지 profile shape — anchor 안 attention의 layer-wise 진행", { x: 0.55, y: 4.15, w: W - 1.1, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  const shapes = [
    { color: TERRA, head: "1) Globally digit-concentrated", model: "gemma4-e4b (SigLIP)", body: "모든 layer에서 0.50-0.63 — SigLIP이 LLM stack 전반에서 digit을 ‘본다’. 초기 SigLIP typographic-attack inheritance 문헌과 일치." },
    { color: NAVY, head: "2) Peaked mid-early then decay", model: "llava-1.5-7b + convllava-7b (mid-stack)", body: "L7 peak (0.47 / 0.55) → L15-17에서 fair share 회귀 → L29-31에서 sub-fair. 두 architecturally distinct 인코더가 같은 shape로 수렴 — mid-stack-cluster의 depth signature." },
    { color: GREEN, head: "3) Sharp early peak with sustained mid plateau", model: "fastvlm-7b (FastViT)", body: "L4 peak (0.53) → L6에서 ~0.32로 drop → L12까지 0.35-0.45 plateau 유지 → L18에서 fair share 회귀. early-stack에서 peaking하되 mid-stack까지 신호가 ‘지속’." },
  ];
  const sw = (W - 1.4) / 3;
  shapes.forEach((sh, i) => {
    const sx = 0.55 + i * (sw + 0.2);
    s.addShape("rect", { x: sx, y: 4.6, w: sw, h: 2.1, fill: { color: "FFFFFF" }, line: { color: sh.color, width: 1.25 } });
    s.addShape("rect", { x: sx, y: 4.6, w: sw, h: 0.4, fill: { color: sh.color }, line: { color: sh.color, width: 0 } });
    s.addText(sh.head, { x: sx + 0.1, y: 4.6, w: sw - 0.2, h: 0.4, fontFace: TITLE_FONT, fontSize: 10.5, bold: true, color: "FFFFFF", valign: "middle" });
    s.addText(sh.model, { x: sx + 0.1, y: 5.05, w: sw - 0.2, h: 0.3, fontFace: TITLE_FONT, fontSize: 10, italic: true, color: sh.color });
    s.addText(sh.body, { x: sx + 0.1, y: 5.35, w: sw - 0.2, h: 1.3, fontFace: BODY_FONT, fontSize: 9.5, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
  });
  note(s, "비-square archetypes (internvl3-8b multi-tile, qwen2.5-vl-7b 17×23 grid)는 per-encoder bbox-to-token mapping 작업이 필요해 deferred — full 6-model coverage + masked-arm causal control은 §8 future work.   note: digit-attention concentration peak (L4-L9)이 §7.2의 total-anchor-mass peak (L14-L22)보다 ‘earlier’.   E1-patch와 E1b는 complementary — where in the anchor (E1-patch) vs which layers (E1b).");
  addPageFooter(s, 20, "§7.2.1 · E1-patch digit concentration");
}

// ---------------- Slide 21 — §7.3 causal ablation E1d ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§7.3 causal ablation (E1d) — single-layer null, upper-half가 작동", "anchor span을 layer-set별로 mask해서 direction-follow 변화 측정 (n=200, 6 모델 × 6 모드)");
  const rows = [
    [tH("Mode"), tH("Δ direction-follow (panel)"), tH("결과")],
    [tBL("ablate_peak (E1b 헤드라인 layer 단독)"), tBL("|Δ df| ≤ 2.0 pp; 모든 CI baseline overlap"), tHL("Null on 6/6 — single-layer 인과적 null", { color: TERRA })],
    [tBL("ablate_layer0 (non-peak control)"), tBL("Δ df ∈ [−2.7, +0.5] pp"), tHL("Null on 6/6", { color: TERRA })],
    [tBL("ablate_upper_half (mitigation candidate)"), tBL("−4.0 ~ −10.5 pp"), tHL("6/6 모델에서 reduce, fluency-clean on 4/6 (mid-stack + Qwen)", { color: GREEN })],
    [tBL("ablate_all"), tBL("−9.6 ~ −24.5 pp"), tBL("3/6 모델 fluency 붕괴 (mean-distance 4-6배 또는 1000배)", { color: TERRA })],
    [tBL("ablate_lower_half (diagnostic)"), tBL("Heterogeneous"), tBL("3/6 BACKFIRE, 1/6 reduce, 2/6 flat", { color: TERRA })],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: W - 1.1, colW: [3.5, 3.7, W - 1.1 - 3.5 - 3.7], rowH: [0.4, ...Array(5).fill(0.55)], fontFace: BODY_FONT, fontSize: 11 });
  s.addShape("rect", { x: 0.55, y: 5.0, w: W - 1.1, h: 1.7, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("Surprise — single-layer ablation이 인과적으로 null", { x: 0.7, y: 5.05, w: W - 1.4, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  s.addText([
    { text: "•  E1b의 peak layer 하나만 마스크해도 effect 없음 → ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "anchor 신호는 LLM stack 전반에 redundant하게 인코딩", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: ".\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "•  peak layer는 신호가 가장 ‘보이는 곳’이지, ‘유일하게 만들어지는 곳’이 아님.\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "•  Upper-half 단독이 6/6 panel-wide 작동하면서 fluency를 깨지 않는 ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "유일한 architecture-blind locus", options: { bold: true, fontSize: 11, color: GREEN } },
    { text: " — §7.4 free-lunch mitigation의 근거.", options: { fontSize: 11, color: CHARCOAL } },
  ], { x: 0.7, y: 5.45, w: W - 1.4, h: 1.2, valign: "top" });
  addPageFooter(s, 21, "§7.3 · Causal ablation");
}

// ---------------- Slide 22 — §7.4 mitigation Phase 2 headline ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§7.4 mitigation — upper-half soft re-weighting (E4 Phase 2, n=88,650/모델)", "df 떨어지고 exact_match 오르며 target_only accuracy 그대로 — ‘free-lunch’");
  const rows = [
    [tH("Model"), tH("s*"), tH("df base"), tH("df treated"), tH("df Δ pp"), tH("df rel"), tH("em base"), tH("em treated"), tH("em Δ pp")],
    [tBL("LLaVA-1.5-7b"), tBL("−3.0"), tB("0.288"), tB("0.246"), tHL("−4.19"), tHL("−14.6 %"), tB("0.334"), tB("0.342"), tHL("+0.77")],
    [tBL("ConvLLaVA-7b"), tBL("−2.0"), tB("0.258"), tB("0.233"), tHL("−2.49"), tHL("−9.6 %"), tB("0.352"), tB("0.365"), tHL("+1.30")],
    [tBL("InternVL3-8b"), tBL("−0.5"), tB("0.126"), tB("0.119"), tB("−0.74"), tB("−5.8 %"), tB("0.590"), tB("0.595"), tHL("+0.49")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.5, w: W - 1.1, colW: [1.5, 0.7, 1.05, 1.3, 1.0, 1.0, 1.05, 1.3, W - 1.1 - 1.5 - 0.7 - 1.05 - 1.3 - 1.0 - 1.0 - 1.05 - 1.3], rowH: [0.4, ...Array(3).fill(0.42)], fontFace: BODY_FONT, fontSize: 10.5 });
  // 3 invariants
  s.addText("3 invariants — 왜 ‘free-lunch’인가", { x: 0.55, y: 3.4, w: W - 1.1, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  const invs = [
    { color: TERRA, n: "①", t: "df는 모든 모델에서 감소, em은 모든 모델에서 +0.49~+1.30 pp 상승 — anchor 조건에서만 예측이 정답 쪽으로 이동." },
    { color: NAVY, n: "②", t: "accuracy_vqa(b) (target_only baseline)이 모든 strength에서 invariant — hook이 single-image inference로 leak하지 않음." },
    { color: GREEN, n: "③", t: "accuracy_vqa(d) (neutral arm)도 ±0.5 pp 이내 — 두 번째 이미지가 숫자가 아니면 hook이 영향 없음. anchor pathway에만 작용." },
  ];
  const cw = (W - 1.4) / 3;
  invs.forEach((iv, i) => {
    const cx = 0.55 + i * (cw + 0.2);
    s.addShape("rect", { x: cx, y: 3.85, w: cw, h: 1.7, fill: { color: "FFFFFF" }, line: { color: iv.color, width: 1.5 } });
    s.addShape("ellipse", { x: cx + 0.2, y: 4.0, w: 0.5, h: 0.5, fill: { color: iv.color }, line: { color: iv.color, width: 0 } });
    s.addText(iv.n, { x: cx + 0.2, y: 4.0, w: 0.5, h: 0.5, fontFace: TITLE_FONT, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(iv.t, { x: cx + 0.85, y: 3.95, w: cw - 1.0, h: 1.55, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
  });
  // bottom note
  s.addShape("rect", { x: 0.55, y: 5.7, w: W - 1.1, h: 1.0, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText([
    { text: "Per-model s*가 필요", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: " (−0.5 ~ −3.0, 한 자릿수 차이) — mitigation은 ‘locus + selection rule’ 수준에서 일반화되지, 단일 strength constant 수준은 아님. df 감소율은 baseline anchor pull과 anti-correlation (LLaVA-1.5 가장 큼, InternVL3 가장 작음). 다음 슬라이드에서 paired damage / recovery 분석.", options: { fontSize: 11, color: CHARCOAL } },
  ], { x: 0.7, y: 5.8, w: W - 1.4, h: 0.8, valign: "middle" });
  addPageFooter(s, 22, "§7.4 · Mitigation headline");
}

// ---------------- Slide 23 — §7.4 paired damage/recovery + anti-correlation ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§7.4 — paired anchor-damage & recovery + anti-correlation conjecture", "anchor가 가져온 정확도 손실 중 mitigation이 얼마나 회수하는지의 paired 분석");
  const rows = [
    [tH("Model"), tH("em(target_only)"), tH("em(num@0)"), tH("em(num@s*)"), tH("damage"), tH("recovery"), tH("% recovered")],
    [tBL("LLaVA-1.5-7b"), tB("0.370"), tB("0.334"), tB("0.342"), tB("−3.55 pp"), tHL("+0.77 pp"), tHL("21.7 %")],
    [tBL("ConvLLaVA-7b"), tB("0.445"), tB("0.352"), tB("0.365"), tB("−9.34 pp"), tHL("+1.31 pp"), tHL("14.0 %")],
    [tBL("InternVL3-8b"), tB("0.633"), tB("0.594"), tB("0.598"), tB("−3.87 pp"), tHL("+0.40 pp"), tHL("10.2 %")],
  ];
  s.addTable(rows, { x: 0.55, y: 1.55, w: W - 1.1, colW: [1.6, 1.7, 1.5, 1.5, 1.5, 1.5, W - 1.1 - 1.6 - 1.7 - 1.5 - 1.5 - 1.5 - 1.5], rowH: [0.4, ...Array(3).fill(0.45)], fontFace: BODY_FONT, fontSize: 11 });
  s.addText("n_paired ~ 17,700 per model (InternVL3 11,848 — parse-loss caveat)", { x: 0.55, y: 3.45, w: W - 1.1, h: 0.3, fontFace: BODY_FONT, fontSize: 9.5, italic: true, color: MUTED, align: "center" });

  // Two cards
  s.addShape("rect", { x: 0.55, y: 3.95, w: 6.0, h: 2.75, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("Paired picture — coherent across cluster", { x: 0.7, y: 4.05, w: 5.7, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: NAVY });
  s.addText([
    { text: "각 모델이 anchor 노출 시 ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "4-9 pp 정확도 손실", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: " 발생 (em(target_only) − em(num@0)).\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "Upper-half re-weighting이 그 손실의 ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "10-22%를 회수", options: { bold: true, fontSize: 11, color: GREEN } },
    { text: " (damage 대비 recovery, no target_only side-effect).\n\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "→ ", options: { fontSize: 11 } },
    { text: "free-lunch는 statistical significance 수준의 잡음이 아니라 paired anchor-damage의 의미 있는 fraction을 회복하는 결과.", options: { italic: true, fontSize: 11, color: NAVY } },
  ], { x: 0.7, y: 4.45, w: 5.7, h: 2.2, valign: "top" });

  s.addShape("rect", { x: 6.85, y: 3.95, w: 5.95, h: 2.75, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("Anti-correlation conjecture (testable)", { x: 7.0, y: 4.05, w: 5.65, h: 0.4, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
  s.addText([
    { text: "관측: relative df-reduction이 baseline anchor pull과 anti-correlated.\n", options: { bold: true, fontSize: 10.5, color: CHARCOAL } },
    { text: "•  InternVL3 (df₀ = 0.126, 가장 작음) → −5.8 % (가장 작은 reduction)\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "•  LLaVA-1.5 (df₀ = 0.288, 가장 큼) → −14.6 % (가장 큰 reduction)\n\n", options: { fontSize: 10, color: CHARCOAL } },
    { text: "Conjecture: ", options: { bold: true, fontSize: 10.5, color: NAVY } },
    { text: "upper-half attention pathway는 anchor 신호 중 ", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "더 큰 fraction", options: { italic: true, bold: true, fontSize: 10.5, color: NAVY } },
    { text: "을 carry — 평소 그 pathway를 덜 쓰는 모델일수록 down-weight 효과 큼. LLaVA/ConvLLaVA의 anchor 신호는 broadly distributed, InternVL3은 narrow concentrated.", options: { fontSize: 10.5, color: CHARCOAL } },
  ], { x: 7.0, y: 4.45, w: 5.65, h: 2.2, valign: "top" });
  addPageFooter(s, 23, "§7.4 · Damage / Recovery");
}

// ---------------- Slide 24 — §7.5 why "free-lunch" + summary ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§7.5 ‘free-lunch’ 해석 + §7 summary", "anchor pathway가 자기 baseline answer-formation에는 load-bearing하지 않다는 것이 핵심");
  // Three invariants visualization
  s.addText("free-lunch가 성립하는 3 자명 조건", { x: 0.55, y: 1.55, w: W - 1.1, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  const conds = [
    { icon: "✅", c: GREEN, head: "accuracy_vqa(b) invariant", body: "target_only baseline은 모든 strength × 모든 모델에서 변화 무 → hook이 single-image inference로 leak하지 않음. ‘mitigation이 자기 정확도를 갉아먹지 않는가’의 직접 검증." },
    { icon: "✅", c: GREEN, head: "accuracy_vqa(d) ±0.5 pp", body: "neutral arm (digit-free 두 번째 이미지)도 baseline ± 0.5 pp 이내 → 두 번째 이미지가 ‘있다’는 것 자체로 hook이 발동하지 않음. digit-bearing pathway에만 작용." },
    { icon: "↑", c: TERRA, head: "exact_match(a) +0.49~+1.30 pp", body: "anchor 조건에서만 prediction이 ‘움직인다’ — 그리고 정답 방향으로. 즉 mitigation이 떼어내는 신호는 정답에 weight되지 않는, ‘anchor 쪽으로만 끌어당기던 redundant signal’." },
  ];
  const cw = (W - 1.4) / 3;
  conds.forEach((c, i) => {
    const cx = 0.55 + i * (cw + 0.2);
    s.addShape("rect", { x: cx, y: 2.0, w: cw, h: 2.4, fill: { color: "FFFFFF" }, line: { color: c.c, width: 1.5 } });
    s.addShape("rect", { x: cx, y: 2.0, w: cw, h: 0.5, fill: { color: c.c }, line: { color: c.c, width: 0 } });
    s.addText(`${c.icon}  ${c.head}`, { x: cx + 0.1, y: 2.0, w: cw - 0.2, h: 0.5, fontFace: TITLE_FONT, fontSize: 11.5, bold: true, color: "FFFFFF", valign: "middle" });
    s.addText(c.body, { x: cx + 0.15, y: 2.6, w: cw - 0.3, h: 1.7, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
  });

  // §7 summary
  s.addShape("rect", { x: 0.55, y: 4.7, w: W - 1.1, h: 2.0, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 1 } });
  s.addText("§7 summary — 한 줄 정리", { x: 0.7, y: 4.78, w: W - 1.4, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  s.addText([
    { text: "• 메커니즘:", options: { bold: true, fontSize: 10.5, color: NAVY } },
    { text: " anchor 신호는 multi-layer-redundant pathway. one observable peak per encoder family (E1/E1b), but single-layer ablation에서 인과적 null on 6/6 (E1d).\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "• 구체화:", options: { bold: true, fontSize: 10.5, color: NAVY } },
    { text: " anchor 안에서는 digit pixels 자체에 attention 집중 (E1-patch, fair share 대비 +24~+40 pp on 4/4 모델).\n", options: { fontSize: 10.5, color: CHARCOAL } },
    { text: "• Mitigation:", options: { bold: true, fontSize: 10.5, color: NAVY } },
    { text: " upper-half soft re-weighting (E4 Phase 2)이 ‘locus + selection rule’ 수준의 architecture-blind mitigation. df −5.8~−14.6 % rel ↓, em +0.49~+1.30 pp ↑, target_only/neutral 모두 invariant — paired damage의 10-22 % 회수.", options: { fontSize: 10.5, color: CHARCOAL } },
  ], { x: 0.7, y: 5.18, w: W - 1.4, h: 1.45, valign: "top" });
  addPageFooter(s, 24, "§7.5 · Free-lunch + summary");
}

// ---------------- Slide 25 — §8.1 γ-β reasoning amplifies ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§8.1 reasoning이 anchoring을 강화한다 (γ-β)", "Qwen3-VL-8B-Instruct vs Thinking, MathVista, S1 anchor arm, n=365 — 동일 architecture / chat template / 4-condition stimuli");
  // Big numbers
  const stats = [
    { v1: "0.117", v2: "0.074", lab: "adopt(a)\nThinking / Instruct", mult: "× 1.6", arrow: "↑", color: TERRA },
    { v1: "0.291", v2: "0.102", lab: "df(a) C-form\nThinking / Instruct", mult: "× 2.9", arrow: "↑↑", color: TERRA },
    { v1: "0.196", v2: "0.216", lab: "acc(b)\nThinking / Instruct", mult: "−0.020", arrow: "↓", color: GREEN, note: "Thinking이 baseline 자체도 더 부정확" },
  ];
  const cw = (W - 1.4) / 3;
  stats.forEach((st, i) => {
    const cx = 0.55 + i * (cw + 0.2);
    s.addShape("rect", { x: cx, y: 1.55, w: cw, h: 3.4, fill: { color: "FFFFFF" }, line: { color: st.color, width: 1.5 } });
    s.addText(`${st.v1} / ${st.v2}`, { x: cx, y: 1.7, w: cw, h: 0.85, fontFace: TITLE_FONT, fontSize: 30, bold: true, color: NAVY, align: "center", valign: "middle" });
    s.addText(st.lab, { x: cx, y: 2.5, w: cw, h: 0.7, fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, align: "center", valign: "middle", paraSpaceAfter: 0 });
    s.addText(st.mult, { x: cx, y: 3.25, w: cw, h: 0.6, fontFace: TITLE_FONT, fontSize: 22, bold: true, color: st.color, align: "center", valign: "middle" });
    s.addText(st.arrow, { x: cx, y: 3.85, w: cw, h: 0.5, fontFace: TITLE_FONT, fontSize: 22, bold: true, color: st.color, align: "center", valign: "middle" });
    if (st.note) s.addText(st.note, { x: cx + 0.15, y: 4.45, w: cw - 0.3, h: 0.45, fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, align: "center" });
  });
  // Reading
  s.addShape("rect", { x: 0.55, y: 5.15, w: W - 1.1, h: 1.55, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("Reading", { x: 0.7, y: 5.2, w: W - 1.4, h: 0.35, fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA });
  s.addText([
    { text: "•  Thinking 체크포인트는 동일 architecture / chat template / 동일 4-condition 자극 — 차이는 trained reasoning 행동뿐.\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "•  reasoning trace가 anchor를 잡지 못할 뿐 아니라 오히려 ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "anchor 정보가 elaboration step에 누적되어 증폭", options: { bold: true, fontSize: 11, color: TERRA } },
    { text: " — Wang 2025 (LRM-judging) + VLMBias (reasoning models)와 일치.\n", options: { fontSize: 11, color: CHARCOAL } },
    { text: "•  단일 pair × 단일 dataset이라 ", options: { fontSize: 11, color: CHARCOAL } },
    { text: "existence-proof", options: { italic: true, fontSize: 11, color: NAVY } },
    { text: " 수준의 hook. §8 future: 30B-A3B 쌍 + Gemma3-Thinking + ChartQA/TallyQA로 generalisation.", options: { fontSize: 11, color: CHARCOAL } },
  ], { x: 0.7, y: 5.55, w: W - 1.4, h: 1.1, valign: "top" });
  addPageFooter(s, 25, "§8.1 · γ-β reasoning amplifies");
}

// ---------------- Slide 26 — §8.4 Limitations ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§8.4 Limitations — Reviewer가 물을 항목 미리 disclose", "open-weight only · 단일 prompt · no human baseline · single-cluster mitigation · γ-β single pair · driver schema audit");
  const lims = [
    { head: "Single-prompt runs", body: "JSON-strict 단일 prompt만 사용. paraphrase robustness (3-5 prompt × bootstrap CI)는 다음 hardening pass — 선행 cognitive-bias LLM paper에서 일관되게 reviewer가 제기.", c: TERRA },
    { head: "Open weights only", body: "메인 7개 모델 모두 open-weight. closed-model defuse (GPT-4o / Gemini 2.5 ~500 sample)는 revision 시 access 확보되면 추가.", c: NAVY },
    { head: "No human baseline", body: "§1/§6의 인지과학 framing은 prior literature 기반. 50명 Prolific replication은 ARR clock 상 미실시.", c: NAVY },
    { head: "Distance window dataset-dependent", body: "VQAv2/TallyQA absolute, ChartQA/MathVista relative cutoff. relative form은 inductive choice — GT 분포 다른 dataset에서 재검증 필요.", c: TERRA },
    { head: "Mid-stack mitigation single-cluster", body: "E4 free-lunch는 LLaVA-1.5/ConvLLaVA/InternVL3 3 mid-stack-cluster 모델만 검증. SigLIP-Gemma early peak / Qwen-ViT late peak로의 일반화는 P3.", c: TERRA },
    { head: "γ-β single pair", body: "reasoning-amplifies-anchoring은 1 pair × 1 dataset. existence proof로 취급, quantitative law로 보지 않음.", c: NAVY },
    { head: "Driver schema audit", body: "M1→M2 refactor 사이 driver-schema gap이 direction_follow_rate를 silently 0으로 만든 적 있음. C-form refactor + reaggregate sweep + migration report로 remediation. 논문의 모든 numerical claim은 post-audit C-form re-aggregate output에서 파생.", c: GREEN },
  ];
  // 4 + 3 grid
  const colW = (W - 1.4) / 3;
  lims.forEach((l, i) => {
    const r = Math.floor(i / 3), c = i % 3;
    const cx = 0.55 + c * (colW + 0.2);
    const cy = 1.55 + r * 1.78;
    s.addShape("rect", { x: cx, y: cy, w: colW, h: 1.7, fill: { color: "FFFFFF" }, line: { color: l.c, width: 1.25 } });
    s.addShape("rect", { x: cx, y: cy, w: colW, h: 0.4, fill: { color: l.c }, line: { color: l.c, width: 0 } });
    s.addText(l.head, { x: cx + 0.1, y: cy, w: colW - 0.2, h: 0.4, fontFace: TITLE_FONT, fontSize: 11, bold: true, color: "FFFFFF", valign: "middle" });
    s.addText(l.body, { x: cx + 0.1, y: cy + 0.45, w: colW - 0.2, h: 1.2, fontFace: BODY_FONT, fontSize: 9.5, color: CHARCOAL, valign: "top", paraSpaceAfter: 1 });
  });
  addPageFooter(s, 26, "§8.4 · Limitations");
}

// ---------------- Slide 27 — §1.6 Contributions + §8.2/§8.3 future work ----------------
{
  const s = pres.addSlide();
  addPageHeader(s, "§1.6 5가지 공헌 + §8.2/§8.3 future work", "what this paper adds · what comes next");
  // Left: 5 contributions
  s.addShape("rect", { x: 0.55, y: 1.55, w: 7.0, h: 5.15, fill: { color: PALE_NAVY }, line: { color: NAVY, width: 0 } });
  s.addText("§1.6 — 5가지 공헌", { x: 0.7, y: 1.65, w: 6.7, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
  const contribs = [
    "VLM 최초의 cross-modal 숫자 anchoring 평가 — 4-condition (target/anchor/mask/neutral) + FLUX-rendered digit anchor 인벤토리 + OpenCV-inpainted mask counterpart.",
    "Canonical M2 metrics (C-form direction-follow `(pa−pb)·(anchor−pb) > 0 AND pa ≠ pb`) — gt-free, baseline-relative anchor pull 측정. 207-cell migration audit 공개.",
    "Cross-dataset evidence — VQAv2 number (n=17,730) / TallyQA / ChartQA / MathVista 4종 + 7-모델 메인 패널 + 3-모델 E5e + 2-모델 γ-β reasoning.",
    "메커니즘 + mitigation — encoder-family별 attention localisation (E1/E1b) + E1-patch digit-pixel concentration (4/6 archetype) + causal ablation panel (E1d) + upper-half mitigation full-scale validation (E4 Phase 2, n=88,650/모델).",
    "VLM에서 reasoning이 anchoring을 amplify하는 결과 (γ-β, MathVista) — text-only LRM 문헌과 일치하는 multimodal counterpart.",
  ];
  let yy = 2.05;
  contribs.forEach((c, i) => {
    s.addShape("ellipse", { x: 0.7, y: yy + 0.03, w: 0.4, h: 0.4, fill: { color: TERRA }, line: { color: TERRA, width: 0 } });
    s.addText(`${i + 1}`, { x: 0.7, y: yy + 0.03, w: 0.4, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    s.addText(c, { x: 1.2, y: yy, w: 6.2, h: 0.85, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
    yy += 0.92;
  });

  // Right: future work F1/F2/F3
  s.addShape("rect", { x: 7.75, y: 1.55, w: 5.05, h: 5.15, fill: { color: PALE_TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("Future work (선호도 순)", { x: 7.9, y: 1.65, w: 4.75, h: 0.4, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA });
  const fws = [
    { tag: "F1 · LLM/VLM diff (§8.2)", body: "동일 numerical question을 (a) text prompt → LLM (b) rendered-digit image → VLM. layer-wise integration profile (E1-style + per-layer logit-lens) 비교. text-anchor LLM 문헌과 image-anchor 본 논문의 ‘gap’을 메우는 가장 깨끗한 single follow-up." },
    { tag: "F2 · image vs text anchor (§8.3)", body: "동일 VLM 위에서 anchor를 image vs prompt-text로 제시 — modality 채널 분리 (LLM 없이도 가능). 두 stimulus inventory (rendered + prompt-text equivalent) 모두 release." },
    { tag: "F3 · γ-β multi-pair scale-up (§8.1)", body: "Qwen3-VL-30B-A3B Instruct/Thinking 추가 + Gemma3-Thinking + GPT-4o reasoning-mode + ChartQA/TallyQA generalisation." },
  ];
  let yy2 = 2.05;
  fws.forEach((f) => {
    s.addText(f.tag, { x: 7.9, y: yy2, w: 4.75, h: 0.35, fontFace: TITLE_FONT, fontSize: 11, bold: true, color: TERRA });
    s.addText(f.body, { x: 7.9, y: yy2 + 0.35, w: 4.75, h: 1.3, fontFace: BODY_FONT, fontSize: 9.5, color: CHARCOAL, valign: "top", paraSpaceAfter: 2 });
    yy2 += 1.55;
  });
  addPageFooter(s, 27, "§1.6 / §8.2-3 · Contributions + Future");
}

// ---------------- Slide 28 — Reproducibility + closing ----------------
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape("rect", { x: 0, y: H - 0.6, w: W, h: 0.6, fill: { color: TERRA }, line: { color: TERRA, width: 0 } });
  s.addText("§1.7 Reproducibility — open release", { x: 0.7, y: 0.7, w: W - 1.4, h: 0.6, fontFace: TITLE_FONT, fontSize: 28, bold: true, color: "FFFFFF" });
  s.addText("permissive license, code/configs/inventories/predictions/aggregate CSVs 모두 공개", { x: 0.7, y: 1.4, w: W - 1.4, h: 0.4, fontFace: BODY_FONT, fontSize: 13, italic: true, color: GOLD });

  const items = [
    { tag: "Code", body: "scripts/run_experiment.py / scripts/analyze_*.py / scripts/extract_attention_mass.py / scripts/causal_anchor_ablation.py / scripts/e4_attention_reweighting.py 등" },
    { tag: "Configs", body: "configs/*.yaml — VQAv2/TallyQA/ChartQA/MathVista 각각 4-cond/E5b/E5c/E5d/E5e/E1-patch/E4 Phase 1+2." },
    { tag: "Stimulus inventories", body: "FLUX-schnell 128 anchor + 128 mask + 128 neutral. FLUX seed + OpenCV inpaint 파라미터 포함." },
    { tag: "Predictions / aggregates", body: "predictions.jsonl + per-row 6 flag (post-audit C-form). docs/insights/_data/*.csv aggregate (per_cell / per_layer / per_model)." },
    { tag: "Audit trail", body: "docs/insights/M2-metric-definition-evidence.md + docs/insights/C-form-migration-report.md (207 cell M1→M2 audit). references/roadmap.md §10 changelog." },
    { tag: "Models", body: "12 모델 모두 Hugging Face open weights — 라이선스 dependent on each repo." },
  ];
  const cw = (W - 1.4) / 2;
  items.forEach((it, i) => {
    const r = Math.floor(i / 2), c = i % 2;
    const cx = 0.7 + c * (cw + 0.1);
    const cy = 2.0 + r * 1.45;
    s.addShape("rect", { x: cx, y: cy, w: cw, h: 1.3, fill: { color: "FFFFFF" }, line: { color: GOLD, width: 1 } });
    s.addText(it.tag, { x: cx + 0.15, y: cy + 0.08, w: cw - 0.3, h: 0.35, fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY });
    s.addText(it.body, { x: cx + 0.15, y: cy + 0.45, w: cw - 0.3, h: 0.8, fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", paraSpaceAfter: 1 });
  });

  s.addText("Cross-modal numerical anchoring — uncertainty-modulated graded pull, digit-pixel causal, upper-half free-lunch mitigation, reasoning amplifies.", {
    x: 0.7, y: H - 1.3, w: W - 1.4, h: 0.5, fontFace: BODY_FONT, fontSize: 13, italic: true, color: GOLD, align: "center", valign: "middle",
  });
  s.addText("End · paper §1-§8 한국어 요약 (확장판) · 2026-04-29", { x: 0.7, y: H - 0.55, w: W - 1.4, h: 0.5, fontFace: BODY_FONT, fontSize: 11, color: "FFFFFF", align: "center", valign: "middle" });
}

// ============================================================================
// Save
// ============================================================================
const OUT = path.join(REPO, "docs/ppt/paper_summary_kr.pptx");
pres.writeFile({ fileName: OUT }).then((f) => {
  console.log("Wrote:", f);
});
