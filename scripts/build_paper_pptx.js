// Build paper-style PPTX summary deck (Korean) for cross-modal anchoring in VLMs.
// Outputs: docs/ppt/paper_summary.pptx
//
// Run: node scripts/build_paper_pptx.js
//
// Companion markdown: docs/insights/paper_summary_slides.md
// Slides reference figures under docs/figures/.

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const REPO_ROOT = path.resolve(__dirname, "..");
const FIG_DIR = path.join(REPO_ROOT, "docs", "figures");
const OUT_PATH = path.join(REPO_ROOT, "docs", "ppt", "paper_summary.pptx");

// ---- palette ----
const NAVY = "1E2761";
const ICE = "CADCFC";
const ACCENT_GOLD = "F2A900";
const ACCENT_RED = "C8102E";
const WHITE = "FFFFFF";
const GREY_DARK = "374151";
const GREY_MED = "6C7280";
const GREY_LIGHT = "E5E7EB";

const HEADER_FONT = "Calibri";
const BODY_FONT = "Calibri";

// ---- helpers ----
function addBgFooter(slide, slideNum, total, sectionLabel) {
  // Top accent strip (navy) for non-title slides
  slide.addShape("rect", { x: 0, y: 0, w: 13.3, h: 0.18, fill: { color: NAVY } });
  // Footer
  slide.addText(`§ ${sectionLabel}`, {
    x: 0.5, y: 7.05, w: 8, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, color: GREY_MED, align: "left", margin: 0,
  });
  slide.addText(`${slideNum} / ${total}`, {
    x: 11.8, y: 7.05, w: 1, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, color: GREY_MED, align: "right", margin: 0,
  });
}

function addTitle(slide, titleText, subtitleText) {
  slide.addText(titleText, {
    x: 0.5, y: 0.45, w: 12.3, h: 0.7,
    fontFace: HEADER_FONT, fontSize: 30, bold: true, color: NAVY, margin: 0,
  });
  if (subtitleText) {
    slide.addText(subtitleText, {
      x: 0.5, y: 1.15, w: 12.3, h: 0.4,
      fontFace: BODY_FONT, fontSize: 16, color: GREY_DARK, italic: true, margin: 0,
    });
  }
}

function maybeImage(slidePath) {
  // Confirm fig file exists; otherwise return null so we can degrade gracefully.
  if (!fs.existsSync(slidePath)) return null;
  return slidePath;
}

// ---- create deck ----
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.title = "Cross-modal anchoring in VLMs — paper summary";
pres.author = "vlm_anchoring (2026-04-29)";

const TOTAL = 22; // updated after we count below

// ====================================================================
// Slide 1 — Title
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Cross-modal Numerical Anchoring in VLMs", {
    x: 0.6, y: 1.6, w: 12.1, h: 1.4,
    fontFace: HEADER_FONT, fontSize: 42, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Uncertainty-modulated graded pull, digit-pixel causality, and a free-lunch mitigation", {
    x: 0.6, y: 3.1, w: 12.1, h: 0.6,
    fontFace: BODY_FONT, fontSize: 20, color: ICE, italic: true, margin: 0,
  });
  // Accent bar
  s.addShape("rect", { x: 0.6, y: 3.85, w: 1.3, h: 0.06, fill: { color: ACCENT_GOLD } });
  s.addText("EMNLP 2026 Main · ARR May 25 target", {
    x: 0.6, y: 4.05, w: 12.1, h: 0.4,
    fontFace: BODY_FONT, fontSize: 16, color: ICE, margin: 0,
  });
  s.addText("논문 스타일 정리 발표 (2026-04-29)\n자료: references/project.md §0 + docs/insights/", {
    x: 0.6, y: 6.4, w: 12.1, h: 0.8,
    fontFace: BODY_FONT, fontSize: 13, color: ICE, margin: 0,
  });
}

// ====================================================================
// Slide 2 — Headline claim
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "헤드라인 클레임 (§1 Introduction)",
    "한 줄로 요약하는 본 연구의 주장");

  // Big quote block
  s.addShape("rect", { x: 0.6, y: 1.95, w: 12.1, h: 2.1,
    fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText([
    { text: "Cross-modal numerical anchoring in VLMs is\n", options: { color: WHITE } },
    { text: "uncertainty-modulated graded pull,\n", options: { color: ACCENT_GOLD, bold: true } },
    { text: "not categorical capture — and concentrates on a\n", options: { color: WHITE } },
    { text: "digit-pixel cue", options: { color: ACCENT_GOLD, bold: true } },
    { text: " inside the anchor image.", options: { color: WHITE } },
  ], { x: 0.9, y: 2.05, w: 11.5, h: 1.9, fontFace: HEADER_FONT, fontSize: 22, valign: "middle", margin: 0 });

  // Three pillars
  const pillars = [
    { title: "Setup novelty", body: "다른 어떤 연구도 stand-alone rendered-number 이미지를 cross-modal anchor로 open numerical VQA에 적용한 적이 없음." },
    { title: "Graded vs. categorical", body: "Phase A A1: 7 모델 전부 wrong-base direction-follow가 correct-base보다 +6.9–19.6 pp 큼. paired adopt는 2–6 %." },
    { title: "Cognitive science 정합", body: "Mussweiler & Strack의 selective accessibility — uncertainty에 비례한 anchor pull과 일치." },
  ];
  pillars.forEach((p, i) => {
    const x = 0.6 + i * 4.15;
    s.addShape("rect", { x, y: 4.25, w: 4.0, h: 2.3,
      fill: { color: ICE }, line: { color: ICE } });
    s.addShape("rect", { x, y: 4.25, w: 0.08, h: 2.3, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD } });
    s.addText(p.title, {
      x: x + 0.25, y: 4.35, w: 3.7, h: 0.5,
      fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, margin: 0,
    });
    s.addText(p.body, {
      x: x + 0.25, y: 4.85, w: 3.7, h: 1.7,
      fontFace: BODY_FONT, fontSize: 12, color: GREY_DARK, margin: 0,
    });
  });

  addBgFooter(s, 2, TOTAL, "1 Introduction");
}

// ====================================================================
// Slide 3 — Motivation / Real-world scenario
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "현실 시나리오와 위험", "사용자가 VLM에 여러 이미지를 함께 보여주는 상황은 늘고 있다");

  const items = [
    { h: "현실 시나리오", b: "앨범 정리 / 멀티-스크린샷 / 문서 복수 첨부 — VLM에 여러 이미지를 동시 입력하는 빈도가 높아짐" },
    { h: "위험", b: "정답과 무관한 이미지가 답변에 영향을 준다면? 특히 *숫자가 그려진 이미지*가 다른 질문의 수치 답변을 끌어당긴다면?" },
    { h: "인지과학 prior", b: "Tversky-Kahneman / Mussweiler-Strack: 사람도 무관한 숫자에 anchored 되며, 효과는 불확실할수록 강함" },
    { h: "LLM 선행 연구", b: "Jones-Steinhardt 2022, Echterhoff EMNLP Findings 2024: 텍스트 anchor 주입 시 유사 효과 확인" },
    { h: "VLM 공백", b: "시각적 anchor (rendered digit image)가 다른 이미지의 수치 질문에 미치는 영향은 미답" },
  ];
  items.forEach((it, i) => {
    const y = 1.95 + i * 0.95;
    s.addShape("oval", { x: 0.6, y: y + 0.05, w: 0.45, h: 0.45,
      fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(String(i + 1), { x: 0.6, y: y + 0.05, w: 0.45, h: 0.45,
      fontFace: HEADER_FONT, fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText(it.h, { x: 1.25, y: y, w: 2.5, h: 0.5,
      fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, valign: "top", margin: 0 });
    s.addText(it.b, { x: 4.0, y: y, w: 8.7, h: 0.85,
      fontFace: BODY_FONT, fontSize: 13, color: GREY_DARK, valign: "top", margin: 0 });
  });

  addBgFooter(s, 3, TOTAL, "1 Introduction");
}

// ====================================================================
// Slide 4 — Related Work matrix
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Related Work — novelty 매트릭스 (§2)",
    "핵심 setup은 직접 선행 없음, 가장 가까운 이웃들과 명확한 차별");

  const head = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: HEADER_FONT, fontSize: 12 } });
  const body = (t, h = false) => ({ text: t, options: { fontFace: BODY_FONT, fontSize: 11, color: GREY_DARK, bold: h } });
  const data = [
    [head("Extension"), head("가장 가까운 선행"), head("Novelty 판정")],
    [body("Cross-modal numerical anchoring (core)", true), body("VLMBias (다른 setup); typographic attacks (다른 task)"), body("Genuinely novel", true)],
    [body("\"Stronger on wrong cases\" 비대칭", true), body("LLM/VLM anchoring 어디에도 없음"), body("Genuinely novel — strongest hook", true)],
    [body("Dataset 확장 (VQAv2 → TallyQA/ChartQA/MathVista)"), body("N/A — 방법론 위생"), body("Required, not novel")],
    [body("ViT vs Conv-encoder ablation"), body("Typographic-attack mech (arXiv 2508.20570)"), body("Novel angle, ablation 가능")],
    [body("Confidence-modulated anchoring (logit, §6)", true), body("Phase A binary; LLM 쪽 entropy 연구 일부"), body("Continuous proxy = novel projection", true)],
    [body("Encoder-blind upper-half mitigation (§7)", true), body("Weng et al. EMNLP 2024 Main (causal mediation)"), body("Cross-architecture locus = novel", true)],
  ];
  s.addTable(data, { x: 0.5, y: 1.85, w: 12.3, colW: [4.5, 4.4, 3.4],
    border: { type: "solid", pt: 0.5, color: GREY_LIGHT } });

  s.addText("Findings-tier 패턴 → Main-tier로 끌어올린 4가지 lever:  attention mass · causal ablation · mid-stack mitigation · confidence proxy",
    { x: 0.5, y: 6.5, w: 12.3, h: 0.5,
      fontFace: BODY_FONT, fontSize: 12, color: NAVY, italic: true, margin: 0 });

  addBgFooter(s, 4, TOTAL, "2 Related Work");
}

// ====================================================================
// Slide 5 — 4 conditions (Problem Definition)
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "4 조건 setup (§3 Problem Definition)",
    "sample_instance 마다 4 conditions: b / a / m / d");

  const conds = [
    { tag: "b", name: "target_only", desc: "두 번째 이미지 없음", role: "baseline (pred_b)" },
    { tag: "a", name: "+ anchor", desc: "디지트 한 글자가 그려진 anchor 이미지", role: "manipulation (pred_a)" },
    { tag: "m", name: "+ anchor_masked", desc: "anchor 이미지의 디지트 픽셀 영역만 inpaint로 가림", role: "digit-pixel control (pred_m)" },
    { tag: "d", name: "+ neutral", desc: "디지트 없는 FLUX 생성 이미지", role: "2-image distraction control (pred_d)" },
  ];
  conds.forEach((c, i) => {
    const x = 0.6 + i * 3.15;
    s.addShape("rect", { x, y: 1.85, w: 3.0, h: 4.0,
      fill: { color: WHITE }, line: { color: NAVY, width: 1.2 } });
    s.addShape("rect", { x, y: 1.85, w: 3.0, h: 0.7, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(c.tag, { x, y: 1.85, w: 0.7, h: 0.7,
      fontFace: HEADER_FONT, fontSize: 28, bold: true, color: ACCENT_GOLD, align: "center", valign: "middle", margin: 0 });
    s.addText(c.name, { x: x + 0.65, y: 1.85, w: 2.3, h: 0.7,
      fontFace: HEADER_FONT, fontSize: 13, color: WHITE, valign: "middle", margin: 0 });
    s.addText(c.desc, { x: x + 0.2, y: 2.7, w: 2.6, h: 1.5,
      fontFace: BODY_FONT, fontSize: 12, color: GREY_DARK, margin: 0 });
    s.addText(c.role, { x: x + 0.2, y: 4.5, w: 2.6, h: 1.0,
      fontFace: BODY_FONT, fontSize: 11, color: ACCENT_GOLD, italic: true, margin: 0 });
  });

  s.addText([
    { text: "(a − d) gap → ", options: { bold: true, color: NAVY } },
    { text: "anchoring vs. distraction 분리 · ", options: { color: GREY_DARK } },
    { text: "(a − m) gap → ", options: { bold: true, color: NAVY } },
    { text: "digit-pixel 인과 분리 · ", options: { color: GREY_DARK } },
    { text: "(wrong-base − correct-base) → ", options: { bold: true, color: NAVY } },
    { text: "uncertainty 변조 분리", options: { color: GREY_DARK } },
  ], { x: 0.6, y: 6.05, w: 12.1, h: 0.8, fontFace: BODY_FONT, fontSize: 13, margin: 0 });

  addBgFooter(s, 5, TOTAL, "3 Problem Definition");
}

// ====================================================================
// Slide 6 — Canonical metrics M2
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Canonical metrics — M2 (§3)",
    "docs/insights/M2-metric-definition-evidence.md");

  // Formula card
  s.addShape("rect", { x: 0.6, y: 1.85, w: 12.1, h: 3.05,
    fill: { color: ICE }, line: { color: ICE } });
  s.addText([
    { text: "adopt_rate          = #(pa == anchor AND pb != anchor)  /  #(pb != anchor)\n", options: { breakLine: true } },
    { text: "direction_follow    = #( (pb−gt)·(pa−gt) > 0  AND  pa != pb )  /  #(numeric pair AND anchor present)\n", options: { breakLine: true } },
    { text: "exact_match         = #(pa == gt)  /  #(numeric pair)\n", options: { breakLine: true } },
    { text: "anchor_effect_M     = M(anchor arm) − M(neutral arm)", options: {} },
  ], { x: 0.95, y: 1.95, w: 11.4, h: 2.85, fontFace: "Consolas", fontSize: 16, color: NAVY, margin: 0 });

  // Notation
  s.addText("표기 통일", { x: 0.6, y: 5.05, w: 6.0, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "pred_b / pred_a / pred_m / pred_d ", options: { bold: true, color: NAVY } },
    { text: "= 각 condition의 prediction\n", options: { breakLine: true, color: GREY_DARK } },
    { text: "anchor ", options: { bold: true, color: NAVY } },
    { text: "= anchor image의 value · ", options: { color: GREY_DARK } },
    { text: "gt ", options: { bold: true, color: NAVY } },
    { text: "= ground truth\n", options: { breakLine: true, color: GREY_DARK } },
    { text: "Boolean flags: pb_eq_a / pa_eq_a / gt_eq_a / pa_ne_pb / pb_eq_gt", options: { color: GREY_DARK } },
  ], { x: 0.6, y: 5.5, w: 6.0, h: 1.5, fontFace: BODY_FONT, fontSize: 13, margin: 0 });

  // Why this exact pair
  s.addText("왜 이 조합?", { x: 7.0, y: 5.05, w: 5.7, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "• 18 변종 중 가장 깔끔 (M2 evidence)\n", options: { breakLine: true } },
    { text: "• wrong > correct gap S0/S1에서 22/22 보존\n", options: { breakLine: true } },
    { text: "• distance decay S1 > S5 100%\n", options: { breakLine: true } },
    { text: "• anchor > masked S0/S1에서 5/6 보존", options: {} },
  ], { x: 7.0, y: 5.5, w: 5.7, h: 1.5, fontFace: BODY_FONT, fontSize: 13, color: GREY_DARK, margin: 0 });

  addBgFooter(s, 6, TOTAL, "3 Problem Definition");
}

// ====================================================================
// Slide 7 — M2 evidence (figure)
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "M2 — Adopt variant signal preservation", "다양한 (numerator, denominator) 조합 비교");
  const img = path.join(FIG_DIR, "paper_M2_variant_comparison.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: 0.6, y: 1.7, w: 8.0, h: 4.5 });
  }
  s.addText([
    { text: "▷ A_paired__D_paired", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "  paired numerator + paired denominator\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "  → wrong>correct gap +0.040, 22/22 cells\n\n", options: { color: NAVY, bold: true, breakLine: true } },
    { text: "▷ A_raw__D_all (pre-M1 marginal)", options: { bold: true, color: ACCENT_RED, breakLine: true } },
    { text: "  → gap −0.028 (wrong < correct, 8/22)\n", options: { color: ACCENT_RED, breakLine: true } },
    { text: "  gt==anchor confound이 신호 역전시킴\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "선택: A_paired__D_paired (M2)", options: { bold: true, color: NAVY } },
  ], { x: 8.8, y: 1.95, w: 4.1, h: 4.5, fontFace: BODY_FONT, fontSize: 12, margin: 0 });

  addBgFooter(s, 7, TOTAL, "3 Problem Definition");
}

// ====================================================================
// Slide 8 — Datasets + Anchor Inventory
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Datasets + Anchor Inventory (§4)",
    "4 datasets · FLUX-rendered digit anchor + masked + neutral");

  const head = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: HEADER_FONT, fontSize: 12 } });
  const body = (t) => ({ text: t, options: { fontFace: BODY_FONT, fontSize: 11, color: GREY_DARK } });
  const data = [
    [head("Dataset"), head("GT range"), head("샘플"), head("anchor selection rule"), head("Status")],
    [body("VQAv2 number"), body("0–8"), body("17,730"), body("anchor ∈ {0..9} (range-restricted)"), body("[done] 7-model main + 7-model strengthen")],
    [body("TallyQA number"), body("0–8"), body("≈11,200"), body("absolute |a−gt| ≤ 5 (S1)"), body("[in flight] E5b 1-model + E5e 1/3 models")],
    [body("ChartQA"), body("1–1000+"), body("≈5,400"), body("relative |a−gt| ≤ max(1, 0.10·gt)"), body("[done] E5e 3 models")],
    [body("MathVista (integer)"), body("1–1000"), body("385"), body("relative_s1 (same)"), body("[done] gamma-alpha 3 models (2026-04-29)")],
  ];
  s.addTable(data, { x: 0.5, y: 1.85, w: 12.3, colW: [2.6, 1.4, 1.5, 4.0, 2.8],
    border: { type: "solid", pt: 0.5, color: GREY_LIGHT } });

  s.addText("Anchor 생성 파이프라인", { x: 0.6, y: 4.6, w: 12.1, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "(1) FLUX로 480×480 anchor 이미지 생성 ({0..10000} 중 128개) → ", options: { color: GREY_DARK } },
    { text: "anchor", options: { bold: true, color: NAVY } },
    { text: "\n(2) 동일 이미지에 OpenCV Telea inpaint로 디지트 픽셀만 가림 → ", options: { color: GREY_DARK, breakLine: true } },
    { text: "masked", options: { bold: true, color: NAVY } },
    { text: "\n(3) FLUX로 디지트 없는 자연 장면 이미지 별도 생성 → ", options: { color: GREY_DARK, breakLine: true } },
    { text: "neutral", options: { bold: true, color: NAVY } },
  ], { x: 0.6, y: 5.05, w: 12.1, h: 1.7, fontFace: BODY_FONT, fontSize: 13, margin: 0 });

  addBgFooter(s, 8, TOTAL, "4 Datasets");
}

// ====================================================================
// Slide 9 — Distance: E5b decay
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Distance × Plausibility Window (E5b)",
    "wrong-base에서 anchor 거리가 가까울수록 adoption ↑");

  const img = path.join(FIG_DIR, "E5b_adopt_cond_curve.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: 0.6, y: 1.75, w: 8.5, h: 4.7, sizing: { type: "contain", w: 8.5, h: 4.7 } });
  }
  s.addText([
    { text: "S1 [|a-gt| ≤ 1] 가장 큰 effect, S5 [301+]에서 노이즈 floor\n\n", options: { breakLine: true } },
    { text: "VQAv2 wrong-base S1 = ", options: { color: GREY_DARK } },
    { text: "0.131", options: { bold: true, color: NAVY } },
    { text: ", S5 = 0.003 (44× 차이)\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "TallyQA wrong-base S1 = ", options: { color: GREY_DARK } },
    { text: "0.092", options: { bold: true, color: NAVY } },
    { text: ", S5 = 0.000 (∞ 차이)\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "결론: anchor가 plausible할 때만 effect 발생.", options: { bold: true, color: NAVY } },
  ], { x: 9.3, y: 2.05, w: 3.6, h: 4.5, fontFace: BODY_FONT, fontSize: 12, margin: 0 });

  addBgFooter(s, 9, TOTAL, "5 Distance & Plausibility Window");
}

// ====================================================================
// Slide 10 — E5b cross-dataset overlay
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Cross-dataset overlay (E5b)",
    "VQAv2 / TallyQA wrong-base만 같은 그래프에");

  const img = path.join(FIG_DIR, "E5b_adopt_cond_overlay.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: 0.6, y: 1.7, w: 8.5, h: 4.7, sizing: { type: "contain", w: 8.5, h: 4.7 } });
  }
  s.addText([
    { text: "두 데이터셋의 baseline 정확도가 매우 다름:\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "  acc(b) VQAv2 0.62 vs TallyQA 0.21\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "그럼에도 ", options: { color: GREY_DARK } },
    { text: "adopt 곡선의 모양은 동일", options: { bold: true, color: NAVY } },
    { text: " — image domain에 의존하지 않는 효과.\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "S1 → S5 단조 감소가 cross-dataset robust.", options: { color: NAVY, italic: true } },
  ], { x: 9.3, y: 2.05, w: 3.6, h: 4.5, fontFace: BODY_FONT, fontSize: 12, margin: 0 });

  addBgFooter(s, 10, TOTAL, "5 Distance & Plausibility Window");
}

// ====================================================================
// Slide 11 — E5c digit-pixel causality
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Digit-pixel causality (E5c)",
    "anchor − masked gap → 디지트 픽셀이 effect의 인과");

  const img1 = path.join(FIG_DIR, "E5c_anchor_vs_masked_adopt.png");
  if (fs.existsSync(img1)) {
    s.addImage({ path: img1, x: 0.4, y: 1.75, w: 6.4, h: 4.7, sizing: { type: "contain", w: 6.4, h: 4.7 } });
  }
  const img2 = path.join(FIG_DIR, "E5c_correct_vs_wrong_adopt.png");
  if (fs.existsSync(img2)) {
    s.addImage({ path: img2, x: 6.9, y: 1.75, w: 6.0, h: 4.7, sizing: { type: "contain", w: 6.0, h: 4.7 } });
  }

  s.addText("디지트 픽셀만 가렸을 뿐 배경은 동일 → adopt 큰 폭 감소. background 자체는 generic distractor 정도의 영향만.",
    { x: 0.5, y: 6.4, w: 12.3, h: 0.5, fontFace: BODY_FONT, fontSize: 12, color: NAVY, italic: true, align: "center", margin: 0 });

  addBgFooter(s, 11, TOTAL, "5 Distance & Plausibility Window");
}

// ====================================================================
// Slide 12 — E5c df distance-invariance
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Direction-follow는 distance-invariant",
    "anchor와 masked가 거의 동일 — generic 2-image distraction artifact");

  const img = path.join(FIG_DIR, "E5c_anchor_vs_masked_df.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: 0.6, y: 1.7, w: 8.5, h: 4.7, sizing: { type: "contain", w: 8.5, h: 4.7 } });
  }
  s.addText([
    { text: "key 관찰\n", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "df_cond는 S1~S5 거의 동일 (S1=0.36, S5=0.37)\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "anchor와 masked의 df_cond도 거의 일치 (gap ≤ 0.005)\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "→ 'sign-based 방향 일치'는 ", options: { color: GREY_DARK } },
    { text: "두 번째 이미지가 있다는 사실 자체", options: { bold: true, color: NAVY } },
    { text: "에서 비롯된 generic perturbation.\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "정작 디지트 effect는 adopt에 집중.", options: { color: NAVY, italic: true } },
  ], { x: 9.3, y: 2.05, w: 3.6, h: 4.5, fontFace: BODY_FONT, fontSize: 11, margin: 0 });

  addBgFooter(s, 12, TOTAL, "5 Distance & Plausibility Window");
}

// ====================================================================
// Slide 13 — E5d per-dataset cutoff
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Per-dataset cutoff 검증 (E5d)",
    "ChartQA: S1-only relative · MathVista: C3 FAIL (γ-α로 이전)");

  const img1 = path.join(FIG_DIR, "E5d_chartqa_decay.png");
  if (fs.existsSync(img1)) {
    s.addImage({ path: img1, x: 0.4, y: 1.75, w: 6.4, h: 4.7, sizing: { type: "contain", w: 6.4, h: 4.7 } });
  }
  const img2 = path.join(FIG_DIR, "E5d_mathvista_decay.png");
  if (fs.existsSync(img2)) {
    s.addImage({ path: img2, x: 6.9, y: 1.75, w: 6.0, h: 4.7, sizing: { type: "contain", w: 6.0, h: 4.7 } });
  }

  s.addText("ChartQA: [done] S1-only |a-gt| <= max(1, 0.1*gt) 채택.   MathVista E5d: [!] 모든 stratum diffuse -> C3 FAIL -> gamma-alpha로 cross-model 재검증.",
    { x: 0.5, y: 6.4, w: 12.3, h: 0.5, fontFace: BODY_FONT, fontSize: 12, color: NAVY, italic: true, align: "center", margin: 0 });

  addBgFooter(s, 13, TOTAL, "5 Distance & Plausibility Window");
}

// ====================================================================
// Slide 14 — Cross-model E5e (cross-dataset summary)
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Cross-model × cross-dataset (E5e)",
    "wrong-base S1 adopt_rate (M2) heatmap");

  const img = path.join(FIG_DIR, "paper_cross_dataset_summary.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: 0.6, y: 1.7, w: 12.1, h: 5.1, sizing: { type: "contain", w: 12.1, h: 5.1 } });
  }
  addBgFooter(s, 14, TOTAL, "5 Distance & Plausibility Window");
}

// ====================================================================
// Slide 15 — MathVista γ-α highlight
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "MathVista γ-α — categorical-replace regime",
    "gemma3-27b adopt(a, wrong-base) = 0.194 — 본 프로그램에서 가장 큰 단일 셀");

  const img = path.join(FIG_DIR, "paper_E5e_mathvista_bars.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: 0.4, y: 1.7, w: 7.5, h: 4.7, sizing: { type: "contain", w: 7.5, h: 4.7 } });
  }

  // Highlight box
  s.addShape("rect", { x: 8.2, y: 1.85, w: 4.7, h: 4.7,
    fill: { color: ICE }, line: { color: ICE } });
  s.addShape("rect", { x: 8.2, y: 1.85, w: 0.08, h: 4.7, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD } });
  s.addText("두 regime의 발견", { x: 8.4, y: 1.95, w: 4.4, h: 0.4,
    fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Graded-tilt (VQAv2/TallyQA/ChartQA)\n", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "anchor가 search direction을 비례적 이동 → df 큼, adopt 작음\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "Categorical-replace (MathVista)\n", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "anchor가 base 답을 통째로 교체 → adopt 큼, df = 0\n\n", options: { color: GREY_DARK, breakLine: true } },
    { text: "gemma3-27b 모델 + MathVista 조합에서 ", options: { color: GREY_DARK } },
    { text: "adopt 0.194", options: { bold: true, color: NAVY } },
    { text: " 기록. 디지트 픽셀 인과 (a − m gap = ", options: { color: GREY_DARK } },
    { text: "+15.2 pp", options: { bold: true, color: ACCENT_GOLD } },
    { text: ") 보존.", options: { color: GREY_DARK } },
  ], { x: 8.4, y: 2.4, w: 4.4, h: 4.0, fontFace: BODY_FONT, fontSize: 12, margin: 0 });

  addBgFooter(s, 15, TOTAL, "5 Distance & Plausibility Window");
}

// ====================================================================
// Slide 16 — L1 confidence quartile
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Confidence-modulated anchoring (L1, §6)",
    "base prediction의 entropy ↑ → anchor pull ↑ (graded)");

  const img = path.join(FIG_DIR, "paper_L1_confidence_quartile.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: 0.4, y: 1.7, w: 8.5, h: 4.7, sizing: { type: "contain", w: 8.5, h: 4.7 } });
  }
  s.addText([
    { text: "측정 방법\n", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "각 base inference에서 답 토큰의 top-k entropy 계산 → 모델·데이터셋 cell마다 4분위로 나눔\n\n",
      options: { color: GREY_DARK, breakLine: true } },
    { text: "Q1 = 가장 confident 25%\n", options: { color: NAVY, breakLine: true } },
    { text: "Q4 = 가장 uncertain 25%\n\n", options: { color: NAVY, breakLine: true } },
    { text: "헤드라인\n", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "Q4 − Q1 mean df gap = ", options: { color: GREY_DARK } },
    { text: "+0.128", options: { bold: true, color: ACCENT_GOLD } },
    { text: "\n34 cells 중 18개가 fully monotone Q1<Q2<Q3<Q4", options: { color: GREY_DARK } },
  ], { x: 9.1, y: 1.95, w: 3.8, h: 4.5, fontFace: BODY_FONT, fontSize: 11, margin: 0 });

  addBgFooter(s, 16, TOTAL, "6 Confidence Modulation");
}

// ====================================================================
// Slide 17 — Q1 vs Q4 worked example
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Q1 vs Q4 — worked example",
    "E5c VQAv2 wrong-base S1, llava-interleave-7b, entropy_top_k 기준");

  const head = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: HEADER_FONT, fontSize: 13 } });
  const cell = (t, h = false) => ({ text: t, options: { fontFace: BODY_FONT, fontSize: 12, color: GREY_DARK, bold: h } });
  const data = [
    [head("quartile"), head("어떤 records?"), head("base 정답률"), head("anchor adopt"), head("direction-follow")],
    [cell("Q1", true), cell("가장 confident (entropy 낮음)"), cell("0.77 (대부분 정답)"), cell("0.077"), cell("0.040")],
    [cell("Q2"), cell("upper-middle"), cell("0.50"), cell("0.090"), cell("0.080")],
    [cell("Q3"), cell("lower-middle"), cell("0.27"), cell("0.110"), cell("0.090")],
    [cell("Q4", true), cell("가장 uncertain (entropy 높음)"), cell("0.07 (대부분 오답)"),
     { text: "0.147", options: { fontFace: BODY_FONT, fontSize: 12, color: NAVY, bold: true } },
     { text: "0.113", options: { fontFace: BODY_FONT, fontSize: 12, color: NAVY, bold: true } }],
    [{ text: "Δ (Q4−Q1)", options: { fill: { color: ICE }, fontFace: HEADER_FONT, bold: true, color: NAVY } },
     { text: "—", options: { fill: { color: ICE } } },
     { text: "−0.70", options: { fill: { color: ICE }, color: NAVY } },
     { text: "+0.070 (+7.0 pp)", options: { fill: { color: ICE }, bold: true, color: ACCENT_GOLD } },
     { text: "+0.074 (+7.4 pp)", options: { fill: { color: ICE }, bold: true, color: ACCENT_GOLD } }],
  ];
  s.addTable(data, { x: 0.5, y: 1.85, w: 12.3, colW: [1.5, 4.0, 2.3, 2.3, 2.2],
    border: { type: "solid", pt: 0.5, color: GREY_LIGHT } });

  // Pull footnote up close to table
  s.addShape("rect", { x: 0.5, y: 4.95, w: 12.3, h: 1.5,
    fill: { color: ICE }, line: { color: ICE } });
  s.addShape("rect", { x: 0.5, y: 4.95, w: 0.08, h: 1.5, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD } });
  s.addText([
    { text: "관계: ", options: { bold: true, color: NAVY } },
    { text: "Phase A의 wrong/correct binary는 confidence quartile의 ", options: { color: GREY_DARK } },
    { text: "coarse projection", options: { bold: true, color: NAVY } },
    { text: ". Q1 mean exact_match=0.77 (정답 ∼ correct), Q4 mean=0.07 (오답 ∼ wrong).\n그러나 quartile은 ", options: { color: GREY_DARK } },
    { text: "confidently wrong / lucky correct", options: { bold: true, color: NAVY } },
    { text: " 같은 예외 케이스를 더 정밀하게 분리한다.", options: { color: GREY_DARK } },
  ], { x: 0.75, y: 5.05, w: 11.9, h: 1.3, fontFace: BODY_FONT, fontSize: 13, valign: "middle", margin: 0 });

  addBgFooter(s, 17, TOTAL, "6 Confidence Modulation");
}

// ====================================================================
// Slide 18 — E1/E1b attention archetypes
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Attention mass 4-archetype (E1 + E1b)",
    "6 모델 panel n=200 stratified — peak 위치 + δ + 메커니즘");

  const head = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: HEADER_FONT, fontSize: 12 } });
  const cell = (t) => ({ text: t, options: { fontFace: BODY_FONT, fontSize: 11, color: GREY_DARK } });
  const data = [
    [head("Archetype"), head("Encoder"), head("Peak L"), head("δ (anchor − neutral)"), head("메커니즘 (budget decomposition)")],
    [cell("SigLIP-Gemma early"), cell("SigLIP-So (gemma4-e4b)"), cell("L5 / 42 (12% depth)"), cell("+0.050"), cell("text-stealing (δ_text −0.038)")],
    [cell("Mid-stack cluster"), cell("CLIP-ViT (llava-1.5)"), cell("L16 / 32"), cell("+0.019"), cell("text-stealing")],
    [cell("Mid-stack cluster"), cell("InternViT (internvl3-8b)"), cell("L14 / 28"), cell("+0.019"), cell("text-stealing")],
    [cell("Mid-stack cluster"), cell("ConvNeXt (convllava-7b)"), cell("L16 / 32"), cell("+0.022"), cell("text-stealing (H3 falsified)")],
    [cell("Qwen-ViT late"), cell("Qwen-ViT (qwen2.5-vl-7b)"), cell("L22 / 28 (82%)"), cell("+0.015"), cell("target-stealing (anchor takes from target image)")],
    [cell("FastVLM late"), cell("FastViT (fastvlm-7b)"), cell("L22"), cell("+0.047"), cell("text-stealing, A7 gap +0.086 (n=75)")],
  ];
  s.addTable(data, { x: 0.5, y: 1.85, w: 12.3, colW: [2.4, 3.0, 2.0, 2.4, 2.5],
    border: { type: "solid", pt: 0.5, color: GREY_LIGHT } });

  // Pull footnote up close to table
  s.addShape("rect", { x: 0.5, y: 5.4, w: 12.3, h: 1.3,
    fill: { color: ICE }, line: { color: ICE } });
  s.addShape("rect", { x: 0.5, y: 5.4, w: 0.08, h: 1.3, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD } });
  s.addText("H3 (\"ConvNeXt < ViT\") falsified — 3개 다른 encoder가 같은 mid-stack text-stealing profile.\n→ \"post-projection LLM stack depth\"가 axis (H6: 2-axis decomposition).",
    { x: 0.75, y: 5.5, w: 11.9, h: 1.1, fontFace: BODY_FONT, fontSize: 13, color: NAVY, italic: true, valign: "middle", margin: 0 });

  addBgFooter(s, 18, TOTAL, "7 Attention");
}

// ====================================================================
// Slide 19 — Causal ablation E1d
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Causal ablation (E1d)",
    "single-layer null · stack-wide breaks fluency · upper-half mitigation locus");

  const head = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: HEADER_FONT, fontSize: 12 } });
  const cell = (t, c = GREY_DARK, b = false) => ({ text: t, options: { fontFace: BODY_FONT, fontSize: 11, color: c, bold: b } });
  const data = [
    [head("Ablation 모드"), head("Δ direction_follow (6 모델)"), head("Fluency 영향"), head("Use case")],
    [cell("ablate_layer0"), cell("[−0.027, +0.005] — null on 6/6"), cell("clean"), cell("layer-0 control: 단일 레이어 효과 없음")],
    [cell("ablate_peak (E1b 피크)"), cell("[−0.032, +0.020] — null on 6/6"), cell("clean"), cell("multi-layer redundancy 입증")],
    [cell("ablate_lower_half"), cell("varies by archetype"), cell("clean → moderate"), cell("아키텍처별 효과 다름")],
    [cell("ablate_upper_half", NAVY, true), cell("[−0.115, −0.055] on 6/6", NAVY, true), cell("clean on 4/6 (mid-stack + Qwen)", NAVY, true), cell("[selected] E4 prototype locus", NAVY, true)],
    [cell("ablate_all (stack-wide)"), cell("[−0.22, −0.11] universal"), cell("breaks on 3/6 (mean-dist 4–6×)"), cell("upper bound only")],
  ];
  s.addTable(data, { x: 0.5, y: 1.85, w: 12.3, colW: [2.5, 3.4, 2.6, 3.8],
    border: { type: "solid", pt: 0.5, color: GREY_LIGHT } });

  // Pull footnote up close to table
  s.addShape("rect", { x: 0.5, y: 5.4, w: 12.3, h: 1.3,
    fill: { color: ICE }, line: { color: ICE } });
  s.addShape("rect", { x: 0.5, y: 5.4, w: 0.08, h: 1.3, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD } });
  s.addText("핵심: anchor 효과는 stack 전반에 redundantly encoded — single-layer 차단은 무의미. upper-half multi-layer 차단만이 architecture-blind mitigation locus.",
    { x: 0.75, y: 5.5, w: 11.9, h: 1.1, fontFace: BODY_FONT, fontSize: 13, color: NAVY, italic: true, valign: "middle", margin: 0 });

  addBgFooter(s, 19, TOTAL, "7 Attention");
}

// ====================================================================
// Slide 20 — Mitigation E4 free lunch
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Mitigation — \"free lunch\" (E4)",
    "mid-stack cluster 3 모델 Phase 2 full validation: df ↓ · em ↑ · acc invariant");

  const head = (t) => ({ text: t, options: { fill: { color: NAVY }, color: WHITE, bold: true, fontFace: HEADER_FONT, fontSize: 12 } });
  const cell = (t, c = GREY_DARK, b = false) => ({ text: t, options: { fontFace: BODY_FONT, fontSize: 12, color: c, bold: b } });
  const data = [
    [head("Model"), head("s* (chosen strength)"), head("Δ direction_follow"), head("Δ exact_match"), head("Δ accuracy_vqa(b)")],
    [cell("llava-1.5-7b", NAVY, true),
     cell("−3.0"),
     cell("0.258 → 0.212 (−17.7 % rel)", NAVY, true),
     cell("0.334 → 0.342 (+0.77 pp)", ACCENT_GOLD, true),
     cell("invariant", GREY_MED)],
    [cell("convllava-7b", NAVY, true),
     cell("−2.0"),
     cell("0.228 → 0.204 (−10.6 %)", NAVY, true),
     cell("0.352 → 0.365 (+1.30 pp)", ACCENT_GOLD, true),
     cell("invariant", GREY_MED)],
    [cell("internvl3-8b", NAVY, true),
     cell("−0.5"),
     cell("0.103 → 0.097 (−5.8 %)", NAVY, true),
     cell("0.590 → 0.595 (+0.49 pp)", ACCENT_GOLD, true),
     cell("invariant", GREY_MED)],
  ];
  s.addTable(data, { x: 0.5, y: 1.85, w: 12.3, colW: [2.4, 2.0, 3.4, 2.7, 1.8],
    border: { type: "solid", pt: 0.5, color: GREY_LIGHT } });

  // Insight callout
  s.addShape("rect", { x: 0.6, y: 4.85, w: 12.1, h: 1.9,
    fill: { color: ICE }, line: { color: ICE } });
  s.addShape("rect", { x: 0.6, y: 4.85, w: 0.08, h: 1.9, fill: { color: ACCENT_GOLD }, line: { color: ACCENT_GOLD } });
  s.addText("\"Free lunch\" 의미", {
    x: 0.85, y: 4.95, w: 11.5, h: 0.5,
    fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "df ↓ 함과 동시에 ", options: { color: GREY_DARK } },
    { text: "em이 오히려 상승", options: { bold: true, color: NAVY } },
    { text: " (anchor에 끌렸던 wrong 답을 회복) → mitigation으로 ", options: { color: GREY_DARK } },
    { text: "정확도 손실 없음", options: { bold: true, color: ACCENT_GOLD } },
    { text: ".\nbase condition (no anchor) 의 acc는 ", options: { color: GREY_DARK } },
    { text: "완전히 invariant", options: { bold: true, color: NAVY } },
    { text: " — anchor-condition-specific hook (single-image inference에 영향 0).", options: { color: GREY_DARK } },
  ], { x: 0.85, y: 5.5, w: 11.7, h: 1.2, fontFace: BODY_FONT, fontSize: 13, margin: 0 });

  addBgFooter(s, 20, TOTAL, "7 Attention");
}

// ====================================================================
// Slide 21 — Future Work
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  addTitle(s, "Future Work (§8)",
    "VLM/LLM architectural diff 가 가장 priority 높음");

  const items = [
    { num: "1", h: "VLM vs LLM architectural diff (preferred)",
      b: "VLM의 anchor pull은 vision encoder → projection → LLM stack의 mid-stack에서 일어남 (E1b). LLM에 anchor를 text로 줬을 때 어느 layer 분포에서 통합되는가? Layer-wise integration profile 비교로 architectural diff 입증." },
    { num: "2", h: "Image vs text anchor 비교",
      b: "anchor를 image로 줬을 때 vs. 동일 anchor value를 텍스트로 줬을 때 effect 차이. cross-modal vs text-only LLM anchoring 정량 비교." },
    { num: "3", h: "Reasoning-mode VLM (γ-β)",
      b: "Qwen3-VL thinking 모드 활성화 + MathVista. reasoning chain 안에서 anchor가 amplify/suppress 되는가? VLMBias 의 \"reasoning models can be more biased\" 를 cross-check." },
  ];
  items.forEach((it, i) => {
    const y = 1.95 + i * 1.55;
    s.addShape("oval", { x: 0.6, y: y + 0.2, w: 0.65, h: 0.65,
      fill: { color: i === 0 ? ACCENT_GOLD : NAVY }, line: { color: i === 0 ? ACCENT_GOLD : NAVY } });
    s.addText(it.num, { x: 0.6, y: y + 0.2, w: 0.65, h: 0.65,
      fontFace: HEADER_FONT, fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText(it.h, { x: 1.5, y: y, w: 11.3, h: 0.5,
      fontFace: HEADER_FONT, fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(it.b, { x: 1.5, y: y + 0.5, w: 11.3, h: 1.0,
      fontFace: BODY_FONT, fontSize: 12, color: GREY_DARK, margin: 0 });
  });

  addBgFooter(s, 21, TOTAL, "8 Future Work");
}

// ====================================================================
// Slide 22 — Conclusion
// ====================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Conclusion", { x: 0.6, y: 0.5, w: 12.1, h: 0.7,
    fontFace: HEADER_FONT, fontSize: 30, bold: true, color: WHITE, margin: 0 });

  s.addText([
    { text: "Cross-modal numerical anchoring in VLMs is\n", options: { color: WHITE } },
    { text: "uncertainty-modulated graded pull,\n", options: { color: ACCENT_GOLD, bold: true } },
    { text: "concentrating on a digit-pixel cue,\n", options: { color: WHITE } },
    { text: "and mitigated by an encoder-blind upper-half locus —\n", options: { color: WHITE } },
    { text: "without sacrificing accuracy.", options: { color: ACCENT_GOLD, bold: true } },
  ], { x: 0.6, y: 1.6, w: 12.1, h: 2.2,
    fontFace: HEADER_FONT, fontSize: 22, color: WHITE, margin: 0 });

  // 4 takeaways
  const takeaways = [
    { num: "01", h: "Behavioural", b: "wrong > correct on direction-follow +6.9–19.6 pp · L1 confidence quartile Q4-Q1 +0.128" },
    { num: "02", h: "Causal", b: "anchor − masked gap 0.5–15 pp on wrong-base S1 across 4 datasets · digit-pixel는 effect의 인과" },
    { num: "03", h: "Mechanistic", b: "4 archetypes의 attention peak · upper-half ablation −5.5 to −11.5 pp on 6/6 architecture-blind" },
    { num: "04", h: "Mitigation", b: "mid-stack cluster 3-model: df −5.8 to −17.7 % rel · em +0.5 to +1.3 pp · acc invariant — \"free lunch\"" },
  ];
  takeaways.forEach((t, i) => {
    const x = 0.6 + (i % 2) * 6.05;
    const y = 4.1 + Math.floor(i / 2) * 1.45;
    s.addText(t.num, { x, y, w: 1.0, h: 1.3,
      fontFace: HEADER_FONT, fontSize: 36, bold: true, color: ACCENT_GOLD, margin: 0 });
    s.addText(t.h, { x: x + 1.1, y, w: 4.7, h: 0.35,
      fontFace: HEADER_FONT, fontSize: 14, bold: true, color: ICE, margin: 0 });
    s.addText(t.b, { x: x + 1.1, y: y + 0.4, w: 4.7, h: 0.95,
      fontFace: BODY_FONT, fontSize: 11, color: WHITE, margin: 0 });
  });

  s.addText("EMNLP 2026 Main · ARR May 25", { x: 0.6, y: 7.0, w: 12.1, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, color: ICE, italic: true, margin: 0 });
}

// ---- write ----
pres.writeFile({ fileName: OUT_PATH }).then(() => {
  console.log("wrote", OUT_PATH);
});
