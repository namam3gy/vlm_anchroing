// Korean paper-summary deck for VLM cross-modal anchoring paper.
// Build: install pptxgenjs in any tmp dir, drop this file there, `node paper_summary_kr.build.js`.
// Output: paper_summary_kr.pptx (20 slides, LAYOUT_WIDE 13.3" × 7.5")
//
// Sourced from `docs/paper/sections/0[1-8]_*.md` + `references/roadmap.md` §3.3
// + `docs/insights/_data/E5c_per_cell.csv` as of commit bb3870c (2026-04-29).
// Numbers are pinned to the C-form M2 metrics (§3.4 of the paper draft).

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const REPO = "/mnt/ddn/prod-runs/thyun.park/src/vlm_anchroing";

// ----------------------------------------------------------------------------
// Palette + typography
// ----------------------------------------------------------------------------
const NAVY = "1E2761";      // primary
const TERRA = "B85042";     // accent — warm anchor pull
const GOLD = "D4AF37";      // numeric highlight
const CREAM = "F8F6F2";     // light bg
const CHARCOAL = "333333";  // body
const MUTED = "6B7280";     // captions / muted text
const GREEN = "2C5F2D";     // positive results

const TITLE_FONT = "Calibri";
const BODY_FONT = "Calibri";

const W = 13.333, H = 7.5; // LAYOUT_WIDE

// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------
function addPageHeader(slide, title, subtitle) {
  // accent vertical bar
  slide.addShape("rect", {
    x: 0.4, y: 0.55, w: 0.10, h: 0.7,
    fill: { color: TERRA }, line: { color: TERRA, width: 0 },
  });
  slide.addText(title, {
    x: 0.65, y: 0.45, w: W - 1.5, h: 0.6,
    fontFace: TITLE_FONT, fontSize: 26, bold: true,
    color: NAVY, valign: "middle", margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.65, y: 1.05, w: W - 1.5, h: 0.35,
      fontFace: BODY_FONT, fontSize: 12, italic: true,
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
  slide.addText(`${pageNum} / 20`, {
    x: W - 1.2, y: H - 0.32, w: 0.8, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, color: MUTED,
    align: "right", valign: "middle", margin: 0,
  });
}

function tableHeaderCell(text, opts = {}) {
  return {
    text,
    options: {
      bold: true, color: "FFFFFF", fill: { color: NAVY },
      fontSize: 11, fontFace: BODY_FONT, align: "center", valign: "middle",
      margin: 0.05,
      ...opts,
    },
  };
}

function tableBodyCell(text, opts = {}) {
  return {
    text: String(text),
    options: {
      color: CHARCOAL, fontSize: 11, fontFace: BODY_FONT,
      align: "center", valign: "middle",
      fill: { color: "FFFFFF" }, margin: 0.05,
      ...opts,
    },
  };
}

function tableHighlight(text, opts = {}) {
  return tableBodyCell(text, {
    bold: true, color: TERRA, fill: { color: CREAM }, ...opts,
  });
}

// ----------------------------------------------------------------------------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "namam3gy";
pres.title = "VLM 교차모달 숫자 anchoring (paper summary, KR)";

// =============================================================================
// SLIDE 1 — Title
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape("rect", {
    x: 0, y: 6.7, w: W, h: 0.8,
    fill: { color: TERRA }, line: { color: TERRA, width: 0 },
  });
  s.addShape("rect", {
    x: 0.6, y: 1.0, w: 0.18, h: 4.2,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("VLM의 교차모달 숫자 anchoring", {
    x: 1.0, y: 1.5, w: 11, h: 1.1,
    fontFace: TITLE_FONT, fontSize: 44, bold: true,
    color: "FFFFFF", margin: 0,
  });
  s.addText("Cross-modal numerical anchoring in VLMs", {
    x: 1.0, y: 2.55, w: 11, h: 0.5,
    fontFace: TITLE_FONT, fontSize: 20, italic: true,
    color: "CADCFC", margin: 0,
  });
  s.addText([
    { text: "불확실성에 비례한 ", options: { color: "FFFFFF" } },
    { text: "graded pull", options: { color: GOLD, bold: true } },
    { text: " · digit-pixel 인과성 · upper-half mitigation · reasoning이 anchoring을 ", options: { color: "FFFFFF" } },
    { text: "강화", options: { color: GOLD, bold: true } },
    { text: "한다", options: { color: "FFFFFF" } },
  ], {
    x: 1.0, y: 3.3, w: 11, h: 0.6,
    fontFace: BODY_FONT, fontSize: 16, margin: 0,
  });
  s.addText("EMNLP 2026 Main · ARR May 25 target · 7 open-weight VLMs · 4 numeric VQA datasets", {
    x: 1.0, y: 4.2, w: 11, h: 0.4,
    fontFace: BODY_FONT, fontSize: 13, color: "CADCFC", italic: true, margin: 0,
  });
  s.addText("paper §1–§8 요약 · 한국어판 · 2026-04-29", {
    x: 1.0, y: 6.85, w: 11, h: 0.5,
    fontFace: BODY_FONT, fontSize: 12, color: "FFFFFF", bold: true,
    valign: "middle", margin: 0,
  });
}

// =============================================================================
// SLIDE 2 — TL;DR
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "한 줄 요약 (TL;DR)", "§1.3 Headline claim");

  s.addShape("rect", {
    x: 0.6, y: 1.7, w: W - 1.2, h: 1.5,
    fill: { color: NAVY }, line: { color: NAVY, width: 0 },
  });
  s.addText([
    { text: "VLM에 무관한 숫자 이미지를 함께 주면 ", options: { color: "FFFFFF" } },
    { text: "예측이 그 숫자 쪽으로 끌립니다", options: { color: GOLD, bold: true } },
    { text: ".", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "이 효과는 ", options: { color: "FFFFFF" } },
    { text: "categorical capture가 아니라 graded pull", options: { color: GOLD, bold: true } },
    { text: "이고, ", options: { color: "FFFFFF" } },
    { text: "기저 예측의 불확실성에 비례", options: { color: GOLD, bold: true } },
    { text: "하며, ", options: { color: "FFFFFF" } },
    { text: "anchor 이미지의 digit pixel이 인과적", options: { color: GOLD, bold: true } },
    { text: "입니다.", options: { color: "FFFFFF" } },
  ], {
    x: 0.85, y: 1.85, w: W - 1.7, h: 1.2,
    fontFace: BODY_FONT, fontSize: 18, valign: "middle", margin: 0,
  });

  // three pillars
  const pillars = [
    { num: "1", title: "Graded vs. categorical", body: "wrong-base direction-follow이 correct-base보다 +6.9~+19.6 pp 큼.\n그러나 paired adopt(literal copy)는 2-7%에 불과 — 모델은 anchor 숫자를 그대로 출력하지 않고, 자기 baseline 예측을 anchor 쪽으로 기울일 뿐." },
    { num: "2", title: "Digit-pixel 인과성", body: "anchor 이미지에서 숫자 픽셀만 OpenCV inpaint로 지우면 효과가 generic distractor 수준으로 떨어짐.\n배경 scene이 아니라 digit pixels이 원인." },
    { num: "3", title: "Confidence-modulated", body: "answer-token entropy로 4분위 분할하면 direction-follow가 단조증가 (Q4-Q1 = +15.2 pp).\nPhase-A의 wrong/correct 이분법은 이 연속 구조의 거친 투영." },
  ];
  pillars.forEach((p, i) => {
    const x = 0.6 + i * 4.15;
    s.addShape("roundRect", {
      x, y: 3.6, w: 3.95, h: 3.4,
      fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 },
      rectRadius: 0.08,
    });
    s.addShape("ellipse", {
      x: x + 0.25, y: 3.85, w: 0.7, h: 0.7,
      fill: { color: TERRA }, line: { color: TERRA, width: 0 },
    });
    s.addText(p.num, {
      x: x + 0.25, y: 3.85, w: 0.7, h: 0.7,
      fontFace: TITLE_FONT, fontSize: 26, bold: true, color: "FFFFFF",
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(p.title, {
      x: x + 1.05, y: 3.85, w: 2.7, h: 0.7,
      fontFace: TITLE_FONT, fontSize: 14, bold: true, color: NAVY,
      valign: "middle", margin: 0,
    });
    s.addText(p.body, {
      x: x + 0.25, y: 4.7, w: 3.45, h: 2.2,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL,
      valign: "top", margin: 0,
    });
  });

  addPageFooter(s, 2, "§1 · 한 줄 요약");
}

// =============================================================================
// SLIDE 3 — 4-condition setup with example images
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "실험 셋업: 4-condition 자극 (per sample_instance)",
    "§3.1 — 동일한 (target image, question) 위에 두 번째 이미지를 b/a/m/d 4가지로 바꿔가며 모델 출력을 비교");

  const conds = [
    { label: "b (target_only)",
      sub: "baseline · 두 번째 이미지 없음",
      pred: "pred_b",
      img: path.join(REPO, "inputs/vqav2_number_val/images/000000000139.jpg"),
      caption: "타겟 이미지만 단독 제시" },
    { label: "a (target + anchor)",
      sub: "두 번째 이미지: 숫자가 그려진 무관 이미지",
      pred: "pred_a",
      img: path.join(REPO, "inputs/irrelevant_number/3.png"),
      caption: "FLUX 생성 — Arabic numeral 1개 (예: 3)" },
    { label: "m (target + masked)",
      sub: "anchor와 동일 scene이지만 digit 픽셀만 inpaint",
      pred: "pred_m",
      img: path.join(REPO, "inputs/irrelevant_number_masked/3.png"),
      caption: "OpenCV Telea inpaint, OCR-validated" },
    { label: "d (target + neutral)",
      sub: "digit-free FLUX 이미지 (generic 2-image distractor)",
      pred: "pred_d",
      img: path.join(REPO, "inputs/irrelevant_neutral/13.png"),
      caption: "scene-stylistic 매칭, 숫자 없음" },
  ];

  conds.forEach((c, i) => {
    const x = 0.4 + i * 3.18;
    s.addShape("roundRect", {
      x, y: 1.7, w: 3.05, h: 5.4,
      fill: { color: CREAM }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
    });
    s.addText(c.label, {
      x: x + 0.1, y: 1.8, w: 2.85, h: 0.4,
      fontFace: TITLE_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
    });
    s.addText(c.sub, {
      x: x + 0.1, y: 2.2, w: 2.85, h: 0.6,
      fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, margin: 0,
    });
    if (fs.existsSync(c.img)) {
      s.addImage({ path: c.img, x: x + 0.3, y: 2.85, w: 2.45, h: 2.45,
        sizing: { type: "contain", w: 2.45, h: 2.45 } });
    }
    s.addText(c.caption, {
      x: x + 0.1, y: 5.45, w: 2.85, h: 0.7,
      fontFace: BODY_FONT, fontSize: 9, color: MUTED, italic: true,
      align: "center", margin: 0,
    });
    s.addShape("rect", {
      x: x + 0.1, y: 6.25, w: 2.85, h: 0.65,
      fill: { color: NAVY }, line: { color: NAVY, width: 0 },
    });
    s.addText(c.pred, {
      x: x + 0.1, y: 6.25, w: 2.85, h: 0.65,
      fontFace: TITLE_FONT, fontSize: 16, bold: true, color: GOLD,
      align: "center", valign: "middle", margin: 0,
    });
  });

  addPageFooter(s, 3, "§3 · Setup");
}

// =============================================================================
// SLIDE 4 — 용어 정리 (notation)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "용어 정리 (notation)",
    "§3.1 — 코드 / 논문 / 대시보드에 일관되게 등장");

  const rows = [
    [tableHeaderCell("기호"), tableHeaderCell("의미"), tableHeaderCell("근거")],
    [tableBodyCell("pred_b"), tableBodyCell("baseline 예측 (target_only condition의 출력)"), tableBodyCell("§3.1, predictions.jsonl 'prediction' (condition=target_only)")],
    [tableBodyCell("pred_a"), tableBodyCell("anchor condition의 모델 출력"), tableBodyCell("a-arm 행")],
    [tableBodyCell("pred_m"), tableBodyCell("masked condition의 모델 출력 (digit pixel inpaint)"), tableBodyCell("m-arm 행")],
    [tableBodyCell("pred_d"), tableBodyCell("neutral condition의 모델 출력 (digit-free 2nd image)"), tableBodyCell("d-arm 행")],
    [tableBodyCell("anchor"), tableBodyCell("anchor 이미지에 그려진 숫자 (filename = 값)"), tableBodyCell("inputs/irrelevant_number/<value>.png")],
    [tableBodyCell("gt"), tableBodyCell("ground truth — 정답 숫자"), tableBodyCell("dataset의 정답 라벨")],
    [tableHighlight("pa, pb"), tableHighlight("논문 본문 약식 표기 — pa = pred_a, pb = pred_b"), tableHighlight("§3.4 metric 정의")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.7, w: W - 1.2, colW: [1.6, 5.5, 5.0],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
  });

  s.addShape("rect", {
    x: 0.6, y: 5.6, w: W - 1.2, h: 1.5,
    fill: { color: "FFFFFF" }, line: { color: TERRA, width: 1 },
  });
  s.addText("핵심 비교 3가지", {
    x: 0.8, y: 5.65, w: 11.6, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "(a − d): ", options: { bold: true, color: NAVY } },
    { text: "anchoring과 generic 2-image distraction 분리 (digit이 핵심인지)", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "(a − m): ", options: { bold: true, color: NAVY } },
    { text: "digit pixel만의 기여 — 같은 scene을 inpaint한 m을 빼면 픽셀 효과 분리", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "(a, base-wrong) − (a, base-correct): ", options: { bold: true, color: NAVY } },
    { text: "불확실성 modulation (§5.2 Phase A → §6 confidence quartile로 연속화)", options: { color: CHARCOAL } },
  ], {
    x: 0.8, y: 6.0, w: 11.6, h: 1.05,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  addPageFooter(s, 4, "§3 · Notation");
}

// =============================================================================
// SLIDE 5 — Canonical M2 metrics
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "핵심 지표 M2 (C-form, 2026-04-28 확정)",
    "§3.4 — 모든 §5/§6/§7 표가 이 정의에 매핑");

  // Three boxes for the three metrics
  const metrics = [
    { name: "adopt_rate (M2 paired)",
      formula: "#(pa = anchor AND pb ≠ anchor) / #(pb ≠ anchor)",
      kr: "기저 예측이 anchor와 다른 샘플 중, anchor 추가 후 anchor 값을 그대로 출력하는 비율 (literal copy)",
      detail: "분모에서 base-prediction confound 제거 — 우연히 base가 anchor와 같았던 케이스는 빼고 계산." },
    { name: "direction_follow_rate (C-form)",
      formula: "#( (pa − pb) · (anchor − pb) > 0 AND pa ≠ pb ) / #(numeric pair AND anchor present)",
      kr: "예측이 baseline에서 anchor 방향으로 움직였는지의 sign-based 측정 (gt 미사용)",
      detail: "pa가 pb 기준으로 anchor 쪽 부호로 이동했으면 1, 아니면 0. C-form은 stimulus draw에 robust." },
    { name: "exact_match",
      formula: "#(pa = gt) / #(numeric pair)",
      kr: "anchor 조건에서의 정답 일치율 (per-arm accuracy)",
      detail: "anchoring으로 떨어진 정확도를 직접 측정 — §7.4에서 mitigation 결과 검증에 사용." },
  ];
  metrics.forEach((m, i) => {
    const y = 1.65 + i * 1.7;
    s.addShape("rect", {
      x: 0.6, y, w: 0.10, h: 1.55,
      fill: { color: TERRA }, line: { color: TERRA, width: 0 },
    });
    s.addText(m.name, {
      x: 0.85, y: y + 0.05, w: 5.5, h: 0.4,
      fontFace: TITLE_FONT, fontSize: 14, bold: true, color: NAVY, margin: 0,
    });
    s.addText(m.formula, {
      x: 0.85, y: y + 0.45, w: 11.5, h: 0.4,
      fontFace: "Consolas", fontSize: 11, color: TERRA, margin: 0,
    });
    s.addText(m.kr, {
      x: 0.85, y: y + 0.85, w: 11.5, h: 0.35,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, bold: true, margin: 0,
    });
    s.addText(m.detail, {
      x: 0.85, y: y + 1.18, w: 11.5, h: 0.35,
      fontFace: BODY_FONT, fontSize: 10, color: MUTED, italic: true, margin: 0,
    });
  });

  s.addText([
    { text: "왜 C-form인가? ", options: { bold: true, color: TERRA } },
    { text: "pb (baseline)을 기준으로 보면 anchor draw에 robust하고 gt에 의존하지 않음. anchor·gt form (옛 metric)은 stimulus에 따라 indicator가 흔들림 → 폐기 (2026-04-28).", options: { color: CHARCOAL } },
  ], {
    x: 0.6, y: 6.85, w: W - 1.2, h: 0.4,
    fontFace: BODY_FONT, fontSize: 11, italic: true, valign: "middle", margin: 0,
  });

  addPageFooter(s, 5, "§3 · Metrics");
}

// =============================================================================
// SLIDE 6 — Datasets + Models
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "데이터셋 + 모델 패널", "§4.1–§4.3");

  // Datasets table (left)
  s.addText("데이터셋 (4종, integer-GT 부분집합)", {
    x: 0.6, y: 1.6, w: 6.0, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  const dsRows = [
    [tableHeaderCell("Dataset"), tableHeaderCell("GT 범위"), tableHeaderCell("샘플"), tableHeaderCell("역할")],
    [tableBodyCell("VQAv2 number"), tableBodyCell("0–8"), tableBodyCell("17,730"), tableBodyCell("메인 패널")],
    [tableBodyCell("TallyQA"), tableBodyCell("0–8 (cap)"), tableBodyCell("38,245"), tableBodyCell("counting under 가림/모호")],
    [tableBodyCell("ChartQA"), tableBodyCell("integer 1–1000"), tableBodyCell("5,390"), tableBodyCell("타겟 안에 정답이 보이는 케이스")],
    [tableBodyCell("MathVista"), tableBodyCell("integer 1–1000"), tableBodyCell("385 (testmini)"), tableBodyCell("math-reasoning 프롬프트")],
  ];
  s.addTable(dsRows, {
    x: 0.6, y: 1.95, w: 6.0, colW: [1.7, 1.3, 1.2, 1.8],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
  });

  // Model panel (right)
  s.addText("모델 패널 (open-weight VLM 12종)", {
    x: 6.85, y: 1.6, w: 6.0, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: NAVY, margin: 0,
  });
  const mdRows = [
    [tableHeaderCell("Model"), tableHeaderCell("Encoder"), tableHeaderCell("Role")],
    [tableBodyCell("Gemma4-e4b (4B)"), tableBodyCell("SigLIP"), tableBodyCell("Main + 메커니즘 (SigLIP)")],
    [tableBodyCell("LLaVA-1.5-7B"), tableBodyCell("CLIP-ViT-L/14"), tableBodyCell("메커니즘 + E4 mid-stack")],
    [tableBodyCell("LLaVA-Next-Interleaved-7B"), tableBodyCell("CLIP-ViT-L/14"), tableBodyCell("메인 + E5b/c 기준")],
    [tableBodyCell("InternVL3-8B"), tableBodyCell("InternViT-300M"), tableBodyCell("메커니즘 + E4 mid-stack")],
    [tableBodyCell("ConvLLaVA-7B"), tableBodyCell("ConvNeXt"), tableBodyCell("메커니즘 + E4 mid-stack")],
    [tableBodyCell("FastVLM-7B"), tableBodyCell("FastViT"), tableBodyCell("메커니즘 (4번째 archetype)")],
    [tableBodyCell("Gemma3-27b-it"), tableBodyCell("SigLIP-So-400m"), tableBodyCell("메인 + E5e (panel-leading)")],
    [tableBodyCell("Qwen2.5-VL-7B"), tableBodyCell("Qwen-ViT"), tableBodyCell("메인 + E5e (가장 anchor 저항적)")],
    [tableBodyCell("Qwen3-VL-8B Instruct"), tableBodyCell("Qwen3-VL"), tableBodyCell("메인 + γ-β instruct arm")],
    [tableHighlight("Qwen3-VL-8B Thinking"), tableHighlight("Qwen3-VL"), tableHighlight("γ-β reasoning-mode")],
  ];
  s.addTable(mdRows, {
    x: 6.85, y: 1.95, w: 6.0, colW: [2.4, 1.6, 2.0],
    border: { type: "solid", pt: 0.4, color: "D0D0D0" },
    fontSize: 9.5,
  });

  s.addText([
    { text: "메커니즘 패널", options: { bold: true, color: NAVY } },
    { text: " (E1/E1b/E1d): 6 모델 — 4가지 encoder archetype.   ", options: { color: CHARCOAL } },
    { text: "Mitigation 패널", options: { bold: true, color: NAVY } },
    { text: " (E4 Phase 2): 3 mid-stack-cluster 모델, n=88,650/모델.   ", options: { color: CHARCOAL } },
    { text: "γ-β reasoning 쌍", options: { bold: true, color: NAVY } },
    { text: ": Qwen3-VL-8B Instruct vs Thinking (동일 아키텍처).", options: { color: CHARCOAL } },
  ], {
    x: 0.6, y: 6.85, w: W - 1.2, h: 0.4,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, valign: "middle", margin: 0,
  });

  addPageFooter(s, 6, "§4 · Datasets + Models");
}

// =============================================================================
// SLIDE 7 — §5.1 Main panel (7 models × VQAv2)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "§5.1 메인 패널: 7 모델 × VQAv2 number (n=17,730)",
    "C-form direction-follow는 모든 모델에서 양수 — anchoring은 패널 전반에서 검출됨");

  const rows = [
    [tableHeaderCell("Model"), tableHeaderCell("acc(b)"), tableHeaderCell("acc(d)"), tableHeaderCell("acc(a)"), tableHeaderCell("adopt(a)"), tableHeaderCell("df(a) C-form")],
    [tableBodyCell("Gemma4-e4b"), tableBodyCell("0.553"), tableBodyCell("0.505"), tableBodyCell("0.541"), tableHighlight("0.066"), tableHighlight("0.274")],
    [tableBodyCell("LLaVA-Interleave-7b"), tableBodyCell("0.619"), tableBodyCell("0.577"), tableBodyCell("0.576"), tableHighlight("0.053"), tableHighlight("0.172")],
    [tableBodyCell("Gemma3-27b-it"), tableBodyCell("0.628"), tableBodyCell("0.623"), tableBodyCell("0.633"), tableHighlight("0.053"), tableHighlight("0.167")],
    [tableBodyCell("Qwen3-VL-30b"), tableBodyCell("0.759"), tableBodyCell("0.709"), tableBodyCell("0.707"), tableHighlight("0.039"), tableHighlight("0.170")],
    [tableBodyCell("Qwen3-VL-8b"), tableBodyCell("0.751"), tableBodyCell("0.709"), tableBodyCell("0.715"), tableHighlight("0.033"), tableHighlight("0.104")],
    [tableBodyCell("Qwen2.5-VL-7b"), tableBodyCell("0.736"), tableBodyCell("0.708"), tableBodyCell("0.711"), tableHighlight("0.021"), tableHighlight("0.094")],
    [tableBodyCell("Gemma4-31b-it"), tableBodyCell("0.749"), tableBodyCell("0.723"), tableBodyCell("0.741"), tableHighlight("0.024"), tableHighlight("0.085")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: 8.5, colW: [2.5, 1.0, 1.0, 1.0, 1.5, 1.5],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
    fontSize: 11,
  });

  // Right side: two callouts
  s.addShape("roundRect", {
    x: 9.4, y: 1.65, w: 3.45, h: 2.5,
    fill: { color: CREAM }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("패턴 1", {
    x: 9.55, y: 1.75, w: 3.2, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA, margin: 0,
  });
  s.addText("df(a) > 0 on every model (0.085–0.274).\nC-form metric은 panel 전반에서 anchor pull을 검출.", {
    x: 9.55, y: 2.1, w: 3.2, h: 1.95,
    fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, valign: "top", margin: 0,
  });

  s.addShape("roundRect", {
    x: 9.4, y: 4.3, w: 3.45, h: 2.5,
    fill: { color: CREAM }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("패턴 2", {
    x: 9.55, y: 4.4, w: 3.2, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA, margin: 0,
  });
  s.addText("adopt(a) (literal copy)는 2-7%로 df(a)보다 훨씬 작음.\nliteral 출력이 아니라 graded shift이 효과의 본체.", {
    x: 9.55, y: 4.75, w: 3.2, h: 1.95,
    fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL, valign: "top", margin: 0,
  });

  s.addText("두 패턴은 (a − d) 조건별 baseline gap에서도 유지 — anchor가 generic 2-image distractor 위에 추가 anchor pull을 더함.", {
    x: 0.6, y: 6.7, w: W - 1.2, h: 0.5,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: MUTED, margin: 0,
  });

  addPageFooter(s, 7, "§5.1 · Main panel");
}

// =============================================================================
// SLIDE 8 — §5.2 wrong/correct asymmetry (Phase A)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "§5.2 wrong-base vs correct-base 비대칭 (Phase A)",
    "기저 예측이 틀렸던 샘플에서 anchor 효과가 훨씬 큼 — 7/7 모델 모두");

  const rows = [
    [tableHeaderCell("Model"), tableHeaderCell("wrong − correct (moved-closer rate)"), tableHeaderCell("부호")],
    [tableBodyCell("Gemma4-e4b"), tableHighlight("+19.6 pp"), tableBodyCell("+", { color: GREEN, bold: true })],
    [tableBodyCell("Gemma3-27b-it"), tableHighlight("+15.9 pp"), tableBodyCell("+", { color: GREEN, bold: true })],
    [tableBodyCell("Qwen3-VL-30b"), tableHighlight("+12.2 pp"), tableBodyCell("+", { color: GREEN, bold: true })],
    [tableBodyCell("Gemma4-31b-it"), tableHighlight("+8.4 pp"), tableBodyCell("+", { color: GREEN, bold: true })],
    [tableBodyCell("Qwen3-VL-8b"), tableHighlight("+8.0 pp"), tableBodyCell("+", { color: GREEN, bold: true })],
    [tableBodyCell("LLaVA-Interleave"), tableHighlight("+7.2 pp"), tableBodyCell("+", { color: GREEN, bold: true })],
    [tableBodyCell("Qwen2.5-VL-7b"), tableHighlight("+6.9 pp"), tableBodyCell("+", { color: GREEN, bold: true })],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: 7.5, colW: [2.5, 4.0, 1.0],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
    fontSize: 12,
  });

  s.addShape("roundRect", {
    x: 8.4, y: 1.65, w: 4.45, h: 5.0,
    fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("Reading", {
    x: 8.6, y: 1.8, w: 4.1, h: 0.4,
    fontFace: TITLE_FONT, fontSize: 14, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "Anchoring은 모델이 anchor 없이는 틀렸을 cohort에 집중", options: { bold: true, color: NAVY } },
    { text: " — 즉 base-prediction entropy가 가장 높은 부분집합.", options: { color: CHARCOAL } },
    { text: "\n\n", options: { breakLine: true } },
    { text: "이 wrong/correct 이분법은 §6에서 ", options: { color: CHARCOAL } },
    { text: "answer-token entropy 4분위", options: { bold: true, color: TERRA } },
    { text: "라는 ", options: { color: CHARCOAL } },
    { text: "연속 confidence proxy", options: { bold: true, color: TERRA } },
    { text: "로 정밀화됨.", options: { color: CHARCOAL } },
    { text: "\n\n", options: { breakLine: true } },
    { text: "인지과학 대응: Mussweiler-Strack의 selective accessibility 모델 — anchor가 search-space의 candidate로 들어가서, 모델이 자기 prior에 의존하는 정도에 비례해 답에 blend됨.", options: { color: CHARCOAL, italic: true } },
  ], {
    x: 8.6, y: 2.25, w: 4.1, h: 4.3,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  s.addText("선행 anchoring 문헌 (LLM, VLM 모두) 중 ‘base correctness별 분리’를 한 사례 없음 — A1이 본 논문의 가장 강한 intellectual hook.", {
    x: 0.6, y: 6.85, w: W - 1.2, h: 0.4,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED, margin: 0,
  });

  addPageFooter(s, 8, "§5.2 · Asymmetry");
}

// =============================================================================
// SLIDE 9 — §5.3 distance decay (E5b)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "§5.3 거리별 anchor 효과 (E5b)",
    "anchor가 정답에서 ‘그럴듯한 거리’를 벗어나면 효과는 사라짐 — plausibility window");

  // Stratum legend
  s.addText("Stratum 정의: |a − gt| 거리에 따라 5단계로 분할", {
    x: 0.6, y: 1.6, w: 12, h: 0.3,
    fontFace: BODY_FONT, fontSize: 11, italic: true, color: MUTED, margin: 0,
  });

  const rows = [
    [tableHeaderCell("Stratum"), tableHeaderCell("|a − gt|"), tableHeaderCell("VQAv2 llava"), tableHeaderCell("VQAv2 qwen2.5"), tableHeaderCell("TallyQA llava"), tableHeaderCell("TallyQA qwen2.5")],
    [tableBodyCell("S1"), tableBodyCell("[0,1]"), tableHighlight("0.130"), tableHighlight("0.070"), tableHighlight("0.092"), tableHighlight("0.033")],
    [tableBodyCell("S2"), tableBodyCell("[2,5]"), tableBodyCell("0.032"), tableBodyCell("0.014"), tableBodyCell("0.006"), tableBodyCell("0.015")],
    [tableBodyCell("S3"), tableBodyCell("[6,30]"), tableBodyCell("0.010"), tableBodyCell("0.003"), tableBodyCell("0.003"), tableBodyCell("0.000")],
    [tableBodyCell("S4"), tableBodyCell("[31,300]"), tableBodyCell("0.010"), tableBodyCell("0.003"), tableBodyCell("0.000"), tableBodyCell("0.000")],
    [tableBodyCell("S5"), tableBodyCell("[301,∞)"), tableBodyCell("0.003"), tableBodyCell("0.003"), tableBodyCell("0.000"), tableBodyCell("0.000")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.95, w: 7.2, colW: [0.9, 1.0, 1.4, 1.4, 1.4, 1.4],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
    fontSize: 10.5,
  });
  s.addText("wrong-base paired adopt_cond", {
    x: 0.6, y: 4.55, w: 7.2, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, align: "center", margin: 0,
  });

  // E5b figure
  const fig = path.join(REPO, "docs/figures/E5c_correct_vs_wrong_adopt.png");
  if (fs.existsSync(fig)) {
    s.addImage({ path: fig, x: 8.0, y: 1.95, w: 5.0, h: 2.5,
      sizing: { type: "contain", w: 5.0, h: 2.5 } });
    s.addText("E5b/c S1 peak / S5 floor pattern (wrong-base 빨강 vs correct-base 파랑)", {
      x: 8.0, y: 4.5, w: 5.0, h: 0.3,
      fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, align: "center", margin: 0,
    });
  }

  s.addShape("roundRect", {
    x: 0.6, y: 5.1, w: W - 1.2, h: 1.7,
    fill: { color: CREAM }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("Reading — 두 gate가 동시에 작동", {
    x: 0.8, y: 5.2, w: 12.0, h: 0.4,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "①  Uncertainty gate", options: { bold: true, color: NAVY } },
    { text: ": correct-base에서는 anchor가 끌어오지 못함 (S1에서도 ≤ 0.10).   ", options: { color: CHARCOAL } },
    { text: "②  Plausibility gate", options: { bold: true, color: NAVY } },
    { text: ": wrong-base여도 |a-gt| > 5 (S3+) 이면 anchor 거부.\n", options: { color: CHARCOAL } },
    { text: "두 model 모두에서 S1 peak / S3-S5 floor 형태가 동일 — base-accuracy (0.62 vs 0.81) 차이가 패턴을 바꾸지 않음.", options: { color: CHARCOAL, italic: true } },
  ], {
    x: 0.8, y: 5.6, w: 12.0, h: 1.15,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  addPageFooter(s, 9, "§5.3 · Distance");
}

// =============================================================================
// SLIDE 10 — §5.4 digit-pixel causality (E5c)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "§5.4 digit-pixel 인과성 (E5c)",
    "anchor에서 숫자 픽셀만 inpaint한 ‘masked’ arm을 추가 — anchor − masked 간격이 digit pixels의 순수 기여");

  const rows = [
    [tableHeaderCell("Dataset"), tableHeaderCell("Model"), tableHeaderCell("anchor"), tableHeaderCell("masked"), tableHeaderCell("a − m gap")],
    [tableBodyCell("VQAv2"), tableBodyCell("llava-interleave-7b"), tableHighlight("0.129"), tableBodyCell("0.068"), tableBodyCell("+6.1 pp", { color: GREEN, bold: true })],
    [tableBodyCell("VQAv2"), tableBodyCell("qwen2.5-vl-7b"), tableBodyCell("0.070"), tableBodyCell("0.066"), tableBodyCell("+0.4 pp", { color: MUTED, italic: true })],
    [tableBodyCell("TallyQA"), tableBodyCell("llava-interleave-7b"), tableHighlight("0.110"), tableBodyCell("0.084"), tableBodyCell("+2.5 pp", { color: GREEN, bold: true })],
    [tableBodyCell("TallyQA"), tableBodyCell("qwen2.5-vl-7b"), tableBodyCell("0.033"), tableBodyCell("0.037"), tableBodyCell("−0.5 pp", { color: MUTED, italic: true })],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: 7.2, colW: [1.2, 2.7, 1.1, 1.1, 1.1],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
    fontSize: 11,
  });
  s.addText("wrong-base × S1, paired adopt_cond", {
    x: 0.6, y: 3.7, w: 7.2, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, align: "center", margin: 0,
  });

  const fig = path.join(REPO, "docs/figures/E5c_anchor_vs_masked_adopt.png");
  if (fs.existsSync(fig)) {
    s.addImage({ path: fig, x: 8.0, y: 1.65, w: 5.0, h: 2.4,
      sizing: { type: "contain", w: 5.0, h: 2.4 } });
    s.addText("anchor (실선 ●) vs masked (점선 ■) × wrong/correct base — llava-interleave", {
      x: 8.0, y: 4.05, w: 5.0, h: 0.3,
      fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, align: "center", margin: 0,
    });
  }

  s.addShape("roundRect", {
    x: 0.6, y: 4.4, w: W - 1.2, h: 2.5,
    fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("Reading — direction-consistent across models", {
    x: 0.8, y: 4.5, w: 12.0, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "•  llava-interleave: anchor > masked 격차 양수 (+6.1 / +2.5 pp). digit pixel이 paired adoption의 인과 경로.", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  qwen2.5-vl: 양 arm 모두 noise floor — §3.3 main panel에서 가장 anchor-resistant 모델 (df(a) = 0.094)이라 E5c도 floor에 위치.", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  Reading: ", options: { bold: true, color: NAVY } },
    { text: "‘pull이 검출 가능한 곳에서는 gap이 양수, pull이 floor면 gap도 floor’ — direction-consistent. ", options: { color: CHARCOAL } },
    { text: "gemma3-27b-it E5c VQAv2 cell pending (~5–6h H200) → mid-panel 모델이 llava-style인지 qwen-style인지 결정.", options: { color: TERRA, italic: true, bold: true } },
    { text: "\n\n", options: { breakLine: true } },
    { text: "(1,3,4) 비교 — masked vs neutral의 acc_drop 차이는 1-2 pp뿐 → anchor scene 배경은 generic distractor 이상의 일을 하지 않음.", options: { color: MUTED, italic: true } },
  ], {
    x: 0.8, y: 4.85, w: 12.0, h: 2.0,
    fontFace: BODY_FONT, fontSize: 10.5, valign: "top", margin: 0,
  });

  addPageFooter(s, 10, "§5.4 · Digit-pixel");
}

// =============================================================================
// SLIDE 11 — §5.5 cross-dataset (E5e)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "§5.5 Cross-dataset 확장 (E5e)",
    "S1 단일-stratum × b/a/m/d × 3 모델 × ChartQA + TallyQA + MathVista — a > m 패턴이 dataset 전반에서 재현");

  const rows = [
    [tableHeaderCell("Dataset"), tableHeaderCell("Model"), tableHeaderCell("adopt(a)"), tableHeaderCell("adopt(m)"), tableHeaderCell("df(a) C-form"), tableHeaderCell("df(m) C-form")],
    [tableBodyCell("ChartQA"), tableBodyCell("gemma3-27b-it"), tableHighlight("0.037"), tableBodyCell("0.022"), tableHighlight("0.096"), tableBodyCell("0.079")],
    [tableBodyCell("ChartQA"), tableBodyCell("llava-interleave"), tableHighlight("0.028"), tableBodyCell("0.009"), tableHighlight("0.152"), tableBodyCell("0.115")],
    [tableBodyCell("ChartQA"), tableBodyCell("qwen2.5-vl-7b"), tableHighlight("0.017"), tableBodyCell("0.013"), tableHighlight("0.051"), tableBodyCell("0.046")],
    [tableBodyCell("TallyQA"), tableBodyCell("gemma3-27b-it"), tableHighlight("0.027"), tableBodyCell("0.016"), tableHighlight("0.073"), tableBodyCell("0.060")],
    [tableBodyCell("TallyQA"), tableBodyCell("llava-interleave"), tableHighlight("0.026"), tableBodyCell("0.014"), tableHighlight("0.066"), tableBodyCell("0.056")],
    [tableBodyCell("TallyQA"), tableBodyCell("qwen2.5-vl-7b"), tableHighlight("0.011"), tableBodyCell("0.011"), tableHighlight("0.029"), tableBodyCell("0.030")],
    [tableBodyCell("MathVista", { bold: true, fill: { color: CREAM } }),
     tableBodyCell("gemma3-27b-it", { bold: true, fill: { color: CREAM } }),
     tableHighlight("0.176", { fill: { color: GOLD }, color: "FFFFFF" }),
     tableBodyCell("0.047", { bold: true, fill: { color: CREAM } }),
     tableHighlight("0.216", { fill: { color: GOLD }, color: "FFFFFF" }),
     tableBodyCell("0.134", { bold: true, fill: { color: CREAM } })],
    [tableBodyCell("MathVista"), tableBodyCell("llava-interleave"), tableHighlight("0.066"), tableBodyCell("0.030"), tableHighlight("0.205"), tableBodyCell("0.125")],
    [tableBodyCell("MathVista"), tableBodyCell("qwen2.5-vl-7b"), tableHighlight("0.020"), tableBodyCell("0.008"), tableHighlight("0.072"), tableBodyCell("0.041")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.6, w: W - 1.2, colW: [1.5, 2.7, 1.6, 1.6, 1.9, 1.9],
    border: { type: "solid", pt: 0.4, color: "D0D0D0" },
    fontSize: 10.5,
  });

  s.addText([
    { text: "관찰 1.  ", options: { bold: true, color: NAVY } },
    { text: "3/3 모델이 모든 dataset에서 a > m 보존 — digit-pixel 인과성이 model family와 dataset 전반에서 일반화.   ", options: { color: CHARCOAL } },
    { text: "관찰 2.  ", options: { bold: true, color: NAVY } },
    { text: "MathVista가 panel-largest cell — gemma3-27b adopt 0.176 / df 0.332 (wrong-base S1).   ", options: { color: CHARCOAL } },
    { text: "관찰 3.  ", options: { bold: true, color: NAVY } },
    { text: "plausibility-window 패턴은 per-dataset cutoff (E5d)로 양적 검증.", options: { color: CHARCOAL } },
  ], {
    x: 0.6, y: 6.2, w: W - 1.2, h: 1.0,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  addPageFooter(s, 11, "§5.5 · E5e");
}

// =============================================================================
// SLIDE 12 — §6 confidence-modulated anchoring
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "§6 confidence-modulated anchoring (L1 분석)",
    "wrong/correct 이분법은 ‘answer-token entropy 4분위 monotone’의 거친 투영");

  const rows = [
    [tableHeaderCell("Quartile"), tableHeaderCell("base correctness"), tableHeaderCell("anchor adopt"), tableHeaderCell("df C-form")],
    [tableBodyCell("Q1 (가장 confident)"), tableBodyCell("0.92"), tableBodyCell("0.043"), tableBodyCell("0.032")],
    [tableBodyCell("Q2"), tableBodyCell("0.72"), tableBodyCell("0.084"), tableBodyCell("0.080")],
    [tableBodyCell("Q3"), tableBodyCell("0.42"), tableBodyCell("0.149"), tableBodyCell("0.137")],
    [tableHighlight("Q4 (가장 uncertain)"), tableHighlight("0.34"), tableHighlight("0.172"), tableHighlight("0.210")],
    [tableBodyCell("Δ (Q4 − Q1)", { bold: true, fill: { color: NAVY }, color: "FFFFFF" }),
     tableBodyCell("−0.58", { bold: true, fill: { color: NAVY }, color: "FFFFFF" }),
     tableBodyCell("+0.130", { bold: true, fill: { color: GOLD }, color: "FFFFFF" }),
     tableBodyCell("+0.178", { bold: true, fill: { color: GOLD }, color: "FFFFFF" })],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: 7.2, colW: [2.4, 1.6, 1.6, 1.6],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
    fontSize: 11,
  });
  s.addText("E5c VQAv2 wrong-base S1, llava-interleave-7b — entropy_top_k proxy 기준", {
    x: 0.6, y: 4.45, w: 7.2, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, italic: true, color: MUTED, align: "center", margin: 0,
  });

  const fig = path.join(REPO, "docs/figures/paper_L1_confidence_quartile.png");
  if (fs.existsSync(fig)) {
    s.addImage({ path: fig, x: 8.0, y: 1.65, w: 5.0, h: 3.1,
      sizing: { type: "contain", w: 5.0, h: 3.1 } });
  }

  s.addShape("roundRect", {
    x: 0.6, y: 4.95, w: W - 1.2, h: 1.95,
    fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("패널 전반 결론", {
    x: 0.8, y: 5.05, w: 12.0, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "•  ", options: { color: NAVY, bold: true } },
    { text: "23/35 (model × dataset × stratum) cell이 fully monotone Q1 < Q2 < Q3 < Q4 on direction-follow.", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  ", options: { color: NAVY, bold: true } },
    { text: "mean Q4 − Q1 gap = +0.152 (df) / +0.044 (adopt) — entropy_top_k proxy가 가장 깨끗한 신호.", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  ", options: { color: NAVY, bold: true } },
    { text: "결론: anchor가 threshold 넘으면 capture하는 step이 아니라, ", options: { color: CHARCOAL } },
    { text: "base-prediction entropy에 비례한 graded blending", options: { bold: true, color: TERRA } },
    { text: " — Mussweiler-Strack의 selective accessibility 모델과 일치.", options: { color: CHARCOAL } },
  ], {
    x: 0.8, y: 5.45, w: 12.0, h: 1.4,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  addPageFooter(s, 12, "§6 · Confidence");
}

// =============================================================================
// SLIDE 13 — §7.1-§7.2 attention mass + per-layer (E1, E1b)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "§7.1–§7.2 anchor attention mass + 4 archetype peak layer (E1, E1b)",
    "encoder family별로 anchor에 attention이 몰리는 ‘peak layer’가 다름");

  const rows = [
    [tableHeaderCell("Archetype"), tableHeaderCell("Reference model"), tableHeaderCell("Peak layer"), tableHeaderCell("δ at peak"), tableHeaderCell("Budget source")],
    [tableBodyCell("SigLIP-Gemma early"), tableBodyCell("gemma4-e4b"), tableBodyCell("L5/42 (12%)"), tableHighlight("+0.050"), tableBodyCell("text-stealing")],
    [tableBodyCell("Mid-stack cluster", { bold: true, fill: { color: CREAM } }),
     tableBodyCell("llava-1.5-7b (CLIP-ViT)", { bold: true, fill: { color: CREAM } }),
     tableBodyCell("L16/32", { bold: true, fill: { color: CREAM } }),
     tableHighlight("+0.019", { fill: { color: GOLD }, color: "FFFFFF" }),
     tableBodyCell("text-stealing", { bold: true, fill: { color: CREAM } })],
    [tableBodyCell("Mid-stack cluster", { fill: { color: CREAM } }),
     tableBodyCell("convllava-7b (ConvNeXt)", { fill: { color: CREAM } }),
     tableBodyCell("L16/32", { fill: { color: CREAM } }),
     tableHighlight("+0.022", { fill: { color: GOLD }, color: "FFFFFF" }),
     tableBodyCell("text-stealing", { fill: { color: CREAM } })],
    [tableBodyCell("Mid-stack cluster", { fill: { color: CREAM } }),
     tableBodyCell("internvl3-8b (InternViT)", { fill: { color: CREAM } }),
     tableBodyCell("L14/28", { fill: { color: CREAM } }),
     tableHighlight("+0.019", { fill: { color: GOLD }, color: "FFFFFF" }),
     tableBodyCell("text-stealing", { fill: { color: CREAM } })],
    [tableBodyCell("Qwen-ViT late"), tableBodyCell("qwen2.5-vl-7b"), tableBodyCell("L22/28 (82%)"), tableBodyCell("+0.015"), tableBodyCell("target-stealing")],
    [tableBodyCell("FastVLM late"), tableBodyCell("fastvlm-7b"), tableBodyCell("L22"), tableBodyCell("+0.047"), tableBodyCell("text-stealing")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: W - 1.2, colW: [2.5, 3.5, 1.8, 1.8, 2.5],
    border: { type: "solid", pt: 0.4, color: "D0D0D0" },
    fontSize: 11,
  });

  s.addShape("roundRect", {
    x: 0.6, y: 5.6, w: W - 1.2, h: 1.3,
    fill: { color: CREAM }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText([
    { text: "Mid-stack cluster에 ", options: { color: CHARCOAL } },
    { text: "아키텍처적으로 다른 3개 encoder", options: { bold: true, color: NAVY } },
    { text: " (CLIP-ViT, ConvNeXt, InternViT)가 같은 layer-depth signature로 수렴 → mitigation의 가장 leverage 큰 target.\n", options: { color: CHARCOAL } },
    { text: "n=200 stratified, generation step에서 anchor token에 갈 attention의 비중 측정. CI excludes 0 on 4/4 base-experiment 모델.", options: { color: MUTED, italic: true } },
    { text: "\n", options: { breakLine: true } },
    { text: "(H3 retired) ", options: { bold: true, color: TERRA } },
    { text: "encoder-architecture per se는 anchoring susceptibility를 예측하지 못함 → depth-axis로 framing 교체.", options: { color: CHARCOAL } },
  ], {
    x: 0.8, y: 5.7, w: 12.0, h: 1.15,
    fontFace: BODY_FONT, fontSize: 10.5, valign: "top", margin: 0,
  });

  addPageFooter(s, 13, "§7.1–§7.2 · Attention");
}

// =============================================================================
// SLIDE 14 — §7.3 causal ablation (E1d)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "§7.3 인과 ablation (E1d) — single-layer null, upper-half가 작동",
    "anchor span을 layer-set별로 mask해서 direction-follow 변화 측정 (n=200, 6 모델)");

  const rows = [
    [tableHeaderCell("Mode"), tableHeaderCell("결과")],
    [tableBodyCell("ablate_peak (E1b 헤드라인 layer 단독)"), tableBodyCell("Null on 6/6 (|Δ df| ≤ 2.0 pp; CI overlap baseline)", { color: MUTED })],
    [tableBodyCell("ablate_layer0 (non-peak control)"), tableBodyCell("Null on 6/6 (Δ df ∈ [−2.7, +0.5] pp)", { color: MUTED })],
    [tableHighlight("ablate_upper_half (mitigation candidate)"),
     tableHighlight("−4.0 ~ −10.5 pp on 6/6 모델, fluency-clean on 4/6")],
    [tableBodyCell("ablate_all"), tableBodyCell("−9.6 ~ −24.5 pp BUT fluency 붕괴 3/6 모델 (mean-distance 4-6배 또는 1000배)", { color: TERRA })],
    [tableBodyCell("ablate_lower_half (diagnostic)"), tableBodyCell("Heterogeneous: 3/6 BACKFIRE, 1/6 reduce, 2/6 flat", { color: MUTED })],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: W - 1.2, colW: [4.5, 7.7],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
    fontSize: 11,
  });

  s.addShape("roundRect", {
    x: 0.6, y: 4.5, w: W - 1.2, h: 2.4,
    fill: { color: "FFFFFF" }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("핵심 surprise — single-layer ablation이 인과적으로 null", {
    x: 0.8, y: 4.6, w: 12.0, h: 0.4,
    fontFace: TITLE_FONT, fontSize: 14, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "•  E1b의 peak layer 하나만 마스크해도 효과 없음 → ", options: { color: CHARCOAL } },
    { text: "anchor 신호는 LLM stack 전반에 redundant하게 인코딩", options: { bold: true, color: NAVY } },
    { text: ".", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  peak layer는 신호가 가장 ‘보이는 곳’이지, ‘유일하게 만들어지는 곳’이 아님.", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  ", options: { color: CHARCOAL } },
    { text: "Upper-half 단독 mitigation locus", options: { bold: true, color: TERRA } },
    { text: "가 6/6 모델 panel-wide 작동하면서 fluency를 깨지 않는 유일한 지점 → §7.4 free-lunch mitigation의 근거.", options: { color: CHARCOAL } },
    { text: "\n\n", options: { breakLine: true } },
    { text: "Reading: ", options: { bold: true, color: NAVY } },
    { text: "anchor 효과는 ‘하나의 critical site’가 아니라 ‘upper half 전체에 분산된 pathway’ — 그래서 단일 layer 제거는 보상되지만 upper half 전체 down-weight는 전달이 줄어든다.", options: { color: CHARCOAL, italic: true } },
  ], {
    x: 0.8, y: 5.0, w: 12.0, h: 1.85,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  addPageFooter(s, 14, "§7.3 · Ablation");
}

// =============================================================================
// SLIDE 15 — §7.4 mitigation (E4 free-lunch)
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "§7.4 mitigation: upper-half attention 재가중 (E4 Phase 2, n=88,650/모델)",
    "df는 떨어지고 exact_match는 오르며 target_only accuracy는 그대로 — ‘free-lunch’");

  const rows = [
    [tableHeaderCell("Model"), tableHeaderCell("s*"),
     tableHeaderCell("df 기준 → 처리"), tableHeaderCell("df Δ pp"), tableHeaderCell("df rel"),
     tableHeaderCell("em 기준 → 처리"), tableHeaderCell("em Δ pp")],
    [tableBodyCell("LLaVA-1.5-7b"), tableBodyCell("−3.0"),
     tableBodyCell("0.288 → 0.246"), tableHighlight("−4.19"), tableHighlight("−14.6 %"),
     tableBodyCell("0.334 → 0.342"), tableBodyCell("+0.77", { color: GREEN, bold: true })],
    [tableBodyCell("ConvLLaVA-7b"), tableBodyCell("−2.0"),
     tableBodyCell("0.258 → 0.233"), tableHighlight("−2.49"), tableHighlight("−9.6 %"),
     tableBodyCell("0.352 → 0.365"), tableBodyCell("+1.30", { color: GREEN, bold: true })],
    [tableBodyCell("InternVL3-8b"), tableBodyCell("−0.5"),
     tableBodyCell("0.126 → 0.119"), tableBodyCell("−0.74"), tableBodyCell("−5.8 %"),
     tableBodyCell("0.590 → 0.595"), tableBodyCell("+0.49", { color: GREEN, bold: true })],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: W - 1.2, colW: [1.8, 0.8, 2.0, 1.4, 1.4, 2.5, 1.4],
    border: { type: "solid", pt: 0.5, color: "C0C0C0" },
    fontSize: 10.5,
  });

  s.addShape("roundRect", {
    x: 0.6, y: 3.7, w: W - 1.2, h: 3.2,
    fill: { color: CREAM }, line: { color: NAVY, width: 1 }, rectRadius: 0.06,
  });
  s.addText("3 가지 invariant — 왜 ‘free-lunch’인가", {
    x: 0.8, y: 3.8, w: 12.0, h: 0.4,
    fontFace: TITLE_FONT, fontSize: 14, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "①  ", options: { color: NAVY, bold: true } },
    { text: "df는 모든 모델에서 감소, em은 모든 모델에서 +0.49 ~ +1.30 pp 상승 — anchor 조건에서만 예측이 정답 쪽으로 이동.", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "②  ", options: { color: NAVY, bold: true } },
    { text: "accuracy_vqa(b) (target_only baseline)이 모든 strength에서 invariant — hook이 single-image inference로 leak하지 않음.", options: { color: CHARCOAL } },
    { text: "\n", options: { breakLine: true } },
    { text: "③  ", options: { color: NAVY, bold: true } },
    { text: "accuracy_vqa(d) (neutral arm)도 ±0.5 pp 이내 — 두 번째 이미지가 숫자가 아니면 hook이 영향 없음. anchor pathway에만 작용.", options: { color: CHARCOAL } },
    { text: "\n\n", options: { breakLine: true } },
    { text: "Per-model s*가 필요 (−0.5 ~ −3.0, 한 자릿수 차이) — mitigation은 ", options: { color: CHARCOAL } },
    { text: "‘locus + selection rule’ 수준에서 일반화", options: { bold: true, color: TERRA } },
    { text: "되지, 단일 strength constant 수준은 아님. df 감소율은 baseline anchor pull과 anti-correlation (LLaVA-1.5 가장 큼, InternVL3 가장 작음).", options: { color: CHARCOAL } },
  ], {
    x: 0.8, y: 4.2, w: 12.0, h: 2.65,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  addPageFooter(s, 15, "§7.4 · Mitigation");
}

// =============================================================================
// SLIDE 16 — γ-β reasoning amplifies anchoring
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  // dark-bg header (no addPageHeader call — would draw duplicate text under our overlay)
  s.addShape("rect", {
    x: 0.4, y: 0.55, w: 0.10, h: 0.7,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("§8.1 reasoning이 anchoring을 강화한다 (γ-β)", {
    x: 0.65, y: 0.45, w: W - 1.5, h: 0.6,
    fontFace: TITLE_FONT, fontSize: 26, bold: true,
    color: "FFFFFF", valign: "middle", margin: 0,
  });
  s.addText("Qwen3-VL-8B-Instruct vs Thinking, MathVista, S1 anchor arm, n=365", {
    x: 0.65, y: 1.05, w: W - 1.5, h: 0.35,
    fontFace: BODY_FONT, fontSize: 12, italic: true,
    color: "CADCFC", valign: "middle", margin: 0,
  });

  // Big stat callouts
  const stats = [
    { num: "0.117 / 0.074", label: "adopt(a) — Thinking / Instruct", multiplier: "×1.6", direction: "↑" },
    { num: "0.291 / 0.102", label: "df(a) C-form — Thinking / Instruct", multiplier: "×2.9", direction: "↑↑" },
    { num: "0.196 / 0.216", label: "acc(b) — Thinking / Instruct", multiplier: "↓", direction: "Thinking이 baseline 자체도 더 부정확" },
  ];
  stats.forEach((st, i) => {
    const x = 0.6 + i * 4.15;
    s.addShape("roundRect", {
      x, y: 1.9, w: 3.95, h: 2.6,
      fill: { color: "FFFFFF" }, line: { color: GOLD, width: 1 }, rectRadius: 0.08,
    });
    s.addText(st.num, {
      x: x + 0.15, y: 2.0, w: 3.65, h: 0.7,
      fontFace: TITLE_FONT, fontSize: 26, bold: true, color: TERRA,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(st.label, {
      x: x + 0.15, y: 2.7, w: 3.65, h: 0.45,
      fontFace: BODY_FONT, fontSize: 11, color: CHARCOAL,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(st.multiplier, {
      x: x + 0.15, y: 3.2, w: 3.65, h: 0.7,
      fontFace: TITLE_FONT, fontSize: 32, bold: true, color: NAVY,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(st.direction, {
      x: x + 0.15, y: 3.95, w: 3.65, h: 0.45,
      fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED,
      align: "center", valign: "middle", margin: 0,
    });
  });

  s.addShape("roundRect", {
    x: 0.6, y: 4.8, w: W - 1.2, h: 2.1,
    fill: { color: TERRA }, line: { color: TERRA, width: 0 }, rectRadius: 0.06,
  });
  s.addText("Reading", {
    x: 0.8, y: 4.9, w: 12.0, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 14, bold: true, color: GOLD, margin: 0,
  });
  s.addText([
    { text: "•  Thinking 체크포인트는 동일 architecture / chat template / 동일 4-condition 자극 — 차이는 ", options: { color: "FFFFFF" } },
    { text: "trained reasoning 행동뿐", options: { bold: true, color: GOLD } },
    { text: ".", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  reasoning trace가 anchor를 잡지 못할 뿐 아니라 ", options: { color: "FFFFFF" } },
    { text: "오히려 anchor 정보가 elaboration step에 누적되어 증폭", options: { bold: true, color: GOLD } },
    { text: " — Wang 2025 (LRM-judging) + VLMBias (reasoning models)와 일치.", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  단일 pair × 단일 dataset이라 ", options: { color: "FFFFFF" } },
    { text: "existence-proof 수준의 hook", options: { italic: true, color: "CADCFC" } },
    { text: ". §8 future: 30B-A3B 쌍 + Gemma3-Thinking + 다른 dataset으로 generalisation.", options: { color: "FFFFFF" } },
  ], {
    x: 0.8, y: 5.3, w: 12.0, h: 1.55,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  // footer (manual since dark bg)
  s.addShape("rect", {
    x: 0, y: H - 0.32, w: W, h: 0.04,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("§8.1 · γ-β reasoning", {
    x: 0.4, y: H - 0.32, w: 6, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, color: "CADCFC",
    valign: "middle", margin: 0,
  });
  s.addText("16 / 20", {
    x: W - 1.2, y: H - 0.32, w: 0.8, h: 0.3,
    fontFace: BODY_FONT, fontSize: 9, color: "CADCFC",
    align: "right", valign: "middle", margin: 0,
  });
}

// =============================================================================
// SLIDE 17 — Related work positioning
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "§2 선행 연구와의 차별화",
    "본 논문이 unmeasured했던 정확한 위치는?");

  const rows = [
    [tableHeaderCell("Extension"), tableHeaderCell("가장 가까운 선행"), tableHeaderCell("차별성")],
    [tableHighlight("Cross-modal numerical anchoring (core)"),
     tableBodyCell("VLMBias (Nguyen 2025), typographic attacks (Wang 2025)"),
     tableHighlight("Genuinely novel — stand-alone rendered-number image as anchor on open numerical VQA")],
    [tableHighlight("Wrong-base asymmetry"),
     tableBodyCell("LLM/VLM anchoring 문헌 모두 미수행"),
     tableHighlight("Genuinely novel — 가장 강한 intellectual hook")],
    [tableBodyCell("Cross-encoder ablation (ViT vs Conv vs SigLIP)"),
     tableBodyCell("Dyslexify (Hufe 2025) — CLIP late-half typographic 회로"),
     tableBodyCell("Novel — depth-axis로 framing, mid-stack cluster 발견")],
    [tableBodyCell("Reasoning amplifies (γ-β)"),
     tableBodyCell("Wang 2025 (LRM-judging text-only), VLMBias on reasoning"),
     tableBodyCell("VLM에서의 first 확인 (단일 pair existence-proof)")],
    [tableBodyCell("Confidence-modulated (continuous)"),
     tableBodyCell("Lou-Sun 2024 (LLM 강도별), Phase-A binary"),
     tableBodyCell("Novel — entropy 4분위 monotone × 23/35 cell")],
    [tableBodyCell("Mid-stack mitigation"),
     tableBodyCell("Weng 2024 EMNLP Main (causal mediation + 22 % mitigation)"),
     tableBodyCell("Same template, multimodal anchor 적용")],
  ];
  s.addTable(rows, {
    x: 0.6, y: 1.65, w: W - 1.2, colW: [3.5, 4.0, 4.7],
    border: { type: "solid", pt: 0.4, color: "D0D0D0" },
    fontSize: 10.5,
  });

  s.addText([
    { text: "Reviewer 방어 핵심: ", options: { bold: true, color: TERRA } },
    { text: "(a) anchor는 의미 라벨이 아닌 ", options: { color: CHARCOAL } },
    { text: "단일 숫자값", options: { bold: true, color: NAVY } },
    { text: ", (b) 측정은 ", options: { color: CHARCOAL } },
    { text: "회귀형 numeric shift", options: { bold: true, color: NAVY } },
    { text: " (classification flip / ASR 아님), (c) ", options: { color: CHARCOAL } },
    { text: "인지과학 grounding (Mussweiler-Strack, Jacowitz-Kahneman)", options: { bold: true, color: NAVY } },
    { text: " — 적합한 testable predictions.", options: { color: CHARCOAL } },
  ], {
    x: 0.6, y: 6.5, w: W - 1.2, h: 0.7,
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, valign: "top", margin: 0,
  });

  addPageFooter(s, 17, "§2 · Related work");
}

// =============================================================================
// SLIDE 18 — limitations
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: CREAM };
  addPageHeader(s, "§8.4 한계 (limitations)",
    "Reviewer가 물을 항목을 미리 disclose");

  const limits = [
    { title: "Single-prompt runs",
      body: "JSON-strict 단일 프롬프트만 사용. paraphrase robustness (3-5 prompt 변형 × bootstrap CI)는 다음 hardening pass." },
    { title: "Open weights only",
      body: "메인 7개 모델 모두 open. closed-model defuse (GPT-4o / Gemini 2.5 ~500 sample)는 revision 시 access 확보 시 추가." },
    { title: "No human baseline",
      body: "§1/§6의 인지과학 framing은 prior literature 기반. 50명 Prolific replication은 ARR clock 상 미실시." },
    { title: "Distance window dataset-dependent",
      body: "VQAv2/TallyQA absolute, ChartQA/MathVista relative. relative cutoff은 inductive choice — GT 분포 다른 dataset에서 재검증 필요." },
    { title: "Mid-stack mitigation single-cluster",
      body: "E4 free-lunch는 LLaVA-1.5 / ConvLLaVA / InternVL3 3개 mid-stack-cluster 모델만 검증. SigLIP-Gemma early peak / Qwen-ViT late peak로의 일반화는 P3." },
    { title: "γ-β single pair",
      body: "reasoning-amplifies-anchoring은 1 pair × 1 dataset. existence proof로 취급, quantitative law로 보지 않음." },
    { title: "Driver schema audit",
      body: "M1→M2 refactor 사이 driver-schema gap이 direction_follow_rate를 silently 0으로 만든 적 있음. C-form refactor + reaggregate sweep + migration report로 remediation." },
  ];
  limits.forEach((lm, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.6 + col * 6.2;
    const y = 1.65 + row * 1.30;
    s.addShape("roundRect", {
      x, y, w: 6.0, h: 1.20,
      fill: { color: "FFFFFF" }, line: { color: NAVY, width: 0.5 }, rectRadius: 0.05,
    });
    s.addText(lm.title, {
      x: x + 0.15, y: y + 0.08, w: 5.7, h: 0.4,
      fontFace: TITLE_FONT, fontSize: 12, bold: true, color: TERRA, margin: 0,
    });
    s.addText(lm.body, {
      x: x + 0.15, y: y + 0.45, w: 5.7, h: 0.7,
      fontFace: BODY_FONT, fontSize: 10, color: CHARCOAL, valign: "top", margin: 0,
    });
  });

  addPageFooter(s, 18, "§8.4 · Limitations");
}

// =============================================================================
// SLIDE 19 — 5 contributions + future work
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: "FFFFFF" };
  addPageHeader(s, "§1.6 5가지 공헌 + §8 future work",
    "");

  const contribs = [
    { num: "1", body: "VLM 최초의 cross-modal 숫자 anchoring 평가 — 4-condition (target/anchor/mask/neutral) + FLUX-rendered digit anchor inventory + OpenCV-inpainted mask counterparts" },
    { num: "2", body: "M2 canonical metric (C-form direction-follow `(pa−pb)·(anchor−pb) > 0 AND pa ≠ pb`) — gt-free, baseline-relative anchor pull 측정" },
    { num: "3", body: "Cross-dataset evidence — VQAv2 number (n=17,730) / TallyQA / ChartQA / MathVista 4종 + 7-model main panel + 3-model E5e + 2-model γ-β reasoning" },
    { num: "4", body: "메커니즘 + mitigation — encoder-family별 attention localisation (E1/E1b) + causal ablation panel (E1d) + upper-half mitigation full-scale validation (E4 Phase 2, n=88,650/모델)" },
    { num: "5", body: "VLM에서 reasoning이 anchoring을 amplify하는 결과 (γ-β, MathVista) — text-only LRM 문헌과 일치하는 multimodal counterpart" },
  ];
  contribs.forEach((c, i) => {
    const y = 1.6 + i * 0.78;
    s.addShape("ellipse", {
      x: 0.6, y, w: 0.6, h: 0.6,
      fill: { color: NAVY }, line: { color: NAVY, width: 0 },
    });
    s.addText(c.num, {
      x: 0.6, y, w: 0.6, h: 0.6,
      fontFace: TITLE_FONT, fontSize: 22, bold: true, color: GOLD,
      align: "center", valign: "middle", margin: 0,
    });
    s.addText(c.body, {
      x: 1.4, y: y - 0.05, w: W - 1.7, h: 0.7,
      fontFace: BODY_FONT, fontSize: 12, color: CHARCOAL, valign: "middle", margin: 0,
    });
  });

  s.addShape("roundRect", {
    x: 0.6, y: 5.7, w: W - 1.2, h: 1.2,
    fill: { color: CREAM }, line: { color: TERRA, width: 1 }, rectRadius: 0.06,
  });
  s.addText("Future work (preferred)", {
    x: 0.8, y: 5.78, w: 12.0, h: 0.35,
    fontFace: TITLE_FONT, fontSize: 13, bold: true, color: TERRA, margin: 0,
  });
  s.addText([
    { text: "F1.  ", options: { color: NAVY, bold: true } },
    { text: "LLM/VLM 아키텍처 차이 — 동일 숫자 question을 (a) text 프롬프트로 LLM에 vs (b) rendered-digit image로 VLM에 — layer-wise integration profile 비교 (ideation 단락 작성됨).   ", options: { color: CHARCOAL } },
    { text: "F2.  ", options: { color: NAVY, bold: true } },
    { text: "image vs text anchor on the same VLM — modality 채널 분리.   ", options: { color: CHARCOAL } },
    { text: "F3.  ", options: { color: NAVY, bold: true } },
    { text: "γ-β를 multi-pair × multi-dataset reasoning panel로 scale-up.", options: { color: CHARCOAL } },
  ], {
    x: 0.8, y: 6.13, w: 12.0, h: 0.75,
    fontFace: BODY_FONT, fontSize: 10.5, valign: "top", margin: 0,
  });

  addPageFooter(s, 19, "§1.6 / §8 · Contributions");
}

// =============================================================================
// SLIDE 20 — Reproducibility + close
// =============================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape("rect", {
    x: 0, y: 0, w: W, h: 0.04,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addShape("rect", {
    x: 0.6, y: 1.0, w: 0.18, h: 5.5,
    fill: { color: GOLD }, line: { color: GOLD, width: 0 },
  });
  s.addText("재현성 + 마무리", {
    x: 1.0, y: 0.9, w: 11, h: 0.7,
    fontFace: TITLE_FONT, fontSize: 32, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText("compute envelope · open code · post-audit C-form numbers", {
    x: 1.0, y: 1.7, w: 11, h: 0.4,
    fontFace: BODY_FONT, fontSize: 14, italic: true, color: "CADCFC", margin: 0,
  });

  // Compute envelope
  s.addText("Compute envelope", {
    x: 1.0, y: 2.4, w: 5.5, h: 0.4,
    fontFace: TITLE_FONT, fontSize: 16, bold: true, color: GOLD, margin: 0,
  });
  s.addText([
    { text: "•  8 × NVIDIA H200 (1 GPU shared with vLLM Qwen2.5-32B server, ~60 GB usable per GPU)", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  Total ~5,760 GPU-hours over project lifetime", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  Total ~1.6M model generations across all experiments", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  Wall: 7-model VQAv2 main panel ≈ 6h; E4 Phase 2 ≈ 16h/모델; γ-β Thinking ≈ 45 min on H200", options: { color: "FFFFFF" } },
  ], {
    x: 1.0, y: 2.8, w: 5.5, h: 2.7,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  // Reproducibility
  s.addText("Open release", {
    x: 7.0, y: 2.4, w: 5.5, h: 0.4,
    fontFace: TITLE_FONT, fontSize: 16, bold: true, color: GOLD, margin: 0,
  });
  s.addText([
    { text: "•  All code, configs, anchor inventories (FLUX seeds + OpenCV inpaint params)", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  Per-sample predictions + aggregate CSVs (post-audit, C-form-aligned)", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  Open-weight Hugging Face models — full reproducibility on commodity 8×H200", options: { color: "FFFFFF" } },
    { text: "\n", options: { breakLine: true } },
    { text: "•  M2 audit + C-form migration: docs/insights/M2-metric-definition-evidence.md, C-form-migration-report.md", options: { color: "CADCFC", italic: true } },
  ], {
    x: 7.0, y: 2.8, w: 5.5, h: 2.7,
    fontFace: BODY_FONT, fontSize: 11, valign: "top", margin: 0,
  });

  // Closing
  s.addShape("roundRect", {
    x: 1.0, y: 5.6, w: W - 2.0, h: 1.5,
    fill: { color: TERRA }, line: { color: TERRA, width: 0 }, rectRadius: 0.08,
  });
  s.addText([
    { text: "Cross-modal numerical anchoring is real on VLMs", options: { bold: true, color: "FFFFFF" } },
    { text: " — gated by uncertainty × plausibility × digit-pixel visibility, mechanistically multi-layer-redundant, mitigable at upper-half attention locus, and reasoning trace amplifies rather than catches the bias.", options: { color: "FFFFFF" } },
  ], {
    x: 1.3, y: 5.75, w: W - 2.6, h: 1.2,
    fontFace: BODY_FONT, fontSize: 14, valign: "middle", margin: 0,
  });

  // footer
  s.addText("paper §1–§8 KR summary · namam3gy · 2026-04-29", {
    x: 0.6, y: H - 0.3, w: W - 1.2, h: 0.25,
    fontFace: BODY_FONT, fontSize: 9, color: "CADCFC",
    align: "center", valign: "middle", margin: 0,
  });
}

// ----------------------------------------------------------------------------
pres.writeFile({ fileName: "/tmp/paper-deck/paper_summary_kr.pptx" }).then(fn => {
  console.log("wrote " + fn);
});
