const CONDITIONS = ["b", "a", "m", "d"];
const COND_LABEL = {
  b: "b · base",
  a: "a · anchor",
  m: "m · masked",
  d: "d · neutral",
};
const COND_HEADER = {
  b: "base cond",
  a: "anchor cond",
  m: "mask cond",
  d: "distractor cond",
};

const state = {
  data: null,
  sampleId: null,
  condition: "a",
};

async function init() {
  const root = document.getElementById("demo-app");
  let resp;
  try {
    resp = await fetch("data/demo.json", { cache: "no-store" });
  } catch (err) {
    root.innerHTML = `<p class="text-red-600">Failed to fetch demo.json: ${escapeHtml(String(err))}</p>`;
    return;
  }
  if (!resp.ok) {
    root.innerHTML = `<p class="text-red-600">Failed to load demo.json (${resp.status}).</p>`;
    return;
  }
  state.data = await resp.json();
  if (!state.data.samples.length) {
    root.textContent = "No samples available.";
    return;
  }
  state.sampleId = state.data.samples[0].id;
  render();
  wireBibtex();
}

function currentSample() {
  return state.data.samples.find((s) => s.id === state.sampleId);
}

function secondImageFor(sample, cond) {
  if (cond === "a") return sample.images.anchor;
  if (cond === "m") return sample.images.masked;
  if (cond === "d") return sample.images.neutral;
  return null;
}

function render() {
  const root = document.getElementById("demo-app");
  const s = currentSample();
  // Preserve page scroll + thumbnail-row horizontal scroll across the
  // innerHTML rebuild — otherwise picking a sample bounces the viewport.
  const pageScrollY = window.scrollY;
  const pageScrollX = window.scrollX;
  const oldThumbs = root.querySelector("[data-thumbs]");
  const thumbsScrollLeft = oldThumbs ? oldThumbs.scrollLeft : 0;

  const thumbs = state.data.samples.map((sm) => `
    <button class="text-left shrink-0" data-sid="${escapeHtml(sm.id)}" type="button">
      <img class="thumb ${sm.id === state.sampleId ? "selected" : ""}"
           src="${escapeHtml(sm.images.target)}" alt="${escapeHtml(sm.id)}" />
      <div class="text-xs text-neutral-500 mt-1">${escapeHtml(sm.dataset)}</div>
    </button>
  `).join("");

  const condButtons = CONDITIONS.map((c) => `
    <button class="cond-btn ${c === state.condition ? "active" : ""}"
            data-cond="${c}" type="button">${COND_LABEL[c]}</button>
  `).join("");

  const rows = state.data.models.map((m) => {
    const baselinePred = s.predictions[m.id]["b"];
    const baselineAtAnchor = baselinePred === s.anchor;
    const cells = CONDITIONS.map((c) => {
      const pred = s.predictions[m.id][c];
      const isGt = pred === s.gt;
      // Anchor cond column carries one of two effect tags. Both follow
      // the canonical M2 definitions (see references/AGENTS.md):
      //   (adopt) = pa == anchor AND pb != anchor — the model wasn't
      //             already producing the anchor on baseline, so the
      //             a-arm move is genuinely toward the anchor
      //   (df)    = (pa - pb)·(anchor - pb) > 0 AND pa != pb — moved
      //             toward the anchor without fully adopting (the
      //             C-form direction-follow condition; if pb already
      //             equals the anchor, the dot-product is 0, so df is
      //             also false there)
      // Other columns get no tag. Bold weight is reserved for adopt
      // (df is partial pull — colour alone is enough).
      let tag = "";
      let isAdopt = false;
      if (c === "a") {
        if (pred === s.anchor && !baselineAtAnchor) {
          tag = '<span class="pred-tag adopt">(adopt)</span>';
          isAdopt = true;
        } else if (
          pred !== baselinePred
          && Math.sign(pred - baselinePred) === Math.sign(s.anchor - baselinePred)
        ) {
          tag = '<span class="pred-tag df">(df)</span>';
        }
      }
      const cls = [
        "pred-cell",
        isGt ? "gt" : "wrong",
        c === state.condition ? "cond-active" : "",
        isAdopt ? "pulled" : "",
      ].filter(Boolean).join(" ");
      return `<td class="${cls}">${pred}${tag}</td>`;
    }).join("");
    return `<tr><td>${escapeHtml(m.label)}</td>${cells}</tr>`;
  }).join("");

  const adoptCount = state.data.models.filter(
    (m) => s.predictions[m.id].a === s.anchor
  ).length;
  const caption = `${escapeHtml(s.dataset)} · GT = ${s.gt} · anchor = ${s.anchor} · ${adoptCount} of ${state.data.models.length} models adopt the anchor under condition a.`;

  const second = secondImageFor(s, state.condition);
  const showSecond = second !== null;

  root.innerHTML = `
    <div>
      <div class="text-xs uppercase tracking-wider text-neutral-500 mb-2">Pick a sample</div>
      <div class="flex gap-3 overflow-x-auto pb-2" data-thumbs>${thumbs}</div>
    </div>
    <div class="rounded-md border border-neutral-200 p-4 space-y-4">
      <div class="text-base md:text-lg leading-snug">
        <span class="font-semibold text-[var(--accent)]">Q:</span> ${escapeHtml(s.question)}
      </div>
      <div class="flex flex-wrap items-baseline gap-x-6 gap-y-1 text-base md:text-lg font-mono">
        <span><span class="text-neutral-500 mr-1">GT</span><span class="font-bold gt-value">${s.gt}</span></span>
        <span><span class="text-neutral-500 mr-1">anchor</span><span class="font-bold anchor-value">${s.anchor}</span></span>
        <span class="text-sm text-neutral-500">|Δ| = ${Math.abs(s.gt - s.anchor)}</span>
      </div>
      <div class="grid ${showSecond ? "md:grid-cols-2" : "grid-cols-1"} gap-4">
        <figure>
          <a href="${escapeHtml(s.images.target)}" target="_blank" rel="noopener" class="image-frame block">
            <img src="${escapeHtml(s.images.target)}" class="image-fit" alt="target" />
          </a>
          <figcaption class="text-xs text-neutral-500 mt-1">target image · click for full size</figcaption>
        </figure>
        ${showSecond ? `
        <figure>
          <a href="${escapeHtml(second)}" target="_blank" rel="noopener" class="image-frame block">
            <img src="${escapeHtml(second)}" class="image-fit" alt="${escapeHtml(state.condition)}" />
          </a>
          <figcaption class="text-xs text-neutral-500 mt-1">condition <code>${escapeHtml(state.condition)}</code> · click for full size</figcaption>
        </figure>` : ""}
      </div>
      <div class="flex gap-2 flex-wrap">${condButtons}</div>
      <div class="overflow-x-auto">
        <table class="pred-table">
          <thead>
            <tr>
              <th>Model</th>
              ${CONDITIONS.map((c) => `<th>${escapeHtml(COND_HEADER[c])}</th>`).join("")}
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <p class="text-sm text-neutral-600">${caption}</p>
    </div>
  `;

  root.querySelectorAll("[data-sid]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.sampleId = btn.dataset.sid;
      render();
    });
  });
  root.querySelectorAll("[data-cond]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.condition = btn.dataset.cond;
      render();
    });
  });

  // Restore both scroll positions after the layout settles. Doing it on
  // the next frame avoids a double-paint flicker during the rebuild.
  const newThumbs = root.querySelector("[data-thumbs]");
  if (newThumbs) newThumbs.scrollLeft = thumbsScrollLeft;
  requestAnimationFrame(() => window.scrollTo(pageScrollX, pageScrollY));
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wireBibtex() {
  const btn = document.getElementById("copy-bibtex");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const text = document.getElementById("bibtex").textContent;
    try {
      await navigator.clipboard.writeText(text);
      const orig = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => { btn.textContent = orig; }, 1200);
    } catch {
      btn.textContent = "Failed";
    }
  });
}

document.addEventListener("DOMContentLoaded", init);
