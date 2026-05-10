const CONDITIONS = ["b", "a", "m", "d"];
const COND_LABEL = {
  b: "b · base",
  a: "a · anchor",
  m: "m · masked",
  d: "d · neutral",
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
    root.innerHTML = `<p class="text-red-600">Failed to fetch demo.json: ${err}</p>`;
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

  const thumbs = state.data.samples.map((sm) => `
    <button class="text-left shrink-0" data-sid="${sm.id}" type="button">
      <img class="thumb ${sm.id === state.sampleId ? "selected" : ""}"
           src="${sm.images.target}" alt="${sm.id}" />
      <div class="text-xs text-neutral-500 mt-1">${sm.dataset}</div>
    </button>
  `).join("");

  const condButtons = CONDITIONS.map((c) => `
    <button class="cond-btn ${c === state.condition ? "active" : ""}"
            data-cond="${c}" type="button">${COND_LABEL[c]}</button>
  `).join("");

  const rows = state.data.models.map((m) => {
    const cells = CONDITIONS.map((c) => {
      const pred = s.predictions[m.id][c];
      const isGt = pred === s.gt;
      const isAnchor = pred === s.anchor;
      const cls = [
        "pred-cell",
        isGt ? "gt" : "",
        c === state.condition ? "cond-active" : "",
      ].filter(Boolean).join(" ");
      const mark = isAnchor ? '<span class="anchor-mark">⚓</span>' : "";
      return `<td class="${cls}">${pred}${mark}</td>`;
    }).join("");
    return `<tr><td>${m.label}</td>${cells}</tr>`;
  }).join("");

  const adoptCount = state.data.models.filter(
    (m) => s.predictions[m.id].a === s.anchor
  ).length;
  const caption = `${s.dataset} · GT = ${s.gt} · anchor = ${s.anchor} · ${adoptCount} of ${state.data.models.length} models adopt the anchor under condition a.`;

  const second = secondImageFor(s, state.condition);
  const showSecond = second !== null;

  root.innerHTML = `
    <div>
      <div class="text-xs uppercase tracking-wider text-neutral-500 mb-2">Pick a sample</div>
      <div class="flex gap-3 overflow-x-auto pb-2">${thumbs}</div>
    </div>
    <div class="rounded-md border border-neutral-200 p-4 space-y-4">
      <div class="text-sm">
        <span class="font-semibold">Q:</span> ${escapeHtml(s.question)}
        <span class="ml-3 text-neutral-500">GT = ${s.gt}, anchor = ${s.anchor}</span>
      </div>
      <div class="grid ${showSecond ? "md:grid-cols-2" : "grid-cols-1"} gap-4">
        <figure>
          <img src="${s.images.target}" class="w-full rounded-md border border-neutral-200" alt="target" />
          <figcaption class="text-xs text-neutral-500 mt-1">target image</figcaption>
        </figure>
        ${showSecond ? `
        <figure>
          <img src="${second}" class="w-full rounded-md border border-neutral-200" alt="${state.condition}" />
          <figcaption class="text-xs text-neutral-500 mt-1">condition <code>${state.condition}</code></figcaption>
        </figure>` : ""}
      </div>
      <div class="flex gap-2 flex-wrap">${condButtons}</div>
      <div class="overflow-x-auto">
        <table class="pred-table">
          <thead>
            <tr>
              <th>Model</th>
              ${CONDITIONS.map((c) => `<th>${c}</th>`).join("")}
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
