const GITHUB_REPO = "https://github.com/tsinghua-fib-lab/AutoSOTA";
const GITHUB_BLOB = `${GITHUB_REPO}/blob/main/`;

const state = {
  items: [],
  activeFilter: "all",
  activeQuery: "",
  activeSort: "id-asc",
};

// --- DOM refs ---
const $ = (sel) => document.querySelector(sel);
const heroHeadline = $("#hero-headline");
const heroSummary = $("#hero-summary");
const statsGrid = $("#stats-grid");
const featuredGrid = $("#featured-grid");
const runsSummary = $("#runs-summary");
const runsList = $("#runs-list");
const searchInput = $("#search-input");
const filterGrade = $("#filter-grade");
const sortSelect = $("#sort-select");
const modalOverlay = $("#modal-overlay");
const modalContent = $("#modal-content");
const modalClose = $("#modal-close");
const fbTotal = $("#fb-total");
const fbStrong = $("#fb-strong");

$("#year").textContent = String(new Date().getFullYear());

// --- Utilities ---
function cleanMD(text) {
  return text.replace(/<[^>]+>/g, " ").replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_`]/g, "").replace(/\s+/g, " ").trim();
}

function esc(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function parsePct(text) {
  const s = cleanMD(text).replace(/↓/g, "-").replace(/↑/g, "+").replace(/−/g, "-");
  const m = s.match(/([+-]?\d+(?:\.\d+)?)\s*%/);
  return m ? Number(m[1]) : 0;
}

function formatPct(v) {
  if (!Number.isFinite(v)) return "--";
  return (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + "%";
}

function gradeInfo(v) {
  const a = Math.abs(v);
  if (a >= 10) return { grade: "strong", label: "Strong" };
  if (a >= 3) return { grade: "moderate", label: "Moderate" };
  return { grade: "modest", label: "Modest" };
}

function metricColor(v) {
  const a = Math.abs(v);
  if (a >= 10) return "amber";
  if (a >= 3) return "green";
  return "";
}

// --- Parse README ---
function parseLeaderboard(md) {
  const lines = md.split(/\r?\n/);
  // Find simplified header: | ID | Paper Title | Ours\_Optimization |
  const hIdx = lines.findIndex(l => l.includes("| ID | Paper Title | Ours\\_Optimization |"));
  if (hIdx === -1) return [];
  const entries = [];
  for (let i = hIdx + 2; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line.startsWith("|")) break;
    const cells = line.split("|").slice(1, -1).map(c => c.trim());
    if (cells.length < 3) continue;
    const id = Number(cleanMD(cells[0]).replace(/\D/g, ""));
    const title = cleanMD(cells[1]);
    const imp = parsePct(cells[2]);
    const impDisp = cells[2].trim();
    const gi = gradeInfo(imp);
    entries.push({ id, title, improvement: imp, improvementDisplay: impDisp, grade: gi.grade, gradeLabel: gi.label });
  }
  return entries;
}

function parseSummaries(md) {
  const section = (md.split("## Per-paper optimization summaries")[1] || "");
  const blocks = section.split(/\n(?=###\s+\d+\s+[—-]\s+)/).map(b => b.trim()).filter(b => b.startsWith("### "));
  const map = new Map();
  for (const block of blocks) {
    const chunks = block.split(/\n\s*\n/).map(c => c.trim()).filter(Boolean).filter(c => c !== "---");
    if (!chunks.length) continue;
    const hm = chunks[0].match(/^###\s+(\d+)\s+[—-]\s+(.+)$/);
    if (!hm) continue;
    const id = Number(hm[1]);
    const method = cleanMD(hm[2]);
    const title = chunks[1] ? cleanMD(chunks[1]) : "";
    const lc = chunks.find(c => c.includes("](./"));
    const lm = lc ? lc.match(/\((\.\/[^)]+)\)/) : null;
    const relPath = lm ? lm[1] : "";
    const paragraphs = chunks.slice(2).filter(c => !c.startsWith("**[")).map(c => cleanMD(c)).filter(Boolean);
    map.set(id, { method, title, summary: paragraphs[0] || "", paragraphs, relPath,
      blobUrl: relPath ? `${GITHUB_BLOB}${relPath.replace(/^\.\//, "")}` : GITHUB_REPO,
      treeUrl: relPath ? `${GITHUB_REPO}/tree/main/${relPath.replace(/^\.\//, "").replace(/\/[^/]+$/, "")}` : GITHUB_REPO,
    });
  }
  return map;
}

function mergeData(lb, summaries) {
  return lb.map(e => {
    const s = summaries.get(e.id) || {};
    const method = s.method || e.title;
    const ftitle = s.title || e.title;
    const summary = s.summary || "";
    return {
      ...e, method, fullTitle: ftitle,
      summary, paragraphs: s.paragraphs || [],
      blobUrl: s.blobUrl || GITHUB_REPO,
      treeUrl: s.treeUrl || GITHUB_REPO,
      searchText: [e.id, method, ftitle, summary].join(" ").toLowerCase(),
    };
  });
}

// --- Render ---
function renderStats(items) {
  const n = items.length;
  const strongN = items.filter(i => i.grade === "strong").length;
  const validGains = items.map(i => Math.abs(i.improvement)).filter(v => isFinite(v));
  const avgGain = validGains.length ? validGains.reduce((sum, v) => sum + v, 0) / validGains.length : 0;
  const best = items.reduce((a, b) => Math.abs(a.improvement) > Math.abs(b.improvement) ? a : b, items[0]);

  statsGrid.innerHTML = [
    { v: n, l: "papers optimized", c: "" },
    { v: strongN, l: "strong wins (≥10%)", c: "amber" },
    { v: best ? best.improvementDisplay : "--", l: "best gain — " + (best ? esc(best.method || best.fullTitle).slice(0, 30) : ""), c: metricColor(best ? best.improvement : 0) },
    { v: formatPct(avgGain), l: "average gain", c: "" },
  ].map(card => `<article class="stat-card"><span class="stat-value ${card.c}">${esc(String(card.v))}</span><span class="stat-label">${esc(card.l)}</span></article>`).join("");

  heroHeadline.textContent = `${n} papers optimized, one pipeline.`;
  heroSummary.textContent = `${strongN} papers cleared the 10% threshold. Every entry links to its optimization note and source folder.`;

  fbTotal.textContent = String(n);
  fbStrong.textContent = String(strongN);
}

function renderFeatured(items) {
  const featured = [...items].sort((a, b) => Math.abs(b.improvement) - Math.abs(a.improvement)).slice(0, 6);
  if (!featured.length) { featuredGrid.innerHTML = '<article class="featured-card"><p>No featured runs available.</p></article>'; return; }

  featuredGrid.innerHTML = featured.map(item => {
    const gi = gradeInfo(item.improvement);
    return `<article class="featured-card" data-id="${item.id}">
      <div class="featured-card-header">
        <span class="featured-id">paper ${item.id}</span>
        <span class="status-pill status-${gi.grade}">${gi.label}</span>
      </div>
      <h3>${esc(item.method || item.fullTitle)}</h3>
      <p class="featured-title">${esc(item.fullTitle)}</p>
      <p class="featured-summary">${esc((item.summary || "").slice(0, 150))}</p>
      <div class="featured-footer">
        <strong class="${metricColor(item.improvement)}">${esc(item.improvementDisplay)}</strong>
        <span class="featured-link">View details →</span>
      </div>
    </article>`;
  }).join("");

  featuredGrid.querySelectorAll(".featured-card").forEach(card => {
    card.addEventListener("click", () => openModal(Number(card.dataset.id)));
  });
}

function filterItems() {
  let items = state.items;
  const q = state.activeQuery.trim().toLowerCase();
  if (state.activeFilter !== "all") {
    items = items.filter(i => i.grade === state.activeFilter);
  }
  if (q) {
    items = items.filter(i => i.searchText.includes(q));
  }
  // Sort
  switch (state.activeSort) {
    case "improvement-desc":
      items = [...items].sort((a, b) => Math.abs(b.improvement) - Math.abs(a.improvement));
      break;
    case "improvement-asc":
      items = [...items].sort((a, b) => Math.abs(a.improvement) - Math.abs(b.improvement));
      break;
    default: // id-asc
      items = [...items].sort((a, b) => a.id - b.id);
  }
  return items;
}

function renderRuns() {
  const visible = filterItems();
  runsSummary.textContent = `Showing ${visible.length} of ${state.items.length} papers.`;

  if (!visible.length) {
    runsList.innerHTML = '<article class="run-card run-card-empty"><p>No papers match the current filter.</p></article>';
    return;
  }

  runsList.innerHTML = visible.map(item => {
    const gi = gradeInfo(item.improvement);
    const mc = metricColor(item.improvement);
    return `<article class="run-card" data-id="${item.id}">
      <span class="run-card-id">#${item.id}</span>
      <div class="run-card-title">
        <h3>${esc(item.method || item.fullTitle)}</h3>
        <p>${esc(item.fullTitle)}</p>
      </div>
      <div class="run-card-metric">
        <span class="metric-value ${mc}">${esc(item.improvementDisplay)}</span>
        <span class="metric-label">optimization gain</span>
      </div>
      <div class="run-card-status">
        <span class="status-pill status-${gi.grade}">${gi.label}</span>
      </div>
    </article>`;
  }).join("");

  runsList.querySelectorAll(".run-card:not(.run-card-empty)").forEach(card => {
    card.addEventListener("click", () => openModal(Number(card.dataset.id)));
  });
}

// --- Modal ---
function openModal(id) {
  const item = state.items.find(i => i.id === id);
  if (!item) return;
  const gi = gradeInfo(item.improvement);
  const mc = metricColor(item.improvement);
  const summaryHTML = (item.paragraphs.length ? item.paragraphs : [item.summary || "No summary available."])
    .map(p => `<p>${esc(p)}</p>`).join("");

  modalContent.innerHTML = `
    <span class="status-pill status-${gi.grade}" style="margin-bottom:12px">${gi.label} · ${esc(item.improvementDisplay)}</span>
    <h2 id="modal-title">${esc(item.method || item.fullTitle)}</h2>
    <p class="modal-title">${esc(item.fullTitle)}</p>
    <div class="modal-meta">
      <div class="modal-meta-item"><span class="mm-label">Paper ID</span><span class="mm-value">#${item.id}</span></div>
      <div class="modal-meta-item"><span class="mm-label">Optimization</span><span class="mm-value ${mc}">${esc(item.improvementDisplay)}</span></div>
      <div class="modal-meta-item"><span class="mm-label">Grade</span><span class="mm-value ${mc}">${gi.label}</span></div>
    </div>
    <div class="modal-summary">${summaryHTML}</div>
    <div class="modal-links">
      <a class="button" href="${esc(item.blobUrl)}" target="_blank" rel="noreferrer">View OPTIMIZATION.md</a>
      <a class="button" href="${esc(item.treeUrl)}" target="_blank" rel="noreferrer">Browse Folder</a>
      <a class="button" href="${GITHUB_REPO}/issues/new" target="_blank" rel="noreferrer">Report Issue</a>
    </div>
  `;
  modalOverlay.classList.add("is-open");
  modalOverlay.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  modalOverlay.classList.remove("is-open");
  modalOverlay.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

modalClose.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modalOverlay.classList.contains("is-open")) closeModal();
});

// --- Filters ---
function setActiveFilter(filter) {
  state.activeFilter = filter;
  filterGrade.querySelectorAll(".filter-button").forEach(b => {
    b.classList.toggle("is-active", b.dataset.filter === filter);
  });
}

filterGrade.addEventListener("click", (e) => {
  if (!(e.target instanceof HTMLButtonElement)) return;
  setActiveFilter(e.target.dataset.filter || "all");
  renderRuns();
});

searchInput.addEventListener("input", (e) => {
  state.activeQuery = e.target.value;
  renderRuns();
});

sortSelect.addEventListener("change", (e) => {
  state.activeSort = e.target.value;
  renderRuns();
});

// --- Init ---
async function init() {
  try {
    const resp = await fetch("site-data/README.md", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const md = await resp.text();
    const lb = parseLeaderboard(md);
    const summaries = parseSummaries(md);
    state.items = mergeData(lb, summaries).sort((a, b) => a.id - b.id);

    renderStats(state.items);
    renderFeatured(state.items);
    renderRuns();
  } catch (err) {
    heroHeadline.textContent = "Could not load repository data.";
    heroSummary.textContent = err.message;
    featuredGrid.innerHTML = `<article class="featured-card"><p>${esc(err.message)}</p></article>`;
    runsSummary.textContent = "Failed to load.";
    runsList.innerHTML = `<article class="run-card run-card-empty"><p>${esc(err.message)}</p></article>`;
  }
}

init();
