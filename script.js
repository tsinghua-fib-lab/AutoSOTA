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

const state = {
  items: [],
  activeFilter: "all",
  activeQuery: "",
  activeSort: "id-asc",
};

$("#year").textContent = String(new Date().getFullYear());

// --- Render ---
function renderStats(items) {
  const n = items.length;
  const strongN = items.filter(i => i.grade === "strong").length;
  const validGains = items.map(i => Math.abs(i.improvement)).filter(v => Number.isFinite(v));
  const avgGain = validGains.length ? validGains.reduce((sum, v) => sum + v, 0) / validGains.length : 0;
  const best = items.reduce((a, b) => Math.abs(a.improvement) > Math.abs(b.improvement) ? a : b, items[0]);

  statsGrid.innerHTML = [
    { v: n, l: "papers enchanted", c: "" },
    { v: strongN, l: "strong wins (≥10%)", c: "amber" },
    { v: best ? best.improvementDisplay : "--", l: "best gain — " + (best ? esc(best.method || best.fullTitle).slice(0, 30) : ""), c: metricColor(best ? best.improvement : 0) },
    { v: formatPct(avgGain), l: "average gain", c: "" },
  ].map(card => `<article class="stat-card"><span class="stat-value ${card.c}">${esc(String(card.v))}</span><span class="stat-label">${esc(card.l)}</span></article>`).join("");

  heroHeadline.textContent = `${n} papers enchanted, one pipeline.`;
  heroSummary.textContent = `${strongN} papers cleared the 10% threshold. Every entry links to its results note and source folder.`;

  fbTotal.textContent = String(n);
  fbStrong.textContent = String(strongN);
}

function renderFeatured(items) {
  const featured = [...items].sort((a, b) => Math.abs(b.improvement) - Math.abs(a.improvement)).slice(0, 6);
  if (!featured.length) { featuredGrid.innerHTML = '<article class="featured-card"><p>No featured runs available.</p></article>'; return; }

  featuredGrid.innerHTML = featured.map(item => {
    const gi = gradeInfo(item.improvement);
    return `<article class="featured-card" data-id="${item.id}" tabindex="0" role="button" aria-label="View details for ${esc(item.method || item.fullTitle)}">
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
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModal(Number(card.dataset.id)); }
    });
  });
}

function filterItems() {
  const items = [...state.items];
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
      items.sort((a, b) => Math.abs(b.improvement) - Math.abs(a.improvement));
      break;
    case "improvement-asc":
      items.sort((a, b) => Math.abs(a.improvement) - Math.abs(b.improvement));
      break;
    default: // id-asc
      items.sort((a, b) => a.id - b.id);
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
    return `<article class="run-card" data-id="${item.id}" tabindex="0" role="button" aria-label="View details for ${esc(item.method || item.fullTitle)}">
      <span class="run-card-id">#${item.id}</span>
      <div class="run-card-title">
        <h3>${esc(item.method || item.fullTitle)}</h3>
        <p>${esc(item.fullTitle)}</p>
      </div>
      <div class="run-card-metric">
        <span class="metric-value ${mc}">${esc(item.improvementDisplay)}</span>
        <span class="metric-label">improvement</span>
      </div>
      <div class="run-card-status">
        <span class="status-pill status-${gi.grade}">${gi.label}</span>
      </div>
    </article>`;
  }).join("");

  runsList.querySelectorAll(".run-card:not(.run-card-empty)").forEach(card => {
    card.addEventListener("click", () => openModal(Number(card.dataset.id)));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModal(Number(card.dataset.id)); }
    });
  });
}

let lastFocusedEl = null;

// --- Modal ---
function openModal(id) {
  const item = state.items.find(i => i.id === id);
  if (!item) return;
  lastFocusedEl = document.activeElement;
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
      <div class="modal-meta-item"><span class="mm-label">Improvement</span><span class="mm-value ${mc}">${esc(item.improvementDisplay)}</span></div>
      <div class="modal-meta-item"><span class="mm-label">Grade</span><span class="mm-value ${mc}">${gi.label}</span></div>
    </div>
    <div class="modal-summary">${summaryHTML}</div>
    <div class="modal-links">
      <a class="button" href="${item.blobUrl}" target="_blank" rel="noreferrer">View README.md</a>
      <a class="button" href="${item.treeUrl}" target="_blank" rel="noreferrer">Browse Folder</a>
      <a class="button" href="${GITHUB_REPO}/issues/new" target="_blank" rel="noreferrer">Report Issue</a>
    </div>
  `;
  modalOverlay.classList.add("is-open");
  modalOverlay.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  modalClose.focus();
}

function closeModal() {
  modalOverlay.classList.remove("is-open");
  modalOverlay.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
  if (lastFocusedEl && typeof lastFocusedEl.focus === "function") lastFocusedEl.focus();
}

modalClose.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modalOverlay.classList.contains("is-open")) { closeModal(); return; }
  if (e.key === "Tab" && modalOverlay.classList.contains("is-open")) {
    const focusable = modalOverlay.querySelectorAll("button, a[href], input, select, textarea, [tabindex]:not([tabindex='-1'])");
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!first) return;
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
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

let searchTimer = null;
searchInput.addEventListener("input", (e) => {
  state.activeQuery = e.target.value;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => renderRuns(), 200);
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
