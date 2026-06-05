// CVPR 2026 Monitor
const GITHUB_REPO = "https://github.com/tsinghua-fib-lab/AutoSOTA";

const state = {
  papers: { day1: [], day2: [], day3: [] },
  activeDay: "day1",
  activeFilters: new Set(),
  activeQuery: "",
  showImprovedOnly: false,
  page: 1,
  pageSize: 50,
};

const $ = (sel) => document.querySelector(sel);
const heroHeadline = $("#hero-headline");
const heroSummary = $("#hero-summary");
const statsGrid = $("#stats-grid");
const dayTabs = $("#day-tabs");
const runsSummary = $("#runs-summary");
const runsList = $("#runs-list");
const searchInput = $("#search-input");
const improvedToggle = $("#improved-toggle");
const fbTotal = $("#fb-total");
const fbCats = $("#fb-cats");

// --- Category classification ---
const CATEGORY_GROUPS = {
  "Reproduction Succeeded": "success",
  "Metrics Below Paper Claims": "success",
  "Incomplete Repository; Not Fully Reproducible": "failure",
  "No GitHub Repo Found Or Clone Failed": "failure",
  "Non-Methodological Paper": "na",
  "Data/Model Resources Unavailable": "failure",
  "Storage/Compute/Network Infrastructure Drawback": "failure",
  "Environment/Dependency Setup Failed": "failure",
  "Reproduction Pending": "pending",
  "Other Reproduction Failure": "failure",
};

function catGroup(cat) { return CATEGORY_GROUPS[cat] || "failure"; }

function esc(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// --- Stats ---
function renderStats() {
  const total = state.papers.day1.length + state.papers.day2.length + state.papers.day3.length;

  const cards = [
    { v: "—", l: "papers indexed", c: "", tip: "Total CVPR 2026 papers tracked and monitored by AutoSOTA across all three conference days." },
    { v: "—", l: "SOTA ratio", c: "", tip: "Percentage of indexed papers where AutoSOTA achieved a measurable improvement over the reported state-of-the-art results." },
    { v: "—", l: "reproduced", c: "", tip: "Papers whose code repositories were successfully cloned, built, and verified to produce the original reported outputs." },
    { v: "—", l: "enchanted", c: "", tip: "Papers where AutoSOTA's automated pipeline went beyond reproduction to discover and apply novel optimizations that improve performance." },
  ];
  statsGrid.innerHTML = cards.map(card => `<article class="stat-card">
    <span class="stat-value ${card.c}">${esc(card.v)}</span>
    <div class="stat-label-row">
      <span class="stat-label">${esc(card.l)}</span>
      <span class="stat-tip" tabindex="0" role="tooltip" aria-label="${esc(card.tip)}">?</span>
    </div>
  </article>`).join("");

  heroHeadline.textContent = `Day by day, as CVPR unfolds.`;
  heroSummary.textContent = `AutoSOTA runs every paper, updates results daily — June 5–7, live from the conference floor.`;

  fbTotal.textContent = String(total);
  fbCats.textContent = "—";
}

// --- Day tabs ---
function renderDayTabs() {
  const days = [
    { key: "day1", label: "Day 1", date: "June 5" },
    { key: "day2", label: "Day 2", date: "June 6" },
    { key: "day3", label: "Day 3", date: "June 7" },
  ];
  dayTabs.innerHTML = days.map(d => {
    const count = state.papers[d.key].length;
    const active = d.key === state.activeDay ? " is-active" : "";
    return `<button class="day-tab${active}" role="tab" aria-selected="${d.key === state.activeDay}" data-day="${d.key}">
      <span>${d.label}</span>
      <span class="day-tab-count">${d.date} · ${count} papers</span>
    </button>`;
  }).join("");

  dayTabs.querySelectorAll(".day-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      state.activeDay = tab.dataset.day;
      state.page = 1;
      renderDayTabs();
      renderRuns();
    });
  });
}

// --- Paper list ---
const STATUS_RANK = { "Oral": 0, "Highlight": 1, "Poster": 2 };

function getVisiblePapers() {
  let papers = state.papers[state.activeDay];
  if (state.showImprovedOnly) {
    papers = papers.filter(p => p["指标提升相对百分比"] != null);
  }
  if (state.activeFilters.size > 0) {
    papers = papers.filter(p => state.activeFilters.has(p.AutoSOTA_Category || "Uncategorized"));
  }
  const q = state.activeQuery.trim().toLowerCase();
  if (q) {
    papers = papers.filter(p =>
      (p.paper_id || "").toLowerCase().includes(q) ||
      (p.title || "").toLowerCase().includes(q) ||
      (p.abstract || "").toLowerCase().includes(q)
    );
  }
  // Sort: improved first, then by status rank (Oral > Highlight > Poster)
  papers = [...papers].sort((a, b) => {
    const aImp = a["指标提升相对百分比"] != null ? 0 : 1;
    const bImp = b["指标提升相对百分比"] != null ? 0 : 1;
    if (aImp !== bImp) return aImp - bImp;
    const aRank = STATUS_RANK[a.status] ?? 3;
    const bRank = STATUS_RANK[b.status] ?? 3;
    return aRank - bRank;
  });
  return papers;
}

function formatImprovement(raw) {
  if (raw == null) return null;
  const s = String(raw).replace(/[？?]/g, "").trim();
  const n = parseFloat(s);
  if (isNaN(n)) return s; // multi-metric text, return as-is
  return (n * 100).toFixed(1) + "%";
}

function renderRuns() {
  const allVisible = getVisiblePapers();
  const total = state.papers[state.activeDay].length;
  const totalPages = Math.ceil(allVisible.length / state.pageSize) || 1;
  if (state.page > totalPages) state.page = 1;
  const start = (state.page - 1) * state.pageSize;
  const visible = allVisible.slice(start, start + state.pageSize);

  const parts = [`${allVisible.length} of ${total} papers`];
  if (state.showImprovedOnly) parts.push("(improved only)");
  else if (state.activeFilters.size) parts.push("(filtered)");
  runsSummary.innerHTML = `<span>${parts.join(" ")}</span>
    <span class="page-nav">
      <button class="page-btn" id="page-prev" ${state.page <= 1 ? "disabled" : ""}>← Prev</button>
      <span class="page-info">${state.page} / ${totalPages}</span>
      <button class="page-btn" id="page-next" ${state.page >= totalPages ? "disabled" : ""}>Next →</button>
    </span>`;
  if (!visible.length) {
    runsList.innerHTML = '<article class="run-card run-card-empty"><p>No papers match the current filters.</p></article>';
    return;
  }

  runsList.innerHTML = visible.map(p => {
    const group = catGroup(p.AutoSOTA_Category);
    const cat = p.AutoSOTA_Category || "Uncategorized";
    const statusClass = (p.status || "").toLowerCase();
    const impPct = p["指标提升相对百分比"];
    const hasImprovement = impPct != null;

    const issueTitle = encodeURIComponent(`[CVPR 2026] ${p.paper_id}: ${(p.title || "").slice(0, 80)}`);
    const issueURL = `${GITHUB_REPO}/issues/new?labels=cvpr2026&title=${issueTitle}`;

    const links = [];
    if (p.pdf_url) links.push(`<a href="${esc(p.pdf_url)}" target="_blank" rel="noreferrer">PDF</a>`);
    if (p.arxiv) links.push(`<a href="https://arxiv.org/abs/${esc(p.arxiv)}" target="_blank" rel="noreferrer">arXiv:${esc(p.arxiv)}</a>`);
    if (p.github_url) links.push(`<a href="${esc(p.github_url)}" target="_blank" rel="noreferrer">GitHub</a>`);

    const impDisplay = formatImprovement(impPct);
    const metricHTML = hasImprovement
      ? `<span class="metric-delta positive">↑ ${esc(impDisplay)}</span>`
      : "";

    const absShort = (p.abstract || "").slice(0, 200);

    return `<article class="cvpr-row${hasImprovement ? " has-improvement" : ""}" data-id="${esc(p.paper_id)}">
      <div class="cvpr-row-main">
        <button class="cvpr-row-title-btn" onclick="this.closest('.cvpr-row').classList.toggle('is-expanded')" aria-expanded="false">
          ${esc(p.title)}
        </button>
        ${absShort ? `<p class="cvpr-row-abstract">${esc(absShort)}${(p.abstract || "").length > 200 ? "…" : ""}</p>` : ""}
        <div class="cvpr-row-links">${links.join(" · ") || '<span style="color:var(--muted)">No links</span>'}</div>
      </div>
      <div class="cvpr-row-status">
        <span class="status-badge status-${statusClass}">${esc(p.status)}</span>
      </div>
      <div class="cvpr-row-category">
        ${cat !== "Reproduction Succeeded" ? `<span class="cat-label cat-${group}">${esc(cat)}</span>` : ""}
        ${hasImprovement ? `<span class="cat-meta">${metricHTML}</span>` : ""}
      </div>
      <div class="cvpr-row-actions">
        <a class="issue-link" href="${issueURL}" target="_blank" rel="noreferrer">Report Issue</a>
      </div>
    </article>`;
  }).join("");
}

// --- Init ---
async function loadDay(dayKey) {
  try {
    const resp = await fetch(`site-data/cvpr_${dayKey}.json`, { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    state.papers[dayKey] = await resp.json();
  } catch (err) {
    console.error(`Failed to load ${dayKey}:`, err);
    state.papers[dayKey] = [];
  }
}

async function init() {
  await Promise.all([loadDay("day1"), loadDay("day2"), loadDay("day3")]);

  const total = state.papers.day1.length + state.papers.day2.length + state.papers.day3.length;
  if (!total) {
    heroHeadline.textContent = "Could not load CVPR data.";
    heroSummary.textContent = "Check that site-data JSON files are available.";
    return;
  }

  renderStats();
  renderDayTabs();
  renderRuns();
}

// --- Event listeners ---
let searchTimer = null;
searchInput.addEventListener("input", (e) => {
  state.activeQuery = e.target.value;
  state.page = 1;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => renderRuns(), 200);
});

improvedToggle.addEventListener("click", () => {
  state.showImprovedOnly = !state.showImprovedOnly;
  state.page = 1;
  improvedToggle.classList.toggle("is-active", state.showImprovedOnly);
  renderRuns();
});

document.addEventListener("click", (e) => {
  if (e.target.id === "page-prev" && state.page > 1) { state.page--; renderRuns(); }
  if (e.target.id === "page-next") { state.page++; renderRuns(); }
});

init();
