// CVPR 2026 Monitor
const GITHUB_REPO = "https://github.com/tsinghua-fib-lab/AutoSOTA";

const state = {
  papers: { day1: [], day2: [], day3: [] },
  activeDay: "day2",
  activeQuery: "",
  showImprovedOnly: false,
  showAllCategories: false,
  page: 1,
  pageSize: 30,
};

const $ = (sel) => document.querySelector(sel);
const heroHeadline = $("#hero-headline");
const statsGrid = $("#stats-grid");
const dayTabs = $("#day-tabs");
const runsSummary = $("#runs-summary");
const runsList = $("#runs-list");
const searchInput = $("#search-input");
const improvedToggle = $("#improved-toggle");
const allCatToggle = $("#allcat-toggle");

function esc(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

// --- Stats ---
function renderStats() {
  statsGrid.innerHTML = `<article class="stat-card countdown-card">
    <span class="countdown-label">Next update</span>
    <span class="countdown-timer" id="countdown-timer">--:--:--</span>
  </article>`;

  heroHeadline.innerHTML = `When AutoSOTA Meets<br>Top Conference Papers`;

  startCountdown();
}

// --- Countdown ---
function startCountdown() {
  function tick() {
    const now = new Date();
    const target = new Date(now);
    target.setHours(24, 0, 0, 0);
    const diff = target - now;
    if (diff <= 0) { document.getElementById("countdown-timer").textContent = "Updating..."; return; }
    const h = Math.floor(diff / 3600000);
    const m = Math.floor((diff % 3600000) / 60000);
    const s = Math.floor((diff % 60000) / 1000);
    document.getElementById("countdown-timer").textContent =
      String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
  }
  tick();
  setInterval(tick, 1000);
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

const CAT_COLORS = {
  "Missing Repo": "cat-gray", "Incomplete Repo": "cat-gray",
  "Non-Method Paper": "cat-gray", "Missing Data": "cat-gray",
  "Insufficient Resources": "cat-amber",
  "Succeeded": "cat-green",
  "Setup Failed": "cat-red", "Missed Claims": "cat-red",
  "No Improvement": "cat-red",
};

const CAT_COLOR_RANK = {
  "Succeeded": 0,
  "Insufficient Resources": 1,
  "Setup Failed": 2, "Missed Claims": 2, "No Improvement": 2,
  "Missing Repo": 3, "Incomplete Repo": 3, "Non-Method Paper": 3, "Missing Data": 3,
};

function getVisiblePapers() {
  let papers = state.papers[state.activeDay];
  if (!state.showAllCategories) {
    papers = papers.filter(p => {
      const color = CAT_COLORS[p["AutoSOTA_Category"]];
      return color && color !== "cat-gray";
    });
  }
  if (state.showImprovedOnly) {
    papers = papers.filter(p => p["AutoSOTA_Category"] === "Succeeded");
  }
  const q = state.activeQuery.trim().toLowerCase();
  if (q) {
    papers = papers.filter(p =>
      String(p.seq || "").includes(q) ||
      (p.paper_id || "").toLowerCase().includes(q) ||
      (p.title || "").toLowerCase().includes(q)
    );
  }
  // Sort: improved first, then status, then category color
  papers = [...papers].sort((a, b) => {
    const aImp = a["AutoSOTA_Category"] === "Succeeded" ? 0 : 1;
    const bImp = b["AutoSOTA_Category"] === "Succeeded" ? 0 : 1;
    if (aImp !== bImp) return aImp - bImp;
    const aStatus = STATUS_RANK[a.status] ?? 3;
    const bStatus = STATUS_RANK[b.status] ?? 3;
    if (aStatus !== bStatus) return aStatus - bStatus;
    const aColor = CAT_COLOR_RANK[a["AutoSOTA_Category"]] ?? 4;
    const bColor = CAT_COLOR_RANK[b["AutoSOTA_Category"]] ?? 4;
    return aColor - bColor;
  });
  return papers;
}

function formatImprovement(raw) {
  if (raw == null) return null;
  const s = String(raw).replace(/[？?]/g, "").trim();
  const n = parseFloat(s);
  if (isNaN(n)) return s;
  // Values with % are already percentages; bare values < 1 are ratios.
  const pct = s.includes("%") ? n.toFixed(1) : (n < 1 ? (n * 100).toFixed(1) : n.toFixed(1));
  return pct + "%";
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
    const statusClass = (p.status || "").toLowerCase();
    const impPct = p["指标提升百分比(绝对值)"];
    const cat = p["AutoSOTA_Category"];
    const hasImprovement = cat === "Succeeded";
    const optNoteRaw = p["优化说明"];
    const optNote = optNoteRaw && !/^Performance\s*enhanced\s*successfully/i.test(optNoteRaw) ? optNoteRaw : null;

    const seqLabel = p.seq ? `seq ${p.seq} · ` : "";
    const issueTitle = encodeURIComponent(`[CVPR 2026] ${seqLabel}${p.paper_id}: ${(p.title || "").slice(0, 80)}`);
    const issueURL = `${GITHUB_REPO}/issues/new?labels=cvpr2026&title=${issueTitle}`;

    const links = [];
    const hasArxiv = p.arxiv && /^\d{4}\.\d{4}/.test(p.arxiv);
    if (hasArxiv) {
      links.push(`<a href="https://arxiv.org/abs/${esc(p.arxiv)}" target="_blank" rel="noreferrer">arXiv:${esc(p.arxiv)}</a>`);
    } else if (p.pdf_url) {
      links.push(`<a href="${esc(p.pdf_url)}" target="_blank" rel="noreferrer">PDF</a>`);
    }
    const gh = p.github_url;
    if (gh && gh.startsWith("http") && !gh.includes("thecvf.com") && gh !== p.pdf_url) {
      links.push(`<a href="${esc(gh)}" target="_blank" rel="noreferrer">GitHub</a>`);
    }
    const autosotaRepo = p.autosota_repo_url;
    if (autosotaRepo && autosotaRepo.startsWith("http")) {
      links.unshift(`<a class="autosota-code-link" href="${esc(autosotaRepo)}" target="_blank" rel="noreferrer" aria-label="Open AutoSOTA optimized code repository"><span class="autosota-code-kicker">AutoSOTA</span><span>Optimized Code Repo</span><span class="autosota-code-arrow" aria-hidden="true">↗</span></a>`);
    }

    const impDisplay = formatImprovement(impPct);
    const catCls = CAT_COLORS[cat] || "";
    const catTag = catCls ? `<span class="cat-badge ${catCls}">${esc(cat)}</span>` : "";

    return `<article class="cvpr-row${hasImprovement ? " has-improvement" : ""}" data-id="${esc(p.paper_id)}">
      <div class="cvpr-row-main">
        <span class="cvpr-row-title">${esc(p.title)}</span>
        <div class="cvpr-row-links">${links.join(" · ") || '<span style="color:var(--muted)">No links</span>'}</div>
      </div>
      <div class="cvpr-row-status">
        <span class="status-badge status-${statusClass}">${esc(p.status)}</span>
      </div>
      <div class="cvpr-row-metric">
        ${impDisplay ? `<span class="metric-delta positive">↑ ${esc(impDisplay)}</span>` : ""}
      </div>
      <div class="cvpr-row-cat">
        ${catTag}
      </div>
      <div class="cvpr-row-enchant">
        ${hasImprovement ? `<span class="enchant-label">AutoSOTA Enhancement</span>${optNote ? `<span class="enchant-text">${esc(optNote)}</span>` : ""}` : ""}
      </div>
      <div class="cvpr-row-actions">
        <a class="issue-link" href="${issueURL}" target="_blank" rel="noreferrer">Feedback</a>
      </div>
    </article>`;
  }).join("");
}

// --- Init ---
function getStaticDay(dayKey) {
  const data = window.CVPR_DATA && window.CVPR_DATA[dayKey];
  return Array.isArray(data) && data.length ? data : null;
}

async function loadDay(dayKey) {
  const staticData = getStaticDay(dayKey);
  if (staticData) {
    state.papers[dayKey] = staticData;
    return;
  }

  try {
    const resp = await fetch(`site-data/cvpr_${dayKey}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    state.papers[dayKey] = await resp.json();
  } catch (err) {
    console.error(`Failed to load ${dayKey}:`, err);
    state.papers[dayKey] = [];
  }
}

async function init() {
  // Load the default day first and render immediately.
  await loadDay(state.activeDay);
  if (!state.papers[state.activeDay].length) {
    heroHeadline.textContent = "Could not load CVPR data.";
    runsList.innerHTML = '<article class="run-card run-card-empty"><p>Could not load CVPR data. Please try again later.</p></article>';
    return;
  }

  renderStats();
  renderDayTabs();
  renderRuns();

  // Load the other days in background and refresh tabs when ready.
  Promise.all(["day1", "day2", "day3"].filter(day => day !== state.activeDay).map(loadDay)).then(() => {
    renderDayTabs();
  });
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

allCatToggle.addEventListener("click", () => {
  state.showAllCategories = !state.showAllCategories;
  state.page = 1;
  allCatToggle.classList.toggle("is-active", state.showAllCategories);
  renderRuns();
});

document.addEventListener("click", (e) => {
  if (e.target.id === "page-prev" && state.page > 1) { state.page--; renderRuns(); }
  if (e.target.id === "page-next") { state.page++; renderRuns(); }
});

init();
