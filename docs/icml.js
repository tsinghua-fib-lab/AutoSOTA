// ICML 2026 Monitor
const GITHUB_REPO = "https://github.com/tsinghua-fib-lab/AutoSOTA";

const state = {
  papers: [],
  activeQuery: "",
  activeStatusFilter: "all",
  sortBy: "status-priority",
  page: 1,
  pageSize: 30,
};

const $ = (sel) => document.querySelector(sel);
const heroHeadline = $("#hero-headline");
const heroSummary = $("#hero-summary");
const heroFacts = $("#hero-facts");
const statsGrid = $("#stats-grid");
const runsSummary = $("#runs-summary");
const runsList = $("#runs-list");
const searchInput = $("#search-input");
const sortSelect = $("#sort-select");
const statusFilters = $("#status-filters");

const STATUS_META = {
  not_started: { label: "Not Started", className: "status-not-started", rank: 0 },
  research: { label: "Researching", className: "status-research", rank: 1 },
  success: { label: "Success", className: "status-success", rank: 2 },
  failed: { label: "Failed", className: "status-failed", rank: 3 },
};

const STATUS_SORT_PRIORITY = {
  success: 0,
  failed: 1,
  research: 2,
  not_started: 3,
};

const STAGES = [
  { key: "reproduction", label: "Reproduction", statusField: "reproduction_status", successField: "reproduction_success" },
  { key: "ideas", label: "Ideas", statusField: "ideas_status" },
  { key: "sota", label: "SOTA", statusField: "sota_status", successField: "sota_success" },
  { key: "evaluation", label: "Evaluation", statusField: "evaluation_status", successField: "evaluation_success" },
  { key: "artifacts", label: "Artifacts", statusField: "artifacts_status", successField: "artifacts_success" },
];

function esc(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function normalize(value) {
  return String(value || "").trim().toLowerCase();
}

function hasValue(value) {
  return normalize(value).length > 0;
}

function isExplicitFalse(value) {
  return /^(0|false|no|n|failed|fail|failure)$/i.test(String(value || "").trim());
}

function isSuccessValue(value) {
  const text = normalize(value);
  if (!text) return false;
  return /^(1|true|yes|y|success|succeeded|done|pass|passed|complete|completed|finished|ok|optimized|reproduced)$/i.test(text) ||
    /\b(success|succeeded|passed|completed|finished|optimized|reproduced)\b/i.test(text);
}

function isFailureValue(value) {
  const text = normalize(value);
  if (!text) return false;
  return isExplicitFalse(text) ||
    /\b(fail|failed|failure|error|exception|timeout|crash|oom|blocked|invalid|missed)\b/i.test(text) ||
    /no[_\s-]?improvement|not[_\s-]?reproduc|insufficient|missing[_\s-]?(repo|data|resource)/i.test(text);
}

function stageState(paper, stage) {
  const statusValue = paper[stage.statusField];
  const successValue = stage.successField ? paper[stage.successField] : "";

  if (isExplicitFalse(successValue) || isFailureValue(statusValue)) return "failed";
  if (isSuccessValue(successValue) || isSuccessValue(statusValue)) return "success";
  if (hasValue(successValue) || hasValue(statusValue)) return "research";
  return "pending";
}

function getStages(paper) {
  return STAGES.map((stage, index) => {
    const stateName = stageState(paper, stage);
    return {
      ...stage,
      index: index + 1,
      state: stateName,
      display: stateName === "pending" ? "Pending" : stateName === "research" ? "In progress" : stateName,
    };
  });
}

function categoryFromStageName(stageName) {
  const text = normalize(stageName);
  if (!text) return "";
  if (text.includes("repro")) return "Reproduction failed";
  if (text.includes("idea")) return "Idea generation failed";
  if (text.includes("sota") || text.includes("optim")) return "Optimization failed";
  if (text.includes("eval")) return "Evaluation failed";
  if (text.includes("artifact") || text.includes("repo")) return "Artifact packaging failed";
  return `${stageName} failed`;
}

function getFailureCategory(paper, stages) {
  if (hasValue(paper.failed_stage)) return categoryFromStageName(paper.failed_stage);

  const failedStage = stages.find((stage) => stage.state === "failed");
  if (failedStage) return categoryFromStageName(failedStage.label);
  if (isFailureValue(paper.pipeline_status)) return "Pipeline failed";
  return "Pipeline incomplete";
}

function derivePaperStatus(paper) {
  const stages = getStages(paper);
  const hasFailure = hasValue(paper.failed_stage) ||
    hasValue(paper.failure_reason) ||
    hasValue(paper.failure_reason_source) ||
    isFailureValue(paper.pipeline_status) ||
    stages.some((stage) => stage.state === "failed");

  const sotaSuccess = stages.some((stage) => stage.key === "sota" && stage.state === "success");
  const pipelineSuccess = isSuccessValue(paper.pipeline_status);
  const repoSuccess = hasValue(paper.autosota_repo_url) && !hasFailure;

  if (hasFailure) {
    return {
      key: "failed",
      label: STATUS_META.failed.label,
      className: STATUS_META.failed.className,
      failureCategory: getFailureCategory(paper, stages),
      stages,
    };
  }

  if (sotaSuccess || pipelineSuccess || repoSuccess) {
    return {
      key: "success",
      label: STATUS_META.success.label,
      className: STATUS_META.success.className,
      failureCategory: "",
      stages,
    };
  }

  const hasActivity = hasValue(paper.started_at_beijing) ||
    hasValue(paper.finished_at_beijing) ||
    hasValue(paper.pipeline_status) ||
    stages.some((stage) => stage.state !== "pending");
  const key = hasActivity ? "research" : "not_started";

  return {
    key,
    label: STATUS_META[key].label,
    className: STATUS_META[key].className,
    failureCategory: "",
    stages,
  };
}

function annotatePapers() {
  return state.papers.map((paper) => ({
    ...paper,
    derived: derivePaperStatus(paper),
  }));
}

function getCounts(papers) {
  return papers.reduce((acc, paper) => {
    acc[paper.derived.key] = (acc[paper.derived.key] || 0) + 1;
    return acc;
  }, { not_started: 0, research: 0, success: 0, failed: 0 });
}

function pct(count, total, digits = 1) {
  if (!total) return "0.0";
  return ((count / total) * 100).toFixed(digits);
}

function displayPct(count, total) {
  if (!total) return "0%";
  const value = (count / total) * 100;
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)}%`;
}

function renderStats() {
  const papers = annotatePapers();
  const total = papers.length;
  const counts = getCounts(papers);
  const completed = counts.success + counts.failed;
  const progressPct = displayPct(completed, total);
  const remaining = Math.max(total - completed, 0);

  heroHeadline.textContent = "ICML 2026";
  heroSummary.innerHTML = `AutoSOTA explores the potential of latest research as the conference unfold. <a class="icml-signup-link" href="https://docs.google.com/forms/d/e/1FAIpQLSdGKf1W2McOrmW9v0326f_mJAn0VhPVZNgF-W-BqlAsPjGiZA/viewform" target="_blank" rel="noreferrer">Sign up</a> your work for AutoSOTA.`;
  heroFacts.innerHTML = `
    <div class="icml-fact">
      <span>Tracked Papers</span>
      <strong>${total}</strong>
    </div>
    <div class="icml-fact">
      <span>Completed</span>
      <strong>${completed}</strong>
    </div>
    <div class="icml-fact">
      <span>Current Phase</span>
      <strong>${counts.research ? "Running" : completed ? "Updated" : "Ready"}</strong>
    </div>
  `;

  const segment = (key) => `style="width:${pct(counts[key], total, 4)}%"`;
  const statusItem = (key) => `<div class="icml-status-item status-item-${key}">
    <span class="icml-status-name"><i class="legend-dot dot-${key.replace("_", "-")}"></i>${STATUS_META[key].label}</span>
    <span class="icml-status-metric"><strong>${counts[key]}</strong><em>${displayPct(counts[key], total)}</em></span>
  </div>`;

  statsGrid.innerHTML = `
    <article class="icml-progress-card">
      <div class="icml-progress-top">
        <div>
          <span class="countdown-label">Overall Progress</span>
          <span class="icml-progress-number">${esc(progressPct)}</span>
        </div>
        <div class="icml-progress-side">
          <span class="icml-total-pill">${total} papers</span>
          <span>${completed} done · ${remaining} left</span>
        </div>
      </div>
      <div class="icml-progress-track" aria-label="ICML overall status distribution">
        <span class="icml-progress-segment segment-success" ${segment("success")}></span>
        <span class="icml-progress-segment segment-failed" ${segment("failed")}></span>
        <span class="icml-progress-segment segment-research" ${segment("research")}></span>
        <span class="icml-progress-segment segment-not-started" ${segment("not_started")}></span>
      </div>
      <div class="icml-status-grid">
        ${statusItem("success")}
        ${statusItem("failed")}
        ${statusItem("research")}
        ${statusItem("not_started")}
      </div>
    </article>
  `;
}

function presentationClass(value) {
  const text = normalize(value || "oral");
  if (text.includes("highlight")) return "type-highlight";
  if (text.includes("poster")) return "type-poster";
  return "type-oral";
}

function presentationLabel(value) {
  return String(value || "Oral").trim() || "Oral";
}

function formatMetric(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  const number = Number(text);
  if (!Number.isFinite(number)) return text;
  const abs = Math.abs(number);
  if (abs >= 1000) return number.toLocaleString(undefined, { maximumFractionDigits: 1 });
  if (abs >= 100) return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (abs >= 10) return number.toLocaleString(undefined, { maximumFractionDigits: 3 });
  if (abs >= 1) return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return number.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

function formatImprovement(value) {
  const text = String(value || "").trim();
  if (!text) return "";
  if (text.includes("%")) return text.startsWith("+") ? text : `+${text}`;
  const number = Number(text);
  if (!Number.isFinite(number)) return text;
  return `+${(number * 100).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;
}

function metricPanel(paper) {
  const improvement = formatImprovement(paper.improvement);
  const originalMetric = formatMetric(paper.original_metric);
  const optimizedMetric = formatMetric(paper.optimized_metric);
  const category = String(paper.sota_category || "").trim();

  if (!improvement && !originalMetric && !optimizedMetric && !category) {
    return '<span class="muted-dash">--</span>';
  }

  return `<div class="icml-metric-panel">
    ${category ? `<span class="icml-category-pill">Class ${esc(category)}</span>` : ""}
    ${improvement ? `<strong>${esc(improvement)}</strong>` : ""}
    ${originalMetric || optimizedMetric ? `<span>${esc(originalMetric || "--")} &rarr; ${esc(optimizedMetric || "--")}</span>` : ""}
  </div>`;
}

function statusBadge(derived) {
  const tooltip = derived.key === "failed" ? ` data-tooltip="${esc(derived.failureCategory)}" title="${esc(derived.failureCategory)}"` : "";
  return `<span class="status-badge ${derived.className}"${tooltip}>${esc(derived.label)}</span>`;
}

function stageBadges(stages) {
  return `<div class="icml-stage-badges">${stages.map((stage) => {
    const tooltip = `${stage.index}. ${stage.label}: ${stage.display}`;
    return `<span class="stage-badge stage-${stage.state}" data-tooltip="${esc(tooltip)}" title="${esc(tooltip)}">${stage.index}</span>`;
  }).join("")}</div>`;
}

function stageBlock(derived) {
  return stageBadges(derived.stages);
}

function comparePaperId(a, b) {
  const aId = parseInt(a.paper_id, 10);
  const bId = parseInt(b.paper_id, 10);
  if (!Number.isNaN(aId) && !Number.isNaN(bId) && aId !== bId) return aId - bId;
  return String(a.paper_id || "").localeCompare(String(b.paper_id || ""));
}

function presentationSortPriority(paper) {
  const text = normalize(paper.presentation_status);
  if (text.includes("oral")) return 0;
  if (text.includes("poster")) return 1;
  return 2;
}

function getVisiblePapers() {
  let papers = annotatePapers();
  const q = state.activeQuery.trim().toLowerCase();

  if (state.activeStatusFilter !== "all") {
    papers = papers.filter((paper) => paper.derived.key === state.activeStatusFilter);
  }

  if (q) {
    papers = papers.filter((paper) =>
      String(paper.seq || "").includes(q) ||
      String(paper.paper_id || "").toLowerCase().includes(q) ||
      String(paper.title || "").toLowerCase().includes(q) ||
      String(paper.enhancement || "").toLowerCase().includes(q) ||
      String(paper.repo_url || "").toLowerCase().includes(q) ||
      String(paper.pdf_url || "").toLowerCase().includes(q)
    );
  }

  papers = [...papers].sort((a, b) => {
    if (state.sortBy === "title-asc") {
      return String(a.title || "").localeCompare(String(b.title || ""));
    }
    if (state.sortBy === "status-priority") {
      const diff = STATUS_SORT_PRIORITY[a.derived.key] - STATUS_SORT_PRIORITY[b.derived.key];
      if (diff) return diff;
      const typeDiff = presentationSortPriority(a) - presentationSortPriority(b);
      if (typeDiff) return typeDiff;
      return comparePaperId(a, b);
    }
    if (state.sortBy === "status-asc") {
      const diff = STATUS_META[a.derived.key].rank - STATUS_META[b.derived.key].rank;
      if (diff) return diff;
    }
    return comparePaperId(a, b);
  });

  return papers;
}

function renderRuns() {
  const allVisible = getVisiblePapers();
  const total = state.papers.length;
  const totalPages = Math.ceil(allVisible.length / state.pageSize) || 1;
  if (state.page > totalPages) state.page = totalPages;
  const start = (state.page - 1) * state.pageSize;
  const visible = allVisible.slice(start, start + state.pageSize);

  const filterLabel = state.activeStatusFilter === "all" ? "" : ` · ${STATUS_META[state.activeStatusFilter].label}`;
  runsSummary.innerHTML = `<span>${allVisible.length} of ${total} papers${filterLabel}</span>
    <span class="page-nav">
      <button class="page-btn" id="page-prev" ${state.page <= 1 ? "disabled" : ""}>Prev</button>
      <span class="page-info">${state.page} / ${totalPages}</span>
      <button class="page-btn" id="page-next" ${state.page >= totalPages ? "disabled" : ""}>Next</button>
    </span>`;

  if (!visible.length) {
    runsList.innerHTML = '<article class="run-card run-card-empty"><p>No papers match the current filters.</p></article>';
    return;
  }

  runsList.innerHTML = visible.map((paper) => {
    const issueTitle = encodeURIComponent(`[ICML 2026] ${paper.paper_id}: ${(paper.title || "").slice(0, 80)}`);
    const issueURL = `${GITHUB_REPO}/issues/new?labels=icml2026&title=${issueTitle}`;
    const links = [];

    if (paper.autosota_repo_url) {
      links.push(`<a class="autosota-code-link" href="${esc(paper.autosota_repo_url)}" target="_blank" rel="noreferrer" aria-label="Open AutoSOTA optimized code repository"><span class="autosota-code-kicker">AutoSOTA</span><span>Optimized Code Repo</span><span class="autosota-code-arrow" aria-hidden="true">↗</span></a>`);
    }
    if (paper.pdf_url) {
      links.push(`<a href="${esc(paper.pdf_url)}" target="_blank" rel="noreferrer">PDF</a>`);
    }
    if (paper.repo_url) {
      links.push(`<a href="${esc(paper.repo_url)}" target="_blank" rel="noreferrer">Paper Code</a>`);
    }

    const enhancement = paper.enhancement && paper.derived.key === "success" ? paper.enhancement : "";
    const typeLabel = presentationLabel(paper.presentation_status);
    const typeClass = presentationClass(typeLabel);

    return `<article class="cvpr-row icml-row status-row-${paper.derived.key}" data-id="${esc(paper.paper_id)}">
      <div class="cvpr-row-main">
        <div class="icml-paper-meta">
          <span class="paper-id-chip">Paper ID ${esc(paper.paper_id)}</span>
          <span>${esc(typeLabel)}</span>
        </div>
        <span class="cvpr-row-title">${esc(paper.title)}</span>
        <div class="cvpr-row-links">${links.join(" · ") || '<span style="color:var(--muted)">No links</span>'}</div>
      </div>
      <div class="icml-row-type">
        <span class="type-badge ${typeClass}">${esc(typeLabel)}</span>
      </div>
      <div class="cvpr-row-status">
        ${statusBadge(paper.derived)}
      </div>
      <div class="icml-row-stages" aria-label="Pipeline stages">
        ${stageBlock(paper.derived)}
      </div>
      <div class="icml-row-metrics">
        ${paper.derived.key === "success" ? metricPanel(paper) : '<span class="muted-dash">--</span>'}
      </div>
      <div class="cvpr-row-enchant">
        ${enhancement ? `<span class="enchant-label">Best Iteration Summary</span><span class="enchant-text">${esc(enhancement)}</span>` : '<span class="enchant-text muted-dash">--</span>'}
      </div>
      <div class="cvpr-row-actions">
        <a class="issue-link" href="${issueURL}" target="_blank" rel="noreferrer">Feedback</a>
      </div>
    </article>`;
  }).join("");
}

function getStaticPapers() {
  const data = window.ICML_DATA;
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data.papers)) return data.papers;
  return null;
}

async function loadPapers() {
  const staticData = getStaticPapers();
  if (staticData) {
    state.papers = staticData;
    return;
  }

  try {
    const resp = await fetch("site-data/icml_papers.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    state.papers = Array.isArray(data) ? data : data.papers || [];
  } catch (err) {
    console.error("Failed to load ICML papers:", err);
    state.papers = [];
  }
}

async function init() {
  await loadPapers();
  if (!state.papers.length) {
    heroHeadline.textContent = "Could not load ICML data.";
    runsList.innerHTML = '<article class="run-card run-card-empty"><p>Could not load ICML data. Please try again later.</p></article>';
    return;
  }

  renderStats();
  renderRuns();
}

let searchTimer = null;
searchInput.addEventListener("input", (event) => {
  state.activeQuery = event.target.value;
  state.page = 1;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => renderRuns(), 180);
});

sortSelect.addEventListener("change", (event) => {
  state.sortBy = event.target.value;
  state.page = 1;
  renderRuns();
});

statusFilters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-status-filter]");
  if (!button) return;
  state.activeStatusFilter = button.dataset.statusFilter;
  state.page = 1;
  statusFilters.querySelectorAll("[data-status-filter]").forEach((item) => {
    item.classList.toggle("is-active", item === button);
  });
  renderRuns();
});

document.addEventListener("click", (event) => {
  if (event.target.id === "page-prev" && state.page > 1) {
    state.page -= 1;
    renderRuns();
  }
  if (event.target.id === "page-next") {
    state.page += 1;
    renderRuns();
  }
});

init();
