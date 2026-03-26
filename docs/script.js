const GITHUB_REPO_URL = "https://github.com/Ericccc02/AutoSota_list";
const GITHUB_BLOB_BASE = `${GITHUB_REPO_URL}/blob/main/`;
const GITHUB_TREE_BASE = `${GITHUB_REPO_URL}/tree/main/`;

const state = {
  activeFilter: "all",
  activeQuery: "",
  items: [],
};

const heroHeadline = document.getElementById("hero-headline");
const heroSummary = document.getElementById("hero-summary");
const statsGrid = document.getElementById("stats-grid");
const featuredGrid = document.getElementById("featured-grid");
const runsSummary = document.getElementById("runs-summary");
const runsList = document.getElementById("runs-list");
const searchInput = document.getElementById("search-input");
const filterGroup = document.getElementById("filter-group");
const yearNode = document.getElementById("year");

if (yearNode) {
  yearNode.textContent = String(new Date().getFullYear());
}

function cleanInlineMarkdown(text) {
  return text
    .replace(/<[^>]+>/g, " ")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_`]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function parseImprovementValue(text) {
  const normalized = cleanInlineMarkdown(text)
    .replace(/↓/g, "-")
    .replace(/↑/g, "+")
    .replace(/−/g, "-");
  const match = normalized.match(/([+-]?\d+(?:\.\d+)?)\s*%/);
  return match ? Number(match[1]) : 0;
}

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }

  if (value >= 10) {
    return `${value.toFixed(1)}%`;
  }

  if (value >= 1) {
    return `${value.toFixed(2)}%`;
  }

  return `${value.toFixed(2)}%`;
}

function extractSnapshot(text) {
  const cleaned = cleanInlineMarkdown(text);
  if (!cleaned) {
    return "Summary available in detail view.";
  }

  const [firstSentence] = cleaned.split(/(?<=[.!?])\s+/);
  const snapshot = firstSentence || cleaned;
  return snapshot.length > 120 ? `${snapshot.slice(0, 117)}...` : snapshot;
}

function parseMetricDisplay(metricText) {
  const cleaned = cleanInlineMarkdown(metricText);

  let match = cleaned.match(/^([+-]?\d+(?:\.\d+)?%)\((.+?)->(.+?)\)(?:\s+(.*))?$/);
  if (match) {
    return {
      primary: `${match[2].trim()} → ${match[3].trim()}`,
      note: match[4] ? match[4].trim() : "",
      prefix: "",
    };
  }

  match = cleaned.match(/^([A-Za-z][A-Za-z0-9_/%.\s-]*?)\s+(.+?)\s*->\s*(.+?)\s*\(([^)]+)\)\s*$/);
  if (match) {
    return {
      primary: `${match[2].trim()} → ${match[3].trim()}`,
      note: match[4].trim(),
      prefix: match[1].trim(),
    };
  }

  return {
    label: "Primary Metric",
    primary: cleaned,
    note: "",
    prefix: "",
  };
}

function buildMetricDisplay(entry, summary) {
  if (entry.metric) {
    return {
      label: "Primary Metric",
      ...parseMetricDisplay(entry.metric),
    };
  }

  return {
    label: "Snapshot",
    primary: extractSnapshot(summary.summary || summary.detailParagraphs?.[0] || entry.title || ""),
    note: "",
    prefix: "",
  };
}

function classifyTrend(trend) {
  if (trend.includes("🚀")) {
    return "strong";
  }
  if (trend.includes("📉")) {
    return "lower";
  }
  return "higher";
}

function buildGitHubBlobUrl(relativePath) {
  if (!relativePath) {
    return GITHUB_REPO_URL;
  }
  const normalized = relativePath.replace(/^\.\//, "");
  return `${GITHUB_BLOB_BASE}${normalized}`;
}

function buildGitHubTreeUrl(relativePath) {
  if (!relativePath) {
    return GITHUB_REPO_URL;
  }
  const normalized = relativePath.replace(/^\.\//, "");
  const folderPath = normalized.includes("/")
    ? normalized.slice(0, normalized.lastIndexOf("/"))
    : normalized;
  return `${GITHUB_TREE_BASE}${folderPath}`;
}

function parseLeaderboard(markdown) {
  const lines = markdown.split(/\r?\n/);
  const legacyHeaderIndex = lines.findIndex((line) =>
    line.includes("| ID | Method | Primary Metric | Improvement | Trend |")
  );
  const simplifiedHeaderIndex = lines.findIndex((line) =>
    line.includes("| ID | Paper Title | Ours\\_Optimization |")
  );

  if (legacyHeaderIndex !== -1) {
    const entries = [];
    for (let index = legacyHeaderIndex + 2; index < lines.length; index += 1) {
      const line = lines[index].trim();
      if (!line.startsWith("|")) {
        break;
      }

      const cells = line
        .split("|")
        .slice(1, -1)
        .map((cell) => cell.trim());

      if (cells.length < 5) {
        continue;
      }

      const [idCell, methodCell, metricCell, improvementCell, trendCell] = cells;
      const id = Number(cleanInlineMarkdown(idCell).replace(/\D/g, ""));
      const method = cleanInlineMarkdown(methodCell);
      const metric = cleanInlineMarkdown(metricCell);
      const improvement = cleanInlineMarkdown(improvementCell);
      const trend = cleanInlineMarkdown(trendCell);
      const improvementValue = parseImprovementValue(improvementCell);
      const trendType = classifyTrend(trendCell);

      entries.push({
        id,
        method,
        metric,
        title: "",
        improvement,
        improvementValue,
        improvementMagnitude: Math.abs(improvementValue),
        trend,
        trendType,
        directionKnown: true,
        isStrong: trendType === "strong",
        isLower: trendType === "lower",
      });
    }

    return entries;
  }

  if (simplifiedHeaderIndex === -1) {
    return [];
  }

  const entries = [];
  for (let index = simplifiedHeaderIndex + 2; index < lines.length; index += 1) {
    const line = lines[index].trim();
    if (!line.startsWith("|")) {
      break;
    }

    const cells = line
      .split("|")
      .slice(1, -1)
      .map((cell) => cell.trim());

    if (cells.length < 3) {
      continue;
    }

    const [idCell, titleCell, improvementCell] = cells;
    const id = Number(cleanInlineMarkdown(idCell).replace(/\D/g, ""));
    const title = cleanInlineMarkdown(titleCell);
    const improvement = cleanInlineMarkdown(improvementCell);
    const improvementValue = parseImprovementValue(improvementCell);

    entries.push({
      id,
      method: "",
      metric: "",
      title,
      improvement,
      improvementValue,
      improvementMagnitude: Math.abs(improvementValue),
      trend: "",
      trendType: improvementValue >= 10 ? "strong" : "improved",
      directionKnown: false,
      isStrong: improvementValue >= 10,
      isLower: false,
    });
  }

  return entries;
}

function parseSummaries(markdown) {
  const summarySection = markdown.split("## Per-paper optimization summaries")[1] || "";
  const blocks = summarySection
    .split(/\n(?=###\s+\d+\s+[—-]\s+)/)
    .map((block) => block.trim())
    .filter((block) => block.startsWith("### "));

  const summaryMap = new Map();

  for (const block of blocks) {
    const chunks = block
      .split(/\n\s*\n/)
      .map((chunk) => chunk.trim())
      .filter(Boolean)
      .filter((chunk) => chunk !== "---");

    if (!chunks.length) {
      continue;
    }

    const headerMatch = chunks[0].match(/^###\s+(\d+)\s+[—-]\s+(.+)$/);
    if (!headerMatch) {
      continue;
    }

    const id = Number(headerMatch[1]);
    const method = cleanInlineMarkdown(headerMatch[2]);
    const title = chunks[1] ? cleanInlineMarkdown(chunks[1]) : "";
    const linkChunk = chunks.find((chunk) => chunk.includes("](./"));
    const linkMatch = linkChunk
      ? linkChunk.match(/\((\.\/[^)]+)\)/)
      : null;
    const relativePath = linkMatch ? linkMatch[1] : "";

    const paragraphs = chunks
      .slice(2)
      .filter((chunk) => !chunk.startsWith("**["))
      .map((chunk) => cleanInlineMarkdown(chunk))
      .filter(Boolean);

    summaryMap.set(id, {
      method,
      title,
      summary: paragraphs[0] || "",
      detailParagraphs: paragraphs,
      relativePath,
      githubBlobUrl: buildGitHubBlobUrl(relativePath),
      githubTreeUrl: buildGitHubTreeUrl(relativePath),
    });
  }

  return summaryMap;
}

function mergeData(leaderboard, summaries) {
  return leaderboard.map((entry) => {
    const summary = summaries.get(entry.id) || {};
    const title = summary.title || entry.title || entry.method;
    const method = summary.method || entry.method || title;
    return {
      ...entry,
      method,
      title,
      summary: summary.summary || "",
      detailParagraphs: summary.detailParagraphs || [],
      metricDisplay: buildMetricDisplay(entry, summary),
      githubBlobUrl: summary.githubBlobUrl || GITHUB_REPO_URL,
      githubTreeUrl: summary.githubTreeUrl || GITHUB_REPO_URL,
      searchText: [
        entry.id,
        method,
        entry.metric,
        entry.improvement,
        title,
        summary.summary || "",
      ]
        .join(" ")
        .toLowerCase(),
    };
  });
}

function formatStatCards(items) {
  const strongCount = items.filter((item) => item.isStrong).length;
  const validGains = items
    .map((item) => item.improvementMagnitude)
    .filter((v) => isFinite(v));
  const avgGain = validGains.length
    ? validGains.reduce((sum, v) => sum + v, 0) / validGains.length
    : NaN;
  const topGain =
    [...items].sort((left, right) => right.improvementMagnitude - left.improvementMagnitude)[0] ||
    items[0];

  const cards = [
    {
      value: String(items.length),
      label: "optimized papers",
      meta: "Parsed from the repository README snapshot",
    },
    {
      value: String(strongCount),
      label: "strong wins",
      meta: "Optimization gains at or above 10%",
    },
    {
      value: topGain ? topGain.improvement : "--",
      label: "best optimization gain",
      meta: topGain ? topGain.method : "No data",
    },
    {
      value: formatPercent(avgGain),
      label: "average gain",
      meta: "Average optimization gain across all papers",
    },
  ];

  statsGrid.innerHTML = cards
    .map(
      (card) => `
        <article class="stat-card">
          <span class="stat-value">${escapeHtml(card.value)}</span>
          <span class="stat-label">${escapeHtml(card.label)}</span>
          <span class="stat-meta">${escapeHtml(card.meta)}</span>
        </article>
      `
    )
    .join("");

  heroHeadline.textContent = `${items.length} optimized papers, one repository.`;
  heroSummary.textContent = `${strongCount} papers cleared a 10% gain threshold, and every entry below links back to its optimization note and source folder.`;
}

function statusClass(item) {
  if (item.isStrong) {
    return "status-strong";
  }
  if (item.directionKnown && item.isLower) {
    return "status-lower";
  }
  if (item.directionKnown) {
    return "status-higher";
  }
  return "status-improved";
}

function statusLabel(item) {
  if (item.isStrong) {
    return "Strong";
  }
  if (item.directionKnown && item.isLower) {
    return "Lower";
  }
  if (item.directionKnown) {
    return "Higher";
  }
  return "Improved";
}

function renderFeatured(items) {
  const featuredItems = [...items]
    .sort((left, right) => right.improvementMagnitude - left.improvementMagnitude)
    .slice(0, 6);

  if (!featuredItems.length) {
    featuredGrid.innerHTML = `
      <article class="featured-card">
        <p>No featured runs available.</p>
      </article>
    `;
    return;
  }

  featuredGrid.innerHTML = featuredItems
    .map(
      (item) => `
        <article class="featured-card">
          <div class="featured-card-header">
            <span class="featured-id">paper ${item.id}</span>
            <span class="status-pill ${statusClass(item)}">${statusLabel(item)}</span>
          </div>
          <h3>${escapeHtml(item.method)}</h3>
          <p class="featured-title">${escapeHtml(item.title)}</p>
          <p class="featured-summary">
            ${escapeHtml(item.summary || item.metricDisplay.primary || item.title)}
          </p>
          <p class="featured-metric">
            ${escapeHtml(item.metricDisplay.label)}: ${escapeHtml(item.metricDisplay.primary)}
          </p>
          <div class="featured-footer">
            <strong>${escapeHtml(item.improvement)}</strong>
            <a
              class="featured-link"
              href="${escapeHtml(item.githubBlobUrl)}"
              target="_blank"
              rel="noreferrer"
            >
              Open note
            </a>
          </div>
        </article>
      `
    )
    .join("");

  featuredGrid.querySelectorAll(".featured-card").forEach((card, index) => {
    const item = featuredItems[index];
    card.addEventListener("click", (event) => {
      const target = event.target;
      if (target instanceof Element && target.closest("a")) {
        return;
      }
      state.activeFilter = "all";
      state.activeQuery = "";
      searchInput.value = "";
      setActiveFilterButton("all");
      renderRuns();
      const targetCard = runsList.querySelector(`[data-id="${item.id}"]`);
      if (targetCard) {
        targetCard.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });
  });
}

function filterItems() {
  const query = state.activeQuery.trim().toLowerCase();

  return state.items.filter((item) => {
    const filterMatch =
      state.activeFilter === "all" ||
      (state.activeFilter === "strong" && item.isStrong);

    const queryMatch = !query || item.searchText.includes(query);
    return filterMatch && queryMatch;
  });
}

function renderRuns() {
  const visibleItems = filterItems();
  runsSummary.textContent = `Showing ${visibleItems.length} of ${state.items.length} optimized papers.`;

  if (!visibleItems.length) {
    runsList.innerHTML = `
      <article class="run-card run-card-empty">
        <p class="loading-row">No runs match the current filter.</p>
      </article>
    `;
    return;
  }

  runsList.innerHTML = visibleItems
    .map((item) => {
      const metricPrefix = item.metricDisplay.prefix
        ? `<span class="run-highlight-prefix">${escapeHtml(item.metricDisplay.prefix)}</span>`
        : "";
      const metricNote = item.metricDisplay.note
        ? `<span class="run-highlight-note">${escapeHtml(item.metricDisplay.note)}</span>`
        : "";
      const detailParagraphs = item.detailParagraphs.length
        ? item.detailParagraphs
            .map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`)
            .join("")
        : `<p>${escapeHtml(item.summary || "No summary available.")}</p>`;

      return `
        <article class="run-card" data-id="${item.id}">
          <div class="run-card-top">
            <div class="run-card-meta">
              <span class="run-id">paper ${String(item.id).padStart(3, "0")}</span>
              <span class="status-pill ${statusClass(item)}">${statusLabel(item)}</span>
            </div>
            <div class="run-actions">
              <a
                class="run-link"
                href="${escapeHtml(item.githubBlobUrl)}"
                target="_blank"
                rel="noreferrer"
              >
                Note
              </a>
              <a
                class="run-link"
                href="${escapeHtml(item.githubTreeUrl)}"
                target="_blank"
                rel="noreferrer"
              >
                Folder
              </a>
            </div>
          </div>
          <div class="run-headline">
            <div>
              <h3>${escapeHtml(item.method)}</h3>
              <p class="run-title">${escapeHtml(item.title)}</p>
            </div>
          </div>
          <div class="run-highlights">
            <article class="run-highlight">
              <span class="run-highlight-label">${escapeHtml(item.metricDisplay.label)}</span>
              <strong class="run-metric">${escapeHtml(item.metricDisplay.primary)}</strong>
              ${metricPrefix}
              ${metricNote}
            </article>
            <article class="run-highlight">
              <span class="run-highlight-label">Optimization Gain</span>
              <strong class="run-improvement">${escapeHtml(item.improvement)}</strong>
            </article>
          </div>
          <div class="run-detail">
            <div class="run-detail-copy">
              ${detailParagraphs}
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

async function loadReadme() {
  const response = await fetch("site-data/README.md", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load repository snapshot: ${response.status}`);
  }
  return response.text();
}

function setActiveFilterButton(nextFilter) {
  filterGroup.querySelectorAll(".filter-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.filter === nextFilter);
  });
}

async function init() {
  try {
    const markdown = await loadReadme();
    const leaderboard = parseLeaderboard(markdown);
    const summaries = parseSummaries(markdown);
    const items = mergeData(leaderboard, summaries).sort((left, right) => left.id - right.id);

    state.items = items;

    formatStatCards(items);
    renderFeatured(items);
    renderRuns();
  } catch (error) {
    heroHeadline.textContent = "Repository snapshot could not be loaded.";
    heroSummary.textContent = "The page could not parse repository results at the moment.";
    featuredGrid.innerHTML = `
      <article class="featured-card">
        <p>${escapeHtml(error.message)}</p>
      </article>
    `;
    runsSummary.textContent = "Failed to load repository snapshot.";
    runsList.innerHTML = `
      <article class="run-card run-card-empty">
        <p class="loading-row">${escapeHtml(error.message)}</p>
      </article>
    `;
  }
}

searchInput.addEventListener("input", (event) => {
  state.activeQuery = event.target.value;
  renderRuns();
});

filterGroup.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement)) {
    return;
  }
  const nextFilter = target.dataset.filter || "all";
  state.activeFilter = nextFilter;
  setActiveFilterButton(nextFilter);
  renderRuns();
});

init();
