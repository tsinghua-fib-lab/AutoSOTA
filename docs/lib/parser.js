(function() {
  const GITHUB_REPO = "https://github.com/tsinghua-fib-lab/AutoSOTA";
  const GITHUB_BLOB = `${GITHUB_REPO}/blob/main/`;

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

  function parseLeaderboard(md) {
    const lines = md.split(/\r?\n/);
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
      const folder = relPath ? relPath.replace(/^\.\//, "").replace(/\/[^/]+$/, "") : "";
      map.set(id, { method, title, summary: paragraphs[0] || "", paragraphs, relPath,
        blobUrl: folder ? `${GITHUB_BLOB}${folder}/README.md` : GITHUB_REPO,
        treeUrl: folder ? `${GITHUB_REPO}/tree/main/${folder}` : GITHUB_REPO,
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
        searchText: [e.id, method, ftitle, summary, e.improvementDisplay].join(" ").toLowerCase(),
      };
    });
  }

  const exports = { cleanMD, esc, parsePct, formatPct, gradeInfo, metricColor, parseLeaderboard, parseSummaries, mergeData, GITHUB_REPO, GITHUB_BLOB };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = exports;
  }
  if (typeof window !== "undefined") {
    Object.assign(window, exports);
  }
})();
