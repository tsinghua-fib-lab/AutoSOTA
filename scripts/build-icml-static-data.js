#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const defaultCsvPath = path.join(root, "site-data", "icml_papers.csv");
const csvPath = process.argv[2] ? path.resolve(process.argv[2]) : defaultCsvPath;

const outputJsonPaths = [
  path.join(root, "site-data", "icml_papers.json"),
  path.join(root, "docs", "site-data", "icml_papers.json"),
];
const outputJsPaths = [
  path.join(root, "icml-data.js"),
  path.join(root, "docs", "icml-data.js"),
];

const fieldAliases = {
  paper_id: ["paper_id", "paper id", "paperID", "id"],
  paper_title: ["paper_title", "paper title", "title"],
  status: ["status", "presentation_status", "presentation status", "type"],
  workspace: ["workspace"],
  started_at_beijing: ["started_at_beijing", "started at beijing", "started_at", "started at"],
  finished_at_beijing: ["finished_at_beijing", "finished at beijing", "finished_at", "finished at"],
  pipeline_status: ["pipeline_status", "pipeline status"],
  reproduction_success: ["reproduction_success", "reproduction success"],
  sota_success: ["sota_success", "sota success"],
  evaluation_success: ["evaluation_success", "evaluation success"],
  artifacts_success: ["artifacts_success", "artifacts success"],
  reproduction_status: ["reproduction_status", "reproduction status"],
  ideas_status: ["ideas_status", "ideas status"],
  sota_status: ["sota_status", "sota status"],
  evaluation_status: ["evaluation_status", "evaluation status"],
  artifacts_status: ["artifacts_status", "artifacts status"],
  failed_stage: ["failed_stage", "failed stage"],
  failure_reason: ["failure_reason", "failure reason", "failure category", "failure_reason_category"],
  failure_reason_source: ["failure_reason_source", "failure reason source"],
  enhancement: ["best_iter_optimization_summary", "summary", "enhancement", "optimization summary"],
  repo_url: ["repo_url", "repo url", "github_url", "github url", "code url"],
  pdf_url: ["pdf_url", "pdf url", "paper url", "openreview url"],
  autosota_repo: ["autosota_repo", "autosota repo", "auto sota repo", "autosota_repo_url", "autosota repo url"],
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];

    if (ch === '"') {
      if (inQuotes && next === '"') {
        field += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === "," && !inQuotes) {
      row.push(field);
      field = "";
      continue;
    }

    if ((ch === "\n" || ch === "\r") && !inQuotes) {
      if (ch === "\r" && next === "\n") i += 1;
      row.push(field);
      if (row.some((cell) => cell !== "")) rows.push(row);
      row = [];
      field = "";
      continue;
    }

    field += ch;
  }

  if (field || row.length) {
    row.push(field);
    if (row.some((cell) => cell !== "")) rows.push(row);
  }

  if (!rows.length) return [];
  const headers = rows[0].map((h) => h.replace(/^\uFEFF/, "").trim());
  return rows.slice(1).map((cells) => {
    const obj = {};
    headers.forEach((header, idx) => {
      obj[header] = (cells[idx] || "").trim();
    });
    return obj;
  });
}

function cleanUrl(value) {
  const text = String(value || "").trim();
  return /^https?:\/\//i.test(text) ? text : "";
}

function normalizeKey(key) {
  return String(key || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
}

function getValue(row, canonicalKey) {
  const aliases = fieldAliases[canonicalKey] || [canonicalKey];
  const normalizedLookup = new Map(
    Object.keys(row).map((key) => [normalizeKey(key), row[key]])
  );

  for (const alias of aliases) {
    if (Object.prototype.hasOwnProperty.call(row, alias)) return row[alias];
    const normalizedAlias = normalizeKey(alias);
    if (normalizedLookup.has(normalizedAlias)) return normalizedLookup.get(normalizedAlias);
  }
  return "";
}

function normalizePaper(row, index) {
  return {
    paper_id: getValue(row, "paper_id") || String(index + 1).padStart(4, "0"),
    seq: index + 1,
    title: getValue(row, "paper_title") || "",
    presentation_status: getValue(row, "status") || "Oral",
    workspace: getValue(row, "workspace"),
    started_at_beijing: getValue(row, "started_at_beijing"),
    finished_at_beijing: getValue(row, "finished_at_beijing"),
    pipeline_status: getValue(row, "pipeline_status"),
    reproduction_success: getValue(row, "reproduction_success"),
    sota_success: getValue(row, "sota_success"),
    evaluation_success: getValue(row, "evaluation_success"),
    artifacts_success: getValue(row, "artifacts_success"),
    reproduction_status: getValue(row, "reproduction_status"),
    ideas_status: getValue(row, "ideas_status"),
    sota_status: getValue(row, "sota_status"),
    evaluation_status: getValue(row, "evaluation_status"),
    artifacts_status: getValue(row, "artifacts_status"),
    failed_stage: getValue(row, "failed_stage"),
    failure_reason: getValue(row, "failure_reason"),
    failure_reason_source: getValue(row, "failure_reason_source"),
    enhancement: getValue(row, "enhancement"),
    repo_url: cleanUrl(getValue(row, "repo_url")),
    pdf_url: cleanUrl(getValue(row, "pdf_url")),
    autosota_repo_url: cleanUrl(getValue(row, "autosota_repo")),
  };
}

if (!fs.existsSync(csvPath)) {
  throw new Error(`CSV not found: ${csvPath}`);
}

const csvText = fs.readFileSync(csvPath, "utf8");
const papers = parseCsv(csvText).map(normalizePaper);
const payload = { conference: "ICML 2026", papers };
const jsonText = `${JSON.stringify(papers, null, 2)}\n`;
const jsText = [
  "// Generated by scripts/build-icml-static-data.js.",
  "// Do not edit by hand; update site-data/icml_papers.csv or pass a CSV path and rerun this script.",
  `window.ICML_DATA = ${JSON.stringify(payload)};`,
  "",
].join("\n");

for (const outputPath of outputJsonPaths) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, jsonText);
  console.log(`Wrote ${path.relative(root, outputPath)}`);
}

for (const outputPath of outputJsPaths) {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, jsText);
  console.log(`Wrote ${path.relative(root, outputPath)}`);
}
