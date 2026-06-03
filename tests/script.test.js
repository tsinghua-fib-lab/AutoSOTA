const { describe, it } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const path = require("path");

const {
  cleanMD, esc, parsePct, formatPct, gradeInfo, metricColor,
  parseLeaderboard, parseSummaries, mergeData, GITHUB_REPO, GITHUB_BLOB,
} = require("../docs/lib/parser.js");

// ============================================================
// Tests
// ============================================================

describe("cleanMD", () => {
  it("strips HTML tags", () => {
    assert.equal(cleanMD("<b>hello</b> world"), "hello world");
  });

  it("extracts link text from markdown links", () => {
    assert.equal(cleanMD("[click here](http://example.com)"), "click here");
  });

  it("removes emphasis markers", () => {
    assert.equal(cleanMD("**bold** and *italic* and `code`"), "bold and italic and code");
  });

  it("collapses whitespace", () => {
    assert.equal(cleanMD("a   b\n\nc"), "a b c");
  });
});

describe("esc", () => {
  it("escapes HTML special characters", () => {
    assert.equal(esc("<>&\""), "&lt;&gt;&amp;&quot;");
  });

  it("returns string unchanged when no special chars", () => {
    assert.equal(esc("Hello World"), "Hello World");
  });
});

describe("parsePct", () => {
  it("parses positive percentage", () => {
    assert.equal(parsePct("+63.64%"), 63.64);
  });

  it("parses negative percentage", () => {
    assert.equal(parsePct("↓14.5%"), -14.5);
  });

  it("parses percentage without sign", () => {
    assert.equal(parsePct("7.32%"), 7.32);
  });

  it("parses unicode minus sign", () => {
    assert.equal(parsePct("−5.0%"), -5);
  });

  it("returns 0 for non-matching text", () => {
    assert.equal(parsePct("no number here"), 0);
  });

  it("parses integer percentage", () => {
    assert.equal(parsePct("↑10%"), 10);
  });
});

describe("formatPct", () => {
  it("formats >=10 with one decimal", () => {
    assert.equal(formatPct(63.64), "63.6%");
  });

  it("formats <10 with two decimals", () => {
    assert.equal(formatPct(7.324), "7.32%");
  });

  it("returns -- for non-finite values", () => {
    assert.equal(formatPct(NaN), "--");
    assert.equal(formatPct(Infinity), "--");
  });
});

describe("gradeInfo", () => {
  it("classifies >=10 as strong", () => {
    assert.deepEqual(gradeInfo(14), { grade: "strong", label: "Strong" });
    assert.deepEqual(gradeInfo(-63.64), { grade: "strong", label: "Strong" });
  });

  it("classifies 3-10 as moderate", () => {
    assert.deepEqual(gradeInfo(5), { grade: "moderate", label: "Moderate" });
    assert.deepEqual(gradeInfo(-7.32), { grade: "moderate", label: "Moderate" });
  });

  it("classifies <3 as modest", () => {
    assert.deepEqual(gradeInfo(1.2), { grade: "modest", label: "Modest" });
    assert.deepEqual(gradeInfo(0), { grade: "modest", label: "Modest" });
  });
});

describe("metricColor", () => {
  it("returns amber for >=10", () => {
    assert.equal(metricColor(14), "amber");
  });
  it("returns green for 3-10", () => {
    assert.equal(metricColor(5), "green");
  });
  it("returns empty string for <3", () => {
    assert.equal(metricColor(1), "");
  });
});

describe("parseLeaderboard", () => {
  it("parses a valid leaderboard table", () => {
    const md = `| ID | Paper Title | Ours\\_Optimization |
|----|-------------|-------------------|
| 1  | [Paper A](http://a.com) | +63.64% |
| 2  | Paper B | ↓7.32% |`;
    const result = parseLeaderboard(md);
    assert.equal(result.length, 2);
    assert.equal(result[0].id, 1);
    assert.equal(result[0].title, "Paper A");
    assert.equal(result[0].improvement, 63.64);
    assert.equal(result[0].improvementDisplay, "+63.64%");
    assert.equal(result[0].grade, "strong");
    assert.equal(result[1].id, 2);
    assert.equal(result[1].improvement, -7.32);
    assert.equal(result[1].grade, "moderate");
  });

  it("returns empty array when header not found", () => {
    assert.deepEqual(parseLeaderboard("no table here"), []);
  });

  it("stops parsing at first non-table line", () => {
    const md = `| ID | Paper Title | Ours\\_Optimization |
|----|-------------|-------------------|
| 1  | Only | +10% |
Some other text
| 2  | Skipped | +5% |`;
    const result = parseLeaderboard(md);
    assert.equal(result.length, 1);
  });

  it("skips malformed rows with too few cells", () => {
    const md = `| ID | Paper Title | Ours\\_Optimization |
|----|-------------|-------------------|
| incomplete |
| 1  | Valid | +5% |`;
    const result = parseLeaderboard(md);
    assert.equal(result.length, 1);
  });
});

describe("parseSummaries", () => {
  it("parses summary blocks and builds correct blob URLs", () => {
    const md = `## Per-paper optimization summaries

### 1 — Method A

Paper Title Here

Summary paragraph one.

Summary paragraph two.

**[→ paper-1-FOO/OPTIMIZATION.md](./paper-1-FOO/OPTIMIZATION.md)**

---

### 2 — Method B

Another Paper

Summary text.

**[→ paper-2-BAR/OPTIMIZATION.md](./paper-2-BAR/OPTIMIZATION.md)**`;
    const map = parseSummaries(md);
    assert.equal(map.size, 2);

    const p1 = map.get(1);
    assert.equal(p1.method, "Method A");
    assert.equal(p1.title, "Paper Title Here");
    assert.equal(p1.summary, "Summary paragraph one.");
    assert.equal(p1.blobUrl, `${GITHUB_BLOB}paper-1-FOO/README.md`);
    assert.equal(p1.treeUrl, `${GITHUB_REPO}/tree/main/paper-1-FOO`);

    const p2 = map.get(2);
    assert.equal(p2.method, "Method B");
    assert.equal(p2.blobUrl, `${GITHUB_BLOB}paper-2-BAR/README.md`);
    assert.equal(p2.treeUrl, `${GITHUB_REPO}/tree/main/paper-2-BAR`);
  });

  it("falls back to GITHUB_REPO when no relPath", () => {
    const md = `## Per-paper optimization summaries

### 3 — No Link Method

No Link Paper

Summary only.`;
    const map = parseSummaries(md);
    const p = map.get(3);
    assert.equal(p.blobUrl, GITHUB_REPO);
    assert.equal(p.treeUrl, GITHUB_REPO);
  });
});

describe("mergeData", () => {
  it("includes improvementDisplay in searchText (regression test)", () => {
    const lb = [{ id: 1, title: "Paper", improvement: 63.64, improvementDisplay: "+63.64%", grade: "strong", gradeLabel: "Strong" }];
    const summaries = new Map();
    summaries.set(1, { method: "Method", title: "Full Paper Title", summary: "A summary.", paragraphs: [], blobUrl: "http://x", treeUrl: "http://y" });
    const merged = mergeData(lb, summaries);
    const st = merged[0].searchText;
    assert.ok(st.includes("63.64"), "searchText should contain the numeric improvement value");
    assert.ok(st.includes("+63.64%"), "searchText should contain the improvement display string");
  });

  it("falls back to leaderboard title when summary has no method", () => {
    const lb = [{ id: 1, title: "Fallback Title", improvement: 5, improvementDisplay: "+5.00%", grade: "moderate", gradeLabel: "Moderate" }];
    const merged = mergeData(lb, new Map());
    assert.equal(merged[0].method, "Fallback Title");
    assert.equal(merged[0].fullTitle, "Fallback Title");
    assert.equal(merged[0].summary, "");
  });
});

describe("filterItems logic", () => {
  function filterItems(items, activeFilter, activeQuery, activeSort) {
    let result = items;
    if (activeFilter !== "all") {
      result = result.filter(i => i.grade === activeFilter);
    }
    if (activeQuery) {
      result = result.filter(i => i.searchText.includes(activeQuery));
    }
    switch (activeSort) {
      case "improvement-desc":
        result = [...result].sort((a, b) => Math.abs(b.improvement) - Math.abs(a.improvement));
        break;
      case "improvement-asc":
        result = [...result].sort((a, b) => Math.abs(a.improvement) - Math.abs(b.improvement));
        break;
      default:
        result = [...result].sort((a, b) => a.id - b.id);
    }
    return result;
  }

  const items = [
    { id: 1, grade: "strong", improvement: 63.64, improvementDisplay: "+63.64%", searchText: "1 methoda full title a +63.64%" },
    { id: 2, grade: "moderate", improvement: 7.32, improvementDisplay: "↓7.32%", searchText: "2 methodb full title b ↓7.32%" },
    { id: 3, grade: "modest", improvement: 1.5, improvementDisplay: "+1.50%", searchText: "3 methodc full title c +1.50%" },
  ];

  it("filters by grade", () => {
    const result = filterItems(items, "strong", "", "id-asc");
    assert.equal(result.length, 1);
    assert.equal(result[0].id, 1);
  });

  it("searches by improvementDisplay value", () => {
    const result = filterItems(items, "all", "63.64", "id-asc");
    assert.equal(result.length, 1);
    assert.equal(result[0].id, 1);
  });

  it("searches by percent symbol in improvement", () => {
    const result = filterItems(items, "all", "7.32%", "id-asc");
    assert.equal(result.length, 1);
    assert.equal(result[0].id, 2);
  });

  it("searches by method name", () => {
    const result = filterItems(items, "all", "methodb", "id-asc");
    assert.equal(result.length, 1);
    assert.equal(result[0].id, 2);
  });

  it("sorts by improvement descending", () => {
    const result = filterItems(items, "all", "", "improvement-desc");
    assert.deepEqual(result.map(i => i.id), [1, 2, 3]);
  });

  it("sorts by id ascending (default)", () => {
    const result = filterItems(items, "all", "", "id-asc");
    assert.deepEqual(result.map(i => i.id), [1, 2, 3]);
  });
});

describe("Integration: parse README.md end-to-end", () => {
  it("parses all papers and every item has a searchable improvementDisplay", () => {
    const readmePath = path.join(__dirname, "..", "site-data", "README.md");
    const md = fs.readFileSync(readmePath, "utf8");
    const lb = parseLeaderboard(md);
    const summaries = parseSummaries(md);
    const items = mergeData(lb, summaries);

    assert.ok(items.length >= 100, `Expected >=100 papers, got ${items.length}`);

    for (const item of items) {
      // Every item must have improvementDisplay in searchText
      const rawDisp = String(item.improvementDisplay).toLowerCase();
      assert.ok(
        item.searchText.includes(rawDisp),
        `Paper #${item.id}: searchText must include improvementDisplay "${rawDisp}". searchText was: ${item.searchText}`
      );

      // blobUrl must point to README.md, not OPTIMIZATION.md
      if (item.blobUrl !== GITHUB_REPO) {
        assert.ok(
          item.blobUrl.endsWith("/README.md"),
          `Paper #${item.id}: blobUrl should end with /README.md, got ${item.blobUrl}`
        );
      }

      // Verify grade matches improvement value
      const gi = gradeInfo(item.improvement);
      assert.equal(item.grade, gi.grade, `Paper #${item.id}: grade mismatch`);
    }
  });

  it("builds correct GitHub URLs for the first 5 papers", () => {
    const readmePath = path.join(__dirname, "..", "site-data", "README.md");
    const md = fs.readFileSync(readmePath, "utf8");
    const lb = parseLeaderboard(md);
    const summaries = parseSummaries(md);
    const items = mergeData(lb, summaries).sort((a, b) => a.id - b.id);

    for (const item of items.slice(0, 5)) {
      // blobUrl must use the correct repo
      assert.ok(item.blobUrl.startsWith(GITHUB_REPO), `Paper #${item.id}: blobUrl must start with ${GITHUB_REPO}`);
      // treeUrl must use the correct repo
      assert.ok(item.treeUrl.startsWith(GITHUB_REPO), `Paper #${item.id}: treeUrl must start with ${GITHUB_REPO}`);
      // No reference to old repo
      assert.ok(!item.blobUrl.includes("Ericccc02"), `Paper #${item.id}: blobUrl contains old repo reference`);
      assert.ok(!item.treeUrl.includes("Ericccc02"), `Paper #${item.id}: treeUrl contains old repo reference`);
    }
  });
});
