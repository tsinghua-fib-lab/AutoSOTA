---
name: align-docs
description: >-
  Reconcile the TSL mkdocs documentation under docs/ with recent code changes.
  Run this AFTER code work on a branch is finished: it reads the recent commits
  (or a range you pass as args), maps each changed code path to the doc page(s)
  that describe it, and edits those pages so the docs match the code. Trigger on
  "align the docs with the code", "update the docs for these changes",
  "sync the docs", "the docs are stale after my refactor", or when wrapping up a
  code change and the public docs need to follow.
---

# Align docs with code

The TSL docs (`docs/`, a mkdocs-material site) are a hand-written mirror of the
code. When code changes, the matching pages drift out of date. This skill brings
them back in sync **from the actual diff** — not by rewriting pages from scratch.

Core principle: **the code is the source of truth.** Read what the code now does,
then make the smallest doc edits that make the page true again. Touch only pages
whose underlying code actually changed.

## 1. Scope the change set

Figure out which code commits the docs still need to catch up with.

- **If the user passed args**, honor them. Accept any of:
  - a git range — `A..B` (reconcile commits in that range);
  - a count — `5` (reconcile the last 5 commits);
  - a ref — `main` (reconcile `main..HEAD`).
- **Otherwise default to "since the docs were last aligned":**

  ```sh
  LAST_DOCS=$(git log -1 --format=%H -- docs/)
  # code commits made after the last docs-touching commit:
  git log --oneline "$LAST_DOCS"..HEAD -- \
    src/ tsl-py/src/ tsl-py/python/ tsl-split-evolution-dashboard/ Cargo.toml
  ```

  If that range is empty, or the most recent commit touched *both* docs and code,
  fall back to inspecting the last handful of commits (`git log --oneline -10`) and
  confirm the intended scope with the user before editing.
- **Also check the working tree** (`git status --short`, `git diff`). Code changes
  may not be committed yet; if there are uncommitted code edits, include them.

Then read the real diffs, restricted to code paths, e.g.:

```sh
git diff "$LAST_DOCS"..HEAD -- src/ tsl-py/src/ tsl-py/python/
```

Build a list of *what behavior/API/struct/math actually changed* — renamed or added
public symbols, changed function signatures or defaults, new/removed struct fields
that appear in docs, new or removed hyperparameters, changed numeric behavior, new
modules. **Ignore pure internal refactors** (private renames, code movement) that
don't change anything a doc page asserts.

## 2. Map changed code → doc pages

| Changed code | Doc page(s) to check |
|---|---|
| `src/grid_tensor.rs`, `src/grid_tensor/**` | `docs/docs/code/grid-tensor.md`; math in `docs/docs/math/model.md`, `math/fitting.md` |
| `src/stage_predictor.rs`, `src/stage_predictor/**` | `docs/docs/code/stage-predictor.md`; `math/bagging-aggregation.md` |
| `src/forest.rs`, `src/forest/**` (`fitter.rs`) | `docs/docs/code/forest.md`; `math/fitting.md` |
| `src/logging.rs`, `src/logging/**`, `grid_tensor/logging_helpers.rs` | `docs/docs/code/logging.md` |
| any `params.rs` builder / `src/forest/params.rs` (hyperparameters) | `docs/docs/guides/hyperparameters.md` **and** the parameter tables in `code/python-api.md` |
| `tsl-py/src/lib.rs` (PyO3 bindings: `TSL`, `GridTensor`, `StagePredictor`, `FitResult`) | `docs/docs/code/python-api.md` |
| `tsl-py/python/tensorsl/sklearn.py` (`TSLRegressor`) | `docs/docs/code/python-api.md` |
| `tsl-py/python/tensorsl/__init__.py` (public exports) | `docs/docs/code/python-api.md` (the import line + symbol list) |
| `tsl-py/python/tensorsl/plot/**` | `docs/docs/code/plotting.md` |
| partial dependence (`compute_partial_dependence_function`) | `code/python-api.md`, `math/partial-dependence.md` |
| `tsl-py/examples/**` | `docs/docs/guides/getting-started.md`, `docs/docs/index.md` |
| `tsl-split-evolution-dashboard/**` (`tslviz`) | `docs/docs/guides/visualizing.md`, `code/logging.md` |
| cross-cutting invariants / module hierarchy | `docs/docs/code/architecture.md` |

Each `code/*.md` page opens by naming the `src/...` path it documents — use that as
a sanity check on the mapping. If a change introduces a genuinely new area with no
home, a **new page** may be warranted: create it and add a `nav:` entry in
`docs/mkdocs.yml`.

## 3. Edit the pages

For each affected page: read the page, read the new code it describes, then make
surgical edits. Match what's actually there now — struct fields, signatures, default
values, enum variants, return types, method names, described behavior, and any
worked numbers. Keep the page's existing structure, voice, and tables.

When a public symbol is renamed/removed, also fix every cross-reference and anchor
link to it across the docs (CI builds `--strict`, so a dangling `#anchor` fails).

## 4. Honor the docs conventions

These are established project rules — follow them or the build/voice breaks:

- **Math is inline LaTeX.** Write all math in prose and table cells as `$...$`
  (arithmatex `generic: true`), never Unicode/plain-text glyphs. Match the notation
  in `docs/docs/math/*.md` and cross-link the anchor that defines the quantity (e.g.
  backbone/tilt → `math/model.md#backbone-and-exponential-tilt`). Keep code
  identifiers like `scaling_plus` in backticks.
- **API kind pills** come from heading `id` prefixes: `cls-` class, `meth-` method,
  `cmeth-` classmethod, `fn-` function, `dc-` dataclass — e.g. `## `foo` { #meth-x-foo }`.
  A *new* prefix must be added to all four selector groups in
  `docs/docs/stylesheets/extra.css` (shared chrome, heading size, nav size, color).
- **Pipe-in-table gotcha:** inside an inline-code span in a Markdown table, use a
  plain `|` (`` `str | None` ``), not `\|` — the backslash renders literally there.
- **Stay code-faithful.** Describe what the code does. Do **not** introduce
  paper-vs-code discrepancy notes, and never reference the private review file. Only
  benign, mathematically-equivalent implementation choices may carry a neutral note.

## 5. Verify the build

CI builds the site in strict mode; do the same to catch broken links/anchors:

```sh
mkdocs build --strict -f docs/mkdocs.yml
```

If `mkdocs` isn't installed, install the documented dev deps into the project venv
first, then build:

```sh
/Users/jin/Documents/TSL/.venv/bin/pip install -r docs/requirements.txt
/Users/jin/Documents/TSL/.venv/bin/mkdocs build --strict -f docs/mkdocs.yml
```

If a build still isn't possible, fall back to manually checking that every
`[...](...)` link and `#anchor` you touched resolves, and report that the strict
build was not run.

## 6. Commit & report

- **Commits:** don't commit unless the user asks. When you do, use the `docs:`
  Conventional Commits prefix (per `CONTRIBUTING.md`), stage doc paths **explicitly**
  (never `git add -A`), and keep any local paper-vs-code notes unstaged.
- **Report** a concise summary: which commits/range were reconciled, each page you
  changed and why, any code changes that needed no doc update (and why), and the
  strict-build result.
