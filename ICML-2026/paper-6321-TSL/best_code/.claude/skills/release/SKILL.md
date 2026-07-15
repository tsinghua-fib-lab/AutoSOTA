---
name: release
description: >-
  Cut and publish a new release of the tensorsl Python package (and the tsl_rust
  Rust crate) to PyPI. Bumps the single source-of-truth version, regenerates the
  changelog with git-cliff, opens a release PR (main is protected), dry-runs the
  build with no publish, and — after the PR merges and you give an explicit
  go-ahead — pushes the vX.Y.Z tag that triggers the PyPI publish. Trigger on
  "release X.Y.Z", "publish a new version", "cut a release", "bump the version
  and release", "ship a new tensorsl version".
---

# Release tensorsl

The version lives in **one place**: `[workspace.package].version` in the root
`Cargo.toml`. Both Rust crates inherit it (`version.workspace = true`) and the
Python package reads it through maturin's dynamic version, so the number never
needs copying. A release is: **bump that version → regenerate the changelog →
land it on `main` via a PR → push a `vX.Y.Z` tag**, which triggers
[`.github/workflows/release.yml`](../../../.github/workflows/release.yml) to build
the distribution and publish to PyPI via Trusted Publishing (OIDC, no token).

Two invariants that shape every step:

- **`main` is protected.** The release commit (version bump + changelog) cannot be
  pushed straight to `main`; it goes through a PR. The **tag is pushed only after
  the PR merges**, against the merged commit.
- **Pushing the tag is irreversible and outward-facing.** A PyPI version can never
  be reused (only yanked). **Always get an explicit go-ahead from the user before
  pushing the tag.** Everything before that is safe and reversible.

## 0. Inputs and preconditions

- **Version `X.Y.Z`** comes from the skill args. If absent, ask. Guidance for a
  0.x project: `fix`/`perf` → patch (`0.1.1`); a notable change (`feat`, new
  backend, different numerics) → minor (`0.2.0`).
- One-time local tools: `cargo install cargo-edit git-cliff`.
- Confirm the working tree is clean and `main` is up to date:
  `git checkout main && git pull --ff-only && git status --porcelain` (must be empty).

## 1. Prepare the release on a branch

```sh
git checkout -b release/vX.Y.Z
cargo set-version --workspace X.Y.Z        # the single source of truth; refuses downgrades
python3 scripts/sync-version.py            # sync the R package version (no-op until tsl-r lands)
git-cliff --tag vX.Y.Z -o CHANGELOG.md     # changelog from Conventional Commits since the last tag
```

`cargo set-version`/`git-cliff` are exactly what `scripts/release.sh` runs — but
that script also commits **and tags**, which doesn't fit a protected `main`, so run
the commands directly and tag later (step 4). Review the changelog diff; if commits
are mis-grouped, fix `cliff.toml` (`feat`/`fix`/`perf`/`docs` are shown,
`chore`/`ci`/`build`/`test` are dropped).

## 2. Open the release PR

```sh
git add -A
git commit -m "chore(release): X.Y.Z"
git push -u origin release/vX.Y.Z
gh pr create --base main --title "chore(release): X.Y.Z" --body "<summary of the release>"
```

## 3. Dry-run the build (no publish)

```sh
gh workflow run release.yml --ref release/vX.Y.Z
gh run watch "$(gh run list --workflow=release.yml --branch release/vX.Y.Z -L1 --json databaseId -q '.[0].databaseId')" --exit-status
```

The `publish` job is gated to **tags**, so a `workflow_dispatch` run only builds
and tests the artifacts — it can never publish. Use this to confirm the build is
green before committing to a tag. If a job sits **queued** for a long time it's
waiting for a runner (macOS/Windows queues), not hung.

## 4. Merge, then tag to publish

After the user merges the PR **and explicitly approves publishing**:

```sh
git checkout main && git pull --ff-only
git tag -a vX.Y.Z -m "vX.Y.Z"             # annotate the merged commit
git push origin vX.Y.Z                     # THIS triggers the publish — irreversible
```

Then watch the release run:

```sh
gh run watch "$(gh run list --workflow=release.yml -L1 --json databaseId -q '.[0].databaseId')" --exit-status
```

The run builds the distribution → checks the built version equals the tag →
publishes to PyPI → drafts the GitHub Release. If the `pypi` environment has a
required reviewer, the publish job pauses for approval in the Actions tab.

## 5. Verify

```sh
gh release view vX.Y.Z
```

Confirm the version on PyPI (`https://pypi.org/project/tensorsl/`) and, optionally,
`pip install tensorsl==X.Y.Z` in a clean venv.

## Recovery

If the run fails **before** PyPI accepts the upload (e.g. a metadata error), the
version is still free: fix on a branch, merge, then **move the tag** to the fixed
commit and re-push —
`git tag -d vX.Y.Z && git push origin --delete vX.Y.Z`, then re-tag and push.
Only do this while nothing has been accepted by PyPI.

## Gotchas (already configured — do not regress)

- The tag **must** be `vX.Y.Z` (the `v` prefix is the workflow trigger).
- The `pypi` GitHub environment must allow `v*` **tags** to deploy, else the publish
  job is rejected ("not allowed to deploy due to environment protection rules").
- **No direct-URL/git dependencies** in `tsl-py/pyproject.toml` — PyPI rejects them
  (optional git deps live in `tsl-py/examples/requirements.txt`).
- The package `readme` must point at `tsl-py/README.md` (sibling to `pyproject.toml`),
  not `../README.md` — the sdist flattens to its root and a parent path dangles.
- The Python distribution is **`tensorsl`** on PyPI and imported under the same name.
