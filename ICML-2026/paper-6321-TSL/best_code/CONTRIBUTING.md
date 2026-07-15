# Contributing

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary).
Each commit message has the form:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- **type** — one of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, or `chore`.
- **scope** *(optional)* — the area touched, e.g. `grid_tensor`, `forest`, `tsl-py`, `dashboard`.
- **description** — short, imperative, lower-case, no trailing period.
- **body** *(optional)* — what changed and why, wrapped at ~72 columns.
- **breaking changes** — append `!` after the type/scope (e.g. `feat(forest)!:`) **and/or** add a `BREAKING CHANGE:` footer describing the break.

Examples:

```
feat(forest): add orthogonal-greedy OLS refit across stages
fix(grid_tensor): apply OLS scaling exactly once in predict
docs: restyle README header with floated logo and badge row
perf(grid_tensor): use binned prefix sums for split candidates
```

### Examples and other non-shipped code

`feat`, `fix`, and `perf` map to SemVer bumps of the published Rust crate and Python
package, so reserve them for changes to shipped code (`src/`, `tsl-py/src/`,
`tsl-py/python/`). Changes that don't affect the published packages take a non-release
type with a scope:

- `docs(examples):` — example scripts (`tsl-py/examples/*.py`) and their README.
- `chore(examples):` — regenerated figures (`tsl-py/examples/figures/`) or pretrained
  model binaries (`tsl-py/examples/models/`).

(This replaces the bare `example:` type used in earlier history.)

See the [Conventional Commits summary](https://www.conventionalcommits.org/en/v1.0.0/#summary)
for the full specification.

## Releasing

There is a single source-of-truth version: `[workspace.package].version` in the root
`Cargo.toml`. Both Rust crates inherit it, `tsl-py` exposes it to Python through maturin's
dynamic version (so `pyproject.toml` carries no version of its own), and the R connector
package (`tslr`, in `tsl-r/`) is kept in step by `scripts/sync-version.py`. They cannot drift.

The Python package is distributed on PyPI as **`tensorsl`** and imported under the same
name: `import tensorsl`.

To cut a release, from a clean `main`:

```sh
scripts/release.sh X.Y.Z      # bumps the version, regenerates CHANGELOG.md, commits, tags
git push origin main
git push origin vX.Y.Z        # this tag is what triggers publishing
```

`scripts/release.sh` needs two tools once: `cargo install cargo-edit git-cliff`. The version
bump is deliberate and manual; `git-cliff` writes the changelog from the Conventional Commits
since the previous tag (`cliff.toml` controls grouping).

Pushing the `vX.Y.Z` tag runs [`.github/workflows/release.yml`](.github/workflows/release.yml),
which builds platform wheels and an sdist, checks the built version matches the tag, and
publishes them to PyPI via Trusted Publishing (OIDC — no stored token), then drafts a GitHub
Release. The build is pure Rust, so the wheels are self-contained.

**Always create the tag on `main`, last.** The `vX.Y.Z` tag must point at a commit that
lives on `main`. When the version bump goes through a pull request (because `main` is
protected), wait for it to merge, then `git checkout main && git pull` and tag the merged
commit — never tag the feature branch. A squash-merge rewrites the branch into a new commit
on `main`, so a tag created on the branch is orphaned, pointing at a commit that is not on
`main`. `skip-existing` keeps the publish idempotent if a tag is ever re-pushed or moved.

One-time project setup (already done for the canonical repo): register a PyPI Trusted
Publisher for project `tensorsl` pointing at this repo's `release.yml` and the `pypi`
environment, and create that `pypi` environment in the GitHub repo settings.
