#!/usr/bin/env bash
#
# Cut a release: bump the one canonical version, refresh the changelog, commit,
# and tag. Pushing the tag is left to you — that push is what triggers the PyPI
# publish workflow (.github/workflows/release.yml).
#
# Usage:
#     scripts/release.sh X.Y.Z
#
# Requires: cargo-edit (`cargo set-version`) and git-cliff, both installable with
#     cargo install cargo-edit git-cliff
#
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

[ $# -eq 1 ] || die "usage: scripts/release.sh X.Y.Z"
VERSION="$1"
TAG="v${VERSION}"

# Accept semver, optionally with a prerelease suffix (e.g. 0.2.0-rc1).
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] \
  || die "'$VERSION' is not a semantic version (expected X.Y.Z)"

command -v cargo >/dev/null      || die "cargo not found"
cargo set-version --help >/dev/null 2>&1 || die "cargo-edit missing: cargo install cargo-edit"
command -v git-cliff >/dev/null  || die "git-cliff not found: cargo install git-cliff"

cd "$(git rev-parse --show-toplevel)"

[ -z "$(git status --porcelain)" ] || die "working tree is dirty; commit or stash first"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" != "main" ] && [ "${TSL_RELEASE_ALLOW_BRANCH:-0}" != "1" ]; then
  die "not on main (on '$BRANCH'); set TSL_RELEASE_ALLOW_BRANCH=1 to override"
fi

git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null \
  && die "tag ${TAG} already exists"

echo "==> bumping workspace version to ${VERSION}"
cargo set-version --workspace "$VERSION"

echo "==> syncing R package version (no-op until the R package exists)"
python3 scripts/sync-version.py

echo "==> rendering CHANGELOG.md for ${TAG}"
git-cliff --tag "$TAG" --output CHANGELOG.md

echo "==> committing and tagging"
git add -A
git commit -m "chore(release): ${TAG}"
git tag -a "$TAG" -m "$TAG"

cat <<EOF

Release ${TAG} prepared locally. Review the commit, then publish with:

    git push origin ${BRANCH}
    git push origin ${TAG}

Pushing the tag triggers the PyPI publish workflow.
EOF
