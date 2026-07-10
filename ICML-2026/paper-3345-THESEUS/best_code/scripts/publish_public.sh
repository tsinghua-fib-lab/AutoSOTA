#!/usr/bin/env bash


usage() {
  cat <<'USAGE'
Usage: scripts/publish_public.sh /path/to/public/repo

Builds a curated public snapshot from this private repo and syncs it into an
existing checkout of the public repository.

Included by default:
  - README.md
  - CITATION.cff
  - pyproject.toml
  - uv.lock
  - src/merge_and_rebase
  - configs
  - docs
  - tests
  - LICENSE*

Excluded from the public snapshot:
  - todo_list.md
  - src/.cache contents
  - src/checkpoints contents
  - review-sensitive two-stage docs, tests, and configs

The script recreates src/.cache and src/checkpoints as empty tracked folders
containing only .gitkeep files.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

target_dir="$(cd "$1" && pwd)"

if [[ ! -d "$target_dir/.git" ]]; then
  echo "error: $target_dir is not a git repository checkout" >&2
  exit 1
fi

if [[ -n "$(git -C "$target_dir" status --porcelain)" ]]; then
  echo "error: public repo checkout has uncommitted changes: $target_dir" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ ! -d src/merge_and_rebase ]]; then
  echo "error: expected source tree at src/merge_and_rebase" >&2
  exit 1
fi

stage_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$stage_dir"
}
trap cleanup EXIT

copy_if_exists() {
  local rel_path="$1"

  if [[ ! -e "$rel_path" ]]; then
    return
  fi

  mkdir -p "$stage_dir/$(dirname "$rel_path")"
  cp -R "$rel_path" "$stage_dir/$rel_path"
}

remove_if_exists() {
  local rel_path="$1"

  if [[ -e "$stage_dir/$rel_path" ]]; then
    rm -rf "$stage_dir/$rel_path"
  fi
}

copy_if_exists README.md
copy_if_exists CITATION.cff
copy_if_exists pyproject.toml
copy_if_exists uv.lock
copy_if_exists src/merge_and_rebase
copy_if_exists configs
copy_if_exists docs
copy_if_exists tests
copy_if_exists LICENSE
copy_if_exists LICENSE.md
copy_if_exists LICENSE.txt

# Drop generated Python artifacts from the exported tree.
find "$stage_dir" \
  \( -name '__pycache__' -o -name '*.pyc' -o -name '*.pyo' \) \
  -exec rm -rf {} +

# Keep these paths visible in public without shipping private or heavy contents.
mkdir -p "$stage_dir/src/.cache" "$stage_dir/src/checkpoints"
: > "$stage_dir/src/.cache/.gitkeep"
: > "$stage_dir/src/checkpoints/.gitkeep"

# Remove review-sensitive and generated files from the public export.
remove_if_exists tests/test_text_embeddings_finetune_stage.py
remove_if_exists configs/vision8_task_arithmetic_two_stage.json
remove_if_exists configs/vision_connectivity_vision8_all_pairs.yaml
remove_if_exists configs/vision_connectivity_vision20_all_pairs_barrier_only.yaml
remove_if_exists docs/repo-overview-slides.html

if [[ -f "$stage_dir/docs/repo-overview-slides.md" ]]; then
  perl -0pi -e 's@\n---\n\n# \*\*Text Pre-Stages\*\*.*?\n---\n\n# \*\*What Gets Saved After Vision Training\*\*@\n---\n\n# **What Gets Saved After Vision Training**@s' "$stage_dir/docs/repo-overview-slides.md"
fi

cat > "$stage_dir/.gitignore" <<'GITIGNORE'
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.venv/
src/.cache/*
!src/.cache/.gitkeep
src/checkpoints/*
!src/checkpoints/.gitkeep
GITIGNORE

find "$target_dir" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -R "$stage_dir/." "$target_dir/"

echo "Public snapshot synced to: $target_dir"
echo "Next steps:"
echo "  git -C \"$target_dir\" status"
echo "  git -C \"$target_dir\" add -A"
echo "  git -C \"$target_dir\" commit -m 'Update public snapshot'"
echo "  git -C \"$target_dir\" push origin main"
