#!/usr/bin/env bash


usage() {
  cat <<'USAGE'
Usage: scripts/publish_and_push_public.sh /path/to/public/repo [commit message]

Builds the public snapshot, commits any resulting changes, and pushes them to
origin using ~/.ssh/nello.

Examples:
  scripts/publish_and_push_public.sh /tmp/merge-and-rebase-public
  scripts/publish_and_push_public.sh /tmp/merge-and-rebase-public "Update public snapshot"
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

target_dir="$1"
commit_message="${2:-}"
repo_root="$(git rev-parse --show-toplevel)"

if [[ ! -x "$repo_root/scripts/publish_public.sh" ]]; then
  echo "error: missing executable publish script at $repo_root/scripts/publish_public.sh" >&2
  exit 1
fi

if [[ -z "$commit_message" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "Commit message [Update public snapshot]: " commit_message
  fi
  commit_message="${commit_message:-Update public snapshot}"
fi

"$repo_root/scripts/publish_public.sh" "$target_dir"

if [[ -n "$(git -C "$target_dir" status --porcelain)" ]]; then
  git -C "$target_dir" add -A
  git -C "$target_dir" commit -m "$commit_message"
  GIT_SSH_COMMAND='ssh -i ~/.ssh/nello -o IdentitiesOnly=yes' \
    git -C "$target_dir" push
  echo "Published public snapshot from $repo_root to $target_dir"
else
  echo "No public changes to publish."
fi
