#!/usr/bin/env bash

set -euo pipefail

usage() {
  printf '%s\n' \
    "Usage: $0 [--check-only] [release-tag]" \
    "" \
    "Publishes the processed protein fragment archive and manifest as GitHub" \
    "Release assets. The archive checksum is verified against the manifest" \
    "before any upload." \
    "" \
    "Defaults:" \
    "  release-tag: protein-data-v1" \
    "  PROTEIN_ARTIFACT: data/protein/casp12_fragments_L10_N20000.npz" \
    "  PROTEIN_MANIFEST: data/protein/casp12_fragments_L10_N20000.manifest.json" \
    "" \
    "Examples:" \
    "  $0 --check-only" \
    "  $0 protein-data-v1"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi

RELEASE_TAG="${1:-protein-data-v1}"
RELEASE_TITLE="${PROTEIN_RELEASE_TITLE:-Protein fragment data artifact v1}"
ARTIFACT="${PROTEIN_ARTIFACT:-data/protein/casp12_fragments_L10_N20000.npz}"
MANIFEST="${PROTEIN_MANIFEST:-data/protein/casp12_fragments_L10_N20000.manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ ! -f "$ARTIFACT" ]]; then
  echo "Missing protein artifact: $ARTIFACT" >&2
  echo "Generate it with: python training/process_protein_fragments.py --name casp12 --fragment-length 10 --max-data-length 20000" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing protein manifest: $MANIFEST" >&2
  exit 1
fi

expected_sha="$(
  "$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    manifest = json.load(f)
print(manifest["fragments"]["sha256"])
PY
)"

actual_sha="$(
  "$PYTHON_BIN" - "$ARTIFACT" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"

if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "Checksum mismatch for $ARTIFACT" >&2
  echo "  manifest: $expected_sha" >&2
  echo "  actual:   $actual_sha" >&2
  exit 1
fi

echo "Verified protein artifact checksum: $actual_sha"

if [[ "$CHECK_ONLY" == "1" ]]; then
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "Missing GitHub CLI 'gh'. Install it and authenticate with 'gh auth login'." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run 'gh auth login' first." >&2
  exit 1
fi

notes_file="$(mktemp)"
trap 'rm -f "$notes_file"' EXIT

"$PYTHON_BIN" - "$MANIFEST" "$notes_file" <<'PY'
import json
import os
import sys

manifest_path, notes_path = sys.argv[1], sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

fragments = manifest["fragments"]
lines = [
    "# Protein Fragment Data Artifact",
    "",
    "Processed SideChainNet backbone fragments for the protein experiments.",
    "",
    f"- Dataset source: {manifest.get('dataset_source')}",
    f"- SideChainNet dataset: {manifest.get('sidechainnet_dataset')}",
    f"- SideChainNet version: {manifest.get('sidechainnet_version')}",
    f"- OpenMM version: {manifest.get('openmm_version')}",
    f"- Fragment length: {manifest.get('fragment_length')}",
    f"- Requested fragments: {manifest.get('max_data_length')}",
    f"- Archive path in repo workflow: {fragments.get('path')}",
    f"- Shape: {fragments.get('shape')}",
    f"- Dtype: {fragments.get('dtype')}",
    f"- Bytes: {fragments.get('bytes')}",
    f"- SHA-256: `{fragments.get('sha256')}`",
    "",
    "Download both the `.npz` archive and `.manifest.json` file into `data/protein/` before running `training/protein.sh`.",
]

with open(notes_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY

if gh release view "$RELEASE_TAG" >/dev/null 2>&1; then
  echo "Using existing release: $RELEASE_TAG"
else
  gh release create "$RELEASE_TAG" --title "$RELEASE_TITLE" --notes-file "$notes_file"
fi

gh release upload "$RELEASE_TAG" "$ARTIFACT" "$MANIFEST" --clobber
release_url="$(gh release view "$RELEASE_TAG" --json url -q .url)"
echo "Release URL: $release_url"

repo_full_name="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)"
if [[ -n "$repo_full_name" ]]; then
  artifact_name="$(basename "$ARTIFACT")"
  manifest_name="$(basename "$MANIFEST")"
  base_url="https://github.com/${repo_full_name}/releases/download/${RELEASE_TAG}"
  echo "Artifact URL: ${base_url}/${artifact_name}"
  echo "Manifest URL: ${base_url}/${manifest_name}"
fi
