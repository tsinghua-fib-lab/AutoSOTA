#!/bin/bash
# Download the Who&When benchmark from the upstream Agents_Failure_Attribution repo
# (Zhang et al., ICML 2025), placing it at data/whoandwhen/.
#
# We do not redistribute Who&When; this is just a convenience wrapper around
# `git clone` + a copy. Run from the CORRECT/ project root.
#
# Usage: bash scripts/download_whoandwhen.sh [destination]
#   destination  defaults to data/whoandwhen

set -e

DEST=${1:-"data/whoandwhen"}
UPSTREAM_URL="https://github.com/mingyin1/Agents_Failure_Attribution.git"
TMP_CLONE=$(mktemp -d)

if [ -d "$DEST" ] && [ "$(ls -A "$DEST" 2>/dev/null)" ]; then
    echo "Destination $DEST already exists and is non-empty. Aborting to avoid overwrite."
    echo "Remove it first or pass a different destination as an argument."
    exit 1
fi

echo "========================================"
echo "Downloading Who&When benchmark"
echo "========================================"
echo "Upstream:    $UPSTREAM_URL"
echo "Destination: $DEST"
echo "========================================"

trap 'rm -rf "$TMP_CLONE"' EXIT

echo "Cloning into temporary directory..."
git clone --depth=1 "$UPSTREAM_URL" "$TMP_CLONE"

mkdir -p "$DEST"

# Copy the two Who&When subsets if present in the upstream tree.
# Upstream layout (as of mid-2025) places them at the repo root:
#   Who&When/Algorithm-Generated/*.json
#   Who&When/Hand-Crafted/*.json
# Fall back to a recursive find if that path doesn't exist.
if [ -d "$TMP_CLONE/Who&When" ]; then
    cp -r "$TMP_CLONE/Who&When/." "$DEST/"
else
    echo "Note: Who&When/ not found at upstream root. Searching..."
    found=$(find "$TMP_CLONE" -maxdepth 4 -type d -name "Algorithm-Generated" | head -1)
    if [ -n "$found" ]; then
        parent=$(dirname "$found")
        cp -r "$parent/." "$DEST/"
    else
        echo "ERROR: Could not locate Who&When data in upstream clone."
        echo "Inspect $TMP_CLONE manually and copy the right directories into $DEST."
        exit 1
    fi
fi

echo ""
echo "Downloaded to $DEST. Top-level entries:"
ls "$DEST"
echo ""
echo "Cite Who&When if you use it:"
echo "  Zhang et al., \"Which Agent Causes Task Failures and When?\", ICML 2025."
