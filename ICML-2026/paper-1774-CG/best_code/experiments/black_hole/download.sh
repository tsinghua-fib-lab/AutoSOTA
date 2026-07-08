#!/usr/bin/env bash
# Fetch the public assets for the black-hole imaging experiment.
#
#   bash experiments/black_hole/download.sh
#
# Downloads into the vendored InverseBench engine so the default paths in
# `run.py` resolve:
#   third_party/InverseBench/checkpoints/blackhole-50k.pt   (diffusion prior)
#   third_party/InverseBench/data/blackhole/{measure,test}  (test instances)
#
# The mean-flow checkpoint (for `--posterior meanflow`) is authors' own and is
# NOT yet public — see the TODO at the bottom.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IB="$REPO_ROOT/third_party/InverseBench"
CKPT_DIR="$IB/checkpoints"
DATA_DIR="$IB/data/blackhole"

mkdir -p "$CKPT_DIR" "$DATA_DIR"

# 1) Diffusion prior (public InverseBench GitHub release).
#    The release asset is `blackhole.pt`; we save it under the name the configs
#    expect (`blackhole-50k.pt`).
PRIOR_URL="https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/blackhole.pt"
PRIOR_DST="$CKPT_DIR/blackhole-50k.pt"
if [[ -f "$PRIOR_DST" ]]; then
  echo "[skip] prior already present: $PRIOR_DST"
else
  echo "[get ] diffusion prior -> $PRIOR_DST"
  curl -L --fail -o "$PRIOR_DST" "$PRIOR_URL"
fi

# 2) Black-hole test/validation data (CaltechData).
#    The InverseBench data record bundles all problems; download the black-hole
#    archive from the data page and extract `measure/` and `test/` here:
#
#      https://data.caltech.edu/records/jfdr4-6ws87
#
#    The record is token-gated and may rotate; if the direct link below 404s,
#    open the page, grab the black-hole archive URL, and set BH_DATA_URL.
BH_DATA_URL="${BH_DATA_URL:-}"
if [[ -d "$DATA_DIR/test" && -d "$DATA_DIR/measure" ]]; then
  echo "[skip] black-hole data already present: $DATA_DIR"
elif [[ -n "$BH_DATA_URL" ]]; then
  echo "[get ] black-hole data -> $DATA_DIR"
  tmp="$(mktemp -d)"
  curl -L --fail -o "$tmp/bh.zip" "$BH_DATA_URL"
  unzip -q "$tmp/bh.zip" -d "$DATA_DIR"
  rm -rf "$tmp"
else
  echo "[TODO] Set BH_DATA_URL to the black-hole archive from"
  echo "       https://data.caltech.edu/records/jfdr4-6ws87 and re-run, or place"
  echo "       'measure/' and 'test/' under: $DATA_DIR"
fi

# 3) Mean-flow checkpoint (fast `--posterior meanflow`).
echo
echo "[TODO] Mean-flow checkpoint is not yet public. Once uploaded, fetch it here and"
echo "       pass --mf-ckpt <path> (or set BH_MF_CKPT). Use the EMA network-snapshot-*.pkl"
echo "       produced by easy_meanflow training (https://github.com/...)."

echo
echo "Done. Prior: $PRIOR_DST"
