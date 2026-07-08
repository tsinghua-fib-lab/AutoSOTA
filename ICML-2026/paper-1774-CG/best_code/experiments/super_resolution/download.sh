#!/usr/bin/env bash
# Fetch the public assets for the 4x super-resolution experiment (paper §6.3a).
#
#   bash experiments/super_resolution/download.sh
#
# Downloads two things so the default paths in `run.py` resolve:
#
#   1. The public pixel-mean-flow (pMF) inference repo, cloned to the location
#      `pixel_mean_flow_adapter.py` expects for `from pmf import pixelMeanFlow`:
#        experiments/super_resolution/pixel_space_inverse_problems/external/pMF
#      (the adapter inserts `<adapter_dir>/external/pMF` on sys.path).
#
#   2. The pMF-B/16 checkpoint from HuggingFace `Lyy0725/pMF`, saved to:
#        experiments/super_resolution/checkpoints/pMF-B-16.pt
#
# ImageNet-val-256 is NOT downloaded here — it is user-provided via --val-root.
#
# The script is idempotent (skips anything already present).
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXEL_SPACE_DIR="$THIS_DIR/pixel_space_inverse_problems"
EXTERNAL_DIR="$PIXEL_SPACE_DIR/external"
PMF_DIR="$EXTERNAL_DIR/pMF"
CKPT_DIR="$THIS_DIR/checkpoints"
CKPT_DST="$CKPT_DIR/pMF-B-16.pt"

mkdir -p "$EXTERNAL_DIR" "$CKPT_DIR"

# 1) Public pMF inference repo (provides `pmf.py` + `models/`).
PMF_REPO_URL="https://github.com/Lyy-iiis/pmf"
if [[ -f "$PMF_DIR/pmf.py" ]]; then
  echo "[skip] pMF repo already present: $PMF_DIR"
else
  echo "[get ] cloning pMF inference repo -> $PMF_DIR"
  git clone --depth 1 "$PMF_REPO_URL" "$PMF_DIR"
fi

# 2) pMF-B/16 checkpoint (HuggingFace Lyy0725/pMF).
#    NOTE: confirm the exact filename on the HF repo before a fresh download:
#      https://huggingface.co/Lyy0725/pMF/tree/main
#    The pMF README and the adapter both reference `pMF-B-16.pt`.
CKPT_URL="https://huggingface.co/Lyy0725/pMF/resolve/main/pMF-B-16.pt"
if [[ -f "$CKPT_DST" ]]; then
  echo "[skip] checkpoint already present: $CKPT_DST"
else
  echo "[get ] pMF-B/16 checkpoint (~473MB) -> $CKPT_DST"
  curl -L --fail -o "$CKPT_DST" "$CKPT_URL"
fi

echo
echo "Done."
echo "  pMF repo:     $PMF_DIR"
echo "  checkpoint:   $CKPT_DST"
echo
echo "ImageNet-val-256 is user-provided: pass --val-root <dir> at run time."
