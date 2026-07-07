#!/bin/bash
# Run streaming prototype OOD detection on the ImageNet benchmark.
#
# Usage:
#   bash scripts/run_streaming_ood_imagenet.sh
#   bash scripts/run_streaming_ood_imagenet.sh --data_root /your/data/path
#   bash scripts/run_streaming_ood_imagenet.sh --backbone ViT-B/32 --batch_size 128
#
# Any extra arguments are forwarded directly to main.py, so any config key
# that main.py supports can be overridden from the command line here.

set -e

# Resolve the project root relative to this script's location so the script
# can be called from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_ROOT}"

python "${PROJECT_ROOT}/main.py" \
    --config "${PROJECT_ROOT}/configs/streaming_ood_imagenet.yaml" \
    "$@"
