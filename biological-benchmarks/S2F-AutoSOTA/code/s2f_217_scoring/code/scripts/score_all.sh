#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${DMS_DATA:?Set DMS_DATA to extracted DMS substitution CSVs}"
: "${STRUCTURES:?Set STRUCTURES to ProteinGym AF2 structures}"
: "${MSA_DATA:?Set MSA_DATA to ProteinGym a2m files}"
: "${WEIGHTS:?Set WEIGHTS to the directory containing all eight verified files}"
: "${SCORES:?Set SCORES to the output directory}"
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
exec "${PYTHON:-python}" "$SCRIPT_DIR/score.py" \
    --DMS-data-folder "$DMS_DATA" --structure-folder "$STRUCTURES" \
    --msa-folder "$MSA_DATA" --checkpoint "$WEIGHTS/model_best.pth" \
    --esm-weights "$WEIGHTS" --output-scores-folder "$SCORES" \
    --gpu "${GPU:-0}" --all --verify-weights
