#!/bin/bash

# PREDICTION_FILENAME
# --------------------
# Set CITEEVAL_ROOT
if [ -z "$CITEEVAL_ROOT" ]; then
    # Infer CITEEVAL_ROOT
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CITEEVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# SYSTEM EVAL
# --------------------
SYSTEM_EVAL_FILENAME=system_eval_examples.citeeval
SYSTEM_EVAL_FILE="$CITEEVAL_ROOT/data/system_eval/$SYSTEM_EVAL_FILENAME"

EVAL_OUTPUT_DIR="$CITEEVAL_ROOT/data/system_eval_outputs"
CR_ITERCOE_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/${SYSTEM_EVAL_FILENAME}.citeeval-auto-12272024.cr_itercoe.gpt-4o.out"
CR_EDITDIST_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/${SYSTEM_EVAL_FILENAME}.citeeval-auto-12272024.cr_editdist.gpt-4o.out"
CR_EMSEMBLE_METRIC_OUTPUT="${CR_ITERCOE_METRIC_OUTPUT},${CR_EDITDIST_METRIC_OUTPUT}"

python -m scripts.evaluate_system \
    --system_output $SYSTEM_EVAL_FILE \
    --metric_output $CR_EMSEMBLE_METRIC_OUTPUT \
    --cited