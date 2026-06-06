#!/bin/bash

# CITEEVAL CONFIG
# --------------------
VERSION="citeeval-auto-12272024"
MODEL=deepseek-chat
MODULES=ca,ce,cr_itercoe,cr_editdist  # ca, ce, cr_itercoe, cr_editdist

# Set CITEEVAL_ROOT
if [ -z "$CITEEVAL_ROOT" ]; then
    # Infer CITEEVAL_ROOT
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CITEEVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi


# METRIC EVAL
# --------------------
DATA_SPLIT=test
PREDICTION_FILE="$CITEEVAL_ROOT/data/citebench/metric_eval/metric_${DATA_SPLIT}/citebench.metric_${DATA_SPLIT}"
EVAL_OUTPUT_DIR="$CITEEVAL_ROOT/data/metric_eval_outputs"


# SYSTEM EVAL
# --------------------
# SYSTEM_EVAL_FILENAME="system_eval_examples.citeeval"
# PREDICTION_FILE="$CITEEVAL_ROOT/data/system_eval/${SYSTEM_EVAL_FILENAME}"
# EVAL_OUTPUT_DIR="$CITEEVAL_ROOT/data/system_eval_outputs"


# LOGGING
# --------------------
# Log set up info
CITEEVAl_SETTING="CiteEval version ${VERSION} | modules ${MODULES} | model ${MODEL}"
DATA_SETTING="Eval file: ${PREDICTION_FILE}"
echo $CITEEVAl_SETTING
echo $DATA_SETTING


# COMMAND
# --------------------
# Core command for evaluation
python -m scripts.run_citeeval \
    --response_output_file $PREDICTION_FILE \
    --eval_output_dir  $EVAL_OUTPUT_DIR \
    --modules $MODULES \
    --version $VERSION \
    --model_name $MODEL \
    --n_threads 24