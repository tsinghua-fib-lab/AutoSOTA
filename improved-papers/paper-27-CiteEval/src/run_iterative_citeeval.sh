#!/bin/bash

# CITEEVAL CONFIG
# --------------------
VERSION="citeeval-auto-12272024"
MODEL=gpt-4o

# Set CITEEVAL_ROOT
if [ -z "$CITEEVAL_ROOT" ]; then
    # Infer CITEEVAL_ROOT
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CITEEVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi


# CITEBENCH_DEV_FILE
# --------------------
CITEBENCH_DEV_FILE="$CITEEVAL_ROOT/data/citebench/metric_eval/metric_dev/citebench.metric_dev"
EVAL_OUTPUT_DIR="$CITEEVAL_ROOT/data/iterative_metric_eval_outputs"

# LOGGING
# --------------------
# Log set up info
CITEEVAl_SETTING="CiteEval version ${VERSION} | modules ${MODULES} | model ${MODEL}"
echo $CITEEVAl_SETTING


# COMMAND
# --------------------
# Core command for evaluation
python -m scripts.run_iterative_citeeval \
    --citebench_dev_file $CITEBENCH_DEV_FILE \
    --eval_output_dir $EVAL_OUTPUT_DIR \
    --version $VERSION \
    --model_name $MODEL \
