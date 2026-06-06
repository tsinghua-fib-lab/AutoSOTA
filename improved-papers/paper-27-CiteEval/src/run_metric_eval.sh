#!/bin/bash

# Set CITEEVAL_ROOT
if [ -z "$CITEEVAL_ROOT" ]; then
    # Infer CITEEVAL_ROOT
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CITEEVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi


# PREDICTION_FILENAME: CITEEVAL-AUTO
# --------------------
DATA_SPLIT="test"  # dev, test
PREDICTION_FILE="$CITEEVAL_ROOT/data/citebench/metric_eval/metric_${DATA_SPLIT}/citebench.metric_${DATA_SPLIT}"
EVAL_OUTPUT_DIR="$CITEEVAL_ROOT/data/metric_eval_outputs"

CR_ITERCOE_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/citebench.metric_test.citeeval-auto-12272024.cr_itercoe.deepseek-chat.out"
CR_EDITDIST_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/citebench.metric_test.citeeval-auto-12272024.cr_editdist.deepseek-chat.out"
CR_EMSEMBLE_METRIC_OUTPUT="${CR_ITERCOE_METRIC_OUTPUT},${CR_EDITDIST_METRIC_OUTPUT}"
CR_EDITDIST_DEV_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/citebench.metric_dev.citeeval-auto-12272024.cr_itercoe.deepseek-chat.out"


# PREDICTION_FILENAME: BASELINES
# --------------------
AUTOAIS_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/citebench.metric_test.autoais.out"
LQAC_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/citebench.metric_test.lqac.out"
ATTRISCORE_METRIC_OUTPUT="$EVAL_OUTPUT_DIR/citebench.metric_test.attriscore-attribution-with-definition-zs.gpt-4.out"


# COMMANDS
# --------------------
python -m scripts.evaluate_metric \
    --metric "citeeval-auto-12272024.ca" \
    --metric_output $CR_EDITDIST_METRIC_OUTPUT \
    --split $DATA_SPLIT


python -m scripts.evaluate_metric \
    --metric "citeeval-auto-12272024.cr" \
    --metric_output $CR_EMSEMBLE_METRIC_OUTPUT \
    --split $DATA_SPLIT


python -m scripts.evaluate_metric \
    --metric "autoais" \
    --metric_output $AUTOAIS_METRIC_OUTPUT \
    --split $DATA_SPLIT


python -m scripts.evaluate_metric \
    --metric "lqac" \
    --metric_output $LQAC_METRIC_OUTPUT \
    --split $DATA_SPLIT


python -m scripts.evaluate_metric \
    --metric "attriscore" \
    --metric_output $ATTRISCORE_METRIC_OUTPUT \
    --split $DATA_SPLIT
