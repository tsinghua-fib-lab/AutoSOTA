#!/bin/bash
PYTHON=$(which python)
echo "Using Python: $PYTHON"

# Paths - Auto-detect demo root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="${DEMO_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR}"

DATASET="${DATASET}"
TRAIN_FILE="${TRAIN_FILE}"
CAL_FILE="${CAL_FILE}"
TEST_FILE="${TEST_FILE}"

## Models
# Update these paths to point to your model locations
MODEL_PATH="${MODEL_PATH:-../models/mdeberta-v3-base}"

# Experiment Settings
N_SPLITS=$1
TEST_RATIO=$2 
ALPHA=$3
knearest=$4
DATA_FORMAT=$5
HELD_OUT_RATIO=$6

run_repeated_eval() {
    local ROUTER_NAME=$1
    local RACER_NONC_SCORE=$2
    local SAVE_SUBDIR=$3
    local TRAINED_PATH=$4 
    local KNEAREST=$5
    local DATA_FORMAT=$6
    local EXTRA_ARGS=$7   
    
    echo "----------------------------------------------------------------"
    echo "Running Repeated Eval for ${ROUTER_NAME}+RACER (Nonconformity Score: ${RACER_NONC_SCORE})"
    echo "----------------------------------------------------------------"
    
    local SAVE_FOLDER="${OUTPUT_DIR}/${DATASET}/repeated_racer_alpha${ALPHA}_test_ratio${TEST_RATIO}_n_splits${N_SPLITS}/${SAVE_SUBDIR}"
    mkdir -p "${SAVE_FOLDER}"
    local LOG_FILE="${SAVE_FOLDER}/run.log"

    $PYTHON ${ROOT_DIR}/main.py \
        --router_name ${ROUTER_NAME} \
        --model_path ${MODEL_PATH} \
        --trained_router_path "${TRAINED_PATH}" \
        --data_name ${DATASET} \
        --train_paths ${TRAIN_FILE} \
        --cal_paths ${CAL_FILE} \
        --test_paths ${TEST_FILE} \
        --answer_path ${TEST_FILE} \
        --save_folder ${SAVE_FOLDER} \
        --alpha ${ALPHA} \
        --racer_nonc_score ${RACER_NONC_SCORE} \
        --n_splits ${N_SPLITS} \
        --test_ratio ${TEST_RATIO} \
        --held_out_ratio ${HELD_OUT_RATIO} \
        --use_softmax True \
        --data_types multi_attempt \
        --data_format ${DATA_FORMAT} \
        --knearest ${KNEAREST} \
        ${EXTRA_ARGS} 2>&1 | tee "${LOG_FILE}"
}
# --- 2. KNN ---
knearest=40
run_repeated_eval "knn" "gap" "knn_gap" "" "${knearest}" "${DATA_FORMAT}" "--seed 42"
run_repeated_eval "knn" "one_minus_prob" "knn_prob" "" "${knearest}" "${DATA_FORMAT}" "--seed 42"

echo "All repeated evaluations complete."
