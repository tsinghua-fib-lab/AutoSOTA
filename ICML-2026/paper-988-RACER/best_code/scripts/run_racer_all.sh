#!/bin/bash
set -e

ROOT=".."
RUN_SH="${ROOT}/scripts/run_racer.sh"
OUTPUT_DIR="${ROOT}/results"

export CUDA_VISIBLE_DEVICES="5"

# Default Parameters
N_SPLITS=100
TEST_RATIO=0.4
KNN=40
HELD_OUT_RATIO=0.1

# Dataset list
# DATASETS=(gsm8k mmlu arc_challenge cmmlu)
DATASETS=(gsm8k)
ALPHAS=(0.03) 
FORMATS=(label)

DATA_DIR="${ROOT}/data"
TRAIN_FILEs_same="${DATA_DIR}/mmlu_train.json,${DATA_DIR}/gsm8k_train.json,${DATA_DIR}/arc_challenge_train.json,${DATA_DIR}/cmmlu_train.json"

declare -A TRAIN_FILEs CAL_FILEs TEST_FILEs
# Configure data paths for each dataset
TRAIN_FILEs[gsm8k]="${TRAIN_FILEs_same}"
CAL_FILEs[gsm8k]="${DATA_DIR}/gsm8k_cal.json"
TEST_FILEs[gsm8k]="${DATA_DIR}/gsm8k_test.json"

for ds in "${DATASETS[@]}"; do
  export DATASET="$ds"
  export TRAIN_FILE="${TRAIN_FILEs[$ds]}"
  export CAL_FILE="${CAL_FILEs[$ds]}"
  export TEST_FILE="${TEST_FILEs[$ds]}"
  export OUTPUT_DIR="${OUTPUT_DIR}"

  for alpha in "${ALPHAS[@]}"; do
    for fmt in "${FORMATS[@]}"; do
      
      echo ">>> DATASET=$ds ALPHA=$alpha DATA_FORMAT=$fmt HELD_OUT_RATIO=$HELD_OUT_RATIO"
      bash "$RUN_SH" "$N_SPLITS" "$TEST_RATIO" "$alpha" "$KNN" "$fmt" "$HELD_OUT_RATIO"
    done
  done
done
