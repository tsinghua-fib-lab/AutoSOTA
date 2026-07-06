#!/bin/bash
# Run DNN-family baselines on RelBench medium-table datasets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH}:src:."

DATA_DIR_ROOT="datasets/fit-medium-table"
SCRIPT_PATH="scripts/relbench/dnn_baseline_table_data.py"
OUTPUT_DIR="run_outputs/data/relbench/baselines/deep_tabular"
mkdir -p "${OUTPUT_DIR}"

# List of datasets to test
DATA_LIST=(
    "avito-user-clicks"
    "avito-ad-ctr"
    "event-user-repeat"
    "event-user-attendance"
    "ratebeer-beer-positive"
    "ratebeer-place-positive"
    "ratebeer-user-active"
    "trial-site-success"
    "trial-study-outcome"
    "hm-item-sales"
    "hm-user-churn"
)

# Model types to test
MODELS=("MLP" "FTTrans" "ResNet" "DFM" "TabM")


# Run a single deep tabular baseline on one dataset.
run_dnn_baseline() {
    local dataset=$1
    local model=$2
    local data_dir="${DATA_DIR_ROOT}/${dataset}"

    python "${SCRIPT_PATH}" \
        --data_dir "${data_dir}" \
        --model "${model}" \
        --log_dir "${OUTPUT_DIR}"
}

# Loop through all datasets
for dataset in "${DATA_LIST[@]}"; do

    for model in "${MODELS[@]}"; do
        run_dnn_baseline "${dataset}" "${model}"
    done

    echo ""
done

echo "All DNN baseline tests completed!"
