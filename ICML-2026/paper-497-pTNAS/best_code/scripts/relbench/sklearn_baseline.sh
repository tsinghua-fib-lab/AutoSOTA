#!/bin/bash
# Run Logistic Regression and Random Forest on RelBench medium-table datasets.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH}:src:."

DATA_DIR_ROOT="datasets/fit-medium-table"
SCRIPT_PATH="scripts/relbench/sklearn_baseline.py"
LOG_DIR="run_outputs/data/relbench/baselines/classical/logs"
mkdir -p "${LOG_DIR}"

# List of datasets to test
DATA_LIST=(
    # "avito-user-clicks"
    # "avito-ad-ctr"
    # "event-user-repeat"
    # "event-user-attendance"
    # "ratebeer-beer-positive"
    # "ratebeer-place-positive"
    # "ratebeer-user-active"
    # "trial-site-success"
    # "trial-study-outcome"
    # "hm-item-sales"
    "hm-user-churn"
)

# Model types to test
MODELS=("logistic" "randomforest")

# Run a single sklearn baseline on one dataset.
run_sklearn_baseline() {
    local dataset=$1
    local model=$2
    local data_dir="${DATA_DIR_ROOT}/${dataset}"
    local log_file="${LOG_DIR}/sklearn_${model}_${dataset}.log"

    python "${SCRIPT_PATH}" \
        --data_dir "${data_dir}" \
        --method "${model}" \
        > "${log_file}" 2>&1
}

# Loop through all datasets
for dataset in "${DATA_LIST[@]}"; do

    for model in "${MODELS[@]}"; do
        run_sklearn_baseline "${dataset}" "${model}"
    done

    echo ""
done

echo "All sklearn baseline tests completed!"
