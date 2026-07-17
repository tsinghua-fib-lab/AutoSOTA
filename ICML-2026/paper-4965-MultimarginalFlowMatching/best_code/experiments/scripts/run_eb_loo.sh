#!/bin/bash
# Reproduce EB leave-one-out experiments (WLF paper comparison, Table 1).
#
# Runs 3 folds (holdout t1, t2, t3) for W2 (OT-coupled, 768d/8L) and/or
# W2Inf (no OT coupling, 256d/4L).
#
# Usage:
#   ./experiments/scripts/run_eb_loo.sh [w2|w2inf|all] [date_tag] [--parallel]
#
# Options:
#   w2|w2inf|all  Which method to run (default: all)
#   date_tag      Tag for results directory (default: current date, e.g. 25Mar26)
#   --parallel    Run all 3 folds in parallel instead of sequentially
#
# Expected results (W1 on held-out marginal, normalized space):
#
#   W2 (768d/8L, OT-coupled):    avg W1 ≈ 0.643
#   W2Inf (256d/4L, no OT):      avg W1 ≈ 0.748
#   OT-CFM (WLF paper baseline): avg W1 = 0.822
#   WLF-OT (WLF paper):          avg W1 = 0.641

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PARALLEL=false
DATE_TAG=""
METHOD="all"
for arg in "$@"; do
    if [[ "$arg" == "--parallel" ]]; then
        PARALLEL=true
    elif [[ "$arg" == "w2" || "$arg" == "w2inf" || "$arg" == "all" ]]; then
        METHOD="$arg"
    elif [[ -z "$DATE_TAG" ]]; then
        DATE_TAG="$arg"
    fi
done
DATE_TAG="${DATE_TAG:-$(date +%d%b%y)}"

SAVE_BASE="results/${DATE_TAG}"

echo "============================================================"
echo "EB Leave-One-Out Experiment (WLF Paper Comparison)"
echo "============================================================"
echo "Method:      $METHOD"
echo "Date tag:    $DATE_TAG"
echo "Results dir: $SAVE_BASE"
echo "Mode:        $([ "$PARALLEL" = true ] && echo "parallel" || echo "sequential")"
echo "============================================================"
echo ""

run_fold() {
    local method=$1
    local fold=$2
    local config tag

    if [[ "$method" == "w2" ]]; then
        config="configs/singlecell/eb_loo_fold${fold}.json"
        tag="w2_fold${fold}"
    else
        config="configs/singlecell/eb_loo_w2inf_fold${fold}.json"
        tag="w2inf_fold${fold}"
    fi

    if [[ ! -f "$config" ]]; then
        echo "ERROR: Config not found: $config"
        return 1
    fi

    echo "--- ${method} fold ${fold}: holdout t${fold} (config: $config) ---"
    python experiments/train.py \
        --dataset singlecell \
        --config "$config" \
        --save-dir "${SAVE_BASE}" \
        --tag "$tag"
    echo "--- ${method} fold ${fold} complete ---"
    echo ""
}

run_method() {
    local method=$1
    echo "======== Running $method ========"
    if [[ "$PARALLEL" == true ]]; then
        PIDS=()
        for fold in 1 2 3; do
            run_fold "$method" "$fold" &
            PIDS+=($!)
            sleep 2
        done
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
    else
        for fold in 1 2 3; do
            run_fold "$method" "$fold"
        done
    fi
    echo "======== $method complete ========"
    echo ""
}

if [[ "$METHOD" == "w2" || "$METHOD" == "all" ]]; then
    run_method "w2"
fi

if [[ "$METHOD" == "w2inf" || "$METHOD" == "all" ]]; then
    run_method "w2inf"
fi

echo "============================================================"
echo "All runs complete! Results saved to: $SAVE_BASE"
echo "============================================================"
