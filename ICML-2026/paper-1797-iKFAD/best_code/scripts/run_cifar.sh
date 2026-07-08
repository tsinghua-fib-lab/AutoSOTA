#!/bin/bash

# =================================================================
# Usage: bash run_experiments.sh
# This runs 5 optimizers in PARALLEL, each on a dedicated GPU.
# =================================================================

# --- Self-Nohup Logic ---
if [ -t 1 ]; then
    echo "Launching parallel experiments in background..."
    echo "Logs will be written to: cifar/global.log (and cifar/logs/*.log)"
    nohup bash "$0" "$@" > cifar/global.log 2>&1 &
    echo "Background process ID (PID): $!"
    echo "You can close this terminal now."
    exit 0
fi

# =================================================================
# Main Logic
# =================================================================

SEEDS="1 2 3 4 5 6 7 8 9 10"
mkdir -p cifar/logs

echo "Starting Parallel Experiments at $(date)"
echo "------------------------------------------------"

# --- Helper Function to Run 5 Seeds on 1 GPU ---
run_on_gpu() {
    local OPT_NAME=$1
    local GPU_ID=$2
    shift 2 # Remove first 2 args (Name and GPU), keep the rest as python flags
    
    local LOG_FILE="cifar/logs/${OPT_NAME}.log"
    local DEVICE="cuda:${GPU_ID}"

    echo "[GPU ${GPU_ID}] Starting ${OPT_NAME} (Seeds: $SEEDS)... Log: $LOG_FILE"
    
    # Clear the log file before starting the first seed
    > "$LOG_FILE"

    for seed in $SEEDS; do
        python cifar/train.py \
            --optimizer "$OPT_NAME" \
            --seed "$seed" \
            --device "$DEVICE" \
            "$@" \
            >> "$LOG_FILE" 2>&1
    done

    echo "[GPU ${GPU_ID}] Finished ${OPT_NAME}."
}

# =================================================================
# Launch Jobs (Note the '&' at the end of each block)
# =================================================================

# 1. iKFAD -> GPU 0
run_on_gpu "ikfad" 0 \
    --lr 0.0973 \
    --gamma 0.40 \
    --alpha 0.402 \
    --mu 1.87 \
    --epochs 40 &

# 2. CD -> GPU 1
run_on_gpu "cd" 1 \
    --lr 0.0448 \
    --gamma 0.17 \
    --c 56814.47 \
    --epochs 40 &

# 3. CADAM -> GPU 3 (Skipping 2)
run_on_gpu "cadam" 3 \
    --lr 0.0037 \
    --gamma 3.55 \
    --alpha 0.005 \
    --c 82552.90 \
    --epochs 40 &

# 4. mSGD -> GPU 5 (Skipping 4)
run_on_gpu "msgd" 5 \
    --lr 0.0260 \
    --momentum 0.86 \
    --epochs 40 &

# 5. Adam -> GPU 6
run_on_gpu "adam" 6 \
    --lr 0.0008 \
    --beta1 0.97 \
    --beta2 0.99 \
    --epochs 40 &

# =================================================================
# Wait for all background jobs to finish
# =================================================================
echo "All tasks launched. Waiting for completion..."
wait

echo "------------------------------------------------"
echo "All parallel experiments completed at $(date)"