#!/bin/bash

# =================================================================
# Usage: bash scripts/run_sst2.sh
# This runs SST-2 experiments in PARALLEL, each on a dedicated GPU.
# =================================================================

# --- Self-Nohup Logic ---
if [ -t 1 ]; then
    echo "Launching SST-2 parallel experiments in background..."
    echo "Logs will be written to: sst2/global.log (and sst2/logs_besthparams_2epochs/*.log)"
    
    nohup bash "$0" "$@" > sst2/global.log 2>&1 &
    
    echo "Background process ID (PID): $!"
    echo "You can close this terminal now."
    exit 0
fi

# =================================================================
# Main Logic
# =================================================================

# mkdir -p sst2/logs_besthparams

echo "Starting SST-2 Parallel Experiments (10 Seeds per optimizer) at $(date)"
echo "------------------------------------------------"

# --- Helper Function to Run 10 Seeds on 1 GPU ---
run_on_gpu() {
    local OPT_NAME=$1
    local GPU_ID=$2
    shift 2 # Remove first 2 args, keep the rest as python flags
    
    local LOG_FILE="sst2/logs_besthparams_2epochs/${OPT_NAME}.log"
    local DEVICE="cuda:${GPU_ID}"

    echo "[GPU ${GPU_ID}] Starting ${OPT_NAME}... Log: $LOG_FILE"
    
    # Call run_best_seeds.py which internally loops 0-9
    python sst2/run_best_seeds.py \
        --optimizer "$OPT_NAME" \
        --device "$DEVICE" \
        "$@" \
        > "$LOG_FILE" 2>&1

    echo "[GPU ${GPU_ID}] Finished ${OPT_NAME}."
}

# =================================================================
# Launch Jobs (Note the '&' at the end of each block)
# =================================================================

# iKFAD
# run_on_gpu "ikfad" 0 \
#     --lr 0.027043 \
#     --gamma 1.28e-06 \
#     --alpha 0.0465 \
#     --mu 3.61e-07 \
#     --epochs 2 &

# iKFAD0
run_on_gpu "ikfad0" 5 \
    --lr 0.016464179405195838 \
    --gamma 0. \
    --alpha 0.05736039355699625 \
    --mu 9.894800041353303e-06 \
    --epochs 2 &

# # CD 
# run_on_gpu "cd" 3 \
#     --lr 0.04223 \
#     --gamma 0.051996 \
#     --c 3.34e7 \
#     --epochs 2 &

# # CD0
# run_on_gpu "cd0" 5 \
#     --lr 0.0363 \
#     --gamma 0 \
#     --c 9.78e6 \
#     --epochs 2 &

# # CADAM
# run_on_gpu "cadam" 6 \
#     --lr 0.0003485 \
#     --gamma 9.2188 \
#     --alpha 0.032325 \
#     --c 0.0237 \
#     --epochs 2 &

# CADAM0
# run_on_gpu "cadam0" 6 \
#     --lr 3.3881783195372295e-05 \
#     --gamma 0. \
#     --alpha 0.02140656025200693 \
#     --c 338638.98570018134 \
#     --epochs 2 &

# mSGD
# run_on_gpu "msgd" 5 \
#     --lr 0.00237 \
#     --momentum 0.8335 \
#     --epochs 2 &

# # Adam
# run_on_gpu "adam" 7 \
#     --lr 3.82e-05 \
#     --beta1 0.9187 \
#     --beta2 0.98254 \
#     --epochs 2 &

# =================================================================
# Wait for all background jobs to finish
# =================================================================
echo "All tasks launched. Waiting for completion..."
wait

echo "------------------------------------------------"
echo "All SST-2 parallel experiments completed at $(date)"