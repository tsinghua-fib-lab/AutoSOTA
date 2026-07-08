#!/bin/bash

# =================================================================
# Usage: bash scripts/run_vit.sh
# This runs ViT experiments in PARALLEL, each on a dedicated GPU.
# =================================================================

# --- Self-Nohup Logic ---
if [ -t 1 ]; then
    echo "Launching ViT parallel experiments in background..."
    echo "Logs will be written to: vit/global.log (and vit/logs_besthparams/*.log)"
    
    nohup bash "$0" "$@" > vit/global.log 2>&1 &
    
    echo "Background process ID (PID): $!"
    echo "You can close this terminal now."
    exit 0
fi

# =================================================================
# Main Logic
# =================================================================

mkdir -p vit/logs_besthparams

echo "Starting VIT Parallel Experiments (10 Seeds per optimizer) at $(date)"
echo "------------------------------------------------"

# --- Helper Function to Run 10 Seeds on 1 GPU ---
run_on_gpu() {
    local OPT_NAME=$1
    local GPU_ID=$2
    shift 2 # Remove first 2 args, keep the rest as python flags
    
    local LOG_FILE="vit/logs_besthparams/${OPT_NAME}.log"
    local DEVICE="cuda:${GPU_ID}"

    echo "[GPU ${GPU_ID}] Starting ${OPT_NAME}... Log: $LOG_FILE"
    
    # Call run_best_seeds.py which internally loops 0-9
    python vit/run_best_seeds.py \
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
run_on_gpu "ikfad" 0 \
    --lr 0.0899069592965965 \
    --gamma 0.045371333867075814 \
    --alpha 0.1010979726904078 \
    --mu 6.293766175067193e-06 \
    --epochs 50 &

# iKFAD0
run_on_gpu "ikfad0" 7 \
    --lr 0.07248512251690599 \
    --gamma 0. \
    --alpha 0.08985756672252108 \
    --mu 1.264346301276473e-05 \
    --epochs 50 &

# CD 
run_on_gpu "cd" 0 \
    --lr 0.19983382612713718 \
    --gamma 0.0022049346829463405 \
    --c 655446.008302313 \
    --epochs 50 &

# CD0
run_on_gpu "cd0" 1 \
    --lr 0.11707460172563208 \
    --gamma 0 \
    --c 535134.420880812 \
    --epochs 50 &

# CADAM
run_on_gpu "cadam" 6 \
    --lr 0.001967286159417574 \
    --gamma 4.468696227637524 \
    --alpha 0.9137670875238709 \
    --c 24314309.96094943 \
    --epochs 50 &

# CADAM0
run_on_gpu "cadam0" 5 \
    --lr 0.0008172766761937721 \
    --gamma 0. \
    --alpha 0.17253661877559273 \
    --c 8838234638.602621 \
    --epochs 50 &

# mSGD
run_on_gpu "msgd" 3 \
    --lr 0.021327084768258682 \
    --momentum 0.8743927942685524 \
    --epochs 50 &

# Adam
run_on_gpu "adam" 4 \
    --lr 0.0005531145824647459 \
    --beta1 0.8608226206342482 \
    --beta2 0.8852058634949964 \
    --epochs 50 &

# =================================================================
# Wait for all background jobs to finish
# =================================================================
echo "All tasks launched. Waiting for completion..."
wait

echo "------------------------------------------------"
echo "All ViT parallel experiments completed at $(date)"