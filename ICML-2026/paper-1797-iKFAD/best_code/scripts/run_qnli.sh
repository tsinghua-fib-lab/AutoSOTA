#!/bin/bash

# =================================================================
# Usage: bash run_qnli.sh
# This runs 5 QNLI experiments in PARALLEL, each on a dedicated GPU.
# =================================================================

# --- Self-Nohup Logic ---
if [ -t 1 ]; then
    echo "Launching QNLI parallel experiments in background..."
    echo "Logs will be written to: qnli/global_cdvcd0_seed1_10.log (and qnli/logs/*.log)"
    
    nohup bash "$0" "$@" > qnli/global.log 2>&1 &
    
    echo "Background process ID (PID): $!"
    echo "You can close this terminal now."
    exit 0
fi

# =================================================================
# Main Logic
# =================================================================

SEEDS="1 2 3 4 5 6 7 8 9 10"

echo "Starting QNLI Parallel Experiments at $(date)"
echo "------------------------------------------------"

# --- Helper Function to Run 3 Seeds on 1 GPU ---
run_on_gpu() {
    local OPT_NAME=$1
    local GPU_ID=$2
    shift 2 # Remove first 2 args, keep the rest as python flags
    
    local LOG_FILE="qnli/logs_besthparams/${OPT_NAME}.log"
    local DEVICE="cuda:${GPU_ID}"

    echo "[GPU ${GPU_ID}] Starting ${OPT_NAME} (Seeds: $SEEDS)... Log: $LOG_FILE"
    
    # Clear the log file before starting the first seed
    > "$LOG_FILE"

    for seed in $SEEDS; do
        python3 qnli/train.py \
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

# # 1. iKFAD -> GPU 0
# run_on_gpu "ikfad" 0 \
#     --lr 0.04132653104317665 \
#     --gamma 1.360825242358726e-08 \
#     --alpha 2.8041726079801736 \
#     --mu 6.712205546958609e-09 \
#     --epochs 2 &

# # 1. iKFAD0 -> GPU 0
run_on_gpu "ikfad0" 0 \
    --lr 0.044796758137489415 \
    --gamma 0. \
    --alpha 0.0016741631151323088 \
    --mu 2.772426437221421e-07 \
    --epochs 2 &

# # 2. CD -> GPU 1
# run_on_gpu "cd" 6 \
#     --lr 0.078912635018664 \
#     --gamma 2.782838168232168 \
#     --c 112262624.35478136 \
#     --epochs 2 &

# run_on_gpu "cd0" 7 \
#     --lr 0.031440324711587087 \
#     --gamma 0.0 \
#     --c 189769097.4797418 \
#     --epochs 2 &

# # 3. CADAM -> GPU 3
# run_on_gpu "cadam" 3 \
#     --lr 0.00016044362291342404 \
#     --gamma 9.662966777273317 \
#     --alpha 0.0025752546928434973 \
#     --c 22754.577654435445 \
#     --epochs 2 &

#  3. CADAM0 -> GPU 3
# run_on_gpu "cadam0" 7 \
#     --lr 1.736183853226889e-05 \
#     --gamma 0. \
#     --alpha 0.033206767043794286 \
#     --c 15228940.534624306 \
#     --epochs 2 &


# # 4. mSGD -> GPU 5
# run_on_gpu "msgd" 4 \
#     --lr 0.001514080018177748 \
#     --momentum 0.8957512756657571 \
#     --epochs 2 &   

# # 5. Adam -> GPU 6
# run_on_gpu "adam" 5 \
#     --lr 3.524943264195254e-05  \
#     --beta1 0.8553205203248113 \
#     --beta2 0.932701313016893 \
#     --epochs 2 &

# =================================================================
# Wait for all background jobs to finish
# =================================================================
echo "All tasks launched. Waiting for completion..."
wait

echo "------------------------------------------------"
echo "All QNLI parallel experiments completed at $(date)"