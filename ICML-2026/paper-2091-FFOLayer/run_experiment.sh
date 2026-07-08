#!/bin/bash
# Reproduction experiment for paper 2091
# Runs ffocp_eq on synthetic DFL task with specified parameters

cd /repo
export PYTHONPATH=/repo:$PYTHONPATH

METHOD="ffocp_eq"
YDIM=800
EPOCHS=4
BATCH_SIZE=32
BACKWARD_EPS=1e-6
DEVICE="cpu"

SEEDS=${1:-"0 1 2"}
echo "Running experiment: method=$METHOD, ydim=$YDIM, epochs=$EPOCHS, batch_size=$BATCH_SIZE, backward_eps=$BACKWARD_EPS, seeds=[$SEEDS]"

for SEED in $SEEDS; do
    echo ""
    echo "===== Seed $SEED ====="
    echo "Start: $(date)"
    python3 synthetic_task/main_synthetic.py \
        --method $METHOD \
        --ydim $YDIM \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --device $DEVICE \
        --backward_eps $BACKWARD_EPS
    echo "End: $(date)"
    
    # Print results
    RESULT_FILE="/synthetic_results_${BATCH_SIZE}/${METHOD}/${METHOD}_ydim${YDIM}_lr0.001_seed${SEED}_backwardTol${BACKWARD_EPS}.csv"
    if [ -f "$RESULT_FILE" ]; then
        echo "Results for seed $SEED:"
        cat "$RESULT_FILE"
    fi
done

echo ""
echo "===== All seeds completed ====="
