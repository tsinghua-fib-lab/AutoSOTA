#!/bin/bash
cd /repo/src/experiments
export PYTHONPATH=/repo/src:$PYTHONPATH
for seed in $(seq 1 49); do
    echo "Starting seed $seed at $(date)"
    python3 p_val_perm.py --setting adjacent --model GB --seed $seed 2>&1 | tail -1
    echo "Finished seed $seed at $(date)"
done
echo "All seeds done at $(date)"
