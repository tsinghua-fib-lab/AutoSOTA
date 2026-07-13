#!/bin/bash
# Multi-seed reproduction for LieStoNet Example 1
echo "=== Multi-seed Reproduction ==="
for seed in 42 123; do
    echo ""
    echo "============================== SEED=$seed =============================="
    XLA_FLAGS="--xla_gpu_enable_command_buffer=" stdbuf -oL -eL python3 -u EX1_SDE_repro_clean.py --seed $seed >> /repo/repro_multi_seed.log 2>&1
    echo "SEED=$seed completed with exit code $?"
    echo ""
done
echo "=== ALL SEEDS COMPLETE ==="
