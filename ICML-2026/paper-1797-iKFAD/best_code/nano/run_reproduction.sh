#!/bin/bash
# Reproduce iKFAD Test Loss on GPT2-Nano (Shakespeare)
set -euo pipefail
cd "$(dirname "$0")"

ALL_SEEDS=false
if [ "${1:-}" = "--all-seeds" ]; then
    ALL_SEEDS=true
fi

if $ALL_SEEDS; then
    echo "Running 10 seeds (1337-1346)..."
    RESULTS=()
    for seed in $(seq 1337 1346); do
        OUT=$(CUDA_VISIBLE_DEVICES=0 python3 eval.py --seed "$seed" --max-iters 5001 2>/dev/null)
        LOSS=$(echo "$OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['test_loss'])")
        RESULTS+=("$LOSS")
        echo "Seed $seed: $LOSS"
    done
    echo "${RESULTS[@]}" > /tmp/repro_results.txt
    python3 -c "
import math
with open('/tmp/repro_results.txt') as f:
    vals = [float(x) for x in f.read().strip().split()]
mean = sum(vals)/len(vals)
std = math.sqrt(sum((x-mean)**2 for x in vals)/(len(vals)-1))
print(f'Test Loss: {mean:.4f} +/- {std:.4f}  n={len(vals)}')
"
else
    CUDA_VISIBLE_DEVICES=0 python3 eval.py --seed 1337 --max-iters 5001 2>/dev/null
fi
