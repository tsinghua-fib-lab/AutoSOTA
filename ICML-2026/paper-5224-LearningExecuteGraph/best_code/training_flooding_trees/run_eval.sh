#!/bin/bash
# Eval script for Message Flooding on Trees
# Iter 6: Reduced hidden_dim = k*1500
set -e

N=${1:-7}
D=${2:-2}
L=${3:-1}
TRIALS=${4:-1}
EPOCHS=${5:-3500}
MAX_MODELS=${6:-300}
DEVICE=${7:-cuda}
OUTDIR=${8:-results}

# Compute hidden_dim as k * 1500 (reduced from k*2000)
K=$((L + 4 * (D + 1) * L))
HIDDEN_DIM=$((K * 2000))

cd "$(dirname "$0")"

# Step 1: Generate test samples (deterministic, fast)
echo "=== Generating test samples ==="
python3 generate_test_samples.py --n "$N" --D "$D" --l "$L"

# Step 2: Generate test cases (deterministic)
echo "=== Generating test cases ==="
python3 generate_test_cases.py --n "$N" --D "$D" --l "$L"

# Step 3: Train ensemble and evaluate
echo "=== Training ensemble (hidden_dim=$HIDDEN_DIM, epochs=$EPOCHS) ==="
python3 train_flooding.py \
    --n "$N" --D "$D" --l "$L" \
    --trials "$TRIALS" --epochs "$EPOCHS" \
    --max_models "$MAX_MODELS" --device "$DEVICE" \
    --hidden_dim "$HIDDEN_DIM" \
    --outdir "$OUTDIR"

# Step 4: Report final metric
echo ""
echo "=== FINAL RESULTS ==="
RESULTS_FILE="$OUTDIR/n${N}_D${D}_l${L}/trial_0/results.csv"
if [ -f "$RESULTS_FILE" ]; then
    echo "--- Full training curve ---"
    cat "$RESULTS_FILE"
    echo ""
    echo "--- Final metrics ---"
    tail -1 "$RESULTS_FILE" | while IFS=',' read -r num_models error sample_acc case_acc; do
        echo "ensemble_size: $num_models"
        echo "mse_error: $error"
        echo "sample_accuracy: $sample_acc"
        echo "case_accuracy: $case_acc"
        echo "METRIC:case_accuracy=$case_acc"
    done
fi
echo "=== EVAL COMPLETE ==="
