#!/bin/bash
# Runs evaluation on all 8 commonsense datasets using MAS model
# Parallelizes on 2 GPUs: first 4 datasets on GPU 0, last 4 on GPU 1

BASE_MODEL="/models/Llama-3.2-1B"
LORA_WEIGHTS="/repo/trained_models_and_results/Llama-3.2-1B_epoch3_MAS"
RESULTS_DIR="/tmp/eval_results_$$"
mkdir -p $RESULTS_DIR

cd /repo

echo "=== Starting parallel evaluation on all datasets ==="

# Run 4 datasets on GPU 0
eval_gpu0() {
    export CUDA_VISIBLE_DEVICES=0
    for dataset in boolq piqa social_i_qa hellaswag; do
        case $dataset in
            *) bs=1 ;;
        esac
        echo "GPU0: Evaluating $dataset (batch_size=$bs)..."
        output=$(python3 mas_eval.py --dataset $dataset --base_model $BASE_MODEL --lora_weights $LORA_WEIGHTS --batch_size $bs 2>&1)
        accuracy=$(echo "$output" | grep -E "^${dataset}:" | tail -1 | sed 's/.*: //' | sed 's/%//')
        echo "$accuracy" > "$RESULTS_DIR/${dataset}"
        echo "GPU0: $dataset = ${accuracy}%"
    done
}

# Run 4 datasets on GPU 1
eval_gpu1() {
    export CUDA_VISIBLE_DEVICES=1
    for dataset in winogrande ARC-Challenge ARC-Easy openbookqa; do
        bs=1
        echo "GPU1: Evaluating $dataset (batch_size=$bs)..."
        output=$(python3 mas_eval.py --dataset $dataset --base_model $BASE_MODEL --lora_weights $LORA_WEIGHTS --batch_size $bs 2>&1)
        accuracy=$(echo "$output" | grep -E "^${dataset}:" | tail -1 | sed 's/.*: //' | sed 's/%//')
        echo "$accuracy" > "$RESULTS_DIR/${dataset}"
        echo "GPU1: $dataset = ${accuracy}%"
    done
}

# Run both in parallel
eval_gpu0 &
PID0=$!
eval_gpu1 &
PID1=$!

# Wait for both
wait $PID0
wait $PID1

echo ""
echo "=== FINAL RESULTS ==="
total=0
count=0
for dataset in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    acc=$(cat "$RESULTS_DIR/${dataset}" 2>/dev/null || echo "0")
    case $dataset in
        boolq) label="boolq" ;;
        piqa) label="piqa" ;;
        social_i_qa) label="siqa" ;;
        hellaswag) label="hellaswag" ;;
        winogrande) label="winogrande" ;;
        ARC-Challenge) label="arc_c" ;;
        ARC-Easy) label="arc_e" ;;
        openbookqa) label="obqa" ;;
    esac
    echo "${label}: ${acc}%"
    total=$(python3 -c "print($total + $acc)")
    count=$((count + 1))
done

avg=$(python3 -c "print(round($total / $count, 2))")
echo "average: ${avg}%"

rm -rf $RESULTS_DIR
