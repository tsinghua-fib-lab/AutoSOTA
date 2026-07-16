#!/bin/bash
# Batch evaluation of HaluEval generation results

# cd to project root if needed
cd "$(dirname "$0")"

echo "=========================================="
echo "HaluEval Evaluation Tasks Starting"
echo "=========================================="

# Define files to evaluate
FILES=(
    "outputs/llm_generation/HaluEval_llama3.1-8b_20260114_113628.json"
    "outputs/llm_generation/HaluEval_ministral-8b_20260114_112810.json"
    "outputs/llm_generation/HaluEval_qwen3-8b_20260114_115355.json"
)

# Evaluate one by one
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo ""
        echo "=========================================="
        echo "Evaluating: $file"
        echo "=========================================="
        python ./tools/evaluate_halueval_batch.py -i "$file" -b 32 -c 4
    else
        echo "⚠️  File does not exist: $file"
    fi
done

echo ""
echo "=========================================="
echo "All evaluation tasks completed!"
echo "=========================================="

# Summary of results
echo ""
echo "📊 Result files:"
for file in "${FILES[@]}"; do
    eval_file="${file%.json}_evaluated.json"
    if [ -f "$eval_file" ]; then
        echo "  ✅ $eval_file"
    fi
done

