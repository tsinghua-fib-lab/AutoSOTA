#!/bin/bash
# Generate error schemata for the gpt-5-nano subset of CORRECT-Error.
# Adapted from upstream Automated_FA_3 (see NOTICE) — original: generate_all_templates_gpt5nano.sh
# Run from the CORRECT/ project root.
#
# Note: GAIA is treated as "gaia_level1" only for gpt-5-nano (per paper Table 2).
#
# Usage: bash scripts/generate_schemata_gpt5nano.sh [model_name] [tensor_parallel_size]

set -e

MODEL_NAME=${1:-"Qwen/Qwen2.5-72B-Instruct"}
TENSOR_PARALLEL_SIZE=${2:-8}
INPUT_DIR="data/correct_error_gpt5nano"
OUTPUT_DIR="data/schemata_correct_error_gpt5nano"
BATCH_SIZE=64

echo "======================================"
echo "Error Schema Generation (gpt-5-nano)"
echo "======================================"
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Input Directory: $INPUT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "======================================"

python src/error_schema_generator.py \
    --model_name "$MODEL_NAME" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --results_dir "$INPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --datasets arc hotpot musique wikimqa math500 mmlu_pro gaia_level1

echo "======================================"
echo "Schema generation complete."
echo "======================================"
