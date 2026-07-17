#!/bin/bash
# Precompute trajectory-trajectory embedding similarities for the gpt-5-nano subset.
# Adapted from upstream Automated_FA_3 (see NOTICE) — original: generate_all_similarities_gpt5nano.sh
# Run from the CORRECT/ project root.
#
# Note: GAIA is treated as "gaia_level1" only for gpt-5-nano (per paper Table 2).
#
# Usage: bash scripts/generate_similarities_gpt5nano.sh [model_name]

set -e

MODEL_NAME=${1:-"BAAI/bge-m3"}
RESULTS_DIR="data/correct_error_gpt5nano"
OUTPUT_DIR="data/similarities_gpt5nano"
DATASETS=(arc hotpot musique wikimqa math500 mmlu_pro gaia_level1)

echo "========================================"
echo "Trajectory Similarities (gpt-5-nano)"
echo "========================================"
echo "Embedding model:  $MODEL_NAME"
echo "Input:            $RESULTS_DIR"
echo "Output:           $OUTPUT_DIR"
echo "Datasets:         ${DATASETS[*]}"
echo "========================================"

mkdir -p "$OUTPUT_DIR" logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/similarity_generation_${TIMESTAMP}.log"
echo "Logging to: $LOG_FILE"

python src/generate_trajectory_similarities.py \
    --results_dir "$RESULTS_DIR" \
    --model "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --datasets "${DATASETS[@]}" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Generated similarity files:"
for dataset in "${DATASETS[@]}"; do
    FILE="$OUTPUT_DIR/${dataset}_trajectory_similarities.json"
    if [ -f "$FILE" ]; then
        NUM_ENTRIES=$(python -c "import json; print(len(json.load(open('$FILE'))))" 2>/dev/null || echo "?")
        echo "  ok  $FILE (entries: $NUM_ENTRIES)"
    else
        echo "  --  $FILE (missing)"
    fi
done
echo "========================================"
