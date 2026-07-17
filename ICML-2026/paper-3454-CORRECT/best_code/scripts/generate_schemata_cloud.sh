#!/bin/bash
# Generate error schemata via cloud APIs (OpenAI GPT or Google Gemini).
# Produces the error_schemata.txt files under data/schemata*/.
#
# Adapted from upstream Automated_FA_3 (see NOTICE) — original: error_template_generator_cloud.py
# Run from the CORRECT/ project root.
#
# Usage:
#   OPENAI_API_KEY=sk-... bash scripts/generate_schemata_cloud.sh [model] [variant]
#
# Arguments:
#   model    OpenAI / Gemini model name (default: gpt-4o)
#   variant  "gpt4omini" or "gpt5nano" — which CORRECT-Error subset to run on.
#            Defaults to gpt4omini (writes data/schemata_correct_error_gpt4omini/).

set -e

MODEL=${1:-"gpt-4o"}
VARIANT=${2:-"gpt4omini"}

case "$VARIANT" in
    gpt5nano)
        INPUT_DIR="data/correct_error_gpt5nano"
        OUTPUT_DIR="data/schemata_correct_error_gpt5nano"
        DATASETS=(arc hotpot musique wikimqa math500 mmlu_pro gaia_level1)
        ;;
    gpt4omini)
        INPUT_DIR="data/correct_error"
        OUTPUT_DIR="data/schemata_correct_error_gpt4omini"
        DATASETS=(arc hotpot musique wikimqa math500 mmlu_pro gaia)
        ;;
    *)
        echo "Unknown VARIANT='$VARIANT' (use 'gpt4omini' or 'gpt5nano')"; exit 1 ;;
esac

echo "======================================"
echo "Error Schema Generation (Cloud / $MODEL)"
echo "======================================"
echo "Variant:     $VARIANT"
echo "Input:       $INPUT_DIR"
echo "Output:      $OUTPUT_DIR"
echo "Datasets:    ${DATASETS[*]}"
echo "======================================"

python src/error_schema_generator_cloud.py \
    --model "$MODEL" \
    --results_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --datasets "${DATASETS[@]}"

echo "======================================"
echo "Schema generation complete."
echo "======================================"
