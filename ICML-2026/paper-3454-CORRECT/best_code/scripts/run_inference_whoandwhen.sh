#!/bin/bash
# Schema-guided error recognition on the Who&When benchmark (paper Table 1).
# Adapted from upstream Automated_FA_3 (see NOTICE).
# Run from the CORRECT/ project root.
#
# Defaults reproduce the paper recipe:
#   - baseline (k=0) + CORRECT with the paper's per-subset schema count
#     (Hand-Crafted: k=10, Algorithm-Generated: k=1; paper §A.3)
#   - cloud detectors (gpt-*/gemini-*) go through the paper-byte-identical
#     path (--cloud_paper_path: utils_cloud_parallel baseline + cloud_model
#     CORRECT, THOUGHT TEMPLATE wording), with reasoning_effort=medium for gpt-5
#   - local detectors (qwen-*, etc.) run through vLLM
#
# Prerequisites:
#   1. Who&When data:   bash scripts/download_whoandwhen.sh
#   2. Schemata under data/schemata_whoandwhen/<subset>/ (GPT-5, paper §A.3) are included.
#   3. Trajectory similarities under data/similarities_whoandwhen/ are included.
#      To regenerate, run src/generate_trajectory_similarities.py on the
#      Who&When subsets (the generate_similarities*.sh scripts cover CORRECT-Error).
#
# Usage:
#   bash scripts/run_inference_whoandwhen.sh                              # qwen-7b (local)
#   MODEL=gpt-5 OPENAI_API_KEY=sk-... bash scripts/run_inference_whoandwhen.sh
#   MODEL=gemini-2.5-flash GOOGLE_APPLICATION_CREDENTIALS=/path/key.json \
#     GOOGLE_CLOUD_PROJECT=my-proj bash scripts/run_inference_whoandwhen.sh
#
# Overridable env:
#   MODEL, MAX_TOKENS, K_HC, K_AG, RUN_BASELINE, BATCH, WORKERS,
#   TENSOR_PARALLEL_SIZE, CUDA_DEVICES, OPENAI_REASONING_EFFORT, OUTPUT_DIR

set -e

MODEL=${MODEL:-"qwen-7b"}
MAX_TOKENS=${MAX_TOKENS:-8192}
K_HC=${K_HC:-10}                 # paper §A.3: Hand-Crafted uses k=10
K_AG=${K_AG:-1}                  # paper §A.3: Algorithm-Generated uses k=1
RUN_BASELINE=${RUN_BASELINE:-1}  # also run k=0 baseline (LLM-as-a-judge)
BATCH=${BATCH:-30}
WORKERS=${WORKERS:-10}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}
# gpt-5 family defaults to medium reasoning effort (paper / server default).
export OPENAI_REASONING_EFFORT=${OPENAI_REASONING_EFFORT:-medium}

SCHEMATA_BASE="data/schemata_whoandwhen"
SIMILARITIES_BASE="data/similarities_whoandwhen"
OUTPUT_DIR=${OUTPUT_DIR:-"outputs_whoandwhen"}
mkdir -p "$OUTPUT_DIR"

is_cloud() { [[ "$1" == gpt-* || "$1" == gpt4* || "$1" == gemini-* ]]; }

echo "========================================"
echo "Schema-Guided Error Recognition (Who&When)"
echo "========================================"
echo "Detector model:   $MODEL  ($(is_cloud "$MODEL" && echo 'cloud / --cloud_paper_path' || echo 'local / vLLM'))"
echo "Schemata path:    $SCHEMATA_BASE/<subset>/error_schemata.txt"
echo "Recipe:           baseline=$([ "$RUN_BASELINE" = 1 ] && echo yes || echo no), HC k=$K_HC, AG k=$K_AG"
echo "Output dir:       $OUTPUT_DIR"
echo "========================================"

run_one() {
    local subset=$1 num_schemata=$2
    local is_hc="False"; [ "$subset" = "Hand-Crafted" ] && is_hc="True"
    local schemata_file="$SCHEMATA_BASE/$subset/error_schemata.txt"
    local similarities_file="$SIMILARITIES_BASE/${subset}_trajectory_similarities.json"
    local subset_slug=$(echo "$subset" | tr '[:upper:]-' '[:lower:]_')
    local out="${OUTPUT_DIR}/all_at_once_${num_schemata}schemata_${MODEL//\//_}_${subset_slug}.txt"

    if [ ! -f "$schemata_file" ]; then echo "skip $subset: $schemata_file not found"; return 0; fi
    if [ "$num_schemata" -gt 0 ] && [ ! -f "$similarities_file" ]; then
        echo "skip $subset k=$num_schemata: $similarities_file not found"
        echo "  (shipped under data/similarities_whoandwhen/; regenerate via src/generate_trajectory_similarities.py)"; return 0
    fi

    echo ""
    echo "--- $subset, k=$num_schemata  ($(date +%T)) ---"

    local args=(
        --method all_at_once --model "$MODEL"
        --directory_path "data/whoandwhen/${subset}"
        --is_handcrafted "$is_hc"
        --schemata_file "$schemata_file"
        --schema_selection similarity
        --similarities_file "$similarities_file"
        --num_schemata "$num_schemata"
        --max_tokens "$MAX_TOKENS"
        --output_file "$out"
    )
    if is_cloud "$MODEL"; then
        args+=( --cloud_paper_path --batch_size "$BATCH" --max_workers "$WORKERS" )
        python src/inference_whoandwhen.py "${args[@]}"
    else
        args+=( --use_vllm --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" )
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python src/inference_whoandwhen.py "${args[@]}"
    fi
    echo "ok  $out"
}

START_TIME=$(date +%s)

# Hand-Crafted: baseline (k=0) + paper k=K_HC
[ "$RUN_BASELINE" = 1 ] && run_one "Hand-Crafted" 0
run_one "Hand-Crafted" "$K_HC"

# Algorithm-Generated: baseline (k=0) + paper k=K_AG
[ "$RUN_BASELINE" = 1 ] && run_one "Algorithm-Generated" 0
run_one "Algorithm-Generated" "$K_AG"

DURATION=$(( $(date +%s) - START_TIME ))
echo ""
echo "========================================"
echo "Done in $((DURATION / 60))m $((DURATION % 60))s. Outputs in $OUTPUT_DIR/"
echo "Score with: python src/evaluate.py --tolerance 0 \\"
echo "              --eval_file $OUTPUT_DIR/<out>.txt --data_path data/whoandwhen/<subset>"
echo "========================================"
