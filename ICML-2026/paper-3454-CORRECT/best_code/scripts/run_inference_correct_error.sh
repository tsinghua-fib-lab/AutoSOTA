#!/bin/bash
# Schema-guided error recognition on CORRECT-Error (paper Table 2).
# Adapted from upstream Automated_FA_3 (see NOTICE).
# Run from the CORRECT/ project root.
#
# CORRECT-Error has two generator-model splits; pick one with SPLIT:
#   SPLIT=gpt5nano  (default)  "Synthesized by GPT-5-Nano" (1,908 trajectories)
#   SPLIT=gpt4omini            "Synthesized by GPT-4o-mini" (318 trajectories)
# Schemata and similarities for both splits are included in data/.
#
# Uses k=5 schemata per query (paper §A.3) over all 7 datasets with the
# Qwen-2.5-7B detector. Override the count with SCHEMA_NUMS if needed.
# The detector runs locally through vLLM; MODEL must be a local model alias
# (e.g. qwen-7b, qwen-72b, llama-8b).
#
# Usage:
#   bash scripts/run_inference_correct_error.sh                   # gpt5nano, k=5
#   SPLIT=gpt4omini bash scripts/run_inference_correct_error.sh   # gpt-4o-mini split
#   MODEL=qwen-72b bash scripts/run_inference_correct_error.sh    # larger detector
#
# Overridable env: SPLIT, MODEL, SCHEMA_NUMS, TENSOR_PARALLEL_SIZE, CUDA_DEVICES, OUTPUT_DIR

set -e

SPLIT=${SPLIT:-gpt5nano}
case "$SPLIT" in
    gpt5nano)
        RESULTS_DIR="data/correct_error_gpt5nano"
        SCHEMATA_DIR="data/schemata_correct_error_gpt5nano"
        SIMILARITIES_DIR="data/similarities_gpt5nano"
        GAIA_NAME="gaia_level1"
        ;;
    gpt4omini)
        RESULTS_DIR="data/correct_error"
        SCHEMATA_DIR="data/schemata_correct_error_gpt4omini"
        SIMILARITIES_DIR="data/similarities"
        GAIA_NAME="gaia"
        ;;
    *)
        echo "Unknown SPLIT='$SPLIT' (use 'gpt5nano' or 'gpt4omini')"; exit 1 ;;
esac

DATASETS=(arc hotpot musique wikimqa math500 mmlu_pro "$GAIA_NAME")
MODEL=${MODEL:-"qwen-7b"}
IFS=' ' read -ra SCHEMA_NUMS <<< "${SCHEMA_NUMS:-5}"   # paper §A.3 main result: k=5
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
CUDA_DEVICES=${CUDA_DEVICES:-"0,1,2,3"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs"}
mkdir -p "$OUTPUT_DIR"

# CORRECT-Error detection runs locally through vLLM only; reject cloud models
# (inference_correct_error.py would otherwise no-op on them).
if [[ "$MODEL" == gpt-* || "$MODEL" == gpt4* || "$MODEL" == gemini-* ]]; then
    echo "Error: MODEL='$MODEL' is a cloud model. CORRECT-Error detection is"
    echo "local-only; use a vLLM alias such as qwen-7b, qwen-72b, or llama-8b."
    exit 1
fi

if [ ! -d "$SCHEMATA_DIR" ]; then
    echo "Error: schemata dir '$SCHEMATA_DIR' not found."
    exit 1
fi

echo "========================================"
echo "Schema-Guided Error Recognition (CORRECT-Error, $SPLIT split)"
echo "========================================"
echo "Detector model:   $MODEL  (local / vLLM)"
echo "Schema counts:    ${SCHEMA_NUMS[*]}   ($([ "${#SCHEMA_NUMS[@]}" = 1 ] && echo 'paper main result' || echo 'ablation sweep'))"
echo "Datasets:         ${DATASETS[*]}"
echo "Output dir:       $OUTPUT_DIR"
echo "========================================"

run_inference() {
    local dataset=$1 num_schemata=$2
    local out="${OUTPUT_DIR}/all_at_once_similarity_${num_schemata}schemata_${MODEL//\//_}_${SPLIT}_${dataset}.txt"

    echo ""
    echo "--- Dataset=$dataset, k=$num_schemata  ($(date +%T)) ---"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python src/inference_correct_error.py \
        --method all_at_once --model "$MODEL" --dataset "$dataset" \
        --results_dir "$RESULTS_DIR" --schemata_dir "$SCHEMATA_DIR" \
        --similarities_dir "$SIMILARITIES_DIR" --num_schemata "$num_schemata" \
        --is_handcrafted true --output_file "$out" \
        --use_vllm --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    echo "ok  $out"
}

TOTAL_RUNS=$((${#DATASETS[@]} * ${#SCHEMA_NUMS[@]}))
CURRENT_RUN=0
START_TIME=$(date +%s)

for num_schemata in "${SCHEMA_NUMS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Processing..."
        run_inference "$dataset" "$num_schemata"
    done
done

DURATION=$(( $(date +%s) - START_TIME ))
echo ""
echo "========================================"
echo "Done in $((DURATION / 60))m $((DURATION % 60))s. Outputs in $OUTPUT_DIR/"
echo "Score with: python src/evaluate.py --tolerance 0 \\"
echo "              --eval_file $OUTPUT_DIR/<out>.txt --data_path $RESULTS_DIR/<dataset>/individual_trajectories"
echo "========================================"
