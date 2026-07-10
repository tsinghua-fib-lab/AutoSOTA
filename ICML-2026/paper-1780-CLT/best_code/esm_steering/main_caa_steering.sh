#!/bin/bash
set -e 

# Usage:
#   sh main_caa_steering.sh [dataset_name]
#   Example: sh main_caa_steering.sh HIS7_YEAST_Pokusaeva_2019

DMS_NAME_FILTER="${1:-}"

REPO_ROOT="$(dirname "$(pwd)")"
DMS_DATA_DIR="$REPO_ROOT/function_circuit/DMS" 
ESM_PATH="../models/esm2_t6_8M_UR50D.pt"
EVAL_MODELS_DIR="eval_models"  # Directory containing the trained CLT/BlockCLT models to evaluate (e.g., "eval_models_35M" or "eval_models_8M")
MODEL_TAG="8M"
FOLDS="0,1,2,3,4"
SPLIT_TYPE="rand_multiples"
SEED=42  # Match CLT base seed
MAX_MUTATIONS=5
TRIALS=10
ALPHA_MIN=0.1
ALPHA_MAX=5
K=25
POS_NEG_PROP="0.1,all"  # Empty = use all sequences with bin=1 (pos) and bin=0 (neg). Set to comma-separated values between 0-1 or "all" (e.g., "0.1,0.2,0.3,all" for >=90th, >=80th, >=70th percentile bin=1, and "all" for all bin=1/bin=0)

echo "========================================"
echo " [Setup] Configuration - CAA STEERING"
echo "========================================"
echo "  > DMS Data Directory:   $DMS_DATA_DIR"
echo "  > Eval Models Dir:       $EVAL_MODELS_DIR"
echo "  > ESM Weights:           $ESM_PATH"
echo "  > Split Type:            $SPLIT_TYPE"
echo "  > Folds:                 $FOLDS"
echo "  > Seed:                  $SEED"
echo "  > Trials:                $TRIALS"
echo "  > Alpha Range:           [$ALPHA_MIN, $ALPHA_MAX] ($K steps)"
if [ -z "$POS_NEG_PROP" ]; then
    echo "  > Pos/Neg Prop:          All (bin=1 for pos, bin=0 for neg)"
else
    echo "  > Pos/Neg Prop:          $POS_NEG_PROP (comma-separated, >= (100-X)th percentile bin=1, <= Xth percentile bin=0)"
fi
echo "  > Similarity Filter:      DISABLED (using only mutation constraint)"
echo "========================================"

echo ""
echo " [Running] CAA Steering"
cd "$(dirname "$0")"

# Create log file (with dataset name if filtering)
if [ -n "$DMS_NAME_FILTER" ]; then
    LOG_FILE="caa_steering_log_${MODEL_TAG}_${DMS_NAME_FILTER}.txt"
else
    LOG_FILE="caa_steering_log_${MODEL_TAG}.txt"
fi
echo "Logging to: $LOG_FILE"
if [ -n "$DMS_NAME_FILTER" ]; then
    echo "  > Dataset Filter: $DMS_NAME_FILTER" | tee -a "$LOG_FILE"
fi

python run_caa_steering.py \
    --dms_dir "$DMS_DATA_DIR" \
    --output_dir "results_caa_steering" \
    --eval_models_dir "$EVAL_MODELS_DIR" \
    --esm_weights "$ESM_PATH" \
    --split_type "$SPLIT_TYPE" \
    --folds "$FOLDS" \
    --seed "$SEED" \
    --trials "$TRIALS" \
    --alpha_min "$ALPHA_MIN" \
    --alpha_max "$ALPHA_MAX" \
    --k "$K" \
    --max_mutations "$MAX_MUTATIONS" \
    --disable_similarity_filter \
    ${POS_NEG_PROP:+--pos_neg_prop "$POS_NEG_PROP"} \
    ${DMS_NAME_FILTER:+--dms_name_filter "$DMS_NAME_FILTER"} \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Pipeline Complete."
echo "Full log saved to: $LOG_FILE"
