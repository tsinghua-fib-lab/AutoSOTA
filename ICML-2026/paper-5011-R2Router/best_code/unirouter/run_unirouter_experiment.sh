#!/bin/bash
# UniRouter Experiment Pipeline
#
# This script demonstrates:
# 1. Training original UniRouter on initial model pool
# 2. Training Uni-R2 on the same pool
# 3. Dynamically adding new (unseen) models without retraining
# 4. Comparing routing performance across methods

set -e

# ============================================================================
# CONFIGURATION - Edit this section
# ============================================================================

# Initial LLM pool (format: "Name|Size|CSVPath")
# These models will be used for initial training
INITIAL_POOL=(
    # "Mistral-7B-Instruct-v0.2|0.20|data/Mistral-7B-Instruct-v0.2.csv"
    # "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    "GLM-4.6|1.75|data/GLM-4.6.csv"
    "gemma-3-4b-it|0.06815|data/gemma-3-4b-it.csv"
    # "gemma-3-1b-it|0.0170375|data/gemma-3-1b-it.csv"
    "gemma-3-270m-it|0.004259375|data/gemma-3-270m-it.csv"
    "Llama-3.1-70B-Instruct|0.40|data/Llama-3.1-70B-Instruct.csv"
    # "Llama-3.2-3B-Instruct|0.02|data/Llama-3.2-3B-Instruct.csv"
    "Qwen2.5-Math-1.5B-Instruct|0.09|data/Qwen2.5-Math-1.5B-Instruct.csv"
    # "Qwen2.5-Math-7B-Instruct|0.35|data/Qwen2.5-Math-7B-Instruct.csv"
    "Qwen3-0.6B|0.0173|data/Qwen3-0.6B.csv"
    # "Qwen3-235B-A22B-Instruct-2507|0.55|data/Qwen3-235B-A22B-Instruct-2507.csv"
    # "Qwen3-Next-80B-A3B-Instruct|0.6|data/Qwen3-Next-80B-A3B-Instruct.csv"
)

# NEW models (format: "Name|Size|CSVPath")
# These models will be added AFTER initial training to test dynamic addition
NEW_MODELS=(
    "Llama-3.2-3B-Instruct|0.02|data/Llama-3.2-3B-Instruct.csv"
    "gemma-3-1b-it|0.0170375|data/gemma-3-1b-it.csv"
    "Mistral-7B-Instruct-v0.2|0.20|data/Mistral-7B-Instruct-v0.2.csv"
    "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    "Qwen2.5-Math-7B-Instruct|0.35|data/Qwen2.5-Math-7B-Instruct.csv"
)

# Uni-R2 Configuration
VAL_SIZE=500                    # Validation set size
N_CLUSTERS=100                  # Number of clusters
USE_CLUSTERING=true             # Use clustering (recommended)

# Token budgets to test (must match dataset!)
# Available in R2-Bench dataset: 10,20,30,40,50,80,100,150,200,300,500,800,1200,2000,4000,unlimited
TOKEN_BUDGETS="10,20,30,40,50,80,100,150,200,300,500,800,1200,2000,4000,9999"  # 9999 = unlimited

# Lambda Distribution - Controls cost-performance tradeoff sampling
# Format: "min,max,num_points" for each segment (segments are concatenated)
# Formula: score = (1-λ)*quality - λ*cost, where λ ∈ [0,1]
# Examples:
#   "0,0.2,20;0.2,1.0,50"  - Default: denser sampling at low lambda (quality-focused)
#   "0,1.0,100"            - Uniform: evenly spaced across full range
#   "0,0.5,80;0.5,1.0,20"  - Dense low lambda, sparse high lambda
LAMBDA_DISTRIBUTION="0,0.001,100;0.001,0.01,100;0.01,0.1,100;0.1,1.0,100"

# QNC Target Accuracy Rate - Controls what percentage of best LLM to target
# 1.0 = 100% of best LLM (default), 0.9 = 90% of best LLM
TARGET_ACCURACY_RATE=0.95

# Output directories (following IID/OOD structure)
CHECKPOINT_DIR="./checkpoints/unirouter"
RESULTS_DIR="./comparison_results/unirouter"

# Activate virtual environment
source .venv/bin/activate

echo "=========================================="
echo "UniRouter Experiment Pipeline"
echo "=========================================="
echo "Initial pool: ${#INITIAL_POOL[@]} models"
echo "New models: ${#NEW_MODELS[@]} models"
echo "Validation size: $VAL_SIZE"
echo "Clusters: $N_CLUSTERS"
echo "Token budgets: $TOKEN_BUDGETS"
echo "=========================================="
echo ""

# Create output directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$RESULTS_DIR"

# ============================================================================
# Step 1: Check if validation set exists
# ============================================================================

echo "Step 1: Checking validation set..."
VAL_SET_FILE="$CHECKPOINT_DIR/validation_set.pkl"

if [ -f "$VAL_SET_FILE" ]; then
    echo "  [✓] Validation set exists: $VAL_SET_FILE"
else
    echo "  [ ] Validation set does not exist - will create during evaluation"
fi

# ============================================================================
# Step 2: Check/Build Original UniRouter
# ============================================================================

echo ""
echo "=========================================="
echo "Step 2: Original UniRouter (Unlimited Tokens Only)"
echo "=========================================="

ORIGINAL_CHECKPOINT="$CHECKPOINT_DIR/original_unirouter.pkl"

if [ -f "$ORIGINAL_CHECKPOINT" ]; then
    echo "  [✓] Original UniRouter checkpoint exists"
    echo "  Checkpoint: $ORIGINAL_CHECKPOINT"
else
    echo "  [ ] Original UniRouter checkpoint does not exist"
    echo "  Will create during evaluation"
fi

# ============================================================================
# Step 3: Check/Build Uni-R2
# ============================================================================

echo ""
echo "=========================================="
echo "Step 3: Uni-R2 (Multiple Token Budgets)"
echo "=========================================="

UNICORE_CHECKPOINT="$CHECKPOINT_DIR/uni_r2_feature_matrix.pkl"

if [ -f "$UNICORE_CHECKPOINT" ]; then
    echo "  [✓] Uni-R2 feature matrix exists"
    echo "  Checkpoint: $UNICORE_CHECKPOINT"
else
    echo "  [ ] Uni-R2 feature matrix does not exist"
    echo "  Will create during evaluation"
fi

# ============================================================================
# Step 4: Run Comparison Evaluation
# ============================================================================

echo ""
echo "=========================================="
echo "Step 4: Running Comparative Evaluation"
echo "=========================================="

# Build model arguments for initial pool
INITIAL_MODELS_ARG=()
for llm in "${INITIAL_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    INITIAL_MODELS_ARG+=(--initial-model "$name" "$size" "$csv")
done

# Build model arguments for new models
NEW_MODELS_ARG=()
for llm in "${NEW_MODELS[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    NEW_MODELS_ARG+=(--new-model "$name" "$size" "$csv")
done

echo "Evaluating:"
echo "  Initial models: ${#INITIAL_POOL[@]}"
for llm in "${INITIAL_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    echo "    - $name (size: ${size}B)"
done

echo ""
echo "  New models (added dynamically): ${#NEW_MODELS[@]}"
for llm in "${NEW_MODELS[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    echo "    - $name (size: ${size}B)"
done

echo ""
echo "Running evaluation script..."

python unirouter/eval_compare.py \
    "${INITIAL_MODELS_ARG[@]}" \
    "${NEW_MODELS_ARG[@]}" \
    --val-size "$VAL_SIZE" \
    --n-clusters "$N_CLUSTERS" \
    --token-budgets "$TOKEN_BUDGETS" \
    --lambda-dist "$LAMBDA_DISTRIBUTION" \
    --target-accuracy-rate "$TARGET_ACCURACY_RATE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$RESULTS_DIR"

# ============================================================================
# Step 5: Display Results
# ============================================================================

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE!"
echo "=========================================="
echo ""

# Display metrics if available
METRICS_FILE="$RESULTS_DIR/comparison_metrics.csv"
CURVES_FILE="$RESULTS_DIR/comparison_curves.csv"
PLOT_FILE="$RESULTS_DIR/comparison_plot.png"

if [ -f "$METRICS_FILE" ]; then
    echo "Performance Metrics:"
    echo "-------------------"
    cat "$METRICS_FILE"
    echo ""
fi

echo "Results saved to:"
echo "  - Metrics: $METRICS_FILE"
echo "  - Curves: $CURVES_FILE"
echo "  - Plot: $PLOT_FILE"
echo ""

echo "Checkpoints saved to:"
echo "  - Original UniRouter: $ORIGINAL_CHECKPOINT"
echo "  - Uni-R2: $UNICORE_CHECKPOINT"
echo "  - Validation set: $VAL_SET_FILE"
echo ""

echo "Key Findings:"
echo "-------------"
echo "1. Original UniRouter:"
echo "   - Routes to best LLM (always unlimited tokens)"
echo "   - Validation cost: $VAL_SIZE API calls per model"
echo ""
echo "2. Uni-R2:"
echo "   - Routes to best (LLM, token_budget) pair"
IFS=',' read -ra BUDGETS <<< "$TOKEN_BUDGETS"
echo "   - Validation cost: $((VAL_SIZE * ${#BUDGETS[@]})) API calls per model"
echo "   - Enables cost-quality tradeoff"
echo ""
echo "3. Dynamic Model Addition:"
echo "   - Added ${#NEW_MODELS[@]} new models WITHOUT retraining"
echo "   - Only required running new models on validation set"
echo "   - Total models after addition: $((${#INITIAL_POOL[@]} + ${#NEW_MODELS[@]}))"
echo ""

# ============================================================================
# Usage Examples
# ============================================================================

echo "=========================================="
echo "Usage Examples"
echo "=========================================="
echo ""
echo "To customize the experiment, edit this script and modify:"
echo ""
echo "1. INITIAL_POOL - Models for initial training"
echo "   Example:"
echo "     INITIAL_POOL=("
echo "       \"GPT-3.5|10|data/gpt3.5.csv\""
echo "       \"GPT-4|50|data/gpt4.csv\""
echo "     )"
echo ""
echo "2. NEW_MODELS - Models to add dynamically"
echo "   Example:"
echo "     NEW_MODELS=("
echo "       \"Claude-3|25|data/claude3.csv\""
echo "     )"
echo ""
echo "3. VAL_SIZE - Validation set size (default: 500)"
echo "   Larger = more accurate features, slower validation"
echo ""
echo "4. N_CLUSTERS - Number of clusters (default: 100)"
echo "   More clusters = finer-grained routing"
echo ""
echo "5. TOKEN_BUDGETS - Comma-separated list (9999 = unlimited)"
echo "   Example: \"50,100,200,400,800\""
echo ""
echo "6. LAMBDA_DISTRIBUTION - Multi-segment lambda sampling"
echo "   Format: \"min,max,num;min,max,num;...\""
echo "   Example: \"0,0.001,100;0.001,0.01,100;0.01,0.1,100;0.1,1.0,100\""
echo "   Allows denser sampling in quality-focused regions"
echo ""
