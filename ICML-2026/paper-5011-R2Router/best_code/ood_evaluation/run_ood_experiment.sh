#!/bin/bash
# OOD Evaluation Pipeline: Train R2-Router and baselines, evaluate generalization
#
# This script trains models on 19 categories and tests on 1 held-out category
# to evaluate out-of-distribution generalization.

set -e

# ============================================================================
# CONFIGURATION - Edit this section
# ============================================================================

# Define your LLM pool here (format: "Name|Size|CSVPath")

LLM_POOL=(
    "Mistral-7B-Instruct-v0.2|0.20|data/Mistral-7B-Instruct-v0.2.csv"
    "GLM-4.5-Air|0.85|data/GLM-4.5-Air.csv"
    "GLM-4.6|1.75|data/GLM-4.6.csv"
    "gemma-3-4b-it|0.06815|data/gemma-3-4b-it.csv"
    "gemma-3-1b-it|0.0170375|data/gemma-3-1b-it.csv"
    "gemma-3-270m-it|0.004259375|data/gemma-3-270m-it.csv"
    "Llama-3.1-70B-Instruct|0.40|data/Llama-3.1-70B-Instruct.csv"
    "Llama-3.2-3B-Instruct|0.02|data/Llama-3.2-3B-Instruct.csv"
    "Qwen2.5-Math-1.5B-Instruct|0.09|data/Qwen2.5-Math-1.5B-Instruct.csv"
    "Qwen2.5-Math-7B-Instruct|0.35|data/Qwen2.5-Math-7B-Instruct.csv"
    "Qwen3-0.6B|0.0173|data/Qwen3-0.6B.csv"
    # "Qwen3-235B-A22B-Instruct-2507|0.55|data/Qwen3-235B-A22B-Instruct-2507.csv"
    # "Qwen3-Next-80B-A3B-Instruct|0.6|data/Qwen3-Next-80B-A3B-Instruct.csv"
)

# R2-Router Hyperparameters - IMPORTANT: Must match IID training!
CORE_MODEL_TYPE="ridge"        # Use Ridge for better OOD generalization
CORE_ALPHA=10.0                # Regularization strength (MUST match IID)

# Lambda Distribution - Controls cost-performance tradeoff sampling
# Format: "min,max,num_points" for each segment (segments are concatenated)
# Formula: score = (1-λ)*quality - λ*cost, where λ ∈ [0,1]
# Default: "0,0.2,20;0.2,1.0,50" - denser sampling at low lambda (quality-focused)
LAMBDA_DISTRIBUTION="0,0.001,100;0.001,0.01,100;0.01,0.1,100;0.1,1.0,100"

# OOD Evaluation Settings
TEST_CATEGORY="Idavidrein/gpqa/gpqa_extended"  # Category to hold out for testing (GPQA: 384 queries)
QUICK_MODE=false                     # Set to true for quick demo with 1 model
TARGET_ACCURACY_RATE=1.0             # Target accuracy rate for QNC (1.0 = 100%, 0.9 = 90% of best LLM)
# OUTPUT_DIR will be set automatically based on TEST_CATEGORY (can override with --output)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            TEST_CATEGORY="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --target-rate)
            TARGET_ACCURACY_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --category CATEGORY      Test category (default: TIGER-Lab/MMLU-Pro)"
            echo "  --quick                  Quick demo with first model only"
            echo "  --output DIR             Output directory (default: ./comparison_results/ood_evaluation)"
            echo "  --target-rate RATE       Target accuracy rate for QNC (default: 1.0)"
            echo "                           1.0 = 100% of best LLM, 0.9 = 90% of best LLM"
            echo "  --help                   Show this help message"
            echo ""
            echo "Available Categories (20 total):"
            echo ""
            echo "Large Categories (>1000 queries):"
            echo "  1. openhermes/teknium                    (13,670 queries - 44.1%)"
            echo "  2. TIGER-Lab/MMLU-Pro                    (8,264 queries - 26.7%) [DEFAULT]"
            echo "  3. lighteval/MATH/all                    (5,122 queries - 16.5%)"
            echo "  4. lighteval/MATH/all/test               (1,735 queries - 5.6%)"
            echo ""
            echo "Small Categories (100-400 queries):"
            echo "  5. Idavidrein/gpqa/gpqa_extended         (384 queries - Graduate science)"
            echo "  6. TAUR-Lab/MuSR/team_allocation         (176 queries - Multi-step reasoning)"
            echo "  7. TAUR-Lab/MuSR/object_placements       (185 queries - Multi-step reasoning)"
            echo "  8. TAUR-Lab/MuSR                         (163 queries - Multi-step reasoning)"
            echo ""
            echo "RAGBench Categories (82-167 queries - Domain-specific QA):"
            echo "  9.  rungalileo/ragbench/covidqa          (167 queries - COVID medical)"
            echo "  10. rungalileo/ragbench/cuad             (137 queries - Legal contracts)"
            echo "  11. rungalileo/ragbench/delucionqa       (131 queries - Delucion QA)"
            echo "  12. rungalileo/ragbench/emanual          (100 queries - E-manuals)"
            echo "  13. rungalileo/ragbench/expertqa         (97 queries - Expert QA)"
            echo "  14. rungalileo/ragbench/finqa            (100 queries - Finance)"
            echo "  15. rungalileo/ragbench/hagrid           (82 queries - HAGRID)"
            echo "  16. rungalileo/ragbench/hotpotqa         (94 queries - Multi-hop QA)"
            echo "  17. rungalileo/ragbench/msmarco          (94 queries - MS MARCO)"
            echo "  18. rungalileo/ragbench/pubmedqa         (86 queries - PubMed medical)"
            echo "  19. rungalileo/ragbench/tatqa            (90 queries - Table QA)"
            echo "  20. rungalileo/ragbench/techqa           (91 queries - Technical QA)"
            echo ""
            echo "Examples:"
            echo "  $0                                                    # Default: MMLU-Pro"
            echo "  $0 --category 'openhermes/teknium'                    # Largest category"
            echo "  $0 --category 'lighteval/MATH/all'                    # Math problems"
            echo "  $0 --category 'Idavidrein/gpqa/gpqa_extended'         # Graduate science"
            echo "  $0 --category 'TAUR-Lab/MuSR/team_allocation'         # Multi-step reasoning"
            echo "  $0 --category 'rungalileo/ragbench/finqa'             # Finance domain"
            echo "  $0 --quick                                            # Quick test (1 model)"
            echo ""
            echo "To see category details:"
            echo "  cat ./ood_evaluation/category_splits/ood_splits_summary.csv"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Generate training scheme suffix (same logic as run_experiment.sh)
SCHEME_SUFFIX="ridge_alpha${CORE_ALPHA}"

# Create category-safe name (replace / with _)
CATEGORY_SAFE=$(echo "$TEST_CATEGORY" | sed 's/\//_/g')

# Set default output directory based on category (can be overridden with --output)
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./comparison_results/ood_evaluation/${CATEGORY_SAFE}"
fi

# Build model arguments
MODELS_ARG=()
for llm in "${LLM_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    MODELS_ARG+=(--model "$name" "$size" "$csv")
done

echo "=========================================="
echo "OOD EVALUATION - Out-of-Distribution Test"
echo "=========================================="
echo "Test Category: $TEST_CATEGORY"
echo "Model Pool: ${#LLM_POOL[@]} models"
echo "R2-Router Config: $CORE_MODEL_TYPE (alpha=$CORE_ALPHA)"
echo "Lambda Distribution: $LAMBDA_DISTRIBUTION"
echo "Quick Mode: $QUICK_MODE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ============================================================================
# Step 1: Check if OOD category splits exist
# ============================================================================

echo "Checking OOD category splits..."
if [ ! -f "./ood_evaluation/category_splits/ood_splits.pkl" ]; then
    echo "⚠️  Category splits not found!"
    echo "   Running map_and_split_data.py to create splits..."
    python ood_evaluation/map_and_split_data.py
    echo "✓ Category splits created"
else
    echo "✓ Category splits exist"
fi

# ============================================================================
# Step 2: Check existing checkpoints
# ============================================================================

echo ""
echo "Checking OOD checkpoints..."

# Count existing R2-Router checkpoints
EXISTING_CORE=0
TOTAL_CORE=${#LLM_POOL[@]}

if [ "$QUICK_MODE" = true ]; then
    TOTAL_CORE=1
fi

for llm in "${LLM_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    # Convert name to checkpoint key (replace special chars)
    checkpoint_key=$(echo "$name" | sed 's/[.-]/_/g')
    checkpoint_dir="./checkpoints/ood_evaluation/${CATEGORY_SAFE}/${checkpoint_key}_${SCHEME_SUFFIX}"

    if [ -d "$checkpoint_dir" ] && \
       [ -f "$checkpoint_dir/limited_score_predictors.joblib" ] && \
       [ -f "$checkpoint_dir/unlimited_score_predictor.joblib" ] && \
       [ -f "$checkpoint_dir/unlimited_token_predictor.joblib" ]; then
        EXISTING_CORE=$((EXISTING_CORE + 1))
        echo "  [✓] $name"
    else
        echo "  [ ] $name - will train"
    fi

    # In quick mode, only check first model
    if [ "$QUICK_MODE" = true ]; then
        break
    fi
done

echo ""
echo "R2-Router checkpoints: $EXISTING_CORE/$TOTAL_CORE exist"

# Check baseline checkpoints
CARROT_KNN_EXISTS=false
CARROT_LINEAR_EXISTS=false

if [ -d "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_knn" ] && \
   [ -f "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_knn/knn_score.joblib" ]; then
    CARROT_KNN_EXISTS=true
    echo "  [✓] CARROT-KNN"
else
    echo "  [ ] CARROT-KNN - will train"
fi

if [ -d "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_linear" ] && \
   [ -f "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_linear/linear_score.joblib" ]; then
    CARROT_LINEAR_EXISTS=true
    echo "  [✓] CARROT-Linear"
else
    echo "  [ ] CARROT-Linear - will train"
fi

# ============================================================================
# Step 3: Run OOD Evaluation
# ============================================================================

echo ""
echo "=========================================="
echo "Running OOD Evaluation"
echo "=========================================="

# Build command
CMD="python ood_evaluation/run_ood.py --category \"$TEST_CATEGORY\" --output \"$OUTPUT_DIR\" --lambda-dist \"$LAMBDA_DISTRIBUTION\" --target-accuracy-rate $TARGET_ACCURACY_RATE"

if [ "$QUICK_MODE" = true ]; then
    CMD="$CMD --quick"
fi

# Add model arguments
for arg in "${MODELS_ARG[@]}"; do
    CMD="$CMD $arg"
done

echo "Command: $CMD"
echo ""

# Execute
eval $CMD

# ============================================================================
# Step 4: Create config.txt files in checkpoint and results directories
# ============================================================================

echo ""
echo "Creating config.txt files..."

# Build config content
CONFIG_CONTENT="# OOD Evaluation Configuration
# Category: ${TEST_CATEGORY}
# Training: 19 categories (excluding test category)
# Testing: 1 held-out category (${TEST_CATEGORY})

[R2-Router Hyperparameters]
model_type = ${CORE_MODEL_TYPE}
alpha = ${CORE_ALPHA}

[Lambda Distribution]
lambda_dist = ${LAMBDA_DISTRIBUTION}

[LLM Pool]
# Format: Name | Size (B) | CSV Path
"

# Add each LLM to config
for llm in "${LLM_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    CONFIG_CONTENT+="${name} | ${size} | ${csv}
"
done

# Save config to results directory
echo "$CONFIG_CONTENT" > "${OUTPUT_DIR}/config.txt"
echo "  [✓] ${OUTPUT_DIR}/config.txt"

# Save config to each checkpoint directory
for llm in "${LLM_POOL[@]}"; do
    IFS='|' read -r name size csv <<< "$llm"
    checkpoint_key=$(echo "$name" | sed 's/[.-]/_/g')
    checkpoint_dir="./checkpoints/ood_evaluation/${CATEGORY_SAFE}/${checkpoint_key}_${SCHEME_SUFFIX}"

    if [ -d "$checkpoint_dir" ]; then
        echo "$CONFIG_CONTENT" > "${checkpoint_dir}/config.txt"
        echo "  [✓] ${checkpoint_dir}/config.txt"
    fi

    # In quick mode, only create config for first model
    if [ "$QUICK_MODE" = true ]; then
        break
    fi
done

# Also save config to baseline checkpoint directories
if [ -d "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_knn" ]; then
    echo "$CONFIG_CONTENT" > "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_knn/config.txt"
    echo "  [✓] ./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_knn/config.txt"
fi

if [ -d "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_linear" ]; then
    echo "$CONFIG_CONTENT" > "./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_linear/config.txt"
    echo "  [✓] ./checkpoints/ood_evaluation/${CATEGORY_SAFE}/carrot_linear/config.txt"
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo "=========================================="
echo "OOD EVALUATION COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_DIR}/"
echo ""
echo "Checkpoints saved to:"
echo "  - ./checkpoints/ood_evaluation/${CATEGORY_SAFE}/"
echo ""

echo "Key result files:"
echo "  - ${OUTPUT_DIR}/${CATEGORY_SAFE}_metrics.csv"
echo "  - ${OUTPUT_DIR}/${CATEGORY_SAFE}_curves.csv"
echo "  - ${OUTPUT_DIR}/${CATEGORY_SAFE}_plot.png"
echo ""

# Display metrics if available
METRICS_FILE="${OUTPUT_DIR}/${CATEGORY_SAFE}_metrics.csv"
if [ -f "$METRICS_FILE" ]; then
    echo "Performance Summary:"
    echo "-------------------"
    cat "$METRICS_FILE"
    echo ""
fi

echo "To test a different category:"
echo "  bash ood_evaluation/run_ood_experiment.sh --category 'lighteval/MATH/all'"
echo ""
