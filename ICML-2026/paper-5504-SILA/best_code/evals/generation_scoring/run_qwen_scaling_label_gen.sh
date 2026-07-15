#!/bin/bash
# Helper script to run label generation for Qwen scaling experiments
#
# Usage:
#   ./run_qwen_scaling_label_gen.sh 7b trained     # Run 7B with trained adapter
#   ./run_qwen_scaling_label_gen.sh 14b untrained  # Run 14B with untrained (grid search)
#   ./run_qwen_scaling_label_gen.sh all trained    # Run all sizes with trained adapter
#   ./run_qwen_scaling_label_gen.sh all untrained  # Run all sizes with untrained

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
MODEL_SIZE=${1:-"72b"}
VARIANT=${2:-"untrained"}

# Validate arguments
if [[ ! "$MODEL_SIZE" =~ ^(7b|14b|32b|72b|all)$ ]]; then
    echo -e "${RED}Error: Invalid model size '$MODEL_SIZE'${NC}"
    echo "Usage: $0 <model_size> <variant>"
    echo "  model_size: 7b, 14b, 32b, 72b, or all"
    echo "  variant: trained or untrained"
    exit 1
fi

if [[ ! "$VARIANT" =~ ^(trained|untrained)$ ]]; then
    echo -e "${RED}Error: Invalid variant '$VARIANT'${NC}"
    echo "Usage: $0 <model_size> <variant>"
    echo "  variant: trained or untrained"
    exit 1
fi

# Function to run a single experiment
run_experiment() {
    local size=$1
    local variant=$2
    local config_file="evals/generation_scoring/configs/qwen_scaling/qwen25_${size}_${variant}_label_gen.json"
    local output_dir="results/qwen_scaling/${size}_${variant}"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Running: Qwen2.5-${size^^} (${variant})${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Config: $config_file"
    echo "Output: $output_dir"
    echo ""
    
    # Check if config exists
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}Error: Config file not found: $config_file${NC}"
        return 1
    fi
    
    # Check if dataset exists (sample counts vary by model size)
    # Full VAL split: 4964 topics at every layer
    local n_samples
    case $size in
        7b)  n_samples=69496 ;;
        14b) n_samples=119136 ;;
        32b) n_samples=158848 ;;
        72b) n_samples=198560 ;;
    esac
    local dataset_file="outputs/qwen_scaling/qwen25_${size}_combined_val_${n_samples}_master.json"
    if [ ! -f "$dataset_file" ]; then
        echo -e "${YELLOW}Warning: Dataset not found: $dataset_file${NC}"
        echo -e "${YELLOW}Please run: python data_prep/prepare_qwen_scaling_datasets.py --model-size $size${NC}"
        return 1
    fi
    
    # For trained variant, check if checkpoint path needs updating
    if [ "$variant" == "trained" ]; then
        if grep -q "PLACEHOLDER_TRAINED_CHECKPOINT" "$config_file"; then
            echo -e "${YELLOW}Warning: Config still has placeholder checkpoint path${NC}"
            echo -e "${YELLOW}Please update the 'adapter_checkpoint_path' in $config_file${NC}"
            echo ""
            read -p "Continue anyway? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Skipping..."
                return 0
            fi
        fi
    fi
    
    # Run the evaluation
    /tmp/venv/bin/python evals/generation_scoring/run_eval.py \
        --config-file "$config_file" \
        --output-dir "$output_dir" \
        --no-wandb
    
    echo -e "${GREEN}✓ Completed: Qwen2.5-${size^^} (${variant})${NC}"
    echo ""
}

# Main execution
if [ "$MODEL_SIZE" == "all" ]; then
    SIZES=("7b" "14b" "32b" "72b")
else
    SIZES=("$MODEL_SIZE")
fi

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Qwen Scaling Label Generation        ║${NC}"
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo ""
echo "Model sizes: ${SIZES[@]}"
echo "Variant: $VARIANT"
echo ""

for size in "${SIZES[@]}"; do
    run_experiment "$size" "$VARIANT"
done

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  All experiments completed!            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
