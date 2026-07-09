#!/bin/bash
#SBATCH --job-name=olmo-60m-len2048-zeroshot
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=128G


export PYTHONPATH="/home/xxxxxxxxx/03-proj/PE/OLMo:$PYTHONPATH"
export HF_HOME="/data/xxxxxxxxx/03-proj/PE/hf_cache"
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_EXE="/home/xxxxxxxxx/anaconda3/bin/python"

BASELINE_CKPT="/data/xxxxxxxxx/03-proj/PE/checkpoints/olmo-60m-Baseline-RoPE-flash-len2048-1.5B/final_model.pt"
SCALED_CKPT="/data/xxxxxxxxx/03-proj/PE/checkpoints/olmo-60m-ScaledRoPE-flash-len2048-1.5B/final_model.pt"
DATA_PATH="/data/xxxxxxxxx/03-proj/PE/c4_30M_validation"

CONFIG_PATH="./configs/olmo_60m.yaml"

echo "===================================================================="
echo "Starting Zero-shot Eval: Baseline"
echo "Checkpoint: $BASELINE_CKPT"
echo "Target Lengths: 2048, 4096, 8192"
echo "===================================================================="

$PYTHON_EXE eval_extrapolation.py \
    --config $CONFIG_PATH \
    --checkpoint "$BASELINE_CKPT" \
    --data_path "$DATA_PATH" \
    --lengths 2048 4096 8192

echo "Finished Eval: Baseline"
echo "--------------------------------------------------------------------"
echo ""

SIGMA=85.0

echo "===================================================================="
echo "Starting Zero-shot Eval: Scaled RoPE (Inductive Bias Check)"
echo "Checkpoint: $SCALED_CKPT"
echo "Forcing Scaled RoPE with Sigma: $SIGMA"
echo "Target Lengths: 2048, 4096, 8192"
echo "===================================================================="

$PYTHON_EXE eval_extrapolation.py \
    --config $CONFIG_PATH \
    --checkpoint "$SCALED_CKPT" \
    --data_path "$DATA_PATH" \
    --lengths 2048 4096 8192 \
    --force_scaled_rope \
    --sigma $SIGMA

echo "Finished Eval: Scaled RoPE"
echo "===================================================================="