#!/bin/bash
#SBATCH --job-name=longbenchv2_long
#SBATCH --account=torch_pr_637_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200"
#SBATCH --nodes=1 --tasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128GB --time=24:00:00
#SBATCH --output=slurm-%j.out

source ${CONDA_BASE:-$HOME/miniconda3}/bin/activate
conda activate qwen3
cd ${REPO_ROOT:-$HOME/kv-cache-dev}

echo "=== LongBench v2 Evaluation (long samples) ==="
echo "Config: configs/test_long_context_long.yaml"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/inference_with_recompute_kv.py configs/test_long_context_long.yaml
