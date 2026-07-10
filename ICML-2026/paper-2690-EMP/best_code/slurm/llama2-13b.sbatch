#!/bin/bash
#SBATCH --job-name=LLama2-13b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --partition hpg-b200
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:b200:4

## Logging
##SBATCH --output=logs/%x_%j.out
##SBATCH --error=logs/%x_%j.err

set -euo pipefail
LOG_DIR="${PWD}/logs"
mkdir -p logs


TS="$(date +'%Y%m%d_%H%M%S')"
COMBINED_LOG="$LOG_DIR/LLama2-13b.log"
exec > >(tee -a "$COMBINED_LOG") 2>&1
# -------------------- Environment --------------------

module purge
module load conda
module load cuda/12.8
conda activate emp

# Hugging Face setup — caches on node-local scratch for speed
export HF_TOKEN=""   # <-- your token
export HF_HOME="$SLURM_TMPDIR/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$SLURM_TMPDIR/hf_datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN

# Put weights on fast local storage (node tmp)
CACHE_DIR="$SLURM_TMPDIR/llm_weights"
mkdir -p "$CACHE_DIR"

echo "Starting run at: $(date)"


srun python -u wanda.py \
  --model_id meta-llama/Llama-2-13b-hf \
  --cache_dir "$CACHE_DIR" \
  --calib_seq_len 512 \
  --calib_samples 128 \
  --block_size_eval 2048 \
  --beta 1.0 \
  --eval_zeroshot \
  --zs_max_samples 0 \
  --zs_batch_size 8 \
  --zs_max_length 256

echo "Finished at: $(date)"
