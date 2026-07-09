#!/bin/bash
#SBATCH --job-name=kv-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --account=pr_118_tandon_priority
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --export=ALL

# Single evaluation job
# Called by eval_benchmark.slurm with environment variables:
# BENCHMARK, STRATEGY, CHUNK_K, LOG_DIR

echo "=============================================="
echo "Benchmark: $BENCHMARK"
echo "Strategy: $STRATEGY"
echo "Chunk K: $CHUNK_K"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

# Setup environment
source ${CONDA_BASE:-$HOME/miniconda3}/bin/activate
source ${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_DIR:-/path/to/conda_envs/qwen3_vl}

cd ${REPO_ROOT:-$HOME/kv-cache-optimization}

# Run evaluation
python scripts/eval_vlmeval.py \
    --data $BENCHMARK \
    --strategy $STRATEGY \
    --chunk_k $CHUNK_K \
    2>&1 | tee ${LOG_DIR}/${STRATEGY}_k${CHUNK_K}.log

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "End time: $(date)"
echo "=============================================="
