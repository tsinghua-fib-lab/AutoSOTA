#!/bin/bash
#SBATCH --job-name=imagenet_cluster
#SBATCH --partition=mi3008x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --output=logs/evaluate/evaluate_%j.out
#SBATCH --error=logs/evaluate/evaluate_%j.err


mkdir -p logs/evaluate
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate rectflow

export NCCL_P2P_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "======================================"

echo "GPU Information:"
rocm-smi || nvidia-smi

echo ""
echo "Starting Evaluation..."
torchrun --nproc-per-node=8 compute_fid_multi_gpu.py --path_pattern cotfm_block_4

echo ""
echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"