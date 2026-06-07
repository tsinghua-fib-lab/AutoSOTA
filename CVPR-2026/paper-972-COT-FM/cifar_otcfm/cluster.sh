#!/bin/bash
#SBATCH --job-name=imagenet_cluster
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=logs/imagenet_cluster_%j.out
#SBATCH --error=logs/imagenet_cluster_%j.err

mkdir -p logs

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
echo "Starting ImageNet clustering..."
CUDA_VISIBLE_DEVICES=0 python clustering.py

echo ""
echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"