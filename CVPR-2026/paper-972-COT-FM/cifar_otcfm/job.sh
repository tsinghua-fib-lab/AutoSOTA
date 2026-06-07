#!/bin/bash
#SBATCH --job-name=cotfm
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=logs/cotfm/cotfm_%j.out
#SBATCH --error=logs/cotfm/cotfm_%j.err

mkdir -p logs/cotfm
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate rectflow

export NCCL_P2P_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1

export PYTHONPATH="${PYTHONPATH}:./AlignFlow"

CUDA_VISIBLE_DEVICES=0 python train_cifar10.py --model cotfm --seed 10 --wandb_name cotfm_block_4 --output_dir ./results/cotfm_block_4