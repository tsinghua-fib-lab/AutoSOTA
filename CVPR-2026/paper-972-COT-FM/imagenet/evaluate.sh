#!/bin/bash
#SBATCH --job-name=cotfm_eval
#SBATCH --partition=mi3008x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=05:00:00
#SBATCH --output=logs/cotfm_eval_%j.out
#SBATCH --error=logs/cotfm_eval_%j.err

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate rf

export NCCL_P2P_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=8 --nnodes=1 --master-port=29503 evaluate.py     --ckpt "/path/to/checkpoint.pt"     --model "SiT-B/2"     --resolution 256     --cfg-scale 2     --per-proc-batch-size 256     --num-fid-samples 50000     --sample-dir "./fid_dir"     --compute-metrics     --num-steps 100    --fid-statistics-file "./fid_stats/adm_in256_stats.npz" --num-classes 1000 --global-seed 0
