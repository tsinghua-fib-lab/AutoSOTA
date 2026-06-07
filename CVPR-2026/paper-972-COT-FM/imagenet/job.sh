#!/bin/bash
#SBATCH --job-name=cotfm
#SBATCH --partition=mi3008xl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cotfm_%j.out
#SBATCH --error=logs/cotfm_%j.err

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate rf

export NCCL_P2P_DISABLE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=8 --nnodes=1  --master-port=29502  train_cotfm.py     --exp-name "cotfm"     --output-dir "work_dir"     --data-dir "./imagenet_data/train_vae_latents_lmdb"     --model "SiT-B/4"     --resolution 256     --batch-size 256     --allow-tf32     --mixed-precision "bf16"     --epochs 160   --path-type "linear"     --weighting "adaptive"     --time-sampler "logit_normal"     --time-mu -0.4     --time-sigma 1.0     --ratio-r-not-equal-t 0.25     --adaptive-p 1.0     --cfg-omega 3     --cfg-kappa 0.    --cfg-min-t 0.0    --cfg-max-t 1.0  --checkpointing-steps 20000 --num-classes 1000  --seed 0
