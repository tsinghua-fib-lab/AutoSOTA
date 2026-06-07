#!/bin/bash
#SBATCH --job-name=imagenet_cluster
#SBATCH --partition=mi3008x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=5:00:00
#SBATCH --output=logs/imagenet_cluster_%j.out
#SBATCH --error=logs/imagenet_cluster_%j.err

mkdir -p logs

source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate rf

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
torchrun --nproc_per_node=8 --nnodes=1 reverse.py \
    --data-dir "./imagenet_data/train_vae_latents_lmdb" \
    --model "SiT-B/4" \
    --resolution 256 \
    --batch-size 256 \
    --num-classes 1000 \
    --ckpt "./work_dir/rectified_flow/checkpoints/0800000.pt" \
    --output-dir "./imagenet_data/reconstructed_images" \
    --num-steps 50

echo ""
echo "======================================"
echo "Job finished at: $(date)"
echo "======================================"