#!/bin/bash
#SBATCH --job-name=maltree_image
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/image_training_%j.out
#SBATCH --error=logs/image_training_%j.err

# MalTree Image Embedding Training Script
# For distributed training on SLURM cluster

module purge
module load cuda/11.8
module load python/3.10

# Activate conda environment
source activate malware_analysis

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Create logs directory
mkdir -p logs

# Navigate to image analysis directory
cd $SLURM_SUBMIT_DIR/malware_image_analysis

# Run distributed training
# Hyperparameters from paper (Table 4, Appendix I)
srun python train_distributed.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.01 \
    --weight-decay 0.0001 \
    --early-stopping-patience 10 \
    --lr-scheduler-patience 10 \
    --lr-scheduler-factor 0.1 \
    --seed 42

echo "Training completed at $(date)"
