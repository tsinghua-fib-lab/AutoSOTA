#!/bin/bash
#SBATCH --job-name=hispatial
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --partition=hpc-high
#SBATCH --output=slurm_logs/training/hispatial_%j.log
#SBATCH --error=slurm_logs/training/hispatial_%j.err

# ===== GPU Monitoring =====
mkdir -p slurm_logs/monitoring
monitor_log="slurm_logs/monitoring/gpu_monitor_$SLURM_JOB_ID.log"
(
    while true; do
        echo "===== $(date) =====" >> "$monitor_log"
        nvidia-smi >> "$monitor_log"
        sleep 600
    done
) &

# ===== Configuration =====
CONFIG_FILE=${1:-"configs/train_default.json"}

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_LAUNCH_TIMEOUT=6000000
export NCCL_SOCKET_TIMEOUT=3600
export OMP_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_API_KEY="${WANDB_API_KEY}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== Multi-node setup from Slurm =====
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export NODE_RANK=$SLURM_PROCID
NNODE=$SLURM_JOB_NUM_NODES
NGPU=$SLURM_GPUS_ON_NODE

echo "============================================"
echo " HiSpatial Training"
echo " Config : $CONFIG_FILE"
echo " Nodes  : $NNODE"
echo " GPUs   : $NGPU per node"
echo " Master : $MASTER_ADDR:$MASTER_PORT"
echo "============================================"

# ===== Launch via srun + torchrun =====
srun bash -c "
    torchrun \
    --nnodes $NNODE \
    --nproc-per-node $NGPU \
    --rdzv-backend=c10d \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/train.py \
    $CONFIG_FILE \
    --gpus $NGPU \
    --num_nodes $NNODE"
