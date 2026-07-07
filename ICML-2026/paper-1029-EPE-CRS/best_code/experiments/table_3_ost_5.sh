#!/bin/bash
#SBATCH --job-name=table_3_ost
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-999
#SBATCH --output=/home/zach/cox-subgroup/results/slurm_logs/table_3_ost_%A_%a.out
#SBATCH --error=/home/zach/cox-subgroup/results/slurm_logs/table_3_ost_%A_%a.err

OFFSET=5000

echo "My experiment number is " $((SLURM_ARRAY_TASK_ID + OFFSET))

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate survival
cd /home/zach/cox-subgroup/experiments
python run_slurm.py table_3_ost $((SLURM_ARRAY_TASK_ID + OFFSET))
