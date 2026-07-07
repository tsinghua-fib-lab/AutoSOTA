#!/bin/bash
#SBATCH --job-name=nonlinear_d4
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --array=0-249
#SBATCH --output=/home/zach/cox-subgroup/results/slurm_logs/nonlinear_d4_%A_%a.out
#SBATCH --error=/home/zach/cox-subgroup/results/slurm_logs/nonlinear_d4_%A_%a.err

OFFSET=3000

# Print this sub-job's task ID
echo "My experiment number is " $((SLURM_ARRAY_TASK_ID + OFFSET))

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate survival
cd /home/zach/cox-subgroup/experiments
python run_slurm.py nonlinear_d4 $((SLURM_ARRAY_TASK_ID + OFFSET))
