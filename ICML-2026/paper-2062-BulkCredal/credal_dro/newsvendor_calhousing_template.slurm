#!/bin/bash
#
#SBATCH --job-name=<insert job name>
#SBATCH --partition=<insert partition name>
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000
#SBATCH --time=2-00:00:00
#SBATCH --array=0-{num_batches}
#

## Loop over each batch ##
start=$(($SLURM_ARRAY_TASK_ID * 1))
srun credaldro batch {experiment_dir} $start {batch_size} --experiment-data-dir {experiment_data_dir}
