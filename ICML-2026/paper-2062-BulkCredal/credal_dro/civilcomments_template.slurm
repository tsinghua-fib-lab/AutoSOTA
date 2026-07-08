#!/bin/bash
#
#SBATCH --job-name=<job_name>
#SBATCH --partition=<insert_partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --time=2-00:00:00
#SBATCH --array=0-{num_batches}
#

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

batch_id=$SLURM_ARRAY_TASK_ID

srun credaldro batch_real_world {experiment_dir} $batch_id {batch_size} \
  --experiment-data-dir {experiment_data_dir} \
  --rw-dataset-dir {rw_dataset_dir}
