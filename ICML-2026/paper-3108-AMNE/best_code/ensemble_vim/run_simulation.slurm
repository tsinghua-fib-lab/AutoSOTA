#!/bin/bash
#SBATCH --job-name=run_simulation
#SBATCH --output=log/simulation_%A_%a.out
#SBATCH --error=log/simulation_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-100%25
#SBATCH --qos=1d


python simulation.py \
    --n_samples 128 256 512 1024 2048 \
    --n_features 20 \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --n_jobs 5 \
    --n_splits 5 \
    --snr 1 \
    --n_ensemble 10 \
    --results_dir ./results \
    --model_name rf \
    --ensemble voting \
    --dataset_name g_function \
    --sage