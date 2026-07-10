#!/bin/bash
#SBATCH --job-name=run_asymptotic
#SBATCH --output=log/asymptotic_%A_%a.out
#SBATCH --error=log/asymptotic_%A_%a.err
#SBATCH --time=23:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --qos=1d

python asymptotic.py \
    --n_samples 100000 \
    --n_features 20 \
    --seed 1 \
    --n_jobs 1 \
    --n_splits 5 \
    --snr 1 \
    --n_ensemble 10 \
    --results_dir ./results \
    --ensemble voting \
    --model_name rf \
    --dataset_name ishigami \
    --sage