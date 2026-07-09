#!/bin/bash
# Reproduction script for Paper 2529
# Settings: polynomial mixing degree 3, dz=35, MMD-Poly2 + HSIC-Poly2, 400 epochs

SEED=$1
if [ -z "$SEED" ]; then
    echo "Usage: $0 <data_seed>"
    exit 1
fi

GPU_ID=$2
if [ -z "$GPU_ID" ]; then
    GPU_ID=0
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_MODE=offline
export SLURM_JOB_ID=local
export SLURM_NODELIST=localhost
export HOSTNAME=localhost

cd /repo

echo "[seed=$SEED gpu=$GPU_ID] Starting training..."
python train.py     data_seed=$SEED     mix_seed=100     data.mixing_type=normalclamppoly     data.polymix_degree=3     data.dim_v_true=2     data.dim_w_true=2     data.n_pop=2     encoder=mlpnorm     decoder=poly     model.dim_v=2     model.dim_w=2     loss.inv_loss_type=poly     loss.inv_ker_poly_degree=2     loss.ind_loss_type=poly     loss.ind_ker_poly_degree=2     loss.lam1=1.0     loss.lam2=1.0     loss.lam3=0.0     trainer.max_epochs=400     exp_id=repro_poly3     sim_id=1

echo "[seed=$SEED gpu=$GPU_ID] Training complete."
