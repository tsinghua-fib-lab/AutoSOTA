#!/bin/bash
# Simple parallel runner: runs seeds in background, max 4 concurrent (2 per GPU)

SEED=$1
GPU_ID=$2

export CUDA_VISIBLE_DEVICES=$GPU_ID
export WANDB_MODE=offline
export SLURM_JOB_ID=local
export SLURM_NODELIST=localhost
export HOSTNAME=localhost
cd /repo

echo "[$(date +%H:%M:%S)] seed=$SEED gpu=$GPU_ID START"
python train.py     data_seed=$SEED     mix_seed=100     data.mixing_type=normalclamppoly     data.polymix_degree=3     data.dim_v_true=2     data.dim_w_true=2     data.n_pop=2     encoder=mlpnorm     decoder=poly     model.dim_v=2     model.dim_w=2     loss.inv_loss_type=poly     loss.inv_ker_poly_degree=2     loss.ind_loss_type=poly     loss.ind_ker_poly_degree=2     loss.lam1=1.0     loss.lam2=1.0     loss.lam3=0.0     trainer.max_epochs=400     exp_id=repro_poly3     sim_id=1     > /repo/log_seed_${SEED}.txt 2>&1
echo "[$(date +%H:%M:%S)] seed=$SEED gpu=$GPU_ID DONE"
