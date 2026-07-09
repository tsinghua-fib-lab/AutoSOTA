#!/bin/bash
# Master launcher - runs all seeds sequentially, 2 at a time
# This script is designed to be run with nohup inside the container

cd /repo
LOG=/repo/master_train.log

run_seed() {
    local SEED=$1
    local GPU=$2
    (
        export CUDA_VISIBLE_DEVICES=$GPU
        export WANDB_MODE=offline
        export SLURM_JOB_ID=local
        export SLURM_NODELIST=localhost
        export HOSTNAME=localhost
        
        echo "[$(date)] seed=$SEED gpu=$GPU START" >> $LOG
        python train.py             data_seed=$SEED             mix_seed=100             data.mixing_type=normalclamppoly             data.polymix_degree=3             data.dim_v_true=2             data.dim_w_true=2             data.n_pop=2             encoder=mlpnorm             decoder=poly             model.dim_v=2             model.dim_w=2             loss.inv_loss_type=poly             loss.inv_ker_poly_degree=2             loss.ind_loss_type=poly             loss.ind_ker_poly_degree=2             loss.lam1=1.0             loss.lam2=1.0             loss.lam3=0.0             trainer.max_epochs=400             exp_id=repro_poly3             sim_id=1             >> /repo/log_seed_${SEED}.txt 2>&1
        echo "[$(date)] seed=$SEED gpu=$GPU DONE (exit=$?)" >> $LOG
    )
}

echo "Master launcher started at $(date)" > $LOG

for SEED in 43 45 47 49 51 53 55 57 59 61; do
    SEED2=$((SEED + 1))
    echo "[$(date)] Launching seeds $SEED (GPU 0) and $SEED2 (GPU 1)" >> $LOG
    run_seed $SEED 0 &
    PID1=$!
    run_seed $SEED2 1 &
    PID2=$!
    wait $PID1 $PID2
    echo "[$(date)] Pair ($SEED, $SEED2) DONE" >> $LOG
done

echo "[$(date)] ALL SEEDS COMPLETE" >> $LOG
