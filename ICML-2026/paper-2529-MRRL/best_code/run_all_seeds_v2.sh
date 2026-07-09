#!/bin/bash
cd /repo
LOG=/repo/seed_progress.log
echo "=== Started at $(date) ===" > $LOG

run_one() {
    local S=$1 G=$2
    export CUDA_VISIBLE_DEVICES=$G
    export WANDB_MODE=offline
    export SLURM_JOB_ID=local_${S}
    export SLURM_NODELIST=localhost
    export HOSTNAME=localhost
    echo "[$(date)] seed=$S gpu=$G START" >> $LOG
    python train.py \
        data_seed=$S mix_seed=100 \
        data.mixing_type=normalclamppoly data.polymix_degree=3 \
        data.dim_v_true=2 data.dim_w_true=2 data.n_pop=2 \
        encoder=mlpnorm decoder=poly \
        model.dim_v=2 model.dim_w=2 \
        loss.inv_loss_type=poly loss.inv_ker_poly_degree=2 \
        loss.ind_loss_type=poly loss.ind_ker_poly_degree=2 \
        loss.lam1=1.0 loss.lam2=1.0 loss.lam3=0.0 \
        trainer.max_epochs=400 \
        exp_id=repro_poly3 sim_id=1 \
        >> /repo/log_seed_${S}.txt 2>&1
    echo "[$(date)] seed=$S gpu=$G DONE (rc=$?)" >> $LOG
}

for S in 43 45 47 49 51 53 55 57 59 61; do
    S2=$((S+1))
    echo "[$(date)] Pair: $S (gpu0) + $S2 (gpu1)" >> $LOG
    run_one $S 0 &
    P1=$!
    run_one $S2 1 &
    P2=$!
    wait $P1 $P2
    echo "[$(date)] Pair ($S,$S2) complete" >> $LOG
done
echo "=== ALL DONE at $(date) ===" >> $LOG
