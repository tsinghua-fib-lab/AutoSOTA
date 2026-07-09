#!/bin/bash
# Run all 20 seeds (42-61) for paper 2529 reproduction
# Usage: run_all_seeds.sh

SEEDS=$(seq 42 61)
GPU0=0
GPU1=1
PIDS=()

run_seed() {
    local SEED=$1
    local GPU=$2
    export CUDA_VISIBLE_DEVICES=$GPU
    export WANDB_MODE=offline
    export SLURM_JOB_ID=local
    export SLURM_NODELIST=localhost
    export HOSTNAME=localhost
    
    cd /repo
    echo "[seed=$SEED gpu=$GPU] Starting..."
    python train.py         data_seed=$SEED         mix_seed=100         data.mixing_type=normalclamppoly         data.polymix_degree=3         data.dim_v_true=2         data.dim_w_true=2         data.n_pop=2         encoder=mlpnorm         decoder=poly         model.dim_v=2         model.dim_w=2         loss.inv_loss_type=poly         loss.inv_ker_poly_degree=2         loss.ind_loss_type=poly         loss.ind_ker_poly_degree=2         loss.lam1=1.0         loss.lam2=1.0         loss.lam3=0.0         trainer.max_epochs=400         exp_id=repro_poly3         sim_id=1         > /repo/log_seed_${SEED}.txt 2>&1
    echo "[seed=$SEED gpu=$GPU] Done. Exit=$?"
}

GPU_NEXT=0
for SEED in $SEEDS; do
    if [ $GPU_NEXT -eq 0 ]; then
        run_seed $SEED 0 &
        PIDS+=($!)
        GPU_NEXT=1
    else
        run_seed $SEED 1 &
        PIDS+=($!)
        GPU_NEXT=0
        # Wait for both to finish
        wait ${PIDS[-2]} ${PIDS[-1]}
    fi
done

# Wait for any remaining jobs
wait

echo "All seeds completed."
