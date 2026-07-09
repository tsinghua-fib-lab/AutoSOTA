#!/bin/bash

# Define Experiment Setups
MIXING_TYPES=("normalclamppoly")
POLYMIX_DEGREES=(1 2 3)
DATA_SEEDS=({42..61})

for mix in "${MIXING_TYPES[@]}"; do
    for degree in "${POLYMIX_DEGREES[@]}"; do
        for dseed in "${DATA_SEEDS[@]}"; do
            # Export variables so the sbatch script can see them
            export DATA_SEED="$dseed"
            export MIX_SEED=123 # 100
            export MIX_TYPE="$mix"
            export POLYMIX_DEGREE="$degree"
            export DV_TRUE=2
            export DW_TRUE=2
            export DZ=4
            export NPOP=3
            export NTRAIN=10000
            export ENC_TYPE="mlpnorm"
            export DEC_TYPE="poly"
            export DV=2
            export DW=2
            export INV_LOSS_TYPE="poly"
            export IND_LOSS_TYPE="poly"
            export INV_KER_POLY_DEGREE=2
            export IND_KER_POLY_DEGREE=2
            export MAX_EPOCHS=800

            # Note: named new2 sicne adding inv_loss /= len(pop_pairs) in mdcrl.py
            # For MIX_SEED=100, new_3pop_normalclamppolymixpoly{1,2,3} were run before this change.
            # For MIX_SEED=100, only new2_3pop_normalclamppolymixpoly2 was run after this change
            # For MIX_SEED=123, all run after this change
            # Name for Hydra/WandB (exp_id)
            export EXP_NAME="new2_${NPOP}pop_${mix}mix${degree}_ms${MIX_SEED}"
            echo "Submitting $EXP_NAME for Seed: $dseed"

            # export RESUME_ARGS="" # NOTE: for fresh start
            CHECKPOINT_PATH="checkpoints/last.ckpt" # NOTE: for resuming from a check point
            export RESUME_ARGS="+resume_from_checkpoint='$CHECKPOINT_PATH'"
            
            echo $RESUME_ARGS
            
            sbatch run_grid.sh
        done
    done
done