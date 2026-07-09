#!/bin/bash

# Define Experiment Setups
MIXING_TYPES=("normalclamppoly") # "normalsmoothpoly"
HWS=(1 2 3 4)
DATA_SEEDS=({42..61})

for mix in "${MIXING_TYPES[@]}"; do
    for hw in "${HWS[@]}"; do
        for dseed in "${DATA_SEEDS[@]}"; do
            # Export variables so the sbatch script can see them
            export DATA_SEED="$dseed"
            export MIX_SEED=100
            export MIX_TYPE="$mix"
            export POLYMIX_DEGREE=3
            export DV_TRUE=2
            export DW_TRUE=2
            export DZ=4 # This is modified internally if poly mixing
            export NPOP=2
            export NTRAIN=10000
            export ENC_TYPE="mlpnorm"
            export DEC_TYPE="poly"
            export DV=2
            export DW="$hw"
            export INV_LOSS_TYPE="poly"
            export IND_LOSS_TYPE="poly"
            export INV_KER_POLY_DEGREE="$inv_ker_poly_degree"
            export IND_KER_POLY_DEGREE=2
            export MAX_EPOCHS=400
            
            # Name for Hydra/WandB (exp_id)
            export EXP_NAME="new_${mix}mix${POLYMIX_DEGREE}_${ENC_TYPE}enc_${INV_LOSS_TYPE}inv_${IND_LOSS_TYPE}ind_hw${hw}_ms${MIX_SEED}"
            
            echo "Submitting $EXP_NAME for Seed: $dseed"

            export RESUME_ARGS="" # NOTE: for fresh start
            # CHECKPOINT_PATH="checkpoints/last.ckpt" # NOTE: for resuming from a check point
            # export RESUME_ARGS="+resume_from_checkpoint='$CHECKPOINT_PATH'"
            
            echo $RESUME_ARGS
            
            # This triggers the array job for this specific combination
            sbatch run_grid.sh
        done
    done
done