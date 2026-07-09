#!/bin/bash

# Define Experiment Setups
MIXING_TYPES=("invmlp") # "cayleypoly" "normalsmoothpoly" "normalclamppoly" "invmlp"
INV_LOSSES=("rbf") # ("meanvar" "poly2" "poly3" "rbf")
DATA_SEEDS=({42..61}) # ({42..51})

for mix in "${MIXING_TYPES[@]}"; do
    for inv_loss in "${INV_LOSSES[@]}"; do
        inv_loss_type="${inv_loss//[0-9]/}"
        inv_ker_poly_degree="${inv_loss//[^0-9]/}"
        for dseed in "${DATA_SEEDS[@]}"; do
            export DATA_SEED="$dseed"
            export MIX_SEED=100
            export MIX_TYPE="$mix"
            export POLYMIX_DEGREE=3 # NOTE: 1, 2, 3 for poly mixing types or null if invmlp mixing
            export DV_TRUE=2
            export DW_TRUE=2
            export DZ=4
            export NPOP=2
            export NTRAIN=10000
            export ENC_TYPE="mlpnorm" # or "mlp" which is without layernorm
            export DEC_TYPE="poly" # NOTE: "mlp" if invmlp mixing or "poly" if poly mixing
            export DV=2
            export DW=2
            export INV_LOSS_TYPE="$inv_loss_type"
            export IND_LOSS_TYPE="rbf" # "orth" "poly" "rbf"
            export INV_KER_POLY_DEGREE="$inv_ker_poly_degree"
            export IND_KER_POLY_DEGREE=null
            export MAX_EPOCHS=400 # NOTE: if want to extend to 400, change the RESUME_ARGS below
            export GLOBAL_SIGMA=true
            # export L3=0.1
            
            # Name for Hydra/WandB (exp_id)
            export EXP_NAME="new_${mix}mix${POLYMIX_DEGREE/null/}_${ENC_TYPE}enc_${inv_loss}inv_${IND_LOSS_TYPE}${IND_KER_POLY_DEGREE}ind_gs_ms${MIX_SEED}"
            # export EXP_NAME="new_${mix}mix${POLYMIX_DEGREE/null/}_${ENC_TYPE}enc_${inv_loss}inv_${IND_LOSS_TYPE}${IND_KER_POLY_DEGREE}ind_${L3}lam3_ms${MIX_SEED}"
            echo "Submitting $EXP_NAME for Seed: $dseed"

            export RESUME_ARGS="" # NOTE: for fresh start
            # CHECKPOINT_PATH="checkpoints/last.ckpt" # NOTE: for resuming from a check point
            # export RESUME_ARGS="+resume_from_checkpoint='$CHECKPOINT_PATH'"
            
            echo $RESUME_ARGS

            # Submit array jobs
            sbatch run_grid.sh
        done
    done
done