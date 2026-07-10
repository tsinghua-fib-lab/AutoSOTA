#!/bin/bash
set -e 

# Usage:
#   sh main_probe_steering.sh

REPO_ROOT="$(dirname "$(pwd)")"
CIRCUIT_BASE="../function_circuit/functions"
DMS_DATA_DIR="$REPO_ROOT/function_circuit/DMS" 
CLT_PATH="../models/CLT_L6_D3200/checkpoints/last.ckpt"
PLT_PATH="../models/PLT_L6_D3200/checkpoints/last.ckpt" 
ESM_PATH="../models/esm2_t6_8M_UR50D.pt"
SUPP_VALUES="4 8 16"
FOLDS="0,1,2,3,4"
MAX_MUTATIONS="5"
OUTPUT_BASE="probe_results_8M"

echo "========================================"
echo " [Setup] Configuration - PROBE STEERING"
echo "========================================"
echo "  > DMS Data Directory:   $DMS_DATA_DIR"
echo "  > Circuit Base Dir:     $CIRCUIT_BASE"
echo "  > CLT Checkpoint:       $CLT_PATH"
echo "  > PLT Checkpoint:       $PLT_PATH"
echo "  > ESM Weights:          $ESM_PATH"
echo "========================================"

for SUPP in $SUPP_VALUES; do
    for MUT in $MAX_MUTATIONS; do
        echo " [Running] Probe Steering with SUPP=$SUPP, MAX_MUTATIONS=${MUT}"
        python run_probe_steering.py \
            --dms_dir "$DMS_DATA_DIR" \
            --output_dir "${OUTPUT_BASE}/supp${SUPP}" \
            --clt_ckpt "$CLT_PATH" \
            --plt_ckpt "$PLT_PATH" \
            --esm_weights "$ESM_PATH" \
            --circuit_base "$CIRCUIT_BASE" \
            --supp "$SUPP" \
            --configs "CLT_sequential,CLT_sequential_no_frozen,PLT_no_frozen" \
            --folds "$FOLDS" \
            --max_mutations "$MUT" \
            --alpha_steps 25

        echo ""
    done
done

echo "Pipeline Complete."