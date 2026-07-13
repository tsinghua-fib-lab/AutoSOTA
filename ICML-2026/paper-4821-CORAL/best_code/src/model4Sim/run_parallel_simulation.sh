#!/bin/bash

# Define the list of datasets
DATASETS=("ML1m" "Steam" "Amazon")

echo "======================================"
echo "Starting Sequential Multi-Dataset Simulation"
echo "======================================"

for DATASET in "${DATASETS[@]}"
do
    echo ""
    echo ">>> Current Dataset: $DATASET"
    echo "--------------------------------------"

    # The simulation command
    python simulate_llm_rec_parallel.py \
        --dataset "$DATASET" \
        --hiddenSize 100 \
        --max_len 200 \
        --batch_size 32 \
        --n_layers 2 \
        --n_heads 2 \
        --lambda_max 0.77 \
        --rho 1.0 \
        --kappa 1.0 \
        --tau 0.5 \
        --window_size 30 \
        --update_freq 5 \
        --lr_hawkes 0.1 \
        --epoch_hawkes 10

    # Optional: check if the previous command failed
    if [ $? -ne 0 ]; then
        echo "Error: Simulation failed for $DATASET. Exiting loop."
        exit 1
    fi
    
    echo "Done with $DATASET."
done

echo ""
echo "======================================"
echo "All datasets completed successfully!"
echo "Check logs/ folder for results"
echo "======================================"