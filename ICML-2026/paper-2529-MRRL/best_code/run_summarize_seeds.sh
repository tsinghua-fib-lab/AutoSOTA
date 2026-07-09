#!/bin/bash

CKPT_STRATEGY="last"
METRIC_KEY="val/inv_loss"
METHOD="tsls"  # tsls or liml
EXP_GRP="new_normalclamppolymix3_mlpnormenc_polyinv_polyind"
EXCLUDE_SIM_IDS=(0 4 8) # exclude IDs for ind loss being 0
# EXCLUDE_SIM_IDS=(1 2 3 5 6 7 9 10 11)
DATA_SEEDS=({42..61})

# List of experiment IDs
EXP_IDS=(
    "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.001lam3_ms100"
    "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.01lam3_ms100"
    "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.1lam3_ms100"
)

# Loop through each experiment ID and run the summary script
for EXP_ID in "${EXP_IDS[@]}"; do
    echo "Summarizing seeds for: $EXP_ID"

    args=(
        --exp_id "$EXP_ID"
        --exp_grp "$EXP_GRP"
        --ckpt_strategy "$CKPT_STRATEGY"
        --metric_key "$METRIC_KEY"
        --data_seeds "${DATA_SEEDS[@]}"
        --exclude_sim_ids "${EXCLUDE_SIM_IDS[@]}"
        --method "$METHOD"
        --compute_extras
    )

    # Run the command using the array
    python summarize_seeds.py "${args[@]}"
        
    echo "--------------------------------------"
done

EXP_GROUPS=(
    "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_{0.001,0.01,0.1}lam3_ms100"
)

for EXP_GROUP_PATTERN in "${EXP_GROUPS[@]}"; do
    echo "Making boxplot for group: $EXP_GROUP_PATTERN"

    args=(
        --exp_ids $(eval echo $EXP_GROUP_PATTERN)
        --exp_grp "$EXP_GRP"
        --ckpt_strategy "$CKPT_STRATEGY"
        --metric_key "$METRIC_KEY"
        --exclude_sim_ids "${EXCLUDE_SIM_IDS[@]}"
        --pop_num -1
    )

    # Run the command using the array
    python make_boxplot.py "${args[@]}"
    
    echo "--------------------------------------"
done
