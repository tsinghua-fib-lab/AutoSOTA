#!/bin/bash
# Evaluate all 20 seeds for paper 2529 reproduction
cd /repo

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export SLURM_JOB_ID=local
export SLURM_NODELIST=localhost
export HOSTNAME=localhost
export HF_HOME=/autosota_cache/hf

OUTDIR=results/repro_poly3
mkdir -p $OUTDIR

ALL_RESULTS="$OUTDIR/all_estimates.csv"
echo "seed,instrument,pop_num,estimate" > $ALL_RESULTS

for SEED in $(seq 42 61); do
    echo "Evaluating seed $SEED..."
    
    python evaluate.py         --exp_id repro_poly3         --data_seed $SEED         --ckpt_strategy best         --metric_key 'val/tot_loss'         --selection_mode min         --batch_size 3000 2>&1 | tail -5
    
    # Append hW estimates to combined results
    RESULT_FILE="results/repro_poly3-ds${SEED}_bestsim*_insample_estimates.csv"
    if [ -f $RESULT_FILE ]; then
        python3 -c "
import pandas as pd
df = pd.read_csv('$RESULT_FILE')
hw = df[df['instrument']=='hW']
for _, row in hw.iterrows():
    print(f\"${SEED},hW,{int(row['pop_num'])},{row['estimate']}\")
" >> $ALL_RESULTS
    fi
done

echo "All evaluations complete."
echo "Results saved to $ALL_RESULTS"

# Compute summary statistics
python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$ALL_RESULTS')

# True causal effect theta = 1.0
theta = 1.0

for pop in [-1, 0, 1]:
    sub = df[(df['instrument']=='hW') & (df['pop_num']==pop)]
    if len(sub) > 0:
        bias = sub['estimate'].values - theta
        mean_bias = np.mean(bias)
        sd_bias = np.std(bias, ddof=1)
        print(f'Pop {pop}: Mean bias={mean_bias:.6f}, SD={sd_bias:.6f}, N={len(sub)}')
        print(f'  Raw estimates: {sub["estimate"].values}')
"
