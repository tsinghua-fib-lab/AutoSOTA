#!/bin/bash
#SBATCH --job-name=wellbeing
#SBATCH --output=/scratch3/bon136/wellbeing/results/wellbeing_%A_%a.out
#SBATCH --error=/scratch3/bon136/wellbeing/results/wellbeing_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=OD-233666

# shellcheck disable=SC1090
source ~/causalpe-env/bin/activate

# --- Define your parameters here ---
POLICIES=("eig" "uncertainty" "random" "static_eig")

# shellcheck disable=SC2207
SEEDS=($(seq 1 10))

# Derive policy and seed index from SLURM_ARRAY_TASK_ID
IDX=$((SLURM_ARRAY_TASK_ID - 1))
POLICY_INDEX=$((IDX / ${#SEEDS[@]}))
SEED_INDEX=$((IDX % ${#SEEDS[@]}))

POLICY=${POLICIES[$POLICY_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

OUTDIR=./results/showcase/${POLICY}
mkdir -p $OUTDIR

echo "Running policy=$POLICY seed=$SEED"

python -u synthetic_hitl_causal_dpo.py \
    --outdir "$OUTDIR" \
    --D 20 \
    --S 10000 \
    --T 190 \
    --edge_prob_true 0.25 \
    --flip_prob 0.10 \
    --add_remove_prob 0.05 \
    --weight_noise 0.20 \
    --beta_edge 10.0 \
    --beta_dir 10.0 \
    --lam 0.0 \
    --screen_k 200 \
    --resample_threshold 0.6 \
    --policy "$POLICY" \
    --rejuvenate_samples \
    --rejuvenate_steps 2 \
    --seed "$SEED" \
    --save_prefix "${POLICY}"_seed"${SEED}"


