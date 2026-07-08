#!/bin/bash
#SBATCH --job-name=cb50_hitl
#SBATCH --output=/scratch3/bon136/wellbeing/results/cb50_%A_%a.out
#SBATCH --error=/scratch3/bon136/wellbeing/results/cb50_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=OD-233666

# shellcheck disable=SC1090
source ~/causalpe-env/bin/activate

# --- Parameters ---
# POLICIES=("eig" "uncertainty" "random")
POLICIES=("static_eig" "static_uncertainty" "static_random")

# shellcheck disable=SC2207
SEEDS=($(seq 1 10))

# Total array jobs = num policies * num seeds
# Submit with: sbatch --array=1-30 cb50_array.sh
IDX=$((SLURM_ARRAY_TASK_ID - 1))
POLICY_INDEX=$((IDX / ${#SEEDS[@]}))
SEED_INDEX=$((IDX % ${#SEEDS[@]}))

POLICY=${POLICIES[$POLICY_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

# --- Paths ---
DATASET_NPZ=data/causalbench/exports/weissmann_k562_50.npz
OUTROOT=./results/cb50
OUTDIR=${OUTROOT}/${POLICY}
mkdir -p "$OUTDIR"

echo "Running CausalBench50 policy=$POLICY seed=$SEED"
echo "Dataset: $DATASET_NPZ"
echo "Outdir:  $OUTDIR"

# --- Run single job (one policy, one seed) ---
python -u causalbench_hitl_causal_dpo.py \
  --dataset_npz "$DATASET_NPZ" \
  --outdir "$OUTDIR" \
  --policy "$POLICY" \
  --seed "$SEED" \
  --S 1000 \
  --T 200 \
  --screen_k 800 \
  --resample_threshold 0.5 \
  --rejuvenate_samples \
  --rejuvenate_steps 2 \
  --max_parents 3 \
  --corr_screen_k 8 \
  --ridge 1e-2 \
  --beta_edge 10.0 \
  --beta_dir 10.0 \
  --lam 0.0

