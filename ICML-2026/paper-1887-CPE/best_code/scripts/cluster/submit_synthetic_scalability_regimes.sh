#!/bin/bash
#SBATCH --job-name=cape_scale
#SBATCH --output=/scratch3/bon136/wellbeing/results/cape_scale_%A_%a.out
#SBATCH --error=/scratch3/bon136/wellbeing/results/cape_scale_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=OD-233666

source ~/causalpe-env-torch/bin/activate

POLICIES=("eig" "uncertainty" "random" "static_eig")
DS=(10 20 50)
SEEDS=($(seq 1 10))

# Total tasks = |DS| * |POLICIES| * |SEEDS|
NUM_POLICIES=${#POLICIES[@]}
NUM_DS=${#DS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_DS * NUM_POLICIES * NUM_SEEDS))

# Derive indices
IDX=$((SLURM_ARRAY_TASK_ID - 1))

D_INDEX=$((IDX / (NUM_POLICIES * NUM_SEEDS)))
REM=$((IDX % (NUM_POLICIES * NUM_SEEDS)))
POLICY_INDEX=$((REM / NUM_SEEDS))
SEED_INDEX=$((REM % NUM_SEEDS))

D=${DS[$D_INDEX]}
POLICY=${POLICIES[$POLICY_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running D=$D policy=$POLICY seed=$SEED"

python -u run_synthetic_scalability_regimes.py \
  --D "$D" \
  --policy "$POLICY" \
  --seed "$SEED" \
  --outroot ./results/scalability \
  --expected_indegree 5.0 \
  --max_edge_prob_true 0.25
