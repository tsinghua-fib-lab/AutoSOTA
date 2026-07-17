#!/bin/bash
#SBATCH --job-name=ab_n100k
#SBATCH --account=gts-smussmann3
#SBATCH --partition=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ab_n100k_%A_%a.out
#SBATCH --error=logs/ab_n100k_%A_%a.err
#SBATCH --array=0-18

cd /storage/project/r-smussmann3-0/kkang68/projects/cost-sensitive-learning
mkdir -p logs

# 19 jobs total:
# 0-9: tfidf seeds [0,1,7,13,42,99,123,314,456,2024]
# 10-18: roberta seeds [1,7,13,42,99,123,314,456,2024] (skip 0, already done)

TFIDF_SEEDS=(0 1 7 13 42 99 123 314 456 2024)
ROBERTA_SEEDS=(1 7 13 42 99 123 314 456 2024)

if [ $SLURM_ARRAY_TASK_ID -lt 10 ]; then
    MODEL="tfidf"
    SEED=${TFIDF_SEEDS[$SLURM_ARRAY_TASK_ID]}
else
    MODEL="roberta"
    IDX=$((SLURM_ARRAY_TASK_ID - 10))
    SEED=${ROBERTA_SEEDS[$IDX]}
fi

echo "=============================================="
echo "MODEL: $MODEL"
echo "SEED: $SEED"
echo "SAMPLE_SIZE: 100000"
echo "WEIGHTING: alpha_balanced"
echo "NORMALIZE: True (set in src/tasks/classify.py)"
echo "SOLVER: saga, MAX_ITER: 1000 (defaults in src/models/text_embed.py)"
echo "=============================================="

python -m src.runners.run_experiment \
    --dataset jigsaw \
    --model $MODEL \
    --method classification \
    --weighting alpha_balanced \
    --seed $SEED \
    --sample_size 100000 \
    --output_dir results/jigsaw_norm_true
