#!/bin/bash
#SBATCH --job-name=roberta_s13
#SBATCH --account=gts-smussmann3
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/roberta_s13_%j.out
#SBATCH --error=logs/roberta_s13_%j.err

cd /storage/project/r-smussmann3-0/kkang68/projects/cost-sensitive-learning
mkdir -p logs

echo "=============================================="
echo "Re-running failed RoBERTa alpha_balanced seed 13"
echo "=============================================="

python -m src.runners.run_experiment \
    --dataset jigsaw \
    --model roberta \
    --method classification \
    --weighting alpha_balanced \
    --seed 13

echo "Done!"
