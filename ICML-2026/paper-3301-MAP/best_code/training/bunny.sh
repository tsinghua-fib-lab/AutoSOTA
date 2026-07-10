#!/usr/bin/env bash

set -euo pipefail

module load python
module load pytorch/2.8.0
export WANDB_ENTITY=katiekeegan-home

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
if [ -d "$PARENT_DIR/.venv" ]; then
  source "$PARENT_DIR/.venv/bin/activate"
else
  echo "Warning: Virtual environment not found at $PARENT_DIR/.venv"
  echo "Run 'bash setup.sh' to create it"
fi

# Non-lifted common args (noise_level stays 0.0)
COMMON_ARGS="--num_samples=100000 --noise_level=0.0 --problem=bunny --epochs=200 --hidden_dim=128 --batch_size=64 --time_concat --learning_rate=1e-3"

NOISE_LEVELS=(0.0001 0.0005 0.001 0.005 0.05 0.1 0.5 1.0)

echo "Starting experiments for bunny at $(date)"
mkdir -p logs

# Run baseline/non-lifted trainers with the fixed reproducibility seed
for trainer in DDPM_NONPROJECT PIDM; do
  echo "Running ${trainer} (seed=42)..."
  python driver.py --trainer=${trainer} ${COMMON_ARGS} --seed=42 --no_wandb > logs/${trainer}_seed_42.log 2>&1
  echo "Finished ${trainer} (seed=42)"
done

# Run lifted DDPM for each noise level
for nl in "${NOISE_LEVELS[@]}"; do
  nl_fname=${nl//./_}
  echo "Running DDPM (lifted, noise_level=${nl}, seed=42)..."
  python driver.py --trainer=DDPM --num_samples=100000 --noise_level=${nl} --problem=bunny --epochs=200 --lifted --hidden_dim=128 --batch_size=64 --time_concat --learning_rate=1e-3 --seed=42 --no_wandb > logs/DDPM_LIFTED_noise_${nl_fname}_seed_42.log 2>&1
  echo "Finished DDPM (lifted, noise=${nl}, seed=42)"
done

echo "All runs completed successfully at $(date)"
