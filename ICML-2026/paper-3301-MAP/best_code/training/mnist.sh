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

# Non-lifted common args (noise_level stays 0.0). Add --image flag if your driver expects it.
COMMON_ARGS="--noise_level=0.0 --problem=mnist --epochs=1000 --hidden_dim=1024 --time_conditioning=default --time_embed_dim=64 --batch_size=32 --learning_rate=1e-3 --include_num_samples_in_ckpt"

NOISE_LEVELS=(1.0)
DATASET_SIZES=(10000)

echo "Starting experiments for mnist at $(date)"
mkdir -p logs

# Run each MNIST experiment with the fixed reproducibility seed
for data_size in "${DATASET_SIZES[@]}"; do
  # Run lifted DDPM for each noise level
  for nl in "${NOISE_LEVELS[@]}"; do
    nl_fname=${nl//./_}
    echo "Running DDPM (lifted, data_size=${data_size}, noise_level=${nl}, seed=42)..."
    python driver.py --trainer=DDPM --num_samples=${data_size} --noise_level=${nl} --problem=mnist --epochs=1000 --lifted --hidden_dim=1024 --time_conditioning=default --time_embed_dim=64 --batch_size=32 --learning_rate=1e-3 --seed=42 --include_num_samples_in_ckpt --no_wandb > logs/DDPM_LIFTED_samples_${data_size}_noise_${nl_fname}_seed_42.log 2>&1
    echo "Finished DDPM (lifted, data_size=${data_size}, noise=${nl}, seed=42)"
  done

  for trainer in DDPM_NONPROJECT PIDM; do
    echo "Running ${trainer} (data_size=${data_size}, seed=42)..."
    python driver.py --trainer=${trainer} --num_samples=${data_size} ${COMMON_ARGS} --seed=42 --no_wandb > logs/${trainer}_samples_${data_size}_seed_42.log 2>&1
    echo "Finished ${trainer} (data_size=${data_size}, seed=42)"
  done
done

echo "All runs completed successfully at $(date)"
