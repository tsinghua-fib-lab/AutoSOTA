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
COMMON_ARGS="--num_samples=100000 --noise_level=0.0 --problem=smileyface_sphere --epochs=40 --hidden_dim=64 --time_concat --batch_size=128 --learning_rate=1e-3"

NOISE_LEVELS=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1.0)

echo "Starting experiments for smileyface_sphere using Normalizing Flows at $(date)"
mkdir -p logs

# Run each normalizing-flow experiment with the fixed reproducibility seed
echo "========================================="
echo "Seed: 42"
echo "========================================="

echo ""
echo "--- Lifted noise experiments (RealNVP/Glow) ---"

# Lifted noise experiments for RealNVP and Glow (batch_size=64)
for nl in "${NOISE_LEVELS[@]}"; do
  nl_fname=${nl//./_}
  echo "Noise level: ${nl}"

  # RealNVP lifted
  echo "  Running RealNVP (lifted, noise_level=${nl}, seed=42)..."
  echo "    Launching: python driver.py --trainer=REALNVP --lifted"
  python driver.py --trainer=REALNVP --num_samples=100000 --noise_level=${nl} --problem=smileyface_sphere --epochs=40 --lifted --hidden_dim=64 --time_concat --batch_size=128 --learning_rate=1e-3 --seed=42 --no_wandb > logs/REALNVP_LIFTED_noise_${nl_fname}_seed_42.log 2>&1
  # RealNVP isotropic
  echo "    Launching: python driver.py --trainer=REALNVP --isotropic"
  python driver.py --trainer=REALNVP --num_samples=100000 --noise_level=${nl} --problem=smileyface_sphere --epochs=40 --isotropic --hidden_dim=64 --time_concat --batch_size=128 --learning_rate=1e-3 --seed=42 --no_wandb > logs/REALNVP_ISOTROPIC_noise_${nl_fname}_seed_42.log 2>&1
  echo "  Finished RealNVP (noise=${nl}, seed=42)"

  # Glow lifted
  echo "  Running Glow (lifted, noise_level=${nl}, seed=42)..."
  echo "    Launching: python driver.py --trainer=GLOW --lifted"
  python driver.py --trainer=GLOW --num_samples=100000 --noise_level=${nl} --problem=smileyface_sphere --epochs=40 --lifted --hidden_dim=64 --time_concat --batch_size=128 --learning_rate=1e-3 --seed=42 --no_wandb > logs/GLOW_LIFTED_noise_${nl_fname}_seed_42.log 2>&1
  # Glow isotropic
  echo "    Launching: python driver.py --trainer=GLOW --isotropic"
  python driver.py --trainer=GLOW --num_samples=100000 --noise_level=${nl} --problem=smileyface_sphere --epochs=40 --isotropic --hidden_dim=64 --time_concat --batch_size=128 --learning_rate=1e-3 --seed=42 --no_wandb > logs/GLOW_ISOTROPIC_noise_${nl_fname}_seed_42.log 2>&1
  echo "  Finished Glow (noise=${nl}, seed=42)"

  echo ""
done

# RealNVP baseline (batch_size=64)
echo "Running RealNVP (seed=42)..."
echo "    Launching: python driver.py --trainer=REALNVP baseline"
python driver.py --trainer=REALNVP --num_samples=100000 --noise_level=0.0 --problem=smileyface_sphere --epochs=40 --hidden_dim=64 --time_concat --batch_size=128 --learning_rate=1e-3 --seed=42 --no_wandb > logs/REALNVP_seed_42.log 2>&1
echo "Finished RealNVP (seed=42)"

# Glow baseline (batch_size=64)
echo "Running Glow (seed=42)..."
echo "    Launching: python driver.py --trainer=GLOW baseline"
python driver.py --trainer=GLOW --num_samples=100000 --noise_level=0.0 --problem=smileyface_sphere --epochs=40 --hidden_dim=64 --time_concat --batch_size=128 --learning_rate=1e-3 --seed=42 --no_wandb > logs/GLOW_seed_42.log 2>&1
echo "Finished Glow (seed=42)"

echo "========================================="
echo "All runs completed successfully at $(date)"
echo "========================================="
echo ""
echo "Summary:"
echo "- 2 trainers:  RealNVP, Glow"
echo "- Fixed seed: 42"
echo "- Baseline (noise=0.0) + ${#NOISE_LEVELS[@]} lifted noise levels"
echo "- Total runs: $((2 * (1 + 2 * ${#NOISE_LEVELS[@]}))) experiments"
