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

PROTEIN_FRAGMENTS_PATH="data/protein/casp12_fragments_L10_N20000.npz"
if [ ! -f "$PROTEIN_FRAGMENTS_PATH" ]; then
  echo "Protein fragments not found at $PROTEIN_FRAGMENTS_PATH"
  echo "Generating SidechainNet fragment archive and manifest..."
  python training/process_protein_fragments.py --name casp12 --fragment-length 10 --max-data-length 20000
fi

# Non-lifted common args (noise_level stays 0.0)
COMMON_ARGS_BASE="--num_samples=20000 --noise_level=0.0 --problem=protein --hidden_dim=1024 --time_embed_dim=16 --time_conditioning=default --batch_size=32 --learning_rate=1e-3 --protein_fragments_path=${PROTEIN_FRAGMENTS_PATH}"

# NOISE_LEVELS=(0.01 0.05 0.1 0.5 1.0)

echo "Starting experiments for protein at $(date)"
mkdir -p logs

# # Run each protein experiment with the fixed reproducibility seed
# for nl in "${NOISE_LEVELS[@]}"; do
#   nl_fname=${nl//./_}
#   echo "Running DDPM (lifted, noise_level=${nl}, epochs=1000, seed=42)..."
#   python driver.py --trainer=DDPM --num_samples=20000 --noise_level=${nl} --problem=protein --epochs=1000 --lifted --hidden_dim=1024 --time_embed_dim=16 --time_conditioning=default --batch_size=32 --learning_rate=1e-3 --protein_fragments_path=${PROTEIN_FRAGMENTS_PATH} --seed=42 --no_wandb > logs/DDPM_LIFTED_noise_${nl_fname}_seed_42.log 2>&1
#   echo "Finished DDPM (lifted, noise=${nl}, epochs=1000, seed=42)"
# done

# echo "Running DDPM_NONPROJECT (epochs=1000, seed=42)..."
# python driver.py --trainer=DDPM_NONPROJECT ${COMMON_ARGS_BASE} --epochs=1000 --seed=42 --no_wandb > logs/DDPM_NONPROJECT_seed_42.log 2>&1
# echo "Finished DDPM_NONPROJECT (epochs=1000, seed=42)"

echo "Running PIDM (epochs=1000, seed=42)..."
python driver.py --trainer=PIDM ${COMMON_ARGS_BASE} --epochs=1000 --seed=42 --no_wandb > logs/PIDM_seed_42.log 2>&1
echo "Finished PIDM (epochs=1000, seed=42)"

echo "All runs completed successfully at $(date)"

# Verify and optionally publish the processed protein artifact.
# By default we only verify checksums. To auto-publish set AUTO_PUBLISH_PROTEIN=1
PUBLISH_SCRIPT="$PARENT_DIR/scripts/publish_protein_artifact.sh"
if [ -x "$PUBLISH_SCRIPT" ] || [ -f "$PUBLISH_SCRIPT" ]; then
  echo "Verifying protein artifact checksum..."
  if "$PUBLISH_SCRIPT" --check-only; then
    if [ "${AUTO_PUBLISH_PROTEIN:-0}" = "1" ]; then
      echo "AUTO_PUBLISH_PROTEIN=1 => attempting to publish artifact to GitHub release"
      if "$PUBLISH_SCRIPT" "${PROTEIN_RELEASE_TAG:-protein-data-v1}"; then
        echo "Protein artifact published (release: ${PROTEIN_RELEASE_TAG:-protein-data-v1})"
      else
        echo "Protein artifact checksum verified but publish failed. Run $PUBLISH_SCRIPT manually." >&2
      fi
    else
      echo "Protein artifact checksum verified. To publish, set AUTO_PUBLISH_PROTEIN=1 or run: $PUBLISH_SCRIPT <release-tag>"
    fi
  else
    echo "Protein artifact verification failed or artifact missing; see training/process_protein_fragments.py" >&2
  fi
else
  echo "Publish helper not found at $PUBLISH_SCRIPT; skipping artifact verification/publish." >&2
fi
