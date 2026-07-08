#!/usr/bin/env bash
# Submit an experiment's Slurm job using your local config (env.local.sh).
#
#   ./submit.sh <experiment> [sbatch flags...]
#
# where <experiment> is one of: smoke, sbi, black_hole, super_resolution.
# Experiment knobs are passed as environment variables (exported to the job):
#
#   REINF_K=512 POSTERIOR=meanflow ./submit.sh black_hole --array=0-24
#   NUM_PARTICLES=256 ./submit.sh super_resolution --array=0-19
#   ./submit.sh smoke
#
# Reads env.local.sh (gitignored) for paths, $PY and $SLURM_PARTITION.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$REPO/env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO/env.local.sh"
else
  echo "warning: $REPO/env.local.sh not found — copy env.example.sh and fill it in." >&2
fi

[[ $# -ge 1 ]] || { echo "usage: ./submit.sh <smoke|sbi|black_hole|super_resolution> [sbatch flags...]" >&2; exit 2; }
exp="$1"; shift
script="$REPO/slurm/${exp}.sbatch"
[[ -f "$script" ]] || { echo "no such experiment script: $script" >&2; exit 2; }

exec sbatch \
  --partition="${SLURM_PARTITION:-gpu}" \
  --export=ALL \
  "$@" \
  "$script"
