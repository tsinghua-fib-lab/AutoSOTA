#!/usr/bin/env bash
#SBATCH --job-name=WLM_run
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --requeue
#
# Generic runlist-driven sbatch.
#
# Usage:
#   RUNS_DIR=<top_level_out_dir> sbatch --array=0-N scripts/sbatch_run.sbatch <runlist.tsv>
#
# Example:
#   RUNS_DIR=eb sbatch --array=0-14 scripts/sbatch_run.sbatch runlists/eb.tsv
#
# Runlist format (tab-separated; # and blank lines skipped):
#   master_tag  data_cfg  train_cfg  run_tag  seed  [extra args for train.py...]
#
# Cluster knobs (override on the sbatch invocation):
#   WLM_REPO     : path to repo root          (default: $PWD)
#   WLM_ENV      : conda/micromamba env name  (default: WLM)
#   RUNS_DIR     : top-level output dir       (default: runs)
#   DEVICE       : cuda | cpu                 (default: cuda)
#   WANDB_ON     : 1 to enable wandb          (default: 1)
#   REGEN_DATA   : 1 to force re-gen bundle   (default: 0)
#   SKIP_DONE    : 1 to skip if final.pt exists (default: 1)

set -euo pipefail

# --- Activate environment ---
# Prefer micromamba; fall back to conda. Override with WLM_ENV=<name>.
WLM_ENV="${WLM_ENV:-WLM}"
set +u
if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "${WLM_ENV}"
elif command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${WLM_ENV}"
else
    echo "ERROR: neither micromamba nor conda is on PATH" >&2
    exit 1
fi
set -u

# --- Locate repo ---
WLM_REPO="${WLM_REPO:-$PWD}"
if [[ ! -f "${WLM_REPO}/train.py" ]]; then
    echo "ERROR: WLM_REPO does not look like the repo root: ${WLM_REPO}" >&2
    echo "       (expected to find train.py there)" >&2
    echo "       Set WLM_REPO=<path> on the sbatch invocation." >&2
    exit 1
fi
cd "${WLM_REPO}"
mkdir -p logs

# --- Pick the runlist row ---
RUNLIST="${1:-}"
[[ -n "${RUNLIST}" && -f "${RUNLIST}" ]] || {
    echo "ERROR: runlist not found: ${RUNLIST:-<missing>}" >&2; exit 1
}

line="$(
  awk 'BEGIN{i=0}
       $0 ~ /^[[:space:]]*#/ {next}
       NF==0 {next}
       { if (i==ENVIRON["SLURM_ARRAY_TASK_ID"]) {print; exit} i++ }' \
    "${RUNLIST}"
)"
[[ -n "${line}" ]] || {
    echo "ERROR: no data line for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1
}

# Tab-split into the 5 fixed positional fields + everything else as one string.
master_tag="$(awk -F'\t' '{print $1}' <<< "${line}")"
data_cfg="$(  awk -F'\t' '{print $2}' <<< "${line}")"
train_cfg="$( awk -F'\t' '{print $3}' <<< "${line}")"
run_tag="$(   awk -F'\t' '{print $4}' <<< "${line}")"
seed="$(      awk -F'\t' '{print $5}' <<< "${line}")"
extra_args="$(awk -F'\t' '{for(i=6;i<=NF;i++) printf $i" "; print ""}' <<< "${line}")"

# --- Defaults forwarded to run_from_configs.sh ---
export RUNS_DIR="${RUNS_DIR:-runs}"
export DEVICE="${DEVICE:-cuda}"
export WANDB_ON="${WANDB_ON:-1}"
export REGEN_DATA="${REGEN_DATA:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# Local-disk wandb cache on the compute node (avoid hammering NFS).
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export WANDB_DIR="${SLURM_TMPDIR}"
    export WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb"
    export WANDB_CONFIG_DIR="${SLURM_TMPDIR}/.config/wandb"
    mkdir -p "${WANDB_DIR}"
fi

echo "==================================================="
echo "[run] runlist    = ${RUNLIST}"
echo "[run] task       = ${SLURM_ARRAY_TASK_ID}  job=${SLURM_JOB_ID}"
echo "[run] repo       = ${WLM_REPO}"
echo "[run] env        = ${WLM_ENV}"
echo "[run] RUNS_DIR   = ${RUNS_DIR}"
echo "[run] master_tag = ${master_tag}"
echo "[run] data_cfg   = ${data_cfg}"
echo "[run] train_cfg  = ${train_cfg}"
echo "[run] run_tag    = ${run_tag}"
echo "[run] seed       = ${seed}"
echo "[run] extra_args = ${extra_args}"
echo "[run] start at $(date)"
echo "==================================================="

# `${extra_args}` is intentionally unquoted so word-splitting forwards each
# token (e.g. `--holdout-marginals 1 3 5 7` becomes 5 separate argv tokens).
bash scripts/run_from_configs.sh \
    "${master_tag}" \
    "${data_cfg}" \
    "${train_cfg}" \
    "${run_tag}" \
    "${seed}" \
    ${extra_args}

echo "[run] done at $(date)"
