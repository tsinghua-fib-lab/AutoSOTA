#!/bin/bash
#SBATCH --job-name=mdcrl_grid
#SBATCH --array=0-11
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --exclude=gpu286
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shimeng.huang@ista.ac.at

# --- Calculate sleep time: Task ID multiplied by 1 second ---
# If Task 0, sleep 0; If Task 1, sleep 1; ... 
SLEEP_TIME=$((SLURM_ARRAY_TASK_ID * 1))
echo "Staggering start: Task ${SLURM_ARRAY_TASK_ID} sleeping for ${SLEEP_TIME} seconds..."
sleep ${SLEEP_TIME}

# --- Load modules and environments ----

mkdir -p logs

module load cuda

unset PYENV_VERSION
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate rl4iv 
which python
python --version

echo "Checking GPU on $(hostname):"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
python -c "import torch; print('Torch version:', torch.__version__)"

echo "Slurm allocated GPUs: $SLURM_JOB_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# tmp files and wandb cache
# export WANDB_CACHE_DIR="/nfs/scistore19/locatgrp/shuang/scratch/wandb_cache"
export JOB_SCRATCH="/nfs/scistore19/locatgrp/shuang/scratch/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export WANDB_CACHE_DIR="${JOB_SCRATCH}/wandb_cache"
export TMPDIR="${JOB_SCRATCH}/python_tmp"
mkdir -p $WANDB_CACHE_DIR $TMPDIR
trap 'rm -rf "$JOB_SCRATCH"' EXIT
echo "Job started. Scratch is at $JOB_SCRATCH"
# export WANDB_MODE="offline" 
# to sync back: wandb sync --include-offline outputs/sweep_normalclamppolymix2_mlpnormenc_poly2inv_polyind_ms100-ds46

# --- Hyperparameter Arrays ---
lam1_vals=(1 5 10)
lam2_vals=(0 1 5 10)

# --- Calculate indices for the grid ---
# Task ID 0 -> lam1=1, lam2=0
# Task ID 1 -> lam1=1, lam2=1
# Task ID 2 -> lam1=1, lam2=5
# Task ID 3 -> lam1=1, lam2=10
# Task ID 4 -> lam1=5, lam2=0
# Task ID 5 -> lam1=5, lam2=1
# Task ID 6 -> lam1=5, lam2=5
# Task ID 7 -> lam1=5, lam2=10
# Task ID 8 -> lam1=10, lam2=0
# Task ID 9 -> lam1=10, lam2=1
# Task ID 10 -> lam1=10, lam2=5
# Task ID 11 -> lam1=10, lam2=10

idx=${SLURM_ARRAY_TASK_ID:-0}
len2=${#lam2_vals[@]} # Automatically gets '4'
idx1=$((idx / len2))
idx2=$((idx % len2))

L1=${lam1_vals[$idx1]}
L2=${lam2_vals[$idx2]}

# --- Execute with Overrides ---
# Added ${} for safety and fixed the missing $ on EXP_NAME
python train.py \
    data_seed="${DATA_SEED}" \
    mix_seed="${MIX_SEED}" \
    data.mixing_type="${MIX_TYPE}" \
    data.polymix_degree="${POLYMIX_DEGREE}" \
    data.invmlp_actfun="${INVMLP_ACTFUN:-leaky_relu}"\
    data.dim_v_true="${DV_TRUE}" \
    data.dim_w_true="${DW_TRUE}" \
    data.dim_z="${DZ}" \
    data.n_pop="${NPOP}" \
    data.n_train="${NTRAIN}" \
    encoder="${ENC_TYPE}" \
    decoder="${DEC_TYPE}" \
    model.dim_v="${DV}" \
    model.dim_w="${DW}" \
    loss.inv_loss_type="${INV_LOSS_TYPE:-poly}" \
    loss.inv_ker_poly_degree="${INV_KER_POLY_DEGREE:-2}" \
    loss.ind_loss_type="${IND_LOSS_TYPE:-poly}" \
    loss.ind_ker_poly_degree="${IND_KER_POLY_DEGREE:-2}" \
    loss.rbf_global_sigma="${GLOBAL_SIGMA:-false}" \
    loss.lam1="${L1}" \
    loss.lam2="${L2}" \
    loss.lam3="${L3:-0}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    exp_id="${EXP_NAME}" \
    sim_id="${idx}" \
    ${RESUME_ARGS}
  
