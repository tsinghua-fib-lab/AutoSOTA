#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ─────────────────────────  CONFIG  ─────────────────────────
AVAILABLE_GPUS=(6)                 # physical GPU IDs
CONCURRENCY_PER_GPU=1              # → TOTAL_SLOTS = 1, jobs run round-robin
NUM_GPUS=${#AVAILABLE_GPUS[@]}
TOTAL_SLOTS=$(( NUM_GPUS * CONCURRENCY_PER_GPU ))

CONDA_ENV_NAME='partial'              # conda env with your code
ENTRY_SCRIPT='train.py'            # ← changed from uot_alae.py
RUN_DIR='runs_uot'                      # log directory
mkdir -p "$RUN_DIR"

# ─────────────────────────  PYTHON  ─────────────────────────
if ! command -v conda &>/dev/null; then
  echo 'Error: conda not in PATH' >&2 ; exit 1
fi
CONDA_BASE=$(conda info --base 2>/dev/null)
PYTHON_EXE="$CONDA_BASE/envs/$CONDA_ENV_NAME/bin/python"
[[ -x $PYTHON_EXE ]] || {
  PYTHON_EXE=$(conda run -n "$CONDA_ENV_NAME" which python) || {
    echo "Cannot find python in env '$CONDA_ENV_NAME'." >&2 ; exit 1; }
}
echo "Using Python : $PYTHON_EXE"
echo "Using GPUs   : ${AVAILABLE_GPUS[*]}  (total $NUM_GPUS)"

command -v parallel >/dev/null   || { echo "GNU Parallel missing." >&2; exit 1; }
[[ -f $ENTRY_SCRIPT ]]           || { echo "$ENTRY_SCRIPT not found in $(pwd)." >&2; exit 1; }

# ─────────────────────  BUILD ARGUMENT LIST  ─────────────────
ARGS_FILE=$(mktemp)
trap 'rm -f "$ARGS_FILE"' EXIT

declare -a EXPERIMENTS=(
  # SW
  '--loss_type sw  --exp_name sw'
  # TWD
  '--loss_type twd --exp_name twd'

  # Partial-TWD
  '--loss_type twd --twd_unbalanced --exp_name partial_twd_03 --max_mass_generated 0.3'
  '--loss_type twd --twd_unbalanced --exp_name partial_twd_04 --max_mass_generated 0.4'
  '--loss_type twd --twd_unbalanced --exp_name partial_twd_05 --max_mass_generated 0.5'
  '--loss_type twd --twd_unbalanced --exp_name partial_twd_06 --max_mass_generated 0.6'
  '--loss_type twd --twd_unbalanced --exp_name partial_twd_07 --max_mass_generated 0.7'
  '--loss_type twd --twd_unbalanced --exp_name partial_twd_08 --max_mass_generated 0.8'
  '--loss_type twd --twd_unbalanced --exp_name partial_twd_09 --max_mass_generated 0.9'

  # Faster-UOT (POT)
  '--loss_type pot --exp_name faster_uot_02_02 --pot_reg 0.2 --pot_reg_m_kl 0.2'
  '--loss_type pot --exp_name faster_uot_03_03   --pot_reg 0.3  --pot_reg_m_kl 0.3'
  '--loss_type pot --exp_name faster_uot_04_04   --pot_reg 0.4  --pot_reg_m_kl 0.4'
  '--loss_type pot --exp_name faster_uot_05_05   --pot_reg 0.5  --pot_reg_m_kl 0.5'
  '--loss_type pot --exp_name faster_uot_07_07   --pot_reg 0.7  --pot_reg_m_kl 0.7'
  '--loss_type pot --exp_name faster_uot_09_09   --pot_reg 0.9  --pot_reg_m_kl 0.9'
  '--loss_type pot --exp_name faster_uot_05_01  --pot_reg 0.5  --pot_reg_m_kl 0.1'
  '--loss_type pot --exp_name faster_uot_01_05  --pot_reg 0.1 --pot_reg_m_kl 0.5'
  '--loss_type pot --exp_name faster_uot_03_05  --pot_reg 0.3 --pot_reg_m_kl 0.5'
  '--loss_type pot --exp_name faster_uot_05_03  --pot_reg 0.5 --pot_reg_m_kl 0.3'
  '--loss_type pot --exp_name faster_uot_09_01  --pot_reg 0.9 --pot_reg_m_kl 0.1'
  '--loss_type pot --exp_name faster_uot_01_09  --pot_reg 0.1 --pot_reg_m_kl 0.9'


  # PAWL  (k = 256 … 10)
  # best
  '--loss_type pawl --exp_name pawl_256 --pawl_k 256'
  '--loss_type pawl --exp_name pawl_200 --pawl_k 200'
  '--loss_type pawl --exp_name pawl_100 --pawl_k 100'
  '--loss_type pawl --exp_name pawl_75  --pawl_k 75'
  '--loss_type pawl --exp_name pawl_50  --pawl_k 50'
  '--loss_type pawl --exp_name pawl_25  --pawl_k 25'
  '--loss_type pawl --exp_name pawl_10  --pawl_k 10'

  # USOT (ρ₁, ρ₂)
  '--loss_type usot --exp_name usot_001_001 --rho1 0.01 --rho2 0.01'
  '--loss_type usot --exp_name usot_1_1     --rho1 1    --rho2 1'
  '--loss_type usot --exp_name usot_100_100 --rho1 100  --rho2 100'
  '--loss_type usot --exp_name usot_200_200 --rho1 200  --rho2 200'
  '--loss_type usot --exp_name usot_300_300 --rho1 300  --rho2 300'
  '--loss_type usot --exp_name usot_400_400 --rho1 400  --rho2 400'
  '--loss_type usot --exp_name usot_500_500 --rho1 500  --rho2 500'
  '--loss_type usot --exp_name usot_1000_1000 --rho1 1000  --rho2 1000'
  '--loss_type usot --exp_name usot_10000_10000 --rho1 10000  --rho2 10000'
  '--loss_type usot --exp_name usot_1000_1 --rho1 1000  --rho2 1'
  '--loss_type usot --exp_name usot_1_1000 --rho1 1  --rho2 1000'
  
  # SUOT (ρ₁, ρ₂)
  '--loss_type suot --exp_name suot_001_001 --rho1 0.01 --rho2 0.01'
  '--loss_type suot --exp_name suot_1_1     --rho1 1    --rho2 1'
  '--loss_type suot --exp_name suot_100_100 --rho1 100  --rho2 100'
  '--loss_type suot --exp_name suot_200_200 --rho1 200  --rho2 200'
  '--loss_type suot --exp_name suot_300_300 --rho1 300  --rho2 300'
  '--loss_type suot --exp_name suot_400_400 --rho1 400  --rho2 400'
  '--loss_type suot --exp_name suot_500_500 --rho1 500  --rho2 500'
  '--loss_type suot --exp_name suot_1000_1000 --rho1 1000  --rho2 1000'
  '--loss_type suot --exp_name suot_10000_10000 --rho1 10000  --rho2 10000'
  '--loss_type suot --exp_name suot_1000_1 --rho1 1000  --rho2 1'
  '--loss_type suot --exp_name suot_1_1000 --rho1 1  --rho2 1000'

  # SOPT
  '--loss_type sopt --exp_name sopt_001 --sopt_reg 0.01'
  '--loss_type sopt --exp_name sopt_005 --sopt_reg 0.05'
  '--loss_type sopt --exp_name sopt_01  --sopt_reg 0.1'
  '--loss_type sopt --exp_name sopt_05  --sopt_reg 0.5'
  '--loss_type sopt --exp_name sopt_1   --sopt_reg 1'
  '--loss_type sopt --exp_name sopt_10  --sopt_reg 10'
  '--loss_type sopt --exp_name sopt_100 --sopt_reg 100'

  # SPOT
  '--loss_type spot --exp_name spot_256 --spot_k 256'
  '--loss_type spot --exp_name spot_150 --spot_k 150'
  '--loss_type spot --exp_name spot_160 --spot_k 160'
  '--loss_type spot --exp_name spot_170 --spot_k 170'
  '--loss_type spot --exp_name spot_180 --spot_k 180'
  '--loss_type spot --exp_name spot_190 --spot_k 190'
  '--loss_type spot --exp_name spot_210 --spot_k 210'
  '--loss_type spot --exp_name spot_220 --spot_k 220'
  '--loss_type spot --exp_name spot_230 --spot_k 230'
  '--loss_type spot --exp_name spot_200 --spot_k 200'
  '--loss_type spot --exp_name spot_100 --spot_k 100'
  '--loss_type spot --exp_name spot_75  --spot_k 75'
  '--loss_type spot --exp_name spot_50  --spot_k 50'
  '--loss_type spot --exp_name spot_25  --spot_k 25'
  '--loss_type spot --exp_name spot_10  --spot_k 10'
)

# Write each experiment to ARGS_FILE
for args in "${EXPERIMENTS[@]}"; do
  echo "$args" >>"$ARGS_FILE"
done

echo -e "\n=== Argument lines (cat -A) ==="
cat -A "$ARGS_FILE"
echo '================================'

export ENTRY_SCRIPT PYTHON_EXE NUM_GPUS RUN_DIR AVAILABLE_GPUS_STR="${AVAILABLE_GPUS[*]}"

# ─────────────────────  GNU PARALLEL LAUNCH  ─────────────────
parallel -j "$TOTAL_SLOTS" --line-buffer --halt soon,fail=1 \
  --joblog "$RUN_DIR/joblog.tsv" \
  '
    slot={%}      # 1 … TOTAL_SLOTS   –> "worker slot"
    job={#}       # 1 … #lines        –> unique job counter

    # Map slot → real GPU:
    GPUS=($AVAILABLE_GPUS_STR)
    export CUDA_VISIBLE_DEVICES="${GPUS[(( (slot-1) % NUM_GPUS ))]}"

    out="$RUN_DIR/job_${job}_stdout.log"
    err="$RUN_DIR/job_${job}_stderr.log"

    ARGS=$(echo {} | tr -d "'\''")   # strip GNU Parallel quotes
    echo "[Job $job / Slot $slot | GPU $CUDA_VISIBLE_DEVICES] $PYTHON_EXE $ENTRY_SCRIPT $ARGS"
    eval "$PYTHON_EXE $ENTRY_SCRIPT $ARGS" >"$out" 2>"$err"
  ' :::: "$ARGS_FILE"

echo "✓ All jobs finished. Logs in '$RUN_DIR' and $RUN_DIR/joblog.tsv"
