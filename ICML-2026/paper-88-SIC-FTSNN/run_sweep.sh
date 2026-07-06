#!/usr/bin/env bash
# Run SNN fault-ratio sweep from a shell script
# - Edit the "USER CONFIG" block or override via environment variables
# - Works for VGG/ResNet/MLP; picks correct depth flags automatically
# - Forwards any extra args you pass to the Python runner

set -euo pipefail

# ===== USER CONFIG (edit here or override as env vars) =====
PYTHON_BIN="${PYTHON_BIN:-python3}"
SWEEP_PY="${SWEEP_PY:-./auto_sweep.py}"        # <-- set the path to sweep_faults.py
SCRIPT="${SCRIPT:-./vgg_snn.py}"                  # <-- set the training script path (vgg_snn.py / resnet_snn.py / simple_snn.py)

NUM_STEPS="${NUM_STEPS:-2}"
FAULT_TYPE="${FAULT_TYPE:-stuck}"
R_START="${R_START:-0.0}"
R_STOP="${R_STOP:-0.5}"
R_STEP="${R_STEP:-0.1}"

OPTIONS="${OPTIONS:-baseline,ecoc,soft,routing}"  # comma-separated
DEVICES="${DEVICES:-0}"                         # e.g., "0,1" or "cpu"
MAX_PROCS="${MAX_PROCS:-4}"

EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-100}"
DATA_PATH="${DATA_PATH:-propdata/CIFAR10}"
RESULTS_DIR="${RESULTS_DIR:-results_auto_sweep}"

# For depth-enabled models only
VGG_DEPTHS="${VGG_DEPTHS:-7,11,15}"               # used if SCRIPT ends with vgg_snn.py
RESNET_DEPTHS="${RESNET_DEPTHS:-18,34}"           # used if SCRIPT ends with resnet_snn.py
# ==========================================================

usage() {
  cat <<'USAGE'
Usage:
  ./run_sweep.sh [extra sweep_faults.py args]

Edit the USER CONFIG block at the top (or set env vars) to control the run.
Common env var overrides:
  SWEEP_PY=./sweep_faults.py
  SCRIPT=./vgg_snn.py           # or ./resnet_snn.py / ./simple_snn.py
  NUM_STEPS=2 FAULT_TYPE=stuck R_START=0.0 R_STOP=0.5 R_STEP=0.1
  OPTIONS=baseline,ecoc,soft,routing
  DEVICES=0,1 MAX_PROCS=4
  EPOCHS=50 BATCH=100 DATA_PATH=/abs/path/to/dataset RESULTS_DIR=results_fault_sweep
  VGG_DEPTHS=7,11,15 RESNET_DEPTHS=18,34

Examples:
  # VGG (depth sweep) on GPUs 0,1
  SWEEP_PY=./sweep_faults.py SCRIPT=./vgg_snn.py VGG_DEPTHS=7,11,15 \
  DEVICES=0,1 MAX_PROCS=4 DATA_PATH=propdata/CIFAR10 \
  ./run_sweep.sh

  # ResNet (single depth) on GPU 0
  SWEEP_PY=./sweep_faults.py SCRIPT=./resnet_snn.py RESNET_DEPTHS=34 \
  DEVICES=0 MAX_PROCS=2 DATA_PATH=propdata/CIFAR100 \
  ./run_sweep.sh

  # MLP (no depth) on CPU
  SWEEP_PY=./sweep_faults.py SCRIPT=./simple_snn.py DEVICES=cpu MAX_PROCS=2 \
  DATA_PATH=propdata/FMNIST ./run_sweep.sh

  # Forward any extra args directly to the Python runner
  ./run_sweep.sh --results_dir my_runs --learning_rate 0.001
USAGE
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Basic checks
if [[ ! -f "$SWEEP_PY" ]]; then
  echo "[ERR] SWEEP_PY not found: $SWEEP_PY" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERR] Training SCRIPT not found: $SCRIPT" >&2
  exit 1
fi

script_base="$(basename "$SCRIPT")"
depth_args=()
case "$script_base" in
  vgg_snn.py)
    # Only pass depth flags for the matching model
    [[ -n "${VGG_DEPTHS:-}" ]] && depth_args+=(--vgg_depths "$VGG_DEPTHS")
    ;;
  resnet_snn.py)
    [[ -n "${RESNET_DEPTHS:-}" ]] && depth_args+=(--resnet_depths "$RESNET_DEPTHS")
    ;;
  simple_snn.py)
    # MLP: no depth flags
    ;;
  *)
    echo "[WARN] Unrecognized script name. Proceeding without depth flags: $script_base" >&2
    ;;
esac

cmd=(
  "$PYTHON_BIN" "$SWEEP_PY"
  --script "$SCRIPT"
  "${depth_args[@]}"
  --num_steps "$NUM_STEPS"
  --fault_type "$FAULT_TYPE"
  --fault_ratio_start "$R_START"
  --fault_ratio_stop "$R_STOP"
  --fault_ratio_step "$R_STEP"
  --options "$OPTIONS"
  --devices "$DEVICES"
  --max_procs "$MAX_PROCS"
  --epochs "$EPOCHS"
  --batch_size "$BATCH"
  --data_path "$DATA_PATH"
  --results_dir "$RESULTS_DIR"
  "$@"
)

echo "[RUN] ${cmd[*]}"
exec "${cmd[@]}"
