#!/bin/bash
#SBATCH -J macaque_kfold_multiday  # job name
#SBATCH -p gpu-l40                 # partition
#SBATCH -N 1                       # nodes
#SBATCH -n 4                       # tasks
#SBATCH --gres=gpu:4               # gpus (adjust per job)
#SBATCH -c 16                      # cpus per task
#SBATCH -t 0-04:00:00              # time (d-hh:mm:ss)
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=user@domain.com
#SBATCH --output=/path/to/codejob_outputs/%j

# Root directory for data/code/experiments
ROOT_DIR="/path/to/code"
SCRIPT_HELPER="${ROOT_DIR}/code/scripts/python/run_macaque_multiday_cv.py"

show_help() {
  cat <<'EOF' 
Usage: sbatch macaque_kfold_multiday.sh --model=MODEL --days=DAY_SPEC [options] [-- EXTRA_ARGS...]

Required:
  --model=MODEL        One of: cebra, marble, vdw, lfads
  --days=DAY_SPEC      Day list e.g. "0-9,34-43" (inclusive ranges) or "3"

Optional:
  --config=PATH        Base experiment YAML (default: config/yaml_files/macaque/experiment.yaml)
  --model_config=PATH  Model YAML to merge (ESC GNN only; default: config/yaml_files/macaque/vdw_supcon_2.yaml)
  --save_dir_base=DIR  Base save dir (default: <ROOT_DIR>/experiments/macaque_reaching)
  --conda_env=NAME     Conda env to activate (default: torch-env3)
  --dry_run            Print commands only
  -h, --help           Show this help

All args after "--" are passed through to the model runner (e.g., --patience 10).

GPU allocation:
  The script detects available GPUs (CUDA_VISIBLE_DEVICES or nvidia-smi) and splits
  the requested days evenly across GPUs. Each GPU runs its assigned days sequentially.
EOF
}

# Defaults
CONDA_ENV="torch-env3"
MODEL=""
DAYS_SPEC=""
BASE_CONFIG=""
MODEL_CONFIG=""
SAVE_DIR_BASE=""
DRY_RUN=false

PASS_THROUGH=()

# Parse arguments (stop at --)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model=*)
      MODEL="${1#*=}"; shift ;;
    --days=*)
      DAYS_SPEC="${1#*=}"; shift ;;
    --config=*)
      BASE_CONFIG="${1#*=}"; shift ;;
    --model_config=*)
      MODEL_CONFIG="${1#*=}"; shift ;;
    --save_dir_base=*)
      SAVE_DIR_BASE="${1#*=}"; shift ;;
    --conda_env=*)
      CONDA_ENV="${1#*=}"; shift ;;
    --dry_run)
      DRY_RUN=true; shift ;;
    -h|--help)
      show_help; exit 0 ;;
    --)
      shift
      PASS_THROUGH=("$@")
      break ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$DAYS_SPEC" ]]; then
  echo "Error: --model and --days are required."
  show_help
  exit 1
fi

MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_LOWER" != "cebra" && "$MODEL_LOWER" != "marble" && "$MODEL_LOWER" != "vdw" && "$MODEL_LOWER" != "lfads" ]]; then
  echo "Error: --model must be one of cebra|marble|vdw|lfads"
  exit 1
fi

# SLURM environment
. ~/.bashrc
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=1
export TQDM_DISABLE=1

# Determine GPU list
GPU_ENV=${CUDA_VISIBLE_DEVICES:-}
if [[ -n "$GPU_ENV" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_ENV"
else
  GPU_COUNT=$(nvidia-smi -L | wc -l)
  GPU_IDS=()
  for ((i=0; i<GPU_COUNT; i++)); do GPU_IDS+=("$i"); done
fi
NUM_GPUS=${#GPU_IDS[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
  echo "No GPUs detected."
  exit 1
fi

echo "[INFO] GPUs detected: ${GPU_IDS[*]}"

# Build CSV GPU string (do not rely on env inside python)
GPU_IDS_STR=$(IFS=,; echo "${GPU_IDS[*]}")

# Split days across GPUs using a helper python snippet (contiguous chunks)
DAY_GROUPS=$(python3 - "${DAYS_SPEC}" "${GPU_IDS_STR}" <<'PY'
import sys
import math

def parse_days(spec: str):
    days = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            days.update(range(a, b + 1))
        else:
            days.add(int(part))
    return sorted(days)

if len(sys.argv) < 3:
    print("ERROR: missing DAY_SPEC or GPU_IDS", file=sys.stderr)
    sys.exit(1)

spec = sys.argv[1]
gpu_str = sys.argv[2]
gpu_ids = [g for g in gpu_str.split(",") if g != ""]
days = parse_days(spec)

# Assign contiguous chunks of days to GPUs (not round-robin)
num_gpus = max(len(gpu_ids), 1)
total = len(days)
base = total // num_gpus
extra = total % num_gpus
groups = []
start = 0
for i in range(num_gpus):
    take = base + (1 if i < extra else 0)
    end = start + take
    groups.append(days[start:end])
    start = end

for g in groups:
    print(",".join(str(x) for x in g))
PY
)

# Split parsed days string into GPU groups
mapfile -t GROUP_ARRAY <<< "${DAY_GROUPS}"

if [[ ${#GROUP_ARRAY[@]} -eq 0 ]]; then
  echo "[ERROR] No day groups produced from DAYS_SPEC='${DAYS_SPEC}'."
  echo "[ERROR] Helper output was: ${DAY_GROUPS}"
  exit 1
fi

# Sanity: if helper returned fewer groups than GPUs, pad with empty strings
while [[ ${#GROUP_ARRAY[@]} -lt ${#GPU_IDS[@]} ]]; do
  GROUP_ARRAY+=("")
done

# Define function to launch python script by GPU group
launch_gpu_group() {
  local gpu_id="$1"
  local day_csv="$2"
  shift 2
  if [[ -z "$day_csv" ]]; then
    echo "[INFO] GPU $gpu_id: no days assigned, skipping."
    return
  fi
  local env_prefix="CUDA_VISIBLE_DEVICES=${gpu_id}"
  local cmd=(
    python3 "$SCRIPT_HELPER"
    --model "$MODEL_LOWER"
    --days "$day_csv"
    --root_dir "$ROOT_DIR"
  )
  [[ -n "$BASE_CONFIG" ]] && cmd+=(--base_config "$BASE_CONFIG")
  [[ -n "$MODEL_CONFIG" ]] && cmd+=(--model_config "$MODEL_CONFIG")
  [[ -n "$SAVE_DIR_BASE" ]] && cmd+=(--save_dir_base "$SAVE_DIR_BASE")
  $DRY_RUN && cmd+=(--dry_run)
  if [[ ${#PASS_THROUGH[@]} -gt 0 ]]; then
    cmd+=(-- "${PASS_THROUGH[@]}")
  fi

  echo "[INFO] GPU ${gpu_id} days {${day_csv}} -> ${cmd[*]}"
  if ! $DRY_RUN; then
    eval "$env_prefix ${cmd[*]}" &
  fi
}

# Launch each GPU group with its assigned days
for idx in "${!GPU_IDS[@]}"; do
  launch_gpu_group "${GPU_IDS[$idx]}" "${GROUP_ARRAY[$idx]}"
done

if ! $DRY_RUN; then
  wait
fi

