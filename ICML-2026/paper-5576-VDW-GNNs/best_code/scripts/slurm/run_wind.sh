#!/bin/bash
#SBATCH -J wind_runs             # job name
#SBATCH -p gpu-l40               # queue (partition)
#SBATCH -N 1                     # number of nodes requested
#SBATCH -n 1                     # number of tasks (MPI processes)
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --gres=gpu:1             # request gpu(s)
#SBATCH -c 16                    # cpus per task
#SBATCH -t 0-00:30:00            # run time (d-hh:mm:ss)
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=user@domain.com
#SBATCH --output=/path/to/codejob_outputs/%j

# Example CLI call of this script: 
# NOTES: 
# - use 'experiment.yaml' for --model dd-tnn 
# set --dataset=wind_rot for wind rotation dataset

# sbatch scripts/slurm/run_wind.sh --dataset wind --replications 5 --knn_k 3 --local_pca_k 10 --sample_n 2000 --mask_prop 0.3 --do_rotation_eval --model vdw_layer --config vdw.yaml --exp_name vdw


ROOT_DIR="/path/to/code"
CONDA_ENV="torch-env3"
PYTHON_BIN="python3"

show_help() {
  echo "Usage: sbatch $0 --model vdw|vdw_layer|dd-tnn|gcn|gat|gin|legs|egnn|tfn --sample_n LIST --mask_prop LIST [--root_dir PATH] [--conda_env ENV] [script args...]"
  echo ""
  echo "Runs run_wind_experiments.py (vdw/vdw_layer/gcn/gat/gin/legs/egnn/tfn) or run_wind_tnn.py (dd-tnn)."
  echo "All additional args are forwarded to the chosen Python script."
  echo "Rotation eval note: by default we rotate precomputed Q via Q_rot=(I⊗R)Q(I⊗R)^T; pass --reconstruct_Q_rot to rebuild Q on rotated geometry (legacy)."
  echo "sample_n and mask_prop lists are split across available GPUs, distributing"
  echo "the Cartesian-product combos evenly (contiguously) per GPU."
  echo ""
  echo "Examples:"
  echo "  sbatch $0 --model vdw \\"
  echo "    --root_dir /path/to/code \\"
  echo "    --config code/config/yaml_files/wind/vdw.yaml \\"
  echo "    --replications 5 --knn_k 3 --local_pca_k 16 --sample_n 100,200,300,400 --mask_prop 0.1,0.3,0.5"
}

MODEL=""
PASS_THROUGH_ARGS=()
SAMPLE_N_RAW=""
MASK_PROP_RAW=""
DATASET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model=*) MODEL="${1#*=}"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --root_dir=*) ROOT_DIR="${1#*=}"; shift ;;
    --root_dir) ROOT_DIR="$2"; shift 2 ;;
    --conda_env=*) CONDA_ENV="${1#*=}"; shift ;;
    --conda_env) CONDA_ENV="$2"; shift 2 ;;
    --sample_n=*) SAMPLE_N_RAW="${1#*=}"; shift ;;
    --sample_n) SAMPLE_N_RAW="$2"; shift 2 ;;
    --mask_prop=*) MASK_PROP_RAW="${1#*=}"; shift ;;
    --mask_prop) MASK_PROP_RAW="$2"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --dataset) DATASET="$2"; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    *) PASS_THROUGH_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required (vdw|vdw_layer|dd-tnn|gcn|gat|gin|legs|egnn|tfn)."
  show_help
  exit 1
fi

if [[ "$MODEL" != "vdw" && "$MODEL" != "vdw_layer" && "$MODEL" != "dd-tnn" && "$MODEL" != "gcn" && "$MODEL" != "gat" && "$MODEL" != "gin" && "$MODEL" != "legs" && "$MODEL" != "egnn" && "$MODEL" != "tfn" ]]; then
  echo "Error: --model must be one of 'vdw', 'vdw_layer', 'dd-tnn', 'gcn', 'gat', 'gin', 'legs', 'egnn', 'tfn'."
  show_help
  exit 1
fi

if [[ -z "$SAMPLE_N_RAW" || -z "$MASK_PROP_RAW" ]]; then
  echo "Error: --sample_n and --mask_prop are required (comma-separated lists)."
  show_help
  exit 1
fi

parse_csv() {
  # Trim spaces and split on commas; ignore empty entries.
  local raw="$1"
  local cleaned
  cleaned=$(echo "$raw" | tr -d '[:space:]')
  IFS=',' read -r -a arr <<< "$cleaned"
  local out=()
  for v in "${arr[@]}"; do
    [[ -n "$v" ]] && out+=("$v")
  done
  printf "%s\n" "${out[@]}"
}

mapfile -t SAMPLE_N_LIST < <(parse_csv "$SAMPLE_N_RAW")
mapfile -t MASK_PROP_LIST < <(parse_csv "$MASK_PROP_RAW")

if [[ ${#SAMPLE_N_LIST[@]} -eq 0 || ${#MASK_PROP_LIST[@]} -eq 0 ]]; then
  echo "Error: could not parse non-empty --sample_n/--mask_prop lists."
  exit 1
fi

# Build Cartesian product combos.
COMBOS=()
for n in "${SAMPLE_N_LIST[@]}"; do
  for p in "${MASK_PROP_LIST[@]}"; do
    COMBOS+=("${n}:${p}")
  done
done

if [[ ${#COMBOS[@]} -eq 0 ]]; then
  echo "Error: no (sample_n, mask_prop) combinations generated."
  exit 1
fi

# Resolve ROOT_DIR to an absolute path and ensure it exists.
if [[ "$ROOT_DIR" != /* ]]; then
  if ! ROOT_DIR="$(cd "$ROOT_DIR" 2>/dev/null && pwd)"; then
    echo "Error: could not resolve ROOT_DIR '$ROOT_DIR'."
    exit 1
  fi
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: ROOT_DIR '$ROOT_DIR' does not exist."
  exit 1
fi

# Normalize --config to point into code/config/yaml_files/wind when the user
# passes just the filename (e.g., 'vdw.yaml').
CONFIG_BASE_REL="code/config/yaml_files/wind"
CONFIG_BASE="${ROOT_DIR}/${CONFIG_BASE_REL}"

resolve_config_value() {
  local val="$1"
  if [[ "$val" == /* ]]; then
    echo "$val"
  elif [[ "$val" != */* ]]; then
    echo "${CONFIG_BASE}/${val}"
  else
    echo "${ROOT_DIR}/${val}"
  fi
}

if [[ ${#PASS_THROUGH_ARGS[@]} -gt 0 ]]; then
  NORMALIZED_ARGS=()
  i=0
  while [[ $i -lt ${#PASS_THROUGH_ARGS[@]} ]]; do
    arg="${PASS_THROUGH_ARGS[$i]}"
    if [[ "$arg" == --config=* ]]; then
      raw="${arg#*=}"
      NORMALIZED_ARGS+=("--config=$(resolve_config_value "$raw")")
    elif [[ "$arg" == --config ]]; then
      next_idx=$((i + 1))
      if [[ $next_idx -lt ${#PASS_THROUGH_ARGS[@]} ]]; then
        raw="${PASS_THROUGH_ARGS[$next_idx]}"
        NORMALIZED_ARGS+=("--config" "$(resolve_config_value "$raw")")
        i=$next_idx
      else
        NORMALIZED_ARGS+=("$arg")
      fi
    else
      NORMALIZED_ARGS+=("$arg")
    fi
    ((i++))
  done
  PASS_THROUGH_ARGS=("${NORMALIZED_ARGS[@]}")
fi

# Environment setup
. ~/.bashrc
export OMP_NUM_THREADS=1
export TQDM_DISABLE=1
conda activate "$CONDA_ENV"
export PYTHONPATH="${ROOT_DIR}/code:${PYTHONPATH}"
cd "$ROOT_DIR" || exit 1

# Pick the appropriate script based on the model
if [[ "$MODEL" == "vdw" || "$MODEL" == "vdw_layer" || "$MODEL" == "gcn" || "$MODEL" == "gat" || "$MODEL" == "gin" || "$MODEL" == "legs" || "$MODEL" == "egnn" || "$MODEL" == "tfn" ]]; then
  PY_SCRIPT="code/scripts/python/run_wind_experiments.py"
elif [[ "$MODEL" == "dd-tnn" ]]; then
  PY_SCRIPT="code/scripts/python/run_wind_tnn.py"
fi

if [[ -z "$PY_SCRIPT" ]]; then
  echo "Error: no script resolved for model '$MODEL'."
  exit 1
fi

# Detect GPUs
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

# Chunk combos contiguously across GPUs
TOTAL_COMBOS=${#COMBOS[@]}
BASE=$((TOTAL_COMBOS / NUM_GPUS))
EXTRA=$((TOTAL_COMBOS % NUM_GPUS))

# Prepare base command without sample_n/mask_prop (added per combo)
CMD_BASE=("$PYTHON_BIN" "$PY_SCRIPT" "--root_dir" "$ROOT_DIR")
if [[ -n "$DATASET" ]]; then
  CMD_BASE+=("--dataset" "$DATASET")
fi
if [[ ${#PASS_THROUGH_ARGS[@]} -gt 0 ]]; then
  CMD_BASE+=("${PASS_THROUGH_ARGS[@]}")
fi

echo "[INFO] Total combinations: ${TOTAL_COMBOS}"
echo "[INFO] Distributing ~${BASE} combos per GPU (+1 for first ${EXTRA})"

launch_gpu_group() {
  local gpu_id="$1"
  shift
  local combos_for_gpu=("$@")
  if [[ ${#combos_for_gpu[@]} -eq 0 ]]; then
    echo "[INFO] GPU ${gpu_id}: no combos assigned, skipping."
    return
  fi

  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  for combo in "${combos_for_gpu[@]}"; do
    local n_val="${combo%%:*}"
    local p_val="${combo##*:}"
    local cmd=("${CMD_BASE[@]}" "--sample_n" "$n_val" "--mask_prop" "$p_val")
    echo "[INFO] GPU ${gpu_id}: (n=${n_val}, p=${p_val}) -> ${cmd[*]}"
    "${cmd[@]}"
  done
}

START=0
for idx in "${!GPU_IDS[@]}"; do
  COUNT=$BASE
  if [[ $idx -lt $EXTRA ]]; then
    COUNT=$((COUNT + 1))
  fi
  END=$((START + COUNT))
  SLICE=("${COMBOS[@]:START:COUNT}")
  launch_gpu_group "${GPU_IDS[$idx]}" "${SLICE[@]}" &
  START=$END
done

wait

