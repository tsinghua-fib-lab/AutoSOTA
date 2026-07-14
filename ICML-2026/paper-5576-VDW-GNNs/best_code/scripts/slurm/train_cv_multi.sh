#!/bin/bash
#SBATCH -J cv_multi_models      # job name
#SBATCH -p gpu-l40              # queue (partition)
#SBATCH -N 1                    # number of nodes requested
#SBATCH -n 1                    # number of tasks (MPI processes)
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --gres=gpu:1            # request gpu(s)
#SBATCH -c 16                   # cpus per task
#SBATCH -t 0-04:00:00           # run time (d-hh:mm:ss)
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=user@domain.com
#SBATCH --output=/path/to/codejob_outputs/%j

# Notes:
: << 'END'
Run k-fold CV for multiple models found in a directory.
For each model YAML in the directory, we run accelerate launch on main_training.py.
The directory should also contain an experiment.yaml to provide shared settings.

Example usage:
sbatch scripts/slurm/train_cv_multi.sh \
--root_dir=/path/to/code/ \
--config_dir=code/config/yaml_files/ellipsoids \
--dataset=ellipsoids \
--k_folds=5 \
--target=random_harmonic_normals \
--models=vdw,vdw_ablate_vec,legs,gcn,gat,gin,egnn \

AUTOMATED ARGUMENTS:
(1) Setting --target=diameter (default) will use the flag 
  --invariant_pred 
  for the 'vdw' model (no ablations).

(2) Setting --target=random_harmonic_normals will cause the script to use 
  --task_key=vector_node_regression \
  --target_key=y_random_harmonic_normals \ 
  --target_dim=3
  For 'vdw' model, --scalar_feat_key=x will also be used here, to prevent collision between x and pos (scalar and vector features, respectively).
END

ROOT_DIR="/path/to/code"

# Defaults
CONFIG_DIR=""
DATASET=""
MODELS_LIST=""    # comma-separated model keys (without .yaml); if empty, use all
K_FOLDS=""        # optional override for number of CV folds
TARGET="diameter" # dataset target, e.g. random_harmonic_normals
NUM_GPUS=1
VERBOSITY=0
MIXED_PRECISION="no"
CONDA_ENV="torch-env3"
DEBUG_DDP=false
SUBSAMPLE_N=""
N_EPOCHS=""
LEARN_RATE=""
BATCH_SIZE=""
PRETRAINED_WEIGHTS_DIR=""

show_help() {
  echo "Usage: sbatch $0 --config_dir=DIR --dataset=DATASET [--models=gin,gat,gcn] [options]"
  echo ""
  echo "Required:"
  echo "  --config_dir=DIR      Directory containing model YAML files and experiment.yaml"
  echo "                        (model-specific YAMLs used are parsed based on --models)."
  echo "  --dataset=DATASET     Dataset key (e.g., ellipsoids, qm9)"
  echo ""
  echo "Options:"
  echo "  --models=gin,gat,gcn  Only run these models (filenames: gcn.yaml, gat.yaml, etc.)"
  echo "  --target=TARGET       Optional dataset target (e.g., random_harmonic_normals)"
  echo "  --root_dir=PATH       Project root (default: /path/to/code/)"
  echo "                        If --config_dir is relative, it is resolved against this root"
  echo "  --num_gpus=N          Number of GPUs (default: 1)"
  echo "  --verbosity=LEVEL     Verbosity (default: 0)"
  echo "  --mixed_precision=K   no|fp16|bf16 (default: no)"
  echo "  --subsample_n=N       Limit samples for quick tests"
  echo "  --n_epochs=N          Override n_epochs"
  echo "  --k_folds=N           Override number of CV folds (default: 5)"
  echo "  --learn_rate=LR       Override learning rate"
  echo "  --batch_size=BS       Override batch size"
  echo "  --pretrained_weights_dir=PATH  Directory with pretrained weights"
  echo "  --conda_env=ENV       Conda environment (default: torch-env3)"
  echo "  --debug_ddp           Enable verbose NCCL/DDP logs"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --config_dir=*) CONFIG_DIR="${1#*=}"; shift ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --models=*) MODELS_LIST="${1#*=}"; shift ;;
    --target=*) TARGET="${1#*=}"; shift ;;
    --root_dir=*) ROOT_DIR="${1#*=}"; shift ;;
    --num_gpus=*) NUM_GPUS="${1#*=}"; shift ;;
    --verbosity=*) VERBOSITY="${1#*=}"; shift ;;
    --mixed_precision=*) MIXED_PRECISION="${1#*=}"; shift ;;
    --subsample_n=*) SUBSAMPLE_N="${1#*=}"; shift ;;
    --n_epochs=*) N_EPOCHS="${1#*=}"; shift ;;
    --k_folds=*) K_FOLDS="${1#*=}"; shift ;;
    --learn_rate=*) LEARN_RATE="${1#*=}"; shift ;;
    --batch_size=*) BATCH_SIZE="${1#*=}"; shift ;;
    --pretrained_weights_dir=*) PRETRAINED_WEIGHTS_DIR="${1#*=}"; shift ;;
    --conda_env=*) CONDA_ENV="${1#*=}"; shift ;;
    --debug_ddp) DEBUG_DDP=true; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

if [[ -z "$CONFIG_DIR" || -z "$DATASET" ]]; then
  echo "Error: --config_dir and --dataset are required."
  show_help
  exit 1
fi

# Normalize CONFIG_DIR relative to ROOT_DIR when needed.
# - If CONFIG_DIR is absolute, leave it as-is
# - If CONFIG_DIR already starts with ROOT_DIR, leave it as-is
# - Otherwise, prefix ROOT_DIR
if [[ "$CONFIG_DIR" = /* ]]; then
  :
elif [[ "$CONFIG_DIR" == "$ROOT_DIR"* ]]; then
  :
else
  CONFIG_DIR="${ROOT_DIR%/}/${CONFIG_DIR#/}"
fi

# --- Cluster environment setup ---
. ~/.bashrc

# Useful environment variables
export OMP_NUM_THREADS=1
export TQDM_DISABLE=1

if [ "$DEBUG_DDP" = true ]; then
  export NCCL_DEBUG=INFO
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_SHOW_CPP_STACKTRACES=1
fi

# Activate conda environment
conda activate "$CONDA_ENV"

# Ensure own python files/modules can be imported in other python files
export PYTHONPATH="${ROOT_DIR}/code":$PYTHONPATH

# Build list of YAMLs and associated model keys
MODEL_YAMLS=( )
MODEL_KEYS=( )

# Determine if special task args should be attached
ATTACHED_ARGS=""
if [[ "$DATASET" == "ellipsoids" && "$TARGET" == "random_harmonic_normals" ]]; then
  ATTACHED_ARGS="--task_key=vector_node_regression --target_key=y_random_harmonic_normals --target_dim=3"
fi

# Normalize and detect whether vdw_ablate_vec was requested
VDW_ABLATE_VEC_IN_LIST=false
REQUESTED_MODELS=( )
if [[ -n "$MODELS_LIST" ]]; then
  # remove all whitespace and split on commas
  MODELS_LIST="${MODELS_LIST//[[:space:]]/}"
  IFS=',' read -r -a REQUESTED_MODELS <<< "$MODELS_LIST"
  for m in "${REQUESTED_MODELS[@]}"; do
    if [[ "$m" == "vdw_ablate_vec" ]]; then
      VDW_ABLATE_VEC_IN_LIST=true
      break
    fi
  done
fi

# Select model YAML files based on model names and target
if [[ -n "$MODELS_LIST" ]]; then
  # Filter by requested model names
  for m in "${REQUESTED_MODELS[@]}"; do
    if [[ "$m" == "vdw_ablate_vec" ]]; then
      cand="${CONFIG_DIR%/}/vdw.yaml"
    elif [[ "$m" == "tfn" ]]; then
      if [[ "$TARGET" == "diameter" ]]; then
        cand="${CONFIG_DIR%/}/tfn_diameter.yaml"
      elif [[ "$TARGET" == "random_harmonic_normals" ]]; then
        cand="${CONFIG_DIR%/}/tfn_normals.yaml"
      else
        cand="${CONFIG_DIR%/}/tfn.yaml"
      fi
      # fallback if chosen variant doesn't exist
      if [[ ! -f "$cand" ]]; then
        alt="${CONFIG_DIR%/}/tfn.yaml"
        [[ -f "$alt" ]] && cand="$alt"
      fi
    elif [[ "$m" == "egnn" ]]; then
      if [[ "$TARGET" == "diameter" ]]; then
        cand="${CONFIG_DIR%/}/egnn_diameter.yaml"
      elif [[ "$TARGET" == "random_harmonic_normals" ]]; then
        cand="${CONFIG_DIR%/}/egnn_normals.yaml"
      else
        cand="${CONFIG_DIR%/}/egnn.yaml"
      fi
      # fallback if chosen variant doesn't exist
      if [[ ! -f "$cand" ]]; then
        alt="${CONFIG_DIR%/}/egnn.yaml"
        [[ -f "$alt" ]] && cand="$alt"
      fi
    else
      cand="${CONFIG_DIR%/}/$m.yaml"
    fi
    if [[ -f "$cand" ]]; then
      MODEL_YAMLS+=("$cand")
      MODEL_KEYS+=("$m")
    else
      echo "Warning: requested model '$m' not found as $cand"
    fi
  done
else
  # Collect all (exclude experiment.yaml)
  while IFS= read -r -d '' file; do
    base=$(basename "$file")
    name_no_ext="${base%.yaml}"

    # Target-specific filtering for TFN/EGNN variants
    if [[ "$name_no_ext" == tfn_* ]]; then
      if [[ "$TARGET" == "diameter" && "$name_no_ext" != "tfn_diameter" ]]; then
        continue
      fi
      if [[ "$TARGET" == "random_harmonic_normals" && "$name_no_ext" != "tfn_normals" ]]; then
        continue
      fi
    fi
    if [[ "$name_no_ext" == egnn_* ]]; then
      if [[ "$TARGET" == "diameter" && "$name_no_ext" != "egnn_diameter" ]]; then
        continue
      fi
      if [[ "$TARGET" == "random_harmonic_normals" && "$name_no_ext" != "egnn_normals" ]]; then
        continue
      fi
    fi
    # If generic files exist alongside variants, prefer variants and skip generic
    if [[ "$name_no_ext" == "tfn" ]]; then
      if [[ "$TARGET" == "diameter" && -f "${CONFIG_DIR%/}/tfn_diameter.yaml" ]]; then
        continue
      fi
      if [[ "$TARGET" == "random_harmonic_normals" && -f "${CONFIG_DIR%/}/tfn_normals.yaml" ]]; then
        continue
      fi
    fi
    if [[ "$name_no_ext" == "egnn" ]]; then
      if [[ "$TARGET" == "diameter" && -f "${CONFIG_DIR%/}/egnn_diameter.yaml" ]]; then
        continue
      fi
      if [[ "$TARGET" == "random_harmonic_normals" && -f "${CONFIG_DIR%/}/egnn_normals.yaml" ]]; then
        continue
      fi
    fi

    MODEL_YAMLS+=("$file")
    MODEL_KEYS+=("$name_no_ext")
  done < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.yaml" ! -name "experiment.yaml" -print0)
fi

if [ ${#MODEL_YAMLS[@]} -eq 0 ]; then
  echo "No model YAML files found to run."
  exit 1
fi

# Loop through each model YAML file and run CV training
for idx in "${!MODEL_YAMLS[@]}"; do
  MODEL_YAML="${MODEL_YAMLS[$idx]}"
  MODEL_KEY="${MODEL_KEYS[$idx]}"
  echo "\n--- Launching CV run for: $MODEL_YAML ---\n"
  MODEL_EXTRA_ARGS=""
  if [[ "$MODEL_KEY" == "vdw_ablate_vec" ]]; then
    MODEL_EXTRA_ARGS+=" --ablate_vector_track"
  fi
  if [[ "$DATASET" == "ellipsoids" && "$MODEL_KEY" == "vdw" ]]; then
    # avoid collision between x and pos (scalar and vector features)
    MODEL_EXTRA_ARGS+=" --scalar_feat_key x"
    if [[ "$TARGET" == "diameter" ]]; then
      # enable invariant prediction mode for diameter prediction
      MODEL_EXTRA_ARGS+=" --invariant_pred"
    fi
  fi
  # Determine accelerate launch arguments based on NUM_GPUS
  LAUNCH_FLAGS=""
  if [[ ${NUM_GPUS} -ge 2 ]]; then
    LAUNCH_FLAGS="--multi_gpu --num_processes=${NUM_GPUS}"
  else
    LAUNCH_FLAGS="--num_processes=1"
  fi

  accelerate launch \
    ${LAUNCH_FLAGS} \
    --num_machines=1 \
    --mixed_precision=${MIXED_PRECISION} \
    --dynamo_backend=no \
"${ROOT_DIR}/code/scripts/python/main_training.py" \
    --root_dir "$ROOT_DIR" \
    --config "$MODEL_YAML" \
    --dataset "$DATASET" \
    --experiment_type kfold \
    --verbosity "$VERBOSITY" \
    --dataloader_split_batches \
    $([ ! -z "$K_FOLDS" ] && echo "--k_folds $K_FOLDS") \
    $([ ! -z "$SUBSAMPLE_N" ] && echo "--subsample_n $SUBSAMPLE_N") \
    $([ ! -z "$N_EPOCHS" ] && echo "--n_epochs $N_EPOCHS") \
    $([ ! -z "$LEARN_RATE" ] && echo "--learn_rate $LEARN_RATE") \
    $([ ! -z "$BATCH_SIZE" ] && echo "--batch_size $BATCH_SIZE") \
    $([ ! -z "$PRETRAINED_WEIGHTS_DIR" ] && echo "--pretrained_weights_dir $PRETRAINED_WEIGHTS_DIR") \
    $ATTACHED_ARGS $MODEL_EXTRA_ARGS
done


