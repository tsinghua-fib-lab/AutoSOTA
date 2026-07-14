#!/bin/bash
#SBATCH -J test_model         # job name
#SBATCH -p gpu-l40           # queue (partition)
#SBATCH -N 1                 # number of nodes requested
#SBATCH -n 1                 # number of tasks (MPI processes)
#SBATCH --gres=gpu:1         # request 1 gpu
#SBATCH -c 8                 # cpus per task
#SBATCH -t 0-00:15:00        # run time (d-hh:mm:ss)
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=user@domain.com
#SBATCH --output=/path/to/codejob_outputs/%j

# Set root directory
ROOT_DIR="/path/to/code"
CONDA_ENV="torch-env3"  # Default conda environment

# Function to display help message
show_help() {
    echo "Usage: sbatch $0 [--config=CONFIG_FILE] --dataset=DATASET_NAME --experiment_dir=EXPERIMENT_DIR [--conda_env=ENV_NAME]"
    echo ""
    echo "Required options:"
    echo "  (none)"
    echo "  --dataset=DATASET_NAME      Specify the dataset name"
    echo "  --experiment_dir=EXPERIMENT_DIR  Path to experiment directory (containing models/, config/, etc.)"
    echo ""
    echo "Optional options:"
    echo "  --config=CONFIG_FILE        Manually specify a config file (overrides default newest-in-experiment behavior)"
    echo "  --conda_env=ENV_NAME        Conda environment to activate (default: torch-env)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Let the script pick the newest config automatically"
    echo "  sbatch $0 --dataset=QM9 --experiment_dir=/path/to/exp_dir"
    echo ""
    echo "  # Override and supply a specific config file"
    echo "  sbatch $0 --config=borah_qm9_vdw.yaml --dataset=QM9 --experiment_dir=/path/to/exp_dir --conda_env=myenv"
    echo ""
    echo "Note: If --config is omitted, the newest YAML file in <experiment_dir>/config/ is selected automatically."
}

# Parse command line arguments
for ARG in "$@"; do
  case $ARG in
    --config=*)
      CONFIG="${ARG#*=}"
      shift
      ;;
    --dataset=*)
      DATASET="${ARG#*=}"
      shift
      ;;
    --experiment_dir=*)
      EXPERIMENT_DIR="${ARG#*=}"
      shift
      ;;
    --conda_env=*)
      CONDA_ENV="${ARG#*=}"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $ARG"
      show_help
      exit 1
      ;;
  esac
  shift
 done

# Check required arguments
if [ -z "${DATASET:-}" ]; then
    echo "Error: --dataset argument is required"
    show_help
    exit 1
fi
if [ -z "${EXPERIMENT_DIR:-}" ]; then
    echo "Error: --experiment_dir argument is required"
    show_help
    exit 1
fi

# If CONFIG is not provided, use the newest config file in the experiment_dir/config/ directory
if [ -z "${CONFIG:-}" ]; then
    CONFIG_DIR="${EXPERIMENT_DIR}/config"
    if [ ! -d "$CONFIG_DIR" ]; then
        echo "Error: Config directory not found: $CONFIG_DIR"
        exit 1
    fi
    NEWEST_CONFIG=$(ls -t "$CONFIG_DIR"/*.yaml "$CONFIG_DIR"/*.yml 2>/dev/null | head -n 1)
    if [ -z "$NEWEST_CONFIG" ]; then
        echo "Error: No config YAML files found in $CONFIG_DIR"
        exit 1
    fi
    echo "No --config provided, using newest config: $NEWEST_CONFIG"
    CONFIG_PATH="$NEWEST_CONFIG"
else
    # Let the Python ConfigManager resolve relative configs (e.g., qm9/vdw.yaml)
    # using --root_dir and its internal default config_subdir (code/config/yaml_files)
    CONFIG_PATH="${CONFIG}"
fi

echo "Using config file at: $CONFIG_PATH"

# execute bashrc stuff
. ~/.bashrc

# useful env variables
export OMP_NUM_THREADS=1
export TQDM_DISABLE=1

# if using conda environment
conda activate "$CONDA_ENV"

# ensure own python files/modules can be imported in other python files
export PYTHONPATH="${ROOT_DIR}/code":$PYTHONPATH

# run test script with accelerate launch
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
"${ROOT_DIR}/code/scripts/python/test_model.py" \
  --root_dir "${ROOT_DIR}" \
  --config "$CONFIG_PATH" \
  --dataset "${DATASET}" \
  --experiment_dir "${EXPERIMENT_DIR}" 