#!/bin/bash
#SBATCH -J qm9_vdw       # job name
#SBATCH -p gpu-l40          # queue (partition)
#SBATCH -N 1                # number of nodes requested
#SBATCH -n 4                # number of tasks (MPI processes)
#SBATCH --ntasks-per-node=4 # number of tasks per node
#SBATCH --gres=gpu:4        # request gpu(s)
#SBATCH -c 16               # cpus per task
#SBATCH -t 0-08:00:00       # run time (d-hh:mm:ss)
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=user@domain.com
#SBATCH --output=/path/to/codejob_outputs/%j

# Notes:
: << 'END'
Example slurm scheduler calls (note use of relative paths):

sbatch code/scripts/slurm/train.sh \
--config=ellipsoids/tfn.yaml \
--dataset=ellipsoids

sbatch code/scripts/slurm/train.sh \
--config=qm9/vdw.yaml \
--dataset=QM9

Old error (I think from trying to run 2 jobs with 2 GPUs each, on one 4-GPU node)
# /bsuhome/davejohnson408/miniforge3/envs/torch-env/lib/python3.11/site-packages/accelerate/utils/launch.py
File "/bsuhome/davejohnson408/miniforge3/envs/torch-env/lib/python3.11/site-packages/accelerate/utils/launch.py", line 217, in prepare_multi_gpu_env
     raise ConnectionError(
 ConnectionError: Tried to launch distributed communication on port `29500`, but another process is utilizing it. Please specify a different port (such as using the `--main_process_port` flag or specifying a different `main_process_port` in your config file) and rerun your script. To automatically use the next open port (on a single node), you can set this to `0`.
END

# Set root directory
ROOT_DIR="/path/to/code"

# Default values
NUM_GPUS=4  # Default to match slurm num processes/gpus
CONDA_ENV="torch-env3"  # Default conda environment
VERBOSITY=0  # Default verbosity (printed output) level
MIXED_PRECISION="no"  # Default to no mixed precision (doesn't work with sparse tensor cuda ops)
LEARN_RATE=""  # Optional learning rate override
SUBSAMPLE_N=""  # Default to empty (to use all data)
N_EPOCHS=""  # Default to empty (use config value)
DEBUG_DDP=false  # Default: do not enable verbose NCCL/DDP logging
SNAPSHOT_PATH=""  # Default: no snapshot resume
BATCH_SIZE=""  # Default to empty (use config value)
PRETRAINED_WEIGHTS_DIR=""  # Directory with pretrained weights

# Function to display help message
show_help() {
    echo "Usage: sbatch $0 [options]"
    echo ""
    echo "Required options (new run):"
    echo "  --config=CONFIG         YAML config to start a new experiment"
    echo "                          (in <root_dir>/config/yaml_files/)."
    echo "                          For resuming with --snapshot_path you may omit --config."
    echo "  --dataset=DATASET_NAME  Dataset key (still required)."
    echo ""
    echo "Optional options:"
    echo "  --num_gpus=NUM_GPUS     Maximum number of GPUs to use (default: 4)"
    echo "                          Will use fewer if fewer are available"
    echo "  --verbosity=LEVEL       Set verbosity level (default: 0)"
    echo "                          Higher values provide more detailed output"
    echo "  --mixed_precision=TYPE  Set mixed precision type (default: no)"
    echo "                          Options: no, fp16, bf16"
    echo "  --learn_rate=LR        Override learning rate in config file"
    echo "  --batch_size=BS        Override batch size in config file"
    echo "  --subsample_n=N      Temporarily use only N samples for debugging DDP (default: use all)"
    echo "                          Overrides subsample_n from config if specified"
    echo "  --n_epochs=N_EPOCHS     Set the maximum number of epochs (overrides n_epochs in config)"
    echo "  --debug_ddp           Enable verbose DDP/NCCL logging (default: off)"
    echo "  --snapshot_path=PATH  Full path to snapshot directory or file for resuming training"
    echo "  --pretrained_weights_dir=PATH  Full path to directory with pretrained weights"
    echo "  --conda_env=ENV_NAME    Conda environment to activate (default: torch-env)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  sbatch $0 --config=qm9/vdw.yaml --dataset=QM9 --subsample_n=1024 --conda_env=myenv"
    echo ""
    echo "Note: The config file path will be automatically constructed as:"
    echo "      ${ROOT_DIR}/code/config/yaml_files/<CONFIG>"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
    --num_gpus=*)
      NUM_GPUS="${1#*=}"
      shift
      ;;
    --verbosity=*)
      VERBOSITY="${1#*=}"
      shift
      ;;
    --mixed_precision=*)
      MIXED_PRECISION="${1#*=}"
      shift
      ;;
    --learn_rate=*)
      LEARN_RATE="${1#*=}"
      shift
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --pretrained_weights_dir=*)
      PRETRAINED_WEIGHTS_DIR="${1#*=}"
      shift
      ;;
    --subsample_n=*)
      SUBSAMPLE_N="${1#*=}"
      shift
      ;;
    --n_epochs=*)
      N_EPOCHS="${1#*=}"
      shift
      ;;
    --debug_ddp)
      DEBUG_DDP=true
      shift
      ;;
    --snapshot_path=*)
      SNAPSHOT_PATH="${1#*=}"
      shift
      ;;
    --conda_env=*)
      CONDA_ENV="${1#*=}"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "${SNAPSHOT_PATH:-}" ] && [ -z "${CONFIG:-}" ]; then
    echo "Error: --config is required for a new run (no --snapshot_path provided)"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "${DATASET:-}" ]; then
    echo "Error: --dataset argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Validate mixed precision value
if [[ ! "$MIXED_PRECISION" =~ ^(no|fp16|bf16)$ ]]; then
    echo "Error: --mixed_precision must be one of: no, fp16, bf16"
    exit 1
fi

# execute bashrc stuff
. ~/.bashrc

# useful env variables
export OMP_NUM_THREADS=1
export TQDM_DISABLE=1

# Verbose NCCL/DPP logs (enabled only if --debug_ddp flag is passed)
if [ "$DEBUG_DDP" = true ]; then
  export NCCL_DEBUG=INFO                     # – full NCCL call logging
  export TORCH_DISTRIBUTED_DEBUG=DETAIL      # – DDP state-machine logs
  export NCCL_ASYNC_ERROR_HANDLING=1         # – propagates async errors
  # (optional) show per-rank stack traces when watchdog fires
  export TORCH_SHOW_CPP_STACKTRACES=1
fi

# if using conda environment
conda activate "$CONDA_ENV"

# if using python venv:
# module load python/3.9.7
# module load cudnn8.7-cuda11/8.7.0.84
# source /path/to/code.venv/bin/activate

# ensure own python files/modules can be imported in other python files
export PYTHONPATH="${ROOT_DIR}/code":$PYTHONPATH

# run training script with accelerate launch
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
  --root_dir "${ROOT_DIR}" \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --dataloader_split_batches \
  --verbosity "${VERBOSITY}" \
  $([ ! -z "${SUBSAMPLE_N}" ] && echo "--subsample_n ${SUBSAMPLE_N}") \
  $([ ! -z "${N_EPOCHS}" ] && echo "--n_epochs ${N_EPOCHS}") \
  $([ ! -z "${LEARN_RATE}" ] && echo "--learn_rate ${LEARN_RATE}") \
  $([ ! -z "${BATCH_SIZE}" ] && echo "--batch_size ${BATCH_SIZE}") \
  $([ ! -z "${PRETRAINED_WEIGHTS_DIR}" ] && echo "--pretrained_weights_dir ${PRETRAINED_WEIGHTS_DIR}") \
  $([ ! -z "${SNAPSHOT_PATH}" ] && echo "--snapshot_path ${SNAPSHOT_PATH}")
