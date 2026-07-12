export NCCL_TIMEOUT=1800000
export NCCL_P2P_LEVEL=NVL
# Suppress verbose NCCL and distributed training logs
export NCCL_DEBUG=WARN          
export NCCL_DEBUG_SUBSYS=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF
export TRANSFORMERS_VERBOSITY=warning
export HOST=$(hostname)
export WANDB_SERVICE_WAIT=120
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="configs/example_config.yaml"

deepspeed --num_gpus 1 \
    run.py $CONFIG_FILE