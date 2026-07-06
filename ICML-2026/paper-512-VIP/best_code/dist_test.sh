CONFIG=$1

if [ -z "$CONFIG" ]; then
    echo "Usage: bash $0 <config> [extra args...]"
    exit 1
fi

WORK_DIR_ROOT=${WORK_DIR:-"./work_logs"}
CONFIG_NAME=$(basename "${CONFIG%.*}")
DATASET_NAME=${CONFIG_NAME#cfg_}
WORK_DIR="${WORK_DIR_ROOT}/${DATASET_NAME}"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29700}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS=${GPUS:-6}
VISIBLE_GPUS=${VISIBLE_GPUS:-"1,2,3,5,6,7"}

mkdir -p "$WORK_DIR"

export CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/eval_seg.py \
    --config "$CONFIG" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    ${@:4}
