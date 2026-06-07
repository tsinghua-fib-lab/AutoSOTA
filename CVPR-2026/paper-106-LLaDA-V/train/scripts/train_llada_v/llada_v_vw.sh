export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

num_node=$1
gpu_num=$2

# need to change num_node and gpu_num! 
# Configuration note: This script is typically run with 8 nodes and 8 GPUs per node.
# The gradient_accumulation_steps should be adjusted based on your GPU count to maintain effective batch size.
# For example, with 8 GPUs, set gradient_accumulation_steps=32.

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

LLM_VERSION="exp/llada_v_ov2M"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="llava_llada"

BASE_RUN_NAME="llada_v_si_ov_fur_viweb"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "data/visualwebinstruct/mixed_conversation.json" \
    --image_folder "data/visualwebinstruct/images" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "exp/$BASE_RUN_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 4 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --use_conversation_mask False