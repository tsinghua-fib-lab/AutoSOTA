export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

num_node=$1
gpu_num=$2

# need to change num_node and gpu_num! 
# in our case, we use 1 node and 4 gpus

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

echo "master_addr ${MASTER_ADDR}"
echo "master_port ${MASTER_PORT}"
echo "node_rank ${RANK}"
echo "gpu_num ${gpu_num}"
echo "num_node ${num_node}"

LLM_VERSION="model/LLaDA-8B-Instruct-HF/"

LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
############### Pretrain ################

PROMPT_VERSION=llada_plain

BASE_RUN_NAME="llada_v_pretrain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node=${gpu_num} --nnodes=${num_node} --master_addr=${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank=${RANK} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "data/llava_pretrain/blip_laion_cc_sbu_558k.json" \
    --image_folder "data/llava_pretrain/images" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "exp/$BASE_RUN_NAME" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa