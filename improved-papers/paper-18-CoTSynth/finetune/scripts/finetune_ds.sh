
set -ex

run_finetune() {
    local LR=$1
    local NUM_GPUS=$2
    local MAX_SOURCE_LEN=$3
    local MAX_TARGET_LEN=$4
    local DEV_BATCH_SIZE=$5
    local GRAD_ACCUMULATION_STEPS=$6
    local MAX_STEP=$7
    local SAVE_INTERVAL=$8
    local RUN_NAME=$9
    local DATASET_PATH=${10}

    DATESTR=`date +%Y%m%d-%H%M%S`
    OUTPUT_DIR=OUTPUT_PATH/${RUN_NAME}-${DATESTR}
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)

    mkdir -p $OUTPUT_DIR

    torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
        --train_format input-output \
        --train_file $DATASET_PATH \
        --model_name_or_path $BASE_MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --max_source_length $MAX_SOURCE_LEN \
        --max_target_length $MAX_TARGET_LEN \
        --per_device_train_batch_size $DEV_BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
        --max_steps $MAX_STEP \
        --logging_steps 1 \
        --save_steps $SAVE_INTERVAL \
        --learning_rate $LR \
        --bf16 \
        --deepspeed configs/deepspeed.json 2>&1 | tee ${OUTPUT_DIR}/train.log
}

BASE_MODEL_PATH=BASE_MODEL_PATH

run_finetune 2e-6 8 3584 512 1 16 3120 520 RUN_NAME DATASET_PATH