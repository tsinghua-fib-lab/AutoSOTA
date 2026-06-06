[ -z "$EPOCH" ] && EPOCH=3
[ -z "$DATASET" ] && DATASET=gsm8k_math_cot
[ -z "$SEED" ] && SEED=42
[ -z "$RUN_NAME" ] && RUN_NAME=deepseek_math_7b_${DATASET//,/_}_eps${EPOCH}_seed${SEED}
[ -z "$MODEL_PATH" ] && MODEL_PATH=pretrained_model_path

DS_CONFIG_PATH=examples/deepspeed/ds_z2_config.json
OUTPUT_PATH=saves/${RUN_NAME}

set -x

llamafactory-cli train \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn fa2 \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET \
    --template empty \
    --seed $SEED \
    --finetuning_type full \
    --preprocessing_num_workers 8 \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 180000000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --cutoff_len 4096 \
    --save_steps 180000000 \
    --plot_loss \
    --num_train_epochs $EPOCH \
    --bf16 \
    --save_only_model \
    --report_to wandb \
    --run_name $RUN_NAME

rm -rf $OUTPUT_PATH/checkpoint*