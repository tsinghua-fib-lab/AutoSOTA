export CUDA_VISIBLE_DEVICES=0

python ./sdr/train.py \
    --model_path <model_path>  \
    --dataset_name squad \
    --num_remote 1 \
    --slm_name <slm_name> \
    --llm_names <llm_names> \
    --local_answer_paths data/<slm_name>/squad-validation.parquet \
    --remote_answer_paths data/<llm_names>/squad-validation.parquet \
    --embed_path <embed_path> \
    --output_dir <output_dir> \
    --head_model mlp \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 100 \
    --warmup_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 0.01  \
    --bf16 \
    --seed 42 \
    --save_total_limit 2 \
    --multi_remote_strategy head \