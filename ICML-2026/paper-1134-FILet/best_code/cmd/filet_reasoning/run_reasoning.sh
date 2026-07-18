export root= # Root Path
export env_path="$root/miniconda3/envs/{ENV_NAME}/bin/python"
export cache_path="$root/cache/"
export doc_path="$root/FILet---Learning-in-the-Fisher-Subspace-A-Guided-Initialization-for-LoRA-Fine-Tuning/"
export seed= # random seed
export rank=32
export lora_alpha=32
export data_num= # Initialization minibatch size
export num_train_epochs= # Number of Training Epochs
export num_warmup_steps= # Warmup Steps
export batch_size= # Batch Size
export max_length= # Max Length
export eval_steps= # Evaluation Steps
export learning_rate= # Learning Rate
export task_name= # Task Name (boolq, piqa, siqa, hellaswag, winogrande, arce, arcc, obqa)
export model_name_or_path= # Model name, e.g., meta-llama/Llama-2-7b-hf
export sxsy_on_gpu= # Whether to store the sxsy on GPU, which can speed up the computation but may cause out-of-memory issues.

CUDA_VISIBLE_DEVICES=0 $env_path $doc_path/run_reasoning.py \
--model_name_or_path $model_name_or_path \
--mode filet \
--version ver1 \
--task_name $task_name \
--per_device_train_batch_size $batch_size \
--per_device_eval_batch_size $batch_size \
--num_train_epochs $num_train_epochs \
--num_warmup_steps $num_warmup_steps \
--seed $seed \
--weight_decay 0.1 \
--eval_steps $eval_steps \
--learning_rate $learning_rate \
--max_length $max_length \
--data_num $data_num \
--lora_rank $rank \
--lora_alpha $lora_alpha \
--cache_dir $cache_path \
--log_dir $doc_path/logs/filet_reasoning/ \
--output_dir $doc_path/outputs/filet_reasoning/ \
--time 0 \
--sxsy_on_gpu $sxsy_on_gpu;

