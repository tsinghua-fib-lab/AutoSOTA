

models=("Qwen/Qwen2.5-32B-Instruct")

# for long CoT
lora_path="UWNSL/Qwen2.5-32B-Instruct_Long_CoT_lora"

# for short CoT
# lora_path="UWNSL/Qwen2.5-32B-Instruct_Short_CoT_lora"

tasks=("AIME" "AMC" "Olympiad" "gsm8k_cot_zeroshot" "hendrycks_math")
# replace hendrycks_math with hendrycks_math_500 if it takes a lot of time for long CoT 

max_gen_tokens=16000
model_args="dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=2,tensor_parallel_size=2"
batch_size="auto"
output_path="long_cot_vs_short_cot"

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running lm_eval with model: $model, lora: $lora_path, task: $task"
        lm_eval --model vllm \
            --model_args pretrained="$model",lora_local_path="$lora_path",enable_lora=True,$model_args \
            --gen_kwargs do_sample=false,temperature=0,max_gen_toks=$max_gen_tokens \
            --tasks "$task" \
            --batch_size "$batch_size" \
            --log_samples \
            --trust_remote_code \
            --output_path "$output_path" \
            --apply_chat_template \


        SANTIZED_MODEL_SAVE_LABEL=$(echo ${model} | sed 's/\//__/g')
        echo ${SANTIZED_MODEL_SAVE_LABEL}
        if [ "$task" != "gsm8k_cot_zeroshot" ]; then
            python math_metric_llm_eval_general.py --directory_path ${output_path}/${SANTIZED_MODEL_SAVE_LABEL} --task ${task}
        elif [ "$task" == "gsm8k_cot_zeroshot" ]; then
            python math_metric_gsm8k.py --directory_path ${output_path}/${SANTIZED_MODEL_SAVE_LABEL} 
        fi

        
    done
done




