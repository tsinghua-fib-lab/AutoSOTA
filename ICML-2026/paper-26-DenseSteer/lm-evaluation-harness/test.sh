
models=("UWNSL/Qwen2.5-3B-Instruct_Long_CoT" "UWNSL/Qwen2.5-3B-Instruct_Short_CoT")
tasks=("AIME" "AMC" "Olympiad" "hendrycks_math_500" "gsm8k_cot_zeroshot" "hendrycks_math")

max_model_tokens=16000
max_gen_tokens=16000
model_args="tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=$max_model_tokens,dtype=bfloat16"
output_path="long_cot_vs_short_cot"
batch_size="auto"

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        echo "Running lm_eval with model: $model, task: $task"
        lm_eval --model vllm \
            --model_args pretrained="$model",$model_args \
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
