#!/bin/bash

LOG_FILE="create_profile_runs.log"
echo "Starting model runs at $(date)" > $LOG_FILE

MODELS=('Llama' 'LlamaR1' 'Qwen' 'Mistral' 'Phi' 'Gemma' 'GLM' 'Exaone' 'Granite' 'QwenMath' 'QwenCode' 'DeepSeekMath' 'QwenR1' 'InternLM' 'Mathstral' 'BioLlama')
for MODEL in "${MODELS[@]}"; do
    echo "=====================================" | tee -a $LOG_FILE
    echo "Starting $MODEL at $(date)" | tee -a $LOG_FILE
    echo "=====================================" | tee -a $LOG_FILE
    
    CUDA_VISIBLE_DEVICES=2,3,5,6 python create_profile.py --agent $MODEL --task GPQA --gpus 4 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        echo "$MODEL completed successfully" | tee -a $LOG_FILE
    else
        echo "ERROR: $MODEL failed" | tee -a $LOG_FILE
    fi
    
    sleep 5
    
    echo "" | tee -a $LOG_FILE
done

echo "All models completed at $(date)" | tee -a $LOG_FILE