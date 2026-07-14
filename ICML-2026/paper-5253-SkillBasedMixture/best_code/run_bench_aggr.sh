#!/bin/bash

LOG_FILE="aggr_bench_runs.log"
echo "Starting model runs at $(date)" > $LOG_FILE

MODELS=('Llama' 'LlamaR1' 'Qwen' 'Mistral' 'Phi' 'Gemma' 'GLM' 'Exaone' 'Granite' 'QwenMath' 'QwenCode' 'DeepSeekMath' 'QwenR1' 'InternLM' 'Mathstral' 'BioLlama')
for MODEL in "${MODELS[@]}"; do
    echo "=====================================" | tee -a $LOG_FILE
    echo "Starting $MODEL at $(date)" | tee -a $LOG_FILE
    echo "=====================================" | tee -a $LOG_FILE
    
    CUDA_VISIBLE_DEVICES=3,7 python bench_aggr.py --agent $MODEL --task AIME24 --gpus 2 2>&1 | tee -a $LOG_FILE
    
    if [ $? -eq 0 ]; then
        echo "$MODEL completed successfully" | tee -a $LOG_FILE
    else
        echo "ERROR: $MODEL failed" | tee -a $LOG_FILE
    fi
    
    sleep 5
    
    echo "" | tee -a $LOG_FILE
done

echo "All models completed at $(date)" | tee -a $LOG_FILE