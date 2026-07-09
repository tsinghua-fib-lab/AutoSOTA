function check_rval {
    if [[ $? -ne 0 ]]; then
        echo "❌ $1 failed."
        exit 1
    else
        echo "✅ $1 succeeded."
    fi
}

# cd ../../
check_rval "cd to project root"

#!/bin/bash
models=(   
  # "TinyLlama/TinyLlama_v1.1"
  # "google/gemma-2-2b"
  "meta-llama/Llama-2-7b-hf"
  # "meta-llama/Llama-2-13b-hf" 
  # "meta-llama/Llama-3.1-8B" 
  # "meta-llama/Llama-3.1-70B"
)

# Declare an associative array for token lengths.
declare -A token_length=(
  ["TinyLlama/TinyLlama_v1.1"]=2048
  ["google/gemma-2-2b"]=8192
  ["microsoft/Phi-3.5-mini-instruct"]=2048
  ["meta-llama/Llama-2-7b-hf"]=4096
  ["meta-llama/Llama-2-13b-hf"]=4096
  ["meta-llama/Llama-3.1-8B"]=4096
  ["meta-llama/Llama-3.1-70B"]=4096
)

# Declare an associative array for sequence lengths.
declare -A seq_length=(
  ["TinyLlama/TinyLlama_v1.1"]=2048
  ["google/gemma-2-2b"]=8192
  ["meta-llama/Llama-2-7b-hf"]=4096
  ["meta-llama/Llama-2-13b-hf"]=4096
  ["meta-llama/Llama-3.1-8B"]=4096
  ["meta-llama/Llama-3.1-70B"]=4096
)

# Define scaling modes as an array.
scaling_modes=("cholesky") #cholesky is for Qera-exact. you can choose from ('lqer', 'identity (ZeroQuantv2)', 'diag (Qera-approx)')

# Loop over each model.
for model_name in "${models[@]}"; do
  # Optional: Escape the model name if needed.
  model_name_esc=$(echo "$model_name" | sed 's/\//-/g')
  
  # Get token and sequence lengths for the model.
  current_token_length=${token_length["$model_name"]}
  current_seq_length=${seq_length["$model_name"]}
  
  # For each model, iterate over each scaling mode.
  for scaling_mode in "${scaling_modes[@]}"; do
    echo "Running for model: $model_name with token length: $current_token_length, sequence length: $current_seq_length, scaling mode: $scaling_mode"
    
    
    # Optional: add --apply-rand-svd for faster SRR initialization via randomized SVD
    python -u ptq_pipeline.py experiments/configs/srr_3bit_rank32_iter1.yaml \
      --model-name "$model_name" \
      --perplexity-eval-batch-size 1 \
      --max-position-embeddings "$current_token_length" \
      --perplexity-max-seq-length "$current_seq_length" \
      --lr-scaling-mode "$scaling_mode" \
      --num-calibration-samples 256 \
      --srr-seed 42 \
      --disable-lm-eval \
      -ow

    python -u ptq_pipeline.py experiments/configs/srr_3bit_rank32_iter1.yaml \
      --model-name "$model_name" \
      --perplexity-eval-batch-size 1 \
      --max-position-embeddings "$current_token_length" \
      --perplexity-max-seq-length "$current_seq_length" \
      --lr-scaling-mode "$scaling_mode" \
      --num-calibration-samples 256 \
      --srr-seed 1234 \
      --disable-lm-eval \
      -ow

    python -u ptq_pipeline.py experiments/configs/srr_3bit_rank32_iter1.yaml \
      --model-name "$model_name" \
      --perplexity-eval-batch-size 1 \
      --max-position-embeddings "$current_token_length" \
      --perplexity-max-seq-length "$current_seq_length" \
      --lr-scaling-mode "$scaling_mode" \
      --num-calibration-samples 256 \
      --srr-seed 4321 \
      --disable-lm-eval \
      -ow
  done
done