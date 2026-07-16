#!/bin/bash
set -e
source /autosota_cache/paper-4041-venv/bin/activate
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export CUDA_VISIBLE_DEVICES=0,1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf
export TMPDIR=/autosota_cache/tmp

MODEL_PATH="${1:-/autosota_cache/checkpoints/numina-cot-ranktuner-Qwen2.5-Math-7B/global_step_39}"
OUTPUT_DIR="${2:-/autosota_cache/eval_results/numina-cot-ranktuner-Qwen2.5-Math-7B}"

echo "Evaluating: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd /repo/math_evaluation
TOKENIZERS_PARALLELISM=false python3 -u math_eval.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_name math_oai \
    --output_dir "$OUTPUT_DIR" \
    --split test \
    --prompt_type qwen25-math-cot \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 1.0 \
    --n_sampling 16 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm

echo "Evaluation complete."
echo "Results:"
python3 -c "
import json
with open(/math_oai_metrics.json) as f:
    data = json.load(f)
print(f\"Pass@1: {data[\"pass_at_k\"][\"pass@1\"]}\")
print(f\"Pass@16: {data[\"pass_at_k\"][\"pass@16\"]}\")
"
