#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_DIR="$ROOT/GLUE_PIPELINE"
PYTHON_BIN=${PYTHON_BIN:-python}
TOKENIZER_PATH=${TOKENIZER_PATH:-${GLUE_TOKENIZER_PATH:-"$ROOT/tok/fineweb_bpe_16000.json"}}
DATA_CACHE=${DATA_CACHE:-${GLUE_DATA_CACHE:-"$ROOT/data"}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${GLUE_OUTPUT_ROOT:-"$ROOT/outputs"}}

PROTO_GPU=${PROTO_GPU:-0}
LLAMA_GPU=${LLAMA_GPU:-1}
MAMBA_GPU=${MAMBA_GPU:-2}
DELTA_GPU=${DELTA_GPU:-3}
MODELS=${MODELS:-"protoattn llama mamba deltanet"}

mkdir -p "$OUTPUT_ROOT"

gpu_for_model() {
  case "$1" in
    protoattn) echo "$PROTO_GPU" ;;
    llama) echo "$LLAMA_GPU" ;;
    mamba) echo "$MAMBA_GPU" ;;
    deltanet) echo "$DELTA_GPU" ;;
    *) echo "0" ;;
  esac
}

launch() {
  local model=$1
  local gpu
  gpu=$(gpu_for_model "$model")
  echo "[GLUE] Launching $model on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" "$PIPELINE_DIR/GLUE_TRAINER.py" \
    --model "$model" \
    --device cuda:0 \
    --output_root "$OUTPUT_ROOT" \
    --data_cache "$DATA_CACHE" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --batch_size 16 \
    --epochs 3 \
    --max_length 512 \
    --log_steps 100 \
    > "$OUTPUT_ROOT/${model}_train.log" 2>&1 &
}

for model in $MODELS; do
  launch "$model"
done

wait
echo "All GLUE fine-tuning jobs finished."
