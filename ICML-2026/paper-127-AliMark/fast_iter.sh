#!/bin/bash
# Fast iteration: detection + evaluation only (attacks pre-computed)
set -euo pipefail
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/opt/conda/lib/python3.10/site-packages/nvidia/cu13
export PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cu13/bin:$PATH
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
export LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/nccl/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cu13/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf
export HF_HUB_CACHE=/autosota_cache/hf/hub
export TRANSFORMERS_CACHE=/autosota_cache/hf
export HUGGINGFACE_HUB_CACHE=/autosota_cache/hf
export SENTENCE_TRANSFORMERS_HOME=/autosota_cache/hf
rm -rf /root/.cache/flashinfer
mkdir -p /autosota_cache/hf/hub /repo/_result

DATASET="${1:-c4}"
BLOCK_SIZE="${2:-8}"
CANDIDATE_BUDGET="${3:-64}"
N_SENTENCES="${4:-12}"

cd /repo

echo "=== Fast Iteration $(date) ==="
echo "=== Detection ==="
python3 -u 3_detection.py \
    --watermark_algorithm AliMark \
    --watermark_model facebook/opt-1.3b \
    --watermark_embedder all-mpnet-base-v2 \
    --watermark_embedding_dim 768 \
    --watermark_block_size ${BLOCK_SIZE} \
    --watermark_num_next_sentence_candidates ${CANDIDATE_BUDGET} \
    --min_new_sentences ${N_SENTENCES} \
    --dataset_name ${DATASET} \
    --device cuda \
    --seed 42

echo "=== Evaluation ==="
python3 -u 4_evaluation.py \
    --watermark_algorithm AliMark \
    --watermark_model facebook/opt-1.3b \
    --watermark_embedder all-mpnet-base-v2 \
    --watermark_embedding_dim 768 \
    --watermark_block_size ${BLOCK_SIZE} \
    --watermark_num_next_sentence_candidates ${CANDIDATE_BUDGET} \
    --min_new_sentences ${N_SENTENCES} \
    --dataset_name ${DATASET}

echo "=== Metrics ==="
python3 -u eval_metrics.py ${BLOCK_SIZE}
echo "=== Done $(date) ==="
