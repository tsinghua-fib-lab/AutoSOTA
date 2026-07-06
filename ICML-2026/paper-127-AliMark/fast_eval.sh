#!/bin/bash
# Fast eval: detection + evaluation only (skip generation)
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
export TRANSFORMERS_CACHE=/autosota_cache/hf
export HUGGINGFACE_HUB_CACHE=/autosota_cache/hf
export SENTENCE_TRANSFORMERS_HOME=/autosota_cache/hf
rm -rf /root/.cache/flashinfer
mkdir -p /autosota_cache/hf /repo/_result

DATASET="${1:-c4}"
BLOCK_SIZE="${2:-8}"
CANDIDATE_BUDGET="${3:-64}"
N_SENTENCES="${4:-12}"

cd /repo

echo "=== Step 1: Detection ==="
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

echo "=== Step 2: Evaluation ==="
python3 -u 4_evaluation.py \
    --watermark_algorithm AliMark \
    --watermark_model facebook/opt-1.3b \
    --watermark_embedder all-mpnet-base-v2 \
    --watermark_embedding_dim 768 \
    --watermark_block_size ${BLOCK_SIZE} \
    --watermark_num_next_sentence_candidates ${CANDIDATE_BUDGET} \
    --min_new_sentences ${N_SENTENCES} \
    --dataset_name ${DATASET} 2>&1 || true

echo ""
echo "=== Detailed Metrics from detection file ==="
python3 -c '
import numpy as np, pandas as pd
from sklearn.metrics import auc, roc_curve
df = pd.read_json("_result/detection/block_size_'"${BLOCK_SIZE}"'/"${DATASET}"'_AliMark_facebook_opt-1.3b.json", orient="index")
so,sw=[],[]
for _,r in df.iterrows():
    if r.get("original_result") and "detect_result" in r["original_result"]:
        so.append(r["original_result"]["detect_result"]["score"])
    if r.get("watermarked_result") and "detect_result" in r["watermarked_result"]:
        sw.append(r["watermarked_result"]["detect_result"]["score"])
y=[0]*len(so)+[1]*len(sw); ys=so+sw
fpr,tpr,_=roc_curve(y,ys)
roc=auc(fpr,tpr)
def si(x):
    if x < fpr[0]: return tpr[0]
    elif x > fpr[-1]: return tpr[-1]
    return float(np.interp(x,fpr,tpr))
print("n=%d | AUROC=%.2f%% | TPR@0.1%%=%.2f%% | TPR@0.5%%=%.2f%% | TPR@1%%=%.2f%% | TPR@5%%=%.2f%%" % (
    len(so), roc*100, si(0.001)*100, si(0.005)*100, si(0.01)*100, si(0.05)*100))
'
echo "=== Done ==="
