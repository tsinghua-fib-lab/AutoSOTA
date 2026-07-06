#!/bin/bash
set -euo pipefail
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/autosota_cache/hf
export TRANSFORMERS_CACHE=/autosota_cache/hf
export HUGGINGFACE_HUB_CACHE=/autosota_cache/hf
export SENTENCE_TRANSFORMERS_HOME=/autosota_cache/hf
mkdir -p /autosota_cache/hf /repo/_result

cd /repo
# Quick test: generate just the first sample
python3 -u << "PYEOF"
import os, sys, json
for k in ["http_proxy","HTTP_PROXY","https_proxy","HTTPS_PROXY","all_proxy","ALL_PROXY","no_proxy","NO_PROXY"]:
    os.environ.pop(k, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TRANSFORMERS_CACHE"] = "/autosota_cache/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/autosota_cache/hf"

import nltk
nltk.download("punkt", quiet=True)

class Args:
    watermark_algorithm = "AliMark"
    watermark_model = "facebook/opt-1.3b"
    watermark_embedder = "all-mpnet-base-v2"
    watermark_embedding_dim = 768
    watermark_block_size = 8
    watermark_num_next_sentence_candidates = 8
    min_new_sentences = 4
    dataset_name = "c4"
    vllm_gpu_mem_util = 0.8
    device = "cuda"
    seed = 42
    watermark_rs_dropout = 0.0

sys.path.insert(0, "/repo")
from watermark.alimark import AliMark

print("Loading AliMark with HF transformers...")
watermark = AliMark(Args(), load_llm=True)
print("AliMark loaded!")

# Load first sample
with open("dataset/c4.json", "r") as f:
    line = json.loads(f.readline())

prompt = line["prompt"]
reference = line["natural_text"]
print(f"Prompt: {prompt}")
print(f"Reference: {reference[:100]}...")

print("\nGenerating unwatermarked text...")
unwatermarked = watermark.generate_unwatermarked_text(prompt)
print(f"Unwatermarked ({len(nltk.sent_tokenize(unwatermarked))} sents): {unwatermarked[:200]}...")

print("\nGenerating watermarked text...")
watermarked = watermark.generate_watermarked_text(prompt)
print(f"Watermarked ({len(nltk.sent_tokenize(watermarked))} sents): {watermarked[:200]}...")

print("\nRunning detection...")
detect_unwatermarked = watermark.detect_watermark(unwatermarked)
detect_watermarked = watermark.detect_watermark(watermarked)
detect_original = watermark.detect_watermark(reference)

print(f"Detection scores:")
print(f"  Original (human): {detect_original['score']:.4f}")
print(f"  Unwatermarked:    {detect_unwatermarked['score']:.4f}")
print(f"  Watermarked:      {detect_watermarked['score']:.4f}")

print("\nQUICK TEST PASSED!")
PYEOF
