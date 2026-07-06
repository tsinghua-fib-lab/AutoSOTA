import json, os, sys, time, numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
sys.path.insert(0, '/repo')
os.chdir('/repo')

from watermark.alimark import AliMark
import argparse

# Parse minimal args for AliMark
class Args:
    watermark_algorithm = 'AliMark'
    watermark_model = 'facebook/opt-1.3b'
    watermark_embedder = 'all-mpnet-base-v2'
    watermark_embedding_dim = 768
    watermark_block_size = 8
    watermark_num_next_sentence_candidates = 64
    min_new_sentences = 12
    dataset_name = 'c4'
    vllm_gpu_mem_util = 0.2
    device = 'cuda'
    seed = 42

args = Args()

# Load generation results
gen_file = '_result/generation/block_size_8/c4_AliMark_facebook_opt-1.3b.json'
with open(gen_file) as f:
    gen_data = json.load(f)

# Initialize watermark detector
print('Initializing AliMark detector...')
wm = AliMark(args, load_llm=False)
print('Done.')

# Collect all texts that need detection
texts_to_detect = {}  # (sample_idx, col_name) -> text
for idx_str, row in gen_data.items():
    for col in row:
        if col in ['question', 'reference']:
            continue
        val = row[col]
        if isinstance(val, dict) and 'text' in val and val['text']:
            texts_to_detect[(idx_str, col)] = val['text']

print(f'Total texts to detect: {len(texts_to_detect)}')

# Detection results per sample per column
detection_results = {}

# For each text, run full detection (this is the slow part)
for (idx_str, col), text in list(texts_to_detect.items())[:]:
    result = wm.detect_watermark(text=text)
    detection_results[(idx_str, col)] = result['score']

print(f'Detection complete: {len(detection_results)} results')

# Now compute evaluation metrics for different aggregation methods
# But wait - the aggregation is INSIDE detect_watermark, so scores are already aggregated
# To test different methods, we need different detection runs

# For now, just print the current results
scores_original = []
scores_watermarked = []
for (idx_str, col), score in detection_results.items():
    if col == 'original_result':
        scores_original.append(score)
    elif col == 'watermarked_result':
        scores_watermarked.append(score)

if scores_original and scores_watermarked:
    y_true = [0]*len(scores_original) + [1]*len(scores_watermarked)
    y_scores = scores_original + scores_watermarked
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    print(f'Current (mean) aggregation: AUROC={auc(fpr, tpr):.4f}')
