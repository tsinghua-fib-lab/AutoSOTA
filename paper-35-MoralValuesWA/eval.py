"""
Eval script for paper-482: Comparing Moral Values in Western English-speaking societies and LLMs
with Word Associations.

Uses pre-computed moral scores (human_moral.json, llama_2.1_moral.json) with eMFD scoring
to compute Spearman correlations for 5 moral dimensions (GMN-L, Table 1 results).

Reproduced baseline values (ground truth):
  care:      0.4570
  sanctity:  0.4381
  fairness:  0.3223
  authority: 0.2507
  loyalty:   0.2994
"""

import json
import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ---- Load pre-computed moral scores ----
with open('./data/human_moral.json', 'r') as f:
    human_moral = json.load(f)

with open('./data/llama_2.1_moral.json', 'r') as f:
    llm_moral = json.load(f)

with open('./data/mag_words.json', 'r') as f:
    mag_words = json.load(f)

# ---- Load eMFD scoring CSV ----
emfd_path = './data/emfd_scoring.csv'
if not os.path.exists(emfd_path):
    # Try to download from GitHub
    import urllib.request
    url = 'https://raw.githubusercontent.com/medianeuroscience/emfd/master/dictionaries/emfd_scoring.csv'
    try:
        urllib.request.urlretrieve(url, emfd_path)
    except Exception as e:
        sys.exit(f"ERROR: emfd_scoring.csv not found at {emfd_path} and download failed: {e}")

df = pd.read_csv(emfd_path)

# eMFD scoring: product of probability and sentiment
df['care'] = df['care_p'] * df['care_sent']
df['fairness'] = df['fairness_p'] * df['fairness_sent']
df['loyalty'] = df['loyalty_p'] * df['loyalty_sent']
df['authority'] = df['authority_p'] * df['authority_sent']
df['sanctity'] = df['sanctity_p'] * df['sanctity_sent']

emfd_by_word = df.set_index('word')

dimensions = ['care', 'sanctity', 'fairness', 'authority', 'loyalty']

results = {}

for i, dim in enumerate(dimensions):
    words = mag_words[dim]
    # Apply word substitutions from original mag_comparison function
    words = [w.replace("overall", "overalls").replace("judgment", "judgement") for w in words]

    emfd_scores = []
    llm_scores = []

    for word in words:
        emfd_word = word
        if word == "overalls":
            emfd_word = "overall"
        elif word == "judgement":
            emfd_word = "judgment"

        # Get LLM score for this dimension
        if word not in llm_moral:
            continue
        llm_score_vec = llm_moral[word]
        # llm_moral stores per-word moral scores as list [care, fairness, loyalty, authority, sanctity]
        dim_idx = ['care', 'fairness', 'loyalty', 'authority', 'sanctity'].index(dim)
        llm_score = llm_score_vec[dim_idx]

        # Get eMFD score
        if emfd_word not in emfd_by_word.index:
            continue
        emfd_score = emfd_by_word.loc[emfd_word, dim]

        llm_scores.append(llm_score)
        emfd_scores.append(emfd_score)

    if len(llm_scores) < 2:
        print(f"WARNING: insufficient data for dimension {dim}")
        continue

    corr, pval = spearmanr(llm_scores, emfd_scores)
    results[dim] = corr
    n = len(llm_scores)
    print(f"llm correlation: {dim}, {corr:.4f} (n={n}, p={pval:.4f})")

print("\n=== RESULTS ===")
for dim in dimensions:
    if dim in results:
        print(f"correlation_{dim}: {results[dim]:.4f}")
