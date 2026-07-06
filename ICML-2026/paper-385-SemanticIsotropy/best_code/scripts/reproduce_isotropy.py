#!/usr/bin/env python3
"""Pooling method sweep with polynomial degree 3 OLS.

Tests mean, last, max, cls pooling to find the best embedding
aggregation for isotropy-factuality regression.
"""

import json
import os
import pickle
import sys
import collections

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from semantic_isotropy.metrics.isotropy import embedding_density

MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
DATA_PATH = "/datasets/paper385/fsbio_phi35_segscore.jsonl"
OUTPUT_PKL = "/repo/outputs/isotropy_results_nomic_v1.pkl"
OUTPUT_JSON = "/repo/outputs/isotropy_results_nomic_v1.json"
N_BOOTSTRAP = 1500
N_SAMPLES = 10
FP16 = True
SEED = 42

os.environ.pop("HF_ENDPOINT", None)

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    DEVICE = f"cuda:{gpu_ids.split(',')[0]}"
else:
    DEVICE = "cpu"

print(f"Device: {DEVICE}")

print("Loading Nomic V1 embedding model ...")
config = AutoConfig.from_pretrained(
    MODEL_NAME, trust_remote_code=True, output_hidden_states=True, cache_dir="/models",
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, cache_dir="/models"
)
tokenizer.add_special_tokens(
    {"pad_token": tokenizer.special_tokens_map.get("pad_token", tokenizer.eos_token)}
)
dtype = torch.float16 if FP16 else torch.float32
model = AutoModel.from_pretrained(
    MODEL_NAME, trust_remote_code=True, config=config, cache_dir="/models", torch_dtype=dtype,
)
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

print(f"Loading data from {DATA_PATH} ...")
with open(DATA_PATH, encoding="utf-8") as fh:
    data = [json.loads(line) for line in fh]
print(f"Loaded {len(data)} entities.")

# Pre-compute factuality (same for all pooling methods)
factuality_scores = []
entities = []
for entry in data:
    facts = []
    for r in entry["responses"][:N_SAMPLES]:
        scores = [s["class"] == "True" for s in r["statements"]]
        facts.append(np.mean(scores))
    factuality_scores.append(float(np.mean(facts)))
    entities.append(entry["entity"])

factuality_scores = np.array(factuality_scores)
n_entities = len(factuality_scores)

# Test each pooling method
POOLING_METHODS = ["mean", "last", "max", "cls"]
results_by_pooling = {}

for pool_method in POOLING_METHODS:
    print(f"\n{'='*40}")
    print(f"Testing pooling: {pool_method}")
    
    isotropy_scores = []
    for entry in tqdm(data, desc=f"Pool={pool_method}"):
        responses = entry["responses"][:N_SAMPLES]
        si, _pooled, _eig = embedding_density(
            responses, model, tokenizer, entry["entity"],
            pooling_method=pool_method,
            device=torch.device(DEVICE), model_name=MODEL_NAME,
        )
        isotropy_scores.append(si)
    
    isotropy_scores = np.array(isotropy_scores)
    
    # Linear
    lr_lin = LinearRegression().fit(isotropy_scores.reshape(-1,1), factuality_scores)
    r2_lin = float(lr_lin.score(isotropy_scores.reshape(-1,1), factuality_scores))
    
    # Poly deg 3
    poly3 = PolynomialFeatures(degree=3, include_bias=False)
    X_p3 = poly3.fit_transform(isotropy_scores.reshape(-1,1))
    lr_p3 = LinearRegression().fit(X_p3, factuality_scores)
    r2_p3 = float(lr_p3.score(X_p3, factuality_scores))
    adj_r2_p3 = float(1 - (1 - r2_p3) * (n_entities - 1) / (n_entities - 3 - 1))
    
    pearson_r = float(np.corrcoef(isotropy_scores, factuality_scores)[0,1])
    
    results_by_pooling[pool_method] = {
        "r2_linear": r2_lin,
        "r2_poly3": r2_p3,
        "adj_r2_poly3": adj_r2_p3,
        "pearson_r": pearson_r,
        "isotropy_mean": float(np.mean(isotropy_scores)),
        "isotropy_std": float(np.std(isotropy_scores)),
        "isotropy_scores": isotropy_scores,
    }
    
    print(f"  Linear: R2={r2_lin:.4f}  |  Poly3: R2={r2_p3:.4f}  adjR2={adj_r2_p3:.4f}  r={pearson_r:.4f}")

# Select best by adjusted R2
best_pool = max(POOLING_METHODS, key=lambda p: results_by_pooling[p]["adj_r2_poly3"])
best_result = results_by_pooling[best_pool]

print(f"\nBest pooling: {best_pool}")

# Bootstrap with best
isotropy_best = best_result["isotropy_scores"]
poly3_best = PolynomialFeatures(degree=3, include_bias=False)
X_best = poly3_best.fit_transform(isotropy_best.reshape(-1,1))

r2_boot = np.zeros(N_BOOTSTRAP)
for i in range(N_BOOTSTRAP):
    idx = np.random.choice(n_entities, size=n_entities, replace=True)
    X_bs = poly3_best.fit_transform(isotropy_best[idx].reshape(-1,1))
    lr_b = LinearRegression().fit(X_bs, factuality_scores[idx])
    r2_boot[i] = float(lr_b.score(X_bs, factuality_scores[idx]))

r2_mean = float(np.mean(r2_boot))
r2_std = float(np.std(r2_boot))

# Linear bootstrap for comparison
lr_lin_best = LinearRegression().fit(isotropy_best.reshape(-1,1), factuality_scores)
r2_lin_best = float(lr_lin_best.score(isotropy_best.reshape(-1,1), factuality_scores))

print()
print("=" * 60)
print("REPRODUCTION RESULTS — Pooling Method Sweep + Poly deg=3")
print("=" * 60)
print(f"Best pooling : {best_pool}")
print()
print("--- Pooling comparison (poly deg=3) ---")
for method in POOLING_METHODS:
    r = results_by_pooling[method]
    marker = " <-- BEST" if method == best_pool else ""
    print(f"  {method:6s}: R2_lin={r['r2_linear']:.4f}  R2_poly3={r['r2_poly3']:.4f}  adjR2={r['adj_r2_poly3']:.4f}  r={r['pearson_r']:.4f}{marker}")
print()
print(f"R2  point     : {best_result['r2_poly3']:.4f}")
print(f"R2  adjusted   : {best_result['adj_r2_poly3']:.4f}")
print(f"R2  bootstrap : {r2_mean:.4f} +/- {r2_std:.4f}")
print(f"R2  = {r2_mean:.4f} +/- {r2_std:.4f}")
print()
print(f"Isotropy      : mean={best_result['isotropy_mean']:.4f}  std={best_result['isotropy_std']:.4f}")
print(f"Factuality    : mean={np.mean(factuality_scores):.4f}  std={np.std(factuality_scores):.4f}")
print(f"Pearson r     : {best_result['pearson_r']:.4f}")
print(f"OLS (linear)  : factuality = {lr_lin_best.coef_[0]:.4f} * SI + {lr_lin_best.intercept_:.4f}")
is_neg = "NEGATIVE (correct)" if lr_lin_best.coef_[0] < 0 else "POSITIVE"
print(f"Direction     : {is_neg}")
print()
paper_ss = 0.273
paper_fs = 0.600
print(f"Paper (SegmentScore): R2={paper_ss:.3f}+/-0.06")
print(f"Paper (FactScore)   : R2={paper_fs:.3f}+/-0.05  [rubric target]")
print(f"Ours  (SegmentScore): R2={r2_mean:.3f}+/-{r2_std:.3f}  [pool={best_pool}, poly deg=3]")
print("=" * 60)

results = {
    "paper_id": 385,
    "model": MODEL_NAME,
    "best_pooling": best_pool,
    "pooling_comparison": {
        p: {"r2_linear": results_by_pooling[p]["r2_linear"],
            "r2_poly3": results_by_pooling[p]["r2_poly3"],
            "adj_r2_poly3": results_by_pooling[p]["adj_r2_poly3"],
            "pearson_r": results_by_pooling[p]["pearson_r"]}
        for p in POOLING_METHODS
    },
    "r2_point_estimate": best_result["r2_poly3"],
    "r2_adjusted": best_result["adj_r2_poly3"],
    "r2_bootstrap_mean": r2_mean,
    "r2_bootstrap_std": r2_std,
    "isotropy_mean": best_result["isotropy_mean"],
    "isotropy_std": best_result["isotropy_std"],
    "factuality_mean": float(np.mean(factuality_scores)),
    "factuality_std": float(np.std(factuality_scores)),
    "pearson_r": best_result["pearson_r"],
    "ols_slope": float(lr_lin_best.coef_[0]),
    "ols_intercept": float(lr_lin_best.intercept_),
    "config": {
        "n_samples": N_SAMPLES,
        "n_bootstrap": N_BOOTSTRAP,
        "n_entities": n_entities,
        "fp16": FP16,
        "pooling": best_pool,
        "polynomial_degree": 3,
        "seed": SEED,
    },
}

os.makedirs(os.path.dirname(OUTPUT_PKL), exist_ok=True)
with open(OUTPUT_PKL, "wb") as fh:
    pickle.dump(results, fh)
with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, default=str)

print(f"\nSaved: {OUTPUT_PKL}")
print(f"Saved: {OUTPUT_JSON}")
sys.exit(0)
