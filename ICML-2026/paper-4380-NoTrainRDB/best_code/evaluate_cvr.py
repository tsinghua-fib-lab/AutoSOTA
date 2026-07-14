#!/usr/bin/env python3
"""RDBLearn Retailrocket CVR evaluation script.
Reproduces Table 1 AUC metric from paper "No Need to Train Your RDB Foundation Model".
Paper AUC: 0.8469 on Retailrocket CVR (4DBInfer benchmark).
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier
from rdblearn.constants import TABPFN_DEFAULT_CONFIG, RDBLEARN_DEFAULT_CONFIG
from loguru import logger
logger.enable("rdblearn")

def evaluate(depth=2, seed=42):
    # Fix proxy issues for HuggingFace model cache access
    for var in ["ALL_PROXY", "all_proxy", "HF_ENDPOINT"]:
        os.environ.pop(var, None)

    np.random.seed(seed)

    print(f"Loading Retailrocket dataset...", flush=True)
    dataset = RDBDataset.from_4dbinfer("retailrocket")
    task = dataset.tasks["cvr"]

    print(f"Task: {task.name}, Train: {task.train_df.shape}, Test: {task.test_df.shape}", flush=True)

    # Use TabPFN-v2 finetuned checkpoint as per paper
    tabpfn_config = TABPFN_DEFAULT_CONFIG.copy()
    tabpfn_config["model_path"] = "/models/tabpfn-v2-clf/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
    tabpfn_config["n_estimators"] = 32
    
    base_model = TabPFNClassifier(**tabpfn_config)

    config = RDBLEARN_DEFAULT_CONFIG.copy()
    config["dfs"]["max_depth"] = depth
    config["dfs"]["agg_primitives"] = ["max", "min", "mean", "count", "mode", "std", "sum"]
    config["max_train_samples"] = 10000
    config["balanced_sampling"] = True

    clf = RDBLearnClassifier(base_estimator=base_model, config=config)

    print(f"Training (depth={depth}, 10k samples, balanced, n_est=32, sum_agg)...", flush=True)
    X_train = task.train_df.drop(columns=[task.metadata.target_col])
    y_train = task.train_df[task.metadata.target_col]

    clf.fit(
        X=X_train, y=y_train,
        rdb=dataset.rdb,
        key_mappings=task.metadata.key_mappings,
        cutoff_time_column=task.metadata.time_col
    )

    print(f"Predicting...", flush=True)
    X_test = task.test_df.drop(columns=[task.metadata.target_col])
    y_test = task.test_df[task.metadata.target_col]
    y_pred_proba = clf.predict_proba(X=X_test)

    if y_pred_proba.shape[1] == 2:
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba[:, 1]

    auc = roc_auc_score(y_test, y_scores)
    print(f"Test AUC: {auc:.6f}", flush=True)
    return auc

if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    auc = evaluate(depth=depth, seed=seed)
    print(json.dumps({"auc": float(auc), "depth": depth, "seed": seed, "balanced": True, "n_estimators": 32}))
