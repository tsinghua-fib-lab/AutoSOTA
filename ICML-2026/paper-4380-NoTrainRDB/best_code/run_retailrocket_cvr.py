import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier
from rdblearn.datasets import RDBDataset
from rdblearn.estimator import RDBLearnClassifier
from rdblearn.constants import TABPFN_DEFAULT_CONFIG
from loguru import logger
logger.enable("rdblearn")

def main():
    # Unset problematic proxy env vars for HF access
    for var in ["ALL_PROXY", "all_proxy", "HF_ENDPOINT"]:
        os.environ.pop(var, None)

    print("=" * 60)
    print("RDBLearn Retailrocket CVR Experiment")
    print("=" * 60)

    # 1. Load Dataset
    print("\n[1/5] Loading Retailrocket dataset from 4DBInfer...")
    dataset = RDBDataset.from_4dbinfer("retailrocket")
    task = dataset.tasks["cvr"]
    print(f"  Task: {task.name}")
    print(f"  Train: {task.train_df.shape}")
    print(f"  Test:  {task.test_df.shape}")
    print(f"  Target: {task.metadata.target_col}")

    for depth in [2, 3]:
        print(f"\n{'='*60}")
        print(f"  DFS max_depth = {depth}")
        print(f"{'='*60}")

        # 2. Initialize Model with local checkpoint
        print(f"\n[2/5] Initializing TabPFNClassifier (depth={depth})...")
        tabpfn_config = TABPFN_DEFAULT_CONFIG.copy()
        # Use specific checkpoint file matching paper config
        tabpfn_config["model_path"] = "/models/tabpfn-v2-clf/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
        base_model = TabPFNClassifier(**tabpfn_config)
        print(f"  Device: {tabpfn_config['device']}")
        print(f"  Model: {tabpfn_config['model_path']}")

        clf = RDBLearnClassifier(
            base_estimator=base_model,
            config={
                "dfs": {"max_depth": depth},
                "max_train_samples": 10000,
            }
        )

        # 3. Train
        print(f"\n[3/5] Training (depth={depth})...")
        X_train = task.train_df.drop(columns=[task.metadata.target_col])
        y_train = task.train_df[task.metadata.target_col]

        clf.fit(
            X=X_train,
            y=y_train,
            rdb=dataset.rdb,
            key_mappings=task.metadata.key_mappings,
            cutoff_time_column=task.metadata.time_col
        )

        # 4. Predict
        print(f"\n[4/5] Predicting (depth={depth})...")
        X_test = task.test_df.drop(columns=[task.metadata.target_col])
        y_test = task.test_df[task.metadata.target_col]
        y_pred_proba = clf.predict_proba(X=X_test)
        if y_pred_proba.shape[1] == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            y_scores = y_pred_proba[:, 1]

        # 5. Evaluate
        auc = roc_auc_score(y_test, y_scores)
        print(f"\n[5/5] Evaluation (depth={depth}):")
        print(f"  Test AUC: {auc:.6f}")
        print(f"  Paper reported AUC: 0.8469")

if __name__ == "__main__":
    main()
