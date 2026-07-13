#!/usr/bin/env python3
"""
Train per-category KNN quality and token count predictors for category-aware R2-Router.

Uses KNeighborsRegressor(cosine, distance-weighted) per (category x model x budget).
K-fold CV to evaluate, then trains final model on full training set.

Requires: training_data.pkl from build_category_training_data.py

Usage:
    .venv/bin/python scripts/train_category_predictors.py [--k 5]
    .venv/bin/python scripts/train_category_predictors.py --full
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from collections import defaultdict

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
from joblib import dump

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    TRAINING_DATA_PATH, CHECKPOINT_DIR,
    CATEGORY_NAMES, NUM_CATEGORIES,
    MODELS, get_budgets, ROUTER_DATA_10_PATH,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="K for K-fold CV")
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA components per category (0=disabled)")
    parser.add_argument("--full", action="store_true",
                        help="Train final model on ALL data (no held-out test set)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for held-out test (default: 0.2)")
    parser.add_argument("--sub10", action="store_true",
                        help="Train only on sub_10 queries (RouterArena 10%% split)")
    args = parser.parse_args()

    log(f"=== Category-Aware KNN Predictor Training (K={args.k}-fold CV) ===")
    if args.pca > 0:
        log(f"PCA: {args.pca} components per category")
    if args.full:
        log(f"FULL MODE: training final model on ALL data (no held-out test)")
    if args.sub10:
        log(f"SUB10 MODE: training only on RouterArena sub_10 queries")
    log("")

    # 1. Load training data
    log("Loading training data...")
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]      # (8400, 1024)
    categories = data["categories"]       # (8400,) int
    models_data = data["models"]          # {model: {budget: {accuracy, output_tokens}}}
    n_queries = embeddings.shape[0]
    log(f"  Queries: {n_queries}, Dim: {embeddings.shape[1]}")
    log(f"  Models: {list(models_data.keys())}")

    # 2. Global train/test split
    all_idx = np.arange(n_queries)
    global_indices = data["global_indices"]  # [str, ...]
    if args.sub10:
        # Use RouterArena sub_10 as training set, rest as test
        with open(ROUTER_DATA_10_PATH) as f:
            sub10_data = json.load(f)
        sub10_gis = set(e["global index"] for e in sub10_data)
        gi_to_idx = {gi: i for i, gi in enumerate(global_indices)}
        train_idx = np.array(sorted([gi_to_idx[gi] for gi in sub10_gis if gi in gi_to_idx]))
        test_idx = np.array(sorted(set(all_idx.tolist()) - set(train_idx.tolist())))
        train_set = set(train_idx.tolist())
        log(f"  SUB10 MODE: {len(train_idx)} train, {len(test_idx)} test")
    elif args.full:
        train_idx = all_idx.copy()
        test_idx = np.array([], dtype=int)
        train_set = set(train_idx.tolist())
        log(f"  FULL MODE: {len(train_idx)} train, 0 test (all data)")
    else:
        train_idx, test_idx = train_test_split(
            all_idx, test_size=args.test_size, random_state=42, stratify=categories,
        )
        train_set = set(train_idx.tolist())
        log(f"  Split: {len(train_idx)} train, {len(test_idx)} test (held-out, test_size={args.test_size})")

    # Save split
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    split_path = os.path.join(CHECKPOINT_DIR, "train_test_split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump({"train_idx": train_idx, "test_idx": test_idx}, f)

    # 3. K-fold CV + final training per (category x model)
    K = args.k
    results_summary = {}

    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        log(f"\n{'='*70}")
        log(f"Category {cat_idx}: {cat_name}")
        log(f"{'='*70}")

        cat_all = np.where(categories == cat_idx)[0]
        cat_train = np.array([i for i in cat_all if i in train_set])
        cat_test = np.array([i for i in cat_all if i not in train_set])
        log(f"  Train: {len(cat_train)}, Test: {len(cat_test)}")

        if len(cat_train) < 20:
            log(f"  SKIP: too few samples")
            continue

        X_cat_train = embeddings[cat_train]
        X_cat_test = embeddings[cat_test] if len(cat_test) > 0 else np.zeros((0, embeddings.shape[1]))

        # Optional PCA dimensionality reduction
        if args.pca > 0:
            n_components = min(args.pca, X_cat_train.shape[0] - 1, X_cat_train.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_cat_train = pca.fit_transform(X_cat_train)
            if len(X_cat_test) > 0:
                X_cat_test = pca.transform(X_cat_test)
            log(f"  PCA: {embeddings.shape[1]} -> {n_components} (explained var: {pca.explained_variance_ratio_.sum():.3f})")

        cat_dir = os.path.join(CHECKPOINT_DIR, "predictors", cat_name)
        os.makedirs(cat_dir, exist_ok=True)

        if args.pca > 0:
            dump(pca, os.path.join(cat_dir, "pca.joblib"))

        results_summary[cat_name] = {
            "n_train": len(cat_train), "n_test": len(cat_test), "models": {}
        }

        kf = KFold(n_splits=min(K, len(cat_train)), shuffle=True, random_state=42)

        for model_name in MODELS:
            if model_name not in models_data:
                continue

            model_budgets_data = models_data[model_name]
            budgets = get_budgets(model_name)

            # Collect valid budgets (have accuracy data)
            valid_budgets = []
            for b in budgets:
                if b in model_budgets_data:
                    y = model_budgets_data[b]["accuracy"][cat_train]
                    if y.sum() > 0:
                        valid_budgets.append(b)

            if not valid_budgets:
                continue

            num_heads = len(valid_budgets)

            # Build quality target matrices
            Y_quality_train = np.column_stack([
                model_budgets_data[b]["accuracy"][cat_train] for b in valid_budgets
            ])
            if len(cat_test) > 0:
                Y_quality_test = np.column_stack([
                    model_budgets_data[b]["accuracy"][cat_test] for b in valid_budgets
                ])
            else:
                Y_quality_test = np.zeros((0, num_heads))

            # Token targets (from concise setting)
            token_budget = "concise"
            has_token = (token_budget in model_budgets_data and
                         model_budgets_data[token_budget]["output_tokens"][cat_train].sum() > 0)
            if has_token:
                y_token_train = model_budgets_data[token_budget]["output_tokens"][cat_train]
                y_token_test = model_budgets_data[token_budget]["output_tokens"][cat_test] if len(cat_test) > 0 else np.zeros(0, dtype=np.float32)
            else:
                y_token_train = np.zeros(len(cat_train), dtype=np.float32)
                y_token_test = np.zeros(len(cat_test), dtype=np.float32)

            # KNN K value: sqrt(n_train_fold), capped at 256, minimum 3
            knn_k = min(256, max(3, int(np.sqrt(len(cat_train) * (K - 1) / K))))

            # --- K-fold CV ---
            knn_cv_r2s = []
            knn_token_cv_r2s = []

            for fold_idx, (fold_train, fold_val) in enumerate(kf.split(X_cat_train)):
                Xf_train = X_cat_train[fold_train]
                Xf_val = X_cat_train[fold_val]
                Yqf_train = Y_quality_train[fold_train]
                Yqf_val = Y_quality_train[fold_val]
                ytf_train = y_token_train[fold_train]
                ytf_val = y_token_train[fold_val]

                fold_knn_k = min(knn_k, len(fold_train) - 1)

                fold_knn_r2s = []
                for h in range(num_heads):
                    knn = KNeighborsRegressor(
                        n_neighbors=fold_knn_k, metric="cosine", weights="distance"
                    )
                    knn.fit(Xf_train, Yqf_train[:, h])
                    y_pred = knn.predict(Xf_val)
                    r2 = r2_score(Yqf_val[:, h], y_pred) if Yqf_val[:, h].var() > 1e-10 else 0.0
                    fold_knn_r2s.append(r2)
                knn_cv_r2s.append(np.mean(fold_knn_r2s))

                if has_token and ytf_train.sum() > 0:
                    tknn = KNeighborsRegressor(
                        n_neighbors=fold_knn_k, metric="cosine", weights="distance"
                    )
                    tknn.fit(Xf_train, ytf_train)
                    tp = tknn.predict(Xf_val)
                    tr2 = r2_score(ytf_val, tp) if ytf_val.var() > 1e-10 else 0.0
                    knn_token_cv_r2s.append(tr2)

            knn_mean_r2 = np.mean(knn_cv_r2s) if knn_cv_r2s else 0.0
            log(f"  {model_name:<15} heads={num_heads:>2}  k={knn_k:>3}  KNN R2={knn_mean_r2:.4f}")

            # --- Train final KNN on full category training set ---
            final_knn_k = min(knn_k, len(cat_train) - 1)

            for h, b in enumerate(valid_budgets):
                knn = KNeighborsRegressor(
                    n_neighbors=final_knn_k, metric="cosine", weights="distance"
                )
                knn.fit(X_cat_train, Y_quality_train[:, h])
                dump(knn, os.path.join(cat_dir, f"{model_name}_{b}_quality.joblib"))

            if has_token:
                token_knn = KNeighborsRegressor(
                    n_neighbors=final_knn_k, metric="cosine", weights="distance"
                )
                token_knn.fit(X_cat_train, y_token_train)
                dump(token_knn, os.path.join(cat_dir, f"{model_name}_token.joblib"))

            meta = {"architecture": "KNN", "budgets": valid_budgets,
                    "knn_k": final_knn_k, "cv_mean_r2": float(knn_mean_r2)}
            with open(os.path.join(cat_dir, f"{model_name}_quality_meta.json"), "w") as f:
                json.dump(meta, f)

            # --- Evaluate on held-out test set ---
            if len(cat_test) == 0:
                test_mean_r2 = 0.0
                test_token_r2 = 0.0
            else:
                test_r2s = []
                for h, b in enumerate(valid_budgets):
                    knn = KNeighborsRegressor(
                        n_neighbors=final_knn_k, metric="cosine", weights="distance"
                    )
                    knn.fit(X_cat_train, Y_quality_train[:, h])
                    y_pred = knn.predict(X_cat_test)
                    r2 = r2_score(Y_quality_test[:, h], y_pred) if Y_quality_test[:, h].var() > 1e-10 else 0.0
                    test_r2s.append(r2)
                test_mean_r2 = np.mean(test_r2s)

                test_token_r2 = 0.0
                if has_token:
                    tknn = KNeighborsRegressor(
                        n_neighbors=final_knn_k, metric="cosine", weights="distance"
                    )
                    tknn.fit(X_cat_train, y_token_train)
                    tp = tknn.predict(X_cat_test)
                    test_token_r2 = r2_score(y_token_test, tp) if y_token_test.var() > 1e-10 else 0.0

            log(f"    Test R2: quality={test_mean_r2:.4f}, token={test_token_r2:.4f}")

            results_summary[cat_name]["models"][model_name] = {
                "best_architecture": "KNN",
                "n_budgets": num_heads,
                "knn_k": knn_k,
                "KNN_cv_r2": float(knn_mean_r2),
                "test_quality_r2": float(test_mean_r2),
                "test_token_r2": float(test_token_r2),
                "knn_token_cv_r2": float(np.mean(knn_token_cv_r2s)) if knn_token_cv_r2s else 0.0,
            }

    # 3b. Compute and save category means for shrinkage
    log("\nComputing category means for shrinkage...")
    category_means = {}
    category_token_means = {}
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        cat_train = np.array([i for i in np.where(categories == cat_idx)[0] if i in train_set])
        if len(cat_train) == 0:
            continue
        category_means[cat_name] = {}
        category_token_means[cat_name] = {}
        for model_name in MODELS:
            if model_name not in models_data:
                continue
            model_budgets_data = models_data[model_name]
            budgets = get_budgets(model_name)
            category_means[cat_name][model_name] = {}
            for b in budgets:
                if b in model_budgets_data:
                    y = model_budgets_data[b]["accuracy"][cat_train]
                    category_means[cat_name][model_name][b] = float(y.mean())
            for b in ["concise"] + budgets:
                if b in model_budgets_data:
                    t = model_budgets_data[b]["output_tokens"][cat_train]
                    category_token_means[cat_name][model_name] = float(t.mean())
                    break

    means_path = os.path.join(CHECKPOINT_DIR, "category_means.pkl")
    with open(means_path, "wb") as f:
        pickle.dump({"quality": category_means, "tokens": category_token_means}, f)
    log(f"Saved category means to {means_path}")

    # 4. Summary
    log(f"\n{'='*70}")
    log(f"SUMMARY")
    log(f"{'='*70}")

    log(f"\nPer category:")
    for cat_name in CATEGORY_NAMES:
        if cat_name not in results_summary:
            continue
        cat_r2s = []
        for model_name, minfo in results_summary[cat_name]["models"].items():
            cat_r2s.append(minfo["test_quality_r2"])
        mean_test_r2 = np.mean(cat_r2s) if cat_r2s else 0.0
        log(f"  {cat_name:<12} KNN  mean_test_R2={mean_test_r2:.4f}")

    summary_path = os.path.join(CHECKPOINT_DIR, "predictor_results.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    log(f"\nSaved results to {summary_path}")

    log("\nDone!")


if __name__ == "__main__":
    main()
