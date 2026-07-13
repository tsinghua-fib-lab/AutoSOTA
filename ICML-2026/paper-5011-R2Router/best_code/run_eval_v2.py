#!/usr/bin/env python3
"""
R2-Router evaluation v2: Use KNN-based predictors (matching repo's approach)
and evaluate routing performance.

The repo uses KNeighborsRegressor with cosine distance and distance-weighted
averaging for category-aware routing. This script adapts that approach to
a global (non-category-aware) setting.

Usage:
    python run_eval_v2.py --training-data /datasets/training_data.pkl \
        --embeddings /datasets/routerarena_embeddings.pkl \
        --router-data-10 /datasets/router_data_10.json \
        --output-dir ./results_v3
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description='R2-Router KNN evaluation')
    parser.add_argument('--training-data', type=str, required=True)
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--router-data-10', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./results_v3')
    parser.add_argument('--lambda-points', type=int, default=200)
    parser.add_argument('--k-neighbors', type=int, default=64)
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--qnc-target-rate', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"R2-Router KNN Evaluation (k={args.k_neighbors})")
    print("=" * 60)

    # Load data
    with open(args.training_data, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']  # (8400, 1024)
    models_data = data['models']
    global_indices = data['global_indices']

    n_queries = embeddings.shape[0]
    print(f"Loaded {n_queries} queries, {embeddings.shape[1]}d embeddings")
    print(f"Models: {sorted(models_data.keys())}")

    # Train/test split
    rng = np.random.RandomState(args.seed)
    all_idx = np.arange(n_queries)
    n_train = int(n_queries * args.train_frac)
    train_idx = rng.choice(all_idx, n_train, replace=False)
    test_idx = np.array(sorted(set(all_idx) - set(train_idx.tolist())))
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train KNN predictors for each (model, budget)
    print(f"\nTraining KNN predictors (k={args.k_neighbors}, cosine, distance-weighted)...")
    preds = {}

    for mn in sorted(models_data.keys()):
        budgets = sorted(models_data[mn].keys())
        if not budgets:
            continue

        preds[mn] = {}
        for budget in budgets:
            bdata = models_data[mn][budget]
            y_all = bdata['accuracy']
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            valid_train = ~np.isnan(y_train)
            if valid_train.sum() < args.k_neighbors:
                continue

            knn = KNeighborsRegressor(
                n_neighbors=min(args.k_neighbors, valid_train.sum()),
                metric='cosine',
                weights='distance',
                n_jobs=-1,
            )
            knn.fit(X_train[valid_train], y_train[valid_train])

            y_pred = np.full(len(test_idx), np.nan)
            valid_test = ~np.isnan(y_test)
            if valid_test.sum() > 0:
                y_pred[valid_test] = knn.predict(X_test[valid_test])

            test_score = knn.score(X_test[valid_test], y_test[valid_test])

            preds[mn][budget] = {
                'pred_test': y_pred,
                'true_test': y_test,
                'test_score': test_score,
            }

        if preds[mn]:
            avg_r2 = np.mean([preds[mn][b]['test_score'] for b in preds[mn]])
            print(f"  {mn}: {len(preds[mn])}/{len(budgets)} budgets, avg R²={avg_r2:.4f}")

    # Evaluate routing
    print(f"\nEvaluating routing...")
    n_test = len(test_idx)
    lambdas = np.linspace(0, 1, args.lambda_points)

    # Build options
    options = []
    for mn in sorted(preds.keys()):
        for budget in sorted(preds[mn].keys()):
            options.append((mn, budget))

    n_opts = len(options)
    print(f"  {n_opts} (model, budget) options")

    pred_q = np.zeros((n_test, n_opts))
    true_q = np.zeros((n_test, n_opts))
    costs = np.zeros((n_test, n_opts))

    for j, (mn, budget) in enumerate(options):
        pdata = preds[mn][budget]
        pred_q[:, j] = pdata['pred_test']
        true_q[:, j] = pdata['true_test']
        costs[:, j] = models_data[mn][budget]['output_tokens'][test_idx]

    # Normalize costs per query
    cost_norm = np.zeros_like(costs)
    for i in range(n_test):
        ci = costs[i]
        valid = ~np.isnan(ci) & (ci > 0)
        if valid.sum() == 0:
            cost_norm[i] = 0
        else:
            cmin, cmax = ci[valid].min(), ci[valid].max()
            if cmax > cmin:
                cost_norm[i] = np.clip((ci - cmin) / (cmax - cmin), 0, 1)
            else:
                cost_norm[i] = 0

    results = []
    for lam in lambdas:
        risk = (1 - lam) * pred_q - lam * cost_norm
        best = np.nanargmax(risk, axis=1)
        sel_q = true_q[np.arange(n_test), best]
        sel_c = costs[np.arange(n_test), best]
        valid = ~np.isnan(sel_q) & ~np.isnan(sel_c) & (sel_c > 0)
        avg_q = sel_q[valid].mean() if valid.sum() > 10 else np.nanmean(sel_q)
        avg_c = sel_c[valid].mean() if valid.sum() > 10 else np.nanmean(sel_c)
        results.append({'lambda': lam, 'cost': avg_c, 'accuracy': avg_q})

    results_df = pd.DataFrame(results)

    # Compute oracle
    oracle_acc = np.nanmax(true_q, axis=1)
    oracle_mean = float(np.nanmean(oracle_acc))
    print(f"Oracle accuracy: {oracle_mean:.4f}")

    # Best single LLM
    best_llm_acc = 0
    best_name = ""
    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            acc = models_data[mn][budget]['accuracy'][test_idx]
            valid = ~np.isnan(acc)
            if valid.sum() > 100:
                m = float(acc[valid].mean())
                if m > best_llm_acc:
                    best_llm_acc = m
                    best_name = f"{mn}/{budget}"
    print(f"Best single LLM: {best_name} = {best_llm_acc:.4f}")

    # Compute metrics
    sorted_df = results_df.sort_values('cost')
    cc = sorted_df['cost'].values
    pc = sorted_df['accuracy'].values

    peak = float(pc.max())
    cmin, cmax = cc.min(), cc.max()
    nc = (cc - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(cc)
    try:
        audc = float(np.trapezoid(pc, nc))
    except AttributeError:
        audc = float(np.trapz(pc, nc))

    target = best_llm_acc * args.qnc_target_rate
    above = pc >= target
    qnc = float(nc[above][0]) if above.any() else 1.0

    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"  Peak Accuracy:    {peak:.4f}")
    print(f"  AUDC (norm cost): {audc:.4f}")
    print(f"  QNC:              {qnc:.4f}")
    print(f"  Oracle acc:       {oracle_mean:.4f}")
    print(f"  Best LLM:         {best_llm_acc:.4f}")

    metrics = {
        'peak_accuracy': peak,
        'AUDC': audc,
        'QNC': qnc,
        'oracle_accuracy': oracle_mean,
        'best_llm_accuracy': float(best_llm_acc),
        'qnc_target': float(target),
        'k_neighbors': args.k_neighbors,
    }

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    results_df.to_csv(os.path.join(args.output_dir, 'routing_curves.csv'), index=False)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
