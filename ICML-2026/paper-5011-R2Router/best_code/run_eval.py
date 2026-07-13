#!/usr/bin/env python3
"""
R2-Router evaluation: train Ridge regression predictors on RouterArena data,
evaluate routing at multiple lambda values, compute AUDC, QNC, Peak Acc.

Usage:
    python run_eval.py --training-data /datasets/training_data.pkl \
        --router-data-10 /datasets/router_data_10.json \
        --output-dir ./results
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description='R2-Router evaluation')
    parser.add_argument('--training-data', type=str, required=True,
                        help='Path to training_data.pkl')
    parser.add_argument('--router-data-10', type=str, required=True,
                        help='Path to router_data_10.json (sub_10 split)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--lambda-points', type=int, default=200,
                        help='Number of lambda points to evaluate')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Ridge regression alpha (default: 10.0)')
    parser.add_argument('--n-runs', type=int, default=1,
                        help='Number of independent runs with different seeds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--use-sub10', action='store_true', default=False,
                        help='Use sub_10 for training')
    parser.add_argument('--train-frac', type=float, default=0.2,
                        help='Fraction for training when not using sub10')
    parser.add_argument('--qnc-target-rate', type=float, default=1.0,
                        help='Target accuracy rate for QNC (1.0 = 100% of best LLM)')
    return parser.parse_args()


def load_data(training_data_path, router_data_10_path, use_sub10=True, train_frac=0.2, seed=42):
    """Load training data and create train/test split."""
    with open(training_data_path, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']  # (8400, 1024)
    models_data = data['models']
    global_indices = data['global_indices']

    n_queries = embeddings.shape[0]
    print(f"Loaded {n_queries} queries, {embeddings.shape[1]}d embeddings")
    print(f"Models: {list(models_data.keys())}")

    if use_sub10:
        # Use sub_10 for training
        with open(router_data_10_path) as f:
            sub10_data = json.load(f)
        sub10_gis = set(e['global index'] for e in sub10_data)
        gi_to_idx = {gi: i for i, gi in enumerate(global_indices)}

        train_idx = np.array(sorted([
            gi_to_idx[gi] for gi in sub10_gis if gi in gi_to_idx
        ]))
        test_idx = np.array(sorted(
            set(range(n_queries)) - set(train_idx.tolist())
        ))
        print(f"Train (sub_10): {len(train_idx)} queries")
    else:
        # Random split
        rng = np.random.RandomState(seed)
        all_idx = np.arange(n_queries)
        n_train = int(n_queries * train_frac)
        train_idx = rng.choice(all_idx, n_train, replace=False)
        test_idx = np.array(sorted(set(all_idx) - set(train_idx.tolist())))
        print(f"Train (random {train_frac:.0%}): {len(train_idx)} queries")

    print(f"Test: {len(test_idx)} queries")

    return embeddings, models_data, global_indices, train_idx, test_idx


def train_predictors(embeddings, models_data, train_idx, test_idx, alpha=10.0, seed=42):
    """
    Train Ridge regression predictors for each (model, budget) pair.
    Includes ALL models, even those with a single budget.
    """
    preds = {}

    # Include all models that have at least one budget
    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    print(f"\nTraining Ridge predictors (alpha={alpha}, seed={seed})...")
    n_trained = 0
    n_skipped = 0

    for mn in sorted(models_data.keys()):
        budgets = sorted(models_data[mn].keys())
        if not budgets:
            continue

        preds[mn] = {}
        mn_trained = 0

        for budget in budgets:
            bdata = models_data[mn][budget]
            y_all = bdata['accuracy']
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            # Filter NaN
            valid_train = ~np.isnan(y_train)
            valid_test = ~np.isnan(y_test)

            if valid_train.sum() < 10:
                n_skipped += 1
                continue

            # Standardize
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[valid_train])
            X_test_scaled = scaler.transform(X_test[valid_test])

            # Train Ridge
            model = Ridge(alpha=alpha, random_state=seed)
            model.fit(X_train_scaled, y_train[valid_train])

            # Predict
            y_pred = np.full(len(test_idx), np.nan)
            y_pred[valid_test] = model.predict(X_test_scaled)

            preds[mn][budget] = {
                'pred_test': y_pred,
                'true_test': y_test,
                'valid_test': valid_test,
                'train_score': model.score(X_train_scaled, y_train[valid_train]),
                'test_score': model.score(X_test_scaled, y_test[valid_test]),
            }
            mn_trained += 1
            n_trained += 1

        if mn_trained > 0:
            avg_test_r2 = np.mean([
                preds[mn][b]['test_score']
                for b in preds[mn] if not np.isnan(preds[mn][b]['test_score'])
            ])
            print(f"  {mn}: {mn_trained}/{len(budgets)} budgets, avg test R²={avg_test_r2:.4f}")

    print(f"Total: {n_trained} predictors trained, {n_skipped} skipped")
    return preds


def evaluate_routing(preds, models_data, test_idx, n_lambda=200):
    """
    Evaluate routing at multiple lambda values.

    risk = (1-lambda) * quality - lambda * cost_normalized
    """
    n_test = len(test_idx)
    lambdas = np.linspace(0, 1, n_lambda)

    # Collect all (model, budget) options
    options = []
    for mn in sorted(preds.keys()):
        for budget in sorted(preds[mn].keys()):
            options.append((mn, budget))

    print(f"\nEvaluating routing with {len(options)} (model, budget) options")
    print(f"Lambda: {n_lambda} points in [{lambdas[0]:.4f}, {lambdas[-1]:.4f}]")

    n_opts = len(options)
    pred_quality = np.zeros((n_test, n_opts))
    true_quality = np.zeros((n_test, n_opts))
    costs = np.zeros((n_test, n_opts))

    for j, (mn, budget) in enumerate(options):
        pdata = preds[mn][budget]
        pred_quality[:, j] = pdata['pred_test']

        bdata = models_data[mn][budget]
        true_quality[:, j] = pdata['true_test']
        costs[:, j] = bdata['output_tokens'][test_idx]

    # Per-query cost normalization for routing
    cost_norm = np.zeros_like(costs)
    for i in range(n_test):
        ci = costs[i]
        valid = ~np.isnan(ci) & (ci > 0)
        if valid.sum() == 0:
            cost_norm[i] = 0
        else:
            cmin = ci[valid].min()
            cmax = ci[valid].max()
            if cmax > cmin:
                cost_norm[i] = np.clip((ci - cmin) / (cmax - cmin), 0, 1)
            else:
                cost_norm[i] = 0

    results = []
    for lam in lambdas:
        risk = (1 - lam) * pred_quality - lam * cost_norm
        best_idx = np.nanargmax(risk, axis=1)

        selected_quality = true_quality[np.arange(n_test), best_idx]
        selected_cost = costs[np.arange(n_test), best_idx]

        valid = ~np.isnan(selected_quality) & ~np.isnan(selected_cost) & (selected_cost > 0)
        if valid.sum() > 10:
            avg_q = selected_quality[valid].mean()
            avg_c = selected_cost[valid].mean()
        else:
            avg_q = np.nanmean(selected_quality)
            avg_c = np.nanmean(selected_cost)

        results.append({
            'lambda': lam,
            'cost': avg_c,
            'accuracy': avg_q,
        })

    return pd.DataFrame(results), lambdas, options


def evaluate_oracle(models_data, test_idx):
    """
    Oracle routing: for each query, pick the (model, budget) with the
    highest TRUE quality (breaking ties by lowest cost). This is the
    theoretical upper bound for routing.
    """
    n_test = len(test_idx)

    # Collect options
    options = []
    qualities = []
    costs_list = []
    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            bdata = models_data[mn][budget]
            q = bdata['accuracy'][test_idx]
            c = bdata['output_tokens'][test_idx]
            options.append((mn, budget))
            qualities.append(q)
            costs_list.append(c)

    n_opts = len(options)
    quality_matrix = np.column_stack(qualities)
    cost_matrix = np.column_stack(costs_list)

    # Best per query
    best_idx = np.nanargmax(quality_matrix, axis=1)
    best_q = quality_matrix[np.arange(n_test), best_idx]
    best_c = cost_matrix[np.arange(n_test), best_idx]

    valid = ~np.isnan(best_q) & ~np.isnan(best_c)
    return float(np.nanmean(best_q[valid])), float(np.nanmean(best_c[valid]))


def find_best_single_llm(models_data, test_idx):
    """Find the single best (model, budget) for QNC target."""
    best_acc = 0.0
    best_name = None

    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            bdata = models_data[mn][budget]
            acc = bdata['accuracy'][test_idx]
            valid = ~np.isnan(acc)
            if valid.sum() > 100:
                mean_acc = float(acc[valid].mean())
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_name = f"{mn}/{budget}"

    print(f"\nBest single (model, budget): {best_name} = {best_acc:.4f}")
    return best_acc


def compute_metrics(results_df, best_llm_accuracy, target_rate=1.0):
    """Compute AUDC, QNC, Peak Acc."""
    sorted_df = results_df.sort_values('cost')
    cost_curve = sorted_df['cost'].values
    perf_curve = sorted_df['accuracy'].values

    peak_accuracy = float(perf_curve.max())

    cmin, cmax = cost_curve.min(), cost_curve.max()
    if cmax > cmin:
        norm_cost = (cost_curve - cmin) / (cmax - cmin)
    else:
        norm_cost = np.zeros_like(cost_curve)

    # AUDC with normalized cost
    audc = float(np.trapezoid(perf_curve, norm_cost))

    # QNC
    target_accuracy = best_llm_accuracy * target_rate
    above = perf_curve >= target_accuracy
    if above.any():
        qnc = float(norm_cost[above][0])
    else:
        qnc = 1.0

    audc_actual = float(np.trapezoid(perf_curve, cost_curve))

    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"  Peak Accuracy:       {peak_accuracy:.4f}")
    print(f"  AUDC (norm cost):    {audc:.4f}")
    print(f"  QNC ({target_rate*100:.0f}% of best LLM): {qnc:.4f}")
    print(f"  Best single LLM:     {best_llm_accuracy:.4f}")
    print(f"  QNC target:          {target_accuracy:.4f}")

    return {
        'peak_accuracy': peak_accuracy,
        'AUDC': audc,
        'QNC': qnc,
        'AUDC_actual': audc_actual,
        'best_llm_accuracy': float(best_llm_accuracy),
        'qnc_target': float(target_accuracy),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("R2-Router Evaluation Pipeline")
    print("=" * 60)

    embeddings, models_data, global_indices, train_idx, test_idx = \
        load_data(args.training_data, args.router_data_10,
                  use_sub10=args.use_sub10, train_frac=args.train_frac, seed=args.seed)

    all_metrics = []
    for run in range(args.n_runs):
        seed = args.seed + run
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{args.n_runs} (seed={seed})")
        print(f"{'='*60}")

        preds = train_predictors(
            embeddings, models_data, train_idx, test_idx,
            alpha=args.alpha, seed=seed
        )

        results_df, lambdas, options = evaluate_routing(
            preds, models_data, test_idx, n_lambda=args.lambda_points
        )

        # Oracle and best single LLM
        oracle_acc, oracle_cost = evaluate_oracle(models_data, test_idx)
        print(f"\nOracle routing: acc={oracle_acc:.4f}, cost={oracle_cost:.1f}")

        best_llm_acc = find_best_single_llm(models_data, test_idx)

        metrics = compute_metrics(results_df, best_llm_acc, args.qnc_target_rate)
        metrics['run'] = run + 1
        metrics['oracle_accuracy'] = oracle_acc
        all_metrics.append(metrics)

        run_dir = os.path.join(args.output_dir, f'run_{run+1}')
        os.makedirs(run_dir, exist_ok=True)
        results_df.to_csv(os.path.join(run_dir, 'routing_curves.csv'), index=False)

    if args.n_runs > 1:
        print(f"\n{'='*60}")
        print(f"AGGREGATE ({args.n_runs} runs)")
        print(f"{'='*60}")
        agg = {}
        for key in ['peak_accuracy', 'AUDC', 'QNC']:
            vals = [m[key] for m in all_metrics]
            mean_v, std_v = np.mean(vals), np.std(vals)
            print(f"  {key}: {mean_v:.4f} +/- {std_v:.4f}")
            agg[key] = float(mean_v)
            agg[f'{key}_std'] = float(std_v)
        with open(os.path.join(args.output_dir, 'aggregate_metrics.json'), 'w') as f:
            json.dump(agg, f, indent=2)
    else:
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(all_metrics[0], f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")
    return all_metrics


if __name__ == '__main__':
    main()
