#!/usr/bin/env python3
"""
Sweep lambda/beta using Global KNN predictor (no per-category split).

1. 10/90 StratifiedShuffleSplit on categories
2. Train one KNN per (model, budget) on the 10% training set (all categories together)
3. Route all 8400 queries using KNN predictions
4. Evaluate on full 8400 using ground-truth from training_data.pkl

Usage:
    .venv/bin/python scripts/sweep_lambda_global_knn.py --models 235b 80b gemini-flash haiku
    .venv/bin/python scripts/sweep_lambda_global_knn.py --models 235b 80b gemini-flash haiku gpt-5.2
"""

import os
import sys
import json
import math
import pickle
import argparse
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    MODELS, TRAINING_DATA_PATH, MODEL_COST_PATH,
    ROUTER_DATA_PATH, ROUTER_DATA_10_PATH,
)


def arena_score(accuracy, cost_per_1kq, beta=0.1):
    """RouterArena Arena Score: S = (1+beta)*A*C / (beta*A + C) * 100
    beta < 1 favors accuracy, beta > 1 favors cost."""
    if cost_per_1kq <= 0:
        cost_per_1kq = 0.001
    C = (math.log2(200) - math.log2(cost_per_1kq)) / (math.log2(200) - math.log2(0.0044))
    C = max(0.0, min(1.0, C))
    A = accuracy
    denom = beta * A + C
    if denom == 0:
        return 0.0
    return ((1 + beta) * A * C) / denom * 100


def load_prices():
    with open(MODEL_COST_PATH) as f:
        cost_data = json.load(f)
    prices = {}
    for model_name, cfg in MODELS.items():
        cost_key = cfg["cost_key"]
        prices[model_name] = cost_data.get(cost_key, {}).get("output_token_price_per_million", 0)
    return prices


def load_sweep_costs(global_indices, allowed_models, excluded_budgets):
    """Load real per-query cost from sweep JSON files.
    Returns: {model: {budget: np.array(n,)}} with real cost per query."""
    gi_to_idx = {gi: i for i, gi in enumerate(global_indices)}
    n = len(global_indices)
    sweep_costs = {}

    for mn in sorted(allowed_models):
        if mn not in MODELS:
            continue
        sweep_dir = MODELS[mn]["sweep_dir"]
        sweep_costs[mn] = {}

        for budget_file in sorted(os.listdir(sweep_dir)):
            if not budget_file.endswith(".json"):
                continue
            budget = budget_file.replace(".json", "")
            if budget in excluded_budgets:
                continue

            filepath = os.path.join(sweep_dir, budget_file)
            with open(filepath) as f:
                entries = json.load(f)

            costs = np.full(n, np.nan)
            for e in entries:
                if e.get("for_optimality"):
                    continue
                gi = e.get("global index")
                idx = gi_to_idx.get(gi)
                if idx is not None and e.get("cost") is not None:
                    costs[idx] = float(e["cost"])

            if not np.all(np.isnan(costs)):
                sweep_costs[mn][budget] = costs

        print(f"  {mn:<15} {len(sweep_costs[mn])} budgets loaded", flush=True)

    return sweep_costs


def main():
    parser = argparse.ArgumentParser(description="Global KNN Lambda/Beta Sweep")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--exclude-budgets", nargs="*", default=[])
    parser.add_argument("--lambdas", nargs="*", type=float,
                        default=[0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0])
    parser.add_argument("--betas", nargs="*", type=float,
                        default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--k", type=int, default=0,
                        help="KNN K value (0=auto: sqrt(n_train), capped at 256)")
    parser.add_argument("--train-size", type=float, default=0.1,
                        help="Fraction of data for training (default: 0.1)")
    parser.add_argument("--use-sub10", action="store_true",
                        help="Use RouterArena sub_10 (809 queries) as training set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()

    allowed_models = set(args.models) if args.models else set(MODELS.keys())
    excluded_budgets = set(args.exclude_budgets)

    train_label = "sub_10 (809)" if args.use_sub10 else f"{int(args.train_size * 100)}%"
    print("=" * 100)
    print(f"Global KNN Lambda/Beta Sweep ({train_label} train, 100% eval)")
    print("=" * 100)
    print(f"  Models ({len(allowed_models)}): {sorted(allowed_models)}")
    print(f"  Excluded budgets: {sorted(excluded_budgets) if excluded_budgets else 'none'}")
    print(f"  Train set: {train_label}")
    print(f"  Lambdas: {args.lambdas}")
    print(f"  Betas: {args.betas}")
    print(f"  Seed: {args.seed}")
    print()

    # 1. Load data
    print("Loading training data...", flush=True)
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]       # (8400, 1024)
    categories = data["categories"]       # (8400,)
    models_data = data["models"]
    n = embeddings.shape[0]
    print(f"  {n} queries, {embeddings.shape[1]}d embeddings")

    # 2. Select training indices
    if args.use_sub10:
        # Use RouterArena's official sub_10 split (809 queries)
        with open(ROUTER_DATA_10_PATH) as f:
            sub10 = json.load(f)
        sub10_gis = set(e["global index"] for e in sub10)
        global_indices = data["global_indices"]
        train_idx = np.array([i for i, gi in enumerate(global_indices) if gi in sub10_gis])
        print(f"  Train: {len(train_idx)} queries (sub_10)")
    else:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.train_size, random_state=args.seed)
        train_idx, _ = next(sss.split(embeddings, categories))
        print(f"  Train: {len(train_idx)} queries ({int(args.train_size * 100)}%)")

    X_train = embeddings[train_idx]
    n_train = len(train_idx)

    # KNN K
    knn_k = args.k if args.k > 0 else min(256, max(3, int(np.sqrt(n_train))))
    print(f"  KNN K: {knn_k}")

    # 3. Train Global KNN per (model, budget)
    print("\nTraining Global KNN predictors...", flush=True)
    quality_knn = {}   # {model: {budget: KNeighborsRegressor}}
    token_knn = {}     # {model: KNeighborsRegressor}

    for mn in sorted(allowed_models):
        if mn not in models_data:
            print(f"  {mn}: not in training data, skipping", flush=True)
            continue
        quality_knn[mn] = {}
        budgets_data = models_data[mn]

        for budget, bdata in budgets_data.items():
            if budget in excluded_budgets:
                continue
            y_train = bdata["accuracy"][train_idx]
            valid = ~np.isnan(y_train)
            if valid.sum() < 3:
                continue
            knn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum() - 1),
                                     metric="cosine", weights="distance")
            knn.fit(X_train[valid], y_train[valid])
            quality_knn[mn][budget] = knn

        # Token predictor: use concise budget's output_tokens if available
        token_budget = "concise" if "concise" in budgets_data else next(iter(budgets_data), None)
        if token_budget and "output_tokens" in budgets_data[token_budget]:
            y_tok = budgets_data[token_budget]["output_tokens"][train_idx]
            valid = ~np.isnan(y_tok)
            if valid.sum() >= 3:
                tknn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum() - 1),
                                           metric="cosine", weights="distance")
                tknn.fit(X_train[valid], y_tok[valid])
                token_knn[mn] = tknn

        n_budgets = len(quality_knn[mn])
        print(f"  {mn:<15} {n_budgets} budget heads trained", flush=True)

    prices = load_prices()

    # Load real costs from sweep files
    global_indices = data["global_indices"]
    print("\nLoading real costs from sweep files...", flush=True)
    sweep_costs = load_sweep_costs(global_indices, allowed_models, excluded_budgets)

    print(f"\n  Setup done in {time.time()-start:.1f}s\n", flush=True)

    # 4. Route all 8400 queries
    def route_all(lam):
        n_total = embeddings.shape[0]
        best_risk = np.full(n_total, -np.inf)
        best_model = [None] * n_total
        best_budget = [None] * n_total

        for mn in allowed_models:
            if mn not in quality_knn:
                continue
            price = prices[mn]

            # Token prediction
            if mn in token_knn:
                tok = np.maximum(1.0, token_knn[mn].predict(embeddings))
            else:
                tok = np.full(n_total, 50.0)

            for budget, knn in quality_knn[mn].items():
                q = knn.predict(embeddings)
                risk = (1 - lam) * q - lam * tok * price / 1e6

                better = risk > best_risk
                for j in np.where(better)[0]:
                    best_risk[j] = risk[j]
                    best_model[j] = mn
                    best_budget[j] = budget

        return list(zip(best_model, best_budget))

    # 5. Evaluate (using real cost from sweep files)
    def evaluate(routes):
        total_acc = 0.0
        total_cost = 0.0
        model_counts = defaultdict(int)
        for i, (mn, budget) in enumerate(routes):
            model_counts[mn] += 1
            if mn in models_data and budget in models_data[mn]:
                total_acc += float(models_data[mn][budget]["accuracy"][i])
            # Use real cost from sweep files
            if mn in sweep_costs and budget in sweep_costs[mn]:
                c = sweep_costs[mn][budget][i]
                if not np.isnan(c):
                    total_cost += c
        accuracy = total_acc / len(routes)
        cost_1kq = total_cost / len(routes) * 1000
        return accuracy, cost_1kq, model_counts

    # 6. Sweep
    results = []
    for lam in args.lambdas:
        t0 = time.time()
        routes = route_all(lam)
        accuracy, cost_1kq, model_counts = evaluate(routes)
        elapsed = time.time() - t0

        scores = {}
        for beta in args.betas:
            scores[beta] = arena_score(accuracy, cost_1kq, beta)

        results.append({
            "lambda": lam, "accuracy": accuracy, "cost_1kq": cost_1kq,
            "scores": scores, "model_counts": dict(model_counts), "elapsed": elapsed,
        })
        print(f"  \u03bb={lam:.4f}  Acc={accuracy*100:6.2f}%  Cost=${cost_1kq:.4f}/1kq  "
              f"Arena(\u03b2=0.1)={scores.get(0.1, 0):6.2f}  [{elapsed:.1f}s]", flush=True)

    # 7. Summary
    LAM = "lambda"
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    beta_headers = "  ".join(f"Arena(b={b})" for b in args.betas)
    print(f"{LAM:>8}  {'Accuracy':>10}  {'Cost/1kq':>10}  {beta_headers}")
    print("-" * (32 + 12 * len(args.betas)))
    for r in results:
        scores_str = "  ".join(f"{r['scores'][b]:>10.2f}" for b in args.betas)
        print(f"{r['lambda']:>8.4f}  {r['accuracy']*100:>9.2f}%  ${r['cost_1kq']:>9.4f}  {scores_str}")

    # Per-model distribution
    print("\n" + "=" * 100)
    print("PER-MODEL DISTRIBUTION (%)")
    print("=" * 100)
    all_models_used = sorted(set().union(*(r["model_counts"].keys() for r in results)))
    model_headers = "  ".join(f"{m:>12}" for m in all_models_used)
    print(f"{LAM:>8}  {model_headers}")
    print("-" * (10 + 14 * len(all_models_used)))
    for r in results:
        total = sum(r["model_counts"].values())
        pcts = []
        for m in all_models_used:
            count = r["model_counts"].get(m, 0)
            pcts.append(f"{count/total*100:>11.1f}%")
        print(f"{r['lambda']:>8.4f}  {'  '.join(pcts)}")

    # Best lambda per beta
    print("\n" + "=" * 100)
    print("BEST LAMBDA PER BETA")
    print("=" * 100)
    print(f"{'Beta':>8}  {'Best lam':>8}  {'Arena':>10}  {'Accuracy':>10}  {'Cost/1kq':>10}")
    print("-" * 55)
    for beta in args.betas:
        best = max(results, key=lambda r: r["scores"][beta])
        print(f"{beta:>8.2f}  {best['lambda']:>8.4f}  {best['scores'][beta]:>10.2f}  "
              f"{best['accuracy']*100:>9.2f}%  ${best['cost_1kq']:>9.4f}")

    print(f"\nTotal time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
