#!/usr/bin/env python3
"""
Route all 8400 queries using Global KNN (trained on sub_10) and export submission JSON.

Usage:
    .venv/bin/python scripts/route_knn_export.py \
        --models 235b gpt4o gemini-flash \
        --lambda_val 0.995 \
        --export routerarena_submission/r2-router-knn.json
"""

import os
import sys
import json
import math
import pickle
import argparse
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    MODELS, TRAINING_DATA_PATH, MODEL_COST_PATH,
    ROUTER_DATA_PATH, ROUTER_DATA_10_PATH,
)


def arena_score(accuracy, cost_per_1kq, beta=0.1):
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


def load_sweep_entry(model_name, budget):
    sweep_dir = MODELS[model_name]["sweep_dir"]
    path = os.path.join(sweep_dir, f"{budget}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {e["global index"]: e for e in data if not e.get("for_optimality")}


def main():
    parser = argparse.ArgumentParser(description="Route with Global KNN and export submission")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--exclude-budgets", nargs="*", default=[])
    parser.add_argument("--lambda_val", type=float, required=True)
    parser.add_argument("--k", type=int, default=28)
    parser.add_argument("--export", type=str, required=True)
    args = parser.parse_args()

    allowed_models = set(args.models) if args.models else set(MODELS.keys())
    excluded_budgets = set(args.exclude_budgets)
    lam = args.lambda_val
    knn_k = args.k

    print("=" * 80)
    print(f"R2-Router KNN Export (sub_10 train, λ={lam})")
    print("=" * 80)
    print(f"  Models: {sorted(allowed_models)}")
    print(f"  Excluded budgets: {sorted(excluded_budgets) if excluded_budgets else 'none'}")
    print(f"  KNN K: {knn_k}")
    print(f"  Export: {args.export}")
    print()

    # 1. Load data
    print("Loading training data...", flush=True)
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    models_data = data["models"]
    global_indices = data["global_indices"]
    n = embeddings.shape[0]
    print(f"  {n} queries, {embeddings.shape[1]}d embeddings")

    # 2. sub_10 training indices
    print("Loading sub_10 split...", flush=True)
    with open(ROUTER_DATA_10_PATH) as f:
        sub10 = json.load(f)
    sub10_gis = set(e["global index"] for e in sub10)
    train_idx = np.array([i for i, gi in enumerate(global_indices) if gi in sub10_gis])
    print(f"  Train: {len(train_idx)} queries (sub_10)")

    X_train = embeddings[train_idx]

    # 3. Train Global KNN per (model, budget)
    print("\nTraining Global KNN predictors...", flush=True)
    quality_knn = {}
    token_knn = {}

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

        token_budget = "concise" if "concise" in budgets_data else next(iter(budgets_data), None)
        if token_budget and "output_tokens" in budgets_data[token_budget]:
            y_tok = budgets_data[token_budget]["output_tokens"][train_idx]
            valid = ~np.isnan(y_tok)
            if valid.sum() >= 3:
                tknn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum() - 1),
                                           metric="cosine", weights="distance")
                tknn.fit(X_train[valid], y_tok[valid])
                token_knn[mn] = tknn

        print(f"  {mn:<15} {len(quality_knn[mn])} budget heads trained", flush=True)

    prices = load_prices()

    # 4. Route all queries
    print(f"\nRouting all {n} queries (λ={lam})...", flush=True)
    best_risk = np.full(n, -np.inf)
    best_model = [None] * n
    best_budget = [None] * n

    for mn in allowed_models:
        if mn not in quality_knn:
            continue
        price = prices[mn]

        if mn in token_knn:
            tok = np.maximum(1.0, token_knn[mn].predict(embeddings))
        else:
            tok = np.full(n, 50.0)

        for budget, knn in quality_knn[mn].items():
            q = knn.predict(embeddings)
            risk = (1 - lam) * q - lam * tok * price / 1e6

            better = risk > best_risk
            for j in np.where(better)[0]:
                best_risk[j] = risk[j]
                best_model[j] = mn
                best_budget[j] = budget

    routes = list(zip(best_model, best_budget))

    # 5. Evaluate using training_data.pkl
    total_acc = 0.0
    total_cost = 0.0
    model_counts = defaultdict(int)
    budget_counts = defaultdict(int)
    for i, (mn, budget) in enumerate(routes):
        model_counts[mn] += 1
        budget_counts[budget] += 1
        if mn in models_data and budget in models_data[mn]:
            total_acc += float(models_data[mn][budget]["accuracy"][i])
            tokens = float(models_data[mn][budget]["output_tokens"][i])
            total_cost += tokens * prices[mn] / 1e6
    accuracy = total_acc / n
    cost_1kq = total_cost / n * 1000

    print(f"\n{'=' * 80}")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Cost/1kq:  ${cost_1kq:.4f}")
    for beta in [0.05, 0.1, 19.0]:
        print(f"  Arena(β={beta}): {arena_score(accuracy, cost_1kq, beta):.2f}")

    print(f"\n  Model distribution:")
    for mn, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {mn:<20} {cnt:>5} ({cnt/n*100:.1f}%)")

    print(f"\n  Budget distribution:")
    for b, cnt in sorted(budget_counts.items(), key=lambda x: -x[1]):
        print(f"    {b:<25} {cnt:>5} ({cnt/n*100:.1f}%)")

    # 6. Export submission JSON
    print(f"\nExporting submission...", flush=True)
    sweep_cache = {}
    entries = []

    for i, (mn, budget) in enumerate(routes):
        gi = global_indices[i]
        key = (mn, budget)
        if key not in sweep_cache:
            sweep_cache[key] = load_sweep_entry(mn, budget)

        entry = sweep_cache[key].get(gi)
        if entry is None:
            print(f"  WARNING: No sweep entry for {gi} -> {mn}/{budget}")
            continue

        entries.append({
            "global index": gi,
            "prompt": entry["prompt"],
            "prediction": MODELS[mn]["ra_name"],
            "generated_result": entry.get("generated_result", ""),
            "cost": entry.get("cost", 0),
            "accuracy": entry.get("accuracy", 0),
            "for_optimality": False,
            "_r2_model": mn,
            "_r2_budget_label": budget,
        })

    # Optimality entries (sub_10 × other models, concise budget)
    used_models = set(mn for mn, _ in routes)
    gi_to_model = {global_indices[i]: routes[i][0] for i in range(n)}

    for rd_entry in sub10:
        gi = rd_entry["global index"]
        chosen_model = gi_to_model.get(gi)

        for mn in used_models:
            if mn == chosen_model:
                continue
            budget = "concise"
            key = (mn, budget)
            if key not in sweep_cache:
                sweep_cache[key] = load_sweep_entry(mn, budget)
            sweep_entry = sweep_cache[key].get(gi)
            if sweep_entry:
                entries.append({
                    "global index": gi,
                    "prompt": sweep_entry["prompt"],
                    "prediction": MODELS[mn]["ra_name"],
                    "generated_result": sweep_entry.get("generated_result", ""),
                    "cost": sweep_entry.get("cost", 0),
                    "accuracy": sweep_entry.get("accuracy", 0),
                    "for_optimality": True,
                    "_r2_model": mn,
                    "_r2_budget_label": budget,
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
    with open(args.export, "w") as f:
        json.dump(entries, f, indent=2)

    n_regular = sum(1 for e in entries if not e["for_optimality"])
    n_opt = sum(1 for e in entries if e["for_optimality"])
    print(f"\nExported {len(entries)} entries ({n_regular} regular, {n_opt} optimality) to {args.export}")
    print("Done.")


if __name__ == "__main__":
    main()
