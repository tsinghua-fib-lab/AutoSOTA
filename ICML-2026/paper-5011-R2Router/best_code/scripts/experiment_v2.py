#!/usr/bin/env python3
"""
Experiment V2: Alternative routing strategies.

Key insight from V1: KNN quality prediction is too noisy with 809 training points
to correctly route both model AND budget. Oracle with 235b-only gets 78% at $0.032,
but KNN routing only gets ~69%.

New strategies:
1. Model-only routing with fixed budget allocation
2. Budget selection by query difficulty (embedding-based heuristic)
3. Two-stage: first pick model, then pick budget
4. KNN classification (binary accuracy) instead of regression
5. Weighted ensemble: combine model-level accuracy with per-query KNN
"""

import os
import sys
import json
import math
import pickle
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import MODELS, TRAINING_DATA_PATH, MODEL_COST_PATH, ROUTER_DATA_10_PATH


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


def load_sweep_costs(global_indices):
    """Load real per-query costs from sweep files, indexed by position."""
    sweep_root = '/orange/qi855292.ucf/ah872032.ucf/budget_sweep'
    dir_to_model = {
        '235b': '235b', '80b': '80b', '30b': '30b',
        'coder-30b': 'coder-30b', 'Qwen3-Coder-Next': 'coder-next',
        'gemini-flash': 'gemini-flash', 'gpt4o': 'gpt4o',
        'haiku': 'haiku', 'ministral-3b': 'ministral-3b',
        'ministral-8b': 'ministral-8b', 'ministral-14b': 'ministral-14b',
        'gemma-3n-e4b': 'gemma-3n-e4b', 'gpt-5.2': 'gpt-5.2',
        'gemini-3-pro': 'gemini-3-pro', 'glm-5': 'glm-5',
    }
    gi_to_idx = {gi: i for i, gi in enumerate(global_indices)}
    n = len(global_indices)

    sweep_costs = {}  # model -> budget -> np.array(n)
    for mn_dir in os.listdir(sweep_root):
        mn = dir_to_model.get(mn_dir, mn_dir)
        sweep_dir = os.path.join(sweep_root, mn_dir)
        if not os.path.isdir(sweep_dir):
            continue
        sweep_costs[mn] = {}
        for fname in os.listdir(sweep_dir):
            if not fname.endswith('.json'):
                continue
            budget = fname.replace('.json', '')
            with open(os.path.join(sweep_dir, fname)) as f:
                data = json.load(f)
            costs = np.zeros(n)
            for e in data:
                if not e.get('for_optimality'):
                    gi = e['global index']
                    if gi in gi_to_idx:
                        costs[gi_to_idx[gi]] = e.get('cost', 0)
            sweep_costs[mn][budget] = costs
    return sweep_costs


def evaluate_routing(routes, models_data, sweep_costs, global_indices, n):
    """Evaluate routing decisions using real sweep costs."""
    total_acc = 0.0
    total_cost = 0.0
    for i in range(n):
        mn, budget = routes[i]
        if mn in models_data and budget in models_data[mn]:
            total_acc += float(models_data[mn][budget]["accuracy"][i])
        if mn in sweep_costs and budget in sweep_costs[mn]:
            total_cost += sweep_costs[mn][budget][i]
    accuracy = total_acc / n
    cost_1kq = total_cost / n * 1000
    return accuracy, cost_1kq


def print_result(label, acc, cost, target_acc=0.72, target_cost=0.05):
    ascore = arena_score(acc, cost, 0.1)
    meets = "YES" if acc > target_acc and cost < target_cost else ""
    print(f"  {label:<60} Acc={acc*100:.2f}%  Cost=${cost:.4f}  Arena={ascore:.2f}  {meets}", flush=True)
    return acc, cost, ascore


def main():
    print("=" * 100, flush=True)
    print("R2-Router Experiment V2: Alternative Strategies", flush=True)
    print("=" * 100, flush=True)

    # Load data
    print("\n[1] Loading data...", flush=True)
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    models_data = data["models"]
    global_indices = data["global_indices"]
    n = embeddings.shape[0]

    with open(ROUTER_DATA_10_PATH) as f:
        sub10 = json.load(f)
    sub10_gis = set(e["global index"] for e in sub10)
    train_idx = np.array([i for i, gi in enumerate(global_indices) if gi in sub10_gis])
    X_train = embeddings[train_idx]
    print(f"  {n} queries, {len(train_idx)} train", flush=True)

    sweep_costs = load_sweep_costs(global_indices)

    # Available budgets for vLLM models
    vllm_budgets = ["concise", "budget_200", "budget_400", "budget_800"]
    excluded_budgets = {"budget_unlimited", "budget_1500"}

    best_results = []

    # =========================================================================
    # Strategy 1: 235b-only with smart budget allocation
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("[Strategy 1] 235b-only, smart budget by query difficulty", flush=True)
    print("=" * 100, flush=True)

    # Train a KNN to predict 235b accuracy at different budgets
    # Use the DIFFERENCE between concise and budget_800 as "difficulty"
    mn = "235b"
    acc_concise = models_data[mn]["concise"]["accuracy"]
    acc_800 = models_data[mn]["budget_800"]["accuracy"]

    # For training: learn which queries benefit from higher budget
    # If concise already gets it right, use concise. If not, try higher budget.

    # Simple heuristic: predict P(correct | concise) using KNN
    for knn_k in [5, 10, 20, 28, 50, 80]:
        y_train_concise = acc_concise[train_idx]
        valid = ~np.isnan(y_train_concise)
        knn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum()-1),
                                   metric="cosine", weights="distance")
        knn.fit(X_train[valid], y_train_concise[valid])
        pred_concise = knn.predict(embeddings)

        # For each threshold, route: if P(correct|concise) > threshold, use concise
        # else use budget_800 (or budget_400, budget_200)
        for fallback_budget in ["budget_200", "budget_400", "budget_800"]:
            for threshold in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
                routes = []
                for i in range(n):
                    if pred_concise[i] >= threshold:
                        routes.append((mn, "concise"))
                    else:
                        routes.append((mn, fallback_budget))

                acc, cost = evaluate_routing(routes, models_data, sweep_costs, global_indices, n)
                if acc > 0.715 and cost < 0.06:
                    pct_concise = sum(1 for r in routes if r[1] == "concise") / n * 100
                    print_result(f"235b k={knn_k} thresh={threshold} fb={fallback_budget} ({pct_concise:.0f}% concise)",
                               acc, cost)

    # =========================================================================
    # Strategy 2: Multi-budget 235b with graduated thresholds
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("[Strategy 2] 235b multi-budget graduated routing", flush=True)
    print("=" * 100, flush=True)

    # Train KNN to predict accuracy at each budget
    budget_order = ["concise", "budget_200", "budget_400", "budget_800"]

    for knn_k in [10, 20, 28, 50]:
        knn_preds = {}
        for budget in budget_order:
            if budget not in models_data[mn]:
                continue
            y = models_data[mn][budget]["accuracy"][train_idx]
            valid = ~np.isnan(y)
            knn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum()-1),
                                       metric="cosine", weights="distance")
            knn.fit(X_train[valid], y[valid])
            knn_preds[budget] = knn.predict(embeddings)

        # Graduated: start with concise, upgrade if predicted gain > threshold
        for upgrade_thresh in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
            routes = []
            for i in range(n):
                chosen = "concise"
                chosen_pred = knn_preds["concise"][i]
                for budget in ["budget_200", "budget_400", "budget_800"]:
                    if budget in knn_preds:
                        if knn_preds[budget][i] - chosen_pred > upgrade_thresh:
                            chosen = budget
                            chosen_pred = knn_preds[budget][i]
                routes.append((mn, chosen))

            acc, cost = evaluate_routing(routes, models_data, sweep_costs, global_indices, n)
            if acc > 0.715 and cost < 0.06:
                budget_dist = defaultdict(int)
                for _, b in routes:
                    budget_dist[b] += 1
                dist_str = " ".join(f"{b}={c/n*100:.0f}%" for b, c in sorted(budget_dist.items()))
                print_result(f"235b grad k={knn_k} thresh={upgrade_thresh} [{dist_str}]", acc, cost)

    # =========================================================================
    # Strategy 3: Model routing (235b vs cheap model) + budget
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("[Strategy 3] Model routing: 235b + secondary model + budget heuristic", flush=True)
    print("=" * 100, flush=True)

    # For each secondary model, predict: is 235b better or is secondary better?
    # Route to whichever is predicted better, with budget based on confidence
    secondary_models = ["gemini-flash", "30b", "coder-30b", "ministral-3b"]

    for sec_mn in secondary_models:
        if sec_mn not in models_data:
            continue

        # Get best budget for secondary (concise for API models)
        if MODELS[sec_mn]["type"] == "api":
            sec_budgets = ["concise"]
        else:
            sec_budgets = ["concise", "budget_200"]

        for sec_budget in sec_budgets:
            if sec_budget not in models_data[sec_mn]:
                continue

            for knn_k in [10, 20, 28, 50]:
                # Train KNN for 235b at various budgets
                knn_235b = {}
                for budget in ["concise", "budget_200", "budget_400", "budget_800"]:
                    if budget not in models_data["235b"]:
                        continue
                    y = models_data["235b"][budget]["accuracy"][train_idx]
                    valid = ~np.isnan(y)
                    knn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum()-1),
                                               metric="cosine", weights="distance")
                    knn.fit(X_train[valid], y[valid])
                    knn_235b[budget] = knn.predict(embeddings)

                # Train KNN for secondary model
                y_sec = models_data[sec_mn][sec_budget]["accuracy"][train_idx]
                valid_sec = ~np.isnan(y_sec)
                if valid_sec.sum() < 3:
                    continue
                knn_sec = KNeighborsRegressor(n_neighbors=min(knn_k, valid_sec.sum()-1),
                                               metric="cosine", weights="distance")
                knn_sec.fit(X_train[valid_sec], y_sec[valid_sec])
                pred_sec = knn_sec.predict(embeddings)

                # Route: pick model+budget with highest predicted accuracy
                for budget_235b in ["concise", "budget_200", "budget_400"]:
                    if budget_235b not in knn_235b:
                        continue
                    pred_235b = knn_235b[budget_235b]

                    routes = []
                    for i in range(n):
                        if pred_235b[i] >= pred_sec[i]:
                            routes.append(("235b", budget_235b))
                        else:
                            routes.append((sec_mn, sec_budget))

                    acc, cost = evaluate_routing(routes, models_data, sweep_costs, global_indices, n)
                    if acc > 0.715 and cost < 0.06:
                        n_235b = sum(1 for r in routes if r[0] == "235b")
                        print_result(
                            f"235b({budget_235b})+{sec_mn}({sec_budget}) k={knn_k} "
                            f"[235b={n_235b/n*100:.0f}%]",
                            acc, cost)

    # =========================================================================
    # Strategy 4: KNN classification (binary)
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("[Strategy 4] KNN Classification (binary accuracy)", flush=True)
    print("=" * 100, flush=True)

    pool_configs = [
        ("235b+flash", ["235b", "gemini-flash"]),
        ("235b+m3b+flash", ["235b", "ministral-3b", "gemini-flash"]),
        ("235b+30b+flash", ["235b", "30b", "gemini-flash"]),
        ("baseline_4", ["235b", "80b", "gemini-flash", "haiku"]),
    ]

    for pool_name, pool_models in pool_configs:
        for knn_k in [10, 20, 28, 50]:
            # Train KNN classifiers for each (model, budget)
            classifiers = {}
            for mn in pool_models:
                if mn not in models_data:
                    continue
                for budget in models_data[mn]:
                    if budget in excluded_budgets:
                        continue
                    y = models_data[mn][budget]["accuracy"][train_idx]
                    valid = ~np.isnan(y)
                    if valid.sum() < 3:
                        continue
                    # Binary: round to 0/1
                    y_bin = (y[valid] >= 0.5).astype(int)
                    if len(np.unique(y_bin)) < 2:
                        continue
                    clf = KNeighborsClassifier(n_neighbors=min(knn_k, valid.sum()-1),
                                               metric="cosine", weights="distance")
                    clf.fit(X_train[valid], y_bin)
                    # Get probability of correct
                    proba = clf.predict_proba(embeddings)
                    # proba[:, 1] = P(correct)
                    if proba.shape[1] == 2:
                        classifiers[(mn, budget)] = proba[:, 1]
                    else:
                        classifiers[(mn, budget)] = clf.predict(embeddings).astype(float)

            # Route: for each query, pick (model, budget) with highest P(correct),
            # breaking ties by lowest cost
            for lam in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]:
                routes = []
                for i in range(n):
                    best_score = -np.inf
                    best_choice = ("235b", "concise")

                    for (mn, budget), proba in classifiers.items():
                        # Use token cost as tie-breaker
                        cost_i = sweep_costs.get(mn, {}).get(budget, np.zeros(1))[i] if mn in sweep_costs and budget in sweep_costs.get(mn, {}) else 0
                        score = (1 - lam) * proba[i] - lam * cost_i
                        if score > best_score:
                            best_score = score
                            best_choice = (mn, budget)

                    routes.append(best_choice)

                acc, cost = evaluate_routing(routes, models_data, sweep_costs, global_indices, n)
                if acc > 0.715 and cost < 0.06:
                    print_result(f"Classify {pool_name} k={knn_k} λ={lam}", acc, cost)

    # =========================================================================
    # Strategy 5: Global mean accuracy routing (no per-query prediction)
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("[Strategy 5] Mean-accuracy routing (model-level, no per-query)", flush=True)
    print("=" * 100, flush=True)

    # Compute mean accuracy per (model, budget) on training set
    for pool_name, pool_models in pool_configs:
        train_means = {}
        for mn in pool_models:
            if mn not in models_data:
                continue
            for budget in models_data[mn]:
                if budget in excluded_budgets:
                    continue
                y = models_data[mn][budget]["accuracy"][train_idx]
                valid = ~np.isnan(y)
                if valid.sum() == 0:
                    continue
                train_means[(mn, budget)] = float(np.nanmean(y[valid]))

        # Train cost estimate
        train_cost_means = {}
        for mn in pool_models:
            if mn not in sweep_costs:
                continue
            for budget in sweep_costs[mn]:
                if budget in excluded_budgets:
                    continue
                costs = sweep_costs[mn][budget][train_idx]
                train_cost_means[(mn, budget)] = float(np.mean(costs))

        for lam in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]:
            best_choice = None
            best_score = -np.inf
            for (mn, budget), mean_acc in train_means.items():
                mean_cost = train_cost_means.get((mn, budget), 0)
                score = (1 - lam) * mean_acc - lam * mean_cost
                if score > best_score:
                    best_score = score
                    best_choice = (mn, budget)

            if best_choice:
                routes = [best_choice] * n
                acc, cost = evaluate_routing(routes, models_data, sweep_costs, global_indices, n)
                print_result(f"MeanRoute {pool_name} λ={lam} -> {best_choice}", acc, cost)

    # =========================================================================
    # Strategy 6: Hybrid — KNN model selection + fixed budget
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("[Strategy 6] KNN model selection + fixed budget per model", flush=True)
    print("=" * 100, flush=True)

    # For each model, pre-assign a fixed budget. Then KNN only decides WHICH model.
    budget_assignments = {
        "235b": ["concise", "budget_200", "budget_400"],
        "gemini-flash": ["concise"],
        "haiku": ["concise"],
        "30b": ["concise", "budget_200"],
        "80b": ["concise"],
        "ministral-3b": ["concise"],
        "coder-30b": ["concise"],
    }

    for pool_name, pool_models in pool_configs:
        for knn_k in [10, 20, 28, 50]:
            for budget_235b in ["concise", "budget_200", "budget_400"]:
                # Train KNN regressor per model (at their fixed budget)
                model_preds = {}
                for mn in pool_models:
                    if mn not in models_data:
                        continue
                    if mn == "235b":
                        budget = budget_235b
                    else:
                        budget = "concise"  # API or cheap models use concise

                    if budget not in models_data[mn]:
                        continue

                    y = models_data[mn][budget]["accuracy"][train_idx]
                    valid = ~np.isnan(y)
                    if valid.sum() < 3:
                        continue
                    knn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum()-1),
                                               metric="cosine", weights="distance")
                    knn.fit(X_train[valid], y[valid])
                    model_preds[(mn, budget)] = knn.predict(embeddings)

                if not model_preds:
                    continue

                # Route: pick model with highest predicted accuracy
                routes = []
                for i in range(n):
                    best_pred = -1
                    best_choice = ("235b", "concise")
                    for (mn, budget), pred in model_preds.items():
                        if pred[i] > best_pred:
                            best_pred = pred[i]
                            best_choice = (mn, budget)
                    routes.append(best_choice)

                acc, cost = evaluate_routing(routes, models_data, sweep_costs, global_indices, n)
                if acc > 0.715 and cost < 0.06:
                    model_dist = defaultdict(int)
                    for mn, _ in routes:
                        model_dist[mn] += 1
                    dist = " ".join(f"{mn}={c/n*100:.0f}%" for mn, c in sorted(model_dist.items(), key=lambda x: -x[1]))
                    print_result(f"Hybrid {pool_name} 235b@{budget_235b} k={knn_k} [{dist}]", acc, cost)

    # =========================================================================
    # Strategy 7: Aggressive — mostly 235b, upgrade hard queries
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("[Strategy 7] 235b default + upgrade hard queries to higher budget", flush=True)
    print("=" * 100, flush=True)

    # Use KNN to predict P(correct|235b,concise)
    # For queries with low P(correct), upgrade budget
    for knn_k in [10, 20, 28, 50, 80, 100]:
        y_train = models_data["235b"]["concise"]["accuracy"][train_idx]
        valid = ~np.isnan(y_train)
        knn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum()-1),
                                   metric="cosine", weights="distance")
        knn.fit(X_train[valid], y_train[valid])
        pred_concise = knn.predict(embeddings)

        # Also predict P(correct|235b,budget_X)
        for upgrade_budget in ["budget_200", "budget_400", "budget_800"]:
            y_up = models_data["235b"][upgrade_budget]["accuracy"][train_idx]
            valid_up = ~np.isnan(y_up)
            knn_up = KNeighborsRegressor(n_neighbors=min(knn_k, valid_up.sum()-1),
                                          metric="cosine", weights="distance")
            knn_up.fit(X_train[valid_up], y_up[valid_up])
            pred_up = knn_up.predict(embeddings)

            # Upgrade if: pred_up - pred_concise > threshold AND pred_concise < low_threshold
            for low_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                for gain_thresh in [0.0, 0.01, 0.02, 0.05, 0.1]:
                    routes = []
                    for i in range(n):
                        if pred_concise[i] < low_thresh and pred_up[i] - pred_concise[i] > gain_thresh:
                            routes.append(("235b", upgrade_budget))
                        else:
                            routes.append(("235b", "concise"))

                    acc, cost = evaluate_routing(routes, models_data, sweep_costs, global_indices, n)
                    if acc > 0.72 and cost < 0.05:
                        pct_up = sum(1 for r in routes if r[1] != "concise") / n * 100
                        print_result(
                            f"235b k={knn_k} low<{low_thresh} gain>{gain_thresh} up={upgrade_budget} ({pct_up:.0f}% up)",
                            acc, cost)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 100, flush=True)
    print("EXPERIMENT V2 COMPLETE", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
