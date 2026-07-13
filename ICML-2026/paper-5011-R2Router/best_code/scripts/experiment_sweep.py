#!/usr/bin/env python3
"""
Comprehensive experiment sweep for R2-Router optimization.

Goals: Acc > 72%, Cost < $0.05/1kq (using real sweep costs)
Constraint: Train on sub_10 only, evaluate on full 8400.
Constraint: vLLM models must use >= 2 budgets, API models use concise only.

Sweeps: model pools, lambda, KNN k, budget subsets.
"""

import os
import sys
import json
import math
import pickle
import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.neighbors import KNeighborsRegressor

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


def load_sweep_costs():
    """Load real per-query costs from sweep files."""
    sweep_costs = {}
    for mn, cfg in MODELS.items():
        sweep_dir = cfg["sweep_dir"]
        if not os.path.isdir(sweep_dir):
            continue
        sweep_costs[mn] = {}
        for fname in os.listdir(sweep_dir):
            if not fname.endswith(".json"):
                continue
            budget = fname.replace(".json", "")
            path = os.path.join(sweep_dir, fname)
            with open(path) as f:
                data = json.load(f)
            cost_map = {}
            for e in data:
                if not e.get("for_optimality"):
                    cost_map[e["global index"]] = e.get("cost", 0)
            sweep_costs[mn][budget] = cost_map
    return sweep_costs


def main():
    print("=" * 80, flush=True)
    print("R2-Router Optimization Experiment Sweep", flush=True)
    print("=" * 80, flush=True)

    # 1. Load data
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
    print(f"  {n} total queries, {len(train_idx)} train (sub_10)", flush=True)

    with open(MODEL_COST_PATH) as f:
        cost_data = json.load(f)
    prices = {}
    for mn, cfg in MODELS.items():
        prices[mn] = cost_data.get(cfg["cost_key"], {}).get("output_token_price_per_million", 0)

    # 2. Load real sweep costs
    print("\n[2] Loading sweep costs...", flush=True)
    sweep_costs = load_sweep_costs()
    for mn in sorted(sweep_costs):
        budgets = sorted(sweep_costs[mn].keys())
        print(f"  {mn:<15} {len(budgets)} budgets: {budgets}", flush=True)

    # 3. Compute per-model per-budget accuracy (oracle data)
    print("\n[3] Per-model per-budget accuracy on full set...", flush=True)
    model_budget_acc = {}
    model_budget_cost = {}
    for mn in sorted(models_data.keys()):
        model_budget_acc[mn] = {}
        model_budget_cost[mn] = {}
        for budget in sorted(models_data[mn].keys()):
            acc_arr = models_data[mn][budget]["accuracy"]
            valid = ~np.isnan(acc_arr)
            if valid.sum() == 0:
                continue
            mean_acc = float(np.nanmean(acc_arr))
            # Real sweep cost
            if mn in sweep_costs and budget in sweep_costs[mn]:
                costs = sweep_costs[mn][budget]
                total_cost = sum(costs.get(gi, 0) for gi in global_indices)
                mean_cost_1kq = total_cost / n * 1000
            else:
                # Fallback: output_tokens * price
                tok_arr = models_data[mn][budget].get("output_tokens", np.zeros(n))
                mean_cost_1kq = float(np.nanmean(tok_arr)) * prices[mn] / 1e6 * 1000
            model_budget_acc[mn][budget] = mean_acc
            model_budget_cost[mn][budget] = mean_cost_1kq

        # Print best budget
        if model_budget_acc[mn]:
            best_b = max(model_budget_acc[mn], key=model_budget_acc[mn].get)
            print(f"  {mn:<15} best={best_b} acc={model_budget_acc[mn][best_b]*100:.1f}% "
                  f"cost=${model_budget_cost[mn][best_b]:.4f}", flush=True)

    # 4. Oracle analysis: for each query, pick best (model, budget)
    print("\n[4] Oracle analysis (best possible per-query routing)...", flush=True)

    # Build per-query accuracy and cost arrays
    all_options = []  # list of (model, budget)
    for mn in sorted(models_data.keys()):
        for budget in sorted(models_data[mn].keys()):
            acc_arr = models_data[mn][budget]["accuracy"]
            if np.isnan(acc_arr).all():
                continue
            all_options.append((mn, budget))

    print(f"  Total (model, budget) options: {len(all_options)}", flush=True)

    # Oracle with different model pools
    model_pools = {
        "all_12": list(models_data.keys()),
        "235b_only": ["235b"],
        "235b+flash": ["235b", "gemini-flash"],
        "235b+m3b": ["235b", "ministral-3b"],
        "235b+m3b+flash": ["235b", "ministral-3b", "gemini-flash"],
        "235b+m8b+flash": ["235b", "ministral-8b", "gemini-flash"],
        "235b+30b+flash": ["235b", "30b", "gemini-flash"],
        "235b+30b+m3b": ["235b", "30b", "ministral-3b"],
        "235b+30b+m3b+flash": ["235b", "30b", "ministral-3b", "gemini-flash"],
        "235b+coder-next+flash": ["235b", "coder-next", "gemini-flash"],
        "235b+coder-next+m3b": ["235b", "coder-next", "ministral-3b"],
        "235b+coder-30b+flash": ["235b", "coder-30b", "gemini-flash"],
        "235b+coder-30b+m3b": ["235b", "coder-30b", "ministral-3b"],
        "235b+80b+flash": ["235b", "80b", "gemini-flash"],
        "235b+80b+haiku": ["235b", "80b", "haiku"],
        "235b+flash+haiku": ["235b", "gemini-flash", "haiku"],
        "baseline_4": ["235b", "80b", "gemini-flash", "haiku"],
        "235b+coder-next+m3b+flash": ["235b", "coder-next", "ministral-3b", "gemini-flash"],
        "235b+30b+coder-30b+m3b": ["235b", "30b", "coder-30b", "ministral-3b"],
    }

    budget_sets = {
        "all": None,  # no exclusion
        "no_unlim_1500": {"budget_unlimited", "budget_1500"},
        "concise+800": {"budget_10", "budget_20", "budget_40", "budget_80", "budget_150", "budget_200", "budget_400", "budget_unlimited", "budget_1500"},
        "concise+400+800": {"budget_10", "budget_20", "budget_40", "budget_80", "budget_150", "budget_200", "budget_unlimited", "budget_1500"},
        "concise+200+400+800": {"budget_10", "budget_20", "budget_40", "budget_80", "budget_150", "budget_unlimited", "budget_1500"},
        "top4_budgets": {"budget_10", "budget_20", "budget_40", "budget_80", "budget_150", "budget_unlimited", "budget_1500"},  # concise+200+400+800
    }

    print(f"\n  {'Pool':<35} {'Budgets':<25} {'Oracle Acc':>10} {'Oracle Cost':>12} {'Arena(0.1)':>10}", flush=True)
    print(f"  {'-'*35} {'-'*25} {'-'*10} {'-'*12} {'-'*10}", flush=True)

    for pool_name, pool_models in model_pools.items():
        for bset_name, excluded in budget_sets.items():
            if excluded is None:
                excluded = set()

            # Oracle: per-query, pick (model, budget) with highest accuracy,
            # breaking ties by lowest cost
            oracle_acc = np.zeros(n)
            oracle_cost = np.zeros(n)

            for i in range(n):
                gi = global_indices[i]
                best_a = -1
                best_c = float('inf')
                for mn in pool_models:
                    if mn not in models_data:
                        continue
                    for budget in models_data[mn]:
                        if budget in excluded:
                            continue
                        a = float(models_data[mn][budget]["accuracy"][i])
                        if np.isnan(a):
                            continue
                        # Real cost
                        if mn in sweep_costs and budget in sweep_costs[mn]:
                            c = sweep_costs[mn][budget].get(gi, 0)
                        else:
                            tok = float(models_data[mn][budget].get("output_tokens", np.zeros(1))[0] if isinstance(models_data[mn][budget].get("output_tokens"), np.ndarray) and len(models_data[mn][budget]["output_tokens"]) > i else 0)
                            c = tok * prices.get(mn, 0) / 1e6

                        if a > best_a or (a == best_a and c < best_c):
                            best_a = a
                            best_c = c

                oracle_acc[i] = best_a if best_a >= 0 else 0
                oracle_cost[i] = best_c

            acc = float(oracle_acc.mean())
            cost_1kq = float(oracle_cost.sum()) / n * 1000
            ascore = arena_score(acc, cost_1kq, 0.1)
            print(f"  {pool_name:<35} {bset_name:<25} {acc*100:>9.2f}% ${cost_1kq:>10.4f} {ascore:>9.2f}", flush=True)

    # 5. KNN routing sweep
    print("\n\n[5] KNN Routing Sweep (sub_10 train, full eval with real sweep costs)...", flush=True)
    print(f"  {'Pool':<30} {'Budgets':<20} {'λ':>6} {'K':>4} {'Acc':>8} {'Cost/1kq':>10} {'Arena(0.1)':>10}", flush=True)
    print(f"  {'-'*30} {'-'*20} {'-'*6} {'-'*4} {'-'*8} {'-'*10} {'-'*10}", flush=True)

    # Promising model pools to test
    test_pools = {
        "235b_only": ["235b"],
        "235b+m3b+flash": ["235b", "ministral-3b", "gemini-flash"],
        "235b+flash": ["235b", "gemini-flash"],
        "235b+30b+flash": ["235b", "30b", "gemini-flash"],
        "235b+coder-next+flash": ["235b", "coder-next", "gemini-flash"],
        "235b+coder-30b+flash": ["235b", "coder-30b", "gemini-flash"],
        "235b+coder-next+m3b": ["235b", "coder-next", "ministral-3b"],
        "235b+coder-30b+m3b": ["235b", "coder-30b", "ministral-3b"],
        "235b+30b+m3b": ["235b", "30b", "ministral-3b"],
        "235b+m3b": ["235b", "ministral-3b"],
        "baseline_4": ["235b", "80b", "gemini-flash", "haiku"],
        "235b+coder-next+m3b+flash": ["235b", "coder-next", "ministral-3b", "gemini-flash"],
        "235b+30b+coder-30b+m3b": ["235b", "30b", "coder-30b", "ministral-3b"],
    }

    test_budget_sets = {
        "no_unlim_1500": {"budget_unlimited", "budget_1500"},
        "top4_budgets": {"budget_10", "budget_20", "budget_40", "budget_80", "budget_150", "budget_unlimited", "budget_1500"},
    }

    lambda_vals = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    k_vals = [5, 10, 20, 28, 50]

    best_result = {"arena": 0, "acc": 0, "cost": 999, "config": ""}
    results = []

    for pool_name, pool_models in test_pools.items():
        for bset_name, excluded_budgets in test_budget_sets.items():
            # Pre-train KNN for all k values
            for knn_k in k_vals:
                # Train KNN predictors
                quality_knn = {}
                token_knn = {}

                for mn in pool_models:
                    if mn not in models_data:
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
                        k_use = min(knn_k, valid.sum() - 1)
                        if k_use < 1:
                            continue
                        knn = KNeighborsRegressor(n_neighbors=k_use,
                                                  metric="cosine", weights="distance")
                        knn.fit(X_train[valid], y_train[valid])
                        quality_knn[mn][budget] = knn

                    # Token predictor
                    token_budget = "concise" if "concise" in budgets_data else next(iter(budgets_data), None)
                    if token_budget and "output_tokens" in budgets_data.get(token_budget, {}):
                        y_tok = budgets_data[token_budget]["output_tokens"][train_idx]
                        valid = ~np.isnan(y_tok)
                        if valid.sum() >= 3:
                            k_use = min(knn_k, valid.sum() - 1)
                            if k_use >= 1:
                                tknn = KNeighborsRegressor(n_neighbors=k_use,
                                                           metric="cosine", weights="distance")
                                tknn.fit(X_train[valid], y_tok[valid])
                                token_knn[mn] = tknn

                # Pre-compute predictions
                pred_quality = {}
                pred_tokens = {}
                for mn in pool_models:
                    if mn not in quality_knn:
                        continue
                    pred_quality[mn] = {}
                    for budget, knn in quality_knn[mn].items():
                        pred_quality[mn][budget] = knn.predict(embeddings)

                    if mn in token_knn:
                        pred_tokens[mn] = np.maximum(1.0, token_knn[mn].predict(embeddings))
                    else:
                        pred_tokens[mn] = np.full(n, 50.0)

                # Sweep lambda
                for lam in lambda_vals:
                    best_risk = np.full(n, -np.inf)
                    best_model = [None] * n
                    best_budget = [None] * n

                    for mn in pool_models:
                        if mn not in pred_quality:
                            continue
                        price = prices[mn]
                        tok = pred_tokens[mn]

                        for budget, q in pred_quality[mn].items():
                            risk = (1 - lam) * q - lam * tok * price / 1e6
                            better = risk > best_risk
                            for j in np.where(better)[0]:
                                best_risk[j] = risk[j]
                                best_model[j] = mn
                                best_budget[j] = budget

                    # Evaluate with REAL sweep costs
                    total_acc = 0.0
                    total_cost = 0.0
                    for i in range(n):
                        mn = best_model[i]
                        budget = best_budget[i]
                        gi = global_indices[i]
                        if mn and mn in models_data and budget in models_data[mn]:
                            total_acc += float(models_data[mn][budget]["accuracy"][i])
                            if mn in sweep_costs and budget in sweep_costs[mn]:
                                total_cost += sweep_costs[mn][budget].get(gi, 0)
                            else:
                                tok = float(models_data[mn][budget]["output_tokens"][i])
                                total_cost += tok * prices[mn] / 1e6

                    accuracy = total_acc / n
                    cost_1kq = total_cost / n * 1000
                    ascore = arena_score(accuracy, cost_1kq, 0.1)

                    results.append({
                        "pool": pool_name, "budgets": bset_name, "lambda": lam,
                        "k": knn_k, "acc": accuracy, "cost": cost_1kq, "arena": ascore
                    })

                    # Check if meets target
                    meets_target = accuracy > 0.72 and cost_1kq < 0.05
                    marker = " <<<TARGET>>>" if meets_target else ""

                    if ascore > best_result["arena"] or meets_target:
                        if ascore > best_result["arena"]:
                            best_result = {"arena": ascore, "acc": accuracy,
                                          "cost": cost_1kq, "config": f"{pool_name}/{bset_name}/λ={lam}/k={knn_k}"}

                        print(f"  {pool_name:<30} {bset_name:<20} {lam:>6.3f} {knn_k:>4} "
                              f"{accuracy*100:>7.2f}% ${cost_1kq:>9.4f} {ascore:>9.2f}{marker}", flush=True)

    # 6. Print top results
    print(f"\n\n{'=' * 80}", flush=True)
    print("TOP 20 RESULTS (sorted by Arena score):", flush=True)
    print(f"{'=' * 80}", flush=True)
    results.sort(key=lambda x: -x["arena"])
    print(f"  {'#':>3} {'Pool':<30} {'Budgets':<20} {'λ':>6} {'K':>4} {'Acc':>8} {'Cost/1kq':>10} {'Arena':>8} {'Target?':>8}", flush=True)
    for i, r in enumerate(results[:20]):
        meets = "YES" if r["acc"] > 0.72 and r["cost"] < 0.05 else "no"
        print(f"  {i+1:>3} {r['pool']:<30} {r['budgets']:<20} {r['lambda']:>6.3f} {r['k']:>4} "
              f"{r['acc']*100:>7.2f}% ${r['cost']:>9.4f} {r['arena']:>7.2f} {meets:>8}", flush=True)

    # 7. Print results meeting target
    print(f"\n\n{'=' * 80}", flush=True)
    print("RESULTS MEETING TARGET (Acc > 72%, Cost < $0.05):", flush=True)
    print(f"{'=' * 80}", flush=True)
    target_results = [r for r in results if r["acc"] > 0.72 and r["cost"] < 0.05]
    if target_results:
        target_results.sort(key=lambda x: -x["arena"])
        for i, r in enumerate(target_results[:30]):
            print(f"  {i+1:>3} {r['pool']:<30} {r['budgets']:<20} {r['lambda']:>6.3f} {r['k']:>4} "
                  f"{r['acc']*100:>7.2f}% ${r['cost']:>9.4f} {r['arena']:>7.2f}", flush=True)
    else:
        print("  No configurations met both targets.", flush=True)
        # Print closest
        print("\n  Closest to target (Acc > 71.5%, Cost < $0.06):", flush=True)
        close = [r for r in results if r["acc"] > 0.715 and r["cost"] < 0.06]
        close.sort(key=lambda x: -x["arena"])
        for i, r in enumerate(close[:20]):
            print(f"  {i+1:>3} {r['pool']:<30} {r['budgets']:<20} {r['lambda']:>6.3f} {r['k']:>4} "
                  f"{r['acc']*100:>7.2f}% ${r['cost']:>9.4f} {r['arena']:>7.2f}", flush=True)

    print(f"\n  Best overall: {best_result['config']}")
    print(f"    Acc={best_result['acc']*100:.2f}%, Cost=${best_result['cost']:.4f}, Arena={best_result['arena']:.2f}")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
