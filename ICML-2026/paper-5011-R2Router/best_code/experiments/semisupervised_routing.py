#!/usr/bin/env python3
"""
Semi-supervised routing experiments: Label Propagation, Cluster-Then-Route,
and Difficulty-Gated Neighbor Voting.

Key insight: We have embeddings for ALL 8400 queries but labels for only 809 (sub_10).
Semi-supervised methods use the unlabeled embeddings for structural information
while only using sub_10 labels for supervision.

Usage:
    .venv/bin/python experiments/semisupervised_routing.py
"""

import os
import sys
import json
import math
import pickle
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.semi_supervised import LabelSpreading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
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


def load_sweep_costs(allowed_models, allowed_budgets, n):
    """Load actual costs from sweep files."""
    print("  Loading sweep costs...", flush=True)
    costs = {}
    for mn in allowed_models:
        costs[mn] = {}
        sweep_dir = MODELS[mn]["sweep_dir"]
        for budget in allowed_budgets:
            path = os.path.join(sweep_dir, f"{budget}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            cost_map = {}
            for e in data:
                if not e.get("for_optimality"):
                    cost_map[e["global index"]] = e.get("cost", 0)
            costs[mn][budget] = cost_map
    print(f"  Loaded sweep costs for {len(costs)} models", flush=True)
    return costs


def evaluate_routes(routes, models_data, global_indices, prices, sweep_costs, n, label=""):
    """Evaluate routing decisions against ground truth."""
    total_acc = 0.0
    total_cost = 0.0
    model_counts = defaultdict(int)
    budget_counts = defaultdict(int)

    for i, (mn, budget) in enumerate(routes):
        if mn is None:
            continue
        model_counts[mn] += 1
        budget_counts[budget] += 1
        if mn in models_data and budget in models_data[mn]:
            total_acc += float(models_data[mn][budget]["accuracy"][i])
            # Use sweep costs if available, else estimate
            gi = global_indices[i]
            if mn in sweep_costs and budget in sweep_costs[mn] and gi in sweep_costs[mn][budget]:
                total_cost += sweep_costs[mn][budget][gi]
            else:
                tokens = float(models_data[mn][budget]["output_tokens"][i])
                total_cost += tokens * prices[mn] / 1e6

    accuracy = total_acc / n
    cost_1kq = total_cost / n * 1000
    arena = arena_score(accuracy, cost_1kq)

    print(f"  [{label}] Acc={accuracy*100:.2f}% Cost=${cost_1kq:.4f}/1kq Arena={arena:.2f}")
    top_models = sorted(model_counts.items(), key=lambda x: -x[1])[:4]
    model_str = ", ".join(f"{m}={c/n*100:.1f}%" for m, c in top_models)
    print(f"    Models: {model_str}")
    top_budgets = sorted(budget_counts.items(), key=lambda x: -x[1])[:4]
    budget_str = ", ".join(f"{b}={c/n*100:.1f}%" for b, c in top_budgets)
    print(f"    Budgets: {budget_str}")
    return accuracy, cost_1kq, arena


def route_knn_baseline(embeddings, train_idx, models_data, prices, allowed_models,
                       allowed_budgets, lam, k, n):
    """Standard KNN routing (baseline)."""
    X_train = embeddings[train_idx]
    best_risk = np.full(n, -np.inf)
    best_model = [None] * n
    best_budget = [None] * n

    for mn in allowed_models:
        if mn not in models_data:
            continue
        price = prices[mn]

        # Token predictor (from concise)
        if "concise" in models_data[mn] and models_data[mn]["concise"]["output_tokens"][train_idx].sum() > 0:
            y_tok = models_data[mn]["concise"]["output_tokens"][train_idx]
            valid = ~np.isnan(y_tok)
            if valid.sum() >= 3:
                tknn = KNeighborsRegressor(n_neighbors=min(k, valid.sum()-1), metric="cosine", weights="distance")
                tknn.fit(X_train[valid], y_tok[valid])
                tok = np.maximum(1.0, tknn.predict(embeddings))
            else:
                tok = np.full(n, 50.0)
        else:
            tok = np.full(n, 50.0)

        for budget in allowed_budgets:
            if budget not in models_data[mn]:
                continue
            y_train = models_data[mn][budget]["accuracy"][train_idx]
            valid = ~np.isnan(y_train)
            if valid.sum() < 3:
                continue
            knn = KNeighborsRegressor(n_neighbors=min(k, valid.sum()-1), metric="cosine", weights="distance")
            knn.fit(X_train[valid], y_train[valid])
            q = knn.predict(embeddings)
            risk = (1 - lam) * q - lam * tok * price / 1e6

            better = risk > best_risk
            for j in np.where(better)[0]:
                best_risk[j] = risk[j]
                best_model[j] = mn
                best_budget[j] = budget

    return list(zip(best_model, best_budget))


def route_label_propagation(embeddings, train_idx, models_data, prices, allowed_models,
                            allowed_budgets, lam, n_neighbors=30, alpha=0.2, n=None):
    """Semi-supervised Label Propagation routing.

    Uses LabelSpreading to propagate accuracy labels from sub_10 to all queries.
    The graph structure is built from ALL 8400 embeddings.
    """
    best_risk = np.full(n, -np.inf)
    best_model = [None] * n
    best_budget = [None] * n

    # Precompute token estimates using KNN (simpler, works OK)
    X_train = embeddings[train_idx]
    token_pred = {}
    for mn in allowed_models:
        if mn not in models_data:
            continue
        if "concise" in models_data[mn]:
            y_tok = models_data[mn]["concise"]["output_tokens"][train_idx]
            valid = ~np.isnan(y_tok)
            if valid.sum() >= 3:
                tknn = KNeighborsRegressor(n_neighbors=min(28, valid.sum()-1), metric="cosine", weights="distance")
                tknn.fit(X_train[valid], y_tok[valid])
                token_pred[mn] = np.maximum(1.0, tknn.predict(embeddings))

    for mn in allowed_models:
        if mn not in models_data:
            continue
        price = prices[mn]
        tok = token_pred.get(mn, np.full(n, 50.0))

        for budget in allowed_budgets:
            if budget not in models_data[mn]:
                continue
            y_all = models_data[mn][budget]["accuracy"]

            # Binarize for label propagation (93% are already binary)
            labels = np.full(n, -1)  # -1 = unlabeled
            labels[train_idx] = (y_all[train_idx] >= 0.5).astype(int)

            # Check if we have enough labels of each class
            n_pos = (labels[train_idx] == 1).sum()
            n_neg = (labels[train_idx] == 0).sum()
            if n_pos < 2 or n_neg < 2:
                # Degenerate case - use mean
                q = np.full(n, y_all[train_idx].mean())
            else:
                try:
                    ls = LabelSpreading(
                        kernel='knn',
                        n_neighbors=min(n_neighbors, len(train_idx) - 1),
                        alpha=alpha,
                        max_iter=100,
                        n_jobs=-1
                    )
                    ls.fit(embeddings, labels)
                    q = ls.label_distributions_[:, 1]  # P(correct)
                except Exception as e:
                    print(f"    LP failed for {mn}/{budget}: {e}")
                    q = np.full(n, y_all[train_idx].mean())

            risk = (1 - lam) * q - lam * tok * price / 1e6

            better = risk > best_risk
            for j in np.where(better)[0]:
                best_risk[j] = risk[j]
                best_model[j] = mn
                best_budget[j] = budget

    return list(zip(best_model, best_budget))


def route_cluster(embeddings, train_idx, models_data, prices, allowed_models,
                  allowed_budgets, lam, n_clusters=80, min_samples=3, n=None):
    """Cluster-Then-Route: cluster all 8400 embeddings, use cluster means for routing."""
    emb_norm = normalize(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(emb_norm)

    # Global means as fallback
    global_means = {}
    global_token_means = {}
    for mn in allowed_models:
        if mn not in models_data:
            continue
        global_means[mn] = {}
        for budget in allowed_budgets:
            if budget not in models_data[mn]:
                continue
            global_means[mn][budget] = float(models_data[mn][budget]["accuracy"][train_idx].mean())
        if "concise" in models_data[mn]:
            global_token_means[mn] = float(models_data[mn]["concise"]["output_tokens"][train_idx].mean())

    # Per-cluster means
    cluster_quality = {}  # {cluster: {model: {budget: mean_acc}}}
    cluster_tokens = {}   # {cluster: {model: mean_tokens}}
    for c in range(n_clusters):
        cluster_mask = cluster_labels == c
        cluster_train = np.intersect1d(np.where(cluster_mask)[0], train_idx)
        cluster_quality[c] = {}
        cluster_tokens[c] = {}

        for mn in allowed_models:
            if mn not in models_data:
                continue
            cluster_quality[c][mn] = {}
            for budget in allowed_budgets:
                if budget not in models_data[mn]:
                    continue
                if len(cluster_train) >= min_samples:
                    y = models_data[mn][budget]["accuracy"][cluster_train]
                    cluster_quality[c][mn][budget] = float(y.mean())
                else:
                    cluster_quality[c][mn][budget] = global_means.get(mn, {}).get(budget, 0.5)

            if "concise" in models_data[mn] and len(cluster_train) >= min_samples:
                cluster_tokens[c][mn] = float(models_data[mn]["concise"]["output_tokens"][cluster_train].mean())
            else:
                cluster_tokens[c][mn] = global_token_means.get(mn, 50.0)

    # Route
    routes = [None] * n
    for i in range(n):
        c = cluster_labels[i]
        best_risk = -np.inf
        best_choice = ("235b", "concise")

        for mn in allowed_models:
            if mn not in models_data:
                continue
            price = prices[mn]
            tok = cluster_tokens[c].get(mn, 50.0)

            for budget in allowed_budgets:
                if budget not in models_data[mn]:
                    continue
                q = cluster_quality[c][mn].get(budget, 0.5)
                risk = (1 - lam) * q - lam * tok * price / 1e6
                if risk > best_risk:
                    best_risk = risk
                    best_choice = (mn, budget)

        routes[i] = best_choice

    return routes


def route_soft_cluster(embeddings, train_idx, models_data, prices, allowed_models,
                       allowed_budgets, lam, n_clusters=80, temperature=5.0, min_samples=3, n=None):
    """Soft Cluster-Then-Route: weight contributions from multiple nearby clusters."""
    emb_norm = normalize(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(emb_norm)
    centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    # Global means as fallback
    global_means = {}
    global_token_means = {}
    for mn in allowed_models:
        if mn not in models_data:
            continue
        global_means[mn] = {}
        for budget in allowed_budgets:
            if budget not in models_data[mn]:
                continue
            global_means[mn][budget] = float(models_data[mn][budget]["accuracy"][train_idx].mean())
        if "concise" in models_data[mn]:
            global_token_means[mn] = float(models_data[mn]["concise"]["output_tokens"][train_idx].mean())

    # Per-cluster means (same as hard cluster)
    cluster_quality = {}
    cluster_tokens = {}
    cluster_train_counts = {}
    for c in range(n_clusters):
        cluster_mask = cluster_labels == c
        cluster_train = np.intersect1d(np.where(cluster_mask)[0], train_idx)
        cluster_train_counts[c] = len(cluster_train)
        cluster_quality[c] = {}
        cluster_tokens[c] = {}

        for mn in allowed_models:
            if mn not in models_data:
                continue
            cluster_quality[c][mn] = {}
            for budget in allowed_budgets:
                if budget not in models_data[mn]:
                    continue
                if len(cluster_train) >= min_samples:
                    cluster_quality[c][mn][budget] = float(models_data[mn][budget]["accuracy"][cluster_train].mean())
                else:
                    cluster_quality[c][mn][budget] = global_means.get(mn, {}).get(budget, 0.5)
            if "concise" in models_data[mn] and len(cluster_train) >= min_samples:
                cluster_tokens[c][mn] = float(models_data[mn]["concise"]["output_tokens"][cluster_train].mean())
            else:
                cluster_tokens[c][mn] = global_token_means.get(mn, 50.0)

    # Compute distances to all cluster centers for each query
    from sklearn.metrics.pairwise import cosine_distances
    dists = cosine_distances(emb_norm, centers)  # (8400, n_clusters)

    routes = [None] * n
    for i in range(n):
        # Soft weights (inverse distance with temperature)
        d = dists[i]
        weights = np.exp(-d * temperature)
        weights /= weights.sum()

        best_risk = -np.inf
        best_choice = ("235b", "concise")

        for mn in allowed_models:
            if mn not in models_data:
                continue
            price = prices[mn]

            # Weighted token estimate
            tok = sum(weights[c] * cluster_tokens[c].get(mn, 50.0) for c in range(n_clusters))

            for budget in allowed_budgets:
                if budget not in models_data[mn]:
                    continue
                # Weighted quality estimate
                q = sum(weights[c] * cluster_quality[c][mn].get(budget, 0.5) for c in range(n_clusters))
                risk = (1 - lam) * q - lam * tok * price / 1e6
                if risk > best_risk:
                    best_risk = risk
                    best_choice = (mn, budget)

        routes[i] = best_choice

    return routes


def route_neighbor_voting(embeddings, train_idx, models_data, prices, allowed_models,
                          allowed_budgets, lam, k=30, n=None):
    """Difficulty-gated neighbor voting.

    For each query, find K nearest sub_10 neighbors.
    For each neighbor, find which (model, budget) is optimal.
    Vote weighted by distance.
    """
    X_train = embeddings[train_idx]
    nn = NearestNeighbors(n_neighbors=min(k, len(train_idx) - 1), metric='cosine')
    nn.fit(X_train)
    distances, indices = nn.kneighbors(embeddings)

    # Precompute: for each training query, what's the best (model, budget)?
    # Considering both accuracy and cost
    train_best = {}  # {train_global_idx: (model, budget, accuracy)}
    for ti_local, ti_global in enumerate(train_idx):
        best_risk = -np.inf
        best_choice = ("235b", "concise")
        best_acc = 0.0
        for mn in allowed_models:
            if mn not in models_data:
                continue
            price = prices[mn]
            for budget in allowed_budgets:
                if budget not in models_data[mn]:
                    continue
                acc = float(models_data[mn][budget]["accuracy"][ti_global])
                tok = float(models_data[mn]["concise"]["output_tokens"][ti_global]) if "concise" in models_data[mn] else 50.0
                risk = (1 - lam) * acc - lam * tok * price / 1e6
                if risk > best_risk:
                    best_risk = risk
                    best_choice = (mn, budget)
                    best_acc = acc
        train_best[ti_local] = (best_choice, best_acc)

    routes = [None] * n
    for i in range(n):
        neighbor_local_indices = indices[i]
        neighbor_dists = distances[i]

        # Inverse distance weights
        weights = 1.0 / (neighbor_dists + 1e-8)
        weights /= weights.sum()

        # Vote: accumulate weighted votes for each (model, budget)
        choice_scores = defaultdict(float)
        for j, (ni, w) in enumerate(zip(neighbor_local_indices, weights)):
            choice, acc = train_best[ni]
            choice_scores[choice] += w

        # Pick best
        routes[i] = max(choice_scores, key=choice_scores.get)

    return routes


def route_neighbor_voting_per_model(embeddings, train_idx, models_data, prices, allowed_models,
                                     allowed_budgets, lam, k=30, n=None):
    """Per-model neighbor voting: for each model/budget, estimate accuracy from neighbors,
    then route using risk formula. Different from pure voting — uses risk formula."""
    X_train = embeddings[train_idx]
    nn = NearestNeighbors(n_neighbors=min(k, len(train_idx) - 1), metric='cosine')
    nn.fit(X_train)
    distances, indices = nn.kneighbors(embeddings)

    best_risk = np.full(n, -np.inf)
    best_model = [None] * n
    best_budget = [None] * n

    for mn in allowed_models:
        if mn not in models_data:
            continue
        price = prices[mn]

        for budget in allowed_budgets:
            if budget not in models_data[mn]:
                continue
            y_all = models_data[mn][budget]["accuracy"]

            # For each query, compute weighted mean accuracy from neighbors
            q = np.zeros(n)
            tok = np.full(n, 50.0)
            for i in range(n):
                neighbor_global = train_idx[indices[i]]
                w = 1.0 / (distances[i] + 1e-8)
                w /= w.sum()
                q[i] = np.dot(w, y_all[neighbor_global])

                if "concise" in models_data[mn]:
                    tok[i] = max(1.0, np.dot(w, models_data[mn]["concise"]["output_tokens"][neighbor_global]))

            risk = (1 - lam) * q - lam * tok * price / 1e6

            better = risk > best_risk
            for j in np.where(better)[0]:
                best_risk[j] = risk[j]
                best_model[j] = mn
                best_budget[j] = budget

    return list(zip(best_model, best_budget))


def route_lp_plus_knn_ensemble(embeddings, train_idx, models_data, prices, allowed_models,
                                allowed_budgets, lam, lp_neighbors=30, lp_alpha=0.2,
                                knn_k=28, blend_alpha=0.5, n=None):
    """Ensemble: blend Label Propagation P(correct) with KNN regression predictions."""
    X_train = embeddings[train_idx]

    best_risk = np.full(n, -np.inf)
    best_model = [None] * n
    best_budget = [None] * n

    # Token pred via KNN
    token_pred = {}
    for mn in allowed_models:
        if mn not in models_data:
            continue
        if "concise" in models_data[mn]:
            y_tok = models_data[mn]["concise"]["output_tokens"][train_idx]
            valid = ~np.isnan(y_tok)
            if valid.sum() >= 3:
                tknn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum()-1), metric="cosine", weights="distance")
                tknn.fit(X_train[valid], y_tok[valid])
                token_pred[mn] = np.maximum(1.0, tknn.predict(embeddings))

    for mn in allowed_models:
        if mn not in models_data:
            continue
        price = prices[mn]
        tok = token_pred.get(mn, np.full(n, 50.0))

        for budget in allowed_budgets:
            if budget not in models_data[mn]:
                continue
            y_all = models_data[mn][budget]["accuracy"]

            # KNN prediction
            y_train = y_all[train_idx]
            valid = ~np.isnan(y_train)
            if valid.sum() < 3:
                continue
            knn = KNeighborsRegressor(n_neighbors=min(knn_k, valid.sum()-1), metric="cosine", weights="distance")
            knn.fit(X_train[valid], y_train[valid])
            q_knn = knn.predict(embeddings)

            # LP prediction
            labels = np.full(n, -1)
            labels[train_idx] = (y_all[train_idx] >= 0.5).astype(int)
            n_pos = (labels[train_idx] == 1).sum()
            n_neg = (labels[train_idx] == 0).sum()
            if n_pos < 2 or n_neg < 2:
                q_lp = np.full(n, y_train.mean())
            else:
                try:
                    ls = LabelSpreading(
                        kernel='knn',
                        n_neighbors=min(lp_neighbors, len(train_idx) - 1),
                        alpha=lp_alpha,
                        max_iter=100,
                        n_jobs=-1
                    )
                    ls.fit(embeddings, labels)
                    q_lp = ls.label_distributions_[:, 1]
                except:
                    q_lp = np.full(n, y_train.mean())

            # Blend
            q = blend_alpha * q_lp + (1 - blend_alpha) * q_knn
            risk = (1 - lam) * q - lam * tok * price / 1e6

            better = risk > best_risk
            for j in np.where(better)[0]:
                best_risk[j] = risk[j]
                best_model[j] = mn
                best_budget[j] = budget

    return list(zip(best_model, best_budget))


def main():
    print("=" * 80)
    print("Semi-Supervised Routing Experiments")
    print("=" * 80)

    # Load data
    print("\nLoading data...", flush=True)
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
    print(f"  {n} queries, {len(train_idx)} train (sub_10)")

    prices = load_prices()

    # Model pools to test
    model_pools = {
        "baseline_4": ["235b", "80b", "gemini-flash", "haiku"],
        "vllm_3": ["235b", "80b", "30b"],
        "vllm_5": ["235b", "80b", "30b", "coder-next", "coder-30b"],
        "wide_6": ["235b", "80b", "30b", "coder-next", "gemini-flash", "haiku"],
    }

    # Budget sets
    budget_sets = {
        "concise_only": ["concise"],
        "top3": ["concise", "budget_200", "budget_400"],
        "top4": ["concise", "budget_200", "budget_400", "budget_800"],
        "all9": ["budget_10", "budget_20", "budget_40", "budget_80", "budget_150",
                 "budget_200", "budget_400", "budget_800", "concise"],
    }

    # Load sweep costs for evaluation
    all_models = set()
    all_budgets = set()
    for models in model_pools.values():
        all_models.update(models)
    for budgets in budget_sets.values():
        all_budgets.update(budgets)
    sweep_costs = load_sweep_costs(all_models, all_budgets, n)

    # ========================================================================
    # Experiment 1: Label Propagation
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Label Propagation (Semi-Supervised)")
    print("=" * 80)

    best_arena = 0
    best_config = ""

    # LP is expensive (~8400-point semi-supervised fit per model/budget pair)
    # Focus on most promising configs
    lp_combos = [
        ("baseline_4", "top4"),
        ("baseline_4", "all9"),
        ("wide_6", "top4"),
    ]
    for pool_name, budget_name in lp_combos:
        allowed_models = model_pools[pool_name]
        allowed_budgets = budget_sets[budget_name]
        for lam in [0.0, 0.995, 0.999]:
            for lp_nn in [20, 40]:
                for lp_alpha in [0.1, 0.2]:
                    routes = route_label_propagation(
                        embeddings, train_idx, models_data, prices,
                        allowed_models, allowed_budgets, lam,
                        n_neighbors=lp_nn, alpha=lp_alpha, n=n
                    )
                    acc, cost, arena = evaluate_routes(
                        routes, models_data, global_indices, prices, sweep_costs, n,
                        label=f"LP {pool_name}/{budget_name} lam={lam} nn={lp_nn} a={lp_alpha}"
                    )
                    if arena > best_arena:
                        best_arena = arena
                        best_config = f"LP {pool_name}/{budget_name} lam={lam} nn={lp_nn} alpha={lp_alpha}"
                        print(f"  *** NEW BEST: {best_config} Arena={best_arena:.2f}")

    print(f"\n  LP Best: {best_config} Arena={best_arena:.2f}")

    # ========================================================================
    # Experiment 2: Cluster-Then-Route
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Cluster-Then-Route")
    print("=" * 80)

    best_arena_cl = 0
    best_config_cl = ""

    for pool_name, allowed_models in [("baseline_4", model_pools["baseline_4"]),
                                       ("vllm_5", model_pools["vllm_5"])]:
        for budget_name, allowed_budgets in [("top4", budget_sets["top4"]),
                                              ("all9", budget_sets["all9"])]:
            for lam in [0.99, 0.995, 0.999]:
                for nc in [40, 60, 80, 120, 160]:
                    for ms in [2, 3, 5]:
                        routes = route_cluster(
                            embeddings, train_idx, models_data, prices,
                            allowed_models, allowed_budgets, lam,
                            n_clusters=nc, min_samples=ms, n=n
                        )
                        acc, cost, arena = evaluate_routes(
                            routes, models_data, global_indices, prices, sweep_costs, n,
                            label=f"Cluster {pool_name}/{budget_name} lam={lam} nc={nc} ms={ms}"
                        )
                        if arena > best_arena_cl:
                            best_arena_cl = arena
                            best_config_cl = f"Cluster {pool_name}/{budget_name} lam={lam} nc={nc} ms={ms}"
                            print(f"  *** NEW BEST: {best_config_cl} Arena={best_arena_cl:.2f}")

    print(f"\n  Cluster Best: {best_config_cl} Arena={best_arena_cl:.2f}")

    # ========================================================================
    # Experiment 2b: Soft Cluster-Then-Route
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2b: Soft Cluster-Then-Route")
    print("=" * 80)

    best_arena_sc = 0
    best_config_sc = ""

    for pool_name, allowed_models in [("baseline_4", model_pools["baseline_4"])]:
        for budget_name, allowed_budgets in [("top4", budget_sets["top4"])]:
            for lam in [0.995, 0.999]:
                for nc in [60, 80, 120]:
                    for temp in [3.0, 5.0, 10.0]:
                        routes = route_soft_cluster(
                            embeddings, train_idx, models_data, prices,
                            allowed_models, allowed_budgets, lam,
                            n_clusters=nc, temperature=temp, min_samples=3, n=n
                        )
                        acc, cost, arena = evaluate_routes(
                            routes, models_data, global_indices, prices, sweep_costs, n,
                            label=f"SoftCluster {pool_name}/{budget_name} lam={lam} nc={nc} temp={temp}"
                        )
                        if arena > best_arena_sc:
                            best_arena_sc = arena
                            best_config_sc = f"SoftCluster {pool_name}/{budget_name} lam={lam} nc={nc} temp={temp}"
                            print(f"  *** NEW BEST: {best_config_sc} Arena={best_arena_sc:.2f}")

    print(f"\n  Soft Cluster Best: {best_config_sc} Arena={best_arena_sc:.2f}")

    # ========================================================================
    # Experiment 3: Neighbor Voting
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Neighbor Voting")
    print("=" * 80)

    best_arena_nv = 0
    best_config_nv = ""

    for pool_name, allowed_models in model_pools.items():
        for budget_name, allowed_budgets in budget_sets.items():
            for lam in [0.0, 0.99, 0.995, 0.999]:
                for k in [10, 20, 30, 50]:
                    routes = route_neighbor_voting(
                        embeddings, train_idx, models_data, prices,
                        allowed_models, allowed_budgets, lam, k=k, n=n
                    )
                    acc, cost, arena = evaluate_routes(
                        routes, models_data, global_indices, prices, sweep_costs, n,
                        label=f"NV {pool_name}/{budget_name} lam={lam} k={k}"
                    )
                    if arena > best_arena_nv:
                        best_arena_nv = arena
                        best_config_nv = f"NV {pool_name}/{budget_name} lam={lam} k={k}"
                        print(f"  *** NEW BEST: {best_config_nv} Arena={best_arena_nv:.2f}")

    print(f"\n  Neighbor Voting Best: {best_config_nv} Arena={best_arena_nv:.2f}")

    # ========================================================================
    # Experiment 3b: Per-model neighbor voting with risk formula
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3b: Per-Model Neighbor Voting (risk formula)")
    print("=" * 80)

    best_arena_pm = 0
    best_config_pm = ""

    for pool_name, allowed_models in [("baseline_4", model_pools["baseline_4"]),
                                       ("vllm_5", model_pools["vllm_5"])]:
        for budget_name, allowed_budgets in [("top4", budget_sets["top4"]),
                                              ("all9", budget_sets["all9"])]:
            for lam in [0.995, 0.999]:
                for k in [20, 30, 50, 80]:
                    routes = route_neighbor_voting_per_model(
                        embeddings, train_idx, models_data, prices,
                        allowed_models, allowed_budgets, lam, k=k, n=n
                    )
                    acc, cost, arena = evaluate_routes(
                        routes, models_data, global_indices, prices, sweep_costs, n,
                        label=f"NVPM {pool_name}/{budget_name} lam={lam} k={k}"
                    )
                    if arena > best_arena_pm:
                        best_arena_pm = arena
                        best_config_pm = f"NVPM {pool_name}/{budget_name} lam={lam} k={k}"
                        print(f"  *** NEW BEST: {best_config_pm} Arena={best_arena_pm:.2f}")

    print(f"\n  Per-Model NV Best: {best_config_pm} Arena={best_arena_pm:.2f}")

    # ========================================================================
    # Experiment 4: LP + KNN Ensemble
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: LP + KNN Ensemble")
    print("=" * 80)

    best_arena_ens = 0
    best_config_ens = ""

    for pool_name, allowed_models in [("baseline_4", model_pools["baseline_4"])]:
        for budget_name, allowed_budgets in [("top4", budget_sets["top4"]),
                                              ("all9", budget_sets["all9"])]:
            for lam in [0.995, 0.999]:
                for blend in [0.3, 0.5, 0.7]:
                    for lp_nn in [30, 50]:
                        routes = route_lp_plus_knn_ensemble(
                            embeddings, train_idx, models_data, prices,
                            allowed_models, allowed_budgets, lam,
                            lp_neighbors=lp_nn, lp_alpha=0.2,
                            knn_k=28, blend_alpha=blend, n=n
                        )
                        acc, cost, arena = evaluate_routes(
                            routes, models_data, global_indices, prices, sweep_costs, n,
                            label=f"Ensemble {pool_name}/{budget_name} lam={lam} blend={blend} lpnn={lp_nn}"
                        )
                        if arena > best_arena_ens:
                            best_arena_ens = arena
                            best_config_ens = f"Ensemble {pool_name}/{budget_name} lam={lam} blend={blend} lpnn={lp_nn}"
                            print(f"  *** NEW BEST: {best_config_ens} Arena={best_arena_ens:.2f}")

    print(f"\n  Ensemble Best: {best_config_ens} Arena={best_arena_ens:.2f}")

    # ========================================================================
    # Baseline comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("BASELINE: Standard KNN (for comparison)")
    print("=" * 80)

    for pool_name, allowed_models in [("baseline_4", model_pools["baseline_4"])]:
        for budget_name, allowed_budgets in [("top4", budget_sets["top4"]),
                                              ("all9", budget_sets["all9"])]:
            for lam in [0.995, 0.999]:
                for k in [28, 60]:
                    routes = route_knn_baseline(
                        embeddings, train_idx, models_data, prices,
                        allowed_models, allowed_budgets, lam, k=k, n=n
                    )
                    evaluate_routes(
                        routes, models_data, global_indices, prices, sweep_costs, n,
                        label=f"KNN {pool_name}/{budget_name} lam={lam} k={k}"
                    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  Label Propagation:     {best_config} Arena={best_arena:.2f}")
    print(f"  Cluster-Then-Route:    {best_config_cl} Arena={best_arena_cl:.2f}")
    print(f"  Soft Cluster:          {best_config_sc} Arena={best_arena_sc:.2f}")
    print(f"  Neighbor Voting:       {best_config_nv} Arena={best_arena_nv:.2f}")
    print(f"  Per-Model NV:          {best_config_pm} Arena={best_arena_pm:.2f}")
    print(f"  LP+KNN Ensemble:       {best_config_ens} Arena={best_arena_ens:.2f}")
    print()


if __name__ == "__main__":
    main()
