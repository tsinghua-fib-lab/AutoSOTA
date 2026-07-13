#!/usr/bin/env python3
"""
Quick semi-supervised routing test: LP, Cluster, Neighbor Voting.
Focuses on the most promising configs only. Runs in ~10 minutes.
"""

import os
import sys
import json
import math
import pickle
import time
import numpy as np
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.semi_supervised import LabelSpreading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from category_config import (
    MODELS, TRAINING_DATA_PATH, MODEL_COST_PATH, ROUTER_DATA_10_PATH,
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


def evaluate_routes(routes, models_data, prices, n, label=""):
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
            tokens = float(models_data[mn][budget]["output_tokens"][i])
            total_cost += tokens * prices[mn] / 1e6

    accuracy = total_acc / n
    cost_1kq = total_cost / n * 1000
    arena = arena_score(accuracy, cost_1kq)
    top_models = sorted(model_counts.items(), key=lambda x: -x[1])[:3]
    model_str = " ".join(f"{m}={c/n*100:.0f}%" for m, c in top_models)
    print(f"  {label:<60} Acc={accuracy*100:.2f}% Cost=${cost_1kq:.4f} Arena={arena:.2f} [{model_str}]", flush=True)
    return accuracy, cost_1kq, arena


def main():
    print("=" * 100, flush=True)
    print("Quick Semi-Supervised Routing Experiments", flush=True)
    print("=" * 100, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    t0 = time.time()
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
    print(f"  {n} queries, {len(train_idx)} train (sub_10), loaded in {time.time()-t0:.1f}s", flush=True)

    prices = load_prices()
    X_train = embeddings[train_idx]

    # Model pools
    pools = {
        "4model": ["235b", "80b", "gemini-flash", "haiku"],
        "5model": ["235b", "80b", "30b", "coder-next", "gemini-flash"],
        "6model": ["235b", "80b", "30b", "coder-next", "gemini-flash", "haiku"],
    }
    budget_sets = {
        "top4": ["concise", "budget_200", "budget_400", "budget_800"],
        "all9": ["budget_10", "budget_20", "budget_40", "budget_80", "budget_150",
                 "budget_200", "budget_400", "budget_800", "concise"],
    }

    # ========================================================================
    # BASELINE: Standard KNN
    # ========================================================================
    print("\n" + "=" * 100, flush=True)
    print("BASELINE: Standard KNN", flush=True)
    print("=" * 100, flush=True)

    best_overall = 0
    best_overall_config = ""

    for pool_name, allowed_models in pools.items():
        for bset_name, allowed_budgets in budget_sets.items():
            for lam in [0.0, 0.995, 0.999]:
                for k in [20, 28, 60]:
                    best_risk = np.full(n, -np.inf)
                    best_model = [None] * n
                    best_budget = [None] * n

                    for mn in allowed_models:
                        if mn not in models_data:
                            continue
                        price = prices[mn]
                        # Token
                        if "concise" in models_data[mn]:
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
                            idx = np.where(better)[0]
                            best_risk[idx] = risk[idx]
                            for j in idx:
                                best_model[j] = mn
                                best_budget[j] = budget

                    routes = list(zip(best_model, best_budget))
                    _, _, arena = evaluate_routes(routes, models_data, prices, n,
                        label=f"KNN {pool_name}/{bset_name} lam={lam} k={k}")
                    if arena > best_overall:
                        best_overall = arena
                        best_overall_config = f"KNN {pool_name}/{bset_name} lam={lam} k={k}"

    print(f"\n  BEST KNN: {best_overall_config} Arena={best_overall:.2f}", flush=True)

    # ========================================================================
    # EXPERIMENT 1: Label Propagation
    # ========================================================================
    print("\n" + "=" * 100, flush=True)
    print("EXPERIMENT 1: Label Propagation", flush=True)
    print("=" * 100, flush=True)

    best_lp = 0
    best_lp_config = ""

    for pool_name, allowed_models in pools.items():
        for bset_name, allowed_budgets in budget_sets.items():
            for lam in [0.0, 0.995, 0.999]:
                for lp_nn in [20, 40]:
                    for lp_alpha in [0.1, 0.2]:
                        t0 = time.time()
                        best_risk = np.full(n, -np.inf)
                        best_model = [None] * n
                        best_budget = [None] * n

                        for mn in allowed_models:
                            if mn not in models_data:
                                continue
                            price = prices[mn]
                            # Token: use KNN (LP not useful for continuous)
                            if "concise" in models_data[mn]:
                                y_tok = models_data[mn]["concise"]["output_tokens"][train_idx]
                                valid = ~np.isnan(y_tok)
                                if valid.sum() >= 3:
                                    tknn = KNeighborsRegressor(n_neighbors=min(28, valid.sum()-1), metric="cosine", weights="distance")
                                    tknn.fit(X_train[valid], y_tok[valid])
                                    tok = np.maximum(1.0, tknn.predict(embeddings))
                                else:
                                    tok = np.full(n, 50.0)
                            else:
                                tok = np.full(n, 50.0)

                            for budget in allowed_budgets:
                                if budget not in models_data[mn]:
                                    continue
                                y_all = models_data[mn][budget]["accuracy"]
                                labels = np.full(n, -1)
                                labels[train_idx] = (y_all[train_idx] >= 0.5).astype(int)
                                n_pos = (labels[train_idx] == 1).sum()
                                n_neg = (labels[train_idx] == 0).sum()

                                if n_pos < 2 or n_neg < 2:
                                    q = np.full(n, y_all[train_idx].mean())
                                else:
                                    try:
                                        ls = LabelSpreading(
                                            kernel='knn',
                                            n_neighbors=min(lp_nn, len(train_idx) - 1),
                                            alpha=lp_alpha,
                                            max_iter=100,
                                            n_jobs=-1
                                        )
                                        ls.fit(embeddings, labels)
                                        q = ls.label_distributions_[:, 1]
                                    except Exception as e:
                                        q = np.full(n, y_all[train_idx].mean())

                                risk = (1 - lam) * q - lam * tok * price / 1e6
                                better = risk > best_risk
                                idx = np.where(better)[0]
                                best_risk[idx] = risk[idx]
                                for j in idx:
                                    best_model[j] = mn
                                    best_budget[j] = budget

                        routes = list(zip(best_model, best_budget))
                        _, _, arena = evaluate_routes(routes, models_data, prices, n,
                            label=f"LP {pool_name}/{bset_name} lam={lam} nn={lp_nn} a={lp_alpha} ({time.time()-t0:.0f}s)")
                        if arena > best_lp:
                            best_lp = arena
                            best_lp_config = f"LP {pool_name}/{bset_name} lam={lam} nn={lp_nn} alpha={lp_alpha}"

    print(f"\n  BEST LP: {best_lp_config} Arena={best_lp:.2f}", flush=True)

    # ========================================================================
    # EXPERIMENT 2: Cluster-Then-Route
    # ========================================================================
    print("\n" + "=" * 100, flush=True)
    print("EXPERIMENT 2: Cluster-Then-Route", flush=True)
    print("=" * 100, flush=True)

    best_cl = 0
    best_cl_config = ""
    emb_norm = normalize(embeddings)

    for pool_name, allowed_models in pools.items():
        for bset_name, allowed_budgets in budget_sets.items():
            for lam in [0.0, 0.995, 0.999]:
                for nc in [40, 80, 120, 200]:
                    t0 = time.time()
                    kmeans = KMeans(n_clusters=nc, random_state=42, n_init=5)
                    cluster_labels = kmeans.fit_predict(emb_norm)

                    # Per-cluster means
                    global_means = {}
                    for mn in allowed_models:
                        if mn not in models_data:
                            continue
                        global_means[mn] = {}
                        for budget in allowed_budgets:
                            if budget not in models_data[mn]:
                                continue
                            global_means[mn][budget] = float(models_data[mn][budget]["accuracy"][train_idx].mean())

                    cluster_quality = {}
                    cluster_tokens = {}
                    for c in range(nc):
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
                                if len(cluster_train) >= 3:
                                    cluster_quality[c][mn][budget] = float(models_data[mn][budget]["accuracy"][cluster_train].mean())
                                else:
                                    cluster_quality[c][mn][budget] = global_means.get(mn, {}).get(budget, 0.5)
                            if "concise" in models_data[mn] and len(cluster_train) >= 3:
                                cluster_tokens[c][mn] = float(models_data[mn]["concise"]["output_tokens"][cluster_train].mean())
                            else:
                                cluster_tokens[c][mn] = 50.0

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

                    _, _, arena = evaluate_routes(routes, models_data, prices, n,
                        label=f"Cluster {pool_name}/{bset_name} lam={lam} nc={nc} ({time.time()-t0:.0f}s)")
                    if arena > best_cl:
                        best_cl = arena
                        best_cl_config = f"Cluster {pool_name}/{bset_name} lam={lam} nc={nc}"

    print(f"\n  BEST Cluster: {best_cl_config} Arena={best_cl:.2f}", flush=True)

    # ========================================================================
    # EXPERIMENT 3: Neighbor Voting (direct oracle imitation from sub_10)
    # ========================================================================
    print("\n" + "=" * 100, flush=True)
    print("EXPERIMENT 3: Neighbor Voting", flush=True)
    print("=" * 100, flush=True)

    best_nv = 0
    best_nv_config = ""

    for pool_name, allowed_models in pools.items():
        for bset_name, allowed_budgets in budget_sets.items():
            for lam in [0.0, 0.995, 0.999]:
                for k in [10, 20, 30, 50]:
                    t0 = time.time()
                    nn = NearestNeighbors(n_neighbors=min(k, len(train_idx) - 1), metric='cosine')
                    nn.fit(X_train)
                    distances, indices = nn.kneighbors(embeddings)

                    # For each training query, find best (model, budget)
                    train_best = {}
                    for ti_local, ti_global in enumerate(train_idx):
                        best_risk = -np.inf
                        best_choice = ("235b", "concise")
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
                        train_best[ti_local] = best_choice

                    # Vote for each query
                    routes = [None] * n
                    for i in range(n):
                        neighbor_local = indices[i]
                        neighbor_dists = distances[i]
                        weights = 1.0 / (neighbor_dists + 1e-8)
                        weights /= weights.sum()

                        choice_scores = defaultdict(float)
                        for ni, w in zip(neighbor_local, weights):
                            choice_scores[train_best[ni]] += w
                        routes[i] = max(choice_scores, key=choice_scores.get)

                    _, _, arena = evaluate_routes(routes, models_data, prices, n,
                        label=f"NV {pool_name}/{bset_name} lam={lam} k={k} ({time.time()-t0:.0f}s)")
                    if arena > best_nv:
                        best_nv = arena
                        best_nv_config = f"NV {pool_name}/{bset_name} lam={lam} k={k}"

    print(f"\n  BEST NV: {best_nv_config} Arena={best_nv:.2f}", flush=True)

    # ========================================================================
    # EXPERIMENT 4: LP + KNN Ensemble
    # ========================================================================
    print("\n" + "=" * 100, flush=True)
    print("EXPERIMENT 4: LP + KNN Ensemble", flush=True)
    print("=" * 100, flush=True)

    best_ens = 0
    best_ens_config = ""

    for pool_name in ["4model", "6model"]:
        allowed_models = pools[pool_name]
        for bset_name in ["top4", "all9"]:
            allowed_budgets = budget_sets[bset_name]
            for lam in [0.0, 0.995, 0.999]:
                for blend in [0.3, 0.5, 0.7]:
                    t0 = time.time()
                    best_risk = np.full(n, -np.inf)
                    best_model = [None] * n
                    best_budget = [None] * n

                    for mn in allowed_models:
                        if mn not in models_data:
                            continue
                        price = prices[mn]
                        if "concise" in models_data[mn]:
                            y_tok = models_data[mn]["concise"]["output_tokens"][train_idx]
                            valid = ~np.isnan(y_tok)
                            if valid.sum() >= 3:
                                tknn = KNeighborsRegressor(n_neighbors=min(28, valid.sum()-1), metric="cosine", weights="distance")
                                tknn.fit(X_train[valid], y_tok[valid])
                                tok = np.maximum(1.0, tknn.predict(embeddings))
                            else:
                                tok = np.full(n, 50.0)
                        else:
                            tok = np.full(n, 50.0)

                        for budget in allowed_budgets:
                            if budget not in models_data[mn]:
                                continue
                            y_all = models_data[mn][budget]["accuracy"]
                            y_train = y_all[train_idx]
                            valid = ~np.isnan(y_train)
                            if valid.sum() < 3:
                                continue

                            # KNN
                            knn = KNeighborsRegressor(n_neighbors=min(28, valid.sum()-1), metric="cosine", weights="distance")
                            knn.fit(X_train[valid], y_train[valid])
                            q_knn = knn.predict(embeddings)

                            # LP
                            labels = np.full(n, -1)
                            labels[train_idx] = (y_all[train_idx] >= 0.5).astype(int)
                            n_pos = (labels[train_idx] == 1).sum()
                            n_neg = (labels[train_idx] == 0).sum()
                            if n_pos < 2 or n_neg < 2:
                                q_lp = np.full(n, y_train.mean())
                            else:
                                try:
                                    ls = LabelSpreading(kernel='knn', n_neighbors=30, alpha=0.2, max_iter=100, n_jobs=-1)
                                    ls.fit(embeddings, labels)
                                    q_lp = ls.label_distributions_[:, 1]
                                except:
                                    q_lp = np.full(n, y_train.mean())

                            q = blend * q_lp + (1 - blend) * q_knn
                            risk = (1 - lam) * q - lam * tok * price / 1e6
                            better = risk > best_risk
                            idx = np.where(better)[0]
                            best_risk[idx] = risk[idx]
                            for j in idx:
                                best_model[j] = mn
                                best_budget[j] = budget

                    routes = list(zip(best_model, best_budget))
                    _, _, arena = evaluate_routes(routes, models_data, prices, n,
                        label=f"Ens {pool_name}/{bset_name} lam={lam} blend={blend} ({time.time()-t0:.0f}s)")
                    if arena > best_ens:
                        best_ens = arena
                        best_ens_config = f"Ens {pool_name}/{bset_name} lam={lam} blend={blend}"

    print(f"\n  BEST Ensemble: {best_ens_config} Arena={best_ens:.2f}", flush=True)

    # ========================================================================
    # EXPERIMENT 5: LP with continuous target (regression-like)
    # ========================================================================
    print("\n" + "=" * 100, flush=True)
    print("EXPERIMENT 5: LP Continuous (harmonic function approximation)", flush=True)
    print("=" * 100, flush=True)

    # Use LP but with continuous targets: threshold at multiple levels
    # and average the LP predictions
    best_lpc = 0
    best_lpc_config = ""

    for pool_name in ["4model", "6model"]:
        allowed_models = pools[pool_name]
        for bset_name in ["top4"]:
            allowed_budgets = budget_sets[bset_name]
            for lam in [0.0, 0.995, 0.999]:
                t0 = time.time()
                best_risk = np.full(n, -np.inf)
                best_model = [None] * n
                best_budget = [None] * n

                for mn in allowed_models:
                    if mn not in models_data:
                        continue
                    price = prices[mn]
                    if "concise" in models_data[mn]:
                        y_tok = models_data[mn]["concise"]["output_tokens"][train_idx]
                        valid = ~np.isnan(y_tok)
                        if valid.sum() >= 3:
                            tknn = KNeighborsRegressor(n_neighbors=min(28, valid.sum()-1), metric="cosine", weights="distance")
                            tknn.fit(X_train[valid], y_tok[valid])
                            tok = np.maximum(1.0, tknn.predict(embeddings))
                        else:
                            tok = np.full(n, 50.0)
                    else:
                        tok = np.full(n, 50.0)

                    for budget in allowed_budgets:
                        if budget not in models_data[mn]:
                            continue
                        y_all = models_data[mn][budget]["accuracy"]

                        # Multi-threshold LP: binarize at 0.3, 0.5, 0.7 and average
                        q_sum = np.zeros(n)
                        n_lps = 0
                        for threshold in [0.3, 0.5, 0.7]:
                            labels = np.full(n, -1)
                            labels[train_idx] = (y_all[train_idx] >= threshold).astype(int)
                            n_pos = (labels[train_idx] == 1).sum()
                            n_neg = (labels[train_idx] == 0).sum()
                            if n_pos < 2 or n_neg < 2:
                                continue
                            try:
                                ls = LabelSpreading(kernel='knn', n_neighbors=30, alpha=0.2, max_iter=100, n_jobs=-1)
                                ls.fit(embeddings, labels)
                                q_sum += ls.label_distributions_[:, 1]
                                n_lps += 1
                            except:
                                pass

                        if n_lps > 0:
                            q = q_sum / n_lps
                        else:
                            q = np.full(n, y_all[train_idx].mean())

                        risk = (1 - lam) * q - lam * tok * price / 1e6
                        better = risk > best_risk
                        idx = np.where(better)[0]
                        best_risk[idx] = risk[idx]
                        for j in idx:
                            best_model[j] = mn
                            best_budget[j] = budget

                routes = list(zip(best_model, best_budget))
                _, _, arena = evaluate_routes(routes, models_data, prices, n,
                    label=f"LPC {pool_name}/{bset_name} lam={lam} ({time.time()-t0:.0f}s)")
                if arena > best_lpc:
                    best_lpc = arena
                    best_lpc_config = f"LPC {pool_name}/{bset_name} lam={lam}"

    print(f"\n  BEST LP Continuous: {best_lpc_config} Arena={best_lpc:.2f}", flush=True)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 100, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"  KNN Baseline:          {best_overall_config:<55} Arena={best_overall:.2f}", flush=True)
    print(f"  Label Propagation:     {best_lp_config:<55} Arena={best_lp:.2f}", flush=True)
    print(f"  Cluster-Then-Route:    {best_cl_config:<55} Arena={best_cl:.2f}", flush=True)
    print(f"  Neighbor Voting:       {best_nv_config:<55} Arena={best_nv:.2f}", flush=True)
    print(f"  LP+KNN Ensemble:       {best_ens_config:<55} Arena={best_ens:.2f}", flush=True)
    print(f"  LP Continuous:         {best_lpc_config:<55} Arena={best_lpc:.2f}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
