#!/usr/bin/env python3
"""
Global KNN routing: train KNN predictors on ALL data (no category split),
then route queries to best (model, budget).

Modes:
  - quality: Predict per-model quality scores, route via risk formula (default)
  - oracle:  Predict Oracle's model choice directly (classification)

Usage:
    .venv/bin/python scripts/global_knn_route.py \
        --lambda_val 0.999 --shrinkage_k 4.0 \
        --models 235b 80b gemini-flash haiku gpt-5.2

    .venv/bin/python scripts/global_knn_route.py \
        --mode oracle --models 235b 80b gemini-flash haiku
"""

import os, sys, json, pickle, math
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    MODELS, TRAINING_DATA_PATH, MODEL_COST_PATH,
    CATEGORY_NAMES, get_budgets,
)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def arena_score(accuracy, cost_per_1kq, beta=0.1):
    if cost_per_1kq <= 0:
        return 0.0
    b2 = beta ** 2
    A = accuracy
    C = (math.log2(200) - math.log2(cost_per_1kq)) / (math.log2(200) - math.log2(0.0044))
    if b2 * A + C == 0:
        return 0.0
    return ((1 + b2) * A * C) / (b2 * A + C) * 100


def run_oracle_mode(args):
    """Oracle Imitation: predict which model the Oracle would pick, then use it."""
    allowed_models = args.models if args.models else list(MODELS.keys())
    knn_k_arg = args.knn_k

    log(f"=== Oracle Imitation Mode ===")
    log(f"  Models: {sorted(allowed_models)}")

    # 1. Load data
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    categories = data["categories"]
    models_data = data["models"]
    gi_list = data["global_indices"]
    n = embeddings.shape[0]

    # Load prices
    with open(MODEL_COST_PATH) as f:
        cost_data = json.load(f)
    prices = {}
    for mn, cfg in MODELS.items():
        prices[mn] = cost_data.get(cfg["cost_key"], {}).get("output_token_price_per_million", 0)

    # 2. Compute Oracle labels (concise-only): which model has best accuracy?
    #    Tie-break: lower cost
    oracle_labels = []
    for i in range(n):
        best_acc = -1
        best_cost = float("inf")
        best_mn = allowed_models[0]
        for mn in allowed_models:
            if mn not in models_data or "concise" not in models_data[mn]:
                continue
            acc = models_data[mn]["concise"]["accuracy"][i]
            tok = models_data[mn]["concise"]["output_tokens"][i]
            cost = tok * prices[mn] / 1e6
            if acc > best_acc or (acc == best_acc and cost < best_cost):
                best_acc = acc
                best_cost = cost
                best_mn = mn
        oracle_labels.append(best_mn)
    oracle_labels = np.array(oracle_labels)

    # Label distribution
    from collections import Counter
    label_counts = Counter(oracle_labels)
    log(f"\nOracle label distribution (full dataset):")
    for mn, cnt in label_counts.most_common():
        log(f"  {mn:<20} {cnt:>5} ({cnt/n*100:.1f}%)")

    # 3. 10/90 split
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
    train_idx, test_idx = next(sss.split(embeddings, categories))
    log(f"\n  Split: {len(train_idx)} train, {len(test_idx)} test")

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train = oracle_labels[train_idx]
    y_test = oracle_labels[test_idx]

    # Auto KNN k
    knn_k = knn_k_arg if knn_k_arg > 0 else max(3, int(np.sqrt(len(train_idx))))
    knn_k = min(knn_k, len(train_idx) - 1)
    log(f"  KNN k={knn_k}")

    # 4. Train classifiers (KNN + LogReg + SVM-RBF)
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    classifiers = {}

    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=knn_k, metric="cosine", weights="distance")
    clf_knn.fit(X_train, y_train)
    classifiers["KNN"] = clf_knn

    # LogisticRegression
    clf_lr = LogisticRegression(C=1.0, max_iter=2000)
    clf_lr.fit(X_train, y_train)
    classifiers["LogReg"] = clf_lr

    # SVM-RBF (needs scaling)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    clf_svm = SVC(kernel="rbf", C=10.0, gamma="scale")
    clf_svm.fit(X_train_sc, y_train)
    classifiers["SVM-RBF"] = clf_svm

    best_arena = -1
    best_name = None

    for clf_name, clf in classifiers.items():
        if clf_name == "SVM-RBF":
            y_pred = clf.predict(X_test_sc)
        else:
            y_pred = clf.predict(X_test)
        cls_acc = accuracy_score(y_test, y_pred)
        log(f"\n  [{clf_name}] Oracle imitation accuracy: {cls_acc*100:.2f}%")

        # Per-class recall
        for mn in sorted(set(oracle_labels)):
            mask = y_test == mn
            if mask.sum() > 0:
                ca = (y_pred[mask] == mn).mean()
                log(f"    {mn:<20} {mask.sum():>5} test, recall={ca*100:.1f}%")

        # Evaluate routing: predicted model + concise budget
        n_test = len(test_idx)
        total_acc = 0
        total_cost = 0
        found = 0

        for j in range(n_test):
            idx = test_idx[j]
            mn = y_pred[j]
            if mn not in models_data or "concise" not in models_data[mn]:
                continue
            acc = models_data[mn]["concise"]["accuracy"][idx]
            tok = models_data[mn]["concise"]["output_tokens"][idx]
            cost = tok * prices[mn] / 1e6
            total_acc += acc
            total_cost += cost
            found += 1

        if found > 0:
            mean_acc = total_acc / found
            mean_cost = total_cost / found * 1000
            arena = arena_score(mean_acc, mean_cost)
        else:
            mean_acc = mean_cost = arena = 0

        log(f"  -> Routing: Acc={mean_acc*100:.2f}%  Cost/1kq=${mean_cost:.4f}  Arena={arena:.2f}")

        pred_counts = Counter(y_pred)
        log(f"  -> Distribution: {', '.join(f'{m}={c}({c/n_test*100:.0f}%)' for m, c in pred_counts.most_common())}")

        if arena > best_arena:
            best_arena = arena
            best_name = clf_name

    # Oracle upper bound
    n_test = len(test_idx)
    oracle_acc = 0
    oracle_cost = 0
    for j in range(n_test):
        idx = test_idx[j]
        mn = y_test[j]
        if mn not in models_data or "concise" not in models_data[mn]:
            continue
        oracle_acc += models_data[mn]["concise"]["accuracy"][idx]
        oracle_cost += models_data[mn]["concise"]["output_tokens"][idx] * prices[mn] / 1e6
    oracle_acc /= n_test
    oracle_cost = oracle_cost / n_test * 1000
    oracle_arena = arena_score(oracle_acc, oracle_cost)

    log(f"\n{'='*70}")
    log(f"[Oracle upper bound] Acc={oracle_acc*100:.2f}%  Cost/1kq=${oracle_cost:.4f}  Arena={oracle_arena:.2f}")
    log(f"[Best classifier]    {best_name} → Arena={best_arena:.2f}")
    log(f"{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_val", type=float, default=0.999)
    parser.add_argument("--shrinkage_k", type=float, default=4.0)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--knn_k", type=int, default=0,
                        help="KNN neighbors (0=auto: sqrt(n_train))")
    parser.add_argument("--exclude-budgets", nargs="*", default=[])
    parser.add_argument("--force-mean-tokens", action="store_true")
    parser.add_argument("--mode", choices=["quality", "oracle"], default="quality",
                        help="quality: predict scores + risk routing. oracle: imitate Oracle model choice.")
    args = parser.parse_args()

    if args.mode == "oracle":
        return run_oracle_mode(args)

    lam = args.lambda_val
    sk = args.shrinkage_k
    excluded_budgets = set(args.exclude_budgets)

    log(f"=== Global KNN Router: lambda={lam}, shrinkage_k={sk} ===")

    # 1. Load data
    log("Loading training data...")
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]  # (8400, 1024)
    categories = data["categories"]
    models_data = data["models"]
    gi_list = data["global_indices"]
    n = embeddings.shape[0]

    allowed_models = set(args.models) if args.models else set(MODELS.keys())
    log(f"  {n} queries, models: {sorted(allowed_models)}")

    # 2. 10/90 split
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
    train_idx, test_idx = next(sss.split(embeddings, categories))
    log(f"  Split: {len(train_idx)} train, {len(test_idx)} test")

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]

    # Auto KNN k
    knn_k = args.knn_k if args.knn_k > 0 else max(3, int(np.sqrt(len(train_idx))))
    knn_k = min(knn_k, len(train_idx) - 1)
    log(f"  KNN k={knn_k}")

    # 3. Load prices
    with open(MODEL_COST_PATH) as f:
        cost_data = json.load(f)
    prices = {}
    for mn, cfg in MODELS.items():
        prices[mn] = cost_data.get(cfg["cost_key"], {}).get("output_token_price_per_million", 0)
    log(f"  Prices: {{{', '.join(f'{m}: ${prices[m]}' for m in sorted(allowed_models, key=lambda x: -prices[x]))}}}")

    # 4. Train global KNN predictors + compute global means
    log("\nTraining global KNN predictors...")
    quality_knns = {}  # {model: {budget: KNeighborsRegressor}}
    token_knns = {}    # {model: KNeighborsRegressor}
    global_mean_acc = {}   # {model: {budget: float}}
    global_mean_tok = {}   # {model: float}
    cv_r2 = {}  # {model: {budget: float}}  -- test R2 for shrinkage

    for mn in sorted(allowed_models):
        if mn not in models_data:
            log(f"  {mn}: NOT in training data, skipping")
            continue
        budgets = get_budgets(mn)
        quality_knns[mn] = {}
        global_mean_acc[mn] = {}
        cv_r2[mn] = {}

        valid_budgets = []
        for b in budgets:
            if b in excluded_budgets:
                continue
            if b not in models_data[mn]:
                continue
            y = models_data[mn][b]["accuracy"]
            if y[train_idx].sum() == 0:
                continue
            valid_budgets.append(b)

        test_r2s = []
        for b in valid_budgets:
            y_all = models_data[mn][b]["accuracy"]
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            # Global mean
            global_mean_acc[mn][b] = float(y_train.mean())

            # Train KNN
            knn = KNeighborsRegressor(n_neighbors=knn_k, metric="cosine", weights="distance")
            knn.fit(X_train, y_train)
            quality_knns[mn][b] = knn

            # Test R2
            y_pred = knn.predict(X_test)
            r2 = r2_score(y_test, y_pred) if y_test.var() > 1e-10 else 0.0
            cv_r2[mn][b] = r2
            test_r2s.append(r2)

        # Token predictor (from concise)
        token_budget = "concise"
        if token_budget in models_data[mn]:
            tok_all = models_data[mn][token_budget]["output_tokens"]
            tok_train = tok_all[train_idx]
            if tok_train.sum() > 0:
                tknn = KNeighborsRegressor(n_neighbors=knn_k, metric="cosine", weights="distance")
                tknn.fit(X_train, tok_train)
                token_knns[mn] = tknn
                global_mean_tok[mn] = float(tok_train.mean())

                tok_pred = tknn.predict(X_test)
                tok_r2 = r2_score(tok_all[test_idx], tok_pred) if tok_all[test_idx].var() > 1e-10 else 0.0
            else:
                global_mean_tok[mn] = 50.0
                tok_r2 = -1.0
        else:
            global_mean_tok[mn] = 50.0
            tok_r2 = -1.0

        mean_test_r2 = np.mean(test_r2s) if test_r2s else 0.0
        log(f"  {mn:<15} budgets={len(valid_budgets)}  mean_quality_R2={mean_test_r2:.4f}  token_R2={tok_r2:.4f}")

    # 5. Route on test set
    log("\nRouting on test set...")
    n_test = len(test_idx)
    best_risk = np.full(n_test, -np.inf)
    best_model = [None] * n_test
    best_budget = [None] * n_test

    for mn in allowed_models:
        if mn not in quality_knns:
            continue
        price = prices[mn]

        # Token prediction
        if args.force_mean_tokens or mn not in token_knns:
            tok = np.full(n_test, global_mean_tok.get(mn, 50.0))
        else:
            tok = np.maximum(1.0, token_knns[mn].predict(X_test))

        for budget, knn in quality_knns[mn].items():
            q = knn.predict(X_test)

            # Shrinkage
            if sk > 0:
                r2 = cv_r2[mn].get(budget, 0.0)
                alpha = min(1.0, max(0.0, r2 * sk))
                cm = global_mean_acc[mn].get(budget, 0.0)
                q = alpha * q + (1 - alpha) * cm

            risk = (1 - lam) * q - lam * tok * price / 1e6

            for j in range(n_test):
                if risk[j] > best_risk[j]:
                    best_risk[j] = risk[j]
                    best_model[j] = mn
                    best_budget[j] = budget

    # 6. Evaluate against sweep ground truth
    log("\nLoading sweep ground truth...")
    sweep_root = "/orange/qi855292.ucf/ah872032.ucf/budget_sweep"
    total_acc = 0
    total_cost = 0
    found = 0

    for j in range(n_test):
        idx = test_idx[j]
        gi = gi_list[idx]
        mn = best_model[j]
        budget = best_budget[j]

        if mn is None:
            continue

        # Load from sweep
        sweep_dir = MODELS[mn]["sweep_dir"]
        sweep_file = os.path.join(sweep_dir, f"{budget}.json")

        # Use training_data.pkl accuracy (matches sweep)
        if mn in models_data and budget in models_data[mn]:
            acc = models_data[mn][budget]["accuracy"][idx]
            # Cost from output_tokens * price
            tok = models_data[mn][budget]["output_tokens"][idx]
            cost = tok * prices[mn] / 1e6
            total_acc += acc
            total_cost += cost
            found += 1

    if found > 0:
        mean_acc = total_acc / found
        mean_cost = total_cost / found * 1000
        arena = arena_score(mean_acc, mean_cost)
    else:
        mean_acc = mean_cost = arena = 0

    log(f"\n{'='*70}")
    log(f"[training_data.pkl]  Acc={mean_acc*100:.2f}%  Cost/1kq=${mean_cost:.4f}  Arena={arena:.2f}")
    log(f"  (evaluated {found}/{n_test} queries)")

    # Model distribution
    from collections import Counter
    model_counts = Counter(best_model)
    budget_counts = Counter(best_budget)
    log(f"\nModel distribution:")
    for mn, cnt in model_counts.most_common():
        log(f"  {mn:<20} {cnt:>5} ({cnt/n_test*100:.1f}%)")
    log(f"\nBudget distribution:")
    for bn, cnt in budget_counts.most_common():
        log(f"  {bn:<20} {cnt:>5} ({cnt/n_test*100:.1f}%)")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
