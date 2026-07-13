#!/usr/bin/env python3
"""
R2-Router: Route queries and evaluate against sweep ground truth.

Steps:
  1. Load predictors (quality KNN per model×budget, token predictor per model)
  2. For each query, compute risk = (1-λ)*quality - λ*tokens*price for all (model, budget)
  3. Pick the best (model, budget) per query
  4. Look up actual accuracy and cost from budget_sweep JSON files
  5. Compute accuracy, cost/1kq, arena score

Usage:
    .venv/bin/python scripts/route_and_eval.py --lambda_val 0.98
    .venv/bin/python scripts/route_and_eval.py --lambda_val 0.999 --shrinkage_k 0
"""

import os
import sys
import json
import math
import pickle
import argparse
import numpy as np
from collections import defaultdict
from joblib import load as joblib_load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    MODELS, CHECKPOINT_DIR, CATEGORY_NAMES,
    TRAINING_DATA_PATH, MODEL_COST_PATH, ROUTER_DATA_10_PATH,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def arena_score(accuracy: float, cost_per_1kq: float, beta: float = 0.1) -> float:
    if cost_per_1kq <= 0:
        cost_per_1kq = 0.001
    C = (math.log2(200) - math.log2(cost_per_1kq)) / (math.log2(200) - math.log2(0.0044))
    C = max(0.0, min(1.0, C))
    A = accuracy
    denom = beta * A + C
    if denom == 0:
        return 0.0
    return ((1 + beta) * A * C) / denom * 100


# ── Load predictors ─────────────────────────────────────────────────────────

def load_predictors():
    """Load per-category quality and token KNN predictors.

    Returns:
        quality: {cat_name: {model_name: {budget: predictor}}}
        token:   {cat_name: {model_name: predictor}}
        confidence: {cat_name: {model_name: {"quality_r2": float, "token_r2": float}}}
    """
    predictors_dir = os.path.join(CHECKPOINT_DIR, "predictors")
    quality = {}
    token = {}

    for cat_name in CATEGORY_NAMES:
        quality[cat_name] = {}
        token[cat_name] = {}
        cat_dir = os.path.join(predictors_dir, cat_name)
        if not os.path.isdir(cat_dir):
            continue
        for model_name in MODELS:
            quality[cat_name][model_name] = {}
            meta_path = os.path.join(cat_dir, f"{model_name}_quality_meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for b in meta["budgets"]:
                path = os.path.join(cat_dir, f"{model_name}_{b}_quality.joblib")
                if os.path.exists(path):
                    quality[cat_name][model_name][b] = joblib_load(path)
            token_path = os.path.join(cat_dir, f"{model_name}_token.joblib")
            if os.path.exists(token_path):
                token[cat_name][model_name] = joblib_load(token_path)

    # Load CV R² for shrinkage
    results_path = os.path.join(CHECKPOINT_DIR, "predictor_results.json")
    with open(results_path) as f:
        results = json.load(f)
    confidence = {}
    for cat_name in CATEGORY_NAMES:
        confidence[cat_name] = {}
        if cat_name not in results:
            continue
        for model_name, minfo in results[cat_name]["models"].items():
            q_r2 = minfo.get("KNN_cv_r2", 0.0)
            t_r2 = minfo.get("knn_token_cv_r2", 0.0)
            confidence[cat_name][model_name] = {
                "quality_r2": max(0.0, q_r2),
                "token_r2": t_r2,
            }

    return quality, token, confidence


# ── Load prices ──────────────────────────────────────────────────────────────

def load_prices():
    """Load output token price per model from model_cost.json."""
    with open(MODEL_COST_PATH) as f:
        cost_data = json.load(f)
    prices = {}
    for model_name, cfg in MODELS.items():
        cost_key = cfg["cost_key"]
        prices[model_name] = cost_data.get(cost_key, {}).get("output_token_price_per_million", 0)
    return prices


# ── Routing ──────────────────────────────────────────────────────────────────

def route(embeddings, categories, quality_preds, token_preds, confidence,
          prices, lam, shrinkage_k, mean_acc, mean_tokens,
          allowed_models=None, excluded_budgets=None,
          force_mean_tokens=False):
    """Route each query to the best (model, budget).

    For each query i in category c:
      For each (model m, budget b):
        quality_shrunk = alpha * predictor(x_i) + (1-alpha) * cat_mean
            where alpha = clip(quality_r2 * shrinkage_k, 0, 1)
        token_pred = predictor(x_i) if token_r2 > 0, else cat_mean
            (binary switch)
        risk = (1-lam) * quality_shrunk - lam * token_pred * price_m / 1e6
      Pick (m*, b*) = argmax risk

    Returns: list of (model_name, budget_name) per query
    """
    if allowed_models is None:
        allowed_models = set(MODELS.keys())
    if excluded_budgets is None:
        excluded_budgets = set()

    n = embeddings.shape[0]
    routes = [None] * n

    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        cat_indices = np.where(categories == cat_idx)[0]
        if len(cat_indices) == 0:
            continue

        X = embeddings[cat_indices]
        n_cat = len(cat_indices)

        # Predict tokens per model (once, shared across budgets)
        token_pred = {}
        for mn in allowed_models:
            if force_mean_tokens:
                token_pred[mn] = np.full(n_cat, mean_tokens.get(cat_name, {}).get(mn, 50.0))
            else:
                conf = confidence.get(cat_name, {}).get(mn, {})
                t_r2 = conf.get("token_r2", -1.0)
                if shrinkage_k > 0 and t_r2 <= 0:
                    token_pred[mn] = np.full(n_cat, mean_tokens.get(cat_name, {}).get(mn, 50.0))
                elif mn in token_preds.get(cat_name, {}):
                    token_pred[mn] = np.maximum(1.0, token_preds[cat_name][mn].predict(X))
                else:
                    token_pred[mn] = np.full(n_cat, mean_tokens.get(cat_name, {}).get(mn, 50.0))

        # Find best (model, budget) per query
        best_risk = np.full(n_cat, -np.inf)
        best_model = [None] * n_cat
        best_budget = [None] * n_cat

        for mn in allowed_models:
            if mn not in quality_preds.get(cat_name, {}):
                continue
            price = prices[mn]
            tok = token_pred[mn]

            conf = confidence.get(cat_name, {}).get(mn, {})
            q_r2 = conf.get("quality_r2", 0.0)
            alpha = min(1.0, max(0.0, q_r2 * shrinkage_k)) if shrinkage_k > 0 else 1.0

            for budget, reg in quality_preds[cat_name][mn].items():
                if budget in excluded_budgets:
                    continue
                # LogisticRegression: use predict_proba for P(correct)
                if hasattr(reg, 'predict_proba'):
                    q = reg.predict_proba(X)[:, 1]
                else:
                    q = reg.predict(X)
                if shrinkage_k > 0:
                    cm = mean_acc.get(cat_name, {}).get(mn, {}).get(budget, 0.0)
                    q = alpha * q + (1 - alpha) * cm
                risk = (1 - lam) * q - lam * tok * price / 1e6

                for j in range(n_cat):
                    if risk[j] > best_risk[j]:
                        best_risk[j] = risk[j]
                        best_model[j] = mn
                        best_budget[j] = budget

        for j, idx in enumerate(cat_indices):
            routes[idx] = (best_model[j], best_budget[j])

    return routes


# ── Evaluate against sweep ground truth ──────────────────────────────────────

def load_sweep_entry(model_name, budget):
    """Load sweep JSON and build {global_index: entry} map for regular entries."""
    sweep_dir = MODELS[model_name]["sweep_dir"]
    path = os.path.join(sweep_dir, f"{budget}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {e["global index"]: e for e in data if not e.get("for_optimality")}


def evaluate(routes, global_indices, prices):
    """Look up actual accuracy and cost from sweep files.

    Returns: accuracy, cost_per_1kq, arena_score, model_counts
    """
    n = len(routes)

    # Cache sweep data: (model, budget) -> {gi: entry}
    sweep_cache = {}
    total_acc = 0.0
    total_cost = 0.0
    found = 0
    model_counts = defaultdict(int)

    for i, (mn, budget) in enumerate(routes):
        gi = global_indices[i]
        model_counts[mn] += 1

        key = (mn, budget)
        if key not in sweep_cache:
            sweep_cache[key] = load_sweep_entry(mn, budget)

        entry = sweep_cache[key].get(gi)
        if entry is None:
            continue

        acc = entry.get("accuracy")
        cost = entry.get("cost")
        if acc is not None:
            total_acc += float(acc)
            found += 1
        if cost is not None:
            total_cost += float(cost)

    accuracy = total_acc / n
    cost_1kq = total_cost / n * 1000
    score = arena_score(accuracy, cost_1kq)

    return accuracy, cost_1kq, score, model_counts, found


# ── Export submission ────────────────────────────────────────────────────────

def export_submission(routes, global_indices, data, output_path):
    """Generate RouterArena submission JSON from routing decisions.

    Loads generated_result from sweep files for regular queries.
    Generates optimality entries (each model appears once per optimality query).
    """
    n = len(routes)

    # Cache sweep data
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

    # Load router_data_10 for optimality queries (809 queries)
    if os.path.exists(ROUTER_DATA_10_PATH):
        with open(ROUTER_DATA_10_PATH) as f:
            router_data_10 = json.load(f)

        # Build map: gi -> chosen model for regular routing
        gi_to_model = {global_indices[i]: routes[i][0] for i in range(n)}

        # Get unique models used in routing
        used_models = set(mn for mn, _ in routes)

        # For each optimality query, add entries for all used models EXCEPT the chosen one
        for rd_entry in router_data_10:
            gi = rd_entry["global index"]
            chosen_model = gi_to_model.get(gi)

            for mn in used_models:
                if mn == chosen_model:
                    continue  # Skip the model already chosen in regular routing
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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    n_regular = sum(1 for e in entries if not e["for_optimality"])
    n_opt = sum(1 for e in entries if e["for_optimality"])
    print(f"\nExported {len(entries)} entries ({n_regular} regular, {n_opt} optimality) to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="R2-Router: Route and evaluate")
    parser.add_argument("--lambda_val", type=float, default=0.98)
    parser.add_argument("--shrinkage_k", type=float, default=3.0,
                        help="0 = raw predictor, >0 = shrinkage (default 3.0)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only use these models (short names). Default: all")
    parser.add_argument("--exclude-budgets", nargs="*", default=[],
                        help="Exclude these budget names (e.g. budget_unlimited budget_1500)")
    parser.add_argument("--test-only", action="store_true",
                        help="Evaluate only on held-out test indices from train_test_split.pkl")
    parser.add_argument("--export", type=str, default=None,
                        help="Export submission JSON to this path")
    parser.add_argument("--force-mean-tokens", action="store_true",
                        help="Always use category mean for token prediction (ignore predictor)")
    args = parser.parse_args()

    lam = args.lambda_val
    sk = args.shrinkage_k
    allowed_models = set(args.models) if args.models else set(MODELS.keys())
    excluded_budgets = set(args.exclude_budgets)
    print(f"=== R2-Router: λ={lam}, shrinkage_k={sk} ===")
    print(f"  models: {sorted(allowed_models)}")
    if excluded_budgets:
        print(f"  excluded budgets: {sorted(excluded_budgets)}")
    if args.test_only:
        print(f"  MODE: test-only (held-out evaluation)")
    if args.export:
        print(f"  EXPORT: {args.export}")
    print()

    # 1. Load data
    print("Loading training data...")
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]      # (8400, 1024)
    categories = data["categories"]      # (8400,)
    models_data = data["models"]         # {model: {budget: {accuracy, output_tokens}}}
    global_indices = data["global_indices"]  # [str, ...]
    n = embeddings.shape[0]
    print(f"  {n} queries, {embeddings.shape[1]}d embeddings\n")

    # 2. Load predictors and prices
    print("Loading predictors...")
    quality_preds, token_preds, confidence = load_predictors()
    prices = load_prices()
    print(f"  Prices: { {m: f'${p}' for m, p in sorted(prices.items(), key=lambda x: -x[1])} }\n")

    # 3. Compute category means (needed for shrinkage)
    mean_acc = {}
    mean_tokens = {}
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        cat_mask = np.where(categories == cat_idx)[0]
        if len(cat_mask) == 0:
            continue
        mean_acc[cat_name] = {}
        mean_tokens[cat_name] = {}
        for mn, budgets_data in models_data.items():
            mean_acc[cat_name][mn] = {}
            for budget, bdata in budgets_data.items():
                mean_acc[cat_name][mn][budget] = float(bdata["accuracy"][cat_mask].mean())
            if "concise" in budgets_data:
                mean_tokens[cat_name][mn] = float(
                    max(1.0, budgets_data["concise"]["output_tokens"][cat_mask].mean())
                )
            else:
                mean_tokens[cat_name][mn] = 50.0

    # 4. Route
    print("Routing...")
    routes = route(embeddings, categories, quality_preds, token_preds, confidence,
                   prices, lam, sk, mean_acc, mean_tokens,
                   allowed_models=allowed_models, excluded_budgets=excluded_budgets,
                   force_mean_tokens=args.force_mean_tokens)

    # 4b. Filter to test-only if requested
    eval_indices = None
    if args.test_only:
        split_path = os.path.join(CHECKPOINT_DIR, "train_test_split.pkl")
        if not os.path.exists(split_path):
            print(f"ERROR: {split_path} not found. Retrain without --full first.")
            sys.exit(1)
        with open(split_path, "rb") as f:
            split = pickle.load(f)
        test_idx = split["test_idx"]
        train_idx = split["train_idx"]
        if len(test_idx) == 0:
            print("ERROR: No held-out test set (trained with --full). Retrain without --full.")
            sys.exit(1)
        eval_indices = set(test_idx.tolist())
        print(f"  Test-only: evaluating on {len(test_idx)}/{n} held-out queries")
        print(f"  (train={len(train_idx)}, test={len(test_idx)})")

    # 5. Evaluate against sweep ground truth
    print("\n" + "=" * 70)

    if eval_indices is not None:
        # Filter routes and data to eval indices
        eval_idx_list = sorted(eval_indices)
        eval_routes = [routes[i] for i in eval_idx_list]
        eval_gi = [global_indices[i] for i in eval_idx_list]
        n_eval = len(eval_idx_list)

        print("Loading sweep ground truth...")
        acc, cost_1kq, score, counts, found = evaluate(
            eval_routes, eval_gi, prices)
        print(f"Acc={acc*100:.2f}%  Cost/1kq=${cost_1kq:.4f}  Arena={score:.2f}  (found {found}/{n_eval})")

        # Model distribution
        print(f"\nModel distribution (test set, n={n_eval}):")
        for mn, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {mn:<20} {cnt:>5} ({cnt/n_eval*100:.1f}%)")
    else:
        # Full evaluation
        print("Loading sweep ground truth...")
        acc, cost_1kq, score, counts, found = evaluate(routes, global_indices, prices)
        print(f"Acc={acc*100:.2f}%  Cost/1kq=${cost_1kq:.4f}  Arena={score:.2f}")
        print(f"  (found {found}/{n} entries with accuracy in sweep files)")

        # Model distribution
        print(f"\nModel distribution:")
        for mn, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {mn:<20} {cnt:>5} ({cnt/n*100:.1f}%)")

    # Budget distribution (always on full routes)
    budget_counts = defaultdict(int)
    for mn, budget in routes:
        budget_counts[budget] += 1
    print(f"\nBudget distribution:")
    for b, cnt in sorted(budget_counts.items(), key=lambda x: -x[1]):
        print(f"  {b:<25} {cnt:>5} ({cnt/n*100:.1f}%)")

    print(f"\n{'=' * 70}")

    # 6. Export submission JSON if requested
    if args.export:
        export_submission(routes, global_indices, data, args.export)


if __name__ == "__main__":
    main()
