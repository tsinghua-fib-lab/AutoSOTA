#!/usr/bin/env python3
"""
Sweep lambda and beta to compare routing performance.

For each lambda, route all queries → look up actual accuracy/cost from
sweep ground truth → compute arena score at multiple beta values.

Usage:
    .venv/bin/python scripts/sweep_lambda.py
    .venv/bin/python scripts/sweep_lambda.py --models 235b 80b gemini-flash haiku
    .venv/bin/python scripts/sweep_lambda.py --exclude-budgets budget_unlimited budget_1500
    .venv/bin/python scripts/sweep_lambda.py --shrinkage_k 3.0
"""

import os
import sys
import json
import math
import pickle
import argparse
import numpy as np
from collections import defaultdict
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    MODELS, CHECKPOINT_DIR, CATEGORY_NAMES,
    TRAINING_DATA_PATH, MODEL_COST_PATH,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def arena_score(accuracy: float, cost_per_1kq: float, beta: float = 0.1) -> float:
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


# ── Load predictors ─────────────────────────────────────────────────────────

def load_predictors():
    from joblib import load as joblib_load
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


def load_prices():
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
          allowed_models=None, excluded_budgets=None):
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

        token_pred = {}
        for mn in allowed_models:
            conf = confidence.get(cat_name, {}).get(mn, {})
            t_r2 = conf.get("token_r2", -1.0)
            if shrinkage_k > 0 and t_r2 <= 0:
                token_pred[mn] = np.full(n_cat, mean_tokens.get(cat_name, {}).get(mn, 50.0))
            elif mn in token_preds.get(cat_name, {}):
                token_pred[mn] = np.maximum(1.0, token_preds[cat_name][mn].predict(X))
            else:
                token_pred[mn] = np.full(n_cat, mean_tokens.get(cat_name, {}).get(mn, 50.0))

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


# TODO: Evaluation should use real cost from sweep files (per-query accuracy + cost
# from eval_sweep.py output), not output_tokens * price which is incorrect.
# This checkpoint-based sweep script needs an evaluate_from_sweep_files() function
# that loads the sweep ground truth JSON files directly.


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep lambda & beta for R2-Router")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only use these models (short names). Default: all")
    parser.add_argument("--exclude-budgets", nargs="*", default=[],
                        help="Exclude these budget names")
    parser.add_argument("--shrinkage_k", type=float, default=0.0,
                        help="Shrinkage multiplier (0=no shrinkage)")
    parser.add_argument("--lambdas", nargs="*", type=float,
                        default=[0.0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0],
                        help="Lambda values to sweep")
    parser.add_argument("--betas", nargs="*", type=float,
                        default=[0.1, 0.2, 0.5, 1.0],
                        help="Beta values for arena score")
    args = parser.parse_args()

    start = time.time()

    allowed_models = set(args.models) if args.models else set(MODELS.keys())
    excluded_budgets = set(args.exclude_budgets)

    print("=" * 100)
    print("R2-Router Lambda/Beta Sweep")
    print("=" * 100)
    print(f"  Models ({len(allowed_models)}): {sorted(allowed_models)}")
    print(f"  Shrinkage k: {args.shrinkage_k}")
    print(f"  Excluded budgets: {sorted(excluded_budgets) if excluded_budgets else 'none'}")
    print(f"  Lambdas: {args.lambdas}")
    print(f"  Betas: {args.betas}")
    print()

    # 1. Load data
    print("Loading training data...", flush=True)
    with open(TRAINING_DATA_PATH, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    categories = data["categories"]
    models_data = data["models"]
    n = embeddings.shape[0]
    print(f"  {n} queries, {embeddings.shape[1]}d embeddings")

    # 2. Load predictors and prices
    print("Loading predictors...", flush=True)
    quality_preds, token_preds, confidence = load_predictors()
    prices = load_prices()

    # 3. Compute category means
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
                    max(1.0, budgets_data["concise"]["output_tokens"][cat_mask].mean()))
            else:
                mean_tokens[cat_name][mn] = 50.0

    print(f"  Setup done in {time.time()-start:.1f}s\n", flush=True)

    # 4. Sweep lambdas
    # TODO: Need evaluate_from_sweep_files() that loads real accuracy + cost from
    # sweep ground truth JSON files. See route_and_eval.py for reference.
    raise NotImplementedError(
        "evaluate_from_training_data was removed (used output_tokens * price which is wrong). "
        "Use scripts/route_and_eval.py for evaluation, or implement evaluate_from_sweep_files() "
        "that loads real cost from sweep ground truth JSON files."
    )

    results = []
    for lam in args.lambdas:
        t0 = time.time()
        routes = route(embeddings, categories, quality_preds, token_preds, confidence,
                       prices, lam, args.shrinkage_k, mean_acc, mean_tokens,
                       allowed_models=allowed_models, excluded_budgets=excluded_budgets)
        # TODO: replace with evaluate_from_sweep_files(routes, ...)
        accuracy, cost_1kq, model_counts = 0.0, 0.0, defaultdict(int)
        elapsed = time.time() - t0

        scores = {}
        for beta in args.betas:
            scores[beta] = arena_score(accuracy, cost_1kq, beta)

        results.append({
            "lambda": lam,
            "accuracy": accuracy,
            "cost_1kq": cost_1kq,
            "scores": scores,
            "model_counts": dict(model_counts),
            "elapsed": elapsed,
        })
        print(f"  λ={lam:.4f}  Acc={accuracy*100:6.2f}%  Cost=${cost_1kq:.4f}/1kq  "
              f"Arena(β=0.1)={scores[0.1]:6.2f}  [{elapsed:.1f}s]", flush=True)

    # 5. Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    # Header
    beta_headers = "  ".join(f"Arena(β={b})" for b in args.betas)
    print(f"{'λ':>8}  {'Accuracy':>10}  {'Cost/1kq':>10}  {beta_headers}")
    print("-" * (32 + 12 * len(args.betas)))

    for r in results:
        scores_str = "  ".join(f"{r['scores'][b]:>10.2f}" for b in args.betas)
        print(f"{r['lambda']:>8.4f}  {r['accuracy']*100:>9.2f}%  ${r['cost_1kq']:>9.4f}  {scores_str}")

    # 6. Print per-model distribution for each lambda
    print("\n" + "=" * 100)
    print("PER-MODEL DISTRIBUTION (%)")
    print("=" * 100)

    # Collect all models that appear
    all_models_used = set()
    for r in results:
        all_models_used.update(r["model_counts"].keys())
    all_models_used = sorted(all_models_used)

    # Header
    model_headers = "  ".join(f"{m:>12}" for m in all_models_used)
    print(f"{'λ':>8}  {model_headers}")
    print("-" * (10 + 14 * len(all_models_used)))

    for r in results:
        total = sum(r["model_counts"].values())
        pcts = []
        for m in all_models_used:
            count = r["model_counts"].get(m, 0)
            pcts.append(f"{count/total*100:>11.1f}%")
        print(f"{r['lambda']:>8.4f}  {'  '.join(pcts)}")

    # 7. Find best lambda per beta
    print("\n" + "=" * 100)
    print("BEST LAMBDA PER BETA")
    print("=" * 100)
    print(f"{'Beta':>8}  {'Best λ':>8}  {'Arena':>10}  {'Accuracy':>10}  {'Cost/1kq':>10}")
    print("-" * 55)

    for beta in args.betas:
        best = max(results, key=lambda r: r["scores"][beta])
        print(f"{beta:>8.2f}  {best['lambda']:>8.4f}  {best['scores'][beta]:>10.2f}  "
              f"{best['accuracy']*100:>9.2f}%  ${best['cost_1kq']:>9.4f}")

    total_elapsed = time.time() - start
    print(f"\nTotal time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
