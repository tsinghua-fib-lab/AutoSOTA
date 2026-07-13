#!/usr/bin/env python3
"""
Efficient parameter sweep: loads all data once, then sweeps parameters.
Uses the existing route_and_eval functions directly.
"""
import os, sys, json, pickle, math, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, "/home/ah872032.ucf/jiaqi/router/scripts")
from category_config import (
    MODELS, CHECKPOINT_DIR, CATEGORY_NAMES,
    TRAINING_DATA_PATH, MODEL_COST_PATH, ROUTER_DATA_PATH,
)
from route_and_eval import (
    load_predictors, load_prices, route, load_sweep_entry, arena_score,
)

# ============================================================================
# Load everything once
# ============================================================================
print("Loading training data...", flush=True)
t0 = time.time()
with open(TRAINING_DATA_PATH, "rb") as f:
    data = pickle.load(f)
embeddings = data["embeddings"]
categories = data["categories"]
models_data = data["models"]
global_indices = data["global_indices"]
n = embeddings.shape[0]
print(f"  {n} queries loaded in {time.time()-t0:.1f}s", flush=True)

print("Loading predictors...", flush=True)
quality_preds, token_preds, confidence = load_predictors()
prices = load_prices()

# Load train/test split
with open(os.path.join(CHECKPOINT_DIR, "train_test_split.pkl"), "rb") as f:
    split = pickle.load(f)
test_idx = split["test_idx"]
train_idx = split["train_idx"]
test_set = set(test_idx.tolist())
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}", flush=True)

# Precompute category means from FULL data (as route_and_eval does)
mean_acc_full = {}
mean_tokens_full = {}
for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
    cat_mask = np.where(categories == cat_idx)[0]
    if len(cat_mask) == 0:
        continue
    mean_acc_full[cat_name] = {}
    mean_tokens_full[cat_name] = {}
    for mn, budgets_data in models_data.items():
        mean_acc_full[cat_name][mn] = {}
        for budget, bdata in budgets_data.items():
            mean_acc_full[cat_name][mn][budget] = float(bdata["accuracy"][cat_mask].mean())
        if "concise" in budgets_data:
            mean_tokens_full[cat_name][mn] = float(
                max(1.0, budgets_data["concise"]["output_tokens"][cat_mask].mean())
            )
        else:
            mean_tokens_full[cat_name][mn] = 50.0

# Also compute means from train-only (sub_10)
mean_acc_train = {}
mean_tokens_train = {}
for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
    cat_mask = np.where(categories == cat_idx)[0]
    cat_train = np.array([i for i in cat_mask if i in set(train_idx.tolist())])
    if len(cat_train) == 0:
        continue
    mean_acc_train[cat_name] = {}
    mean_tokens_train[cat_name] = {}
    for mn, budgets_data in models_data.items():
        mean_acc_train[cat_name][mn] = {}
        for budget, bdata in budgets_data.items():
            mean_acc_train[cat_name][mn][budget] = float(bdata["accuracy"][cat_train].mean())
        if "concise" in budgets_data:
            mean_tokens_train[cat_name][mn] = float(
                max(1.0, budgets_data["concise"]["output_tokens"][cat_train].mean())
            )
        else:
            mean_tokens_train[cat_name][mn] = 50.0

# Pre-load ALL sweep data for evaluation (this is the bottleneck)
print("Pre-loading sweep data...", flush=True)
t0 = time.time()
sweep_cache = {}
for mn in MODELS:
    from category_config import get_budgets
    for budget in get_budgets(mn):
        key = (mn, budget)
        sweep_cache[key] = load_sweep_entry(mn, budget)
print(f"  Loaded {len(sweep_cache)} sweep files in {time.time()-t0:.1f}s", flush=True)

# ============================================================================
# Evaluation function (fast, uses pre-loaded data)
# ============================================================================
def fast_evaluate(routes, eval_indices=None):
    """Evaluate routes against pre-loaded sweep data."""
    if eval_indices is not None:
        idx_list = sorted(eval_indices)
        eval_routes = [routes[i] for i in idx_list]
        eval_gi = [global_indices[i] for i in idx_list]
    else:
        eval_routes = routes
        eval_gi = global_indices
    
    n_eval = len(eval_routes)
    total_acc = 0.0
    total_cost = 0.0
    found = 0
    model_counts = defaultdict(int)
    budget_counts = defaultdict(int)
    
    for i, (mn, budget) in enumerate(eval_routes):
        if mn is None:
            continue
        gi = eval_gi[i]
        model_counts[mn] += 1
        budget_counts[budget] += 1
        
        entry = sweep_cache.get((mn, budget), {}).get(gi)
        if entry is None:
            continue
        
        acc = entry.get("accuracy")
        cost = entry.get("cost")
        if acc is not None:
            total_acc += float(acc)
            found += 1
        if cost is not None:
            total_cost += float(cost)
    
    accuracy = total_acc / n_eval if n_eval > 0 else 0
    cost_1kq = total_cost / n_eval * 1000 if n_eval > 0 else 0
    score = arena_score(accuracy, cost_1kq)
    
    return accuracy, cost_1kq, score, model_counts, budget_counts, found

# ============================================================================
# Helper: run one configuration
# ============================================================================
def run_config(lam, sk, allowed_models=None, excluded_budgets=None,
               force_mean_tokens=False, use_train_means=False):
    """Run routing + evaluation for one parameter config."""
    if allowed_models is None:
        am = set(MODELS.keys())
    else:
        am = set(allowed_models)
    eb = set(excluded_budgets) if excluded_budgets else set()
    
    ma = mean_acc_train if use_train_means else mean_acc_full
    mt = mean_tokens_train if use_train_means else mean_tokens_full
    
    routes = route(embeddings, categories, quality_preds, token_preds, confidence,
                   prices, lam, sk, ma, mt,
                   allowed_models=am, excluded_budgets=eb,
                   force_mean_tokens=force_mean_tokens)
    
    acc, cost, score, mc, bc, found = fast_evaluate(routes, test_set)
    return acc, cost, score, mc, bc, found

results = []

# ============================================================================
# Experiment 1: Lambda sweep (all models, all budgets, shrinkage_k=0)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 1: Lambda sweep (shrinkage_k=0, all models, all budgets)", flush=True)
print("=" * 80, flush=True)

for lam in [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0]:
    acc, cost, score, mc, bc, found = run_config(lam, 0)
    n_models = len(mc)
    top_model = max(mc, key=mc.get) if mc else "?"
    top_pct = mc[top_model] / sum(mc.values()) * 100 if mc else 0
    top_budget = max(bc, key=bc.get) if bc else "?"
    print(f"  lam={lam:<8} Acc={acc*100:6.2f}%  Cost=${cost:8.4f}  Arena={score:6.2f}  "
          f"models={n_models:>2}  top={top_model}({top_pct:.0f}%)  budget={top_budget}", flush=True)
    results.append({
        "exp": "1_lambda", "lambda": lam, "sk": 0,
        "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
        "n_models": n_models, "top_model": top_model,
    })

# ============================================================================
# Experiment 2: Shrinkage sweep (lambda=0.999, all models)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 2: Shrinkage_k sweep (lambda=0.999, all models)", flush=True)
print("=" * 80, flush=True)

for sk in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    acc, cost, score, mc, bc, found = run_config(0.999, sk)
    top_model = max(mc, key=mc.get) if mc else "?"
    top_pct = mc[top_model] / sum(mc.values()) * 100 if mc else 0
    print(f"  sk={sk:<8} Acc={acc*100:6.2f}%  Cost=${cost:8.4f}  Arena={score:6.2f}  "
          f"top={top_model}({top_pct:.0f}%)", flush=True)
    results.append({
        "exp": "2_shrinkage", "lambda": 0.999, "sk": sk,
        "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
    })

# Also test with train means only
print("\n  --- With train-only means ---", flush=True)
for sk in [0.0, 3.0, 10.0, 50.0]:
    acc, cost, score, mc, bc, found = run_config(0.999, sk, use_train_means=True)
    print(f"  sk={sk:<8} Acc={acc*100:6.2f}%  Cost=${cost:8.4f}  Arena={score:6.2f}  (train means)", flush=True)
    results.append({
        "exp": "2b_shrinkage_train_means", "lambda": 0.999, "sk": sk,
        "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
        "note": "train_means_only",
    })

# ============================================================================
# Experiment 3: Lambda x Shrinkage grid
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 3: Lambda x Shrinkage grid (Arena scores)", flush=True)
print("=" * 80, flush=True)

lambdas = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
shrinkages = [0, 1.0, 3.0, 5.0, 10.0, 20.0]

header = f"{'':>14}"
for sk in shrinkages:
    header += f"  sk={sk:<7}"
print(header, flush=True)

for lam in lambdas:
    row = f"  lam={lam:<8}"
    for sk in shrinkages:
        acc, cost, score, mc, bc, found = run_config(lam, sk)
        row += f"  {score:7.2f}"
        results.append({
            "exp": "3_grid", "lambda": lam, "sk": sk,
            "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
        })
    print(row, flush=True)

# ============================================================================
# Experiment 4: Model subset ablation (lambda=0.999, shrinkage_k=0)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 4: Model subset ablation (lambda=0.999, shrinkage_k=0)", flush=True)
print("=" * 80, flush=True)

model_subsets = {
    "all_16": None,
    "feb10_4 (235b,80b,flash,haiku)": ["235b", "80b", "gemini-flash", "haiku"],
    "vllm_9": ["235b", "80b", "30b", "coder-next", "coder-30b",
                "ministral-3b", "ministral-8b", "ministral-14b", "gemma-3n-e4b"],
    "cheap_vllm_7": ["30b", "coder-next", "coder-30b",
                     "ministral-3b", "ministral-8b", "ministral-14b", "gemma-3n-e4b"],
    "top5 (235b,80b,coder-next,30b,gpt4o)": ["235b", "80b", "coder-next", "30b", "gpt4o"],
    "235b_only": ["235b"],
    "coder-next_only": ["coder-next"],
    "gpt4o_only": ["gpt4o"],
    "gpt-5.2_only": ["gpt-5.2"],
    "235b+30b": ["235b", "30b"],
    "235b+coder-next": ["235b", "coder-next"],
    "235b+ministral-3b": ["235b", "ministral-3b"],
    "235b+gemma-3n-e4b": ["235b", "gemma-3n-e4b"],
    "big3 (235b,gpt4o,gpt-5.2)": ["235b", "gpt4o", "gpt-5.2"],
    "diverse (235b,coder-next,ministral-8b,flash,gpt4o)": 
        ["235b", "coder-next", "ministral-8b", "gemini-flash", "gpt4o"],
    "new_models (gpt-5.2,gemini-3-pro,glm-5)": ["gpt-5.2", "gemini-3-pro", "glm-5"],
    "qwen_family (235b,80b,30b,coder-next,coder-30b)": 
        ["235b", "80b", "30b", "coder-next", "coder-30b"],
    "ministral_family": ["ministral-3b", "ministral-8b", "ministral-14b"],
}

for name, models in model_subsets.items():
    acc, cost, score, mc, bc, found = run_config(0.999, 0, allowed_models=models)
    top = max(mc, key=mc.get) if mc else "?"
    print(f"  {name:<45} Acc={acc*100:6.2f}%  Cost=${cost:8.4f}  Arena={score:6.2f}  top={top}", flush=True)
    results.append({
        "exp": "4_model_ablation", "subset": name,
        "lambda": 0.999, "sk": 0,
        "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
    })

# ============================================================================
# Experiment 5: Budget exclusion ablation
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 5: Budget exclusion (lambda=0.999, shrinkage_k=0, all models)", flush=True)
print("=" * 80, flush=True)

budget_configs = {
    "all_budgets": [],
    "no_unlimited_1500": ["budget_unlimited", "budget_1500"],
    "concise_only": ["budget_10", "budget_20", "budget_40", "budget_80",
                      "budget_150", "budget_200", "budget_400", "budget_800",
                      "budget_1500", "budget_unlimited"],
    "concise+small (10-80)": ["budget_150", "budget_200", "budget_400", "budget_800",
                               "budget_1500", "budget_unlimited"],
    "concise+med (10-200)": ["budget_400", "budget_800", "budget_1500", "budget_unlimited"],
    "large_only (400-unlim)": ["budget_10", "budget_20", "budget_40", "budget_80",
                                "budget_150", "budget_200", "concise"],
    "no_concise": ["concise"],
}

for name, excluded in budget_configs.items():
    acc, cost, score, mc, bc, found = run_config(0.999, 0, excluded_budgets=excluded)
    top_bud = max(bc, key=bc.get) if bc else "?"
    top_pct = bc[top_bud] / sum(bc.values()) * 100 if bc else 0
    print(f"  {name:<30} Acc={acc*100:6.2f}%  Cost=${cost:8.4f}  Arena={score:6.2f}  top_budget={top_bud}({top_pct:.0f}%)", flush=True)
    results.append({
        "exp": "5_budget_ablation", "config": name,
        "lambda": 0.999, "sk": 0,
        "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
    })

# ============================================================================
# Experiment 6: force-mean-tokens effect
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 6: Token predictor vs category mean", flush=True)
print("=" * 80, flush=True)

for lam in [0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
    acc_pred, cost_pred, score_pred, _, _, _ = run_config(lam, 0, force_mean_tokens=False)
    acc_mean, cost_mean, score_mean, _, _, _ = run_config(lam, 0, force_mean_tokens=True)
    diff = score_mean - score_pred
    winner = "MEAN" if diff > 0 else "PRED"
    print(f"  lam={lam:<8} predictor: Arena={score_pred:6.2f}  mean: Arena={score_mean:6.2f}  "
          f"diff={diff:+.2f} ({winner})", flush=True)
    results.append({
        "exp": "6_token_pred", "lambda": lam,
        "arena_pred": round(score_pred, 2), "arena_mean": round(score_mean, 2),
        "diff": round(diff, 2),
    })

# ============================================================================
# Experiment 7: Feb10 baseline variants
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 7: Feb10 baseline and variants", flush=True)
print("=" * 80, flush=True)

feb10_configs = [
    {"name": "feb10_exact", "lam": 0.999, "sk": 0,
     "models": ["235b", "80b", "gemini-flash", "haiku"],
     "exclude": ["budget_unlimited", "budget_1500"]},
    {"name": "feb10+shrink3", "lam": 0.999, "sk": 3.0,
     "models": ["235b", "80b", "gemini-flash", "haiku"],
     "exclude": ["budget_unlimited", "budget_1500"]},
    {"name": "feb10+shrink10", "lam": 0.999, "sk": 10.0,
     "models": ["235b", "80b", "gemini-flash", "haiku"],
     "exclude": ["budget_unlimited", "budget_1500"]},
    {"name": "feb10+lam0.99", "lam": 0.99, "sk": 0,
     "models": ["235b", "80b", "gemini-flash", "haiku"],
     "exclude": ["budget_unlimited", "budget_1500"]},
    {"name": "feb10+lam0.995", "lam": 0.995, "sk": 0,
     "models": ["235b", "80b", "gemini-flash", "haiku"],
     "exclude": ["budget_unlimited", "budget_1500"]},
    {"name": "all+noulim", "lam": 0.999, "sk": 0,
     "models": None, "exclude": ["budget_unlimited", "budget_1500"]},
    {"name": "all+concise", "lam": 0.999, "sk": 0,
     "models": None,
     "exclude": ["budget_10", "budget_20", "budget_40", "budget_80",
                  "budget_150", "budget_200", "budget_400", "budget_800",
                  "budget_1500", "budget_unlimited"]},
    {"name": "feb10+concise", "lam": 0.999, "sk": 0,
     "models": ["235b", "80b", "gemini-flash", "haiku"],
     "exclude": ["budget_10", "budget_20", "budget_40", "budget_80",
                  "budget_150", "budget_200", "budget_400", "budget_800",
                  "budget_1500", "budget_unlimited"]},
]

for cfg in feb10_configs:
    acc, cost, score, mc, bc, found = run_config(
        cfg["lam"], cfg["sk"],
        allowed_models=cfg["models"],
        excluded_budgets=cfg["exclude"])
    top = max(mc, key=mc.get) if mc else "?"
    top_pct = mc[top] / sum(mc.values()) * 100 if mc else 0
    print(f"  {cfg['name']:<25} Acc={acc*100:6.2f}%  Cost=${cost:8.4f}  Arena={score:6.2f}  "
          f"top={top}({top_pct:.0f}%)", flush=True)
    results.append({
        "exp": "7_feb10_variants", "name": cfg["name"],
        "lambda": cfg["lam"], "sk": cfg["sk"],
        "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
    })

# ============================================================================
# Experiment 8: Oracle baselines (single best model+budget for all queries)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("EXPERIMENT 8: Single-model baselines (all queries -> one model, concise)", flush=True)
print("=" * 80, flush=True)

for mn in sorted(MODELS.keys()):
    acc, cost, score, mc, bc, found = run_config(0.999, 0, allowed_models=[mn],
                                                  excluded_budgets=["budget_10", "budget_20", "budget_40", "budget_80",
                                                                    "budget_150", "budget_200", "budget_400", "budget_800",
                                                                    "budget_1500", "budget_unlimited"])
    print(f"  {mn:<20} Acc={acc*100:6.2f}%  Cost=${cost:8.4f}  Arena={score:6.2f}", flush=True)
    results.append({
        "exp": "8_single_model", "model": mn,
        "acc": round(acc*100, 2), "cost": round(cost, 4), "arena": round(score, 2),
    })

# ============================================================================
# Summary: Top 20 configurations by arena score
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("TOP 20 CONFIGURATIONS BY ARENA SCORE", flush=True)
print("=" * 80, flush=True)

# Filter to only grid/sweep experiments (not single-model baselines)
sweep_results = [r for r in results if "arena" in r and r["exp"] not in ["8_single_model", "6_token_pred"]]
sorted_results = sorted(sweep_results, key=lambda x: -x["arena"])
for i, r in enumerate(sorted_results[:20]):
    extra = ""
    if "subset" in r:
        extra = f" [{r['subset']}]"
    elif "config" in r:
        extra = f" [{r['config']}]"
    elif "name" in r:
        extra = f" [{r['name']}]"
    elif "note" in r:
        extra = f" [{r['note']}]"
    print(f"  {i+1:>2}. Arena={r['arena']:6.2f}  Acc={r['acc']:6.2f}%  Cost=${r['cost']:8.4f}  "
          f"lam={r.get('lambda','?'):<8} sk={r.get('sk','?'):<6} "
          f"{r['exp']}{extra}", flush=True)

# Also show top by accuracy
print("\n" + "=" * 80, flush=True)
print("TOP 10 BY ACCURACY", flush=True)
print("=" * 80, flush=True)
sorted_by_acc = sorted(sweep_results, key=lambda x: -x["acc"])
for i, r in enumerate(sorted_by_acc[:10]):
    extra = ""
    if "subset" in r:
        extra = f" [{r['subset']}]"
    elif "name" in r:
        extra = f" [{r['name']}]"
    print(f"  {i+1:>2}. Acc={r['acc']:6.2f}%  Arena={r['arena']:6.2f}  Cost=${r['cost']:8.4f}  "
          f"lam={r.get('lambda','?'):<8} sk={r.get('sk','?'):<6} "
          f"{r['exp']}{extra}", flush=True)

# Save
output_path = "/home/ah872032.ucf/jiaqi/router/experiments/sub10_sweep_results.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to {output_path}", flush=True)
