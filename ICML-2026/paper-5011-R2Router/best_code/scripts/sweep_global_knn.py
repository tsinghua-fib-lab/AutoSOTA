#!/usr/bin/env python3
"""
Comprehensive sweep of global KNN routing (matches route_knn_export.py logic).
Optimized: trains KNN once per (model, budget, k), caches predictions, vectorizes routing.
"""
import pickle, json, numpy as np, os, math, warnings, sys, time
from collections import defaultdict
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from category_config import MODELS

t0 = time.time()

# === Load data ===
print("Loading data...", flush=True)
with open('/orange/qi855292.ucf/ah872032.ucf/category_router/training_data.pkl', 'rb') as f:
    data = pickle.load(f)
embeddings = data['embeddings']
categories = data['categories']
models_data = data['models']
global_indices = data['global_indices']

with open('/home/ah872032.ucf/jiaqi/RouterArena/dataset/router_data_10.json') as f:
    sub10 = json.load(f)
sub10_gis = set(e['global index'] for e in sub10)
sub10_mask = np.array([gi in sub10_gis for gi in global_indices])
n = len(global_indices)
train_idx = np.where(sub10_mask)[0]
X_train = embeddings[train_idx]
print(f"  n={n}, train={len(train_idx)}", flush=True)

# Prices
with open('/home/ah872032.ucf/jiaqi/RouterArena/model_cost/model_cost.json') as f:
    cost_data = json.load(f)
prices = {mn: cost_data.get(MODELS[mn]['cost_key'], {}).get('output_token_price_per_million', 0) for mn in MODELS}

# Load sweep costs
def load_sweep_costs(mn, b):
    path = os.path.join(MODELS[mn]['sweep_dir'], f'{b}.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e['global index']: e.get('cost', 0) for e in entries if not e.get('for_optimality')}

ALL_BUDGETS = ['concise', 'budget_10', 'budget_20', 'budget_40', 'budget_80',
               'budget_150', 'budget_200', 'budget_400', 'budget_800']

print("Loading sweep costs...", flush=True)
cost_cache = {}
for mn in models_data:
    if mn not in MODELS:
        continue
    for b in ALL_BUDGETS:
        if MODELS.get(mn, {}).get('type') == 'api' and b != 'concise':
            continue
        if b in models_data[mn]:
            cost_cache[(mn, b)] = load_sweep_costs(mn, b)

# Build cost arrays (n,) for each (model, budget) pair
cost_arrays = {}
for (mn, b), cmap in cost_cache.items():
    arr = np.array([cmap.get(global_indices[i], 0) for i in range(n)])
    cost_arrays[(mn, b)] = arr

# Build accuracy arrays
acc_arrays = {}
for mn in models_data:
    for b in models_data[mn]:
        acc_arrays[(mn, b)] = models_data[mn][b]['accuracy']

def arena_score(acc, c1k, beta=0.1):
    if c1k <= 0:
        c1k = 0.001
    C = (math.log2(200) - math.log2(c1k)) / (math.log2(200) - math.log2(0.0044))
    C = max(0.0, min(1.0, C))
    A = acc
    d = beta * A + C
    return ((1 + beta) * A * C) / d * 100 if d > 0 else 0

# === Pre-train all KNN models and cache predictions ===
ALL_MODELS = sorted(set(m for pool in [
    ['235b', '80b', 'gemini-flash', 'haiku'],
    ['235b', 'gemini-flash', 'haiku'],
    ['235b', '80b', 'gemini-flash'],
    ['235b', 'ministral-3b', 'gemini-flash'],
    ['235b', 'gemini-flash'],
    ['235b', '80b', 'coder-next', 'gemini-flash', 'haiku'],
    ['235b'],
] for m in pool))

K_VALUES = [10, 20, 28, 40, 60]

print(f"Pre-training KNN models for {len(ALL_MODELS)} models x {len(K_VALUES)} k values...", flush=True)

# quality_pred_cache[(mn, budget, k)] = predicted quality array (n,)
quality_pred_cache = {}
# token_pred_cache[(mn, k)] = predicted token array (n,)
token_pred_cache = {}

for mn in ALL_MODELS:
    if mn not in models_data:
        print(f"  WARNING: {mn} not in models_data, skipping", flush=True)
        continue
    for k in K_VALUES:
        # Quality KNN for each budget
        for budget, bdata in models_data[mn].items():
            if budget in ['budget_unlimited', 'budget_1500']:
                continue
            y = bdata['accuracy'][train_idx]
            valid = ~np.isnan(y)
            if valid.sum() < 3:
                continue
            nn = min(k, int(valid.sum()) - 1)
            if nn < 1:
                continue
            knn = KNeighborsRegressor(n_neighbors=nn, metric='cosine', weights='distance')
            knn.fit(X_train[valid], y[valid])
            quality_pred_cache[(mn, budget, k)] = knn.predict(embeddings)

        # Token KNN (from concise budget)
        tb = 'concise' if 'concise' in models_data[mn] else None
        if tb and 'output_tokens' in models_data[mn][tb]:
            y_tok = models_data[mn][tb]['output_tokens'][train_idx]
            valid = ~np.isnan(y_tok)
            if valid.sum() >= 3:
                nn = min(k, int(valid.sum()) - 1)
                if nn >= 1:
                    tknn = KNeighborsRegressor(n_neighbors=nn, metric='cosine', weights='distance')
                    tknn.fit(X_train[valid], y_tok[valid])
                    token_pred_cache[(mn, k)] = np.maximum(1.0, tknn.predict(embeddings))

    print(f"  {mn}: done ({time.time()-t0:.0f}s)", flush=True)

print(f"KNN training complete. {len(quality_pred_cache)} quality, {len(token_pred_cache)} token models. ({time.time()-t0:.0f}s)", flush=True)

# === Vectorized routing ===
def run_global_knn(pool, excluded_budgets, lam, k, use_real_cost=False):
    excl = set(excluded_budgets)
    best_risk = np.full(n, -np.inf)
    best_choice = np.zeros(n, dtype=int)
    choices = []

    for mn in pool:
        if mn not in models_data:
            continue
        price = prices[mn]
        tok = token_pred_cache.get((mn, k), np.full(n, 50.0))

        for budget in models_data[mn]:
            if budget in excl:
                continue
            ck = (mn, budget, k)
            if ck not in quality_pred_cache:
                continue

            q = quality_pred_cache[ck]
            ci = len(choices)
            choices.append((mn, budget))

            if use_real_cost:
                ca = cost_arrays.get((mn, budget), np.zeros(n)).copy()
                miss = ca == 0
                if miss.any():
                    ca[miss] = tok[miss] * price / 1e6
                risk = (1 - lam) * q - lam * ca
            else:
                risk = (1 - lam) * q - lam * tok * price / 1e6

            better = risk > best_risk
            best_risk[better] = risk[better]
            best_choice[better] = ci

    if not choices:
        return 0, 0, 0

    ta = 0.0
    tc = 0.0
    for i in range(n):
        mn, b = choices[best_choice[i]]
        if (mn, b) in acc_arrays:
            ta += acc_arrays[(mn, b)][i]
        if (mn, b) in cost_arrays:
            tc += cost_arrays[(mn, b)][i]

    acc = ta / n
    c1k = tc / n * 1000
    return acc, c1k, arena_score(acc, c1k)


# === Run the sweep ===
print(flush=True)
print('=== Global KNN Sweep (matches route_knn_export.py) ===', flush=True)
print(flush=True)

results = []
pools = {
    '4model': ['235b', '80b', 'gemini-flash', 'haiku'],
    '3model_a': ['235b', 'gemini-flash', 'haiku'],
    '3model_b': ['235b', '80b', 'gemini-flash'],
    '3model_c': ['235b', 'ministral-3b', 'gemini-flash'],
    '2model': ['235b', 'gemini-flash'],
    '5model': ['235b', '80b', 'coder-next', 'gemini-flash', 'haiku'],
    '235b': ['235b'],
}

excl_sets = {
    'no_unlim': ['budget_unlimited', 'budget_1500'],
    'no_small': ['budget_unlimited', 'budget_1500', 'budget_10', 'budget_20'],
    'mid': ['budget_unlimited', 'budget_1500', 'budget_10', 'budget_20', 'budget_400', 'budget_800'],
}

LAMBDAS = [0.80, 0.85, 0.90, 0.95, 0.99, 0.995, 0.999, 0.9995]

cnt = 0
for pool_name, pool in pools.items():
    for excl_name, excl in excl_sets.items():
        for lam in LAMBDAS:
            for k in K_VALUES:
                acc, c1k, arena = run_global_knn(pool, excl, lam, k)
                results.append((pool_name, excl_name, lam, k, acc, c1k, arena, 'token_cost'))
                cnt += 1
    print(f"  {pool_name}: {cnt} configs done ({time.time()-t0:.0f}s)", flush=True)

# Also test with real costs in routing formula
for pool_name, pool in [('4model', ['235b', '80b', 'gemini-flash', 'haiku']),
                         ('3model_c', ['235b', 'ministral-3b', 'gemini-flash'])]:
    for excl_name, excl in excl_sets.items():
        for lam in [0.80, 0.85, 0.90, 0.95, 0.99, 0.995, 0.999]:
            for k in [20, 28, 40, 60]:
                acc, c1k, arena = run_global_knn(pool, excl, lam, k, use_real_cost=True)
                results.append((pool_name, excl_name, lam, k, acc, c1k, arena, 'real_cost'))
                cnt += 1
    print(f"  {pool_name} (real_cost): {cnt} configs total ({time.time()-t0:.0f}s)", flush=True)

results.sort(key=lambda x: -x[6])

print(flush=True)
print('Top 30 by Arena score:', flush=True)
seen = set()
for pool_name, excl, lam, k, acc, c1k, arena, cost_type in results[:60]:
    key = (pool_name, excl, lam, k, cost_type)
    if key in seen:
        continue
    seen.add(key)
    marker = ''
    if acc > 0.72 and c1k < 0.05:
        marker = ' *** TARGET ***'
    elif acc > 0.71 and c1k < 0.06:
        marker = ' * close *'
    print(f'  {pool_name:<12} {excl:<12} lam={lam:.4f} k={k:>2} {cost_type:<10} '
          f'Acc={acc*100:.2f}% Cost=${c1k:.4f} Arena={arena:.2f}{marker}', flush=True)
    if len(seen) >= 30:
        break

print(flush=True)
print('Target check (Acc>72% AND Cost<$0.05):', flush=True)
target = [(p, e, l, k, a, c, ar, ct) for p, e, l, k, a, c, ar, ct in results if a > 0.72 and c < 0.05]
if target:
    for p, e, l, k, a, c, ar, ct in sorted(target, key=lambda x: -x[6])[:10]:
        print(f'  {p:<12} {e:<12} lam={l:.4f} k={k:>2} {ct:<10} '
              f'Acc={a*100:.2f}% Cost=${c:.4f} Arena={ar:.2f}', flush=True)
else:
    print('  NONE FOUND', flush=True)
    print(flush=True)
    print('Closest to target:', flush=True)
    print('  Best acc with cost < $0.05:', flush=True)
    cheap = [(p, e, l, k, a, c, ar, ct) for p, e, l, k, a, c, ar, ct in results if c < 0.05]
    cheap.sort(key=lambda x: -x[4])
    for p, e, l, k, a, c, ar, ct in cheap[:5]:
        print(f'    {p:<12} {e:<12} lam={l:.4f} k={k:>2} {ct:<10} '
              f'Acc={a*100:.2f}% Cost=${c:.4f} Arena={ar:.2f}', flush=True)

    print('  Best acc with cost < $0.06:', flush=True)
    mid = [(p, e, l, k, a, c, ar, ct) for p, e, l, k, a, c, ar, ct in results if c < 0.06]
    mid.sort(key=lambda x: -x[4])
    for p, e, l, k, a, c, ar, ct in mid[:5]:
        print(f'    {p:<12} {e:<12} lam={l:.4f} k={k:>2} {ct:<10} '
              f'Acc={a*100:.2f}% Cost=${c:.4f} Arena={ar:.2f}', flush=True)

    print('  Best arena overall:', flush=True)
    for p, e, l, k, a, c, ar, ct in results[:5]:
        print(f'    {p:<12} {e:<12} lam={l:.4f} k={k:>2} {ct:<10} '
              f'Acc={a*100:.2f}% Cost=${c:.4f} Arena={ar:.2f}', flush=True)

print(flush=True)

# === Model distribution for top configs ===
print('=== Model/Budget distribution for top 5 configs ===', flush=True)
seen2 = set()
for pool_name, excl, lam, k, acc, c1k, arena, cost_type in results[:20]:
    key = (pool_name, excl, lam, k, cost_type)
    if key in seen2:
        continue
    seen2.add(key)
    pool = pools.get(pool_name, [pool_name])
    excl_list = excl_sets.get(excl, [])
    excl_s = set(excl_list)
    choices = []
    best_risk = np.full(n, -np.inf)
    best_choice = np.zeros(n, dtype=int)
    for mn in pool:
        if mn not in models_data: continue
        price = prices[mn]
        tok = token_pred_cache.get((mn, k), np.full(n, 50.0))
        for budget in models_data[mn]:
            if budget in excl_s: continue
            ck = (mn, budget, k)
            if ck not in quality_pred_cache: continue
            q = quality_pred_cache[ck]
            ci = len(choices)
            choices.append((mn, budget))
            if cost_type == 'real_cost':
                ca = cost_arrays.get((mn, budget), np.zeros(n)).copy()
                miss = ca == 0
                if miss.any(): ca[miss] = tok[miss] * price / 1e6
                risk = (1 - lam) * q - lam * ca
            else:
                risk = (1 - lam) * q - lam * tok * price / 1e6
            better = risk > best_risk
            best_risk[better] = risk[better]
            best_choice[better] = ci

    model_counts = defaultdict(int)
    budget_counts = defaultdict(int)
    for i in range(n):
        mn, b = choices[best_choice[i]]
        model_counts[mn] += 1
        budget_counts[b] += 1

    print(f'\n  Config: {pool_name} {excl} lam={lam} k={k} {cost_type} => Arena={arena:.2f}', flush=True)
    print(f'    Models: ', end='', flush=True)
    for mn, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f'{mn}={cnt/n*100:.1f}% ', end='')
    print(flush=True)
    print(f'    Budgets: ', end='', flush=True)
    for b, cnt in sorted(budget_counts.items(), key=lambda x: -x[1]):
        print(f'{b}={cnt/n*100:.1f}% ', end='')
    print(flush=True)

    if len(seen2) >= 5:
        break

# === Compare global KNN vs per-category KNN ===
print(flush=True)
print('=== Global vs Per-Category KNN comparison ===', flush=True)
best_global = results[0]
print(f'Best Global KNN: {best_global[0]} {best_global[1]} lam={best_global[2]} k={best_global[3]} '
      f'=> Acc={best_global[4]*100:.2f}% Cost=${best_global[5]:.4f} Arena={best_global[6]:.2f}', flush=True)

# Per-category KNN
CATEGORY_NAMES = ['Code', 'Math', 'Knowledge', 'NLU', 'Translation', 'Trivia', 'Domain']
pool_cat = ['235b', '80b', 'gemini-flash', 'haiku']
budgets_cat = ['concise', 'budget_40', 'budget_80', 'budget_150', 'budget_200', 'budget_400', 'budget_800']
lam_cat = 0.9995
k_cat = 60
sk = 3.0

pred_quality = {}
for mn in pool_cat:
    for b in budgets_cat:
        if b not in models_data.get(mn, {}):
            continue
        if MODELS[mn]['type'] == 'api' and b != 'concise':
            continue
        pred_q = np.zeros(n)
        for cat_idx in range(7):
            cat_all = np.where(categories == cat_idx)[0]
            cat_train = np.array([i for i in cat_all if sub10_mask[i]])
            if len(cat_train) < 3:
                pred_q[cat_all] = models_data[mn][b]['accuracy'][cat_train].mean() if len(cat_train) > 0 else 0
                continue
            y = models_data[mn][b]['accuracy'][cat_train]
            X = embeddings[cat_train]
            kk = min(k_cat, len(cat_train) - 1)
            reg = KNeighborsRegressor(n_neighbors=kk, metric='cosine', weights='distance')
            reg.fit(X, y)
            pred_q[cat_all] = reg.predict(embeddings[cat_all])
            cm = y.mean()
            alpha = min(1.0, max(0.0, 0.09 * sk))
            pred_q[cat_all] = alpha * pred_q[cat_all] + (1 - alpha) * cm
        pred_quality[(mn, b)] = pred_q

mean_tokens = {}
for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
    ct = (categories == cat_idx) & sub10_mask
    mean_tokens[cat_name] = {}
    for mn in pool_cat:
        if 'concise' in models_data.get(mn, {}):
            mean_tokens[cat_name][mn] = float(max(1, models_data[mn]['concise']['output_tokens'][ct].mean()))
        else:
            mean_tokens[cat_name][mn] = 50.0

routes_cat = [None] * n
for i in range(n):
    cat_name = CATEGORY_NAMES[categories[i]]
    best_risk_val = -np.inf
    best_route = None
    for mn in pool_cat:
        tok = mean_tokens.get(cat_name, {}).get(mn, 50.0)
        for b in budgets_cat:
            if (mn, b) not in pred_quality:
                continue
            q = pred_quality[(mn, b)][i]
            risk = (1 - lam_cat) * q - lam_cat * tok * prices[mn] / 1e6
            if risk > best_risk_val:
                best_risk_val = risk
                best_route = (mn, b)
    routes_cat[i] = best_route

ta = tc = 0
for i, (mn, b) in enumerate(routes_cat):
    ta += models_data[mn][b]['accuracy'][i] if mn in models_data and b in models_data[mn] else 0
    tc += cost_arrays.get((mn, b), np.zeros(n))[i]
cat_acc = ta / n
cat_c1k = tc / n * 1000
print(f'Best Per-Cat KNN: lam=0.9995 k=60 sk=3.0 => Acc={cat_acc*100:.2f}% Cost=${cat_c1k:.4f} '
      f'Arena={arena_score(cat_acc, cat_c1k):.2f}', flush=True)

print(f'\nTotal time: {time.time()-t0:.0f}s', flush=True)
