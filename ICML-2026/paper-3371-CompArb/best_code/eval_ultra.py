#!/usr/bin/env python3
"""Ultra-refined evaluation: 750 budget pts, 3000 perf pts, 3-stage tau search."""

import json, os, sys
import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebooks"))
from utils import (
    load_pass_curves, threshold_to_profit_batch,
    cost_for_pass, get_arbitrage_prices,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "swebench")
MODELS = ["mini-coder-4b", "qwen3-coder-30b", "gpt5mini", "deepseek"]
MODEL_DISPLAY = {
    "mini-coder-4b": "Mini-Coder 4B", "qwen3-coder-30b": "Qwen3-Coder 30B",
    "gpt5mini": "GPT-5 mini", "deepseek": "DeepSeek v3.2",
}
BUDGET_MIN = 1e-4
BUDGET_MAX = 1.0
BUDGET_POINTS = 750
PERFORMANCE_RANGE = (0.68, None)
PERFORMANCE_POINTS = 3000
SCALE_FACTOR = 500
N_MODELS = len(MODELS)

results = {}
for fname in os.listdir(BASE_DIR):
    if not fname.endswith(".jsonl"): continue
    model_name = fname.split(".jsonl")[0]
    data = [json.loads(line) for line in open(os.path.join(BASE_DIR, fname), "r")]
    results[model_name] = data

budget_grid = np.logspace(np.log10(BUDGET_MIN), np.log10(BUDGET_MAX), BUDGET_POINTS)
providers_perf = load_pass_curves(results, MODELS, budget_grid)
n_problems = providers_perf.shape[0]

max_perf = providers_perf.mean(axis=0).max()
perf_min = PERFORMANCE_RANGE[0]
perf_max = PERFORMANCE_RANGE[1] if PERFORMANCE_RANGE[1] is not None else max_perf
performance_grid = jnp.linspace(perf_min, perf_max, PERFORMANCE_POINTS)

# Stage 1: Wide but sparser search
print("Stage 1: Wide coarse search")
N1 = 61
c1 = np.r_[0, np.logspace(np.log10(0.001), np.log10(BUDGET_MAX), N1)]
c1 = np.unique(np.clip(c1, 0, BUDGET_MAX))
rows = []
for t0 in c1:
    remain1 = c1[c1 >= t0]
    # Subsample to keep grid manageable
    if len(remain1) > 30:
        idx = np.linspace(0, len(remain1)-1, 30).astype(int)
        remain1 = remain1[idx]
    for t1 in remain1:
        remain2 = c1[c1 >= t1]
        if len(remain2) > 20:
            idx = np.linspace(0, len(remain2)-1, 20).astype(int)
            remain2 = remain2[idx]
        for t2 in remain2:
            rows.append([t0, t1, t2, BUDGET_MAX])
g1 = np.array(rows)
print("  Grid size: " + str(g1.shape[0]))
if g1.shape[0] > 100000:
    print("  Grid too large, subsampling to 80000 random rows")
    np.random.seed(42)
    idx = np.random.choice(g1.shape[0], 80000, replace=False)
    g1 = g1[idx]
    print("  Subsampled to: " + str(g1.shape[0]))

p1 = threshold_to_profit_batch(budget_grid, providers_perf, jnp.array(g1), performance_grid).mean(axis=-1)
b1 = g1[int(np.argmax(p1))]
print("  Best: tau=(" + str(round(b1[0],4)) + "," + str(round(b1[1],4)) + "," + str(round(b1[2],4)) + ")")

# Stage 2: Focused refinement
print("Stage 2: Focused refinement")
N2 = 31
s2 = 0.3
t0r = np.linspace(max(0.001, b1[0]*(1-s2)), b1[0]*(1+s2), N2)
t1r = np.linspace(max(0.001, b1[1]*(1-s2)), b1[1]*(1+s2), N2)
t2r = np.linspace(max(0.001, b1[2]*(1-s2)), b1[2]*(1+s2), N2)
rows = []
for t0 in t0r:
    for t1 in t1r:
        if t1 < t0: continue
        for t2 in t2r:
            if t2 < t1: continue
            if t2 > BUDGET_MAX: continue
            rows.append([t0, t1, t2, BUDGET_MAX])
g2 = np.array(rows)
print("  Grid size: " + str(g2.shape[0]))
p2 = threshold_to_profit_batch(budget_grid, providers_perf, jnp.array(g2), performance_grid).mean(axis=-1)
b2 = g2[int(np.argmax(p2))]
print("  Best: tau=(" + str(round(b2[0],4)) + "," + str(round(b2[1],4)) + "," + str(round(b2[2],4)) + ")")

# Stage 3: Ultra-fine
print("Stage 3: Ultra-fine refinement")
N3 = 31
s3 = 0.08
t0r3 = np.linspace(max(0.001, b2[0]*(1-s3)), b2[0]*(1+s3), N3)
t1r3 = np.linspace(max(0.001, b2[1]*(1-s3)), b2[1]*(1+s3), N3)
t2r3 = np.linspace(max(0.001, b2[2]*(1-s3)), b2[2]*(1+s3), N3)
rows = []
for t0 in t0r3:
    for t1 in t1r3:
        if t1 < t0: continue
        for t2 in t2r3:
            if t2 < t1: continue
            if t2 > BUDGET_MAX: continue
            rows.append([t0, t1, t2, BUDGET_MAX])
g3 = np.array(rows)
print("  Grid size: " + str(g3.shape[0]))
p3 = threshold_to_profit_batch(budget_grid, providers_perf, jnp.array(g3), performance_grid).mean(axis=-1)
optimal_allocation = g3[int(np.argmax(p3))]
print("  Best: tau=(" + str(round(optimal_allocation[0],4)) + "," + str(round(optimal_allocation[1],4)) + "," + str(round(optimal_allocation[2],4)) + ")")

# Final metrics
arbitrage_cost, arbitrage_expend = get_arbitrage_prices(
    budget_grid, providers_perf, jnp.array(optimal_allocation), performance_grid
)
provider_prices = np.array([
    cost_for_pass(budget_grid, p, performance_grid)
    for p in providers_perf.mean(axis=0)
])
market_price = np.min(provider_prices, axis=0)
arbitrage_profit_margin = np.maximum(0, market_price - arbitrage_cost) / market_price

target_solve = 0.75
idx_75 = np.argmin(np.abs(np.array(performance_grid) - target_solve))
actual_solve_rate = float(performance_grid[idx_75])
arb_cost_75 = float(arbitrage_cost[idx_75]) * SCALE_FACTOR
profit_margin_75 = float(arbitrage_profit_margin[idx_75]) * 100

max_margin_idx = int(np.argmax(arbitrage_profit_margin))
max_profit_margin = float(arbitrage_profit_margin[max_margin_idx]) * 100
max_profit_solve = float(performance_grid[max_margin_idx]) * 100

print("")
print("=" * 60)
print("ULTRA-REFINED - " + str(N_MODELS) + "-MODEL CASCADE")
print("=" * 60)
print("Cascade: " + " -> ".join(MODEL_DISPLAY.get(m,m) for m in MODELS))
print("Problems: " + str(n_problems))
tau_s = ", ".join(str(round(v,4)) for v in optimal_allocation)
print("Optimal tau: (" + tau_s + ")")
print("Max solve rate: " + str(round(max_perf*100,1)) + "%")
print("Budget pts: " + str(BUDGET_POINTS) + ", Perf pts: " + str(PERFORMANCE_POINTS))
print("")
print("--- Cost at " + str(int(target_solve*100)) + "% (actual: " + str(round(actual_solve_rate*100,1)) + "%) ---")
print("  Arbitrageur cost:      $" + str(round(arb_cost_75,1)))
for i, m in enumerate(MODELS):
    raw = float(provider_prices[i, idx_75])
    c = raw * SCALE_FACTOR if not np.isinf(raw) else float("inf")
    label = MODEL_DISPLAY.get(m, m)
    if c == float("inf"):
        print("  " + label + " cost: inf")
    else:
        print("  " + label + " cost: $" + str(round(c,1)))
mkt = round(float(market_price[idx_75])*SCALE_FACTOR,1)
print("  Market price (min):    $" + str(mkt))
print("  Profit margin at 75%:  " + str(round(profit_margin_75,1)) + "%")
print("")
print("--- Profit Margin ---")
print("  Max profit margin:     " + str(round(max_profit_margin,1)) + "%")
print("  At solve rate:         " + str(round(max_profit_solve,1)) + "%")

metrics_json = {
    "paper_id": 3371, "n_problems": n_problems, "n_models": N_MODELS,
    "models": MODELS,
    "optimal_tau": [float(v) for v in optimal_allocation],
    "max_solve_rate": round(float(max_perf)*100,2),
    "cost_at_75": {
        "target_solve_rate_pct": round(actual_solve_rate*100,1),
        "arbitrageur": round(arb_cost_75,1),
        "market_price": mkt,
    },
    "profit_margin": {
        "max_pct": round(max_profit_margin,1),
        "at_solve_rate_pct": round(max_profit_solve,1),
    },
    "scale_factor": SCALE_FACTOR,
    "budget_points": BUDGET_POINTS,
    "performance_points": PERFORMANCE_POINTS,
}
for i, m in enumerate(MODELS):
    raw = float(provider_prices[i, idx_75])
    metrics_json["cost_at_75"][m] = round(raw*SCALE_FACTOR,1) if not np.isinf(raw) else "inf"

print("")
print("JSON_METRICS:", json.dumps(metrics_json))
