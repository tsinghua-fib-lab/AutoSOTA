#!/usr/bin/env python3
"""Refined evaluation with finer grids and local tau search near optimum."""

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
BUDGET_POINTS = 500
PERFORMANCE_RANGE = (0.68, None)
PERFORMANCE_POINTS = 2000
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

# Stage 1: Coarse grid
TAU_COARSE = 50
coarse_candidates = np.r_[0, np.logspace(np.log10(0.001), np.log10(BUDGET_MAX), TAU_COARSE)]
coarse_candidates = np.unique(np.clip(coarse_candidates, 0, BUDGET_MAX))

rows = []
for t0 in coarse_candidates:
    for t1 in coarse_candidates[coarse_candidates >= t0]:
        for t2 in coarse_candidates[coarse_candidates >= t1]:
            rows.append([t0, t1, t2, BUDGET_MAX])
coarse_grid = np.array(rows)
print("Stage 1 coarse grid: " + str(coarse_grid.shape[0]) + " combinations")

mean_profits = threshold_to_profit_batch(
    budget_grid, providers_perf, jnp.array(coarse_grid), performance_grid
).mean(axis=-1)
best_idx = int(np.argmax(mean_profits))
best_tau = coarse_grid[best_idx]
print("Coarse optimum: tau=(" + str(round(best_tau[0],4)) + ", " + str(round(best_tau[1],4)) + ", " + str(round(best_tau[2],4)) + ")")

# Stage 2: Fine local search
fine_per_dim = 21
scale = 0.5
t0_lo = max(0.001, best_tau[0]*(1-scale))
t0_hi = best_tau[0]*(1+scale)
t1_lo = max(0.001, best_tau[1]*(1-scale))
t1_hi = best_tau[1]*(1+scale)
t2_lo = max(0.001, best_tau[2]*(1-scale))
t2_hi = best_tau[2]*(1+scale)
t0_range = np.linspace(t0_lo, t0_hi, fine_per_dim)
t1_range = np.linspace(t1_lo, t1_hi, fine_per_dim)
t2_range = np.linspace(t2_lo, t2_hi, fine_per_dim)

fine_rows = []
for t0 in t0_range:
    for t1 in t1_range:
        if t1 < t0: continue
        for t2 in t2_range:
            if t2 < t1: continue
            if t2 > BUDGET_MAX: continue
            fine_rows.append([t0, t1, t2, BUDGET_MAX])
fine_grid = np.array(fine_rows)
print("Stage 2 fine grid: " + str(fine_grid.shape[0]) + " combinations")

fine_profits = threshold_to_profit_batch(
    budget_grid, providers_perf, jnp.array(fine_grid), performance_grid
).mean(axis=-1)
fine_best_idx = int(np.argmax(fine_profits))
optimal_allocation = fine_grid[fine_best_idx]
print("Fine optimum: tau=(" + str(round(optimal_allocation[0],4)) + ", " + str(round(optimal_allocation[1],4)) + ", " + str(round(optimal_allocation[2],4)) + ")")

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

print("=" * 60)
print("REFINED SEARCH - " + str(N_MODELS) + "-MODEL CASCADE")
print("=" * 60)
cascade_str = " -> ".join(MODEL_DISPLAY.get(m,m) for m in MODELS)
print("Cascade: " + cascade_str)
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
