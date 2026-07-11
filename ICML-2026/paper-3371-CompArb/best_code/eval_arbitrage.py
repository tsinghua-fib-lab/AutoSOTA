#!/usr/bin/env python3
"""Reproduction evaluation script for Computational Arbitrage in AI Model Markets.

Loads pre-computed SWE-bench pass/fail data for GPT-5 mini and DeepSeek v3.2,
computes the optimal two-model arbitrage strategy, and reports key metrics:
  - Cost at 75% solve rate (arbitrageur, individual models)
  - Profit margin at the optimal point

Paper: Computational Arbitrage in AI Model Markets (arXiv:2603.22404)
"""

import json
import os
import sys

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp

# Add notebooks dir for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebooks"))
from utils import (
    load_pass_curves,
    threshold_to_profit_batch,
    cost_for_pass,
    get_arbitrage_prices,
)

# --- Configuration ---
BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "swebench")
MODELS = ["gpt5mini", "deepseek"]
MODEL_DISPLAY = {"gpt5mini": "GPT-5 mini", "deepseek": "DeepSeek v3.2"}
BUDGET_MIN = 1e-4
BUDGET_MAX = 1.0
BUDGET_POINTS = 250
PERFORMANCE_RANGE = (0.68, None)  # min solve rate; None -> use max achievable
PERFORMANCE_POINTS = 1000
COMPUTE_SEARCH_POINTS = 100
SCALE_FACTOR = 500  # Total budget in dollars (maps raw fraction to $)

# --- Load data ---
results = {}
for fname in os.listdir(BASE_DIR):
    if not fname.endswith(".jsonl"):
        continue
    model_name = fname.split(".jsonl")[0]
    data = [json.loads(line) for line in open(os.path.join(BASE_DIR, fname), "r")]
    results[model_name] = data

budget_grid = np.logspace(np.log10(BUDGET_MIN), np.log10(BUDGET_MAX), BUDGET_POINTS)
providers_perf = load_pass_curves(results, MODELS, budget_grid)
n_problems = providers_perf.shape[0]

# --- Grid search for optimal allocation ---
max_perf = providers_perf.mean(axis=0).max()
perf_min = PERFORMANCE_RANGE[0]
perf_max = PERFORMANCE_RANGE[1] if PERFORMANCE_RANGE[1] is not None else max_perf
performance_grid = jnp.linspace(perf_min, perf_max, PERFORMANCE_POINTS)

compute_first = np.r_[0, np.logspace(np.log10(0.01), np.log10(0.5), COMPUTE_SEARCH_POINTS), budget_grid.max()]
compute_second = budget_grid.max() - compute_first
compute_allocation = np.stack([compute_first, compute_second], axis=-1)

mean_profits = threshold_to_profit_batch(
    budget_grid, providers_perf, compute_allocation, performance_grid
).mean(axis=-1)
optimal_allocation = compute_allocation[np.argmax(mean_profits, axis=-1)]

# --- Compute arbitrage prices ---
arbitrage_cost, arbitrage_expend = get_arbitrage_prices(
    budget_grid, providers_perf, optimal_allocation, performance_grid
)
provider_prices = np.array([
    cost_for_pass(budget_grid, p, performance_grid)
    for p in providers_perf.mean(axis=0)
])
market_price = np.min(provider_prices, axis=0)
arbitrage_profit = np.maximum(0, market_price - arbitrage_cost)
arbitrage_profit_margin = arbitrage_profit / market_price

# --- Extract metrics ---
# Metric 1: Cost at 75% solve rate
target_solve = 0.75
idx_75 = np.argmin(np.abs(np.array(performance_grid) - target_solve))
actual_solve_rate = float(performance_grid[idx_75])

arb_cost_75_raw = float(arbitrage_cost[idx_75])
ds_cost_75_raw = float(provider_prices[1, idx_75])
gpt5_cost_75_raw = float(provider_prices[0, idx_75])
market_cost_75_raw = float(market_price[idx_75])
profit_margin_75 = float(arbitrage_profit_margin[idx_75])

arb_cost_75 = arb_cost_75_raw * SCALE_FACTOR
ds_cost_75 = ds_cost_75_raw * SCALE_FACTOR
gpt5_cost_75 = gpt5_cost_75_raw * SCALE_FACTOR if not np.isinf(gpt5_cost_75_raw) else float("inf")
market_cost_75 = market_cost_75_raw * SCALE_FACTOR

# Metric 2: Maximum profit margin
max_margin_idx = int(np.argmax(arbitrage_profit_margin))
max_profit_margin = float(arbitrage_profit_margin[max_margin_idx]) * 100
max_profit_solve_rate = float(performance_grid[max_margin_idx]) * 100

# --- Report ---
print("=" * 60)
print("COMPUTATIONAL ARBITRAGE - REPRODUCTION RESULTS")
print("=" * 60)
models_display = ", ".join(MODEL_DISPLAY[m] for m in MODELS)
print(f"Models: {models_display}")
print(f"Problems: {n_problems}")
print(f"Optimal allocation (tau): ({float(optimal_allocation[0]):.2f}, {float(optimal_allocation[1]):.2f})")
print(f"Max achievable solve rate: {max_perf*100:.1f}%")
print()

print("--- Cost Metric (lower better) ---")
print(f"Target solve rate: {target_solve*100:.0f}% (actual: {actual_solve_rate*100:.1f}%)")
print(f"  GPT-5 mini cost:       ${gpt5_cost_75:.1f}")
print(f"  DeepSeek v3.2 cost:    ${ds_cost_75:.1f}")
print(f"  Market price (min):    ${market_cost_75:.1f}")
print(f"  Arbitrageur cost:      ${arb_cost_75:.1f}")
print(f"  Profit margin at 75%:  {profit_margin_75*100:.1f}%")

print()
print("--- Profit Margin Metric (higher better) ---")
print(f"  Max profit margin:     {max_profit_margin:.1f}%")
print(f"  At solve rate:         {max_profit_solve_rate:.1f}%")

print()
print("--- Comparison with Paper ---")
print(f"  Paper arbitrageur cost:       $80")
print(f"  Reproduced arbitrageur cost:  ${arb_cost_75:.1f}")
print(f"  Paper DeepSeek cost:          $120")
print(f"  Reproduced DeepSeek cost:     ${ds_cost_75:.1f}")
print(f"  Paper max profit margin:      40%")
print(f"  Reproduced max profit margin: {max_profit_margin:.1f}%")

# --- JSON output for parsing ---
metrics_json = {
    "paper_id": 3371,
    "n_problems": n_problems,
    "optimal_tau": [float(optimal_allocation[0]), float(optimal_allocation[1])],
    "max_solve_rate": round(float(max_perf) * 100, 2),
    "cost_at_75": {
        "target_solve_rate_pct": round(actual_solve_rate * 100, 1),
        "arbitrageur": round(arb_cost_75, 1),
        "deepseek_v3_2": round(ds_cost_75, 1),
        "gpt5_mini": round(gpt5_cost_75, 1) if not np.isinf(gpt5_cost_75) else "inf",
        "market_price": round(market_cost_75, 1),
    },
    "profit_margin": {
        "max_pct": round(max_profit_margin, 1),
        "at_solve_rate_pct": round(max_profit_solve_rate, 1),
    },
    "scale_factor": SCALE_FACTOR,
    "budget_range": [BUDGET_MIN, BUDGET_MAX],
}

print()
print("JSON_METRICS:", json.dumps(metrics_json))
