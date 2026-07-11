#!/usr/bin/env python3
"""Generalized N-model arbitrage evaluation with multi-dimensional tau grid search.

Extends the original 2-model grid search to support arbitrary cascade depth,
enabling 3+ model cascades with cheap first-stage filtering.
"""

import json
import os
import sys
import itertools

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebooks"))
from utils import (
    load_pass_curves,
    threshold_to_profit_batch,
    cost_for_pass,
    get_arbitrage_prices,
)

# --- Configuration ---
BASE_DIR = os.path.join(os.path.dirname(__file__), "data", "swebench")
MODELS = ["mini-coder-4b", "qwen3-coder-30b", "gpt5mini", "deepseek"]  # 3-model cascade
MODEL_DISPLAY = {
    "mini-coder-4b": "Mini-Coder 4B",
    "gpt5mini": "GPT-5 mini",
    "deepseek": "DeepSeek v3.2",
    "sonnet": "Sonnet",
    "qwen3-coder-30b": "Qwen3-Coder 30B",
    "qwen3-coder-480b": "Qwen3-Coder 480B",
    "minicoder4b": "MiniCoder 4B",
}
BUDGET_MIN = 1e-4
BUDGET_MAX = 1.0
BUDGET_POINTS = 750
PERFORMANCE_RANGE = (0.68, None)
PERFORMANCE_POINTS = 3000
SCALE_FACTOR = 500
N_MODELS = len(MODELS)

# --- Helper: build N-dimensional tau grid ---
def build_tau_grid_nd(budget_max, n_models, points_per_dim=30):
    """Build an N-dimensional grid of cumulative thresholds.
    
    For N models, we need N-1 free thresholds (last is always budget_max).
    Uses a triangular constraint: 0 <= tau_0 <= tau_1 <= ... <= budget_max.
    
    Returns: (M, N) array of cumulative thresholds, where M = num_combinations.
    """
    if n_models == 1:
        return np.array([[budget_max]])
    
    # Generate candidate threshold values for each level
    # Use log-spaced grid for each threshold dimension, plus zero
    candidates = np.r_[0, np.logspace(np.log10(0.001), np.log10(budget_max), points_per_dim)]
    candidates = np.unique(np.clip(candidates, 0, budget_max))
    
    if n_models == 2:
        # 1D search: just enumerate first threshold
        t0 = candidates
        t1 = np.full_like(t0, budget_max)
        return np.stack([t0, t1], axis=-1)
    
    if n_models == 3:
        # 2D search: tau1 from candidates, tau2 from [tau1, budget_max]
        rows = []
        for t0 in candidates:
            # For tau2, sample from [t0, budget_max]
            remaining_candidates = candidates[candidates >= t0]
            if len(remaining_candidates) == 0:
                continue
            for t1 in remaining_candidates:
                rows.append([t0, t1, budget_max])
        return np.array(rows)
    
    # For 4+ models, use iterative sampling (coarser to avoid explosion)
    if n_models >= 4:
        coarse = np.r_[0, np.logspace(np.log10(0.001), np.log10(budget_max), max(10, points_per_dim//3))]
        coarse = np.unique(np.clip(coarse, 0, budget_max))
        
        # Iteratively build combinations, keeping the monotonic constraint
        # Use a reduced sampling approach
        combos = [[budget_max]]
        for i in range(n_models - 1):
            new_combos = []
            for combo in combos:
                min_val = 0 if i == n_models - 2 else 0
                max_val = combo[0]
                candidates_i = coarse[(coarse >= min_val) & (coarse <= max_val)]
                if len(candidates_i) > 15:
                    indices = np.linspace(0, len(candidates_i)-1, 15).astype(int)
                    candidates_i = candidates_i[indices]
                for c in candidates_i:
                    new_combos.append([c] + combo)
            combos = new_combos
        return np.array(combos)
    
    return np.array([[budget_max]])

# --- Tau search points ---
TAU_POINTS_PER_DIM = 60

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

# --- Build N-d tau grid ---
tau_grid = build_tau_grid_nd(BUDGET_MAX, N_MODELS, TAU_POINTS_PER_DIM)
print(f"Tau grid: {tau_grid.shape[0]} combinations for {N_MODELS}-model cascade")

# --- Grid search ---
max_perf = providers_perf.mean(axis=0).max()
perf_min = PERFORMANCE_RANGE[0]
perf_max = PERFORMANCE_RANGE[1] if PERFORMANCE_RANGE[1] is not None else max_perf
performance_grid = jnp.linspace(perf_min, perf_max, PERFORMANCE_POINTS)

# Evaluate all tau allocations
print("Evaluating tau allocations...")
mean_profits = threshold_to_profit_batch(
    budget_grid, providers_perf, jnp.array(tau_grid), performance_grid
).mean(axis=-1)
optimal_idx = int(np.argmax(mean_profits))
optimal_allocation = tau_grid[optimal_idx]

# --- Compute arbitrage prices ---
arbitrage_cost, arbitrage_expend = get_arbitrage_prices(
    budget_grid, providers_perf, jnp.array(optimal_allocation), performance_grid
)
provider_prices = np.array([
    cost_for_pass(budget_grid, p, performance_grid)
    for p in providers_perf.mean(axis=0)
])
market_price = np.min(provider_prices, axis=0)
arbitrage_profit = np.maximum(0, market_price - arbitrage_cost)
arbitrage_profit_margin = arbitrage_profit / market_price

# --- Extract metrics ---
target_solve = 0.75
idx_75 = np.argmin(np.abs(np.array(performance_grid) - target_solve))
actual_solve_rate = float(performance_grid[idx_75])

arb_cost_75_raw = float(arbitrage_cost[idx_75])
arb_cost_75 = arb_cost_75_raw * SCALE_FACTOR
profit_margin_75 = float(arbitrage_profit_margin[idx_75])

# Per-model costs at 75%
model_costs_75 = {}
for i, m in enumerate(MODELS):
    raw = float(provider_prices[i, idx_75])
    model_costs_75[m] = raw * SCALE_FACTOR if not np.isinf(raw) else float("inf")

market_cost_75 = float(market_price[idx_75]) * SCALE_FACTOR

# Max profit margin
max_margin_idx = int(np.argmax(arbitrage_profit_margin))
max_profit_margin = float(arbitrage_profit_margin[max_margin_idx]) * 100
max_profit_solve_rate = float(performance_grid[max_margin_idx]) * 100

# --- Report ---
print("=" * 60)
print(f"COMPUTATIONAL ARBITRAGE - {N_MODELS}-MODEL CASCADE")
print("=" * 60)
models_display = " -> ".join(MODEL_DISPLAY.get(m, m) for m in MODELS)
print(f"Cascade: {models_display}")
print(f"Problems: {n_problems}")
tau_str = ", ".join(f"{v:.3f}" for v in optimal_allocation)
print(f"Optimal allocation (tau): ({tau_str})")
print(f"Max achievable solve rate: {max_perf*100:.1f}%")
print(f"Tau search points: {tau_grid.shape[0]}")
print()

print("--- Cost Metric (lower better) ---")
print(f"Target solve rate: {target_solve*100:.0f}% (actual: {actual_solve_rate*100:.1f}%)")
for m in MODELS:
    c = model_costs_75[m]
    print(f"  {MODEL_DISPLAY.get(m, m):20s} cost: ${c:.1f}" if not np.isinf(c) else f"  {MODEL_DISPLAY.get(m, m):20s} cost: inf")
print(f"  Market price (min):    ${market_cost_75:.1f}")
print(f"  Arbitrageur cost:      ${arb_cost_75:.1f}")
print(f"  Profit margin at 75%:  {profit_margin_75*100:.1f}%")

print()
print("--- Profit Margin Metric (higher better) ---")
print(f"  Max profit margin:     {max_profit_margin:.1f}%")
print(f"  At solve rate:         {max_profit_solve_rate:.1f}%")

# --- JSON output ---
metrics_json = {
    "paper_id": 3371,
    "n_problems": n_problems,
    "n_models": N_MODELS,
    "models": MODELS,
    "optimal_tau": [float(v) for v in optimal_allocation],
    "max_solve_rate": round(float(max_perf) * 100, 2),
    "cost_at_75": {
        "target_solve_rate_pct": round(actual_solve_rate * 100, 1),
        "arbitrageur": round(arb_cost_75, 1),
        "market_price": round(market_cost_75, 1),
    },
    "profit_margin": {
        "max_pct": round(max_profit_margin, 1),
        "at_solve_rate_pct": round(max_profit_solve_rate, 1),
    },
    "scale_factor": SCALE_FACTOR,
    "tau_search_size": tau_grid.shape[0],
}

for m in MODELS:
    metrics_json["cost_at_75"][m] = round(model_costs_75[m], 1) if not np.isinf(model_costs_75[m]) else "inf"

print()
print("JSON_METRICS:", json.dumps(metrics_json))
