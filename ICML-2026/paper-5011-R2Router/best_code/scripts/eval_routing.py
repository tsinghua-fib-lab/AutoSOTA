#!/usr/bin/env python3
"""Evaluate a routing decision file by looking up actual accuracy/cost from sweep JSONs.

Usage:
    python scripts/eval_routing.py --routing-file PATH [--exclude-models gpt4o haiku]

The routing file must have _r2_model and _r2_budget_label metadata per entry.
Accuracy and cost are extracted from the corresponding sweep JSON files.
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

from category_config import MODEL_COST_PATH, MODELS, SWEEP_ROOT

MODEL_COST_KEY = {name: cfg["cost_key"] for name, cfg in MODELS.items()}
MODEL_SWEEP_DIR = {
    name: os.path.basename(cfg["sweep_dir"].rstrip("/"))
    for name, cfg in MODELS.items()
}

BUDGET_TO_FILE = {
    "10": "budget_10.json",
    "20": "budget_20.json",
    "40": "budget_40.json",
    "80": "budget_80.json",
    "150": "budget_150.json",
    "200": "budget_200.json",
    "400": "budget_400.json",
    "800": "budget_800.json",
    "1500": "budget_1500.json",
    "unlimited": "budget_unlimited.json",
    "concise": "concise.json",
}

# API models don't have sweep data
API_MODELS = {"gpt4o", "gemini-flash", "haiku"}


def load_sweep_cache(needed_files):
    """Pre-load all needed sweep JSON files into memory, indexed by global index."""
    cache = {}  # {model: {budget: {gidx: entry}}}
    loaded = 0
    for model, budget in sorted(needed_files):
        if model not in cache:
            cache[model] = {}

        sweep_dir = MODEL_SWEEP_DIR.get(model, model)
        filename = BUDGET_TO_FILE.get(budget)
        if not filename:
            print(f"  [WARN] Unknown budget '{budget}' for {model}")
            continue

        filepath = os.path.join(SWEEP_ROOT, sweep_dir, filename)
        try:
            with open(filepath) as f:
                data = json.load(f)
            cache[model][budget] = {e["global index"]: e for e in data}
            loaded += 1
        except FileNotFoundError:
            print(f"  [WARN] Missing file: {filepath}")

    return cache, loaded


def compute_entry_cost(tokens, cost_key, costs):
    """Compute input+output cost for an entry."""
    if cost_key not in costs:
        return 0.0
    in_price = costs[cost_key]["input_token_price_per_million"]
    out_price = costs[cost_key]["output_token_price_per_million"]
    return (tokens.get("input_tokens", 0) * in_price +
            tokens.get("output_tokens", 0) * out_price) / 1e6


def main():
    parser = argparse.ArgumentParser(description="Evaluate routing decisions using sweep data")
    parser.add_argument("--routing-file", required=True, help="Path to routing JSON file")
    parser.add_argument("--exclude-models", nargs="*", default=[], help="Models to exclude from analysis")
    args = parser.parse_args()

    start = time.time()

    # Load routing decisions
    print(f"Loading routing file: {args.routing_file}")
    with open(args.routing_file) as f:
        routing_data = json.load(f)

    regular = [e for e in routing_data if not e.get("for_optimality")]
    print(f"  Total entries: {len(routing_data)}, Regular: {len(regular)}")

    if args.exclude_models:
        print(f"  Excluding models: {args.exclude_models}")

    # Load costs
    with open(MODEL_COST_PATH) as f:
        costs = json.load(f)

    # Step 1: Determine which sweep files to load
    needed_files = set()
    for e in regular:
        model = e.get("_r2_model")
        budget = e.get("_r2_budget_label")
        if model and budget and model not in API_MODELS and model not in args.exclude_models:
            needed_files.add((model, budget))

    print(f"\nPre-loading {len(needed_files)} sweep files into memory...")
    cache, loaded = load_sweep_cache(needed_files)
    print(f"  Loaded {loaded}/{len(needed_files)} files in {time.time()-start:.1f}s")

    # Step 2: Extract accuracy and cost
    print("\nExtracting accuracy and cost...")
    total_correct = 0
    total_evaluated = 0
    total_cost = 0.0
    total_queries = 0
    skipped = defaultdict(int)
    model_stats = defaultdict(lambda: {"correct": 0, "evaluated": 0, "cost": 0.0, "count": 0})

    for route_entry in regular:
        model = route_entry.get("_r2_model")
        budget = route_entry.get("_r2_budget_label")
        gidx = route_entry["global index"]

        if model in args.exclude_models:
            skipped["excluded"] += 1
            continue

        total_queries += 1
        cost_key = MODEL_COST_KEY.get(model)

        # API models: no sweep data, use routing file's generated_result for cost only
        if model in API_MODELS:
            gen = route_entry.get("generated_result")
            if gen and gen.get("success") and cost_key:
                tokens = gen.get("token_usage", {})
                cost = compute_entry_cost(tokens, cost_key, costs)
                total_cost += cost
                model_stats[model]["cost"] += cost
            model_stats[model]["count"] += 1
            skipped["api_no_accuracy"] += 1
            continue

        # vLLM models: lookup from sweep cache
        if model in cache and budget in cache[model]:
            sweep_entry = cache[model][budget].get(gidx)
            if not sweep_entry:
                skipped["not_found"] += 1
                continue

            # Accuracy
            acc = sweep_entry.get("accuracy")
            if acc is not None:
                total_evaluated += 1
                model_stats[model]["evaluated"] += 1
                if acc == 1:
                    total_correct += 1
                    model_stats[model]["correct"] += 1
            else:
                skipped["no_accuracy"] += 1

            # Cost
            gen = sweep_entry.get("generated_result")
            if gen and gen.get("success") and cost_key:
                tokens = gen.get("token_usage", {})
                cost = compute_entry_cost(tokens, cost_key, costs)
                total_cost += cost
                model_stats[model]["cost"] += cost
            model_stats[model]["count"] += 1
        else:
            skipped["cache_miss"] += 1

    elapsed = time.time() - start

    # Step 3: Print results
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\nSkipped: {dict(skipped)}")

    accuracy_pct = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0
    cost_per_1k = (total_cost / total_queries * 1000) if total_queries > 0 else 0

    print(f"\n{'='*90}")
    print(f"Results")
    print(f"{'='*90}")
    print(f"  Queries routed: {total_queries}")
    print(f"  Evaluated (has accuracy): {total_evaluated}")
    print(f"  Correct: {total_correct}")
    print(f"  Accuracy: {accuracy_pct:.2f}%")
    print(f"  Cost/1K queries: ${cost_per_1k:.4f}")

    if total_evaluated > 0 and cost_per_1k > 0:
        A = accuracy_pct / 100
        C = (math.log2(200) - math.log2(cost_per_1k)) / (math.log2(200) - math.log2(0.0044))
        beta = 0.1
        arena = ((1 + beta) * A * C) / (beta * A + C)
        print(f"  Arena Score (β=0.1): {arena:.4f}")

    # Per-model breakdown
    print(f"\n{'Model':<15} {'Count':>6} {'Eval':>6} {'Correct':>8} {'Acc%':>8} {'Cost/1K':>10}")
    print("-" * 65)
    for model in sorted(model_stats.keys(), key=lambda m: model_stats[m]["count"], reverse=True):
        s = model_stats[model]
        acc = (s["correct"] / s["evaluated"] * 100) if s["evaluated"] > 0 else float("nan")
        c1k = (s["cost"] / total_queries * 1000) if total_queries > 0 else 0
        print(f"{model:<15} {s['count']:>6} {s['evaluated']:>6} {s['correct']:>8} {acc:>7.1f}% ${c1k:>9.4f}")

    print("-" * 65)
    print(f"{'TOTAL':<15} {total_queries:>6} {total_evaluated:>6} {total_correct:>8} {accuracy_pct:>7.2f}% ${cost_per_1k:>9.4f}")


if __name__ == "__main__":
    main()
