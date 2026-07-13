#!/usr/bin/env python3
"""
Build consolidated training data for category-aware R2-Router.

Reads all budget sweep JSONs + cached embeddings → single pickle file.

Output format:
{
    "embeddings": (8400, 1024) float32,
    "global_indices": [str] * 8400,
    "categories": (8400,) int,
    "models": {
        "235b": {
            "budget_10": {"accuracy": (8400,), "output_tokens": (8400,)},
            ...,
        },
        ...
    }
}

Usage:
    .venv/bin/python scripts/build_category_training_data.py
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    MODELS, EMBEDDINGS_PATH, ROUTER_DATA_PATH, TRAINING_DATA_PATH,
    get_category, get_budgets,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # 1. Load router_data.json for global_index ordering
    log("Loading router_data.json...")
    with open(ROUTER_DATA_PATH) as f:
        router_data = json.load(f)
    global_indices = [q["global index"] for q in router_data]
    n_queries = len(global_indices)
    log(f"  {n_queries} queries")

    # Build index map: global_index -> position
    idx_map = {gi: i for i, gi in enumerate(global_indices)}

    # 2. Load cached embeddings
    log("Loading embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        emb_data = pickle.load(f)
    embeddings = emb_data["embeddings"]
    assert embeddings.shape[0] == n_queries, \
        f"Embeddings {embeddings.shape[0]} != router_data {n_queries}"
    log(f"  Embeddings: {embeddings.shape}")

    # 3. Assign categories
    log("Assigning categories...")
    categories = np.array([get_category(gi) for gi in global_indices], dtype=np.int32)
    from category_config import CATEGORY_NAMES
    for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
        count = (categories == cat_idx).sum()
        log(f"  {cat_idx} {cat_name}: {count}")

    # 4. Load sweep data for each model × budget
    log("Loading sweep data...")
    models_data = {}

    for model_name, model_cfg in MODELS.items():
        sweep_dir = model_cfg["sweep_dir"]
        budgets = get_budgets(model_name)
        models_data[model_name] = {}

        for budget_name in budgets:
            json_path = os.path.join(sweep_dir, f"{budget_name}.json")
            if not os.path.exists(json_path):
                log(f"  WARNING: Missing {json_path}")
                continue

            with open(json_path) as f:
                entries = json.load(f)

            accuracy = np.zeros(n_queries, dtype=np.float32)
            output_tokens = np.zeros(n_queries, dtype=np.float32)
            found = 0
            missing_accuracy = 0

            for e in entries:
                if e.get("for_optimality", False):
                    continue

                gi = e["global index"]
                if gi not in idx_map:
                    continue

                pos = idx_map[gi]
                found += 1

                # Accuracy
                acc = e.get("accuracy")
                if acc is not None:
                    accuracy[pos] = float(acc)
                else:
                    missing_accuracy += 1

                # Output tokens
                gr = e.get("generated_result")
                if gr and isinstance(gr, dict):
                    tu = gr.get("token_usage", {})
                    if tu:
                        output_tokens[pos] = float(tu.get("output_tokens", 0))

            models_data[model_name][budget_name] = {
                "accuracy": accuracy,
                "output_tokens": output_tokens,
            }

            status = f"found={found}/{n_queries}"
            if missing_accuracy > 0:
                status += f", missing_acc={missing_accuracy}"
            log(f"  {model_name}/{budget_name}: {status}")

    # 5. Save consolidated data
    output_dir = os.path.dirname(TRAINING_DATA_PATH)
    os.makedirs(output_dir, exist_ok=True)

    training_data = {
        "embeddings": embeddings,
        "global_indices": global_indices,
        "categories": categories,
        "models": models_data,
    }

    log(f"Saving to {TRAINING_DATA_PATH}...")
    with open(TRAINING_DATA_PATH, "wb") as f:
        pickle.dump(training_data, f, protocol=4)

    file_size = os.path.getsize(TRAINING_DATA_PATH) / (1024 * 1024)
    log(f"  Size: {file_size:.1f} MB")

    # Summary
    log("\nSummary:")
    log(f"  Queries: {n_queries}")
    log(f"  Embedding dim: {embeddings.shape[1]}")
    log(f"  Categories: {len(CATEGORY_NAMES)}")
    log(f"  Models: {len(models_data)}")
    total_budgets = sum(len(v) for v in models_data.values())
    log(f"  Total model×budget combos: {total_budgets}")
    log("Done!")


if __name__ == "__main__":
    main()