"""
EA-NAS baseline for NAS on tabular data.
Standalone script - no external dependencies beyond pTNAS/src.

Original: VLDB_code/TRAILS/.../exps/baseline/train_with_ea.py
Reference: NAS-Bench-101 evolutionary-search baseline.

The core logic of evolutionary search:
  1. Initialize a population of random architectures
  2. Repeat:
     a. Sample a subset of the population (tournament selection)
     b. Select the best from the subset (parent)
     c. Mutate the parent to create a child
     d. Evaluate the child (query ground truth from NAS-Bench-Tabular)
     e. Add the child to the population, remove the oldest member

Usage:
    cd pTNAS
    PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset frappe
    PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset uci_diabetes
    PYTHONPATH=src python scripts/nas_bench_tabular/re_nas.py --dataset criteo
"""

import argparse
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools'))

from search_space.nas_bench_mlp import NASBenchTabularSpace
from utils.query_api import GTMLP


# ============================================================
# Config
# ============================================================

DATASET_CONFIG = {
    "frappe": {
        "epoch": 13,
        "total_explore": 1000,
        "total_runs": 50,
        "population_size": 10,
        "sample_size": 3,
    },
    "uci_diabetes": {
        "epoch": 0,
        "total_explore": 9000,
        "total_runs": 50,
        "population_size": 10,
        "sample_size": 3,
    },
    "criteo": {
        "epoch": 9,
        "total_explore": 5000,
        "total_runs": 100,
        "population_size": 10,
        "sample_size": 3,
    },
}


# ============================================================
# Evolutionary Search
# ============================================================

def evolutionary_search(space, gt, epoch, total_explore, population_size, sample_size):
    """
    Run one repetition of evolutionary search.

    Args:
        space: NASBenchTabularSpace instance
        gt: GTMLP ground truth query instance
        epoch: which epoch's validation AUC to use as ground truth
        total_explore: total number of architectures to evaluate
        population_size: size of the population
        sample_size: tournament selection size

    Returns:
        time_usage_array: cumulative time (minutes) at each step
        best_auc_array: best AUC found so far at each step
    """
    # Population: deque of (arch_id, auc)
    population = collections.deque(maxlen=population_size)

    time_usage_array = []
    best_auc_array = []
    all_time = 0
    best_auc = 0
    explored = 0

    while explored < total_explore:
        if len(population) < population_size:
            # Initialize: random architecture
            arch_id = space.random_architecture_id()
        else:
            # Tournament selection: sample `sample_size` from population, pick best
            sample = random.sample(list(population), sample_size)
            parent_id = max(sample, key=lambda x: x[1])[0]
            # Mutate parent
            arch_id = space.mutate_architecture(parent_id)

        # Evaluate (query ground truth)
        try:
            auc, time_usage = gt.get_valid_auc(arch_id, epoch_num=epoch)
        except (KeyError, TypeError):
            continue

        # Add to population (oldest removed automatically by deque)
        population.append((arch_id, auc))

        explored += 1
        all_time += time_usage
        best_auc = max(best_auc, auc)

        time_usage_array.append(all_time / 60)  # convert to minutes
        best_auc_array.append(best_auc)

    return time_usage_array, best_auc_array


def main():
    parser = argparse.ArgumentParser(description='EA-NAS baseline')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['frappe', 'uci_diabetes', 'criteo'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = DATASET_CONFIG[args.dataset]

    # Load ground truth
    gt = GTMLP(args.dataset)
    space = NASBenchTabularSpace(args.dataset)
    print(f"Search space: {len(space)} architectures")

    # Run EA-NAS multiple times
    result = {"sys_time_budget": [], "sys_acc": []}

    for run_id in range(cfg["total_runs"]):
        random.seed(args.seed + run_id)
        t0 = time.time()

        time_array, auc_array = evolutionary_search(
            space=space,
            gt=gt,
            epoch=cfg["epoch"],
            total_explore=cfg["total_explore"],
            population_size=cfg["population_size"],
            sample_size=cfg["sample_size"],
        )

        result["sys_time_budget"].append(time_array)
        result["sys_acc"].append(auc_array)

        print(f"  Run {run_id+1}/{cfg['total_runs']}: "
              f"best_auc={auc_array[-1]:.4f}, "
              f"wall_time={time.time()-t0:.1f}s")

    # Save
    if args.output_dir is None:
        output_dir = os.path.join('run_outputs', 'data', 'nas_bench_tabular', 'nas_re')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"train_base_line_re_{args.dataset}_epoch_{cfg['epoch']}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
