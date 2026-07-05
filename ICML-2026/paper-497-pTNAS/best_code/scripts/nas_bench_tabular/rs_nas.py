"""
RS-NAS Baseline: Random Search for NAS on tabular data.
Standalone script - no external dependencies beyond pTNAS/src.

Original: VLDB_code/TRAILS/.../exps/baseline/train_with_random.py

Randomly samples architectures, queries ground truth from NAS-Bench-Tabular,
and tracks the best AUC found so far.

Usage:
    cd pTNAS
    PYTHONPATH=src python scripts/nas_bench_tabular/rs_nas.py --dataset frappe
    PYTHONPATH=src python scripts/nas_bench_tabular/rs_nas.py --dataset uci_diabetes
    PYTHONPATH=src python scripts/nas_bench_tabular/rs_nas.py --dataset criteo
"""

import argparse
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
    "frappe": {"epoch": 13, "total_explore": 19000, "total_runs": 50},
    "uci_diabetes": {"epoch": 0, "total_explore": 9000, "total_runs": 50},
    "criteo": {"epoch": 9, "total_explore": 5000, "total_runs": 100},
}


# ============================================================
# Random Search
# ============================================================

def random_search(space, gt, epoch, total_explore):
    """
    Run one repetition of Random Search.

    Returns:
        time_usage_array: cumulative time (minutes) at each step
        best_auc_array: best AUC found so far at each step
    """
    time_usage_array = []
    best_auc_array = []
    all_time = 0
    best_auc = 0

    explored = 0
    while explored < total_explore:
        arch_id = space.random_architecture_id()

        try:
            auc, time_usage = gt.get_valid_auc(arch_id, epoch_num=epoch)
        except (KeyError, TypeError):
            continue

        explored += 1
        all_time += time_usage
        best_auc = max(best_auc, auc)

        time_usage_array.append(all_time / 60)
        best_auc_array.append(best_auc)

    return time_usage_array, best_auc_array


def main():
    parser = argparse.ArgumentParser(description='RS-NAS (Random Search) baseline')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['frappe', 'uci_diabetes', 'criteo'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = DATASET_CONFIG[args.dataset]

    gt = GTMLP(args.dataset)
    space = NASBenchTabularSpace(args.dataset)
    print(f"Search space: {len(space)} architectures")

    result = {"sys_time_budget": [], "sys_acc": []}

    for run_id in range(cfg["total_runs"]):
        random.seed(args.seed + run_id)
        t0 = time.time()

        time_array, auc_array = random_search(
            space=space, gt=gt, epoch=cfg["epoch"],
            total_explore=cfg["total_explore"],
        )

        result["sys_time_budget"].append(time_array)
        result["sys_acc"].append(auc_array)

        print(f"  Run {run_id+1}/{cfg['total_runs']}: "
              f"best_auc={auc_array[-1]:.4f}, "
              f"wall_time={time.time()-t0:.1f}s")

    # Save
    if args.output_dir is None:
        output_dir = os.path.join('run_outputs', 'data', 'nas_bench_tabular', 'nas_rs')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"train_base_line_rs_{args.dataset}_epoch_{cfg['epoch']}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
