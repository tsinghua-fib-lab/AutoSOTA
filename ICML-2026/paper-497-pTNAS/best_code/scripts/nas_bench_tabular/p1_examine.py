"""
Phase-1 Examine: Run EA with pTProxy scoring to generate Figure 11 data.

Original: VLDB_code/TRAILS/.../exps/micro/benchmark_score_metrics.py

Runs evolutionary search (EA) on NAS-Bench-Tabular, recording:
  - AUC file: ground truth AUC of each explored architecture (best-so-far not applied)
  - Score file: pTProxy score of each explored architecture (best-so-far)

This shows the correlation between pTProxy ranking and ground truth AUC
during the search process (Figure 11).

Output:
    run_outputs/data/nas_bench_tabular/nas_p1_examine/re_{dataset}_-1_12_auc.json
    run_outputs/data/nas_bench_tabular/nas_p1_examine/re_{dataset}_-1_12_score.json

Usage:
    cd pTNAS
    PYTHONPATH=src python scripts/nas_bench_tabular/p1_examine.py --dataset frappe
    PYTHONPATH=src python scripts/nas_bench_tabular/p1_examine.py --dataset all
"""

import argparse
import collections
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools'))

import torch
from search_space.nas_bench_mlp import NASBenchTabularSpace
from utils.query_api import GTMLP
from proxies.ptproxy import ptproxy_score


# ============================================================
# Config
# ============================================================

DATASET_CONFIG = {
    "frappe": {"epoch": 13, "total_explore": 5000, "total_runs": 50},
    "uci_diabetes": {"epoch": 0, "total_explore": 5000, "total_runs": 50},
    "criteo": {"epoch": 9, "total_explore": 5000, "total_runs": 50},
}

POPULATION_SIZE = 10
SAMPLE_SIZE = 3
BATCH_SIZE = 32


# ============================================================
# EA + pTProxy scoring
# ============================================================

def run_one_ea_with_scoring(space, gt, epoch, total_explore, device="cpu"):
    """
    Run one repetition of EA, recording both ground truth AUC and pTProxy score
    for each explored architecture.

    Returns:
        arch_indices: [1, 2, 3, ..., N]
        auc_values: ground truth AUC of each explored arch (raw, not best-so-far)
        score_values: pTProxy score best-so-far at each step
    """
    population = collections.deque(maxlen=POPULATION_SIZE)

    arch_indices = []
    auc_values = []
    score_values = []
    best_score = float('-inf')
    explored = 0

    while explored < total_explore:
        if len(population) < POPULATION_SIZE:
            arch_id = space.random_architecture_id()
        else:
            sample = random.sample(list(population), SAMPLE_SIZE)
            parent_id = max(sample, key=lambda x: x[1])[0]
            arch_id = space.mutate_architecture(parent_id)

        # Get ground truth AUC
        try:
            auc, _ = gt.get_valid_auc(arch_id, epoch_num=epoch)
        except (KeyError, TypeError):
            continue

        # Compute pTProxy score
        try:
            model = space.new_architecture(arch_id, use_bn=False)
            batch_data = model.generate_all_ones_embedding(BATCH_SIZE).float().to(device)
            model = model.to(device)

            score, _ = ptproxy_score(
                arch=model.mlp,
                batch_data=batch_data,
                device=device,
                use_wo_embedding=False,
                epsilon=1e-5,
                weight_mode="traj_width",
            )
            del model

            if not math.isfinite(score):
                score = 0.0
        except Exception:
            score = 0.0

        # Use pTProxy score for population selection (Phase 1 uses proxy, not AUC)
        population.append((arch_id, score))

        explored += 1
        best_score = max(best_score, score)

        arch_indices.append(explored)
        auc_values.append(auc)
        score_values.append(best_score)

    return arch_indices, auc_values, score_values


def run_p1_examine(dataset, device="cpu", seed=42):
    cfg = DATASET_CONFIG[dataset]
    gt = GTMLP(dataset)
    space = NASBenchTabularSpace(dataset)

    result_auc = {"explored_arch": [], "achieved_value": []}
    result_score = {"explored_arch": [], "achieved_value": []}

    for run_id in range(cfg["total_runs"]):
        random.seed(seed + run_id)
        torch.manual_seed(seed + run_id)
        t0 = time.time()

        indices, aucs, scores = run_one_ea_with_scoring(
            space=space, gt=gt, epoch=cfg["epoch"],
            total_explore=cfg["total_explore"], device=device,
        )

        result_auc["explored_arch"].append(indices)
        result_auc["achieved_value"].append(aucs)

        result_score["explored_arch"].append(indices)
        result_score["achieved_value"].append(scores)

        print(f"  Run {run_id+1}/{cfg['total_runs']}: "
              f"final_auc={aucs[-1]:.4f}, best_score={scores[-1]:.2f}, "
              f"wall_time={time.time()-t0:.1f}s")

    return result_auc, result_score


def main():
    parser = argparse.ArgumentParser(description='Phase-1 examine (Figure 11 data)')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['frappe', 'uci_diabetes', 'criteo', 'all'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets = ['frappe', 'uci_diabetes', 'criteo'] if args.dataset == 'all' else [args.dataset]

    if args.output_dir is None:
        output_dir = os.path.join('run_outputs', 'data', 'nas_bench_tabular', 'nas_p1_examine')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Phase-1 examine: {ds}")
        print(f"{'='*60}")

        result_auc, result_score = run_p1_examine(ds, device=args.device, seed=args.seed)

        auc_file = os.path.join(output_dir, f"re_{ds}_-1_12_auc.json")
        score_file = os.path.join(output_dir, f"re_{ds}_-1_12_score.json")

        with open(auc_file, 'w') as f:
            json.dump(result_auc, f)
        with open(score_file, 'w') as f:
            json.dump(result_score, f)

        print(f"  Saved: {auc_file}")
        print(f"  Saved: {score_file}")


if __name__ == "__main__":
    main()
