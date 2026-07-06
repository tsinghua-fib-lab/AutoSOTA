"""
RL-NAS Baseline: REINFORCE-based NAS on tabular data.
Standalone script - no external dependencies beyond pTNAS/src.

Original: VLDB_code/TRAILS/.../exps/baseline/train_with_rl.py

Uses REINFORCE policy gradient to learn per-layer sampling distributions
over hidden layer sizes. Each iteration samples an architecture, queries
its ground truth AUC from NAS-Bench-Tabular, and updates the policy.

Usage:
    cd pTNAS
    PYTHONPATH=src python scripts/nas_bench_tabular/rl_nas.py --dataset frappe
    PYTHONPATH=src python scripts/nas_bench_tabular/rl_nas.py --dataset uci_diabetes
    PYTHONPATH=src python scripts/nas_bench_tabular/rl_nas.py --dataset criteo
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools'))

from search_space.nas_bench_mlp import NASBenchTabularSpace, DATASET_CONFIGS
from utils.query_api import GTMLP


# ============================================================
# Config
# ============================================================

DATASET_CONFIG = {
    "frappe": {"epoch": 13, "max_iter": 19000, "total_runs": 50},
    "uci_diabetes": {"epoch": 0, "max_iter": 9000, "total_runs": 50},
    "criteo": {"epoch": 9, "max_iter": 5000, "total_runs": 50},
}

# Approximate training time per epoch (seconds) for time budget estimation
TRAIN_ONE_EPOCH_TIME = {
    "frappe": 5.122203075885773,
    "uci_diabetes": 4.16297769,
    "criteo": 422,
}


def get_parameter_number(hidden_layer_sizes, input_size=2):
    all_layer_sizes = [input_size] + list(hidden_layer_sizes) + [1]
    return int(sum(
        all_layer_sizes[i] * all_layer_sizes[i + 1] + all_layer_sizes[i + 1]
        for i in range(len(all_layer_sizes) - 1)
    ))


# ============================================================
# RL-NAS (REINFORCE)
# ============================================================

def rl_search(gt, dataset, epoch, max_iter, layer_choices, run_id,
              beta=2, lr=0.05):
    """
    Run one repetition of RL-based NAS with REINFORCE.

    Returns:
        time_usage_array: cumulative time (minutes) at each step
        best_auc_array: best AUC found so far at each step
    """
    # Load all ground truth AUC and time
    gt_aucs = gt.get_all_ground_truth_aucs(epoch_num=epoch)

    # Build lookup by tuple
    rewards = {}
    for arch_id, auc in gt_aucs.items():
        arch_tuple = tuple(int(x) for x in arch_id.split("-"))
        rewards[arch_tuple] = auc

    n_layers = 4
    logits = [torch.zeros(len(layer_choices), requires_grad=True) for _ in range(n_layers)]
    optimizer = torch.optim.Adam(logits, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    rl_reward_momentum = 0.9
    moving_avg_numer = 0
    moving_avg_denom = 0

    best_auc_array = []
    time_usage_array = []
    cur_time = 0
    time_per_epoch = TRAIN_ONE_EPOCH_TIME[dataset]

    for it in range(max_iter):
        torch.manual_seed(1000 * run_id + it)

        # Sample architecture
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        indices = [d.sample() for d in dists]
        chosen = tuple(layer_choices[idx] for idx in indices)

        if chosen not in rewards:
            continue

        auc = rewards[chosen]

        # Track best AUC
        if not best_auc_array or auc > best_auc_array[-1]:
            best_auc_array.append(auc)
        else:
            best_auc_array.append(best_auc_array[-1])

        # Time budget (estimated)
        cur_time += time_per_epoch * (epoch + 1)
        time_usage_array.append(cur_time / 60)

        # Compute advantage
        rl_reward = auc - beta * abs(
            get_parameter_number(list(chosen)) / get_parameter_number([3, 3]) - 1)
        moving_avg_numer = rl_reward_momentum * moving_avg_numer + (1 - rl_reward_momentum) * rl_reward
        moving_avg_denom = rl_reward_momentum * moving_avg_denom + (1 - rl_reward_momentum)
        baseline = moving_avg_numer / moving_avg_denom
        advantage = rl_reward - baseline

        # REINFORCE update
        probs = [torch.softmax(d.logits, dim=0) for d in dists]
        sampling_prob = probs[0][indices[0]]
        for k in range(1, n_layers):
            sampling_prob = sampling_prob * probs[k][indices[k]]

        loss = -float(advantage) * torch.log(sampling_prob)
        loss.backward()
        optimizer.step()

    return time_usage_array, best_auc_array


def main():
    parser = argparse.ArgumentParser(description='RL-NAS (REINFORCE) baseline')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['frappe', 'uci_diabetes', 'criteo'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    ds_cfg = DATASET_CONFIGS[args.dataset]
    layer_choices = ds_cfg["layer_choices"]

    gt = GTMLP(args.dataset)
    print(f"Dataset: {args.dataset}, layer_choices: {layer_choices}")

    result = {"sys_time_budget": [], "sys_acc": []}

    for run_id in range(cfg["total_runs"]):
        t0 = time.time()

        time_array, auc_array = rl_search(
            gt=gt, dataset=args.dataset, epoch=cfg["epoch"],
            max_iter=cfg["max_iter"], layer_choices=layer_choices,
            run_id=run_id,
        )

        result["sys_time_budget"].append(time_array)
        result["sys_acc"].append(auc_array)

        print(f"  Run {run_id+1}/{cfg['total_runs']}: "
              f"best_auc={auc_array[-1]:.4f}, "
              f"wall_time={time.time()-t0:.1f}s")

    # Save
    if args.output_dir is None:
        output_dir = os.path.join('run_outputs', 'data', 'nas_bench_tabular', 'nas_rl')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"train_base_line_rl_{args.dataset}_epoch_{cfg['epoch']}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
