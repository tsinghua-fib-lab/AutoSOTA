"""
TabNAS Baseline: RL-based rejection sampling for NAS on tabular data.
Standalone script - no external dependencies beyond pTNAS/src.

Original: VLDB_code/TRAILS/.../exps/baseline/tab_nas.py
Paper: Yang et al., "Tabnas: Rejection sampling for neural architecture search
       on tabular datasets", NeurIPS 2022.

Usage:
    cd pTNAS
    PYTHONPATH=src python scripts/nas_bench_tabular/tab_nas.py --dataset frappe
    PYTHONPATH=src python scripts/nas_bench_tabular/tab_nas.py --dataset uci_diabetes
    PYTHONPATH=src python scripts/nas_bench_tabular/tab_nas.py --dataset criteo
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools'))

from utils.query_api import GTMLP


# ============================================================
# Config
# ============================================================

DEFAULT_LAYER_CHOICES_20 = [8, 16, 24, 32, 48, 64, 80, 96, 112, 128,
                            144, 160, 176, 192, 208, 224, 240, 256, 384, 512]
DEFAULT_LAYER_CHOICES_10 = [8, 16, 32, 48, 96, 112, 144, 176, 240, 384]

DATASET_CONFIG = {
    "frappe": {
        "layer_choices": DEFAULT_LAYER_CHOICES_20,
        "epoch": 13,
        "max_iter": 19000,
        "n_reps": 50,
    },
    "uci_diabetes": {
        "layer_choices": DEFAULT_LAYER_CHOICES_20,
        "epoch": 0,
        "max_iter": 9000,
        "n_reps": 50,
    },
    "criteo": {
        "layer_choices": DEFAULT_LAYER_CHOICES_10,
        "epoch": 9,
        "max_iter": 5000,
        "n_reps": 100,
    },
}


# ============================================================
# TabNAS: RL-based rejection sampling
# ============================================================

def run_sampling(i_rep, layer_choices, rewards, max_iter):
    """
    Run one repetition of TabNAS RL search.
    Uses REINFORCE with rejection-based reward and moving average baseline.
    """
    n_choices = len(layer_choices)
    layer_logits = [torch.zeros(n_choices, requires_grad=True) for _ in range(4)]
    optimizer = torch.optim.Adam(layer_logits, lr=0.1, betas=(0.9, 0.999), eps=1e-8)

    rl_reward_momentum = 0.9
    moving_avg_numer = 0
    moving_avg_denom = 0
    num_samples_mc_start = 5
    num_samples_mc_end = 5

    cur_best_performance = []

    for it in range(max_iter):
        torch.manual_seed(1000 * i_rep + it)

        num_samples_mc = int(np.ceil(
            (max_iter - it) / max_iter * (num_samples_mc_start - num_samples_mc_end) + num_samples_mc_end
        ))

        # Sample architecture from categorical distributions
        dists = [torch.distributions.Categorical(logits=lg) for lg in layer_logits]
        indices = [d.sample() for d in dists]
        choices = tuple(layer_choices[idx] for idx in indices)

        if choices not in rewards:
            continue

        reward = rewards[choices]

        # Track best
        if not cur_best_performance or reward > cur_best_performance[-1]:
            cur_best_performance.append(reward)
        else:
            cur_best_performance.append(cur_best_performance[-1])

        # Moving average baseline
        moving_avg_numer = rl_reward_momentum * moving_avg_numer + (1 - rl_reward_momentum) * reward
        moving_avg_denom = rl_reward_momentum * moving_avg_denom + (1 - rl_reward_momentum)
        baseline = moving_avg_numer / moving_avg_denom
        advantage = reward - baseline

        # Monte Carlo estimate of valid probability
        probs = [torch.softmax(lg, dim=0) for lg in layer_logits]
        mc_samples = [d.sample(torch.Size([num_samples_mc])) for d in dists]
        feasible = [
            (i1, i2, i3, i4)
            for i1, i2, i3, i4 in zip(*mc_samples)
            if (layer_choices[i1], layer_choices[i2], layer_choices[i3], layer_choices[i4]) in rewards
        ]

        if not feasible:
            continue

        est_prob_valid = torch.sum(torch.tensor([
            probs[0][i] * probs[1][j] * probs[2][k] * probs[3][n]
            for i, j, k, n in feasible
        ]))

        cond_prob = probs[0][indices[0]] * probs[1][indices[1]] * probs[2][indices[2]] * probs[3][indices[3]]
        log_cond_prob = torch.log(cond_prob / est_prob_valid)

        loss = -float(advantage) * log_cond_prob
        loss.backward()
        optimizer.step()

    return cur_best_performance


def main():
    parser = argparse.ArgumentParser(description='TabNAS baseline')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['frappe', 'uci_diabetes', 'criteo'])
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    layer_choices = cfg["layer_choices"]
    epoch = cfg["epoch"]

    # Load ground truth from NAS-Bench-Tabular
    gt = GTMLP(args.dataset)
    rewards = {}
    for arch_id in gt.get_all_trained_model_ids():
        arch_tuple = tuple(int(x) for x in arch_id.split("-"))
        try:
            auc, _ = gt.get_valid_auc(arch_id, epoch_num=epoch)
            rewards[arch_tuple] = auc
        except (KeyError, TypeError):
            pass
    print(f"Loaded {len(rewards)} architectures")

    # Run TabNAS
    result = {"sys_time_budget": [], "sys_acc": []}
    for rep in range(cfg["n_reps"]):
        print(f"  Rep {rep+1}/{cfg['n_reps']}...")
        best_perf = run_sampling(rep, layer_choices, rewards, cfg["max_iter"])
        result["sys_time_budget"].append(list(range(1, len(best_perf) + 1)))
        result["sys_acc"].append(best_perf)

    # Save
    if args.output_dir is None:
        output_dir = os.path.join('run_outputs', 'data', 'nas_bench_tabular', 'nas_tabnas')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"tabNAS_benchmark_{args.dataset}_epoch_{epoch}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
