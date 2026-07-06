"""
BOHB-NAS Baseline: BOHB (Bayesian Optimization and HyperBand) for NAS on tabular data.
Standalone script - no external dependencies beyond pTNAS/src + hpbandster.

Original: VLDB_code/TRAILS/.../exps/baseline/train_bohb.py

Uses hpbandster's BOHB optimizer to search over 4-layer MLP hidden sizes.
Queries ground truth AUC from NAS-Bench-Tabular.

Requirements:
    pip install hpbandster ConfigSpace

Usage:
    cd pTNAS
    PYTHONPATH=src python scripts/nas_bench_tabular/bohb_nas.py --dataset frappe
    PYTHONPATH=src python scripts/nas_bench_tabular/bohb_nas.py --dataset uci_diabetes
    PYTHONPATH=src python scripts/nas_bench_tabular/bohb_nas.py --dataset criteo
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools'))

from search_space.nas_bench_mlp import DATASET_CONFIGS
from utils.query_api import GTMLP

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

logging.getLogger('hpbandster').setLevel(logging.WARNING)


# ============================================================
# Config
# ============================================================

DATASET_CONFIG = {
    "frappe": {"epoch": 13, "total_runs": 50, "n_iterations": 180},
    "uci_diabetes": {"epoch": 0, "total_runs": 50, "n_iterations": 180},
    "criteo": {"epoch": 9, "total_runs": 50, "n_iterations": 180},
}


# ============================================================
# BOHB Worker
# ============================================================

class MLPWorker(Worker):
    def __init__(self, performance_results, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_results = performance_results
        self.baseline_acc = []

    def compute(self, config, budget, *args, **kwargs):
        architecture = '-'.join(str(config[f'layer_{i}']) for i in range(4))
        performance = self.performance_results.get(architecture)

        if performance is None:
            logging.warning(f'Architecture {architecture} not found')
            return {'loss': float('inf'), 'info': architecture}

        # Track best-so-far
        if not self.baseline_acc or performance > self.baseline_acc[-1]:
            self.baseline_acc.append(performance)
        else:
            self.baseline_acc.append(self.baseline_acc[-1])

        return {'loss': 1 - performance, 'info': architecture}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='BOHB-NAS baseline')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['frappe', 'uci_diabetes', 'criteo'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    ds_cfg = DATASET_CONFIGS[args.dataset]
    layer_choices = ds_cfg["layer_choices"]

    # Load ground truth
    gt = GTMLP(args.dataset)
    gt_aucs = gt.get_all_ground_truth_aucs(epoch_num=cfg["epoch"])
    performance_results = {arch_id: auc for arch_id, auc in gt_aucs.items()}
    print(f"Dataset: {args.dataset}, {len(performance_results)} architectures loaded")

    # Config space
    config_space = CS.ConfigurationSpace()
    for i in range(4):
        config_space.add_hyperparameter(
            CSH.CategoricalHyperparameter(f'layer_{i}', choices=[str(c) for c in layer_choices]))

    result = {"sys_time_budget": [], "sys_acc": []}

    for run_id in range(cfg["total_runs"]):
        t0 = time.time()

        # Start nameserver
        ns = hpns.NameServer(run_id=f'bohb_{run_id}', host='localhost', port=0)
        ns_host, ns_port = ns.start()

        # Start worker
        w = MLPWorker(
            performance_results=performance_results,
            nameserver=ns_host, nameserver_port=ns_port,
            run_id=f'bohb_{run_id}')
        w.run(background=True)

        # Run BOHB
        bohb = BOHB(
            configspace=config_space,
            run_id=f'bohb_{run_id}',
            nameserver=ns_host,
            nameserver_port=ns_port)
        bohb.run(n_iterations=cfg["n_iterations"])

        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()

        result["sys_time_budget"].append(list(range(1, len(w.baseline_acc) + 1)))
        result["sys_acc"].append(w.baseline_acc)

        print(f"  Run {run_id+1}/{cfg['total_runs']}: "
              f"best_auc={w.baseline_acc[-1]:.4f}, "
              f"explored={len(w.baseline_acc)}, "
              f"wall_time={time.time()-t0:.1f}s")

    # Save
    if args.output_dir is None:
        output_dir = os.path.join('run_outputs', 'data', 'nas_bench_tabular', 'nas_bohb')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"train_base_line_bohb_{args.dataset}_epoch_{cfg['epoch']}.json")
    with open(output_file, 'w') as f:
        json.dump(result, f)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
