# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified experiment: train encoder + BQ sampling for all domain groups.

Runs two stages for each holdout dataset:
  1. Train a cross-benchmark encoder (using same-group or all benchmarks)
  2. Run BQ sampling (encoder + baselines) on the holdout dataset

Domain groups:
  - safety:    dices, dices_t2i, jigsaw, toxicchat
  - reasoning: strategyqa, mmlu
  - math:      svamp, gsm8k

Settings:
  - new_benchmark (default): Exclude entire target benchmark from training.
  - new_pair: Exclude only the (target_model, target_benchmark) pair;
    keep other models' data on the target benchmark for training.

Usage:
    python -m experiment.exp_run_all_groups
    python -m experiment.exp_run_all_groups --groups safety math
    python -m experiment.exp_run_all_groups --setting new_pair --n-runs 10
    python -m experiment.exp_run_all_groups --skip-training  # reuse existing encoders
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Resolve paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_PARENT_ROOT = os.path.dirname(_PROJECT_ROOT)
if _PARENT_ROOT not in sys.path:
    sys.path.insert(0, _PARENT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proeval.encoder import EncoderTrainer
from proeval.sampler import (
    BQEncoderSampler,
    extract_model_predictions,
    load_embeddings,
    load_predictions,
    setup_train_test_split,
)
from proeval.sampler.baselines import (
    bq_vanilla_sampling,
    random_sampling,
    run_lr_is_evaluation,
    run_lr_lure_evaluation,
    run_rf_is_evaluation,
    run_rf_lure_evaluation,
)
from proeval.utils.metrics import (
    compute_samples_to_threshold,
    print_results_table,
)


# Domain Groups

DATASET_GROUPS = {
    "safety": ["dices", "dices_t2i", "jigsaw", "toxicchat"],
    "reasoning": ["strategyqa", "mmlu"],
    "math": ["svamp", "gsm8k"],
}

DISPLAY_NAMES = {
    "strategyqa": "StrategyQA",
    "gsm8k": "GSM8K",
    "svamp": "SVAMP",
    "mmlu": "MMLU",
    "dices": "DICES",
    "dices_t2i": "DIVE",
    "jigsaw": "Jigsaw",
    "toxicchat": "ToxicChat",
}


def get_group_for_dataset(dataset: str) -> str:
    """Return the group name for a dataset."""
    for group, members in DATASET_GROUPS.items():
        if dataset in members:
            return group
    raise ValueError(f"Unknown dataset: {dataset}")


def get_train_benchmarks(holdout: str, mode: str = "group") -> List[str]:
    """Return training datasets for the given holdout.

    The holdout *benchmark* is always included because
    ``prepare_holdout_split`` removes only the exact
    ``(target_model, holdout_benchmark)`` pair, keeping
    ``(other_model, holdout_benchmark)`` data available for training.

    Args:
        holdout: Dataset being held out for testing.
        mode: ``'group'`` uses same-group benchmarks; ``'all'`` uses
            every dataset.
    """
    if mode == "all":
        return [d for grp in DATASET_GROUPS.values() for d in grp]
    group = get_group_for_dataset(holdout)
    return list(DATASET_GROUPS[group])


# Single-Dataset Experiment

def run_single_dataset(
    holdout_dataset: str,
    target_model: str,
    data_dir: str,
    output_dir: str,
    n_runs: int = 10,
    budget: int = None,
    epochs: int = 1000,
    hidden_dim: int = 8,
    learning_rate: float = 0.01,
    kernel_type: str = "linear",
    init_lengthscale: float = 1.0,
    matern_nu: float = 2.5,
    skip_training: bool = False,
    seed: int = 42,
    train_mode: str = "group",
    checkpoint_interval: int = 50,
    setting: str = "new_benchmark",
) -> Dict:
    """
    Run train + sample for a single holdout dataset.

    Returns:
        Dictionary with experiment results and metrics.
    """
    train_benchmarks = get_train_benchmarks(holdout_dataset, mode=train_mode)
    group = get_group_for_dataset(holdout_dataset)

    print(f"\n{'='*70}")
    print(f"Dataset: {DISPLAY_NAMES.get(holdout_dataset, holdout_dataset)}")
    print(f"Group:   {group}")
    print(f"Train:   {[DISPLAY_NAMES.get(d, d) for d in train_benchmarks]}")
    print(f"{'='*70}")

    result = {
        "dataset": holdout_dataset,
        "group": group,
        "train_benchmarks": train_benchmarks,
        "success": False,
        "error": None,
    }

    # Build checkpoint path
    train_str = "_".join(sorted(train_benchmarks))
    ckpt_name = (
        f"encoder_{group}_train_{train_str}_holdout_{holdout_dataset}"
        f"_hdim{hidden_dim}_kernel_{kernel_type}_ep{epochs}.pth"
    )
    checkpoint_path = os.path.join(output_dir, ckpt_name)

    try:
        # Stage 1: Train encoder
        if skip_training and os.path.exists(checkpoint_path):
            print(f"\n[Stage 1] Skipping training — reusing: {ckpt_name}")
            encoder_path = checkpoint_path
        else:
            print(f"\n[Stage 1] Training encoder...")
            trainer = EncoderTrainer(
                train_benchmarks=train_benchmarks,
                target_benchmark=holdout_dataset,
                target_model=target_model,
                setting=setting,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                epochs=epochs,
                kernel_type=kernel_type,
                init_lengthscale=init_lengthscale,
                matern_nu=matern_nu,
                checkpoint_interval=checkpoint_interval,
            )
            encoder_path = trainer.train(
                data_dir=data_dir,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                seed=seed,
            )
            print(f"  Encoder saved: {encoder_path}")

        result["encoder_path"] = encoder_path

        # Stage 2: BQ Sampling
        print(f"\n[Stage 2] Running BQ sampling...")

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        # Load data
        df = load_predictions(holdout_dataset, data_dir=data_dir)
        prediction_matrix, model_names = extract_model_predictions(
            df, holdout_dataset
        )

        if target_model in model_names:
            target_model_index = model_names.index(target_model)
        else:
            print(f"  Warning: '{target_model}' not found. Using first model.")
            target_model_index = 0

        target_model_name = model_names[target_model_index]
        test_y = prediction_matrix[:, target_model_index]
        true_mean = np.mean(test_y)

        # Pre-training matrix for IS/LURE baselines (model predictions as features)
        pretrain_matrix, _, _, _, _ = setup_train_test_split(
            prediction_matrix, target_model_index
        )

        # Budget: 1% of dataset, minimum 10
        if budget is None:
            budget_actual = max(10, int(0.01 * len(test_y)))
        else:
            budget_actual = budget

        print(f"  Dataset size: {len(test_y)}")
        print(f"  Budget:       {budget_actual}")
        print(f"  True mean:    {true_mean:.4f}")
        print(f"  Model:        {target_model_name}")

        # Initialize encoder sampler
        emb_name = f"{holdout_dataset}_embeddings_text_embedding_3_large.npy"
        embeddings_path = os.path.join(data_dir, emb_name)
        if not os.path.exists(embeddings_path):
            embeddings_path = os.path.join(_PROJECT_ROOT, "data", emb_name)

        encoder_sampler = BQEncoderSampler(
            encoder_path=encoder_path,
            embeddings_path=embeddings_path,
        )
        print(f"  Encoder prior mean: {np.mean(encoder_sampler.prior_u):.4f}")

        # Load text embeddings for vanilla BQ
        text_embeddings = None
        try:
            text_embeddings = load_embeddings(
                holdout_dataset, data_dir=data_dir
            )
        except FileNotFoundError:
            pass

        # Results storage
        results = {
            "bq": {"estimates": [], "indices": []},
            "bq_rounded": {"estimates": [], "indices": []},
            "bq_vanilla": {"estimates": [], "indices": []},
            "random": {"estimates": [], "indices": []},
            "lr_lure": {"estimates": [], "indices": []},
            "rf_lure": {"estimates": [], "indices": []},
            "lr_is": {"estimates": [], "indices": []},
            "rf_is": {"estimates": [], "indices": []},
        }

        seed_size = min(8, budget_actual // 2) if budget_actual > 2 else 1

        for run in range(n_runs):
            run_seed = seed + run
            np.random.seed(run_seed)
            random.seed(run_seed)
            torch.manual_seed(run_seed)

            # BQ with encoder prior (Case 2)
            result_enc = encoder_sampler.sample(
                predictions=df,
                target_model=target_model_index,
                budget=budget_actual,
                seed=run_seed,
            )
            results["bq"]["estimates"].append(result_enc.estimates)
            results["bq_rounded"]["estimates"].append(result_enc.rounded_estimates)

            # Vanilla BQ
            if text_embeddings is not None:
                bq_vanilla_est, _, _, _ = bq_vanilla_sampling(
                    text_embeddings, test_y, budget_actual,
                    n_init=0, noise_variance=encoder_sampler.noise_variance,
                )
                results["bq_vanilla"]["estimates"].append(bq_vanilla_est)

            # Random baseline
            rand_est, _ = random_sampling(test_y, budget_actual)
            results["random"]["estimates"].append(rand_est)

            # IS / LURE baselines
            embeddings = pretrain_matrix
            lr_lure_est, _ = run_lr_lure_evaluation(
                test_y, embeddings, steps=budget_actual, seed_size=seed_size
            )
            results["lr_lure"]["estimates"].append(lr_lure_est)

            rf_lure_est, _ = run_rf_lure_evaluation(
                test_y, embeddings, steps=budget_actual, seed_size=seed_size
            )
            results["rf_lure"]["estimates"].append(rf_lure_est)

            lr_is_est, _ = run_lr_is_evaluation(
                test_y, embeddings, steps=budget_actual, seed_size=seed_size
            )
            results["lr_is"]["estimates"].append(lr_is_est)

            rf_is_est, _ = run_rf_is_evaluation(
                test_y, embeddings, steps=budget_actual, seed_size=seed_size
            )
            results["rf_is"]["estimates"].append(rf_is_est)

            if (run + 1) % 5 == 0 or run == 0:
                print(f"  Run {run + 1}/{n_runs} done")

        # Print results table
        print_results_table(results, true_mean, budget_actual, n_runs, test_y)

        # Compute final MAE for summary
        def final_mae(key):
            ests = results[key]["estimates"]
            if not ests:
                return float("nan"), float("nan")
            maes = [np.abs(e[-1] - true_mean) for e in ests]
            return float(np.mean(maes)), float(np.std(maes))

        result["success"] = True
        result["budget"] = budget_actual
        result["n_questions"] = len(test_y)
        result["true_mean"] = float(true_mean)
        result["target_model"] = target_model_name
        result["metrics"] = {
            "bq_mae": final_mae("bq"),
            "bq_rounded_mae": final_mae("bq_rounded"),
            "bq_vanilla_mae": final_mae("bq_vanilla"),
            "random_mae": final_mae("random"),
            "rf_is_mae": final_mae("rf_is"),
            "rf_lure_mae": final_mae("rf_lure"),
            "lr_is_mae": final_mae("lr_is"),
            "lr_lure_mae": final_mae("lr_lure"),
        }

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"\n  ERROR: {e}")
        traceback.print_exc()

    return result


# Summary Table

ALL_METHODS = [
    ("Random", "random_mae"),
    ("RF+IS", "rf_is_mae"),
    ("LR+IS", "lr_is_mae"),
    ("LR+LURE", "lr_lure_mae"),
    ("RF+LURE", "rf_lure_mae"),
    ("BQ Vanilla", "bq_vanilla_mae"),
    ("BQ Rounded", "bq_rounded_mae"),
    ("BQ", "bq_mae"),
]


def print_summary_table(all_results: Dict[str, Dict]) -> None:
    """Print cross-dataset MAE summary table."""
    datasets = [ds for ds in all_results if all_results[ds].get("success")]
    if not datasets:
        print("No successful experiments.")
        return

    header = f"{'Method':<15}"
    for ds in datasets:
        header += f" {DISPLAY_NAMES.get(ds, ds):>15}"
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("MAE at Final Budget (Lower is Better)")
    print(sep)
    print(header)
    print(sep)

    for method_name, key in ALL_METHODS:
        row = f"{method_name:<15}"
        for ds in datasets:
            m, s = all_results[ds]["metrics"].get(key, (float("nan"), 0))
            if np.isnan(m):
                row += f" {'--':>15}"
            else:
                row += f" {m:.3f}±{s:.3f}"
        print(row)

    print(sep)


def save_summary_csv(
    all_results: Dict[str, Dict], output_path: str
) -> None:
    """Save MAE summary to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["group", "dataset", "method", "final_mae_mean", "final_mae_std"]
        )
        for ds, res in all_results.items():
            if not res.get("success"):
                continue
            group = res.get("group", "unknown")
            ds_label = DISPLAY_NAMES.get(ds, ds)
            for method_name, key in ALL_METHODS:
                m, s = res["metrics"].get(key, (float("nan"), 0))
                writer.writerow([group, ds_label, method_name, f"{m:.6f}", f"{s:.6f}"])
    print(f"\nMAE summary CSV saved to: {output_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Train + BQ sample for all domain groups",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--groups", type=str, nargs="+", default=None,
        choices=list(DATASET_GROUPS.keys()),
        help="Groups to run (default: all)",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Specific datasets to run (overrides --groups)",
    )
    parser.add_argument("--target-model", type=str, default="gemini25_flash")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--budget", type=int, default=None,
                        help="Sampling budget (default: 1%% of dataset)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--kernel-type", type=str, default="linear",
                        choices=["linear", "matern", "rbf"])
    parser.add_argument("--init-lengthscale", type=float, default=1.0)
    parser.add_argument("--matern-nu", type=float, default=2.5)
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save checkpoint every N epochs (0=disabled)")
    parser.add_argument("--train-mode", type=str, default="group",
                        choices=["group", "all"],
                        help="'group': same-group benchmarks, "
                             "'all': all except holdout")
    parser.add_argument("--setting", type=str, default="new_benchmark",
                        choices=["new_pair", "new_benchmark", "new_model"],
                        help="'new_benchmark': exclude entire target benchmark from training. "
                             "'new_pair': exclude only the (target_model, target_benchmark) pair, "
                             "keep other models' data on the target benchmark. "
                             "'new_model': same as new_pair for encoder training.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Reuse existing encoder checkpoints")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = args.data_dir or os.path.join(_PROJECT_ROOT, "data")
    output_dir = args.output_dir or os.path.join(
        "results", f"grouped_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Determine which datasets to run
    if args.datasets:
        datasets_to_run = args.datasets
    elif args.groups:
        datasets_to_run = []
        for g in args.groups:
            datasets_to_run.extend(DATASET_GROUPS[g])
    else:
        datasets_to_run = []
        for members in DATASET_GROUPS.values():
            datasets_to_run.extend(members)

    print("=" * 70)
    print("Unified Domain-Grouped Experiment")
    print("=" * 70)
    print(f"Datasets:  {datasets_to_run}")
    print(f"Model:     {args.target_model}")
    print(f"Setting:   {args.setting}")
    print(f"Runs:      {args.n_runs}")
    print(f"Epochs:    {args.epochs}")
    print(f"Kernel:    {args.kernel_type}")
    print(f"Output:    {output_dir}")
    print("=" * 70)
    for group, members in DATASET_GROUPS.items():
        active = [d for d in members if d in datasets_to_run]
        if active:
            print(f"  {group}: {active}")
    print("=" * 70)

    # Run experiments
    all_results = {}

    for dataset in datasets_to_run:
        result = run_single_dataset(
            holdout_dataset=dataset,
            target_model=args.target_model,
            data_dir=data_dir,
            output_dir=output_dir,
            n_runs=args.n_runs,
            budget=args.budget,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            kernel_type=args.kernel_type,
            init_lengthscale=args.init_lengthscale,
            matern_nu=args.matern_nu,
            skip_training=args.skip_training,
            seed=args.seed,
            train_mode=args.train_mode,
            checkpoint_interval=args.checkpoint_interval,
            setting=args.setting,
        )
        all_results[dataset] = result

    # Print summary
    print_summary_table(all_results)

    # Save results
    results_json_path = os.path.join(output_dir, "all_grouped_results.json")
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults JSON saved to: {results_json_path}")

    csv_path = os.path.join(output_dir, "mae_summary.csv")
    save_summary_csv(all_results, csv_path)

    # Print per-group summary
    print(f"\n{'='*70}")
    print("Per-Group Summary")
    print(f"{'='*70}")
    for group, members in DATASET_GROUPS.items():
        active = [d for d in members if d in all_results]
        if not active:
            continue
        print(f"\n  [{group}]")
        for ds in active:
            res = all_results[ds]
            if res.get("success"):
                bq_m, bq_s = res["metrics"]["bq_mae"]
                rnd_m, rnd_s = res["metrics"]["random_mae"]
                print(
                    f"    {DISPLAY_NAMES.get(ds, ds):12s}  "
                    f"BQ: {bq_m:.4f}±{bq_s:.4f}  "
                    f"Random: {rnd_m:.4f}±{rnd_s:.4f}"
                )
            else:
                print(f"    {DISPLAY_NAMES.get(ds, ds):12s}  FAILED: {res.get('error', '?')}")

    print(f"\n{'='*70}")
    print("All experiments complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
