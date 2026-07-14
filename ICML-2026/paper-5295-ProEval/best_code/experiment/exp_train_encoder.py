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
Train a neural encoder for BQ prior.

Trains an encoder on benchmark data and excludes the target model's
labels (or the entire target benchmark) based on the setting.

Settings:
  --setting new_pair (default): Exclude only the (target_model, target_benchmark)
    pair; keep other models' data on the target benchmark for training.
  --setting new_benchmark: Fully exclude the target benchmark from training.

Usage:
    # Same-benchmark training (exclude only target model's labels):
    python -m experiment.exp_train_encoder \
        --train-benchmarks svamp \
        --target-benchmark svamp \
        --target-model gemini25_flash \
        --setting new_pair \
        --hidden-dim 16 --kernel-type matern --epochs 1000

    # Cross-benchmark training (exclude all svamp data):
    python -m experiment.exp_train_encoder \
        --train-benchmarks gsm8k strategyqa \
        --target-benchmark svamp \
        --target-model gemini25_flash \
        --setting new_benchmark \
        --hidden-dim 16 --kernel-type matern --epochs 200
"""

import argparse
import os
import sys

# Resolve paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_PARENT_ROOT = os.path.dirname(_PROJECT_ROOT)
if _PARENT_ROOT not in sys.path:
    sys.path.insert(0, _PARENT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proeval.encoder import EncoderTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train encoder for BQ prior (Setting 1: Cross-Benchmark)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Benchmark configuration
    parser.add_argument("--train-benchmarks", type=str, nargs="+",
                        default=["gsm8k", "strategyqa"],
                        help="Benchmarks for training")
    parser.add_argument("--target-benchmark", type=str, default="svamp",
                        help="Target benchmark for evaluation")
    parser.add_argument("--target-model", type=str, default="gemini25_flash",
                        help="Model to evaluate for BQ testing")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--init-var", type=float, default=0.3,
                        help="Initial noise variance")
    parser.add_argument("--steepness", type=float, default=1.0,
                        help="Sigmoid steepness for psi activation (default: 1.0). "
                             "High values (e.g. 30) cause saturation and vanishing gradients.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Questions per (benchmark, model) pair")
    parser.add_argument("--num-pairs-per-batch", type=int, default=1000,
                        help="(benchmark, model) pairs per batch")

    # Kernel options
    parser.add_argument("--kernel-type", type=str, default="matern",
                        choices=["linear", "matern", "rbf"])
    parser.add_argument("--init-lengthscale", type=float, default=1.0,
                        help="Initial lengthscale for matern/rbf kernels")
    parser.add_argument("--matern-nu", type=float, default=2.5,
                        help="Matérn smoothness (0.5, 1.5, or 2.5)")

    # Paths
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/checkpoints/)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Exact path to save encoder .pth (overrides --output-dir)")
    parser.add_argument("--setting", type=str, default="new_benchmark",
                        choices=["new_pair", "new_benchmark", "new_model"],
                        help="'new_benchmark': exclude entire target benchmark from training. "
                             "'new_pair': exclude only the (target_model, target_benchmark) pair.")
    parser.add_argument("--include-models", type=str, nargs="+", default=None,
                        help="If provided, train only on these models' labels (GMM-selected).")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(_PROJECT_ROOT, "data")
    output_dir = args.output_dir or os.path.join(_PROJECT_ROOT, "data", "checkpoints")

    trainer = EncoderTrainer(
        train_benchmarks=args.train_benchmarks,
        target_benchmark=args.target_benchmark,
        target_model=args.target_model,
        setting=args.setting,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        init_var=args.init_var,
        steepness=args.steepness,
        kernel_type=args.kernel_type,
        init_lengthscale=args.init_lengthscale,
        matern_nu=args.matern_nu,
        data_batch_size=args.batch_size,
        num_pairs_per_batch=args.num_pairs_per_batch,
        include_models=args.include_models,
    )

    encoder_path = trainer.train(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )

    print(f"\nEncoder saved to: {encoder_path}")
    print("\nTo use this encoder for sampling:")
    print(f"  python -m experiment.exp_sampling_w_embedding_case_2 \\")
    print(f"      --dataset {args.target_benchmark} \\")
    print(f"      --encoder-path {encoder_path}")


if __name__ == "__main__":
    main()
