#!/usr/bin/env python3
"""
Main training script for SelfIE adapters.

Usage:
    python train.py --config configs/scalar_affine_8b_goodfire.yaml
    python train.py --config configs/scalar_plus_low_rank_8b.yaml --cache-dir /path/to/hf/cache
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import Config
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train SelfIE adapters")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (no training)",
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Print experiment info
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"Experiment name: {config.experiment_name or '(not set)'}")
    print(f"Model: {config.model.name}")
    print(f"Projection type: {config.projection.type}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Create trainer
    trainer = Trainer(config, cache_dir=args.cache_dir)
    
    # Run
    if args.eval_only:
        trainer.evaluate_only()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
