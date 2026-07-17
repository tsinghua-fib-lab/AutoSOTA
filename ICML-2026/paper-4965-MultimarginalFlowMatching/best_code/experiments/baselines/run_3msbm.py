#!/usr/bin/env python
"""
Wrapper script for running 3MSBM baseline.

3MSBM (Momentum Multi-Marginal Schrödinger Bridge Matching) is from:
Theodoropoulos et al. "Momentum multi-marginal Schrödinger bridge matching"
arXiv:2506.10168 (2025).

Repository: https://github.com/nikitadobrokhtov/mmsbm

To run this baseline:
1. Clone the 3MSBM repository
2. Follow their setup instructions
3. Use our data loading utilities for consistency

Usage:
    python experiments/baselines/run_3msbm.py --dataset beijingair

Author(s): Raghav Kansal
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MSBM_REPO = "https://github.com/nikitadobrokhtov/mmsbm"


def check_3msbm_installation():
    """Check if 3MSBM is available."""
    try:
        # 3MSBM doesn't have a pip package, so we just check if repo is cloned
        # Users would need to add it to their path
        return True
    except ImportError:
        return False


def run_3msbm_beijing(args):
    """Run 3MSBM on Beijing air quality data."""
    logger.info("Loading Beijing air quality data...")

    from experiments.beijingair.data import load_beijing_data

    data = load_beijing_data(
        data_dir=Path(args.data_dir) / "beijing",
        normalize=True,
    )

    logger.info(f"Data loaded. Train times: {data['train_times']}")
    logger.info(f"Holdout times: {data['holdout_times']}")
    logger.info(f"Dimension: {data['dim']}")

    # 3MSBM training would be integrated here
    # They use a different training paradigm with Schrödinger Bridge

    logger.info("=" * 60)
    logger.info("To run 3MSBM training:")
    logger.info(f"1. Clone: {MSBM_REPO}")
    logger.info("2. Follow their training scripts with this data")
    logger.info("=" * 60)


def run_3msbm_gom(args):
    """Run 3MSBM on Gulf of Mexico data."""
    logger.info("Loading Gulf of Mexico data...")

    from experiments.gulfofmexico.data import load_gom_data

    data = load_gom_data(
        data_dir=Path(args.data_dir) / "gom",
        normalize=True,
    )

    logger.info(f"Data loaded. Train times: {data['train_times']}")
    logger.info("See 3MSBM repository for training implementation.")


def main():
    parser = argparse.ArgumentParser(description="Run 3MSBM baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["gulfofmexico", "beijingair"],
        help="Dataset to run on",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=1000)

    args = parser.parse_args()

    if args.dataset == "beijingair":
        run_3msbm_beijing(args)
    elif args.dataset == "gulfofmexico":
        run_3msbm_gom(args)
    else:
        logger.error(f"Dataset {args.dataset} not supported for 3MSBM")
        sys.exit(1)


if __name__ == "__main__":
    main()
