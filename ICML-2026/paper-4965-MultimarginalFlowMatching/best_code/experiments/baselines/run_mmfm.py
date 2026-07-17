#!/usr/bin/env python
"""
Wrapper script for running MMFM (Multi-Marginal Flow Matching) baseline.

MMFM is from: Tong et al. "Improving and generalizing flow-based generative models
with minibatch optimal transport" arXiv:2302.00482 (2023).

To run this baseline:
1. Clone the MMFM repository: https://github.com/atong01/conditional-flow-matching
2. Follow their setup instructions
3. Use our data loading utilities for consistency

Usage:
    python experiments/baselines/run_mmfm.py --dataset singlecell
    python experiments/baselines/run_mmfm.py --dataset gulfofmexico

Author(s): Raghav Kansal
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MMFM_REPO = "https://github.com/atong01/conditional-flow-matching"


def check_mmfm_installation():
    """Check if MMFM is installed."""
    try:
        import importlib.util

        return importlib.util.find_spec("torchcfm") is not None
    except ImportError:
        return False


def run_mmfm_singlecell(args):
    """Run MMFM on single-cell data."""
    logger.info("Loading single-cell data...")

    from experiments.singlecell.data import create_eb_dataloaders, load_eb_data

    data = load_eb_data(
        data_dir=Path(args.data_dir) / "singlecell",
        pca_dim=args.pca_dim,
        normalize=True,
        ot_coupling=True,
    )

    train_loader, val_loader = create_eb_dataloaders(
        marginals=data["marginals_list"],
        train_times=data["train_times"],
        batch_size=args.batch_size,
        ot_alignments=data["ot_alignments"],
    )

    logger.info(f"Data loaded. Training times: {data['train_times']}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # At this point, use the MMFM library to train
    # Example integration would be:
    #
    # from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
    # model = build_model(d=data['dim'])
    # cfm = ConditionalFlowMatcher()
    #
    # for epoch in range(args.epochs):
    #     for batch in train_loader:
    #         ...

    logger.info("=" * 60)
    logger.info("To complete MMFM training, integrate with torchcfm library.")
    logger.info(f"Repository: {MMFM_REPO}")
    logger.info("=" * 60)


def run_mmfm_gom(args):
    """Run MMFM on Gulf of Mexico data."""
    logger.info("Loading Gulf of Mexico data...")

    from experiments.gulfofmexico.data import create_gom_dataloaders, load_gom_data

    data = load_gom_data(
        data_dir=Path(args.data_dir) / "gom",
        normalize=True,
        ot_coupling=True,
    )

    train_loader, _ = create_gom_dataloaders(
        marginals=data["marginals_list"],
        batch_size=args.batch_size,
        holdout_times=data["holdout_times"],
        ot_alignments=data["ot_alignments"],
    )

    logger.info(f"Data loaded. Training times: {data['train_times']}")
    logger.info("See MMFM repository for training implementation.")


def run_mmfm_beijing(args):
    """Run MMFM on Beijing air quality data."""
    logger.info("Loading Beijing air quality data...")

    from experiments.beijingair.data import create_beijing_dataloaders, load_beijing_data

    data = load_beijing_data(
        data_dir=Path(args.data_dir) / "beijing",
        normalize=True,
        ot_coupling=True,
    )

    train_loader, _ = create_beijing_dataloaders(
        marginals=data["marginals_list"],
        batch_size=args.batch_size,
        holdout_times=data["holdout_times"],
        ot_alignments=data["ot_alignments"],
    )

    logger.info(f"Data loaded. Training times: {data['train_times']}")
    logger.info("See MMFM repository for training implementation.")


def main():
    parser = argparse.ArgumentParser(description="Run MMFM baseline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["singlecell", "gulfofmexico", "beijingair"],
        help="Dataset to run on",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--pca-dim", type=int, default=100, help="PCA dimension for single-cell data"
    )

    args = parser.parse_args()

    if not check_mmfm_installation():
        logger.error("MMFM (torchcfm) not installed.")
        logger.error(f"Install from: {MMFM_REPO}")
        logger.error("  pip install torchcfm")
        sys.exit(1)

    if args.dataset == "singlecell":
        run_mmfm_singlecell(args)
    elif args.dataset == "gulfofmexico":
        run_mmfm_gom(args)
    elif args.dataset == "beijingair":
        run_mmfm_beijing(args)


if __name__ == "__main__":
    main()
