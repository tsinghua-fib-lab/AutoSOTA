"""
Entry point for training Neural MJD/BS models.

This script performs the following steps:
- Parse CLI arguments and load a YAML config
- Initialize logging, randomness, and (optional) distributed helpers
- Build train/val/test PyG dataloaders
- Construct the model, optimizer, scheduler, and optional EMA helpers
- Run the training loop and periodically evaluate/save checkpoints
"""

import logging

from utils.shared_utils import init_basics, init_model
from utils.data_loading.dataloader import load_data
from runner.trainer.trainer import go_training


def main():
    """
    Training begins here. See the README for usage examples.
    """

    """Initialize basics"""
    args, config, dist_helper, writer = init_basics()

    """Get dataloader"""
    train_dl, val_dl, test_dl = load_data(config, dist_helper)

    """Get network"""
    model, optimizer, scheduler, ema_helper = init_model(config, dist_helper, training_mode=True)

    """Go training"""
    go_training(model, optimizer, scheduler, ema_helper, train_dl, val_dl, test_dl, config, dist_helper, writer)

    # Clean up DDP utilities after training
    dist_helper.clean_up()

    logging.info('TRAINING IS FINISHED.')


if __name__ == "__main__":
    main()
