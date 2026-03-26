"""
Entry point for evaluating trained Neural MJD/BS models.

This script:
- Parses CLI args and loads the experiment config
- Builds validation and test PyG dataloaders
- Constructs the model and loads weights (optionally EMA variants)
- Runs evaluation and logs metrics
"""

import os
import copy
import logging
import torch

from utils.shared_utils import init_basics, init_model, get_ema_weight_keywords
from utils.learning_utils import load_model
from utils.data_loading.dataloader import load_data

from runner.trainer.trainer_utils import create_logging_dict, print_epoch_learning_status
from runner.trainer.trainer_mjd import move_forward_one_epoch_mjd


def batch_evaluate(model, dist_helper, val_dl, test_dl, config, args_model_path, args_use_ema, writer) -> None:
    """
    Load one or more checkpoints and run evaluation on validation and test sets.

    Args:
        model: Torch model to evaluate.
        dist_helper: Distributed training helper.
        val_dl: Validation dataloader.
        test_dl: Test dataloader.
        config: Experiment configuration.
        args_model_path: List of checkpoint file paths.
        args_use_ema: EMA usage specification from CLI (None, 'all', or list of betas).
        writer: Optional tensorboard writer.
    """
    logging.info("Models to load:")
    [logging.info("{:d}: {:s}".format(i, item)) for i, item in enumerate(args_model_path)]

    for model_path in args_model_path:
        model_nm = os.path.basename(model_path)
        logging.info("{:s} Evaluating model at {:s} {:s}".format('-' * 6, model_path, '-' * 6))
        ckp_data = torch.load(model_path, map_location=lambda storage, loc: storage)
        weight_keywords = get_ema_weight_keywords(ckp_data, copy.copy(args_use_ema))

        for weight_kw in weight_keywords:
            logging.info("Loading weight for {:s} to create samples...".format(weight_kw))
            load_model(ckp_data, model, weight_kw)

            epoch = int(model_nm.split('_')[-1].replace('.pth', ''))

            # Go sampling!
            logger_dict = create_logging_dict(epoch)
            logger_dict['lr'] = -1
            with torch.no_grad():
                move_forward_one_epoch_mjd(model, optimizer=None, ema_helper=None, dist_helper=dist_helper, 
                                           dataloader=val_dl, logger_dict=logger_dict, mode='val', dataset_nm=config.dataset.name,
                                           plot_save_dir=config.plot_save_dir, huber_delta=None, writer=None)
                move_forward_one_epoch_mjd(model, optimizer=None, ema_helper=None, dist_helper=dist_helper, 
                                           dataloader=test_dl, logger_dict=logger_dict, mode='test', dataset_nm=config.dataset.name,
                                           plot_save_dir=config.plot_save_dir, huber_delta=None, writer=None)

            # show the training and testing status
            print_epoch_learning_status(logger_dict, writer, adjust_r2='sp500' in config.dataset.name)

        # sync DDP processes and release GPU memory
        dist_helper.ddp_sync()
        del ckp_data


def main():
    """
    Evaluation begins here. See the README for usage examples.
    """

    """Initialize basics"""
    args, config, dist_helper, writer = init_basics(mode='eval')

    """Get dataloader"""
    train_dl, val_dl, test_dl = load_data(config, dist_helper, eval_mode=True)

    """Get network"""
    model = init_model(config, dist_helper, training_mode=False)

    """Go evaluation"""
    batch_evaluate(model, dist_helper, val_dl, test_dl, config, args.model_path, args.use_ema, writer)

    # Clean up DDP utilities after evaluation
    dist_helper.clean_up()

    logging.info('EVALUATION IS FINISHED.')

if __name__ == "__main__":
    main()
