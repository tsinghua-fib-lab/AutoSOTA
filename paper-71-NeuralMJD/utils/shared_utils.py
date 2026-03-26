import logging
import torch
from typing import Tuple, List, Optional, Any

from utils.arg_parser import parse_arguments, set_seed_and_logger, backup_code
from utils.learning_utils import get_network, get_optimizer, get_ema_helper
from utils.dist_training import DistributedHelper


def init_basics(mode: str = 'train') -> Tuple[Any, Any, DistributedHelper, Optional[Any]]:
    """Initialize CLI args, config, distributed helper, and logger/writer.

    Args:
        mode: Either 'train' or 'eval'. Determines logging dirs and backup behavior.

    Returns:
        A tuple of (args, config, dist_helper, writer).
    """
    # Initialization
    args, config = parse_arguments(mode=mode)
    dist_helper = DistributedHelper(args.dp, args.ddp, args.ddp_gpu_ids, args.ddp_init_method)
    writer = set_seed_and_logger(config, args.log_level, args.comment, dist_helper, eval_mode=mode == 'eval')
    
    # Only backup code during training
    if mode == 'train':
        backup_code(config, args.config_file)
    
    return args, config, dist_helper, writer


def init_model(config: Any, dist_helper: DistributedHelper, training_mode: bool = True):
    """Initialize model and, if training, optimizer/scheduler/EMA helper.

    Args:
        config: Experiment configuration.
        dist_helper: Distributed training helper.
        training_mode: If True, also returns optimizer, scheduler, and EMA helpers.

    Returns:
        If training_mode is True: (model, optimizer, scheduler, ema_helper).
        Otherwise: model only.
    """
    # Initialize network model
    model = get_network(config, dist_helper)
    if isinstance(model, torch.nn.Module):
        pass
    else:
        logging.info("Model is not a torch.nn.Module instance.")

    if training_mode:
        # Initialize optimizer and EMA helper for training
        optimizer, scheduler = get_optimizer(model, config, dist_helper)
        ema_helper = get_ema_helper(config, model)
        return model, optimizer, scheduler, ema_helper
    else:
        # Only return model for evaluation
        return model


def get_ema_weight_keywords(ckp_data: dict, args_use_ema) -> List[str]:
    """Extract which weight keys (online / EMA variants) to evaluate from a checkpoint.

    Args:
        ckp_data: Loaded checkpoint dictionary.
        args_use_ema: None, 'all', or list of EMA betas specifying which EMA weights to use.

    Returns:
        List of weight keys to load (e.g., ['model', 'model_ema_beta_0.9990']).
    """
    all_weight_keywords = []
    for item in list(ckp_data.keys()):
        if item.startswith('model'):
            all_weight_keywords.append(item)

    weight_keywords = ['model']
    if args_use_ema is None:
        logging.info('Not using EMA weight.')
    elif args_use_ema == 'all':
        # lazy init: to use all online and EMA weights
        weight_keywords = all_weight_keywords
        logging.info('Use all possible EMA weights.')
    else:
        logging.info('Using EMA weight with coefficients: {}'.format(args_use_ema))
        if 1.0 not in args_use_ema:
            weight_keywords.remove('model')
        else:
            args_use_ema.remove(1.0)
        for item in args_use_ema:
            _weight_keyword = 'model_ema_beta_{:.4f}'.format(item)
            assert _weight_keyword in all_weight_keywords, "{} not found in the model data!".format(_weight_keyword)
            weight_keywords.append(_weight_keyword)

    logging.info('Model weights to load: {}'.format(weight_keywords))
    return weight_keywords 