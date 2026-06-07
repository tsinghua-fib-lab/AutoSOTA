import os
import sys
import json
import random
import logging
import argparse

import numpy as np
import torch


logger = logging.getLogger(__name__)


def init_logging(log_file=None):
    handlers = []
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    handlers.append(ch)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.INFO)
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', handlers=handlers, force=True)


def replace_config(config: dict, new_config: dict):
    for k, v in new_config.items():
        current_config = config
        father_keys, final_key = k.split('.')[:-1], k.split('.')[-1]
        for key in father_keys:
            if key not in current_config:
                raise ValueError(f'Invalid config key: {k}')
            current_config = current_config[key]
        if final_key in current_config and v is not None:
            logger.info(f'Overriding config: `{k}` from {current_config[final_key]} to {v}')
            current_config[final_key] = v
    return config


def set_default_config(config: dict):
    config.setdefault('seed', 0)

    dataset_default_config = config['dataset'].pop('default', {})
    for dataset in config['dataset'].keys():
        for k, v in dataset_default_config.items():
            config['dataset'][dataset].setdefault(k, v)

    config['predictor']['device'] = config['trainer']['device']


def load_config(config_path: str, args: argparse.Namespace=None):
    logger.info(f'Loading config from {config_path}')

    # Load config from file
    ext = config_path.split('.')[-1]
    if ext == 'json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f'Unsupported config file extension: {ext}')
    
    # Override config with args
    if args is not None:
        args = vars(args)
        replace_config(config, args)

    # Set default values
    set_default_config(config)

    return config


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
