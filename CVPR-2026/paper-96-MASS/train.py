#!/usr/bin/env python3
"""
Main training entry point for MASS.

This script loads a YAML config, initializes distributed training when needed,
builds the requested trainer/model/dataset through the registry system, and
runs pretraining or downstream example training.
"""

import os
import sys
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Any
import utils
import models
import data
import metrics
from training.utils import verify_registry_integrity

verify_registry_integrity()

from config.config_utils import parse_args, prepare_config, setup_output_dir, is_master
from utils.distributed import setup_distributed, cleanup_distributed, set_seed
from utils.registry import get_trainer, list_trainers

import warnings
warnings.filterwarnings("ignore")

def main_worker(proc_idx: int, config: Dict[str, Any]):
    """
    Main training function for a single worker process.
    
    Args:
        proc_idx: Local process index
        config: Configuration dictionary
    """
    config['distributed']['proc_idx'] = proc_idx
    os.environ['LOCAL_RANK'] = str(proc_idx)

    torch.cuda.set_device(proc_idx)
    setup_distributed(config)

    trainer_type = config.get('trainer', {}).get('type', 'mask_guided_self_supervised')

    if is_master():
        available_trainers = list_trainers()
        logging.info(f"Available trainers: {available_trainers}")
        logging.info(f"Using trainer: {trainer_type}")

    try:
        trainer_cls = get_trainer(trainer_type)
    except KeyError:
        raise ValueError(f"Trainer '{trainer_type}' not found. Available trainers: {list_trainers()}")

    trainer = trainer_cls(config)
    trainer.train()
    cleanup_distributed()


def main():
    """Main entry point for training."""
    mp.set_start_method('spawn', force=True)

    args = parse_args()
    config = prepare_config(args)
    output_dir = setup_output_dir(config)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'train.log')) if is_master() else logging.NullHandler()
        ]
    )

    if is_master():
        logging.info(f"Starting training with configuration: {config}")

    set_seed(config.get('seed', 42))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config['distributed']['gpus']))
    world_size = config['distributed']['world_size']

    if world_size > 1:
        # Use scheduler-provided master address/port when running under SLURM.
        master_addr = os.environ.get('MASTER_ADDR', 
                                    config['distributed'].get('master_addr', 'localhost'))
        master_port = os.environ.get('MASTER_PORT', 
                                    str(config['distributed'].get('master_port', 29500)))

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        config['distributed']['master_addr'] = master_addr
        config['distributed']['master_port'] = master_port
        config['distributed']['dist_url'] = f'tcp://{master_addr}:{master_port}'

        logging.info(f"Environment Master address: {master_addr}")
        logging.info(f"Environment Master port: {master_port}")
        logging.info(f"Dist url: {config['distributed']['dist_url']}")
        logging.info(f"Config Master addr: {config['distributed']['master_addr']}")
        logging.info(f"Config Master port: {config['distributed']['master_port']}")
        
        mp.spawn(
            main_worker,
            args=(config,),
            nprocs=config['distributed']['gpus_per_node'],
            join=True
        )
    else:
        main_worker(0, config)


if __name__ == '__main__':
    main()
