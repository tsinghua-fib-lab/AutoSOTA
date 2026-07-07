import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import matplotlib
import logging
import os
from omegaconf import OmegaConf, DictConfig
import wandb
import sys; sys.modules['tensorflow'] = None # avoid tensorflow import error in ot
import ot


def model_info(model: nn.Module, name='Model'):
    return '{}: num of params: {}, size: {:.2f} MB'.format(
        name,
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        )


def compute_ess(log_weights: torch.Tensor) -> float:
    '''
    Compute the normalized effective sample size (ESS) given log weights.

    Args:
        log_weights (torch.Tensor): shape (B,)

    Returns:
        ess: a float number between 1 / B and 1
    '''
    weights = log_weights.softmax(dim=0)
    return 1.0 / torch.sum(weights ** 2) / weights.shape[0]


def compute_energy_w2_distance(samples1: torch.Tensor, samples2: torch.Tensor, energy_fn: callable) -> float:
    '''
    Compute the Wasserstein-2 distance between the empirical energy distributions of two sets of samples.
    Args:
        samples1 (torch.Tensor): shape (..., D)
        samples2 (torch.Tensor): shape (..., D)
        energy_fn (callable): function that takes in samples and outputs their energies
    
    Returns:
        w2_distance (float): Wasserstein-2 distance between the energy distributions of samples1 and samples2
    '''
    energies1 = energy_fn(samples1).view(-1).cpu().numpy()
    energies2 = energy_fn(samples2).view(-1).cpu().numpy()
    return ot.wasserstein_1d(energies1, energies2, p=2).item() ** 0.5


def get_local_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving: # log to file
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying: # log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


class Logger:
    def __init__(self, args, rank: int = 0):
        self.rank = rank
        if self.rank == 0:
            self.local_logger = get_local_logger(logpath=os.path.join(args.logging.dir, '.log'), 
                                                 displaying=args.logging.displaying,
                                                 saving=args.logging.saving,
                                                 debug=args.logging.debug)
        else:
            self.local_logger = None

        if self.rank == 0 and args.logging.use_wandb:
            assert wandb.login(key=args.logging.wandb_api_key)
            self.wandb_logger = wandb.init(
                entity=args.logging.wandb_entity,
                project=args.logging.wandb_project,
                name=args.logging.run_name,
                config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
                settings=wandb.Settings(init_timeout=300),
                dir=os.environ.get("WANDB_DIR", None) # it is recommended to set WANDB_DIR env variable
            )
        else:
            self.wandb_logger = None

    def info(self, msg: str):
        """Log an info message only to the local logger."""
        if self.local_logger is not None:
            self.local_logger.info(msg)

    def log(self, data: dict, step=None):
        """Log key-value pairs to local logger and wandb logger (if enabled)."""
        if self.local_logger is not None or self.wandb_logger is not None:
            msg = []
            for k, v in sorted(data.items()):
                if isinstance(v, (int, float, str)):
                    msg.append(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}')
                elif isinstance(v, matplotlib.figure.Figure):
                    data[k] = wandb.Image(v)
                elif isinstance(v, np.ndarray) and v.ndim == 1:
                    data[k] = wandb.Histogram(v)
            if msg and self.local_logger is not None:
                self.local_logger.info((f"Step {step}: " if step is not None else "") + ', '.join(msg))
            if self.wandb_logger is not None:
                self.wandb_logger.log(data, step=step)

    def close(self, exit_code=0):
        """Close the wandb logger."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish(exit_code=exit_code)
        if dist.is_initialized():
            dist.destroy_process_group()
