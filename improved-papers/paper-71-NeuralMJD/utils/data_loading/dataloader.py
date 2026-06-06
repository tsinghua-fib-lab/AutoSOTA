import logging
import os
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DistributedSampler
import torch_geometric.loader 
from typing import Tuple

from .sp500 import get_sp500_dataset


seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)


def build_dataloader(dataset, batch_size: int, workers: int, is_ddp: bool):
    """
    Internal helper to construct a PyG dataloader.

    If DDP is enabled, uses a `DistributedSampler` and divides the batch size
    across processes. Otherwise, uses standard shuffling.
    """
    if is_ddp:
        sampler = DistributedSampler(dataset)
        batch_size_per_gpu = max(1, batch_size // dist.get_world_size())
        data_loader = torch_geometric.loader.DataLoader(dataset, sampler=sampler, batch_size=batch_size_per_gpu,
                                                        pin_memory=True, num_workers=workers)
    else:
        data_loader = torch_geometric.loader.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                        pin_memory=True, num_workers=workers)
    return data_loader



def load_data(config, dist_helper, eval_mode: bool = False) -> Tuple[object, object, object]:
    """
    Build training/validation/testing dataloaders based on the dataset name.

    Args:
        config: Experiment configuration.
        dist_helper: Distributed helper with DDP/DP flags.
        eval_mode: If True, use evaluation/test batch size.

    Returns:
        Tuple of (train_dl, val_dl, test_dl).
    """

    # init
    batch_size = config.test.batch_size if eval_mode else config.train.batch_size
    workers = config.train.workers if hasattr(config.train, 'workers') else min(6, os.cpu_count())
    dataset_nm = config.dataset.name

    if dataset_nm == 'sp500':
        train_dataset, val_dataset, test_dataset = get_sp500_dataset(config, eval_mode)
    else:
        raise ValueError("Invalid dataset name: {:s}".format(dataset_nm))

    # build pytorch dataloader and return
    eval_bs = batch_size
    train_dl = build_dataloader(train_dataset, batch_size=batch_size, workers=workers, is_ddp=dist_helper.is_ddp)
    val_dl = build_dataloader(val_dataset, batch_size=eval_bs, workers=workers, is_ddp=dist_helper.is_ddp)
    test_dl = build_dataloader(test_dataset, batch_size=eval_bs, workers=workers, is_ddp=dist_helper.is_ddp)

    logging.info("Training / validation / testing set size: {:d} / {:d} / {:d}".format(len(train_dataset), len(val_dataset), len(test_dataset), ))
    logging.info("Training / validation / testing dataloader batch size: {:d} / {:d} / {:d}".format(train_dl.batch_size, val_dl.batch_size, test_dl.batch_size))
    logging.info("Training / validation / testing dataloader length: {:d} / {:d} / {:d}".format(len(train_dl), len(val_dl), len(test_dl)))

    return train_dl, val_dl, test_dl
