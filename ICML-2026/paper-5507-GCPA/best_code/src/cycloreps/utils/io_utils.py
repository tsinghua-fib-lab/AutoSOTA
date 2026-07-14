import logging
import os
import random
from contextlib import contextmanager
from typing import Optional
from latentis.space._base import Space
from latentis.data import DATA_DIR
from cycloreps import PROJECT_ROOT, FULL_TO_SHORT_NAMES
import torch

import dotenv
import numpy as np

pylogger = logging.getLogger(__name__)


def subsample_spaces(spaces, split, subsample_size):

    ref_name = list(spaces[split].keys())[0]

    num_samples = spaces[split][ref_name].shape[0]
    random_indices = torch.randperm(num_samples)[:subsample_size]

    for name, space in spaces[split].items():
        if space.shape[0] > subsample_size:
            print(f"Truncating {name} to {subsample_size} samples.")
            spaces[split][name] = space[random_indices]

    return spaces


def load_space(dataset_name, encoder_name, split, data_dir=None):

    pylogger.info(
        f"Loading space for dataset: {dataset_name}, encoder: {encoder_name}, split: {split}"
    )

    if data_dir:
        path = (
            data_dir / dataset_name / "encodings" / encoder_name.replace("/", "-") / split
        )
    else:
        path = (
            DATA_DIR / dataset_name / "encodings" / encoder_name.replace("/", "-") / split
        )
    space = Space.load_from_disk(path)
    space.encoder_name = FULL_TO_SHORT_NAMES[encoder_name]
    space.dataset_name = dataset_name

    return space
