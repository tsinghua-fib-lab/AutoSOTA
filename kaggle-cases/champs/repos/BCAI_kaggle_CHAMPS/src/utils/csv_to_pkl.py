## Copyright (c) 2017 Robert Bosch GmbH
## All rights reserved.
##
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import gzip
import bz2
import os

import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def get_datasets(root_dir, mode=""):
    root_dir = root_dir.strip("/")
    full_path = f"/torch_proc_train{mode}.pkl.gz"
    p1_path = f"/torch_proc_train{mode}_p1.pkl.gz"
    p2_path = f"/torch_proc_train{mode}_p2.pkl.gz"
    if mode != "":
        print(f"Using {mode} for dataset")

    if mode == "" and (not os.path.exists(root_dir + full_path)) and os.path.exists(root_dir + p1_path):
        # Found part 1 and 2 of the dataset. Concatenate them first
        with gzip.open(root_dir + f"/torch_proc_train{mode}_p1.pkl.gz", "rb") as f:
            print("Wait Patiently! Combining part 1 & 2 of the dataset so that we don't need to do it in the future.")
            D_train_part1 = pickle.load(f)
        with gzip.open(root_dir + f"/torch_proc_train{mode}_p2.pkl.gz", "rb") as f:
            D_train_part2 = pickle.load(f)
        D_train = tuple([torch.cat([D_train_part1[i], D_train_part2[i]], dim=0) for i in range(len(D_train_part1))])
        with gzip.open(root_dir+'/'+f"/torch_proc_train{mode}.pkl.gz", "wb") as f:
            pickle.dump(D_train, f, protocol=4)

    if os.path.exists(root_dir + full_path):
        print("Found gzipped dataset!")
        with gzip.open(root_dir + f"/torch_proc_train{mode}.pkl.gz", "rb") as f:
            train = TensorDataset(*pickle.load(f))
        if mode == "_full":
            return train, None
        with gzip.open(root_dir + f"/torch_proc_val{mode}.pkl.gz", "rb") as f:
            val = TensorDataset(*pickle.load(f))
        return train, val
    else:
        with bz2.open(root_dir + f"/torch_proc_train{mode}.pkl.bz2", "rb") as f:
            train = TensorDataset(*pickle.load(f))
        if mode == "_full":
            return train, None
        with bz2.open(root_dir + f"/torch_proc_val{mode}.pkl.bz2", "rb") as f:
            val = TensorDataset(*pickle.load(f))
        return train, val

def get_submission_set(root_dir, mode=""):
    full_path = "/torch_proc_submission{mode}.pkl.gz"
    if mode != "":
        print(f"Using {mode} for dataset")

    if os.path.exists(root_dir + full_path):
        with gzip.open(root_dir + f"/torch_proc_submission{mode}.pkl.gz", "rb") as f:
            submission = TensorDataset(*pickle.load(f))
    else:
        with bz2.open(root_dir + "/torch_proc_submission.pkl.bz2", "rb") as f:
            submission = TensorDataset(*pickle.load(f))
    return submission
