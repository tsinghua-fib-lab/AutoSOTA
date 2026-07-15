from gridfm_graphkit.datasets.powergrid_datamodule import LitGridDataModule
from gridfm_graphkit.io.param_handler import NestedNamespace

import yaml
import numpy as np
import random

import os


def get_gridfm_graphkit_dataset(
    num_clients: int,
    client_id: int,
):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    config_path = "./resources/configs/grid/gridfm_graphkit.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config_args = NestedNamespace(**config_dict)

    home_dir = os.environ.get("HOME")

    data_path = os.path.join(home_dir, "data")

    data_module = LitGridDataModule(config_args, data_path)
    data_module.setup("train")

    train_dataset = data_module.train_dataset_multi
    val_dataset = data_module.val_dataset_multi

    traing_dataset_size = len(train_dataset)
    validation_dataset_size = len(val_dataset)

    train_indices = list(range(traing_dataset_size))
    random.shuffle(train_indices)

    val_indices = list(range(validation_dataset_size))
    random.shuffle(val_indices)

    client_train_dataset_base_size = traing_dataset_size // num_clients
    client_train_dataset_remainder = traing_dataset_size % num_clients
    client_val_dataset_base_size = validation_dataset_size // num_clients
    client_val_dataset_remainder = validation_dataset_size % num_clients

    start_idx = client_id * client_train_dataset_base_size + min(
        client_id, client_train_dataset_remainder
    )
    subset_size = client_train_dataset_base_size + (
        1 if client_id < client_train_dataset_remainder else 0
    )
    train_subset_indices = train_indices[start_idx : start_idx + subset_size]

    if isinstance(train_dataset, np.ndarray):
        client_train_dataset = train_dataset[train_subset_indices]
    else:
        client_train_dataset = [train_dataset[idx] for idx in train_subset_indices]

    start_idx = client_id * client_val_dataset_base_size + min(
        client_id, client_val_dataset_remainder
    )
    subset_size = client_val_dataset_base_size + (
        1 if client_id < client_val_dataset_remainder else 0
    )
    val_subset_indices = val_indices[start_idx : start_idx + subset_size]

    if isinstance(val_dataset, np.ndarray):
        client_val_dataset = val_dataset[val_subset_indices]
    else:
        client_val_dataset = [val_dataset[idx] for idx in val_subset_indices]

    return client_train_dataset, client_val_dataset
