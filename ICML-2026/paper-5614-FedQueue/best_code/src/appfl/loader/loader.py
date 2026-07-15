from __future__ import annotations

from typing import Any

from appfl.loader.data.customparser import fetch_custom_dataset
from appfl.loader.data.externalparser import fetch_hf_dataset
from appfl.loader.data.leafparser import fetch_leaf
from appfl.loader.data.torchaudioparser import fetch_torchaudio_dataset
from appfl.loader.data.torchvisionparser import fetch_torchvision_dataset

LEAF_DATASETS = {"FEMNIST", "SHAKESPEARE", "SENT140", "CELEBA", "REDDIT"}


def _adapt_dataset_config(config: Any, num_clients: int) -> Any:
    config.num_clients = int(num_clients)
    config.partition_type = config.partition
    config.split_type = config.partition

    for key, value in dict(config.partition_kwargs).items():
        setattr(config, key, value)
    return config


def load_and_split_dataset(config: Any, num_clients: int):
    """Load a dataset and return client datasets, server dataset, and metadata."""
    config = _adapt_dataset_config(config, num_clients)
    mode = str(getattr(config, "dataset_backend", "torchvision")).strip().lower()
    dataset_name = str(config.dataset_name)
    dataset_upper = dataset_name.upper()

    if mode == "custom":
        return fetch_custom_dataset(config)
    if mode == "hf":
        return fetch_hf_dataset(config)
    if mode == "torchvision":
        return fetch_torchvision_dataset(config)
    if mode == "torchaudio":
        return fetch_torchaudio_dataset(config)
    if mode == "leaf":
        assert config.partition_type == "pre", (
            "Only pre-partitioned LEAF datasets are supported."
        )
        assert dataset_upper in LEAF_DATASETS, (
            f"Unsupported LEAF dataset: {dataset_name}"
        )
        return fetch_leaf(config)

    raise ValueError(
        "dataset_backend must be one of: custom, hf, torchvision, torchaudio, leaf."
    )
