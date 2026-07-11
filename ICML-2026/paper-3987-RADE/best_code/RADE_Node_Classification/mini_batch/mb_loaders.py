# nc_minibatch/mb_loaders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

try:
    from torch_geometric.loader import NeighborLoader
except Exception as e:
    NeighborLoader = None
    _NEIGHBORLOADER_IMPORT_ERROR = e
else:
    _NEIGHBORLOADER_IMPORT_ERROR = None


@dataclass
class NeighborLoaderConfig:
    """
    Configuration for PyG NeighborLoader.

    num_neighbors:
      - length should typically match the number of message-passing layers
      - use small fanouts to control memory (e.g., (15,10,5))
      - setting a value to -1 means "all neighbors" at that hop (often defeats mini_batch benefits)

    batch sizes:
      - train should be smaller
      - eval can be larger (but still bounded by GPU memory)
    """
    num_neighbors: Tuple[int, ...] = (15, 10, 5)
    batch_size_train: int = 1024
    batch_size_eval: int = 4096
    shuffle_train: bool = True

    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2  # only used when num_workers > 0


def _as_index_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure split indices are a 1D LongTensor on CPU.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.dtype != torch.long:
        x = x.to(torch.long)
    return x.view(-1).cpu()


def build_neighbor_loaders(
    data,
    split_idx: Dict[str, torch.Tensor],
    *,
    cfg: NeighborLoaderConfig,
) -> Dict[str, object]:
    """
    Build NeighborLoader objects for train/valid/test splits.

    Important:
      - Keep the *full* graph `data` on CPU.
      - Each batch is moved to GPU inside the training/eval loop (batch.to(device)).
    """
    if NeighborLoader is None:
        raise RuntimeError(
            "Failed to import torch_geometric.loader.NeighborLoader. "
            "This usually means you are missing a required dependency "
            "(pyg-lib or torch-sparse) for neighbor sampling.\n"
            f"Original import error: {_NEIGHBORLOADER_IMPORT_ERROR}"
        )

    # Defensive: ensure split_idx keys exist
    for k in ("train", "valid", "test"):
        if k not in split_idx:
            raise KeyError(f"split_idx missing key '{k}'. Got keys={list(split_idx.keys())}")

    train_idx = _as_index_tensor(split_idx["train"])
    valid_idx = _as_index_tensor(split_idx["valid"])
    test_idx = _as_index_tensor(split_idx["test"])

    # Worker/prefetch settings: PyTorch disallows prefetch_factor when num_workers==0
    common_kwargs = dict(
        num_neighbors=list(cfg.num_neighbors),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        persistent_workers=bool(cfg.persistent_workers) if int(cfg.num_workers) > 0 else False,
    )
    if int(cfg.num_workers) > 0:
        common_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    # Train loader (shuffle seeds)
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        batch_size=int(cfg.batch_size_train),
        shuffle=bool(cfg.shuffle_train),
        **common_kwargs,
    )

    # Eval loaders (do not shuffle)
    valid_loader = NeighborLoader(
        data,
        input_nodes=valid_idx,
        batch_size=int(cfg.batch_size_eval),
        shuffle=False,
        **common_kwargs,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=test_idx,
        batch_size=int(cfg.batch_size_eval),
        shuffle=False,
        **common_kwargs,
    )

    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
