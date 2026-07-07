# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#

import os
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GPSConv
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import ( Planetoid,
                                       WebKB,
                                       Actor,
                                       HeterophilousGraphDataset,
                                       Airports, )
from torch_geometric.utils import to_undirected, to_networkx

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
Convert a (2, E) edge_index (possibly with duplicates / both directions)
into a unique list of undirected edges (M, 2) with a < b.
"""
def canonical_undirected_edges(edge_index: torch.Tensor) -> np.ndarray:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    seen = set()
    edges = []

    for u, v in zip(src, dst):
        if u == v:
            continue

        a, b = (int(u), int(v)) if u < v else (int(v), int(u))

        if (a, b) in seen:
            continue

        seen.add((a, b))
        edges.append((a, b))

    return np.asarray(edges, dtype=np.int64)

"""
Randomly split undirected edges (M, 2) into train/val/test (disjoint).
"""
def split_edges_undirected(edges: np.ndarray, seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1,) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert edges.ndim == 2 and edges.shape[1] == 2

    rng = np.random.RandomState(seed)
    perm = rng.permutation(edges.shape[0])
    edges_shuffled = edges[perm]

    n_total = edges_shuffled.shape[0]
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_edges = edges_shuffled[:n_train]
    val_edges = edges_shuffled[n_train : n_train + n_val]
    test_edges = edges_shuffled[n_train + n_val :]

    return train_edges, val_edges, test_edges

"""
Simple random node splits: train/val/test masks.
"""
def make_node_splits(num_nodes: int, seed: int, train_ratio: float = 0.6, val_ratio: float = 0.2) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_nodes)

    n_train = int(train_ratio * num_nodes)
    n_val = int(val_ratio * num_nodes)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask

"""
Sample undirected negative edges (M, 2), disjoint from forbid set.

Optionally restrict endpoints to allowed_nodes_* subsets.

forbid: set of (u, v) with u < v representing any positive edges.
"""
def sample_negative_edges(num_nodes: int, num_samples: int, forbid: set, rng: np.random.RandomState,
                          allowed_nodes_u: Optional[np.ndarray] = None, allowed_nodes_v: Optional[np.ndarray] = None) \
        -> np.ndarray:
    if allowed_nodes_u is None:
        allowed_nodes_u = np.arange(num_nodes, dtype=np.int64)
    if allowed_nodes_v is None:
        allowed_nodes_v = np.arange(num_nodes, dtype=np.int64)

    neg = set()

    while len(neg) < num_samples:
        u = int(allowed_nodes_u[rng.randint(len(allowed_nodes_u))])
        v = int(allowed_nodes_v[rng.randint(len(allowed_nodes_v))])

        if u == v:
            continue

        a, b = (u, v) if u < v else (v, u)

        if (a, b) in forbid or (a, b) in neg:
            continue

        neg.add((a, b))

    return np.asarray(list(neg), dtype=np.int64)

"""
Build edge_index for message passing from undirected train positives (M, 2):
Adds both directions (u, v) and (v, u), then deduplicates.
"""
def build_train_adjacency_from_edges(pos_train_edges: np.ndarray, num_nodes: int) -> torch.Tensor:
    pos_train_edges = np.asarray(pos_train_edges, dtype=np.int64)

    row = torch.from_numpy(pos_train_edges[:, 0])
    col = torch.from_numpy(pos_train_edges[:, 1])

    edge_index = torch.stack(
        [
            torch.cat([row, col]),
            torch.cat([col, row]),
        ],
        dim=0,
    )

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index

@dataclass
class ProtocolConfig:
    setting: str = "transductive"  # or "inductive"
    node_train_ratio: float = 0.6
    node_val_ratio: float = 0.2
    edge_train_ratio: float = 0.8
    edge_val_ratio: float = 0.1
    hidden: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    node_epochs: int = 200
    link_epochs: int = 200
    patience: int = 50
    barrier_steps: int = 11  # kept for compatibility (unused in direct barrier)
    probe_epochs: int = 200