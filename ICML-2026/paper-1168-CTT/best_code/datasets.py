# -----------------------------------------------------------------------------#
# Dataset loading & protocol wiring
# -----------------------------------------------------------------------------#

import os
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch
from torch_geometric.datasets import ( Planetoid,
                                       WebKB,
                                       Actor,
                                       HeterophilousGraphDataset,
                                       Airports, )
from torch_geometric.utils import to_undirected

from util import (canonical_undirected_edges, split_edges_undirected, make_node_splits,
                  sample_negative_edges, build_train_adjacency_from_edges, ProtocolConfig)

"""
Load a single-graph dataset and return (data, num_features, num_classes).

family in {"Planetoid", "WebKB", "Actor", "RomanEmpire", "Airports"}.
"""
def load_dataset(family: str, name: str, root: str,):
    family_l = family.lower()

    if family_l == "planetoid":
        ds = Planetoid(os.path.join(root, "Planetoid"), name)
    elif family_l == "webkb":
        ds = WebKB(os.path.join(root, "WebKB"), name)
    elif family_l == "actor":
        ds = Actor(os.path.join(root, "Actor"))
    elif family_l == "romanempire":
        ds = HeterophilousGraphDataset(
            os.path.join(root, "HeterophilousGraphDataset"),
            name="roman-empire",
        )
    elif family_l == "airports":
        ds = Airports(os.path.join(root, "Airports"), name=name)
    else:
        raise ValueError(f"Unknown family: {family}")

    data = ds[0]
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    num_features = ds.num_features
    num_classes = ds.num_classes

    return data, num_features, num_classes

"""
Implements:
    - node train/val/test splits
    - positive edge train/val/test splits (disjoint)
    - A_train from LP-train positives
    - negative edges for train/val/test from complement
    - inductive option: LP-train edges touching test nodes removed from A_train
"""
def make_splits_and_adjacency(data, seed: int, cfg: ProtocolConfig,) -> Dict[str, object]:
    num_nodes = data.num_nodes

    # Node splits
    train_mask, val_mask, test_mask = make_node_splits(
        num_nodes=num_nodes,
        seed=seed,
        train_ratio=cfg.node_train_ratio,
        val_ratio=cfg.node_val_ratio,
    )

    # All undirected positive edges
    edges_all = canonical_undirected_edges(data.edge_index)

    pos_train, pos_val, pos_test = split_edges_undirected(
        edges_all,
        seed=seed,
        train_ratio=cfg.edge_train_ratio,
        val_ratio=cfg.edge_val_ratio,
    )

    # Inductive: remove any LP-train edges that touch test nodes from A_train
    if cfg.setting == "inductive":
        test_nodes = np.where(train_mask.cpu().numpy() == 0)[0]  # test+val
        test_nodes_set = set(test_nodes.tolist())

        keep = [
            (int(u) not in test_nodes_set) and (int(v) not in test_nodes_set)
            for (u, v) in pos_train
        ]
        pos_train = pos_train[np.asarray(keep, dtype=bool)]

    # Build A_train from pos_train
    edge_index_train = build_train_adjacency_from_edges(pos_train_edges=pos_train, num_nodes=num_nodes,)

    # Negative sampling: forbid ALL positives (across splits)
    all_pos = np.concatenate([pos_train, pos_val, pos_test], axis=0)
    forbid = {(int(u), int(v)) for (u, v) in all_pos}

    rng = np.random.RandomState(seed + 12345)  # separate RNG for negatives

    if cfg.setting == "inductive":
        test_nodes = np.where(train_mask.cpu().numpy() == 0)[0]
        allowed_nodes = np.setdiff1d(
            np.arange(num_nodes, dtype=np.int64),
            test_nodes.astype(np.int64),
        )
        neg_train = sample_negative_edges(
            num_nodes=num_nodes,
            num_samples=pos_train.shape[0],
            forbid=forbid,
            rng=rng,
            allowed_nodes_u=allowed_nodes,
            allowed_nodes_v=allowed_nodes,
        )
    else:
        neg_train = sample_negative_edges(
            num_nodes=num_nodes,
            num_samples=pos_train.shape[0],
            forbid=forbid,
            rng=rng,
        )

    neg_val = sample_negative_edges(
        num_nodes=num_nodes,
        num_samples=pos_val.shape[0],
        forbid=forbid,
        rng=rng,
    )
    neg_test = sample_negative_edges(
        num_nodes=num_nodes,
        num_samples=pos_test.shape[0],
        forbid=forbid,
        rng=rng,
    )

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "pos_train": pos_train,
        "pos_val": pos_val,
        "pos_test": pos_test,
        "neg_train": neg_train,
        "neg_val": neg_val,
        "neg_test": neg_test,
        "edge_index_train": edge_index_train,
    }

def audit_splits(edge_index_train: torch.Tensor, pos_train: np.ndarray, pos_val: np.ndarray,
                 pos_test: np.ndarray, neg_train: np.ndarray, neg_val: np.ndarray, neg_test: np.ndarray,):
    # 1) No val/test positives in A_train
    ei = canonical_undirected_edges(edge_index_train)
    ei_set = {(int(u), int(v)) for (u, v) in ei}

    for name, pos in [("val", pos_val), ("test", pos_test)]:
        overlap = sum((int(u), int(v)) in ei_set for (u, v) in pos)
        # print(f"[AUDIT] {name} positives overlapping A_train: {overlap}")
        assert overlap == 0, f"Leakage: {name} positives found in A_train!"

    # 2) Negatives disjoint from all positives
    pos_all = { (int(u), int(v)) for (u, v) in np.concatenate([pos_train, pos_val, pos_test], axis=0) }

    for name, neg in [("train", neg_train), ("val", neg_val), ("test", neg_test)]:
        overlap = sum((int(u), int(v)) in pos_all for (u, v) in neg)
        # print(f"[AUDIT] {name} negatives overlapping positives: {overlap}")
        assert overlap == 0, f"Leakage: {name} negatives overlap positives!"
