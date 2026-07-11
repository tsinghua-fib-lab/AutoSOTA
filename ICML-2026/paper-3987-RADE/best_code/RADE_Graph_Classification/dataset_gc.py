# graph_classification/dataset_gc.py
from __future__ import annotations

import os
import zipfile
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import LRGBDataset
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    to_undirected,
)
from sklearn.model_selection import StratifiedKFold

try:
    from ogb.graphproppred import PygGraphPropPredDataset
except Exception:
    PygGraphPropPredDataset = None


class GraphIdentityDataset:
    def __init__(self, base_dataset, node_offsets=None, *, attach_node_ids: bool = True):
        self.base_dataset = base_dataset
        self.node_offsets = [int(offset) for offset in node_offsets] if node_offsets is not None else []
        self.attach_node_ids = bool(attach_node_ids)
        self.global_num_nodes_total = int(node_offsets[-1]) if node_offsets else 0

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        if hasattr(data, "clone"):
            data = data.clone()
        num_nodes = int(data.num_nodes)
        if self.attach_node_ids:
            offset = int(self.node_offsets[int(idx)])
            data.n_id = torch.arange(num_nodes, dtype=torch.long) + offset
        data.graph_id = torch.tensor([int(idx)], dtype=torch.long)
        return data

    def __getattr__(self, name):
        return getattr(self.base_dataset, name)


# -------------------------
# Helpers
# -------------------------

def _canon_name(name: str) -> str:
    n = str(name).lower().strip()
    aliases = {
        "mutag": "mutag",
        "proteins": "proteins",
        "imdb-b": "imdb-b",
        "imdb-binary": "imdb-b",
        "imdb-m": "imdb-m",
        "imdb-multi": "imdb-m",
        "imdb-mult": "imdb-m",
        # OGBG names
        "ogbg-molhiv": "ogbg-molhiv",
        "molhiv": "ogbg-molhiv",
        "ogbg-molpcba": "ogbg-molpcba",
        "molpcba": "ogbg-molpcba",
        "peptides-func": "peptides-func",
        "peptides": "peptides-func",
    }
    return aliases.get(n, n)


def _tu_official_name(canon: str) -> str:
    if canon == "mutag":
        return "MUTAG"
    if canon == "proteins":
        return "PROTEINS"
    if canon == "imdb-b":
        return "IMDB-BINARY"
    if canon == "imdb-m":
        return "IMDB-MULTI"
    raise ValueError(f"Not a TU dataset canon name: {canon}")


def _powerful_gnns_tu_name(canon: str) -> str:
    if canon == "mutag":
        return "MUTAG"
    if canon == "proteins":
        return "PROTEINS"
    if canon == "imdb-b":
        return "IMDBBINARY"
    if canon == "imdb-m":
        return "IMDBMULTI"
    raise ValueError(f"Not a TU dataset canon name: {canon}")

def _load_lrgb_peptides(root: str, name: str):
    # name can be "Peptides-func" / PyG lowercases internally
    train_ds = LRGBDataset(root=root, name=name, split="train")
    val_ds   = LRGBDataset(root=root, name=name, split="val")
    test_ds  = LRGBDataset(root=root, name=name, split="test")

    # Concatenate into one dataset (keeps PyG Dataset interface)
    dataset = train_ds + val_ds + test_ds

    n_tr, n_va, n_te = len(train_ds), len(val_ds), len(test_ds)
    split_idx = {
        "train": torch.arange(0, n_tr, dtype=torch.long),
        "valid": torch.arange(n_tr, n_tr + n_va, dtype=torch.long),
        "test":  torch.arange(n_tr + n_va, n_tr + n_va + n_te, dtype=torch.long),
    }
    return dataset, split_idx


def _random_split_graphs(
    num_graphs: int,
    train_prop: float,
    valid_prop: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(num_graphs, generator=g)

    n_train = int(round(train_prop * num_graphs))
    n_valid = int(round(valid_prop * num_graphs))

    train_idx = perm[:n_train].to(torch.long)
    valid_idx = perm[n_train:n_train + n_valid].to(torch.long)
    test_idx = perm[n_train + n_valid:].to(torch.long)

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def _graph_node_offsets(dataset) -> list[int]:
    offsets = [0]
    running = 0
    for idx in range(len(dataset)):
        running += int(dataset[idx].num_nodes)
        offsets.append(running)
    return offsets


def attach_graph_identity(dataset, *, attach_node_ids: bool = True):
    offsets = _graph_node_offsets(dataset) if attach_node_ids else None
    wrapped = GraphIdentityDataset(dataset, offsets, attach_node_ids=attach_node_ids)
    return wrapped, int(wrapped.global_num_nodes_total)


def _read_powerful_gnns_text(raw_root: str, dataset_name: str) -> Tuple[List[str], str]:
    root = os.path.abspath(os.path.expanduser(raw_root))
    txt_rel = os.path.join("dataset", dataset_name, f"{dataset_name}.txt")
    extracted_path = os.path.join(root, txt_rel)
    if os.path.exists(extracted_path):
        with open(extracted_path, "r", encoding="utf-8") as handle:
            return handle.readlines(), extracted_path

    zip_path = os.path.join(root, "dataset.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as archive:
            zip_member = txt_rel.replace(os.sep, "/")
            with archive.open(zip_member, "r") as handle:
                text = handle.read().decode("utf-8")
        return text.splitlines(keepends=True), f"{zip_path}:{zip_member}"

    raise FileNotFoundError(
        "Could not find TU raw data. Expected either "
        f"'{extracted_path}' or '{zip_path}'."
    )


def load_powerful_gnns_tu_dataset(
    raw_root: str,
    name: str,
    *,
    attach_graph_ids: bool = False,
    attach_global_node_ids: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load TU graphs with the same raw parser semantics as powerful-gnns/util.py.

    The returned dataset is a list of PyG Data objects so the RADE training code
    can keep using the existing model and augmentation path unchanged.
    """
    canon = _canon_name(name)
    if canon not in {"mutag", "proteins", "imdb-b", "imdb-m"}:
        raise ValueError(f"powerful-gnns TU loading only supports TU datasets. Got '{name}'.")

    raw_name = _powerful_gnns_tu_name(canon)
    lines, source = _read_powerful_gnns_text(raw_root, raw_name)
    cursor = 0
    num_graphs = int(lines[cursor].strip())
    cursor += 1

    label_to_index: Dict[int, int] = {}
    raw_graphs: List[Dict[str, Any]] = []
    feature_to_index: Dict[int, int] = {}
    degree_as_tag = canon in {"imdb-b", "imdb-m"}

    for _ in range(num_graphs):
        num_nodes, raw_label = [int(value) for value in lines[cursor].strip().split()]
        cursor += 1
        if raw_label not in label_to_index:
            label_to_index[raw_label] = len(label_to_index)

        node_tags: List[int] = []
        neighbors = [set() for _ in range(num_nodes)]
        for node_idx in range(num_nodes):
            row = lines[cursor].strip().split()
            cursor += 1
            num_neighbors = int(row[1])
            adjacency_end = num_neighbors + 2
            node_tag_raw = int(row[0])
            if node_tag_raw not in feature_to_index:
                feature_to_index[node_tag_raw] = len(feature_to_index)
            node_tags.append(feature_to_index[node_tag_raw])

            for neighbor_raw in row[2:adjacency_end]:
                neighbor_idx = int(neighbor_raw)
                neighbors[node_idx].add(neighbor_idx)
                neighbors[neighbor_idx].add(node_idx)

        if degree_as_tag:
            node_tags = [len(neighbor_set) for neighbor_set in neighbors]

        undirected_edges = []
        for src, neighbor_set in enumerate(neighbors):
            for dst in neighbor_set:
                if src < dst:
                    undirected_edges.append((src, dst))

        directed_edges = []
        for src, dst in undirected_edges:
            directed_edges.append((src, dst))
            directed_edges.append((dst, src))

        edge_index = (
            torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
            if directed_edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        raw_graphs.append(
            {
                "node_tags": node_tags,
                "edge_index": edge_index,
                "label": label_to_index[raw_label],
                "num_nodes": num_nodes,
            }
        )

    tagset = set()
    for graph in raw_graphs:
        tagset = tagset.union(set(graph["node_tags"]))
    tag_list = list(tagset)
    tag_to_index = {tag: idx for idx, tag in enumerate(tag_list)}

    dataset: List[Data] = []
    for graph_idx, graph in enumerate(raw_graphs):
        node_tags = graph["node_tags"]
        x = torch.zeros((len(node_tags), len(tag_list)), dtype=torch.float32)
        x[torch.arange(len(node_tags)), torch.tensor([tag_to_index[tag] for tag in node_tags])] = 1.0
        data = Data(
            x=x,
            edge_index=graph["edge_index"],
            y=torch.tensor([int(graph["label"])], dtype=torch.long),
            num_nodes=int(graph["num_nodes"]),
        )
        data.graph_id = torch.tensor([graph_idx], dtype=torch.long)
        dataset.append(data)

    meta = {
        "dataset_family": "tu",
        "dataset_name": raw_name,
        "task_type": "multiclass",
        "metric": "acc",
        "loss": "ce",
        "evaluator_name": None,
        "in_dim": int(dataset[0].x.size(-1)),
        "out_dim": int(len(label_to_index)),
        "num_classes": int(len(label_to_index)),
        "y_format": "long_class_index",
        "raw_source": source,
        "degree_as_tag": bool(degree_as_tag),
    }
    if attach_global_node_ids:
        dataset_wrapped, total_nodes = attach_graph_identity(dataset)
        meta["global_num_nodes_total"] = int(total_nodes)
        return dataset_wrapped, meta
    if attach_graph_ids:
        dataset_wrapped, _ = attach_graph_identity(dataset, attach_node_ids=False)
        return dataset_wrapped, meta
    return dataset, meta


def make_powerful_gnns_tu_folds(dataset, *, seed: int) -> List[Dict[str, torch.Tensor]]:
    labels = []
    for graph in dataset:
        y = graph.y
        labels.append(int(y.view(-1)[0].item()))

    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=int(seed))
    folds: List[Dict[str, torch.Tensor]] = []
    for train_idx, valid_idx in splitter.split(np.zeros(len(labels)), labels):
        folds.append(
            {
                "train": torch.as_tensor(train_idx, dtype=torch.long),
                "valid": torch.as_tensor(valid_idx, dtype=torch.long),
            }
        )
    return folds


def _sanitize_graph(
    data: Data,
    *,
    make_undirected_edges: bool,
    remove_loops: bool = True,
) -> Data:
    edge_index = data.edge_index.to(torch.long)
    edge_attr = getattr(data, "edge_attr", None)

    if remove_loops:
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # For molecules, do NOT call to_undirected unless you explicitly want it.
    if make_undirected_edges:
        if edge_attr is None:
            edge_index = to_undirected(edge_index, num_nodes=int(data.num_nodes))
        else:
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=int(data.num_nodes))

    if edge_attr is None:
        edge_index = coalesce(edge_index, num_nodes=int(data.num_nodes))
    else:
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=int(data.num_nodes))
        data.edge_attr = edge_attr

    data.edge_index = edge_index
    return data


# -------------------------
# Public loader
# -------------------------

def load_gc_dataset(
    root: str,
    name: str,
    *,
    seed: int = 0,
    train_prop: float = 0.8,
    valid_prop: float = 0.1,
    attach_graph_ids: bool = False,
    attach_global_node_ids: bool = False,
) -> Tuple[Any, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Returns:
      dataset: PyG dataset
      split_idx: dict with {'train','valid','test'} graph indices (LongTensor, CPU)
      meta: dict with:
        - in_dim, out_dim, loss, metric, evaluator_name
        - plus dataset/task fields
    """
    canon = _canon_name(name)

    if canon in {"mutag", "proteins", "imdb-b", "imdb-m"}:
        raise ValueError(
            "TU datasets use the dedicated reference 10-fold workflow. "
            "Call load_powerful_gnns_tu_dataset(...) with make_powerful_gnns_tu_folds(...), "
            "or run main_gc.py directly."
        )

    # -------------------------
    # OGBG datasets (official split)
    # -------------------------
    if canon in {"ogbg-molhiv", "ogbg-molpcba"}:
        if PygGraphPropPredDataset is None:
            raise ImportError("OGB is not available. Install 'ogb' to use ogbg-molhiv/molpcba.")

        root_dir = os.path.join(root, "ogbg")
        ds = PygGraphPropPredDataset(name=canon, root=root_dir)

        def ogb_transform(data: Data) -> Data:
            if data.y is not None:
                data.y = data.y.to(torch.float32)
            return data

        ds.transform = ogb_transform

        split = ds.get_idx_split()
        split_idx = {k: torch.as_tensor(v, dtype=torch.long) for k, v in split.items()}

        d0 = ds[int(split_idx["train"][0])]
        in_dim = int(d0.x.size(-1))
        out_dim = int(ds.num_tasks)

        if canon == "ogbg-molhiv":
            meta = {
                "dataset_family": "ogbg",
                "dataset_name": canon,
                "task_type": "binary",
                "metric": "rocauc",
                "loss": "bce",
                "evaluator_name": canon,
                "in_dim": in_dim,
                "out_dim": out_dim,  # typically 1
                "num_tasks": out_dim,
                "y_format": "float_with_possible_missing",
            }
        else:
            meta = {
                "dataset_family": "ogbg",
                "dataset_name": canon,
                "task_type": "multilabel",
                "metric": "ap",
                "loss": "bce",
                "evaluator_name": canon,
                "in_dim": in_dim,
                "out_dim": out_dim,  # 128
                "num_tasks": out_dim,
                "y_format": "float_with_nan_or_minus1_missing",
            }

        if attach_global_node_ids:
            ds, total_nodes = attach_graph_identity(ds)
            meta["global_num_nodes_total"] = int(total_nodes)
        elif attach_graph_ids:
            ds, _ = attach_graph_identity(ds, attach_node_ids=False)
        return ds, split_idx, meta

    if canon in ["peptides-func"]:
        pyg_name = "Peptides-func"
        dataset, split_idx = _load_lrgb_peptides(root, pyg_name)

        def pep_transform(data: Data) -> Data:
            if getattr(data, "y", None) is not None:
                data.y = data.y.to(torch.float32)
            data.edge_index = data.edge_index.to(torch.long)
            if hasattr(data, "edge_attr") and (data.edge_attr is not None) and (
            not torch.is_floating_point(data.edge_attr)):
                data.edge_attr = data.edge_attr.to(torch.float32)
            return data

        dataset.transform = pep_transform

        in_dim = dataset[0].x.size(-1)
        out_dim = 10
        meta = {"in_dim": in_dim, "out_dim": out_dim, "task": "multilabel", "loss": "bce", "metric": "ap"}
        if attach_global_node_ids:
            dataset, total_nodes = attach_graph_identity(dataset)
            meta["global_num_nodes_total"] = int(total_nodes)
        elif attach_graph_ids:
            dataset, _ = attach_graph_identity(dataset, attach_node_ids=False)
        return dataset, split_idx, meta

    raise ValueError(
        f"Unsupported dataset: {name}. "
        f"Supported: mutag, proteins, imdb-b, imdb-m, ogbg-molhiv, ogbg-molpcba, peptides-func."
    )

