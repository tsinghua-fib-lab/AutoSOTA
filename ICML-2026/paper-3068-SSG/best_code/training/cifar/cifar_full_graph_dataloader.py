"""
CIFAR100 full-graph GraphProbPrune dataset.
Supports loading a single full-graph file without class-wise clustering.

Optimized version:
- Uses precomputed edge_scores directly without loading the adjacency matrix.
- Substantially improves loading speed.

Differences from cifar_graph_dataloader.py:
- Loads a single graph.pkl file rather than multiple class_*/cluster_*.graph.pkl files from a directory.
- Matches samples directly by idx.
"""

import os
import torch
import numpy as np
import torchvision
import math
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
from typing import Dict, Iterator, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
import pickle


def load_full_graph(graph_path: str, verbose: bool = True) -> dict:
    """
    Load a single full-graph file.

    Args:
        graph_path: path to the graph.pkl file
        verbose: whether to print loading time

    Returns:
        graph_data: graph-data dictionary
    """
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file does not exist: {graph_path}")

    if verbose:
        print(f"Loading graph files: {graph_path}")
        t0 = time.perf_counter()

    with open(graph_path, 'rb') as f:
        graph_data = pickle.load(f)

    if verbose:
        t1 = time.perf_counter()
        print(f"[Time] Loading graph file time: {t1 - t0:.2f} s")
        print(f"Graph contains {graph_data['num_nodes']} nodes")

    return graph_data


def create_score_array_from_full_graph(
    graph_data: dict,
    dataset_length: int,
    fill_value: float = 0.0,
    verbose: bool = True
) -> np.ndarray:
    """
    Create a score array from graph data.

    Args:
        graph_data: graph-data dictionary; should contain 'edge_scores' or 'nodes' plus 'adj_matrix'
        dataset_length: dataset length
        fill_value: fill value for unmatched samples
        verbose: whether to print information

    Returns:
        score_array: np.ndarray, shape=(dataset_length,)
    """
    if verbose:
        t0 = time.perf_counter()

    score_array = np.full(dataset_length, fill_value, dtype=np.float32)

    # Prefer precomputed edge_scores.
    if 'edge_scores' in graph_data:
        scores = graph_data['edge_scores']
        matched = 0
        for idx, score in scores.items():
            if 0 <= idx < dataset_length:
                score_array[idx] = float(score)
                matched += 1

        if verbose:
            t1 = time.perf_counter()
            print(f"[Time] Time to create array from precomputed edge_scores: {t1 - t0:.2f} s")
            print(f"Created score array with shape: {score_array.shape}")
            print(f"   - matched valid samples: {matched}")
            print(f"   - unmatched samples: {dataset_length - matched} (filled with {fill_value})")

    elif 'adj_matrix' in graph_data:
        # Backward compatibility: compute from the adjacency matrix.
        if verbose:
            print("No precomputed edge_scores in the graph; computing from the adjacency matrix...")

        nodes = graph_data['nodes']
        adj_matrix = graph_data['adj_matrix']
        N = len(nodes)

        W = adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') else adj_matrix

        for i in tqdm(range(N), desc="Computing node scores"):
            weights = []
            for j in range(N):
                if i != j and W[i, j] > 0:
                    weights.append(W[i, j])

            if len(weights) > 0:
                avg_score = float(np.mean(weights))
            else:
                avg_score = 0.0

            idx = nodes[i]['idx']
            if 0 <= idx < dataset_length:
                score_array[idx] = avg_score

        if verbose:
            t1 = time.perf_counter()
            print(f"[Time] Time to compute scores from the adjacency matrix: {t1 - t0:.2f} s")

    else:
        raise ValueError("Graph data contains neither 'edge_scores' nor 'adj_matrix'; cannot create the score array.")

    return score_array


class CIFARFullGraphProbPrune(Dataset):
    """
    CIFAR100 full-graph GraphProbPrune dataset.

    Differences from CIFARGraphProbPrune:
    - Load a single full-graph file instead of multiple clustered graph files from a directory.
    - Intended for ablation experiments without class-wise clustering.
    """

    def __init__(
        self,
        dataset,
        graph_path: str,
        ratio: float = 0.5,
        num_epoch: int = None,
        delta: float = None,
        mode: str = "prob",
        verbose: bool = True
    ):
        """
        Args:
            dataset: torchvision.datasets.CIFAR100 instance
            graph_path: full-graph file path (single .pkl file)
            ratio: pruning ratio
            num_epoch: total number of training epochs
            delta: annealing parameter
            mode: selection mode: 'prob' or 'topk'
            verbose: whether to print loading information
        """
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

        # Initialize score arrays.
        self.node_scores = np.ones([len(self.dataset)])
        self.final_scores = np.ones([len(self.dataset)])
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

        # Time graph loading.
        t_load_start = time.perf_counter()

        # Load the full graph.
        graph_data = load_full_graph(graph_path, verbose=verbose)

        # Create the score array.
        self.edge_scores = create_score_array_from_full_graph(
            graph_data=graph_data,
            dataset_length=len(self.dataset),
            fill_value=0.0,
            verbose=verbose
        )

        t_load_end = time.perf_counter()
        self.load_time = t_load_end - t_load_start

        if verbose:
            print(f"prune selection mode: {self.mode}")
            print(f"dataset length: {len(self.dataset)}")
            print(f"[Time] total graph loading time: {self.load_time:.2f} s")

    def __setscore__(self, indices, values):
        """Update dynamic node scores during training."""
        self.node_scores[indices] = values
        self.final_scores[indices] = self.node_scores[indices] - self.edge_scores[indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        return data, target, index, weight

    def prune(self, xiuzheng_ratio=None):
        """
        Run pruning and return the sample indices used in the current epoch.

        Args:
            xiuzheng_ratio: optional annealed ratio (CODE-02). If None, uses self.ratio.
        """
        t0 = time.perf_counter()

        # Select samples below the 99th percentile of final_scores as well learned.
        b = self.final_scores < np.percentile(self.final_scores, 99)
        well_learned_samples = np.where(b)[0]
        pruned_samples = np.where(np.invert(b))[0]

        if well_learned_samples.shape == (0,) or self.final_scores.min() == self.final_scores.max():
            print('Cut {} samples for next iteration'.format(len(self.dataset) - len(pruned_samples)))
            self.save_num += len(self.dataset) - len(pruned_samples)
            np.random.shuffle(pruned_samples)
            self.prune_time = time.perf_counter() - t0
            return pruned_samples

        # Select samples according to the configured mode.
        prob = torch.softmax(torch.from_numpy(self.final_scores[well_learned_samples]) / 5.0, dim=0).numpy()
        ratio = xiuzheng_ratio if xiuzheng_ratio is not None else self.ratio
        k = int(ratio * len(well_learned_samples))

        if self.mode == 'prob':
            selected = np.random.choice(well_learned_samples, k, p=prob)
        elif self.mode == 'topk':
            topk_idx = np.argsort(prob)[-k:][::-1]
            selected = well_learned_samples[topk_idx]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.reset_weights()
        if len(selected) > 0:
            self.weights[selected] = 1 / self.ratio
            pruned_samples = np.append(pruned_samples, selected)

        print('Cut {} samples for next iteration'.format(len(self.dataset) - len(pruned_samples)))
        self.save_num += len(self.dataset) - len(pruned_samples)
        np.random.shuffle(pruned_samples)

        self.prune_time = time.perf_counter() - t0
        return pruned_samples

    def pruning_sampler(self):
        return InfoBatchSampler(self, self.num_epoch, self.delta)

    def no_prune(self):
        samples = list(range(len(self.dataset)))
        np.random.shuffle(samples)
        return samples

    def mean_score(self):
        return self.final_scores.mean()

    def normal_sampler_no_prune(self):
        return InfoBatchSampler(self.no_prune)

    def get_weights(self, indexes):
        return self.weights[indexes]

    def total_save(self):
        return self.save_num

    def reset_weights(self):
        self.weights = np.ones(len(self.dataset))

    def total_time_cost(self):
        """Maintain compatibility with the ImageNet interface."""
        return 0

    def get_load_time(self):
        """Return graph loading time."""
        return self.load_time

    def get_last_prune_time(self):
        """Return the most recent pruning operation time."""
        return getattr(self, 'prune_time', 0.0)


class InfoBatchSampler:
    """InfoBatch sampler."""

    def __init__(self, infobatch_dataset, num_epoch=float('inf'), delta=1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.start_prune = 2  # CODE-02: start annealing after epoch 2
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed += 1
        if self.seed > self.stop_prune:
            if self.seed <= self.stop_prune + 1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        elif self.seed < self.start_prune:
            # CODE-02: No pruning in first few epochs
            self.seq = self.infobatch_dataset.no_prune()
        else:
            # CODE-02: Anneal pruning ratio from ratio -> 1.0 over training
            xiuzheng_ratio = min(1.0, self.infobatch_dataset.ratio +
                (1 - self.infobatch_dataset.ratio) *
                (1 - (self.seed - self.start_prune) / (self.stop_prune - self.start_prune)))
            self.seq = self.infobatch_dataset.prune(xiuzheng_ratio)
        self.ite = iter(self.seq)
        self.new_length = len(self.seq)

    def __next__(self):
        try:
            return next(self.ite)
        except StopIteration:
            self.reset()
            raise StopIteration

    def __len__(self):
        return len(self.seq)

    def __iter__(self):
        self.ite = iter(self.seq)
        return self

    def set_epoch(self, epoch):
        """Maintain compatibility with the DistributedSampler interface."""
        np.random.seed(epoch)


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output


def is_master():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    if torch.distributed.get_rank() == 0:
        return True
    return False


def split_index(t):
    """Split large indices into low 15 bits and high bits for distributed index transfer."""
    low_mask = 0b111111111111111
    low = torch.tensor([x & low_mask for x in t])
    high = torch.tensor([(x >> 15) & low_mask for x in t])
    return low, high


def recombine_index(low, high):
    """Recombine split indices."""
    original_tensor = torch.tensor([(high[i] << 15) + low[i] for i in range(len(low))])
    return original_tensor