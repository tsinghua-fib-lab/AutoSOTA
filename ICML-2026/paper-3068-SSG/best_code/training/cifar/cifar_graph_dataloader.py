"""
GraphProbPrune dataset specialized for CIFAR100.
Uses index-based matching rather than filenames for preloaded CIFAR100 datasets.

Key differences:
- ImageNet matches samples with image_name (file path)
- CIFAR100 matches samples with idx (dataset index)
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


def load_all_graph_files(graph_dir: str) -> List[dict]:
    """
    Load all *.graph.pkl files under the specified directory.

    Returns: list whose entries contain 'file_path' and 'graph_data'
    """
    graph_dir = Path(graph_dir)
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory does not exist: {graph_dir}")

    graph_files = sorted(graph_dir.rglob("*.graph.pkl"))
    if not graph_files:
        raise FileNotFoundError(f"No .graph.pkl files found in {graph_dir}")

    print(f"Found {len(graph_files)} graph files")
    graph_data_list = []

    for file in tqdm(graph_files, desc="Loading graph files"):
        with open(file, 'rb') as f:
            graph_data = pickle.load(f)
            graph_data_list.append({
                'file_path': str(file),
                'graph_data': graph_data
            })

    return graph_data_list


def compute_average_edge_weight_scores(nodes: List[dict], adj_matrix: np.ndarray) -> Dict[int, float]:
    """
    Compute each node score as the average weight of its incident edges.

    Args:
        nodes: node list; each node contains 'idx' (CIFAR100 dataset index)
        adj_matrix: adjacency matrix (N, N)

    Returns:
        Dict[idx -> score]
    """
    N = len(nodes)
    avg_scores = {}

    W = adj_matrix if not hasattr(adj_matrix, 'toarray') else adj_matrix.toarray()

    for i in range(N):
        weights = []
        for j in range(N):
            if i != j and W[i, j] > 0:
                weights.append(W[i, j])

        if len(weights) == 0:
            avg_score = 0.0
        else:
            avg_score = float(np.mean(weights))

        # Use idx as the key (CIFAR100 dataset index).
        idx = nodes[i]['idx']
        avg_scores[idx] = avg_score

    return avg_scores


def collect_all_scores(graph_data_list: List[dict]) -> Dict[int, float]:
    """
    Iterate over all graph files and extract the average edge-weight score for each sample.

    Returns:
        Dict[idx -> score]
    """
    all_scores = {}

    for item in tqdm(graph_data_list, desc="Computing average edge-weight scores for each graph"):
        graph_data = item['graph_data']
        nodes = graph_data['nodes']
        adj_matrix = graph_data['adj_matrix']

        scores_dict = compute_average_edge_weight_scores(nodes, adj_matrix)

        for idx, score in scores_dict.items():
            if idx in all_scores:
                print(f"Warning: Duplicate idx: {idx}, overwriting the previous value")
            all_scores[idx] = score

    print(f"Computed scores for {len(all_scores)} samples")
    return all_scores


def create_score_array_by_index(
    scores: Dict[int, float],
    dataset_length: int,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Create a score array from the idx-to-score mapping.

    Args:
        scores: Dict[idx -> score]
        dataset_length: dataset length
        fill_value: fill value for unmatched samples

    Returns:
        scores: np.ndarray, shape=(dataset_length,)
    """
    score_array = np.full(dataset_length, fill_value, dtype=np.float32)

    for idx, score in scores.items():
        if 0 <= idx < dataset_length:
            score_array[idx] = float(score)
        else:
            print(f"Warning: Warning: idx {idx} is outside the dataset range [0, {dataset_length-1}], skipping")

    valid_count = np.sum(score_array != fill_value)
    print(f"Created score array with shape: {score_array.shape}")
    print(f"   - matched valid samples: {valid_count}")
    print(f"   - unmatched samples: {dataset_length - valid_count} (filled with {fill_value})")

    return score_array


class CIFARGraphProbPrune(Dataset):
    """
    GraphProbPrune dataset specialized for CIFAR100.

    Main differences from the ImageNet version:
    1. Use idx (dataset index) for matching instead of image_name (file path)
    2. Wrap torchvision.datasets.CIFAR100 directly; no is_valid_file hook is required
    """

    def __init__(
        self,
        dataset,
        input_graph_dir: str,
        ratio: float = 0.5,
        num_epoch: int = None,
        delta: float = None,
        mode: str = "prob"
    ):
        """
        Args:
            dataset: torchvision.datasets.CIFAR100 instance
            input_graph_dir: graph root directory containing class_*/cluster_*.graph.pkl
            ratio: pruning ratio
            num_epoch: total number of training epochs
            delta: annealing parameter
            mode: selection mode: 'prob' or 'topk'
        """
        self.dataset = dataset
        self.ratio = ratio
        self.num_epoch = num_epoch
        self.delta = delta
        self.mode = mode

        # Initialize score arrays.
        self.node_scores = np.ones([len(self.dataset)])
        self.final_scores = np.ones([len(self.dataset)])
        self.weights = np.ones(len(self.dataset))
        self.save_num = 0

        # Load graph data and compute edge scores.
        graph_data_list = load_all_graph_files(input_graph_dir)
        scores = collect_all_scores(graph_data_list)
        self.edge_scores = create_score_array_by_index(
            scores=scores,
            dataset_length=len(self.dataset),
            fill_value=0.0
        )

        print(f"prune selection mode: {self.mode}")
        print(f"Successfully computed edge scores, dataset length: {len(self.dataset)}")

    def __setscore__(self, indices, values):
        """Update dynamic node scores during training."""
        self.node_scores[indices] = values
        # final_scores = node_scores - edge_scores (higher edge scores indicate greater similarity to neighbors and a higher pruning tendency)
        self.final_scores[indices] = self.node_scores[indices] - self.edge_scores[indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = self.weights[index]
        return data, target, index, weight

    def prune(self):
        """
        Run pruning and return the sample indices used in the current epoch.
        """
        # Select samples below the 99th percentile of final_scores as well learned.
        b = self.final_scores < np.percentile(self.final_scores, 99)
        well_learned_samples = np.where(b)[0]
        pruned_samples = np.where(np.invert(b))[0]

        if well_learned_samples.shape == (0,) or self.final_scores.min() == self.final_scores.max():
            print('Cut {} samples for next iteration'.format(len(self.dataset) - len(pruned_samples)))
            self.save_num += len(self.dataset) - len(pruned_samples)
            np.random.shuffle(pruned_samples)
            return pruned_samples

        # Select samples according to the configured mode.
        prob = torch.softmax(torch.from_numpy(self.final_scores[well_learned_samples]) / 5.0, dim=0).numpy()
        k = int(self.ratio * len(well_learned_samples))

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


class InfoBatchSampler:
    """InfoBatch sampler."""

    def __init__(self, infobatch_dataset, num_epoch=float('inf'), delta=1):
        self.infobatch_dataset = infobatch_dataset
        self.seq = None
        self.stop_prune = num_epoch * delta
        self.seed = 0
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.seed += 1
        if self.seed > self.stop_prune:
            if self.seed <= self.stop_prune + 1:
                self.infobatch_dataset.reset_weights()
            self.seq = self.infobatch_dataset.no_prune()
        else:
            self.seq = self.infobatch_dataset.prune()
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
        # Set the random seed to obtain a different shuffle for each epoch.
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