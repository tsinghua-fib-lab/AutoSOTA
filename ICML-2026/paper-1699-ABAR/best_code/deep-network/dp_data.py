# dp_data.py
from __future__ import annotations

import math
import random
from typing import Iterable, List, Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset


def compute_epsilon_for_range(mean_batch_size: int, delta: float) -> float:
    if mean_batch_size <= 0:
        raise ValueError("mean_batch_size must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    u_bound = mean_batch_size / 3
    # Ensure u_bound is not zero to prevent division by zero in log(2/delta)/u_bound
    if u_bound <= 0:
        raise ValueError("u_bound must be > 0 for epsilon calculation")
    return math.log(2 / delta) / u_bound


def sample_truncated_geometric(epsilon: float, delta: float) -> int:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    alpha = math.exp(epsilon)
    u_bound = math.log(2 / delta) / epsilon

    p = 1 - 1 / alpha
    # Ensure p is within (0, 1) for Geometric distribution
    p = max(1e-10, min(1 - 1e-10, p)) 
    
    magnitude = torch.distributions.Geometric(p).sample().item()
    sign = -1 if torch.rand(1).item() < 0.5 else 1
    r = sign * magnitude

    r = max(-u_bound, min(u_bound, r))
    return int(r)


class TruncatedGeometricBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset_size: int,
        mean_batch_size: int,
        delta: float,
        shuffle: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if mean_batch_size <= 0:
            raise ValueError("mean_batch_size must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")

        self.dataset_size = dataset_size
        self.mean_batch_size = mean_batch_size
        self.epsilon = compute_epsilon_for_range(mean_batch_size, delta)
        self.delta = delta
        self.shuffle = shuffle
        
        # Store the sampler's epsilon for later composition
        self.sampler_epsilon = self.epsilon 
        self.sampler_delta = self.delta

    def __iter__(self) -> Iterable[List[int]]:
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.shuffle(indices)

        i = 0
        while i < self.dataset_size:
            # The sampler uses epsilon/2 and delta/2 for its noise generation
            # to ensure the overall sampler privacy is (epsilon, delta)
            noise = sample_truncated_geometric(self.epsilon / 2, self.delta / 2)
            
            batch_size = max(1, self.mean_batch_size + noise)
            
            # Ensure batch_size does not exceed remaining data
            batch_size = min(batch_size, self.dataset_size - i)

            batch = indices[i : i + batch_size]
            if not batch: # Stop if no more data
                break
            yield batch
            i += batch_size

    def __len__(self) -> int:
        # This is an approximation since batch sizes are variable
        return math.ceil(self.dataset_size / self.mean_batch_size)


def get_dp_dataloader(
    trainset_images: torch.Tensor,
    trainset_labels: torch.LongTensor,
    ntrain: int,
    mean_batch_size: int,
    delta: float,
    shuffle: bool = True,
    plan: Optional[Dict[str, object]] = None,
) -> Tuple[DataLoader, float, float, int]: # Returns DataLoader, sampler_epsilon, sampler_delta, num_batches
    """
    Creates a DataLoader with TruncatedGeometricBatchSampler for DP training.

    Args:
        trainset_images: Tensor of training images.
        trainset_labels: Tensor of training labels.
        ntrain: Total number of training samples.
        mean_batch_size: The desired mean batch size.
        delta: The delta value for the sampler's privacy.
        shuffle: Whether to shuffle the dataset indices.

    Returns:
        A tuple containing:
            - pytorch_train_loader: The DataLoader instance.
            - sampler_epsilon: The epsilon cost of the batch sampler.
            - sampler_delta: The delta cost of the batch sampler.
            - num_batches: The number of batches in the plan.
    """
    # Create a PyTorch TensorDataset
    pytorch_train_dataset = TensorDataset(trainset_images, trainset_labels)

    if plan is None:
        plan = plan_dp_batches(
            dataset_size=ntrain,
            mean_batch_size=mean_batch_size,
            delta=delta,
            shuffle=shuffle,
        )

    batch_sampler = PlannedBatchSampler(
        indices=plan["indices"],
        batch_sizes=plan["batch_sizes"],
    )

    pytorch_train_loader = DataLoader(
        pytorch_train_dataset,
        batch_sampler=batch_sampler,
    )

    return (
        pytorch_train_loader,
        plan["sampler_epsilon"],
        plan["sampler_delta"],
        len(plan["batch_sizes"]),
    )


class PlannedBatchSampler(Sampler[List[int]]):
    def __init__(self, indices: List[int], batch_sizes: List[int]) -> None:
        self.indices = indices
        self.batch_sizes = batch_sizes

    def __iter__(self) -> Iterable[List[int]]:
        start = 0
        for size in self.batch_sizes:
            end = start + size
            yield self.indices[start:end]
            start = end

    def __len__(self) -> int:
        return len(self.batch_sizes)


def plan_dp_batches(
    dataset_size: int,
    mean_batch_size: int,
    delta: float,
    shuffle: bool = True,
) -> Dict[str, object]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be > 0")
    if mean_batch_size <= 0:
        raise ValueError("mean_batch_size must be > 0")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")

    epsilon = compute_epsilon_for_range(mean_batch_size, delta)

    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)

    batch_sizes = []
    i = 0
    while i < dataset_size:
        noise = sample_truncated_geometric(epsilon, delta)
        batch_size = max(1, mean_batch_size + noise)
        batch_size = min(batch_size, dataset_size - i)
        batch_sizes.append(batch_size)
        i += batch_size

    return {
        "indices": indices,
        "batch_sizes": batch_sizes,
        "sampler_epsilon": epsilon,
        "sampler_delta": delta,
    }
