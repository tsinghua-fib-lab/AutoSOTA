
import sys
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset

# Add parent directory to path to import datasets_and_dataloaders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def concat_datasets(datasets):
    """
    Concatenate a list of datasets into a single ConcatDataset.
    If the list is empty, returns None. If only one dataset, returns it directly.

    Args:
        datasets (list): List of torch.utils.data.Dataset objects.

    Returns:
        torch.utils.data.Dataset or None
    """
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def make_combined_loader(datasets, batch_size, shuffle=True, num_workers=0, pin_memory=False):
    """
    Create a DataLoader from a single dataset (or ConcatDataset).

    Args:
        datasets (Dataset): A single dataset object (can be ConcatDataset, Dataset, or None).
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to pin memory.

    Returns:
        DataLoader or None
    """
    if datasets is None:
        return None
    combined_dataset = concat_datasets(datasets)
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def make_combined_loader_deprecated(loaders, global_batch_size):
    """
    [DEPRECATED] Old function for combining DataLoader instances by batch proportion.
    Use make_combined_datasets + make_combined_loader instead.
    """
    import math
    from torch.utils.data import DataLoader, IterableDataset, RandomSampler
    n = len(loaders)
    # Extract lengths and loader settings
    lengths = []
    settings = []  # (shuffle, num_workers, drop_last)
    for loader in loaders:
        if not isinstance(loader, DataLoader):
            raise TypeError(f"Expected DataLoader, got {type(loader)}")
        lengths.append(len(loader.dataset))
        # Determine shuffle from sampler type
        shuffle = isinstance(loader.sampler, RandomSampler)
        settings.append((
            shuffle,
            loader.num_workers,
            loader.drop_last,
        ))

    total = sum(lengths)
    # Compute per-loader batch sizes, at least 1 sample each
    batch_sizes = [max(math.floor(global_batch_size * L / total), 1) for L in lengths]
    # Distribute remainder
    remainder = global_batch_size - sum(batch_sizes)
    for i in range(remainder):
        batch_sizes[i % n] += 1

    if any(b <= 0 for b in batch_sizes):
        raise ValueError(f"All per-loader batch sizes must be > 0, got {batch_sizes}")

    # Re-create DataLoader for each with new batch_size, preserving other settings
    new_loaders = []
    for (shuffle, num_workers, drop_last), loader, b in zip(settings, loaders, batch_sizes):
        new_loaders.append(
            DataLoader(
                loader.dataset,
                batch_size=b,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last
            )
        )

    # Infinite cycling iterators
    def infinite_cycle(loader):
        while True:
            for batch in loader:
                yield batch

    iterators = [infinite_cycle(ld) for ld in new_loaders]

    # Define combined IterableDataset
    class CombinedLoader(IterableDataset):
        def __init__(self):
            self.batch_count = 0
            
        def __iter__(self):
            self.batch_count = 0  # Reset counter for each iteration
            return self

        def __next__(self):
            if self.batch_count >= self.num_batches:
                raise StopIteration
            
            batches = [next(it) for it in iterators]
            xs, ys = zip(*batches)
            x_batch = torch.cat(xs, dim=0)
            y_batch = torch.cat(ys, dim=0)
            
            self.batch_count += 1
            return x_batch, y_batch

    combined = CombinedLoader()
    # Track number of global batches per epoch
    combined.num_batches = min(
        math.ceil(L / b) for L, b in zip(lengths, batch_sizes)
    )

    print(f"[DEPRECATED] Combined loader created with {combined.num_batches} batches per epoch.")
    print(f"Batches per dataset: {[math.ceil(L / b) for L, b in zip(lengths, batch_sizes)]}")
    print("[DEPRECATED] This function is deprecated. Use make_combined_datasets + make_combined_loader instead.")
    return combined


# Basic test when running this script directly
if __name__ == "__main__":
    pass