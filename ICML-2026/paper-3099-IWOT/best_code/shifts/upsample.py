from collections import defaultdict
import random
from torch.utils.data import Subset


def build_upsampled_indices_total(labels, desired_size, seed):
    """
    Upsample to an exact total length by repeating class indices in a round-robin way.
    - Never downsamples: requires desired_size >= len(labels).
    - Extras are distributed fairly across classes.
    """
    cur_size = len(labels)
    if desired_size < cur_size:
        raise ValueError(
            f"Desired size {desired_size} must be >= current dataset size ({cur_size})."
        )

    # Gather indices per class
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[int(y)].append(i)

    # Start with every sample exactly once
    idx = list(range(cur_size))

    # Round-robin add extras by cycling class lists
    classes = list(by_class.keys())
    # Shuffle class order for fairness/reproducibility
    gen = random.Random(seed)
    gen.shuffle(classes)

    extra = desired_size - cur_size
    ptr = {c: 0 for c in classes}
    while extra > 0:
        for c in classes:
            if extra == 0:
                break
            lst = by_class[c]
            idx.append(lst[ptr[c] % len(lst)])
            ptr[c] += 1
            extra -= 1

    # One initial shuffle; per-epoch randomness comes from DistributedSampler.set_epoch()
    gen = random.Random(seed)
    gen.shuffle(idx)
    return idx


def extract_labels(dataset):
    """
    Assumes dataset[i] -> (x, y, ...) and y is an int-like class id.
    """
    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i][1]))
    return labels


def upsample_dataset(
    base_dataset,
    desired_size,
    seed,
):
    labels = extract_labels(base_dataset)
    up_idx = build_upsampled_indices_total(labels, desired_size, seed)
    subset = Subset(base_dataset, up_idx)
    return subset
