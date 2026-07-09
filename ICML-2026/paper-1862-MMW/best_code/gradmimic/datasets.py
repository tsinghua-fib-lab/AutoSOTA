import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets


# ---------------------------------------------------------------------------
# Label noise
# ---------------------------------------------------------------------------

def add_noise_to_labels(labels, num_class, noise_ratio=0.1):
    labels = np.array(labels)
    num_noisy = int(noise_ratio * len(labels))
    noisy_indices = random.sample(range(len(labels)), num_noisy)
    noisy_labels = labels.copy()
    for idx in noisy_indices:
        noisy_labels[idx] = (labels[idx] + np.random.randint(1, num_class)) % num_class
    return noisy_labels, noisy_indices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_inner_dataset(name, root, train, transform, download):
    split_str = "train" if train else "test"

    if name == "dtd":
        ds = datasets.DTD(root=root, split=split_str, transform=transform, download=download)
        ds.targets = np.array([label for _, label in ds])

    elif name == "stl10":
        ds = datasets.STL10(root=root, split=split_str, transform=transform, download=download)
        ds.targets = ds.labels

    elif name == "cifar10":
        ds = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)

    elif name == "cifar100":
        ds = datasets.CIFAR100(root=root, train=train, transform=transform, download=download)

    elif name == "flower102":
        ds = datasets.Flowers102(root=root, split=split_str, transform=transform, download=download)
        ds.targets = np.array([label for _, label in ds])

    elif name == "pet":
        pet_split = "trainval" if train else "test"
        ds = datasets.OxfordIIITPet(
            root=root, split=pet_split, target_types="category",
            transform=transform, download=download,
        )
        ds.targets = [ds[i][1] for i in range(len(ds))]

    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    return ds


def _num_classes(name, ds):
    if name == "flower102":
        return 102
    elif name == "cifar10":
        return 10
    elif name == "cifar100":
        return 100
    else:
        return len(ds.classes)


# ---------------------------------------------------------------------------
# Public dataset classes
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    """Wraps a torchvision dataset; returns (index, image, true_label, noisy_label).

    Supports symmetric label noise injection via ``noise_ratio``.
    """

    def __init__(self, root, name, train=True, transform=None, download=False, noise_ratio=0.0):
        self.dataset = _build_inner_dataset(name, root, train, transform, download)
        self.true_labels = list(self.dataset.targets)
        self.num_class = _num_classes(name, self.dataset)

        if noise_ratio > 0.0:
            self.noise_ratio = noise_ratio
            self.noisy_labels, self.noisy_indices = add_noise_to_labels(
                self.true_labels, self.num_class, noise_ratio
            )
        else:
            self.noise_ratio = 0.0
            self.noisy_labels = list(self.true_labels)
            self.noisy_indices = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return index, data, self.true_labels[index], self.noisy_labels[index]


class FewShotDataset(Dataset):
    """Samples exactly ``few_shot_k`` examples per class from the dataset."""

    def __init__(self, root, name, train=True, transform=None, download=False, few_shot_k=10):
        self.dataset = _build_inner_dataset(name, root, train, transform, download)
        self.num_class = _num_classes(name, self.dataset)

        targets = np.array(self.dataset.targets)
        self.sampled_indices = []
        for cls in range(self.num_class):
            cls_indices = np.where(targets == cls)[0]
            k = min(few_shot_k, len(cls_indices))
            self.sampled_indices.extend(np.random.choice(cls_indices, k, replace=False))

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, index):
        data, target = self.dataset[self.sampled_indices[index]]
        return data, target


class RankedDataset(Dataset):
    """Selects the top-p fraction of training samples.

    With ``sampling_method='random'`` a random subset is drawn.
    With ``sampling_method='mimic'`` samples are ranked by their accumulated
    per-sample weight from a previous run's ``*_weights.pkl`` file specified
    via ``ranking_file``; falls back to random if the file is absent.
    """

    def __init__(
        self,
        root,
        name,
        train=True,
        transform=None,
        download=False,
        sampling_method="random",
        top_p=1.0,
        noise_ratio=0.0,
        ranking_file=None,
    ):
        self.dataset = _build_inner_dataset(name, root, train, transform, download)
        self.num_class = _num_classes(name, self.dataset)
        self.true_labels = list(self.dataset.targets)
        self.noisy_labels = list(self.dataset.targets)
        self.noisy_indices = []

        n = len(self.dataset)
        k = max(1, int(top_p * n))

        if sampling_method == "mimic" and ranking_file and os.path.exists(ranking_file):
            with open(ranking_file, "rb") as fh:
                weights_data = pickle.load(fh)
            last_epoch = len(next(iter(weights_data.values()))["per_sample_weights"]) - 1
            scores = [
                (idx, data["per_sample_weights"][last_epoch])
                for idx, data in weights_data.items()
            ]
            scores.sort(key=lambda x: x[1], reverse=True)
            self.selected_indices = [idx for idx, _ in scores[:k]]
        else:
            if sampling_method == "mimic" and (ranking_file is None or not os.path.exists(ranking_file)):
                print("Warning: ranking_file not found; falling back to random subset selection.")
            self.selected_indices = random.sample(range(n), k)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        real_idx = self.selected_indices[index]
        data, _ = self.dataset[real_idx]
        return real_idx, data, self.true_labels[real_idx], self.noisy_labels[real_idx]
