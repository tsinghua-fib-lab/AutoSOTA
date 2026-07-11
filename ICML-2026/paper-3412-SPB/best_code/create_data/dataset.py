import os
import torch
import trimesh
from torch.utils.data import Dataset
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TransformedDataset(Dataset):
    """
    Wrapper around preprocessed tensors stored on disk.

    Expected file format:
        {
            "images": Tensor,
            "labels": Tensor,
            "meta": optional metadata
        }
    """

    def __init__(self, path, transform=None):
        data = torch.load(path, weights_only=False)

        self.images = data["images"]
        self.labels = data["labels"]

        # Optional auxiliary information used in analysis/debugging.
        self.meta = data.get("meta", None)

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        x = self.images[idx]
        y = int(self.labels[idx])

        # Apply augmentation/preprocessing lazily at access time.
        if self.transform:
            x = self.transform(x)

        sample = {"x": x, "y": y}

        if self.meta is not None:
            sample["meta"] = self.meta[idx]

        return sample


class ModelNet10PointCloud:
    """
    Point-cloud version of ModelNet10.

    Meshes are uniformly sampled into fixed-size point clouds and
    normalized to a unit-radius coordinate system.
    """

    def __init__(self, root, split="train", n_points=1024):

        self.samples = []
        self.n_points = n_points

        classes = sorted(os.listdir(root))

        # Deterministic class ordering defines label assignment.
        for label, cls in enumerate(classes):

            cls_path = os.path.join(root, cls, split)

            for fname in os.listdir(cls_path):

                if fname.endswith(".off"):
                    self.samples.append((os.path.join(cls_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        mesh = trimesh.load(path)

        # Uniform surface sampling.
        points = mesh.sample(self.n_points)  # (N, 3)

        points = torch.tensor(points, dtype=torch.float32)

        # Center and scale to improve optimization stability.
        points = points - points.mean(0)
        points = points / points.norm(dim=1).max()

        return points, label


import torch
from torch.utils.data import Dataset

from jetnet.datasets import TopTagging


class TopTaggingDataset(Dataset):
    """
    Top Tagging dataset with constituent four-vectors.

    Returns:
        jet: Tensor of shape (num_particles, 4)

    Convention:
        (E, px, py, pz)
    """

    def __init__(
        self,
        data_dir="./data",
        split="train",
        num_particles=30,
        jet_type="all",
        normalize=True,
    ):

        self.num_particles = num_particles
        self.normalize = normalize

        # Load JetNet Top Tagging dataset
        particle_data, jet_data = TopTagging.getData(
            jet_type=jet_type,
            data_dir=data_dir,
            particle_features=[
                "E",
                "px",
                "py",
                "pz",
            ],
            jet_features=["type"],
            num_particles=num_particles,
            split=split,
            download=True,
        )

        self.jets = torch.tensor(
            particle_data,
            dtype=torch.float32,
        )

        # Labels:
        #   0 -> QCD jet
        #   1 -> top jet
        self.labels = torch.tensor(
            jet_data[:, 0],
            dtype=torch.long,
        )

        # Replace invalid numerical entries introduced during preprocessing.
        self.jets = torch.nan_to_num(self.jets)

    # Lorentz-compatible normalization
    def normalize_jet(self, jet):

        """
        Normalize all four-vector components by the total jet energy.

        This preserves relative geometric structure between components,
        unlike independent feature-wise normalization.
        """

        scale = jet[:, 0].sum()

        if scale > 0:
            jet = jet / scale

        return jet

    # Dataset API
    def __len__(self):

        return len(self.jets)

    def __getitem__(self, idx):

        # Clone to avoid in-place modification of stored tensors.
        jet = self.jets[idx].clone()

        if self.normalize:
            jet = self.normalize_jet(jet)

        label = self.labels[idx]

        return jet, label
# Original file had unconditional import of jetnet - we make it conditional
