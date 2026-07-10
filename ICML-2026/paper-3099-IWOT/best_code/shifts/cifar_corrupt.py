import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import utils


class CIFAR_CORRUPT:
    def __init__(self, dataloader_options, test_dataloader_options, corruptions):
        self.name = "CIFAR10_TO_CIFAR10C"
        self.source_name = "CIFAR10"
        self.target_name = "CIFAR10C"
        self.input_size = 3072
        self.num_channels = 3
        self.num_classes = 10
        self.transforms_source = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.transforms_target = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.source_data = datasets.CIFAR10(
            root="data/CIFAR10",
            download=True,
            train=True,
            transform=self.transforms_source,
        )

        X_target = torch.tensor([])
        y_target = torch.tensor([])
        for corruption in corruptions:
            X = torch.from_numpy(
                np.load("data/CIFAR-10-C/" + corruption + ".npy")
            ).permute((0, 3, 1, 2))
            y = torch.from_numpy(np.load("data/CIFAR-10-C/labels.npy")).long()
            X_target = torch.concatenate((X_target, X))
            y_target = torch.concatenate((y_target, y))
        num_targets = X_target.shape[0]
        idx_shuffle = torch.randperm(num_targets)
        X_target = X_target[idx_shuffle]
        y_target = y_target[idx_shuffle].long()
        num_test_use = int(X_target.shape[0] * 0.8)
        X_target_use, X_target_test = X_target[:num_test_use], X_target[num_test_use:]
        y_target_use, y_target_test = y_target[:num_test_use], y_target[num_test_use:]

        self.target_data = utils.GenericDataset(
            X_target_use, y_target_use, transform=self.transforms_target
        )

        # Load both datasets
        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)

        source_test_data = datasets.CIFAR10(
            root="data/CIFAR10",
            train=False,
            download=True,
            transform=self.transforms_source,
        )
        self.source_test_dataloader = DataLoader(
            source_test_data, **test_dataloader_options
        )
        target_test_data = utils.GenericDataset(
            X_target_test, y_target_test, transform=self.transforms_target
        )
        self.target_test_dataloader = DataLoader(
            target_test_data, **test_dataloader_options
        )
        # calc_w_distance_label_shift(self)
        self.labels = np.array(
            [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
        )
