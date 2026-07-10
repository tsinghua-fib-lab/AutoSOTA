import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class SVHN_to_MNIST:
    def __init__(self, dataloader_options, test_dataloader_options):
        self.name = "SVHN_TO_MNIST"
        self.source_name = "SVHN"
        self.target_name = "MNIST"
        self.input_size = 3072
        self.num_channels = 3
        self.num_classes = 10
        self.transforms_source = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.transforms_target = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),
                self.StackTransform(),
            ]
        )
        self.source_data = datasets.SVHN(
            root="data/SVHN",
            split="train",
            download=True,
            transform=self.transforms_source,
        )
        self.target_data = datasets.MNIST(
            root="data", train=True, download=True, transform=self.transforms_target
        )
        source_test_data = datasets.SVHN(
            root="data/SVHN",
            split="test",
            download=True,
            transform=self.transforms_source,
        )
        target_test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=self.transforms_target
        )

        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)
        self.source_test_dataloader = DataLoader(
            source_test_data, **test_dataloader_options
        )
        self.target_test_dataloader = DataLoader(
            target_test_data, **test_dataloader_options
        )
        # calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])

    class StackTransform:
        # Transform MNIST to SVHN RGB-shape (this is a bit dumb)
        def __call__(self, x):
            return torch.stack((x[0], x[0], x[0]), dim=0)
