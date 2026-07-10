import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from shifts.upsample import upsample_dataset


class MNIST_to_USPS:
    def __init__(self, dataloader_options, test_dataloader_options):
        self.name = "MNIST_TO_USPS"
        self.num_channels = 1
        self.num_classes = 10
        self.source_name = "MNIST"
        self.target_name = "USPS"
        self.transforms_target = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.transforms_source = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=[16, 16], antialias=True),
            ]
        )
        self.input_size = 256
        self.source_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=self.transforms_source,
        )
        target_data = datasets.USPS(
            root="data/USPS",
            train=True,
            download=True,
            transform=self.transforms_target,
        )
        self.target_data = upsample_dataset(target_data, len(self.source_data), seed=1)

        source_test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=self.transforms_source
        )
        target_test_data = datasets.USPS(
            root="data/USPS",
            train=False,
            download=True,
            transform=self.transforms_target,
        )

        # Load both datasets
        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.source_test_dataloader = DataLoader(
            source_test_data, **test_dataloader_options
        )
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)
        self.target_test_dataloader = DataLoader(
            target_test_data,
            **test_dataloader_options,
        )
        # utils.calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])
