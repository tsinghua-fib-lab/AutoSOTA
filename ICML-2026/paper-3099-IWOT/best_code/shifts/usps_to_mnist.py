import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from shifts.upsample import upsample_dataset


class USPS_to_MNIST:
    def __init__(self, dataloader_options, test_dataloader_options):
        self.name = "USPS_TO_MNIST"
        self.num_channels = 1
        self.num_classes = 10
        self.source_name = "USPS"
        self.target_name = "MNIST"
        self.transforms_target = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=[16, 16], antialias=True),
            ]
        )
        self.transforms_source = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.input_size = 256
        source_data = datasets.USPS(
            root="data/USPS",
            train=True,
            download=True,
            transform=self.transforms_source,
        )
        self.target_data = datasets.MNIST(
            root="data", train=True, download=True, transform=self.transforms_target
        )
        self.source_data = upsample_dataset(source_data, len(self.target_data), seed=1)

        # Load both datasets
        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)

        test_data = datasets.USPS(
            root="data/USPS",
            train=False,
            download=True,
            transform=self.transforms_source,
        )
        self.source_test_dataloader = DataLoader(test_data, **test_dataloader_options)
        test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=self.transforms_target
        )
        self.target_test_dataloader = DataLoader(test_data, **test_dataloader_options)
        # utils.calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])
