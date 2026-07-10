import numpy as np
import torch
import os
import pandas
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class MNIST_to_MNIST_M:
    def __init__(self, dataloader_options, test_dataloader_options, preprocess):
        self.name = "MNIST_TO_MNISTM"
        if preprocess is True:
            self.process_MNIST_M_labels(root="data/MNIST-M", use_train=True)
            self.process_MNIST_M_labels(root="data/MNIST-M", use_train=False)
        self.num_classes = 10
        self.num_channels = 3
        self.input_size = 3072
        self.source_name = "MNIST"
        self.target_name = "MNIST_M"
        self.transforms_source = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(32, 32)),
                self.StackTransform(),
            ]
        )
        self.transforms_target = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.source_data = datasets.MNIST(
            root="data", train=True, download=True, transform=self.transforms_source
        )
        self.target_data = datasets.ImageFolder(
            root="data/MNIST-M/train", transform=self.transforms_target
        )

        # Load both datasets
        self.source_dataloader = DataLoader(self.source_data, **dataloader_options)
        self.target_dataloader = DataLoader(self.target_data, **dataloader_options)

        source_test_data = datasets.MNIST(
            root="data", train=False, download=True, transform=self.transforms_source
        )
        self.source_test_dataloader = DataLoader(
            source_test_data, **test_dataloader_options
        )
        target_test_data = datasets.ImageFolder(
            root="data/MNIST-M/test", transform=self.transforms_target
        )
        self.target_test_dataloader = DataLoader(
            target_test_data, **test_dataloader_options
        )
        # calc_w_distance_label_shift(self)
        self.labels = np.array([str(i) for i in range(10)])

    def process_MNIST_M_labels(self, root, use_train):
        if use_train:
            folder = "train"
        else:
            folder = "test"
        labels_file = os.path.join(root, folder + "_labels.txt")
        out = pandas.read_csv(
            labels_file, header=None, names=["name", "label"], sep=" "
        )
        for label in out["label"].unique():
            os.makedirs(os.path.join(root, folder, str(label)), exist_ok=True)
        for index, elem in out.iterrows():
            os.rename(
                src=os.path.join(root, folder, elem["name"]),
                dst=os.path.join(root, folder, str(elem["label"]), elem["name"]),
            )

    class StackTransform:
        # Transform MNIST to SVHN RGB-shape (this is a bit dumb)
        def __call__(self, x):
            return torch.stack((x[0], x[0], x[0]), dim=0)
