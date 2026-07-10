import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


class VisDA17:
    def __init__(
        self,
        dataloader_options,
        test_dataloader_options,
        size,
        train_ratio=0.8,
        grayscale=False,
    ):
        self.name = "VISDA17"
        self.num_channels = 3
        self.num_classes = 12
        operations = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size),
        ]
        if grayscale is True:
            operations.append(v2.Grayscale(1))
        transforms = v2.Compose(operations)
        self.input_size = size[0] * size[1] * self.num_channels

        source_data = datasets.ImageFolder(
            root="data/VisDA17/train", transform=transforms
        )
        target_data = datasets.ImageFolder(
            root="data/VisDA17/validation", transform=transforms
        )

        train_size = int(train_ratio * len(source_data))
        test_size = int(len(source_data)) - train_size
        train_source, test_source = random_split(source_data, [train_size, test_size])

        train_size = int(train_ratio * len(target_data))
        test_size = int(len(target_data)) - train_size
        train_target, test_target = random_split(target_data, [train_size, test_size])

        # Load both datasets
        self.source_dataloader = DataLoader(train_source, **dataloader_options)
        self.target_dataloader = DataLoader(train_target, **dataloader_options)
        self.source_test_dataloader = DataLoader(test_source, **test_dataloader_options)
        self.target_test_dataloader = DataLoader(test_target, **test_dataloader_options)

        self.source_name = "train-sym"
        self.target_name = "validation-real"
