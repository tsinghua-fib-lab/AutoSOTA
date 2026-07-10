import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from shifts.upsample import upsample_dataset


if __name__ == "__main__":
    transforms_source = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=[16, 16], antialias=True),
        ]
    )
    transforms_target = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    source_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms_source,
    )
    target_data = datasets.USPS(
        root="data/USPS",
        train=True,
        download=True,
        transform=transforms_target,
    )
    print(f"Size of dataset before upsampling: {len(target_data)}")
    dl_options = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": False,
    }
    target_data = upsample_dataset(target_data, len(source_data), seed=1)
    loader = DataLoader(target_data, **dl_options)

    print(f"Size after: {len(loader.dataset)}")
