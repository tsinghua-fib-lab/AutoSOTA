import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_imagenet_datasets(
    dataset_path,
    test_split: float = 0.05,  # ignored, uses official train/val split
    image_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
    random_flip: bool = False,
    val_only: bool = False,
) -> Tuple[Optional[DataLoader], DataLoader]:

    g = torch.Generator()
    g.manual_seed(42)

    train_split_dir = os.path.join(dataset_path, "train")
    val_split_dir = os.path.join(dataset_path, "val")

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_dataset = datasets.ImageFolder(val_split_dir, transform=val_transform)

    if val_only:
        print(f"Dataset statistics (val only):")
        print(f"  Validation images: {len(val_dataset)}")
        print(f"  Number of classes: {len(val_dataset.classes)}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            generator=g,
        )
        return None, val_loader

    if random_flip:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    train_dataset = datasets.ImageFolder(train_split_dir, transform=train_transform)

    assert train_dataset.class_to_idx == val_dataset.class_to_idx, \
        "Train and val class_to_idx mismatch!"

    print(f"Dataset statistics:")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")
    print(f"  Number of classes: {len(train_dataset.classes)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=g
    )

    return train_loader, val_loader
