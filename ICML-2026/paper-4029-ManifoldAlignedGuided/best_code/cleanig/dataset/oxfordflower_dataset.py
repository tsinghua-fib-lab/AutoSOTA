import os
from typing import Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_oxfordflower_datasets(
    dataset_path,
    test_split=0.05,  # ignored, uses official split
    image_size=256,
    batch_size=16,
    num_workers=4,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    random_flip=False,
    val_only: bool = False,
) -> Tuple[Optional[DataLoader], DataLoader]:
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_dataset = datasets.Flowers102(
        root=dataset_path,
        split='val',
        transform=val_transform,
        download=True
    )

    if val_only:
        print(f"Dataset statistics (val only):")
        print(f"  Validation images: {len(val_dataset)}")
        print(f"  Number of classes: 102")

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        return None, val_loader

    if random_flip:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    train_dataset = datasets.Flowers102(
        root=dataset_path,
        split='train',
        transform=train_transform,
        download=True
    )

    print(f"Dataset statistics:")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")
    print(f"  Number of classes: 102")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader
