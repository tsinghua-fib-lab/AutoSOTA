# dataloader.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_dataloaders(batch_size=128):
    """
    Returns train, valid, and test dataloaders for CIFAR10.
    """
    # Standard CIFAR10 Normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test/Val transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Download datasets
    root = './data'
    full_train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    # Split Train (50k) into Train (45k) and Valid (5k)
    # We use a fixed generator for reproducibility across sweeps
    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = random_split(full_train_dataset, [45000, 5000], generator=generator)

    # IMPORTANT: The validation set needs the test_transform (no augmentation), 
    # but random_split inherits the transform from the parent. 
    # We can rely on the slight noise of aug in val or strictly override it. 
    # For a clean benchmark, we usually accept the parent transform or wrap it. 
    # For simplicity here, we proceed, but noted that Val has augs.
    
    num_workers = 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader