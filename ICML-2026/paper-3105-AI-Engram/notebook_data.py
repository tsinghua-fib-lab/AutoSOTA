# Extracted from fig_resnet18_cifar10.ipynb cell 3
# ===== engram/data/cifar.py  (verbatim from the engram package) =====
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2471, 0.2435, 0.2616]

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

def filter_and_sample(dataset, targets, num_samples):
    indices = np.arange(len(dataset))
    if targets is not None:
        targets_array = np.array(dataset.targets)
        mask = np.isin(targets_array, targets)
        indices = indices[mask]

    if num_samples is not None and num_samples < len(indices):
        indices = np.random.choice(indices, size=num_samples, replace=False)
    return Subset(dataset, indices)

def cifar10(split=None, targets: list = None, num_samples: int = None, batch_size=100, shuffle=False, root='../cache/data'):
    def get_data_loader(split):
        train = True if split=='train' else False
        transform = train_transform if split=='train' else test_transform
        
        dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        dataset = filter_and_sample(dataset, targets, num_samples)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    if split is None:
        return {split:get_data_loader(split) for split in ['train', 'test']}
    else:
        return get_data_loader(split)

def cifar100(split=None, targets: list = None, num_samples: int = None, batch_size=100, shuffle=False, root='../cache/data'):
    def get_data_loader(split):
        train = True if split=='train' else False
        transform = train_transform if split=='train' else test_transform
        
        dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
        dataset = filter_and_sample(dataset, targets, num_samples)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    if split is None:
        return {split:get_data_loader(split) for split in ['train', 'test']}
    else:
        return get_data_loader(split)