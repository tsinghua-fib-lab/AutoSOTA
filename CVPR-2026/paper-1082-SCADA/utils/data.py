import sys
sys.path.append('.')

import os
import torch

import torch
from torch.utils.data import TensorDataset

from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader, random_split

from tllib.vision.transforms import ResizeImage

from utils.utils import dump, load




def get_loaders(domain, config):
    """
    Returns the following 6 loaders
    - Full (train & test)
    - Retain (train & test)
    - Forget (only test)
    - Retain Subset (only train)

    """

    full_dataset = get_dataset(domain, config)
    retain_dataset = filter_dataset(full_dataset, domain, config['forget_classes'], config, exclude=True)

    full_train_dataset, full_test_dataset = random_split(full_dataset, lengths=[config['split'], 1-config['split']])
    retain_train_dataset, retain_test_dataset = random_split(retain_dataset, lengths=[config['split'], 1-config['split']])
    forget_dataset = filter_dataset(full_dataset, domain, config['forget_classes'], config, exclude=False)
    retain_subset = get_subset(retain_dataset, config)

    full_train_dl = DataLoader(full_train_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    full_test_dl = DataLoader(full_test_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=True)
    retain_train_dl = DataLoader(retain_train_dataset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)
    retain_test_dl = DataLoader(retain_test_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=True)
    forget_dl = DataLoader(forget_dataset, config['batch'], shuffle=False, num_workers=config['workers'], drop_last=False)
    retain_subset_dl = DataLoader(retain_subset, config['batch'], shuffle=True, num_workers=config['workers'], drop_last=True)

    return full_train_dl, full_test_dl, retain_train_dl, retain_test_dl, forget_dl, retain_subset_dl


def get_dataset(domain, config):
    """
    Returns the dataset

    """
    
    
    torch.manual_seed(config['seed'])

    RESIZE = 256
    SIZE = 224
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        ResizeImage(RESIZE),
        CenterCrop(SIZE),
        ToTensor(),
        Normalize(mean=MEANS, std=STDS)
    ])

    if config['dataset'] == 'OfficeHome':
        assert domain in ['Art', 'Clipart', 'Product', 'Real_World']
        config['num_classes'] = 65
        config['size'] = 224
        config['channels'] = 3

    elif config['dataset'] == 'DomainNet':
        assert domain in ['clipart', 'painting', 'real', 'sketch']    
        config['num_classes'] = 126
        config['size'] = 224
        config['channels'] = 3

    elif config['dataset'] == 'Office31':
        assert domain in ['amazon', 'dslr', 'webcam']
        config['num_classes'] = 31
        config['size'] = 224
        config['channels'] = 3
    
    path = os.path.join(config['data_path'], config['dataset'], domain)        
    dataset = ImageFolder(root=path, transform=transform)
        
    return dataset


def filter_dataset(dataset, domain, classes, config, exclude=False):
    """
    Filters Dataset based on classes

    """
    
    
    path = os.path.join(config['dump_path'], config['dataset'], domain, 'labels.p')

    if not os.path.exists(path):
        labels = []
        for _, label in tqdm(dataset, desc=f"Filtering {config['dataset']} {domain}"):
            labels.append(label)
        labels = torch.tensor(labels)
        dump(labels, path)
    
    labels = load(path)

    if exclude:
        mask = ~torch.isin(labels, torch.tensor(classes))
    else:
        mask = torch.isin(labels, torch.tensor(classes))

    indices = torch.nonzero(mask).squeeze().tolist()
    filtered_dataset = Subset(dataset, indices)

    return filtered_dataset


def get_subset(dataset, config):
    """
    Returns the subset of data accessable after training.
    
    Arbitrarily considering approximately 20 samples per class

    """

    subset_size = config['num_classes'] * 20
    subset_size = min(subset_size, len(dataset))
    subset_indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, subset_indices)

    return subset
