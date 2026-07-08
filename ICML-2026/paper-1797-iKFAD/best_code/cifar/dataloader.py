# dataloader.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import numpy as np

def get_mean_std(trainset):
    """
    Calculates mean and std from the dataset.
    Logic taken from reference code.
    """
    imgs = [item[0] for item in trainset] # item[0] is image
    imgs = torch.stack(imgs, dim=0).numpy()
    
    # calculate mean over each channel (r,g,b)
    mean_r = imgs[:,0,:,:].mean()
    mean_g = imgs[:,1,:,:].mean()
    mean_b = imgs[:,2,:,:].mean()
    print('mean',mean_r,mean_g,mean_b)
    
    # calculate std over each channel (r,g,b)
    std_r = imgs[:,0,:,:].std()
    std_g = imgs[:,1,:,:].std()
    std_b = imgs[:,2,:,:].std()
    print('std',std_r,std_g,std_b)
    
    mean = (mean_r, mean_g, mean_b)
    std  = (std_r, std_g, std_b)

    return mean, std

def get_dataloaders(batch_size=64, root='./cifar/data'):
    """
    Returns train, valid, and test dataloaders for CIFAR-10.
    """
    # 1. Load unnormalized data to calculate Mean/Std
    # Reference code downloads to specific folder for this step
    temp_train_dataset = torchvision.datasets.CIFAR10(
        root='./cifar/data', 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
    )
    
    mean, std = get_mean_std(temp_train_dataset)

    # 2. Define transforms using calculated Mean/Std
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]) 

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]) 

    # 3. Load Datasets
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    full_dataset = ConcatDataset([trainset, testset])
    
    # 80/10/10 split
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])

    # 4. Create Loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader