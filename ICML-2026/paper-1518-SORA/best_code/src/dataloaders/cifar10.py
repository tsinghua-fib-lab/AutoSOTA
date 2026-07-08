import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from dataloaders.index_dataset import IndexDataset

def get_loaders(args, index_dataset: bool, device):
    """
    Prepare CIFAR-10 dataset loaders with optional index annotation and normalization.

    This helper handles:
        - Dataset mean/std normalization (optional)
        - Choice between standard augmentation (random crop + flip) or deterministic
          padding for index-aware datasets
        - Creation of train/test DataLoaders with specified batch size and workers
        - Computation of normalized upper/lower pixel bounds for adversarial constraints

    Args:
        args (argparse.Namespace): Contains dataset, root_path, batch_size, num_workers,
            normalize_dataset, and other configuration options.
        index_dataset (bool): If True, wraps the training set with IndexDataset to return
            (sample, label, index) tuples for index tracking in training/evaluation.
        device (str): Device on which normalization boundary tensors will be stored
            (e.g., "cuda").

    Returns:
        tuple:
            trainloader (DataLoader): Augmented CIFAR-10 training loader.
            testloader (DataLoader): Normalized CIFAR-10 test loader.
            upper_limit (torch.Tensor): Per-channel perturbed pixel max bound (normalized).
            lower_limit (torch.Tensor): Per-channel perturbed pixel min bound (normalized).
            mu (torch.Tensor): Dataset channel means (C×1×1, to match input shape).
            std (torch.Tensor): Dataset channel standard deviations (C×1×1).
            classes (tuple[str]): Class name tuple (length 10 for CIFAR-10).
            num_classes (int): Number of classes (10 for CIFAR-10).
            len_trainset (int): Number of training samples (50,000 for CIFAR-10).
            len_testset (int): Number of test samples (10,000 for CIFAR-10).

    References:
        Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.
        Technical Report, University of Toronto.
        URL: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    if args.normalize_dataset:
        cifar10_mean = [0.4914, 0.4822, 0.4465] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        cifar10_std = [0.2471, 0.2435, 0.2616] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        cifar10_mean = [0., 0., 0.]
        cifar10_std = [1., 1., 1.]
    
    mu = torch.tensor(cifar10_mean).view(3,1,1).to(device)
    std = torch.tensor(cifar10_std).view(3,1,1).to(device)
    
    if index_dataset:
        train_transform = transforms.Compose([
                transforms.Pad(padding=4),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
    else:
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    

    # Download the dataset
    trainset = CIFAR10(root=f'{args.root_path}/Datasets/{args.dataset}', train=True, download=True, transform=train_transform)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = CIFAR10(root=f'{args.root_path}/Datasets/{args.dataset}', train=False, download=True, transform=test_transform)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    # Name of Classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes), len(trainset), len(testset)
