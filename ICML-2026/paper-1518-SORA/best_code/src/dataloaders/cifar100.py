import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from dataloaders.index_dataset import IndexDataset

def get_loaders(args, index_dataset: bool, device):
    """
    Prepare CIFAR-100 dataset loaders with optional index annotation and normalization.

    This function handles:
        - Dataset mean/std normalization (optional).
        - Choice between standard augmentation (random crop + horizontal flip)
          or deterministic padding for index-aware datasets.
        - Creation of train/test DataLoaders with specified batch size and workers.
        - Computation of normalized upper/lower pixel bounds for adversarial clamping.

    Args:
        args (argparse.Namespace): Contains dataset, root_path, batch_size, num_workers,
            normalize_dataset, and other configuration options.
        index_dataset (bool): If True, wraps the training set with IndexDataset to return
            (sample, label, index) tuples for dataset index tracking.
        device (str): Device on which normalization tensors and pixel bounds will be stored 
            (e.g., "cuda").

    Returns:
        tuple:
            trainloader (DataLoader): Augmented CIFAR-100 training loader.
            testloader (DataLoader): Normalized CIFAR-100 test loader.
            upper_limit (torch.Tensor): Per-channel perturbed pixel max bound (normalized).
            lower_limit (torch.Tensor): Per-channel perturbed pixel min bound (normalized).
            mu (torch.Tensor): Dataset channel means (C×1×1).
            std (torch.Tensor): Dataset channel standard deviations (C×1×1).
            classes (tuple[str]): Ordered tuple of CIFAR-100 class names.
            num_classes (int): Number of classes (100 for CIFAR-100).
            len_trainset (int): Number of training samples (50,000 for CIFAR-100).
            len_testset (int): Number of test samples (10,000 for CIFAR-100).

    References:
        Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.
        Technical Report, University of Toronto.
        URL: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    if args.normalize_dataset:
        cifar100_mean = [0.5071, 0.4865, 0.4409] # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        cifar100_std =  [0.2673, 0.2564, 0.2762] # equals np.std(train_set.train_data, axis=(0,1,2))/255
    else:
        cifar100_mean = [0., 0., 0.]
        cifar100_std = [1., 1., 1.]
    
    mu = torch.tensor(cifar100_mean).view(3,1,1).to(device)
    std = torch.tensor(cifar100_std).view(3,1,1).to(device)
    
    if index_dataset:
        train_transform = transforms.Compose([
                transforms.Pad(padding=4),
                transforms.ToTensor(),
                transforms.Normalize(cifar100_mean, cifar100_std),
            ])
    else:  
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(cifar100_mean, cifar100_std),
            ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),
    ])
    

    # Download the dataset
    trainset = CIFAR100(root=f'{args.root_path}/Datasets/{args.dataset}', train=True, download=True, transform=train_transform)
    trainset = IndexDataset(trainset) if index_dataset else trainset # Index Dataset

    testset = CIFAR100(root=f'{args.root_path}/Datasets/{args.dataset}', train=False, download=True, transform=test_transform)

    # Create the loaders
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Legal limits of pixles after normalization
    upper_limit = ((1 - mu)/ std).to(device)
    lower_limit = ((0 - mu)/ std).to(device)

    # Name of Classes
    classes = (
        'apple','aquarium_fish','baby',
        'bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle',
        'caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin',
        'elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard',
        'lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter',
        'palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum',
        'rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
        'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout',
        'tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm')

    return trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes), len(trainset), len(testset)
