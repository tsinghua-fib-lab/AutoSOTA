from dataloaders.cifar10 import get_loaders as cifar10_loaders
from dataloaders.cifar100 import get_loaders as cifar100_loaders
from dataloaders.tinyimagenet import get_loaders as tinyimagenet_loaders
from dataloaders.imagenet100 import get_loaders as imagenet100_loaders
from dataloaders.medmnist import get_loaders as medmnist_loaders

# Returns trainloader, testloader, upper_limit, lower_limit, mu, std, classes, len(classes)
def get_loaders(args, index_dataset: bool, device):
    """
    Get the loaders for the dataset.
    
    Args:
        args (argparse.Namespace): Arguments for the training.
        index_dataset (bool): Whether to index the dataset.
        device (torch.device): Device to use for the training.
    """
    match args.dataset:
        case "CIFAR10":
            return cifar10_loaders(args, index_dataset, device)
        case "CIFAR100":
            return cifar100_loaders(args, index_dataset, device)
        case "TinyImageNet":
            return tinyimagenet_loaders(args, index_dataset, device)
        case "ImageNet100":
            return imagenet100_loaders(args, index_dataset, device)
        case args.dataset if args.dataset in ['PathMNIST', 'TissueMNIST']:
            return medmnist_loaders(args, index_dataset, device)
        case _:
            raise ValueError("Invalid Dataset!")
