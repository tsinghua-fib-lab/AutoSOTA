import torch
import torchvision.datasets
import torchvision.transforms


def _balanced_dataset(dataset, data_amount_per_class, num_classes):
    indices = torch.tensor([])
    for c in range(num_classes):
        equals = (dataset.targets == c)
        indices_temp = torch.nonzero(equals)[:,0]
        indices = torch.cat((indices,indices_temp[:data_amount_per_class]),0)
    return torch.utils.data.Subset(dataset, indices[torch.randperm(indices.shape[0])].long())


def load_balanced_dataset(dataset, batch_size, data_amount_per_class, num_classes, shuffle=False, **kwargs):
    dataset = _balanced_dataset(dataset, data_amount_per_class, num_classes)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, **kwargs)


def fetch_cifar10(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(root, train=train, transform=transform, target_transform=target_transform, download=True)
    return dataset


def load_cifar10(root, batch_size, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_cifar10(root, train, transform, target_transform)
    return load_balanced_dataset(dataset, batch_size, **kwargs)


def fetch_mnist(root, train=False, transform=None, target_transform=None):
    transform = transform if transform is not None else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root, train=train, transform=transform, target_transform=target_transform, download=True)
    return dataset


def load_mnist(root, batch_size, data_amount_per_class, num_classes, train=False, transform=None, target_transform=None, **kwargs):
    dataset = fetch_mnist(root, train, transform, target_transform)
    return load_balanced_dataset(dataset, batch_size, data_amount_per_class, num_classes, **kwargs)