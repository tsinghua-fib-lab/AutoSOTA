import torchvision
import numpy as np
import torch
import os
from torchvision import transforms
from typing import Callable
from torch.utils.data import DataLoader


def whitening_zca(x: torch.Tensor, transpose=True, dataset: str = "CIFAR10"):
    path = os.path.join('./data/', dataset + "_zca.pt")
    zca = None
    try:
        zca = torch.load(path, map_location='cpu')['zca']
    except:
        pass

    if zca is None:
        x = x.cpu().numpy()
        if transpose:
            x = x.copy().transpose(0, 3, 1, 2)

        x = x.copy().reshape(x.shape[0], -1)

        cov = np.cov(x, rowvar=False)

        u, s, v = np.linalg.svd(cov)

        SMOOTHING_CONST = 1e-1
        zca = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + SMOOTHING_CONST)), u.T))
        zca = torch.from_numpy(zca).float()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'zca': zca}, path)

    return zca


class FastCIFAR10(torchvision.datasets.CIFAR10):
    
    """
    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', True)
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)

        self.split = split

        mean = torch.tensor((0.4914, 0.48216, 0.44653))
        std = torch.tensor((0.247, 0.2434, 0.2616))


        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)

        if self.train:
            if not isinstance(train_class, str):
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = np.array(self.targets)[index_class]
                self.len = self.data.shape[0]

        if zca:
            self.data = (self.data - mean) / std
            self.zca = whitening_zca(self.data)
            zca_whitening = torchvision.transforms.LinearTransformation(self.zca, torch.zeros(self.zca.size(1)))
        self.data = torch.tensor(self.data, dtype=torch.float)

        self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        if zca:
            self.data = zca_whitening(self.data)
            print("self.data.mean(), self.data.std()", self.data.mean(), self.data.std())

        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]

        return img, target
    
def make_cifar10_dataloader(args):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(
                (32, 32), padding=4, padding_mode="reflect"
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),

        ])
    trainset = FastCIFAR10(root=args.data_dir, train=True, download=True, zca=True, split='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=False)
    
    transform = None
    testset = FastCIFAR10(root=args.data_dir, train=False, download=True, zca=True, split='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=False)
    return trainloader, testloader

class FastCIFAR100(torchvision.datasets.CIFAR100):
    
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', True)
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)

        self.split = split

        mean = torch.tensor([n/255. for n in [129.3, 124.1, 112.4]])
        std = torch.tensor([n/255. for n in [68.2,  65.4,  70.4]])


        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)

        if self.train:
            if not isinstance(train_class, str):
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = np.array(self.targets)[index_class]
                self.len = self.data.shape[0]

        if zca:
            self.data = (self.data - mean) / std
            self.zca = whitening_zca(self.data, transpose=True, dataset='CIFAR100')
            zca_whitening = torchvision.transforms.LinearTransformation(self.zca, torch.zeros(self.zca.size(1)))
        self.data = torch.tensor(self.data, dtype=torch.float)

        self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        if zca:
            self.data = zca_whitening(self.data)
            print("self.data.mean(), self.data.std()", self.data.mean(), self.data.std())

        self.targets = torch.tensor(self.targets, device=device)
    
    def __getitem__(self, index: int):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[index]

        return img, target

def make_cifar100_dataloader(args):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(
                (32, 32), padding=4, padding_mode="reflect"
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),

        ])
    trainset = FastCIFAR100(root=args.data_dir, train=True, download=True, zca=True, split='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=False)
    
    transform = None
    testset = FastCIFAR100(root=args.data_dir, train=False, download=True, zca=True, split='test')
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=False)
    return trainloader, testloader


def classic_mnist_loader(data_dir: str, train_batch: int = 128, test_batch: int = 100,
                         transform_train: Callable = transforms.Compose([transforms.RandomCrop(28, padding=2),
                                                                         transforms.ToTensor(),
                                                                         transforms.Normalize((0.1307,), (0.3081,))]),
                         transform_test: Callable = transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.1307,), (0.3081,))]),
                         **kwargs):
    train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform_train)

    train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, **kwargs)

    test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False, **kwargs)

    return train_loader, test_loader