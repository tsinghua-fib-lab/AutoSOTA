import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
from tinyimagenet import TinyImageNet

DATA_ROOT = "/datasets"
DATA_ROOT_SHARED = "/share/datasets"


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class TorchDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        pin_memory=True,
        timeout=0,
        worker_init_fn=None,
        persistent_workers=True,
        prefetch_factor=2,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=True if batch_sampler is None else None,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )


def get_dataloaders_cifar10(batch_size: int):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        os.path.join(DATA_ROOT, "cifar10"), transform=t, download=True, train=True,
    )
    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.CIFAR10(
        os.path.join(DATA_ROOT, "cifar10"), transform=t_val, download=True, train=False,
    )
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_dataloaders_cifar10F(batch_size: int):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: torch.flatten(x)),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: torch.flatten(x)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        os.path.join(DATA_ROOT, "cifar10"), transform=t, download=True, train=True,
    )
    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.CIFAR10(
        os.path.join(DATA_ROOT, "cifar10"), transform=t_val, download=True, train=False,
    )
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_dataloaders_fmnist(batch_size: int):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        os.path.join(DATA_ROOT, "fmnist"), transform=t, download=True, train=True,
    )
    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.FashionMNIST(
        os.path.join(DATA_ROOT, "fmnist"), transform=t, download=True, train=False,
    )
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_dataloaders_mnist(batch_size: int):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: torch.flatten(x)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.MNIST(
        os.path.join(DATA_ROOT, "mnist"), transform=t, download=True, train=True,
    )
    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.MNIST(
        os.path.join(DATA_ROOT, "mnist"), transform=t, download=True, train=False,
    )
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_dataloaders_cifar100(batch_size: int):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        lambda x: x.numpy()
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        os.path.join(DATA_ROOT, "cifar100"), transform=t, download=True, train=True,
    )
    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.CIFAR100(
        os.path.join(DATA_ROOT, "cifar100"), transform=t_val, download=True, train=False,
    )
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_dataloaders_tinyimagenet(batch_size: int, size: int):
    t_val = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=TinyImageNet.mean, std=TinyImageNet.std),
        lambda x: x.numpy()
    ])

    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size, padding=size - 56),
        transforms.ToTensor(),
        transforms.Normalize(mean=TinyImageNet.mean, std=TinyImageNet.std),
        lambda x: x.numpy()
    ])

    train_dataset = TinyImageNet(
        os.path.join(DATA_ROOT, "tinyimagenet"), split="train", transform=t
    )
    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = TinyImageNet(
        os.path.join(DATA_ROOT, "tinyimagenet"), split="val", transform=t_val
    )
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


class ImageNet32(VisionDataset):
    train_list = [
        ['train_data_batch_9', '58417149b5ce31688f805341e7f57b4f'],
        ['train_data_batch_4', '876f7e6d6ddb1f52ecb654ffdc8293f6'],
        ['train_data_batch_3', '03d3dc4e850e23e1d526f268a0196296'],
        ['train_data_batch_7', '32ecc8ad6c55b1c9cb6cf79a2cc46094'],
        ['train_data_batch_1', '6c4495e65cd24a8c3019857ef9b758ee'],
        ['train_data_batch_5', 'c789bcdd1feed34a9bc58598a1a1bf7d'],
        ['train_data_batch_8', 'bdeb6da3d05771121992b48c59c69439'],
        ['train_data_batch_6', '8ce5344cb1e11f31bc507cae4c856083'],
        ['train_data_batch_2', '3dd727de4155836807dfc19f98c9fe70'],
        ['train_data_batch_10', '46ad60a1144aaf97a143914453b0dabb'],
    ]

    test_list = [
        ['val_data', '06c02b8b4c8de8b3a36b07859a49de6f'],
    ]

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

        downloaded_list = self.train_list if self.train else self.test_list

        self.data: Any = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f)
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target - 1)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


def get_dataloaders_imagenet32(batch_size: int):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        lambda x: x.numpy()
    ])

    train_dataset = ImageNet32(
        os.path.join(DATA_ROOT, "imagenet32"), train=True, transform=t
    )
    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_dataset = ImageNet32(
        os.path.join(DATA_ROOT, "imagenet32"), train=False, transform=t_val
    )
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


class ImageNet1kDataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None, select_classes=1000):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        self.targets = []
        for cls_name in tqdm(self.classes[:select_classes]):
            class_folder = os.path.join(self.root, cls_name)
            for fname in os.listdir(class_folder):
                img_path = os.path.join(class_folder, fname)
                if img_path == os.path.join(root, split, 'n02100877', 'n02100877_6036.JPEG'):
                    continue
                self.samples.append((img_path, self.class_to_idx[cls_name]))
                self.targets.append(self.class_to_idx[cls_name])

        self.class_counts = defaultdict(int)
        for class_idx in self.targets:
            self.class_counts[class_idx] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target


def get_dataloaders_imagenet_10(batch_size: int):
    t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda x: x.numpy()
    ])

    train_dataset = ImageNet1kDataset(
        root=os.path.join(DATA_ROOT_SHARED, "ImageNet_Full"), split='train', transform=t, select_classes=10
    )
    val_dataset = ImageNet1kDataset(
        root=os.path.join(DATA_ROOT_SHARED, "ImageNet_Full"), split='val', transform=t_val, select_classes=10
    )

    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = TorchDataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_dataloaders_imagenette(batch_size: int):
    t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda x: x.numpy()
    ])

    train_dataset = ImageNet1kDataset(
        root=os.path.join(DATA_ROOT_SHARED, "imagenette", "imagenette2"), split='train', transform=t, select_classes=10
    )
    val_dataset = ImageNet1kDataset(
        root=os.path.join(DATA_ROOT_SHARED, "imagenette", "imagenette2"), split='val', transform=t_val, select_classes=10
    )

    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = TorchDataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


class Galaxy10Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, indices=None, transform=None):
        self.transform = transform
        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            self.images = np.array(f['images'])
            self.labels = np.array(f['ans'])

        if indices is not None:
            self.images = self.images[indices]
            self.labels = self.labels[indices]

        self.images = self.images.astype(np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[index])
        return img, label


def get_dataloaders_galaxy10(batch_size: int, test_size=0.1):
    h5_path = os.path.join(DATA_ROOT_SHARED, "galaxy10", "Galaxy10_DECals.h5")

    with h5py.File(h5_path, 'r') as f:
        n_samples = len(f['ans'])

    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

    t = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.Resize(112),
        transforms.ToTensor(),
        lambda x: x.numpy()
    ])

    t_val = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        lambda x: x.numpy()
    ])

    train_dataset = Galaxy10Dataset(h5_path, indices=train_idx, transform=t)
    test_dataset = Galaxy10Dataset(h5_path, indices=test_idx, transform=t_val)

    train_dataloader = TorchDataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = TorchDataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, test_dataloader


def get_loader(dn, batch_size, size=64):
    if dn == 'MNIST':
        return get_dataloaders_mnist(batch_size)
    elif dn == 'FashionMNIST':
        return get_dataloaders_fmnist(batch_size)
    elif dn == 'CIFAR10':
        return get_dataloaders_cifar10(batch_size)
    elif dn == 'CIFAR10F':
        return get_dataloaders_cifar10F(batch_size)
    elif dn == 'CIFAR100':
        return get_dataloaders_cifar100(batch_size)
    elif dn == 'TinyImageNet':
        return get_dataloaders_tinyimagenet(batch_size, size)
    elif dn == 'ImageNet32':
        return get_dataloaders_imagenet32(batch_size)
    elif dn == 'ImageNet':
        return get_dataloaders_imagenet_10(batch_size)
    elif dn == 'Galaxy10':
        return get_dataloaders_galaxy10(batch_size)
    elif dn == 'ImageNette':
        return get_dataloaders_imagenette(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dn}")


def get_datasetinfo(dataset, model_name=None):
    if dataset == "MNIST":
        return 10, 28
    elif dataset == "FashionMNIST":
        return 10, 28
    elif dataset == "CIFAR10":
        return 10, 32
    elif dataset == "CIFAR100":
        return 100, 32
    elif dataset == "TinyImageNet":
        if model_name in ['VGG5', 'VGG7']:
            return 200, 56
        else:
            return 200, 64
    elif dataset == "ImageNet32":
        return 1000, 32
    elif dataset == "ImageNet":
        return 10, 224
    elif dataset == "Galaxy10":
        return 10, 256
    elif dataset == "ImageNette":
        return 10, 224
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
