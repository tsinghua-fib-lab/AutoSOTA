# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, List, Tuple

import lightning.pytorch as pl
import numpy as np
import scipy.io
import torch
import tqdm
import torchvision
from kagglehub import dataset_download as kaggle_download
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.utils.data.dataset import Subset

from .utils.ITS.transform import AffineTransformation, multi_transform


class DataModule(pl.LightningDataModule):
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert hasattr(self.config, 'batch_size')
        assert hasattr(self.config, 'num_workers')

    def prepare_data(self):
        """Download data, split, etc. Only called on 1 GPU/TPU in distributed."""
        raise NotImplementedError

    def setup(self, stage: str):
        """Make assignments here (val/train/test split). Called on every GPU/TPU in DDP."""
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: Any) -> Any:
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.config.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.config.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.config.num_workers
        )


class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def loadmat(filename: str) -> dict:
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(data: dict) -> dict:
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in data:
        if isinstance(data[key], scipy.io.matlab.mat_struct):
            data[key] = _todict(data[key])
    return data


def _todict(matobj: scipy.io.matlab.mat_struct) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    assert hasattr(matobj, '_fieldnames')
    data = {}
    for strg in getattr(matobj, '_fieldnames'):
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            data[strg] = _todict(elem)
        else:
            data[strg] = elem
    return data


class PaddedMNIST(DataModule):
    urls = [
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/training.mat.zip",
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/test.mat.zip"
    ]

    def __init__(self, config):
        super().__init__(config)
        assert hasattr(self.config, 'data_dir')
        data_dir = Path(self.config.data_dir) / self.__class__.__name__
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"

    def prepare_data(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        for url in self.urls:
            filename = os.path.basename(url)
            filepath = self.raw_dir / filename

            if not os.path.exists(filepath):
                print(f"Downloading {url} to {filepath}...")
                download_url(url, filepath)

                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    print(f"Unzipping {filepath}...")
                    zip_ref.extractall(self.raw_dir)

        print("Data download complete.")

    def setup(self, stage: str):
        if stage in ('fit', 'validate'):
            data = loadmat((self.raw_dir / "training.mat").as_posix())
            labels = data['affNISTdata']['label_int']
            img = data['affNISTdata']['image'].transpose().reshape(len(labels), 40, 40)

            self.train_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            self.val_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            print("Warning: Validation dataset is identical to training dataset.")

        if stage == 'test':
            data = loadmat((self.raw_dir / "test.mat").as_posix())
            labels = data['affNISTdata']['label_int']
            img = data['affNISTdata']['image'].transpose().reshape(len(labels), 40, 40)

            test_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            # subset with fixed random seed
            with torch.random.fork_rng():
                torch.manual_seed(42)
                subset_indices = torch.randint(low=0, high=len(test_dataset) - 1, size=(10000, 1)).squeeze().tolist()
            self.test_dataset = Subset(test_dataset, subset_indices)

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Stacks the image and label tensors from a batch of samples."""
        images, labels = zip(*batch)
        images, labels = torch.stack(images, 0), torch.stack(labels, 0)
        # add channel dimension
        if images.ndim == 3:
            images = images[:, None]
        # PIL [0, 255] -> Tensor [0., 1.]
        if images.dtype == torch.uint8:
            images = images.float() / 255
        return images, labels


class AffNIST(PaddedMNIST):
    urls = [
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/training.mat.zip",
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test.mat.zip"
    ]

    def prepare_data(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        filepath = self.raw_dir / "training.mat.zip"

        if not os.path.exists(filepath):
            print(f"Downloading {self.urls[0]} to {filepath}...")
            download_url(self.urls[0], filepath)

            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                print(f"Unzipping {filepath}...")
                zip_ref.extractall(self.raw_dir)

        filepath = self.raw_dir / "affNIST_test.mat.zip"

        if not os.path.exists(filepath):
            print(f"Downloading {self.urls[1]} to {filepath}...")
            download_url(self.urls[1], filepath)

            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                print(f"Unzipping {filepath}...")
                zip_ref.extractall(self.raw_dir)
                os.rename(self.raw_dir / "test.mat", self.raw_dir / "affNIST_test.mat")

        print("Data download complete.")

    def setup(self, stage: str):
        if stage in ('fit', 'validate'):
            data = loadmat((self.raw_dir / "training.mat").as_posix())
            labels = data['affNISTdata']['label_int']
            img = data['affNISTdata']['image'].transpose().reshape(len(labels), 40, 40)

            self.train_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            self.val_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            print("Warning: Validation dataset is identical to training dataset.")

        if stage == 'test':
            data = loadmat((self.raw_dir / "affNIST_test.mat").as_posix())
            labels = data['affNISTdata']['label_int']
            img = data['affNISTdata']['image'].transpose().reshape(len(labels), 40, 40)

            test_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            # subset with fixed random seed
            with torch.random.fork_rng():
                torch.manual_seed(42)
                subset_indices = torch.randint(low=0, high=len(test_dataset) - 1, size=(10000, 1)).squeeze().tolist()
            self.test_dataset = Subset(test_dataset, subset_indices)


class HomNIST(PaddedMNIST):
    urls = [
        "https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/training.mat.zip"
    ]

    def prepare_data(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        filepath = self.raw_dir / "training.mat.zip"

        if not os.path.exists(filepath):
            print(f"Downloading {self.urls[0]} to {filepath}...")
            download_url(self.urls[0], filepath)

            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                print(f"Unzipping {filepath}...")
                zip_ref.extractall(self.raw_dir)

        filepath = self.raw_dir / "homNIST_test.mat"
        if not os.path.exists(filepath):
            download_filepath = kaggle_download("lachlanemacdonald/homnist", path="homNIST_test.mat")
            shutil.move(download_filepath, filepath)

        print("Data download complete.")

    def setup(self, stage: str):
        if stage in ('fit', 'validate'):
            data = loadmat((self.raw_dir / "training.mat").as_posix())
            labels = data['affNISTdata']['label_int']
            img = data['affNISTdata']['image'].transpose().reshape(len(labels), 40, 40)

            self.train_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            self.val_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))

            print("Warning: Validation dataset is identical to training dataset.")

        if stage == 'test':
            data = loadmat((self.raw_dir / "homNIST_test.mat").as_posix())
            labels = data['labels']
            img = data['img_data'].astype(np.float32)

            test_dataset = TensorDataset(torch.tensor(img), torch.tensor(labels))
            # subset with fixed random seed
            with torch.random.fork_rng():
                torch.manual_seed(42)
                subset_indices = torch.randint(low=0, high=len(test_dataset) - 1, size=(10000, 1)).squeeze().tolist()
            self.test_dataset = Subset(test_dataset, subset_indices)


class ITSMNIST(DataModule):
    def __init__(self, config, exclude_class=9, n_samples=9, test_val_split=[5000, 3991]):
        super().__init__(config)
        self.exclude_class = exclude_class
        self.n_samples = n_samples
        self.test_val_split = test_val_split
        self.its_mode = config.its_mode

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(6),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            torchvision.transforms.Lambda(lambda x: x.squeeze(0))
        ])
        assert self.its_mode in [2, 3, 4, 5], "its mode should be in [2, 3, 4, 5]"
        if self.its_mode == 2:
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
            ]
            self.domains = [(torch.pi,), (0.5,)]
        elif self.its_mode == 3:
            # values from the paper (ITS^3)
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
            ]
            self.domains = [(torch.pi,), (0.25,), (0.25,)]
        elif self.its_mode == 4:
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
                AffineTransformation.TRANSLATION.value
            ]
            self.domains = [(torch.pi,), (0.25,), (0.25,), (0.2,)]
        elif self.its_mode == 5:
            # values from the paper (ITS^5)
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
            ]
            self.domains = [(torch.pi,), (0.25,), (0.25,), (torch.pi,), (0.25,)]

        data_dir = Path(self.config.data_dir) / self.__class__.__name__
        self.raw_dir = Path(self.config.data_dir).as_posix()
        self.processed_dir = data_dir / "processed"
        self.processed_file = self.processed_dir / "mnist_transformed.pt"

    def prepare_data(self):
        """download MNIST dataset if not exists"""
        os.makedirs(self.raw_dir, exist_ok=True)
        torchvision.datasets.MNIST(
            root=self.raw_dir,
            train=True,
            download=True,
            target_transform=torch.tensor
        )
        torchvision.datasets.MNIST(
            root=self.raw_dir,
            train=False,
            download=True,
            target_transform=torch.tensor
        )

    def _apply_and_save_affine(self, dataset):
        """apply affine transformation to the test dataset and save the result"""
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_images, all_labels = [], []

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for x, y in tqdm.tqdm(loader, desc="Applying affine to test set"):
            if x.ndim == 3:
                x = x[:, None]  # add channel dimension
            n = torch.randint(0, self.n_samples,
                              (x.shape[0], len(self.transformations)))
            x_trans, _, _ = multi_transform(
                x, self.transformations, n,
                n_samples=self.n_samples, domain=self.domains
            )
            all_images.append(x_trans.cpu())
            all_labels.append(y.cpu())

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        torch.save({"images": all_images, "labels": all_labels}, self.processed_file)
        print(f"[MNIST] Saved transformed test set to {self.processed_file}")

    def _load_processed_dataset(self):
        data = torch.load(self.processed_file)
        images, labels = data["images"], data["labels"]
        return TensorDataset(images, labels)

    def setup(self, stage: str):
        """split data into train/val/test and apply affine transform to the test data"""
        if stage == 'fit':
            full_train = torchvision.datasets.MNIST(
                root=self.raw_dir,
                train=True,
                download=False,
                transform=self.transform
            )
            # exclude 9 due to the rotation (same as 6)
            idxs = np.where(full_train.targets != self.exclude_class)[0].tolist()
            full_train = Subset(full_train, idxs)
            # original mnist
            self.train_dataset = full_train

        if stage in ('validate', 'test', None):
            full_test = torchvision.datasets.MNIST(
                root=self.raw_dir,
                train=False,
                download=False,
                transform=self.transform
            )
            # exclude 9 due to the rotation (same as 6)
            idxs = np.where(full_test.targets != self.exclude_class)[0].tolist()
            full_test = Subset(full_test, idxs)
            test_set, val_set = random_split(full_test, self.test_val_split)
            if not self.processed_file.exists():
                self._apply_and_save_affine(test_set)
            self.val_dataset = val_set
            self.test_dataset = self._load_processed_dataset()

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        labels = torch.tensor(labels)
        if images.ndim == 3:
            images = images[:, None]  # add channel dimension
        return images, labels


class FashionMNIST(DataModule):
    def __init__(self, config, n_samples=9, test_val_split=[5000, 5000]):
        super().__init__(config)
        self.n_samples = n_samples
        self.test_val_split = test_val_split
        self.its_mode = config.its_mode

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(6),
            torchvision.transforms.ToTensor()
        ])
        assert self.its_mode in [2, 3, 4, 5], "its mode should be in [2, 3, 4, 5]"
        if self.its_mode == 2:
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
            ]
            self.domains = [(torch.pi,), (0.5,)]
        elif self.its_mode == 3:
            # values from the paper (ITS^3)
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
            ]
            self.domains = [(torch.pi,), (0.25,), (0.25,)]
        elif self.its_mode == 4:
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
                AffineTransformation.TRANSLATION.value
            ]
            self.domains = [(torch.pi,), (0.25,), (0.25,), (0.2,)]
        elif self.its_mode == 5:
            # values from the paper (ITS^5)
            self.transformations = [
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
                AffineTransformation.SHEARING.value,
                AffineTransformation.ROTATION.value,
                AffineTransformation.SCALING.value,
            ]
            self.domains = [(torch.pi,), (0.25,), (0.25,), (torch.pi,), (0.25,)]

        data_dir = Path(self.config.data_dir) / self.__class__.__name__
        self.raw_dir = Path(self.config.data_dir).as_posix()
        self.processed_dir = data_dir / "processed"
        self.processed_file = self.processed_dir / "fmnist_transformed.pt"

    def prepare_data(self):
        """download FashionMNIST dataset if not exists"""
        os.makedirs(self.raw_dir, exist_ok=True)
        torchvision.datasets.FashionMNIST(root=self.raw_dir, train=True, download=True)
        torchvision.datasets.FashionMNIST(root=self.raw_dir, train=False, download=True)

    def _apply_and_save_affine(self, dataset):
        """apply affine transformation to the test dataset and save the result"""
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_images, all_labels = [], []

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        for x, y in tqdm.tqdm(loader, desc="Applying affine to test set"):
            if x.ndim == 3:
                x = x[:, None]  # add channel dimension
            n = torch.randint(0, self.n_samples,
                              (x.shape[0], len(self.transformations)))
            x_trans, _, _ = multi_transform(
                x, self.transformations, n,
                n_samples=self.n_samples, domain=self.domains
            )
            all_images.append(x_trans.cpu())
            all_labels.append(y.cpu())

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        torch.save({"images": all_images, "labels": all_labels}, self.processed_file)
        print(f"[FashionMNIST] Saved transformed test set to {self.processed_file}")

    def _load_processed_dataset(self):
        data = torch.load(self.processed_file)
        images, labels = data["images"], data["labels"]
        return TensorDataset(images, labels)

    def setup(self, stage: str):
        """split data into train/val/test and apply affine transform to the test data"""
        if stage == 'fit':
            full_train = torchvision.datasets.MNIST(
                root=self.raw_dir,
                train=True,
                download=False,
                transform=self.transform
            )
            # original fmnist
            self.train_dataset = full_train

        if stage in ('validate', 'test'):
            full_test = torchvision.datasets.MNIST(
                root=self.raw_dir,
                train=False,
                download=False,
                transform=self.transform
            )
            test_set, val_set = random_split(full_test, self.test_val_split)
            if not self.processed_file.exists():
                self._apply_and_save_affine(test_set)
            self.val_dataset = val_set
            self.test_dataset = self._load_processed_dataset()

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        labels = torch.tensor(labels) 
        if images.ndim == 3:
            images = images[:, None]  # add channel dimension
        return images, labels


if __name__ == "__main__":
    from easydict import EasyDict
    config_ = EasyDict({
        'data_dir': './experiments/datasets',
        'batch_size': 32,
        'num_workers': 4
    })
    dataset_ = AffNIST(config_)
    dataset_.prepare_data()
    dataset_.setup("test")
