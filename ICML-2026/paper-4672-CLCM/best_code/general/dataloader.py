"""
End-to-end continual learning dataloaders for image classification.

Unlike the main codebase's dataloaders (which pre-encode images through a
frozen ResNet), these provide raw images so that the *entire* network
(including the convolutional backbone) is trained end-to-end by EFC.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Dataset wrapper: adds one-hot labels & optional class remapping
# ---------------------------------------------------------------------------

class OneHotDataset(Dataset):
    """
    Wraps a torchvision-style dataset to return one-hot targets.

    Args:
        base_dataset: Underlying dataset (or Subset) returning (image, int_label).
        num_classes:  Length of the one-hot vector.
        label_offset: Subtracted from the raw label before encoding.
                      For TaskIL this maps global labels to task-local indices.
    """

    def __init__(self, base_dataset, num_classes, label_offset=0):
        self.base = base_dataset
        self.num_classes = num_classes
        self.label_offset = label_offset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        local = label - self.label_offset
        one_hot = torch.zeros(self.num_classes)
        one_hot[local] = 1.0
        return img, one_hot


# ---------------------------------------------------------------------------
# Continual learning dataloader
# ---------------------------------------------------------------------------

class EndToEndCLDataloader:
    """
    Provides per-task (train, test) DataLoader pairs for continual learning
    with raw images (no pre-encoding).

    Supports:
      - TinyImageNet  (64x64, 200 classes, default 10 tasks x 20 classes)
      - CIFAR-10      (32x32,  10 classes, default  5 tasks x  2 classes)
    """

    def __init__(self, config, dataset_name="TinyImageNet"):
        self.config = config
        self.dataset_name = dataset_name
        self.num_tasks = config.num_tasks
        self.classes_per_task = config.classes_per_task
        self.batch_size = config.batch_size
        self.num_workers = getattr(config, "num_workers", 4)

        self._setup_transforms()
        self._load_datasets()
        self._define_tasks()
        self._precompute_indices()

    # ---- transforms ----

    def _setup_transforms(self):
        if self.dataset_name == "TinyImageNet":
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=8),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ])
        elif self.dataset_name == "CIFAR10":
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ])

    # ---- dataset loading ----

    def _load_datasets(self):
        if self.dataset_name == "TinyImageNet":
            self.train_dataset = datasets.ImageFolder(
                root="./data/tiny-imagenet-200/train",
                transform=self.train_transform,
            )
            self.test_dataset = datasets.ImageFolder(
                root="./data/tiny-imagenet-200/val",
                transform=self.test_transform,
            )
        elif self.dataset_name == "CIFAR10":
            self.train_dataset = datasets.CIFAR10(
                root="./data", train=True,
                transform=self.train_transform, download=True,
            )
            self.test_dataset = datasets.CIFAR10(
                root="./data", train=False,
                transform=self.test_transform, download=True,
            )

    # ---- task definitions ----

    def _define_tasks(self):
        self.tasks = [
            list(
                range(
                    i * self.classes_per_task, (i + 1) * self.classes_per_task
                )
            )
            for i in range(self.num_tasks)
        ]

    def _precompute_indices(self):
        train_targets = torch.as_tensor(self.train_dataset.targets)
        test_targets = torch.as_tensor(self.test_dataset.targets)

        self.task_train_idx = {}
        self.task_test_idx = {}
        for t in range(self.num_tasks):
            classes = torch.tensor(self.tasks[t])
            self.task_train_idx[t] = torch.where(
                torch.isin(train_targets, classes)
            )[0].tolist()
            self.task_test_idx[t] = torch.where(
                torch.isin(test_targets, classes)
            )[0].tolist()

    # ---- dataloader creation ----

    def _make_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_dataloaders(self, task_id):
        setting = self.config.setting.lower()
        if "taskil" in setting:
            return self._get_taskil_loaders(task_id)
        else:
            return self._get_classil_loaders(task_id)

    def _get_taskil_loaders(self, task_id):
        cpt = self.classes_per_task
        offset = task_id * cpt

        train_sub = Subset(self.train_dataset, self.task_train_idx[task_id])
        test_sub = Subset(self.test_dataset, self.task_test_idx[task_id])

        train_ds = OneHotDataset(train_sub, num_classes=cpt, label_offset=offset)
        test_ds = OneHotDataset(test_sub, num_classes=cpt, label_offset=offset)

        return (
            self._make_loader(train_ds, shuffle=True),
            self._make_loader(test_ds, shuffle=False),
        )

    def _get_classil_loaders(self, task_id):
        cpt = self.classes_per_task
        num_classes_so_far = (task_id + 1) * cpt

        # Training: only current task's samples
        train_sub = Subset(self.train_dataset, self.task_train_idx[task_id])
        train_ds = OneHotDataset(
            train_sub, num_classes=num_classes_so_far, label_offset=0
        )

        # Testing: all seen classes
        all_test_idx = []
        for t in range(task_id + 1):
            all_test_idx.extend(self.task_test_idx[t])
        test_sub = Subset(self.test_dataset, all_test_idx)
        test_ds = OneHotDataset(
            test_sub, num_classes=num_classes_so_far, label_offset=0
        )

        return (
            self._make_loader(train_ds, shuffle=True),
            self._make_loader(test_ds, shuffle=False),
        )

    def get_all_tasks_dataloaders(self):
        return [self.get_dataloaders(t) for t in range(self.num_tasks)]
