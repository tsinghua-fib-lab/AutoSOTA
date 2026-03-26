import pathlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

TRAIN_TRANSFORM = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3 if x.shape[0] == 1 else 1, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

TEST_TRANSFORM = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3 if x.shape[0] == 1 else 1, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class SkinDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_mapping: Dict[str, int],
        transform: transforms.Compose,
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.images = list(label_mapping.keys())
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_dir / self.images[index]
        image = Image.open(image_path.with_suffix(".jpg"))
        image = self.transform(image)
        target = self.label_mapping[image_path.stem]
        # target = torch.tensor(target, dtype=torch.long)
        return image, target


def get_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """
    weights -- list of class weighting (should be same length as dataset)
    """
    label_count = df["target"].value_counts().sort_index()
    weight_map = dict(1 / label_count)
    sampler = WeightedRandomSampler(
        [weight_map[r.target] for _, r in df.iterrows()],
        len(df),
        replacement=True,
    )
    return sampler
