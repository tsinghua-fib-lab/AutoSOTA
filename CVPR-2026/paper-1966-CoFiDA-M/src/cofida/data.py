import os
from glob import glob

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from cofida.utils import (
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MONET_COLUMNS,
    extract_lesion_id,
    get_label_from_path,
    normalise_image_type,
)


def find_images(root: str) -> list[str]:
    if not root or not os.path.isdir(root):
        return []
    paths: list[str] = []
    for sub in ("mel", "other"):
        subdir = os.path.join(root, sub)
        for extension in IMAGE_EXTENSIONS:
            paths.extend(glob(os.path.join(subdir, extension)))
    return sorted(paths)


def load_monet_lookup(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [column for column in MONET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df[MONET_COLUMNS].copy()
    df["image_type"] = df["image_type"].map(normalise_image_type)
    df = df.drop_duplicates(subset=["lesion_id", "image_type"], keep="first")
    monet_cols = [column for column in df.columns if column.startswith("MONET_")]
    lookup = {}
    for _, row in df.iterrows():
        lookup[(row["lesion_id"], row["image_type"])] = row[monet_cols].astype(float).values
    return lookup, monet_cols


def make_source_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_target_weak_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_target_strong_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.02),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SourceDataset(Dataset):
    def __init__(self, paths: list[str], monet_lut: dict, transform, num_concepts: int):
        self.paths = paths
        self.labels = [get_label_from_path(path) for path in paths]
        self.monet_lut = monet_lut
        self.transform = transform
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        lesion_id = extract_lesion_id(path)
        monet = torch.zeros(self.num_concepts, dtype=torch.float32)
        key = (lesion_id, "dermoscopic")
        if key in self.monet_lut:
            monet = torch.tensor(self.monet_lut[key], dtype=torch.float32)
        return {
            "img": image,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "monet": monet,
        }


class TargetDataset(Dataset):
    def __init__(self, paths: list[str], monet_lut: dict, weak_transform, strong_transform, num_concepts: int):
        self.paths = paths
        self.monet_lut = monet_lut
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        lesion_id = extract_lesion_id(path)
        monet = torch.zeros(self.num_concepts, dtype=torch.float32)
        key = (lesion_id, "clinical")
        if key in self.monet_lut:
            monet = torch.tensor(self.monet_lut[key], dtype=torch.float32)
        return {
            "img_w": self.weak_transform(image),
            "img_s": self.strong_transform(image),
            "monet": monet,
        }


class TargetValDataset(Dataset):
    def __init__(self, paths: list[str], transform, monet_lut: dict, num_concepts: int):
        self.paths = paths
        self.labels = [get_label_from_path(path) for path in paths]
        self.transform = transform
        self.monet_lut = monet_lut
        self.num_concepts = num_concepts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        lesion_id = extract_lesion_id(path)
        monet = torch.zeros(self.num_concepts, dtype=torch.float32)
        key = (lesion_id, "clinical")
        if key in self.monet_lut:
            monet = torch.tensor(self.monet_lut[key], dtype=torch.float32)
        return {
            "img": image,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "monet": monet,
        }


class TargetWithMONET(Dataset):
    def __init__(self, paths: list[str], monet_lut: dict, transform):
        self.paths = paths
        self.monet_lut = monet_lut
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        lesion_id = extract_lesion_id(path)
        monet = torch.tensor(self.monet_lut[(lesion_id, "clinical")], dtype=torch.float32)
        return {"img": image, "monet": monet, "path": path}


class MonetEvalDataset(Dataset):
    def __init__(self, base: datasets.ImageFolder, idx_to_positive: int, monet_lut: dict, num_concepts: int):
        self.base = base
        self.idx_to_positive = idx_to_positive
        self.monet_lut = monet_lut
        self.num_concepts = num_concepts
        self.samples = base.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.base.transform(self.base.loader(path).convert("RGB"))
        lesion_id = extract_lesion_id(path)
        monet = torch.zeros(self.num_concepts, dtype=torch.float32)
        key = (lesion_id, "clinical")
        if key in self.monet_lut:
            monet = torch.tensor(self.monet_lut[key], dtype=torch.float32)
        return {
            "path": path,
            "img": image,
            "y": torch.tensor(1 if label == self.idx_to_positive else 0, dtype=torch.long),
            "monet": monet,
        }


class BinaryImageFolderDataset(Dataset):
    def __init__(self, base: datasets.ImageFolder, idx_to_positive: int):
        self.base = base
        self.idx_to_positive = idx_to_positive
        self.samples = base.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.base.transform(self.base.loader(path).convert("RGB"))
        return {
            "path": path,
            "img": image,
            "y": torch.tensor(1 if label == self.idx_to_positive else 0, dtype=torch.long),
        }
