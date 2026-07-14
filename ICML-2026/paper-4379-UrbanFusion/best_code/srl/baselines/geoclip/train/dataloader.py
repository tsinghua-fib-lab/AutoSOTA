"""
Original code from: https://github.com/VicenteVivan/geo-clip

Original source:
Vivanco, Vicente; Nayak, Gaurav Kumar; Shah, Mubarak.
"GeoCLIP: CLIP-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization."
NeurIPS 2023. arXiv preprint published September 27, 2023.
"""

import os
from os.path import exists

import numpy as np
import pandas as pd
import torch
from PIL import Image as im
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def img_train_transform():
    train_transform_list = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return train_transform_list


def img_val_transform():
    val_transform_list = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return val_transform_list


class GeoDataLoader(Dataset):
    """
    DataLoader for image-gps datasets.

    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.

    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise OSError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        for _, row in tqdm(
            dataset_info.iterrows(), desc="Loading image paths and coordinates"
        ):
            filename = os.path.join(self.dataset_folder, row["IMG_FILE"])
            if exists(filename):
                images.append(filename)
                latitude = float(row["LAT"])
                longitude = float(row["LON"])
                coordinates.append((latitude, longitude))

        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = self.coordinates[idx]

        image = im.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, gps
