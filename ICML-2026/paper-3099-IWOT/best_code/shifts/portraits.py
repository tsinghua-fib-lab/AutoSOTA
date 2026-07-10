import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import v2


class PORTRAITS:
    def __init__(self, dl_options, test_dl_options, size, grayscale, train_ratio=0.8):
        self.name = "PORTRAITS"
        self.num_classes = 2
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size),
        ]
        self.num_channels = 3
        if grayscale is True:
            self.num_channels = 1
            transforms.append(v2.Grayscale(1))
        self.input_size = size[0] * size[1] * self.num_channels
        transforms = v2.Compose(transforms)
        dataset = ImageFolderWithFilenames(root="data/Portraits", transform=transforms)

        # Print first year of dataset
        _, _, filename = dataset.get_item_with_year(0)
        first_year = filename.split("_")[0]
        print(f"First year of source dataset: {first_year}")

        # Check mid-index year and report
        mid_idx = int(len(dataset) / 2)
        _, _, filename = dataset.get_item_with_year(mid_idx)
        year = filename.split("_")[0]
        self.source_name = "P1900-" + str(year)
        self.target_name = "P" + str(year) + "-2010"
        print(f"Year separating source from target datasets: {year}")

        # Print last year of dataset
        _, _, filename = dataset.get_item_with_year(-1)
        final_year = filename.split("_")[0]
        print(f"Final year of target dataset: {final_year}")

        subset1_idx = list(range(0, mid_idx))
        subset2_idx = list(range(mid_idx, len(dataset)))
        subset1 = Subset(dataset, subset1_idx)
        subset2 = Subset(dataset, subset2_idx)

        train_size = int(train_ratio * len(subset1))
        test_size = int(len(subset1)) - train_size
        train_source, test_source = random_split(subset1, [train_size, test_size])
        train_size = int(train_ratio * len(subset2))
        test_size = int(len(subset2)) - train_size
        train_target, test_target = random_split(subset2, [train_size, test_size])

        # Load both datasets
        self.source_dataloader = DataLoader(train_source, **dl_options)
        self.target_dataloader = DataLoader(train_target, **dl_options)
        self.source_test_dataloader = DataLoader(test_source, **test_dl_options)
        self.target_test_dataloader = DataLoader(test_target, **test_dl_options)
        # calc_w_distance_label_shift(self)
        self.labels = np.array(["Female", "Male"])


class ImageFolderWithFilenames(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        # Traverse the folder and store image paths and filenames
        for root_dir, _, files in os.walk(self.root):
            for file_name in files:
                if file_name.endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(root_dir, file_name))
                    # Extract the folder name as label
                    label = os.path.basename(root_dir)
                    if label == "F":
                        self.labels.append(0)
                    else:
                        self.labels.append(1)

        # Sort the images and labels based on file names
        self.images, self.labels = zip(
            *sorted(
                zip(self.images, self.labels),
                key=lambda x: os.path.basename(
                    x[0]
                ),  # Sorting based on image file path (x[0])
            )
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]  # Get the corresponding label
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_item_with_year(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]  # Get the corresponding label
        if self.transform:
            image = self.transform(image)

        # Get the filename from the path
        filename = os.path.basename(image_path)

        return image, label, filename  # return the image and the filename
