import copy
import json
import os
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from typing import Tuple

def load_image(path: str, img_size: int):
    image = Image.open(path).convert("RGB")
    resize_transform = transforms.Resize((img_size, img_size))
    image = resize_transform(image)
    image = to_tensor(image) * 2 - 1
    return image

class DummyDataset(data.Dataset):
    def __init__(self, dataset_dir: str, img_size: int = 512, train: bool = True):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.data = []

        jpg_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
        for img_path in jpg_files:
            json_path = os.path.join(dataset_dir, os.path.splitext(img_path)[0] + ".json")
            assert os.path.exists(json_path)
            with open(json_path, 'r') as json_file:
                json_dict = json.load(json_file)
            assert "caption" in json_dict.keys()
            self.data.append({"img_path": os.path.join(dataset_dir, img_path), "caption": json_dict["caption"]})


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}
        # Load image
        sample["x"] = load_image(item["img_path"], img_size=self.img_size)
        sample["caption"] = item["caption"]
        return sample

class DataModule:
    def __init__(self, dataset_dir: str, batch_size: int = 1, img_size: int = 512):
            self.batch_size = batch_size

            train_dataset = DummyDataset(dataset_dir=dataset_dir, train=True, img_size=img_size)
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

            val_dataset = DummyDataset(dataset_dir=dataset_dir, train=False, img_size=img_size)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


    def train_dataloader(self):
            return self.train_loader

    def val_dataloader(self):
        return self.val_loader