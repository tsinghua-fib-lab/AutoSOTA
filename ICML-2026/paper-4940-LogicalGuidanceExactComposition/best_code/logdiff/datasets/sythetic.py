import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class SytheticData(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        metadata = os.path.join(root_dir, "metadata.json")
        with open(metadata) as f:
            self.metadata = [json.loads(x) for x in f.readlines()]
        self.transform = transform
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        filename = self.metadata[idx]['filename']
        img_name = os.path.join(self.root_dir, "gen", filename)
        if ".png" in img_name:
            image = Image.open(img_name)
        else:
            image = torch.tensor(np.load(img_name))
        if self.transform:
            image = self.transform(image)
        query = torch.tensor(self.metadata[idx]['query'])
        null_token = torch.tensor(self.metadata[idx]['null_token'])
        return {"X": image, "label": query, "label_null": null_token, "idx": idx}