from typing import Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import csv
import numpy as np
from torchvision import transforms

class CSV:
    def __init__(self, headers, indices, data):
        self.header = headers
        self.index= indices
        self.data = data

def default_celeba_transform(split,size = 64):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def blond_male_transform():
    return lambda x: (x[[20,9]]+1)//2

def male_smile_transform():
    return lambda x: (x[[20,31]] +1)//2
        
class CelebADataset(Dataset):
    def __init__(self, root, split='train',size=64,transforms=None,target_transform=None):
        self.root = root
        self.base_folder = 'celeba'
        self.img_dir = 'img_align_celeba'
        attr_file = 'list_attr_celeba.txt'
        partition_file = 'list_eval_partition.txt'
        if transforms is None:
            self.transform = default_celeba_transform(split,size)
        else:
            self.transform = transforms
        self.target_transform = target_transform
        # Load attributes and partitions using the _load_csv method
        self.attributes_csv = self._load_csv(attr_file, header=1)
        self.splits_csv = self._load_csv(partition_file)
        # Filter images based on the split
        mask = self.filter_data(split)

        self.filename = [self.splits_csv.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.attributes = self.attributes_csv.data[mask]
        self.attr_names = self.attributes_csv.header

    def filter_data(self,split):
        partition_map = {'train': 0, 'val': 1, 'test': 2}               
        split_ = partition_map[split]
        mask = slice(None) if split_ is None else (self.splits_csv.data == split_).squeeze()
        return mask

    def _load_csv(self, filename: str, header: Optional[int] = None) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]
        return CSV(headers, indices, torch.tensor(data_int))
        
    def __len__(self) -> int:
        return len(self.attributes)
    
    def __getitem__(self, idx):
        img_name = os.path.join(f"{self.root}/{self.base_folder}/{self.img_dir}", self.filename[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        
        attrs = self.attributes[idx]
        if self.target_transform:
            attrs = self.target_transform(attrs)

        return {"X": image, "label": attrs, "idx": int(self.filename[idx].split(".")[0])}


class AttrCelebALatent(CelebADataset):
    def __init__(self, celeba_dir,latent_dir,split='train'):
        self.root = celeba_dir
        self.base_folder = 'celeba'
        self.img_dir = 'img_align_celeba'
        attr_file = 'list_attr_celeba.txt'
        partition_file = 'list_eval_partition.txt'
        # Load attributes and partitions using the _load_csv method
        self.attributes_csv = self._load_csv(attr_file, header=1)
        self.splits_csv = self._load_csv(partition_file)
        # Filter images based on the split
        mask = self.filter_data(split)
        self.filename = [self.splits_csv.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.attributes = self.attributes_csv.data[mask]
        self.attr_names = self.attributes_csv.header
        self.latent_dir = latent_dir

    
    def filter_data(self,split):
        raise NotImplementedError
        
    def __getitem__(self, index):
        self.images = np.load(self.latent_dir+"/{:06d}.npy".format(int(self.filename[index].split(".")[0])))
        self.labels = self.attributes[index]
        return {"X":torch.tensor(self.images), "label": (self.labels[[20,9]]+1)//2, 'label_null': torch.ones_like(self.labels[[20,9]])*2}
    def __len__(self):
        return len(self.attributes)

def male_hair_colors_one_hot_transform():
    """
    Indices: Black(8), Blond(9), Brown(11), Gray(17), Bald(4)
    Returns: [Male (0-1), HairColor (0-4)] with
    HairColor: Black(0), Blond(1), Brown(2), Gray(3), Bald(4)
    """
    def transform(x):
        hair_indices = [8, 9, 11, 17, 4]
        hair_values = x[hair_indices]
        
        hair_color_idx = torch.argmax((hair_values == 1).float())
        
        male = (x[20] + 1) // 2
        return torch.stack([male, hair_color_idx.long()])
    
    return transform


class CelebAPixelMaleHairColors(CelebADataset):
    def filter_data(self, split):
        partition_map = {'train': 0, 'val': 1, 'test': 2}               
        split_mask = (self.splits_csv.data == partition_map[split]).squeeze()
        
        # Indices: Black(8), Blond(9), Brown(11), Gray(17), Bald(4)
        hair_indices = [8, 9, 11, 17, 4]
        hair_attr = self.attributes_csv.data[:, hair_indices]
        
        # We only want rows where EXACTLY one of these is 1
        clean_mask = (hair_attr == 1).sum(dim=1) == 1
        final_mask = split_mask & clean_mask
        print(f"Split {split}: {final_mask.sum()} samples kept out of {split_mask.sum()}")
        
        return final_mask
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        full_attrs = data["label"]
        
        # Use our updated one-hot transform
        transform = male_hair_colors_one_hot_transform()
        labels = transform(full_attrs)
        
        return {
            "X": data["X"], 
            "label": labels, 
            "label_null": torch.tensor([2, 5]),
            "idx": data["idx"]
        }


class CelebALatentMaleHairColors(AttrCelebALatent):
    """
    - Loads latent features instead of images.
    - Includes ALL groups (Hair colors: Black/Blond/Brown/Gray/Bald, Male/Female).
    - Labels are [Male, HairColor] where HairColor is categorical (0-4).
    """
    def __getitem__(self, index):
        self.images = np.load(self.latent_dir+"/{:06d}.npy".format(int(self.filename[index].split(".")[0])))
        self.labels = self.attributes[index]
        
        transform = male_hair_colors_one_hot_transform()
        labels = transform(self.labels)
        
        return {"X":torch.tensor(self.images), "label": labels, 'label_null': torch.tensor([2, 5])}
    
    def filter_data(self, split):
        partition_map = {'train': 0, 'val': 1, 'test': 2}               
        split_ = partition_map[split]
        mask = slice(None) if split_ is None else (self.splits_csv.data == split_).squeeze()
        return mask 
