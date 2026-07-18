import re
import scipy.io as sio

from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder

from pathlib import Path
from typing import List


def keyword_match(name: str, keywords: List[str]) -> bool:
    for kw in keywords:
        pattern = re.escape(kw.lower()).replace(r"\ ", r"[\s-]")
        pattern = r"\b" + pattern + r"\b"
        
        if re.search(pattern, name):
            return True
            
    return False


def extract_subset(
    dataset: ImageFolder,
    classes: List[str],
    devkit_path: Path,
) -> Subset:
    """
    Extracts ImageNet1K subset from class names
    (provided in natural language / keyword).
    """
    if len(classes) == 0:
        return dataset

    meta = sio.loadmat(devkit_path, squeeze_me=True)['synsets']

    syn_to_name = {
        str(entry[1]): str(entry[2]).lower()
        for entry in meta
    }

    selected_synsets = {
        syn for syn, name in syn_to_name.items()
        if keyword_match(name, classes)
    }

    subset_idxs = [
        idx for idx, (path, _) in enumerate(dataset.samples)
        if Path(path).parts[-2] in selected_synsets
    ]
    return Subset(dataset, subset_idxs)


class ImageNetToySampleDataset(Dataset):
    """
    Toy Sample Dataset for visualization.
    """
    def __init__(self, root, transform=None):
        self.wnid_to_imagenet_idx = {
            "n01491361": 3,
            "n01518878": 9,
            "n02129604": 292,
            "n02165456": 301,
            "n02279972": 323,
            "n02509815": 387,
            "n02510455": 388,
            "n04285008": 817,
        }
        self.ds = ImageFolder(root, transform=transform)

        self.local_to_imagenet = {
            local_idx: self.wnid_to_imagenet_idx[wnid]
            for wnid, local_idx in self.ds.class_to_idx.items()
        }

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, local_y = self.ds[i]
        y = self.local_to_imagenet[local_y]
        return x, y


class SingleImageMultiLabelDataset(Dataset):
    def __init__(self, image, labels, transform=None):
        self.image = image
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.image
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
