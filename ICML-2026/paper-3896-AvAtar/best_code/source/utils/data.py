# Image-Text Retrieval

import os
import json
import numpy as np
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10C(Dataset):
    def __init__(
        self,
        root,
        corruption="gaussian_noise",
        severity=5,
        transform=None,
        *args,
        **kwargs
    ):
        assert 1 <= severity <= 5, "severity must be in [1, 5]"

        self.transform = transform
        self.corruption = corruption
        self.severity = severity

        data_path = os.path.join(root, f"CIFAR-10-C/{corruption}.npy")
        labels_path = os.path.join(root, "CIFAR-10-C/labels.npy")

        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

        start = (severity - 1) * 10000
        end = severity * 10000

        self.data = self.data[start:end]
        self.labels = self.labels[:10000]
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ImageNetC(Dataset):
    def __init__(
        self,
        root,
        corruption="gaussian_noise",
        severity=5,
        transform=None,
        *args,
        **kwargs
    ):
        """
        Args:
            root (str): root directory containing ImageNet-C
                         e.g. datasets/ImageNet-C
            corruption (str): corruption type
            severity (int): severity level in [1, 5]
            transform: torchvision transforms
        """
        assert 1 <= severity <= 5, "severity must be in [1, 5]"

        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        # Load ImageNet class index
        class_index_path = os.path.join(root, "imagenet_class_index.json")
        assert os.path.isfile(class_index_path), \
            f"Missing imagenet_class_index.json at {class_index_path}"
        with open(class_index_path, "r") as f:
            class_index = json.load(f)

        idx_to_synset = {int(k): v[0] for k, v in class_index.items()}
        idx_to_name = {int(k): v[1] for k, v in class_index.items()}
        self.classes = [idx_to_name[i] for i in range(1000)]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        synset_to_idx = {idx_to_synset[i]: i for i in range(1000)}

        # Path: ImageNet-C/{corruption}/{severity}/
        data_root = os.path.join(root, corruption, str(severity))
        assert os.path.isdir(data_root), f"Path not found: {data_root}"

        self.samples = []

        synset_list = sorted(os.listdir(data_root))
        for synset in synset_list:
            synset_dir = os.path.join(data_root, synset)
            if not os.path.isdir(synset_dir):
                continue
            if synset not in synset_to_idx:
                continue

            label = synset_to_idx[synset]

            for fname in os.listdir(synset_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(synset_dir, fname)
                    self.samples.append((path, label))

        assert len(self.samples) > 0, "No images found!"

        # Reorder samples into round-robin order
        samples_by_label = defaultdict(list)
        for path, label in self.samples:
            samples_by_label[label].append(path)

        reordered_samples = []
        num_per_label = 50
        for i in range(num_per_label):
            for label in range(1000):
                if i < len(samples_by_label[label]):
                    path = samples_by_label[label][i]
                    reordered_samples.append((path, label))

        self.samples = reordered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


dataset_class_dict = {
    "CIFAR10-C": CIFAR10C,
    "ImageNet-C": ImageNetC,
}


# Image-Text Grounding
from typing import Any, Dict, Tuple, List
import json
import torch
from torchvision.datasets import Flickr30k
from pycocotools.coco import COCO
from PIL import Image
import os

from flickr30k_entities_utils import (
    get_annotations,
    get_sentence_data,
)


class CocoDetectionWithCaptions:
    def __init__(
        self,
        img_root,
        inst_ann_file,
        cap_ann_file,
        transform=None,
    ):
        self.img_root = img_root
        self.transform = transform

        self.inst_coco = COCO(inst_ann_file)
        self.cap_coco = COCO(cap_ann_file)

        # image ids that appear in both
        self.img_ids = sorted(
            set(self.inst_coco.getImgIds()) &
            set(self.cap_coco.getImgIds())
        )

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # ---- image ----
        img_info = self.inst_coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # ---- detection annotations ----
        ann_ids = self.inst_coco.getAnnIds(imgIds=img_id)
        target = self.inst_coco.loadAnns(ann_ids)

        # ---- captions ----
        cap_ids = self.cap_coco.getAnnIds(imgIds=img_id)
        caps = self.cap_coco.loadAnns(cap_ids)
        captions = [c["caption"] for c in caps]

        return image, target, captions


class Flickr30kEntities:
    """
    Flickr30k Entities dataset built on top of torchvision.datasets.Flickr30k.

    Loads:
      - captions (.token)
      - sentence-level phrase annotations (.txt)
      - entity bounding boxes (.xml)
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        map_file: str,
        entity_ann_root: str,
        sentence_ann_root: str,
        transform=None,
        target_transform=None,
    ):
        """
        Args:
            root: path to image directory
            ann_file: path to results_20130124.token
            entity_ann_root: path to Annotations/ (XML files)
            sentence_ann_root: path to Sentences/ (TXT files)
            map_ann_root: path to some mapping annotations
        """

        self.entity_ann_root = entity_ann_root
        self.sentence_ann_root = sentence_ann_root

        # --- Base Flickr30k (images + captions) ---
        self.base = Flickr30k(
            root=root,
            ann_file=ann_file,
            transform=transform,
            target_transform=target_transform
        )

        # global id to name
        self.global_map = json.load(open(map_file, "r"))
        self.cat_id_to_name = {}
        for img_map in self.global_map.values():
            for global_id, phrase in img_map.values():
                self.cat_id_to_name[int(global_id)] = phrase

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Tuple[Any, List[Dict[str, Any]]]:
        image, _ = self.base[index]

        image_name = self.base.ids[index]                 # e.g. "36979.jpg"
        image_map = self.global_map[image_name]           # phrase_id → (global_id, text)
        image_id = int(os.path.splitext(image_name)[0])

        # --- load annotations ---
        xml_path = os.path.join(self.entity_ann_root, f"{image_id}.xml")
        sentence_path = os.path.join(self.sentence_ann_root, f"{image_id}.txt")

        ann = get_annotations(xml_path)
        sentences = get_sentence_data(sentence_path)
        captions = [s["sentence"] for s in sentences]

        # phrase_id → local label
        # phrase_ids = sorted(ann.get("boxes", {}).keys())
        # phrase_to_label = {pid: i for i, pid in enumerate(phrase_ids)}

        target: List[Dict[str, Any]] = []

        for pid, box_list in ann.get("boxes", {}).items():
            if pid not in image_map:
                continue

            global_id, _ = image_map[pid]
            for x1, y1, x2, y2 in box_list:
                target.append({
                    "image_id": image_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],   # xywh
                    "category_id": int(global_id),
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                })

        return image, target, captions


def load_itg_dataset(name):
    if name == "COCO":
        return CocoDetectionWithCaptions(
            img_root="/home/qiyu6/YOLO-ITG/datasets/coco/images/val2017",
            inst_ann_file="/home/qiyu6/YOLO-ITG/datasets/coco/annotations/instances_val2017.json",
            cap_ann_file="/home/qiyu6/YOLO-ITG/datasets/coco/annotations/captions_val2017.json"
        )
    elif name == "Flickr30K":
        return Flickr30kEntities(
            root="/home/qiyu6/YOLO-ITG/datasets/flickr30k/flickr30k-images",
            ann_file="/home/qiyu6/YOLO-ITG/datasets/flickr30k/annotations/results_20130124.token",
            map_file="/home/qiyu6/YOLO-ITG/datasets/flickr30k/annotations/flickr30k_phrase_map.json",
            entity_ann_root="/home/qiyu6/YOLO-ITG/datasets/flickr30k/annotations/Annotations",
            sentence_ann_root="/home/qiyu6/YOLO-ITG/datasets/flickr30k/annotations/Sentences",
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    

def load_id_map(dataset, name):
    if name == "COCO":
        cats = dataset.inst_coco.loadCats(dataset.inst_coco.getCatIds())
        return {c["id"]: c["name"] for c in cats}
    elif name == "Flickr30K":
        return dataset.cat_id_to_name
    else:
        raise ValueError(f"Unknown dataset: {name}")
