import os
import logging

import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO

from .utils import get_transform


logger = logging.getLogger(__name__)


class COCO2014(Dataset):
    def __init__(self, root, split='train'):
        self.root = os.path.abspath(root)
        self.img_dir = os.path.join(self.root, f'{split}2014')
        self.coco = COCO(os.path.join(root, f'annotations/instances_{split}2014.json'))
        self.split = split
        self.transform = get_transform(224, split == 'train')
        self.ids = list(self.coco.imgToAnns.keys())
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        self.load_img_and_label()
        self.num_classes = 80

    def load_img_and_label(self):
        self.img_files = []
        self.labels = []
        for i in range(len(self.ids)):
            img_id = self.ids[i]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anno = self.coco.loadAnns(ann_ids)

            label = []
            for obj in anno:
                label.append(self.cat2cat[obj['category_id']])
            self.labels.append(label)

            path = self.coco.loadImgs(img_id)[0]['file_name']
            file = os.path.join(self.img_dir, path)
            self.img_files.append(file)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[self.labels[index]] = 1

        img = self.transform(
            Image.open(self.img_files[index]).convert('RGB')
        )
        
        return img, target
