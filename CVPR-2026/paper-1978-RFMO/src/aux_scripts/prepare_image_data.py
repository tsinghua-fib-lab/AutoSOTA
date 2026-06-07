import os
import sys
import argparse
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import VOCDetection, CocoDetection
from torchvision.models import resnet50, ResNet50_Weights
from pycocotools.coco import COCO

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '../data')

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model

    def forward(self, x):
        for name, module in self.model.named_children():
            if name == 'fc':
                break
            x = module(x)
        return x


class CocoDetection(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros(80, dtype=torch.long)
        for obj in target:
            output[self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((448, 448)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).to(DEVICE)
model.eval()


def extract_voc_features(data_root):
    os.makedirs(DATA_DIR, exist_ok=True)

    batch_size = 32

    category2index = {
        'aeroplane': 0,
        'bicycle': 1,
        'bird': 2,
        'boat': 3,
        'bottle': 4,
        'bus': 5,
        'car': 6,
        'cat': 7,
        'chair': 8,
        'cow': 9,
        'diningtable': 10,
        'dog': 11,
        'horse': 12,
        'motorbike': 13,
        'person': 14,
        'pottedplant': 15,
        'sheep': 16,
        'sofa': 17,
        'train': 18,
        'tvmonitor': 19,
    }

    with torch.no_grad():
        for split in ['train', 'val', 'test']:
            dataset = VOCDetection(
                root=os.path.join(data_root, 'voc2007'),
                year='2007',
                image_set=split,
                transform=transforms,
                # target_transform=target_transforms,
            )

            features, labels = [], []

            # some bugs in directly using Dataloader upon VOCDetection
            # so we need to manually iterate over the dataset

            for i in tqdm(range(0, len(dataset), batch_size), desc=f'Extracting {split} features'):
                batch_labels = []
                batch_images = []
                for j in range(i, min(i+batch_size, len(dataset))):
                    image, target = dataset[j]
                    label = np.zeros(len(category2index), dtype=np.int16)
                    for obj in target['annotation']['object']:
                        label[category2index[obj['name']]] = 1

                    batch_labels.append(label)
                    batch_images.append(image)

                batch_images = torch.stack(batch_images)
                features.append(model(batch_images.to(DEVICE)).cpu().numpy())
                labels.append(batch_labels)

            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)

            np.savez_compressed(f'../data/voc2007_{split}.npz', X=features, y=labels)


def extract_coco_features(data_root):
    os.makedirs(DATA_DIR, exist_ok=True)

    batch_size = 32

    for split in ['train', 'val']:
        coco = COCO(os.path.join(data_root, 'coco', f'annotations/instances_{split}2014.json'))

        features, labels = [], []

        dataset = CocoDetection(
            root=os.path.join(data_root, 'coco', f'{split}2014'), 
            annFile=os.path.join(data_root, 'coco', f'annotations/instances_{split}2014.json'), 
            transform=transforms
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(dataloader, desc=f'Extracting {split} features')):
                features.append(model(images.to(DEVICE)).cpu().numpy())
                labels.append(targets.numpy())

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        if split == 'train':
            from sklearn.model_selection import train_test_split
            train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.1, random_state=0)
            np.savez_compressed(f'../data/coco2014_{split}.npz', X=train_features, y=train_labels)
            np.savez_compressed(f'../data/coco2014_val.npz', X=val_features, y=val_labels)
        else:
            np.savez_compressed(f'../data/coco2014_test.npz', X=features, y=labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', '-d', type=str, choices=['voc2007', 'coco2014'])
    parser.add_argument('--data_root', '-r', type=str)
    args = parser.parse_args()
    
    if args.dataset_name == 'voc2007':
        extract_voc_features(args.data_root)
    elif args.dataset_name == 'coco2014':
        extract_coco_features(args.data_root)
    else:
        raise ValueError('Invalid dataset name')
