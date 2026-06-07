import torch

from .utils import get_transform
from torchvision.datasets import VOCDetection


object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']
category2index = dict(zip(object_categories, range(len(object_categories))))


class VOC2007(VOCDetection):
    def __init__(self, root, split):
        super().__init__(root=root, year='2007', image_set=split, transform=get_transform(224, split == 'train'))
        self.num_classes = 20

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        label = torch.zeros(len(object_categories))

        for obj in target['annotation']['object']:
            label[category2index[obj['name']]] = 1

        return img, label
