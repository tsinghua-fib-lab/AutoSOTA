import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, '../data')
os.environ['SCIKIT_ML_LEARN_DATA'] = os.path.join(CURRENT_DIR, '../data')


class MultiLabelDataset(Dataset):
    def __init__(self, X, y, num_features, num_classes):
        self.X = X
        self.y = y
        self.num_features = num_features
        self.num_classes = num_classes

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


def get_raw_data(dataset_name: str, role: str):
    assert role in ['train', 'val', 'test'], f'Invalid dataset role: {role}'

    data = np.load(os.path.join(DATA_DIR, f'{dataset_name}_{role}.npz'))
    X, y = torch.from_numpy(data['X']).float(), torch.from_numpy(data['y']).float()
    return X, y


def get_dataloader(config: dict):
    role = config['role']
    assert role in ['train', 'val', 'test'], f'Invalid dataset role: {role}'

    if config['dataset_name'] == 'coco2014_img':
        from .coco import COCO2014
        dataset = COCO2014(config['data_root'], split=role)
    elif config['dataset_name'] == 'voc2007_img':
        from .voc import VOC2007
        dataset = VOC2007(config['data_root'], split=role)
    else:
        X,y = get_raw_data(config['dataset_name'], role)
        dataset = MultiLabelDataset(X, y, X.size(1), y.size(1))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(role == 'train'),
        num_workers=min(1, os.cpu_count() - 1),
    )
