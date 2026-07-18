# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import List
import os

import numpy as np
import sklearn.utils.extmath
import torch.utils.data
import torchvision.transforms


class IndexManger(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes: List[int]):
        """Index to latent (= features) space and vice versa.
        Args:
          factor_sizes: List of integers with the number of distinct values for
            each of the factors.
        """
        self.factor_sizes = np.array(factor_sizes)
        self.num_total = np.prod(self.factor_sizes)
        self.factor_bases = self.num_total / np.cumprod(self.factor_sizes)

        self.index_to_feat = sklearn.utils.extmath.cartesian(
            [np.array(list(range(i))) for i in self.factor_sizes])


class BenchmarkDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, dataset_name, variant, mode,latent=False):
        super().__init__()
        lat = ""
        if latent:
            lat = "flux_"
            
        images_filename = "{}{}_{}_{}_images.npz".format(lat,dataset_name, variant,
                                                       mode)
        targets_filename = "{}{}_{}_{}_labels.npz".format(lat,dataset_name, variant,
                                                        mode)
        self.transform = torchvision.transforms.ToTensor()
        self._factor_sizes = None
        self._factor_names = None
        if dataset_name == 'dsprites':
            self._factor_sizes = [3, 6, 40, 32, 32]
            self._factor_names = [
                'shape', 'scale', 'orientation', 'x-position', 'y-position'
            ]
        elif dataset_name == 'shapes3d':
            self._factor_sizes = [10, 10, 10, 8, 4, 15]
            self._factor_names = [
                'floor color', 'wall color', 'object color', 'object size',
                'object type', 'azimuth'
            ]
        elif dataset_name == 'mpi3d':
            self._factor_sizes = [6, 6, 2, 3, 3, 40, 40]
            self._factor_names = [
                'color', 'shape', 'size', 'height', 'bg color', 'x-axis',
                'y-axis'
            ]

        self._index_manager = IndexManger(self._factor_sizes)

        def load_data(filename):
            if not os.path.exists(filename):
                print(f"Dataset file {filename} not found.")
                self.download_dataset(filename)
            return np.load(filename,
                           encoding='latin1',
                           allow_pickle=True)['arr_0']

        self._dataset_images = load_data(
            os.path.join(dataset_path, images_filename))
        self._dataset_targets = load_data(
            os.path.join(dataset_path, targets_filename))

    def __len__(self):
        return len(self._dataset_targets)

    @property
    def normalized_targets(self):
        return self._targets / (np.array(self._factor_sizes) - 1)

    @property
    def _targets(self):
        return self._index_manager.index_to_feat

    def __getitem__(self, idx: int, normalize: bool = False):
        image = self._dataset_images[idx]
        targets = self._dataset_targets[idx]
        if normalize:
            targets = (targets - np.array(self._factor_sizes)//2 + ((targets - np.array(self._factor_sizes)//2) >= 0)).astype(np.float32)
            #targets = targets / (np.array(self._factor_sizes) - 1)
        if self.transform is not None:
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)
        return {"X":image.float(), "label":targets, "label_null": torch.tensor(self._factor_sizes)}
    
    @staticmethod
    def download_dataset(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        from urllib import request
        if 'dsprites' in file_path.lower():
            zenodo_code = '4835774'
        elif 'shapes3d' in file_path.lower():
            zenodo_code = '4898937'
        elif 'mpi3d' in file_path.lower():
            zenodo_code = '4899346'
        else:
            raise Exception('datsaet needs to be ')
        
        url = 'https://zenodo.org/record/{}/files/{}?download=1'.\
            format(zenodo_code, os.path.split(file_path)[-1])
        if "full" in file_path:
            
            #download both mode= train and test and then concatenate
            BenchmarkDataset.download_dataset(file_path.replace("full","random"))
            BenchmarkDataset.download_dataset(file_path.replace("full","random").replace("train","test"))
            #concatenate the two files
            data1 = np.load(file_path.replace("full","random"), allow_pickle=True)['arr_0']
            data2 = np.load(file_path.replace("full","random").replace("train","test"), allow_pickle=True)['arr_0']
            data = np.concatenate([data1,data2],axis=0)
            np.savez(file_path, data)
        else:
            print(f'file not found locally, downloading from {url} ...')
            request.urlretrieve(url, file_path, )
    
    def inverse_transform(self,x):
        return x


def load_dataset(dataset_name: str,
                 variant='random',
                 mode='train',
                 dataset_path=None,
                 batch_size=4,
                 shuffle=True,
                 num_workers=0):
    """ Returns a torch dataset loader for the requested split
    Args:
        dataset_name (str): the dataset name, can dbe either '
            shapes3d, 'dsprites' or 'mpi3d'
        variant (str): the split variant, can be either
            'none', 'random', 'composition', 'interpolation', 'extrapolation'
        mode (str): mode, can be either 'train' or 'test', default is 'train'
        dataset_path (str): path to dataset folder
        batch_size (int): batch_size, default is 4
        num_workers (int): num_workers, default = 0
    Returns:
        dataset
    """
    dataset = BenchmarkDataset(dataset_path, dataset_name, variant, mode)

    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)
