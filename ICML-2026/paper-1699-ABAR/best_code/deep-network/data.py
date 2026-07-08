# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reading data."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification import mnist as _emnist_module
from utils import EasyDict

# Patch the broken NIST URL for EMNIST downloads.
_emnist_module.EMNIST.URL = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'

DATA_DIR = os.path.join(os.environ['ML_DATA'], 'TFDS')


def get_data(data_name: str):
    assert data_name in ['mnist', 'emnist_merge', 'cifar10']
    if data_name.startswith('emnist'):
        data_name = 'emnist/by' + data_name[7:]
    data, info = tfds.load(name=data_name, batch_size=-1, data_dir=DATA_DIR, with_info=True)
    data = tfds.as_numpy(data)
    train = EasyDict(image=data['train']['image'].transpose(0, 3, 1, 2) / 127.5 - 1, label=data['train']['label'])
    test = EasyDict(image=data['test']['image'].transpose(0, 3, 1, 2) / 127.5 - 1, label=data['test']['label'])
    ntrain = train['image'].shape[0]
    nclass = info.features['label'].num_classes
    return train, test, ntrain, nclass
