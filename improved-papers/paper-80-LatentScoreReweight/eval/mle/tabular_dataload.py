# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import torch
import numpy as np
import pandas as pd
from tabular_transformer import GeneralTransformer
import json
import logging
import os

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'tabular_datasets')

def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    
    if loader == np.load:
        return loader(local_path, allow_pickle=True)
    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()

    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)

    return categorical_columns


def load_data(name):
    data_dir = f'data/{name}'
    info_path = f'{data_dir}/info.json'

    train = pd.read_csv(f'{data_dir}/train.csv').to_numpy()
    test = pd.read_csv(f'{data_dir}/test.csv').to_numpy()

    with open(f'{data_dir}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    num_cols = info['num_col_idx']
    cat_cols = info['cat_col_idx']
    target_cols = info['target_col_idx']

    if task_type != 'regression':
        cat_cols = cat_cols + target_cols

    return train, test, (cat_cols, info)
        

def get_dataset(FLAGS, evaluation=False):

    batch_size = FLAGS.training_batch_size if not evaluation else FLAGS.eval_batch_size

    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                            f'the number of devices ({torch.cuda.device_count()})')


    # Create dataset builders for tabular data.
    train, test, cols = load_data(FLAGS.dataname)
    cols_idx = list(np.arange(train.shape[1]))
    dis_idx = cols[0]
    con_idx = [x for x in cols_idx if x not in dis_idx]

    #split continuous and categorical
    train_con = train[:,con_idx]
    train_dis = train[:,dis_idx]

    #new index
    cat_idx_ = list(np.arange(train_dis.shape[1]))[:len(cols[0])]

    transformer_con = GeneralTransformer()
    transformer_dis = GeneralTransformer()

    transformer_con.fit(train_con, [])
    transformer_dis.fit(train_dis, cat_idx_)

    train_con_data = transformer_con.transform(train_con)
    train_dis_data = transformer_dis.transform(train_dis)


    return train, train_con_data, train_dis_data, test, (transformer_con, transformer_dis, cols[1]), con_idx, dis_idx
        