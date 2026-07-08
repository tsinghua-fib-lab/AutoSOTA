import numpy as np
import os

import src
from torch.utils.data import Dataset

import torch


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]
    
class TabDiffDataset(Dataset):
    def __init__(self, dataname, data_dir, info, isTrain=True, y_only=False, dequant_dist='none', int_dequant_factor=0.0):
        self.dataname = dataname
        self.data_dir = data_dir
        self.info = info
        self.isTrain = isTrain

        X_num, X_cat, categories, d_numerical, num_inverse, int_inverse, cat_inverse = preprocess(data_dir, y_only, dequant_dist, int_dequant_factor, task_type = info['task_type'], inverse=True)
        categories = np.array(categories)

        X_train_num, _ = X_num
        X_train_cat, _ = X_cat

        X_train_num, X_test_num = X_num
        X_train_cat, X_test_cat = X_cat

        X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
        X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)

        self.X = torch.cat((X_train_num, X_train_cat), dim=1) if isTrain else torch.cat((X_test_num, X_test_cat), dim=1)
        self.num_inverse = num_inverse
        self.int_inverse = int_inverse
        self.cat_inverse = cat_inverse
        self.d_numerical = d_numerical
        self.categories = categories

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]

def preprocess(dataset_path, y_only=False, dequant_dist='none', int_dequant_factor=0.0, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"
    T_dict['dequant_dist'] = dequant_dist
    T_dict['int_dequant_factor'] = int_dequant_factor

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path = dataset_path,
        T = T,
        task_type = task_type,
        change_val = False,
        concat = concat,
        y_only = y_only,
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
        
        categories = src.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)


        if inverse:
            num_inverse = dataset.num_transform.inverse_transform if dataset.num_transform is not None else lambda x: x
            int_inverse = dataset.int_transform.inverse_transform if dataset.int_transform is not None else lambda x: x
            cat_inverse = dataset.cat_transform.inverse_transform if dataset.cat_transform is not None else lambda x: x

            return X_num, X_cat, categories, d_numerical, num_inverse, int_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type,
    change_val: bool,
    concat = True,
    y_only = False,
):

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy'))  else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if y_only:
                X_num_t = X_num_t[:, :0]
                X_cat_t = X_cat_t[:, :0]
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t  
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if y_only:
                X_num_t = X_num_t[:, :0]
                X_cat_t = X_cat_t[:, :0]
            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))
    int_col_idx_wrt_num = info['int_col_idx_wrt_num']

    if y_only:
        int_col_idx_wrt_num = []
    D = src.Dataset(
        X_num,
        X_cat,
        y,
        int_col_idx_wrt_num,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes')
    )

    if change_val:
        D = src.change_val(D)

    return src.transform_dataset(D, T, None)