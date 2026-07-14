"""
Original code from: https://github.com/gengchenmai/csp

Original source:
Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations."
Proceedings of the 40th International Conference on Machine Learning (ICML), 2023.
"""

import json
import os
import pickle
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch

from srl.baselines.CSP.paths import *


def make_model_res_file(
    data_dir,
    dataset="inat2018",
    eval_split="val",
    res_type="preds",
    sample_ratio=None,
):
    assert eval_split in ["train", "val", "test"]
    if res_type == "preds_sparse":
        if sample_ratio is None:
            return f"{data_dir}/{dataset}_{eval_split}_preds_sparse.npz"
        else:
            return f"{data_dir}/{dataset}_{eval_split}_preds_sparse_ratio{sample_ratio:.3f}.npz"
    else:
        assert res_type in ["net_feats", "labels", "ids", "preds"]
        if sample_ratio is None:
            return f"{data_dir}/{dataset}_{eval_split}_{res_type}.npy"
        else:
            return f"{data_dir}/{dataset}_{eval_split}_{res_type}_ratio{sample_ratio:.3f}.npy"


def get_train_sample_ratio_tag(train_sample_ratio, train_sample_method):
    if train_sample_method == "stratified-fix":
        return f"ratio{train_sample_ratio:.3f}"
    else:
        train_sample_ratio_tag = (
            "ratio{train_sample_ratio:.3f}-{train_sample_method:s}".format(
                train_sample_ratio=train_sample_ratio,
                train_sample_method=train_sample_method,
            )
        )
        return train_sample_ratio_tag


def get_classes_sample_idxs(classes, sample_ratio):
    """
    Given a list of classes labels and sample ratio,
    we get samples whose number of samples in each class in propotional of the total number of samples with this class
    Args:
        classes: np.array(int), shape (num_samples, ), a list of class labels
        sample_ratio: float, the sample ratio
    Return:
        class_sample_idxs:, np.array(int), a list of idx of the samples in classes
    """
    un_classes, un_counts = np.unique(classes, return_counts=True)
    class_dict = {}
    for ii, cc in enumerate(classes):
        if cc not in class_dict:
            class_dict[cc] = []
        class_dict[cc].append(ii)

    # sample_cnts = np.round_(un_counts * sample_ratio, decimals=0, out=None).astype(np.int)
    sample_cnts = []
    for cc, cnt in enumerate(un_counts):
        sample_cnt = cnt * sample_ratio
        if sample_cnt < 1:
            sample_cnt = 1
        else:
            sample_cnt = np.round_(sample_cnt)
        sample_cnts.append(sample_cnt)

    sample_cnts = np.array(sample_cnts).astype(np.int)

    class_sample_dict = {}
    class_sample_idxs = []
    for idx, cc in enumerate(un_classes):
        sample_size = sample_cnts[idx]
        sample_idxs = np.random.choice(
            class_dict[cc], size=sample_size, replace=False
        )
        class_sample_idxs += list(sample_idxs)
        class_sample_dict[cc] = sample_idxs

    class_sample_idxs = np.sort(np.array(class_sample_idxs))
    return class_sample_idxs, class_sample_dict


def get_sample_idx_file_path(
    dataset,
    meta_type,
    data_split="train",
    sample_ratio=0.1,
    sample_method="stratified-fix",
):

    sample_ratio_tag = get_train_sample_ratio_tag(sample_ratio, sample_method)

    data_dir = get_paths(f"{dataset}_data_dir")
    sample_idx_dir = f"{data_dir}/sample_idx/"
    if dataset == "birdsnap":
        sample_idx_dir = f"{sample_idx_dir}/{meta_type}/"
    if not os.path.isdir(sample_idx_dir):
        os.makedirs(sample_idx_dir)

    sample_idx_filepath = f"{sample_idx_dir:s}/{data_split:s}_sample_{sample_ratio_tag:s}.npy".format(
        sample_idx_dir=sample_idx_dir,
        data_split=data_split,
        sample_ratio_tag=sample_ratio_tag,
    )
    return sample_idx_filepath


def coord_normalize(coords, extent=(-180, 180, -90, 90), do_global=False):
    """
    Given a list of coords (X, Y), normalize them to [-1, 1]
    Args:
        coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        extent: (x_min, x_max, y_min, y_max)
        do_global:  True - lon/180 and lat/90
                    False - min-max normalize based on extent
    Return:
        coords_mat: np tensor shape (batch_size, num_context_pt, coord_dim)
    """
    if type(coords) == list:
        coords_mat = np.asarray(coords).astype(np.float32)
    elif type(coords) == np.ndarray:
        coords_mat = coords

    if do_global:
        coords_mat[:, :, 0] /= 180.0
        coords_mat[:, :, 1] /= 90.0
    else:
        # x => [0,1]  min_max normalize
        x = (coords_mat[:, :, 0] - extent[0]) * 1.0 / (extent[1] - extent[0])
        # x => [-1,1]
        coords_mat[:, :, 0] = (x * 2) - 1

        # y => [0,1]  min_max normalize
        y = (coords_mat[:, :, 1] - extent[2]) * 1.0 / (extent[3] - extent[2])
        # x => [-1,1]
        coords_mat[:, :, 1] = (y * 2) - 1

    return coords_mat


def json_load(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def json_dump(data, filepath, pretty_format=True):
    with open(filepath, "w") as fw:
        if pretty_format:
            json.dump(data, fw, indent=2, sort_keys=True)
        else:
            json.dump(data, fw)


def pickle_dump(obj, pickle_filepath):
    with open(pickle_filepath, "wb") as f:
        pickle.dump(obj, f, protocol=2)


def pickle_load(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        obj = pickle.load(f)
    return obj
