import os
import sys
import time
import copy
import pickle
import hashlib
import operator
import numpy as np
import scipy as sp
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from numbers import Number
from collections import OrderedDict, defaultdict
from sklearn.model_selection import train_test_split

sys.dont_write_bytecode = True


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def split_train_test_datasets(all_dataset, train_size=0.5, val_size=0., test_size=0.5):
    num_datasets = len(all_dataset)
    domain_idx = np.arange(num_datasets)
    train_domain_idx, test_domain_idx = train_test_split(domain_idx, train_size=train_size,
                                                         test_size=(val_size + test_size), shuffle=False)
    val_domain_idx, test_domain_idx = train_test_split(test_domain_idx, train_size=val_size, test_size=test_size,
                                                       shuffle=False)

    train_datasets = copy.deepcopy(all_dataset)
    train_datasets.datasets = train_datasets.datasets[train_domain_idx[0]:train_domain_idx[-1] + 1]
    train_datasets.Environments = train_datasets.Environments[train_domain_idx[0]:train_domain_idx[-1] + 1]

    val_datasets = copy.deepcopy(all_dataset)
    val_datasets.datasets = val_datasets.datasets[val_domain_idx[0]:val_domain_idx[-1] + 1]
    val_datasets.Environments = val_datasets.Environments[val_domain_idx[0]:val_domain_idx[-1] + 1]

    test_datasets = copy.deepcopy(all_dataset)
    test_datasets.datasets = test_datasets.datasets[test_domain_idx[0]:test_domain_idx[-1] + 1]
    test_datasets.Environments = test_datasets.Environments[test_domain_idx[0]:test_domain_idx[-1] + 1]

    return train_datasets, val_datasets, test_datasets


def extract_by_name(state_dict, start_with: str):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(start_with):
            name = k[len(start_with):]  # remove `module.`
            new_state_dict[name] = v
    return new_state_dict


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def batch_shuffle(data, label, in_place=False):
    if not in_place:
        data, label = data.clone(), label.clone()
    batch_size = data.size(0)
    idx_shuffle = torch.randperm(batch_size).to(data.device)
    return data[idx_shuffle], label[idx_shuffle]


def format_time():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy(%) over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print('prediction:', pred)
        # print('target:', target)
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mean_var(data):
    a = [1.0 * np.array(data[i].cpu()) for i in range(len(data))]
    n = len(a)
    mean = np.mean(a)
    err = np.std(a / np.sqrt(n))
    return mean, err


def one_hot(indices, depth, device=None):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    if device is None:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    else:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).to(device)

    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


class MyDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(MyDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(MyDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def apply_dataparallel(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            setattr(model, name, MyDataParallel(module).cuda())
