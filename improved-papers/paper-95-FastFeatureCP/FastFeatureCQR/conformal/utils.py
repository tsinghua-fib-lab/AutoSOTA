import sys
import os
import numpy as np
import random
import torch
import torch.nn as nn


def seed_torch(seed, verbose=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False  # decrease efficiency

    if verbose:
        print("==> Set seed to {:}".format(seed))


def mse(inputs, targets):
    return ((inputs - targets) ** 2).mean()


def default_loss(inputs, targets):
    return torch.norm(inputs - targets) ** 2


class WeightedMSE(nn.Module):
    def __init__(self, out_shape, weight=None):
        super(WeightedMSE, self).__init__()
        self.out_shape = out_shape
        if weight is not None:
            self.weight = weight / weight.sum()
        else:
            self.weight = None

    def forward(self, inputs, targets):
        if self.weight is not None:
            batch_size = inputs.shape[0]
            mse = ((inputs.view(batch_size, -1, *self.out_shape) - targets.view(batch_size, -1, *self.out_shape)) ** 2).mean(dim=(0, 2, 3))  # (num_classes,)
            mse = (self.weight * mse).sum()
        else:
            mse = ((inputs - targets) ** 2).mean()
        return mse


def compute_coverage_len(y_test, y_lower, y_upper):
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage, avg_length


def compute_coverage(y_test, y_lower, y_upper, significance, class_weight=None, name="", verbose=True):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length
    """
    if y_test.ndim == 1 and y_lower.ndim == 2 and y_upper.ndim == 2:
        y_test = y_test[:, None]

    in_the_range = np.sum(np.all((y_test >= y_lower) & (y_test <= y_upper), axis=1))
    coverage = in_the_range / len(y_test) * 100
    avg_length = abs(np.mean(y_lower - y_upper))

    if class_weight is not None:
        class_weight = class_weight.cpu().numpy()[None, :, None]
        class_num = class_weight.shape[1]
        batch_size = y_test.shape[0]
        y_weight = y_test.reshape(batch_size, class_num, -1)
        y_weight = np.abs((y_weight - 0.5) * 2)
        weight = y_weight * class_weight
        weight /= weight.sum()

        abs_length = np.abs(y_lower - y_upper).reshape(batch_size, class_num, -1)
        weighted_avg_length = np.sum(weight * abs_length)
    else:
        y_weight = y_test
        weight = y_weight / y_weight.sum()
        weighted_avg_length = np.sum(weight * np.abs(y_lower - y_upper))

    if verbose:
        print("%s: Percentage in the range (expecting %.2f): %f  " % (name, 100 - significance * 100, coverage),
              "Average length: %f" % (avg_length),
              "Weighted Avg length: %f" % (weighted_avg_length))
    sys.stdout.flush()
    return coverage, avg_length, weighted_avg_length


def FFCP_compute_coverage(y_test, y_lower, y_upper, significance, name="", verbose=False):
    """ Compute average coverage and length, and print results

    Parameterss
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)
    verbose: bool, whether print the results
    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length
    weighted_avg_length : float, weighted average length
    """
    if y_test.ndim == 1 and y_lower.ndim == 2 and y_upper.ndim == 2:
        y_test = y_test[:, None]

    in_the_range = np.sum(np.all((y_test >= y_lower) & (y_test <= y_upper), axis=1))

    coverage = in_the_range / len(y_test) * 100

    avg_length = abs(np.mean(y_lower - y_upper))

    if verbose:
        print("%s: Percentage in the range (expecting %.2f): %f  " % (name, 100 - significance * 100, coverage),
              "Average length: %f" % (avg_length))
        sys.stdout.flush()
    return coverage, avg_length