"""
Utility classes and functions for 
pytorch neural networks.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    Dataset, 
    DataLoader
)
from torchmetrics.regression import (
    MeanSquaredError,
    R2Score
)

from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Callable,
    Any
)

import os
import warnings
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import pickle
import json
from collections.abc import Iterable

# Note: Activation function mappings have been moved to models/activation_maps.py
# Import them from there as needed

try:
    import torch_scatter
    _TORCH_SCATTER_AVAILABLE = True
except:
    torch_scatter = None
    _TORCH_SCATTER_AVAILABLE = False


def merge_dicts_with_defaults(
        user_dict: Dict[str, Any], 
        default_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge a user-provided dictionary with a default dictionary, recursively 
    handling nested dictionaries.
    
    Args:
        user_dict: Dictionary provided by the user, may be partial
        default_dict: Dictionary containing default values
        
    Returns:
        Merged dictionary with user values taking precedence over defaults
    """
    result = default_dict.copy()
    for key, value in user_dict.items():
        if key in result and isinstance(result[key], dict) \
        and isinstance(value, dict):
            result[key] = merge_dicts_with_defaults(value, result[key])
        else:
            result[key] = value
    return result


def count_parameters(
    model: torch.nn.Module | Any,
    trainable_only: bool = True,
) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model: Object exposing a ``parameters`` iterator (e.g., ``nn.Module``).
        trainable_only: If True, only parameters with ``requires_grad`` are counted.

    Returns:
        Total parameter count as an integer.

    Raises:
        ValueError: If model is None.
        TypeError: If the model does not expose a parameters iterator.
    """
    if model is None:
        raise ValueError("model must not be None.")

    params_iter = None
    if hasattr(model, "parameters") and callable(getattr(model, "parameters")):
        params_iter = model.parameters()
    elif hasattr(model, "named_parameters") and callable(getattr(model, "named_parameters")):
        params_iter = (param for _, param in model.named_parameters())

    if params_iter is None or not isinstance(params_iter, Iterable):
        raise TypeError("model does not expose a parameters() iterator.")

    total = 0
    for param in params_iter:
        if not isinstance(param, torch.Tensor):
            continue
        if trainable_only and not param.requires_grad:
            continue
        total += int(param.numel())
    return total


def raise_if_nonfinite_array(
    arr: np.ndarray,
    *,
    name: str,
) -> None:
    """
    Raise an exception if a numpy array contains NaN or Inf.
    """
    if not np.isfinite(arr).all():
        raise Exception(f"Non-finite value detected in {name}.")


def raise_if_nonfinite_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    """
    Raise an exception if a tensor contains NaN or Inf.
    """
    if not torch.isfinite(tensor).all():
        raise Exception(f"Non-finite value detected in {name}.")


def minmax_scale_tensor(
    v: torch.Tensor,
    min_v: Optional[torch.Tensor | float] = None, 
    max_v: Optional[torch.Tensor | float] = None,
    above_zero_floor: Optional[float] = None, 
    div_0_thresh: float = 1e-10,
    verbosity: int = 0
) -> Optional[torch.Tensor]:
    """
    Min-max scales a tensor onto the interval
    [0, 1] or [above_zero_floor, 1].
    
    Allows min and max values to be passed
    as optional arguments, e.g. in case they
    come from a sorted array (and are simply at
    indices 0 and -1).

    Args:
        v: tensor.
        min_v: optional minimum value in v, if already
            known upstream, to save computation.
        max_v: optional maximum value in v, if already
            known upstream, to save computation.
        above_zero_floor: optional value > 0,
            to map v onto interval [above_zero_floor, 1]
            instead of [0, 1], e.g. if log is going
            to be taken of output.
        div_0_thresh: if the range between max and min
            values is less than this positive threshold,
            we assume they are the same, and v cannot be 
            rescaled (because there's no variance in its
            elements).
        verbosity: integer value controlling the volume
            of print output as this function runs.
    Returns:
        Tensor of rescaled values from v; or None.
    """
    if v is None:
        return None
    else:
        try:
            if min_v is None:
                min_v = torch.min(v)
            if max_v is None:
                max_v = torch.max(v)
                
            if above_zero_floor is not None:
                if isinstance(above_zero_floor, str): 
                    # above_zero_floor is an error string: an algorithm used
                    # to compute a floor likely found an edge case,
                    # such as no variance in v
                    if above_zero_floor == 'error':
                        return None
                    else:
                        raise Exception(
                            f"Unrecognized error key for 'above_zero_floor'."
                        )
                elif above_zero_floor <= 0:
                    raise Exception(f"above_zero_floor = {above_zero_floor:.4f} <= 0")
                
                # adjust min_v such that the new floor will be above zero
                min_v -= above_zero_floor

            # calc the range
            vals_range = (max_v - min_v)

            # make sure the range is > 0
            if torch.abs(vals_range) < div_0_thresh:
                return None
            # if the range is > 0, min-max scale onto [above_zero_floor, 1] interval
            else:
                return (v - min_v) / vals_range
        except:
            print('Error in `nnu.minmax_scale_tensor`.')
            print(f"\tv = {v}, min_v = {min_v}, max_v = {max_v}")
            return None


def norm_tensor_to_prob_mass(
    v: Optional[torch.Tensor],
    min_v: Optional[torch.Tensor | float] = None, 
    max_v: Optional[torch.Tensor | float] = None,
    above_zero_floor: Optional[float] = None
) -> Optional[torch.Tensor]:
    r"""
    Uses 'minmax_scale_tensor' to obtain v', then 
    divides v' by the sum of all elements (i.e. 
    $\ell^1$-normalization) to obtain a probability 
    vector (a vector on [0, 1] where the sum of all
    elements is 1.0).
    
    Args:
        v: tensor or None, if there's an error upstream.
        min_v: optional minimum value in v, if already
            known upstream, to save computation.
        max_v: optional maximum value in v, if already
            known upstream, to save computation.
        above_zero_floor: optional value > 0,
            to map v onto interval [above_zero_floor, 1]
            instead of [0, 1], e.g. if log is going
            to be taken of output.
    Returns:
        1-d tensor of v transformed into a probability 
        vector; or None.
    """
    if v is None:
        return None
    else:
        try:
            # first map all values onto [0, 1]
            v = minmax_scale_tensor(v, min_v, max_v, above_zero_floor)
            if v is not None:
                # lastly, rescale all values so their sum = 1
                sum_v = v.sum()
                return v / sum_v
            else:
                print("[norm_tensor_to_prob_mass] Warning: minmax_scale_tensor returned None (likely zero variance)."
                      f"Returning None for this channel."
                )
                return None
        except:
            print('Error in `nnu.norm_tensor_to_prob_mass`.')
            print(f"\tv = {v}")
            return None


def get_mid_btw_min_and_2nd_low_vector_vals(
    v: torch.Tensor,
    no_var_thresh: float = 1e-10,
) -> Optional[float] | str:
    r"""
    Computes the linear midpoint between the 
    minimum and second-lowest values in tensor.

    Args:
        v: tensor.
        no_var_thresh: if the difference between
        the max and min values in v is less than 
        this positive threshold, then all values
        in v appear to be equal, and no 2nd lowest
        value can be found; None will be returned
        instead.
    Returns:
        The float value of $\frac{1}{2}(min(v) +
        \text{2nd-lowest}(v))$, None, or string
        holding error key.
    """
    if v is not None:
        i_2nd_low = None
        try:
            if v.ndim > 1:
                v = v.ravel()
            v = v.sort().values
            min_v = v[0]
            max_v = v[-1]
            # if the min and max values in v are the same, 
            # return None
            if (max_v - min_v) < no_var_thresh:
                return 'error'
            else:
                # there could be ties for lowest value: get
                # first index of second-lowest value in v
                i_2nd_low = torch.argwhere(v > min_v)[0][0]
                return (min_v + v[i_2nd_low]) / 2.
        except:
            print('Error in `nnu.get_mid_btw_min_and_2nd_low_vector_vals`.')
            print(f"\tv = {v}; i_2nd_low = {i_2nd_low}")
            return None
    else:
        return None


def get_stat_moments(
    x: torch.Tensor,
    moments: Tuple[int] = (1, 2, 3),
    nan_replace_value: Optional[float] = None,
    batch_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes statistical moments of a tensor x along the first dimension (dim=0), or per group if batch_index is provided. Handles NaNs by replacing them with nan_replace_value (or zero if not set), then uses standard .mean(), .var(), .std() methods.
    Args:
        x: tensor of shape (N, ...), where moments are computed along N.
        moments: tuple of integers representing the moments to calculate (1 = mean, 2 = variance, 3 = skewness, 4 = kurtosis).
        nan_replace_value: value to replace NaNs in the output tensor.
        batch_index: optional 1D tensor of shape (N,) with group indices for each row in x.
    Returns:
        If batch_index is None: Tensor of shape (num_moments, ...), where ... are the remaining dimensions of x.
        If batch_index is provided: Tensor of shape (num_moments, num_groups, ...), where num_groups = batch_index.max() + 1
    """
    # Replace NaNs ONCE at the start
    if nan_replace_value is None:
        nan_replace_value = 0.0
    x_clean = torch.nan_to_num(x, nan=nan_replace_value)

    # --- Helper functions for group-wise stats (no torch_scatter) ---
    def groupwise_mean(x_group):
        return x_group.mean(dim=0)
    def groupwise_var(x_group):
        return x_group.var(dim=0, unbiased=True)
    def groupwise_std(x_group):
        return x_group.std(dim=0, unbiased=True)
    def groupwise_z(x_group, mean, std):
        return (x_group - mean) / (std + 1e-8)
    def groupwise_moment(x_group, power):
        mean = groupwise_mean(x_group)
        std = groupwise_std(x_group)
        z = groupwise_z(x_group, mean, std)
        return (z ** power).mean(dim=0)

    # --- Helper functions for torch_scatter grouped stats ---
    def scatter_groupwise_mean(x_flat, batch_index, count_per_group):
        sum_per_group = torch_scatter.scatter_sum(x_flat, batch_index, dim=0)
        mean = sum_per_group / count_per_group.clamp(min=1)
        mean[count_per_group == 0] = nan_replace_value
        return mean
    def scatter_groupwise_var(x_flat, mean, batch_index, count_per_group):
        x_centered = x_flat - mean[batch_index]
        sq = x_centered ** 2
        sq_sum = torch_scatter.scatter_sum(sq, batch_index, dim=0)
        var = sq_sum / (count_per_group - 1).clamp(min=1)
        var[count_per_group <= 1] = nan_replace_value
        return var
    def scatter_groupwise_moment(x_flat, mean, var, batch_index, count_per_group, power):
        std = torch.sqrt(var + 1e-8)
        z = (x_flat - mean[batch_index]) / (std[batch_index] + 1e-8)
        z_pow_sum = torch_scatter.scatter_sum(z ** power, batch_index, dim=0)
        moment = z_pow_sum / count_per_group.clamp(min=1)
        moment[count_per_group == 0] = nan_replace_value
        return moment

    # If batch_index is provided and torch_scatter is available, compute grouped moments efficiently
    if batch_index is not None and _TORCH_SCATTER_AVAILABLE:
        orig_shape = x_clean.shape[1:]
        x_flat = x_clean.view(x_clean.shape[0], -1)  # (N, D)
        mask = torch.ones_like(x_flat, dtype=torch.bool)  # All values are now valid
        count_per_group = torch_scatter.scatter_sum(mask.float(), batch_index, dim=0)  # (num_groups, D)
        stats = []
        mean = None
        var = None
        for m in moments:
            if m == 1:
                mean = scatter_groupwise_mean(x_flat, batch_index, count_per_group)
                stats.append(mean)
            elif m == 2:
                if mean is None:
                    mean = scatter_groupwise_mean(x_flat, batch_index, count_per_group)
                var = scatter_groupwise_var(x_flat, mean, batch_index, count_per_group)
                stats.append(var)
            elif m == 3:
                if mean is None:
                    mean = scatter_groupwise_mean(x_flat, batch_index, count_per_group)
                if var is None:
                    var = scatter_groupwise_var(x_flat, mean, batch_index, count_per_group)
                skew = scatter_groupwise_moment(x_flat, mean, var, batch_index, count_per_group, 3)
                stats.append(skew)
            elif m == 4:
                if mean is None:
                    mean = scatter_groupwise_mean(x_flat, batch_index, count_per_group)
                if var is None:
                    var = scatter_groupwise_var(x_flat, mean, batch_index, count_per_group)
                kurt = scatter_groupwise_moment(x_flat, mean, var, batch_index, count_per_group, 4)
                stats.append(kurt)
            else:
                raise NotImplementedError(f"{m}th statistical moment not implemented.")
        out = torch.stack(stats, dim=0)
        out = out.view(out.shape[0], out.shape[1], *orig_shape)
        return out
    # If batch_index is provided but torch_scatter is NOT available, use slow fallback
    elif batch_index is not None and not _TORCH_SCATTER_AVAILABLE:
        unique_groups = batch_index.unique(sorted=True)
        stats = []
        for m in moments:
            group_stats = []
            for group in unique_groups:
                mask = (batch_index == group)
                x_group = x_clean[mask]
                if m == 1:
                    val = groupwise_mean(x_group)
                elif m == 2:
                    val = groupwise_var(x_group)
                elif m == 3:
                    val = groupwise_moment(x_group, 3)
                elif m == 4:
                    val = groupwise_moment(x_group, 4)
                else:
                    raise NotImplementedError(f"{m}th statistical moment not implemented.")
                group_stats.append(val)
            stats.append(torch.stack(group_stats, dim=0))
        out = torch.stack(stats, dim=0)
        return out
    # If no batch_index is provided, compute moments ungrouped
    else:
        stats = []
        mean = None
        var = None
        for m in moments:
            if m == 1:
                mean = x_clean.mean(dim=0)
                stats.append(mean)
            elif m == 2:
                if mean is None:
                    mean = x_clean.mean(dim=0)
                var = x_clean.var(dim=0, unbiased=True)
                stats.append(var)
            elif m == 3:
                if mean is None:
                    mean = x_clean.mean(dim=0)
                if var is None:
                    var = x_clean.var(dim=0, unbiased=True)
                std = x_clean.std(dim=0, unbiased=True)
                z = (x_clean - mean) / (std + 1e-8)
                skewness = (z ** 3).mean(dim=0)
                stats.append(skewness)
            elif m == 4:
                if mean is None:
                    mean = x_clean.mean(dim=0)
                if var is None:
                    var = x_clean.var(dim=0, unbiased=True)
                std = x_clean.std(dim=0, unbiased=True)
                z = (x_clean - mean) / (std + 1e-8)
                kurtosis = (z ** 4).mean(dim=0)
                stats.append(kurtosis)
            else:
                raise NotImplementedError(f"{m}th statistical moment not implemented.")
        out = torch.stack(stats, dim=0)
        return out


def get_inv_class_wts(
    train_labels: List[Any]
) -> List[float]:
    """
    Calculates inverse class weights from a set of training
    labels. Can be used with torch.nn.CrossEntropyLoss(weight=.),
    for example, to help with training a class-imbalanced
    dataset.

    Args:
        train_labels: a list of labels (ints, floats, strings,
            etc.) for a training set.
    Returns:
        List of float class weights for 'balanced' 
        re-weighting use in, e.g., torch.nn.CrossEntropyLoss.
    """
    from collections import Counter
    cts = Counter(sorted(train_labels))
    inv_wts = [
        1 - (ct / len(train_labels)) \
        for (label, ct) in cts.items()
    ]
    return inv_wts
    

def log_parameter_grads_weights(
    path: str,
    model: torch.nn.Module, 
    grad_track_param_names: Tuple[str],
    epoch_i: int, 
    batch_i: int,
    save_grads: bool = True,
    save_weights: bool = True,
    verbosity: int = 0
) -> None:
    """
    Appends the flattened gradient values for 
    parameters of interest in a model to rows
    of corresponding CSVs (saved in the model save 
    directory), with the first 2 columns saving
    the epoch and batch numbers.

    Args:
        path: path to the directory in which to save
        model: a torch.nn.Module.
        epoch_i: index of the epoch for which
            gradients and weights are being logged.
        batch_i: index of the batch for which
            gradients and weights are being logged.
        save_grads: boolean whether to log gradients.
        save_weights: boolean whether to log weights.
        verbosity: integer value controlling the
            volume of console print output as this
            function runs.
    Returns:
        None; saves csv file(s).
    """
    for name, param in model.named_parameters():
        for tracked_param_name in grad_track_param_names:
            if tracked_param_name in name:
                if save_grads:
                    # note pytorch appends '.1' etc. to repeated parameter names
                    filename = f'grads_{name}'.replace('.', '_') + '.csv'
                    if param.grad is not None:
                        grads = torch.flatten(param.grad).tolist()
                        out = [epoch_i, batch_i] + grads
                        with open(f'{path}/{filename}', 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(out)
                    else:
                        if verbosity > 0:
                            print(
                                f'Warning: {name} grad was None;',
                                f'is_leaf = {param.is_leaf}'
                            )
                if save_weights:
                    filename = f'weights_{name}'.replace('.', '_') + '.csv'
                    weights = torch.flatten(param).tolist()
                    out = [epoch_i, batch_i] + weights
                    filepath = os.path.join(path, filename)
                    with open(filepath, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(out)


def get_trained_model_preds(
    trained_model: torch.nn.Module,
    dataloaders_dict: Dict[str, dict],
    set: str = 'test',
    return_on_cpu: bool = True,
    verbosity: int = 0
) -> torch.Tensor:
    """
    Runs trained_model.forward on batches of 
    a set (train/valid/test), and collects
    model predictions on the set into one tensor,
    optionally moved to cpu.

    Args:
        trained_model: a torch.nn.Module with
            trained weights.
        dataloaders_dict: a dictionary holding
            torch.utils.data.DataLoader objects
            which themselves contain model inputs
            in dictionaries, keyed by set ('train', 
            'valid', 'test').
        set: the string key for the set to get
            model predictions on.
        return_on_cpu: boolean whether to return
            model predictions on the CPU.
        verbosity: integer value controlling the
            volume of console print output as this
            function runs.
    Returns:
        A tensor of model predictions for the 
        specified set.
    """
    test_preds = []
    device = next(trained_model.parameters()).device
    trained_model.eval()
    with torch.no_grad():
        for input_dict in dataloaders_dict[set]:
            # print('input_dict[\'x\'].shape', input_dict['x'].shape)
            # for i, param in enumerate(trained_model.parameters()):
            #     print(f'layer {i + 1} param shape: {param.data.shape}')
            # print(input_dict['x'], '\n')
            # batch_x = input_dict['x'].to(device)
            # print(f'batch_x.shape: {batch_x.shape}')
            trained_model_output_dict = trained_model(input_dict)
            batch_preds = trained_model_output_dict['preds']
            test_preds.extend(batch_preds)
    test_preds_tensor = torch.stack(test_preds).squeeze()
    if return_on_cpu:
        test_preds_tensor = test_preds_tensor.cpu()

    # check
    if verbosity > 0:
        print('test_preds.shape', test_preds.shape)
        print(test_preds, '\n')
    return test_preds_tensor


def get_target_tensors(
    trained_model: torch.nn.Module,
    datasets_dict: Dict[str, dict],
    sets: Tuple[str] = ('test', ),
    target_name: str = 'y'
) -> Tuple[torch.Tensor]:
    """
    Extracts targets from datasets_dict for
    train/valid/test sets, and if more
    than one set was requested, collects into
    a tuple of 1d/vector tensors.

    Args:
        trained_model: a torch.nn.Module with
            trained weights.
        datasets_dict: a dictionary of 
            torch.utils.data.Dataset (or subclass)
            objects, which themselves contain model inputs
            in dictionaries, keyed by set ('train', 'valid',
            'test')
        sets: tuple of string keys for which to
            obtain targets collected in tensors.
        target_name: string key value for the target,
            as keyed in the dictionaries in datasets_dict.
    Returns:
        Tuple of tensors containing set targets.
    """
    target_tensors = [
        torch.stack([
            dict['target'][target_name] \
            for dict in datasets_dict[set]
        ]) for set in sets
    ]
    if len(sets) > 1:
        target_tensors = tuple(target_tensors)
    return target_tensors


def regressor_preds_plots(
    trained_model: torch.nn.Module,
    datasets_dict: Dict[str, dict],
    dataloaders_dict: Dict[str, DataLoader],
    target_name: str = 'y',
    # device: str = 'mps:0',
    train_targets_bins: Optional[int] = None,
    test_preds_bins: Optional[int] = None,
    fig_size: Tuple[float, float] = (6., 4.)
) -> None:
    """
    Prints useful analytic plots for the predictions
    of a regressor model versus a mean model.

    Args:
        trained_model: a torch.nn.Module with
            trained weights.
        datasets_dict: a dictionary of 
            torch.utils.data.Dataset (or subclass)
            objects, which themselves contain model inputs
            in dictionaries, keyed by set ('train', 'valid',
            'test')
        dataloaders_dict: a dictionary holding
            torch.utils.data.DataLoader objects
            which themselves contain model inputs
            in dictionaries, keyed by set ('train', 
            'valid', 'test').
        target_name: string key value for the target,
            as keyed in the dictionaries in datasets_dict.
        train_targets_bins: optional int arg to pass 
            to 'bins' arg in plt.hist, when making a
            histogram of train set targets.
        test_preds_bins: optional int arg to pass 
            to 'bins' arg in plt.hist, when making a
            histogram of test set predictions.
        fig_size: 2-tuple of floats to pass to plt
            to set output figure size.
    Returns:
        None; prints plots instead (e.g. in a Jupyter 
        notebook).
    """
    # extract train and test targets from datasets_dict
    # train_targets = torch.stack([
    #     dict['target'][target_name] \
    #     for dict in datasets_dict['train']
    # ])
    
    # test_targets = torch.stack([
    #     dict['target'][target_name] \
    #     for dict in datasets_dict['test']
    # ])

    train_targets, test_targets = get_target_tensors(
        trained_model,
        datasets_dict,
        ('train', 'test'),
        target_name
    )
    # print('test_targets.shape', test_targets.shape)
    # print(test_targets, '\n')

    # get trained model's test set predictions
    test_preds = get_trained_model_preds(
        trained_model,
        dataloaders_dict
    )

    # compute mean model's regression metrics
    train_mean = torch.mean(train_targets)
    mean_model_preds = train_mean.repeat(test_targets.shape[0])
    # print('mean_model_preds.shape', mean_model_preds.shape)
    # print(mean_model_preds, '\n')
    mse = MeanSquaredError()
    R2 = R2Score()
    mse.update(mean_model_preds, test_targets)
    R2.update(mean_model_preds, test_targets)
    mean_model_mse = mse.compute()
    mean_model_R2 = R2.compute()
    mse.reset()
    R2.reset()
    
    # trained model's test set regression metrics
    MSE_test = MeanSquaredError()
    MSE_test.update(test_preds, test_targets)
    trained_model_mse = MSE_test.compute()
    MSE_test.reset()
    
    R2_test = R2Score()
    R2_test.update(test_preds, test_targets)
    trained_model_R2 = R2_test.compute()
    R2_test.reset()

    # set plot size
    plt.rcParams["figure.figsize"] = fig_size
    
    # histogram of test set targets
    plt.hist(test_targets, bins=train_targets_bins)
    plt.ylabel('count')
    plt.xlabel(f'{target_name}')
    plt.title(f'Histogram of \'{target_name}\' test set target values')
    test_xlim = plt.gca().get_xlim()
    plt.show()

    # histogram of preds
    plt.title(
        f'distribution of model predictions'
        f' for \'{target_name}\'\n'
        f'MSE = {trained_model_mse:.4f} '
        f'(mean model: {mean_model_mse:.4f})'
    )
    plt.hist(test_preds, bins=test_preds_bins)
    # set preds xlim to that of test targets
    plt.xlim(test_xlim)
    plt.xlabel('model prediction')
    plt.ylabel('count')
    plt.show()
    
    # scatterplot of preds vs. targets
    plt.title(
        f'model predictions vs. test-set targets'
        f' for \'{target_name}\'\n'
        rf'$R^2$ = {trained_model_R2:.4f} '
        f'(mean model: {mean_model_R2:.4f})'
    )
    plt.scatter(
        test_targets, 
        test_preds, 
        color='C0',
        alpha=0.5,
        zorder=np.inf
    )

    # simple y = x line (since preds = targets in perfect model)
    plt.plot(
        np.unique(test_targets), 
        np.poly1d(np.polyfit(
            test_targets, test_preds, 1)
        )(np.unique(test_targets)),
        c='C0'
    )
    # plt.axis('equal')
    plt.axvline(train_mean, linestyle='--', c='gray', zorder=0)
    plt.axhline(train_mean, linestyle='--', c='gray', zorder=0)
    lims = [
        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
    ]
    plt.text(
        lims[0],
        train_mean + 0.1, 
        f'train set mean = {train_mean:.2f}', 
        rotation=0.,
        c='gray'
    )
    # now plot both limits against eachother
    plt.plot(lims, lims, c='gray', zorder=0)
    plt.xlabel('test set target')
    plt.ylabel('model prediction')
    plt.ylim(plt.gca().get_xlim())
    plt.show()


class LinearDecider(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        wts_init_fn: Callable = \
            torch.nn.init.xavier_uniform_
    ):
        super(LinearDecider, self).__init__()    
        self.ll = nn.Linear(input_size, output_size)
        wts_init_fn(self.ll.weight)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['x']
        x = self.ll(x)
        # contain model output in dict
        model_output_dict = {'preds': x}
        return model_output_dict



def build_ffnn(
    input_dim: int, 
    output_dim: int, 
    hidden_dims_list: List[int], # e.g. [1024, 256, 64, 16], 
    bias_in_hidden_layers: bool,
    nonlin_fn: torch.nn.Module, # a torch.nn activation fn
    nonlin_fn_kwargs: Dict[str, Any]
) -> Tuple[nn.ModuleList, nn.Module, nn.Module]:
    """
    Builds a simple feed-forward, fully-connected neural 
    network (aka multilayer perceptron, or MLP) programatically.
    
    Note: returns pieces that model's 'forward()' must iterate
    through, e.g.:
    def forward(self, x):
        for i in range(len(self.lin_fns)):
            x = self.lin_fns[i](x)
            x = self.nonlin_fns[i](x)
            if self.use_dropout:
                x = nn.Dropout(self.dropout_p)
        x = self.lin_out(x)

    Args:
        input_dim: int value of network's input dimension.
        output_dim: int value of network's final output
            dimension.
        hidden_dims_list: list of int values of dimension
            for each hidden linear layer.
        bias_in_hidden_layers: bool whether to include a
            bias term in the hidden layers.
        nonlin_fn: the torch.nn activation function to 
            apply to each linear layer.
        nonlin_fn_kwargs: any kwargs to pass to nonlin_fn.
    Returns:
        3-tuple of the linear layers (as nn.ModuleList), 
        the activation function (as nn.Module), and final 
        the linear layer (as a single nn.Module, with no 
        activation function).
    """
    lin_fns = [None] * len(hidden_dims_list)
    next_input_dim = input_dim
    for i, hidden_dim in enumerate(hidden_dims_list):
        lin_fns[i] = nn.Linear(
            next_input_dim, 
            hidden_dim, 
            bias=bias_in_hidden_layers
        )
        next_input_dim = hidden_dim
    lin_fns = nn.ModuleList(lin_fns)
    nonlin_fns = nonlin_fn(**nonlin_fn_kwargs)
    final_lin = nn.Linear(
        next_input_dim, 
        output_dim, 
        bias=True
    )
    return (lin_fns, nonlin_fns, final_lin)


class ProjectionMLP(nn.Module):
    """
    Configurable MLP. Supports one or more hidden layers with optional BatchNorm,
    customizable activation, and optional residual connections.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | list[int] | tuple[int, ...] = 128,
        embedding_dim: int = 128,
        activation: type[nn.Module] = nn.ReLU,
        use_batch_norm: bool = True,
        dropout_p: float | None = None,
        residual_style: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, (list, tuple)):
            hidden_dims = [int(h) for h in hidden_dim]
        else:
            hidden_dims = [int(hidden_dim)]

        self.residual_style = bool(residual_style)
        self.use_batch_norm = bool(use_batch_norm)
        self.dropout_p = float(dropout_p) if dropout_p is not None else None
        self.activation = activation

        if self.residual_style:
            self._validate_residual_dims(hidden_dims)
            first_dim = hidden_dims[0]
            self.input_proj = nn.Linear(in_dim, first_dim)
            self.blocks = nn.ModuleList([self._make_block(first_dim) for _ in hidden_dims])
            self.output_proj = nn.Linear(first_dim, embedding_dim)
            linear_layers = self._gather_residual_linears()
        else:
            self.net, linear_layers = self._build_standard_net(
                in_dim=in_dim,
                hidden_dims=hidden_dims,
                embedding_dim=embedding_dim,
            )

        self._init_linear_layers(linear_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.residual_style:
            return self.net(x)

        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(x)

    def _make_block(self, dim: int) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Linear(dim, dim),
        ]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(self.activation())
        if self.dropout_p is not None and self.dropout_p > 0.0:
            layers.append(nn.Dropout(p=self.dropout_p))
        return nn.Sequential(*layers)

    def _build_standard_net(
        self,
        *,
        in_dim: int,
        hidden_dims: list[int],
        embedding_dim: int,
    ) -> tuple[nn.Sequential, list[nn.Linear]]:
        layers: list[nn.Module] = []
        linear_layers: list[nn.Linear] = []
        last_dim = in_dim
        for hid in hidden_dims:
            lin = nn.Linear(last_dim, hid)
            layers.append(lin)
            linear_layers.append(lin)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hid))
            layers.append(self.activation())
            if self.dropout_p is not None and self.dropout_p > 0.0:
                layers.append(nn.Dropout(p=self.dropout_p))
            last_dim = hid
        out_lin = nn.Linear(last_dim, embedding_dim)
        layers.append(out_lin)
        linear_layers.append(out_lin)
        return nn.Sequential(*layers), linear_layers

    def _gather_residual_linears(self) -> list[nn.Linear]:
        linear_layers: list[nn.Linear] = [self.input_proj]
        for block in self.blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)
        linear_layers.append(self.output_proj)
        return linear_layers

    def _init_linear_layers(self, linear_layers: list[nn.Linear]) -> None:
        """
        Initialize hidden layers with Kaiming uniform when using ReLU-like
        activations, and the final projection with Xavier uniform.
        """
        nonlin = 'relu'
        if self.activation in (nn.LeakyReLU,):
            nonlin = 'leaky_relu'
        elif self.activation not in (nn.ReLU, nn.LeakyReLU, nn.ReLU6):
            nonlin = 'linear'

        total = len(linear_layers)
        for idx, layer in enumerate(linear_layers):
            is_final = idx == total - 1
            if is_final or nonlin == 'linear':
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity=nonlin)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _validate_residual_dims(hidden_dims: list[int]) -> None:
        if not hidden_dims:
            raise ValueError("Residual projection requires at least one hidden dimension.")
        first_dim = hidden_dims[0]
        if any(h != first_dim for h in hidden_dims):
            raise ValueError("All hidden dimensions must match when residual_style is True.")


class EpochCounter:
    """
    Class for counting epochs and best metrics achieved at which
    epoch, with state_dict implementations, for use in saving and 
    loading model states for continuing training, etc. Useful with 
    the 'accelerate' library, which requires state_dict methods
    for saving/reloading object states.

    Note that validation loss is tracked by default, but an 
    additional metric of interest can also be tracked, when
    its string key is passed to 'metric_name' in __init__.
    """
    def __init__(
        self,
        n: int = 0,
        metric_name: str = 'loss_valid'
    ):
        """
        Args:
            n: int starting epoch.
            metric_name: string key value of the primary 
                metric of interest, for tracking by epoch.
        """
        self.n = n
        self.best = {
            metric_name: {
                'epoch': 0,
                'score': 0.0
            },
            # track valid loss separately, even if 
            # it's the metric of interest
            '_valid_loss': {
                'epoch': 0,
                'score': float('inf')
            }
        }

    def __iadd__(self, m: int):
        self.n = self.n + m
        return self

    def state_dict(self) -> dict:
        return self.__dict__

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def set_best(
        self, 
        metric: str, 
        epoch: int, 
        score: float
    ) -> None:
        if metric in self.best.keys():
            self.best[metric]['epoch'] = epoch
            self.best[metric]['score'] = score
        else:
            self.best[metric] = {
                'epoch': epoch,
                'score': score
            }

    def __str__(self) -> str:
        return str(self.n)


class Class1PredsCounter:
    """
    For binary classification tasks, this
    object keeps track of class 1 logit prediction
    counts, in order to print the proportion
    of class 1 predictions within each epoch 
    and each train and valid phase).
    """
    def __init__(self):
        self.reset()
    
    def update(self, output_dict, phase):
        preds = output_dict['preds'].squeeze()
        # print("preds.shape:", preds.shape)
        # note: a (logit > 0) = (p > 0.5) = class 1 pred
        self.ctr[phase]['class1'] += torch.sum(preds > 0.)
        self.ctr[phase]['total'] += preds.shape[0]

    def print_preds_counts(self):
        print(f'class 1 predictions:')
        for phase in ('train', 'valid'):
            preds_d = self.ctr[phase]
            class1_preds_ct, all_preds_ct = preds_d['class1'], preds_d['total']
            perc = 100 * (class1_preds_ct / all_preds_ct)
            print(f'\t{phase}: {class1_preds_ct} / {all_preds_ct} ({perc:.1f}%)')

    def reset(self):
        self.ctr = {
            'train': {'class1': 0, 'total': 0},
            'valid': {'class1': 0, 'total': 0}
        }

