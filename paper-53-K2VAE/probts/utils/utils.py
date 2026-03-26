# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import re
import os
import torch
import numpy as np
from typing import Optional, Dict
import torch.nn as nn
import importlib

def repeat(tensor: torch.Tensor, n: int, dim: int = 0):
    return tensor.repeat_interleave(repeats=n, dim=dim)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


    
def weighted_average(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: int = None,
    reduce: str = 'mean',
):
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        x: Input tensor, of which the average must be computed.
        weights: Weights tensor, of the same shape as `x`.
        dim: The dim along which to average `x`

    Returns:
        Tensor: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        if reduce != 'mean':
            return weighted_tensor
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim) if dim else x
    
    
def convert_to_list(s):
    '''
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    '''
    if (type(s).__name__=='int'):
        return [s]
    elif (type(s).__name__=='list'):
        return s
    elif (type(s).__name__=='str'):
        elements = re.split(r'\D+', s)
        return list(map(int, elements))
    else:
        return None
    
def find_best_epoch(ckpt_folder):
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: Integer of the highest epoch reached by the checkpoints.
    """
    pattern = r"val_CRPS=([0-9]*\.[0-9]+)"
    ckpt_files = os.listdir(ckpt_folder)  # list of strings
    epochs = []
    for filename in ckpt_files:
        match = re.search(pattern, filename)
        if match:  
            epochs.append(match.group(1))
    # epochs = [float(filename[18:-5]) for filename in ckpt_files]  # 'epoch={int}.ckpt' filename format
    best_crps = min(epochs)
    
    best_epoch = epochs.index(best_crps)
    return best_epoch, ckpt_files[best_epoch]

def ensure_list(input_value, default_value=None):
    """
    Ensures that the input is converted to a list. If the input is None,
    it converts the default value to a list instead.
    """
    result = convert_to_list(input_value)
    if result is None:
        result = convert_to_list(default_value)
    return result


def init_class_helper(class_name):
    """
    Dynamically imports a module and retrieves a class.

    Args:
        class_name (str): The fully qualified name of the class in the format "module_name.ClassName".

    Returns:
        type: The class object retrieved from the specified module.
    """
    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class