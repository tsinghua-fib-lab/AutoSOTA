"""
DIMAT utility functions for model alignment and merging.
Ported from DIMAT/utils/am_utils.py.
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm.auto import tqdm


class SpaceInterceptor(nn.Module):
    """
    Intercepts computational flows between two layers.
    Inserting this module between two layers allows computing a merge/unmerge
    on each layer separately. Most useful for controlling transformations
    learned over residual connections.

    Contains a single weight (identity matrix) that will be transformed
    according to the unmerge/merge applied over it.
    """

    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(dim))

    def forward(self, input, kind="linear"):
        if kind == "conv":
            input = input.permute(0, 2, 3, 1)

        output = input @ self.weight.T

        if kind == "conv":
            output = output.permute(0, 3, 1, 2)

        return output


def reset_bn_stats(model, loader, reset=True):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    device = get_device(model)
    has_bn = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if reset:
                m.momentum = None  # use simple average
                m.reset_running_stats()
            has_bn = True

    if not has_bn:
        return model

    model.train()
    with torch.no_grad(), autocast("cuda" if device.type == "cuda" else "cpu"):
        for images, _ in tqdm(loader, desc="Resetting batch norm"):
            _ = model(images.to(device))
    return model


def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device
