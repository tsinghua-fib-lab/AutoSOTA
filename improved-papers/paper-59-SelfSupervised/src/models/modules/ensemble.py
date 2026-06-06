# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import warnings

import torch
import torch.nn as nn
from tensordict import LazyStackedTensorDict, TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.common import TensorDictBase, TensorDictModuleBase

from typing import Optional


class TensorDictModule(TensorDictModule):
    def __init__(self, *args, **kwargs):
        super(TensorDictModule, self).__init__(*args, **kwargs)

    def reset_parameters_recursive(self, parameters=None):
        if isinstance(self.module, nn.Module):
            for m in self.module.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()


class EnsembleOutputDict(dict):
    """Simple dict wrapper to support key access like TensorDict."""
    def __getitem__(self, key):
        return super().__getitem__(key)


class EnsembleModule(nn.Module):
    """Ensemble module using ModuleList for proper device handling.

    Each copy of the module is stored in a ModuleList, ensuring parameters
    are properly tracked and moved to GPU with proper gradient support.
    """

    def __init__(
        self,
        module: TensorDictModuleBase,
        num_copies: int,
        expand_input: bool = True,
        randomness: str = 'error',
    ):
        super().__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        self.num_copies = num_copies
        self.expand_input = expand_input

        # Get the inner nn.Module
        base_inner = module.module if hasattr(module, 'module') else module

        # Create num_copies with different random initializations
        copies = [copy.deepcopy(base_inner) for _ in range(num_copies)]
        for m in copies:
            for layer in m.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        self.copies = nn.ModuleList(copies)
        # Keep reference to base module for in_keys/out_keys
        self.module = module

    def forward(self, tensordict):
        """Forward pass - returns a dict with stacked outputs.

        Output shape: [num_copies, B, C, L] for y_hat key.
        """
        x = tensordict['x']
        x_mark = tensordict.get('x_mark', None)
        out_list = []

        for copy_module in self.copies:
            if x_mark is not None:
                out = copy_module(x, x_mark)
            else:
                out = copy_module(x)

            if isinstance(out, dict):
                out = out[self.out_keys[0]]
            out_list.append(out)

        # Stack along dim 0: [num_copies, B, C, L]
        stacked = torch.stack(out_list, dim=0)

        # Return dict-like object with same interface as TensorDict
        result = {self.out_keys[0]: stacked}
        return EnsembleOutputDict(result)

    def reset_parameters_recursive(self, parameters=None):
        """No-op: parameters are initialized in __init__."""
        pass
