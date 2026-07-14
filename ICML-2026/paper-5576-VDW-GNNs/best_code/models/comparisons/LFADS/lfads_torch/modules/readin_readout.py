import abc
import math
from glob import glob

import h5py
import numpy as np
import torch
from torch import nn


class FanInLinear(nn.Linear):
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.in_features))
        nn.init.constant_(self.bias, 0.0)


class _MultisessionModuleList(abc.ABC, nn.ModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        pcr_init: bool,
        requires_grad: bool,
    ):
        modules = []
        # Identify paths that match the datafile pattern
        data_paths = sorted(glob(datafile_pattern))
        for data_path in data_paths:
            if pcr_init:
                # Load the pre-computed transformations
                state_dict = self._get_state_dict(data_path)
                out_features, in_features = state_dict["weight"].shape
                layer = nn.Linear(in_features, out_features)
                layer.load_state_dict(state_dict)
            else:
                # Infer only the input dimension from the file
                in_features, out_features = self._get_layer_shape(data_path)
                layer = nn.Linear(in_features, out_features)
            modules.append(layer)
        # Create the nn.ModuleList
        super().__init__(modules)
        # Allow the user to set requires_grad
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    @abc.abstractmethod
    def _get_layer_shape(self, data_path):
        pass

    @abc.abstractmethod
    def _get_state_dict(self, data_path):
        pass


class MultisessionReadin(_MultisessionModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        out_features: int = None,
        pcr_init: bool = True,
        requires_grad: bool = False,
    ):
        assert (
            out_features is not None
        ) != pcr_init, "Setting `out_features` mutually excludes `pcr_init`."
        self.out_features = out_features
        super().__init__(
            datafile_pattern=datafile_pattern,
            pcr_init=pcr_init,
            requires_grad=requires_grad,
        )

    def _get_layer_shape(self, data_path):
        with h5py.File(data_path) as h5file:
            in_features = h5file["train_encod_data"].shape[-1]
        return in_features, self.out_features

    def _get_state_dict(self, data_path):
        with h5py.File(data_path) as h5file:
            weight = h5file["readin_weight"][()]
            bias = -np.dot(h5file["readout_bias"][()], weight)
        return {"weight": torch.tensor(weight.T), "bias": torch.tensor(bias)}


class MultisessionReadout(_MultisessionModuleList):
    def __init__(
        self,
        datafile_pattern: str,
        in_features: int = None,
        pcr_init: bool = True,
        requires_grad: bool = True,
        recon_params: int = 1,
    ):
        assert (
            in_features is not None
        ) != pcr_init, "Setting `in_features` mutually excludes `pcr_init`."
        self.in_features = in_features
        self.recon_params = recon_params
        super().__init__(
            datafile_pattern=datafile_pattern,
            pcr_init=pcr_init,
            requires_grad=requires_grad,
        )

    def _get_layer_shape(self, data_path):
        with h5py.File(data_path) as h5file:
            out_features = h5file["train_recon_data"].shape[-1]
            out_features *= self.recon_params
        return self.in_features, out_features

    def _get_state_dict(self, data_path):
        with h5py.File(data_path) as h5file:
            weight = np.linalg.pinv(h5file["readin_weight"][()])
            bias = h5file["readout_bias"][()]
        return {"weight": torch.tensor(weight.T), "bias": torch.tensor(bias)}
