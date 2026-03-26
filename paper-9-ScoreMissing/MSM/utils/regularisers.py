import numpy as np
import torch
from torch import nn
from ..models.density_models import NormalDensity, NormalCholDensity
from typing import Union


class GenRegulariser(nn.Module):
    def __init__(self, model: nn.Module, regsize=1, regtype=2):
        super().__init__()
        self.model = model
        self.regsize = regsize
        self.regtype = regtype

    def forward(self, *args, **kwargs):
        out = 0
        for parameter in self.model.parameters():
            out += self.regsize*torch.linalg.vector_norm(parameter, ord=self.regtype)
        return out


class OffdiagRegulariser(GenRegulariser):
    def forward(self, *args, **kwargs):
        self.model: Union[NormalDensity, NormalCholDensity]
        if (isinstance(self.model, NormalCholDensity)):
            self.model.update_params()

        parameter = (1-torch.eye(self.model.Precision.shape[0]))*self.model.Precision
        out = self.regsize*torch.linalg.vector_norm(parameter, ord=self.regtype)
        return out


class DetReg(GenRegulariser):
    def forward(self, *args, **kwargs):
        if (isinstance(self.model, NormalCholDensity)):
            return self.regsize*torch.abs(
                np.log(2.)+torch.sum(torch.log(torch.diagonal(self.model.chol_Precision))))

        elif (isinstance(self.model, NormalDensity)):
            return self.regsize*torch.abs(torch.logdet(self.model.Precision))


class WeightReg(GenRegulariser):
    def forward(self, *args, **kwargs):
        out = 0
        for name, parameter in self.model.named_parameters():
            if "weight" in name:
                out += self.regsize*torch.linalg.vector_norm(parameter, ord=self.regtype)
        return out
