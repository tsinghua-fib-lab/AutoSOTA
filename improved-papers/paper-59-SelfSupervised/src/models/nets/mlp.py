import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, List
from models.modules.snrlinear import SNLinear


class Model(nn.Module):
    def __init__(
        self,
        name: str,
        input_size: int,
        output_size: int,
        num_channels: int,
        loss_names: List[str],
        dim: int,
        with_revin: bool,
        if_sample: bool = False,
        if_snr: bool = False,
        sample_size: int = 0,
        eps: float = 1e-5,
        num_models: int = 1,
        **kwargs
    ):
        super(Model, self).__init__()
        if if_snr:
            self.model = nn.Sequential(
                SNLinear(input_size, dim // 2), 
                nn.Linear(dim // 2, dim),
                nn.SiLU(),
                nn.Linear(dim, output_size)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_size, dim),
                nn.SiLU(),
                nn.Linear(dim, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, output_size)
            )
        self.with_revin = with_revin

    def init_params(self):
        pass

    def save_parameters(self):
        pass

    def predict(self, x, *args, **kwargs):
        if self.with_revin:
            means = x.mean(dim=-1, keepdim=True)
            stds = x.std(dim=-1, keepdim=True)
            x = (x - means) / (stds + 1e-6)
        y_hat = self.model(x)
        if self.with_revin:
            y_hat = y_hat * (stds + 1e-6) + means
        return y_hat

    def forward(self, x, y, x_mark=None, y_mark=None):
        result = {}
        y_hat = self.predict(x)
        result["y_hat"] = y_hat
        result["loss_tar"] = F.mse_loss(y_hat, y)
        result["loss_total"] = result["loss_tar"]
        return result
