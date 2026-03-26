# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Multi-variate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import copy
import torch
import torch.nn as nn
from probts.model.nn.prob.flow_model import FlowModel, BatchNorm, FlowSequential


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer("mask", mask)

        # scale function
        s_net = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (
            1 - self.mask
        )

        # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t
        # u = (x - t) * torch.exp(log_s)
        # u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)

        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        # log_abs_det_jacobian = -(1 - self.mask) * s
        # log_abs_det_jacobian = -log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = log_s

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (
            1 - self.mask
        )

        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)
        # x = u * torch.exp(log_s) + t
        # x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        # log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        # log_abs_det_jacobian = log_s #.sum(-1, keepdim=True)
        log_abs_det_jacobian = -log_s

        return x, log_abs_det_jacobian


class RealNVP(FlowModel):
    def __init__(
        self,
        n_blocks,
        target_dim,
        hidden_size,
        n_hidden,
        f_hidden_size,
        conditional_length,
        dequantize,
        batch_norm=True
    ):
        super().__init__(target_dim, f_hidden_size, conditional_length, dequantize)

        # construct model
        modules = []
        mask = torch.arange(target_dim).float() % 2
        for i in range(n_blocks):
            modules += [
                LinearMaskedCoupling(
                    target_dim, hidden_size, n_hidden, mask, conditional_length
                )
            ]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(target_dim)]

        self.net = FlowSequential(*modules)