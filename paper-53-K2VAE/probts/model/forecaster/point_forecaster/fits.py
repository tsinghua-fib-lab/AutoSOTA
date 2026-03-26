# ---------------------------------------------------------------------------------
# Portions of this file are derived from LTSF-Linear
# - Source: https://github.com/cure-lab/LTSF-Linear
# - Paper: Are Transformers Effective for Time Series Forecasting?
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
from probts.model.forecaster import Forecaster
import sys
import torch.nn.functional as F
from torch.distributions import Normal

import torch
import torch.nn as nn


class FITS(Forecaster):
    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self,
                 cut_freq: int,
                 individual: bool,
                 **kwargs):

        super().__init__(**kwargs)
        self.seq_len = self.context_length
        self.pred_len = self.prediction_length
        self.individual = individual
        self.channels = self.target_dim

        self.dominance_freq = cut_freq  # 720/24
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(
                        self.dominance_freq,
                        int(self.dominance_freq * self.length_ratio),
                    ).to(torch.cfloat)
                )

        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, int(self.dominance_freq * self.length_ratio)
            ).to(
                torch.cfloat
            )

        self.encoder_mu = nn.Linear(self.prediction_length,self.prediction_length)
        self.encoder_sigma = nn.Linear(self.prediction_length,self.prediction_length)
        self.loss_fn = lambda dist, target: -dist.log_prob(target)

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        dist = self.forward(inputs)
        targets = batch_data.future_target_cdf
        loss = self.loss_fn(dist, targets)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        dist = self.forward(inputs)
        return dist.rsample(sample_shape=(num_samples,))

    def forward(self, x):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0  # LPF
        low_specx = low_specx[:, 0: self.dominance_freq, :]  # LPF
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros(
                [
                    low_specx.size(0),
                    int(self.dominance_freq * self.length_ratio),
                    low_specx.size(2),
                ],
                dtype=low_specx.dtype,
            ).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](
                    low_specx[:, :, i].permute(0, 1)
                ).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(
                0, 2, 1
            )
        # print(low_specxy_)
        low_specxy = torch.zeros(
            [
                low_specxy_.size(0),
                int((self.seq_len + self.pred_len) / 2 + 1),
                low_specxy_.size(2),
            ],
            dtype=low_specxy_.dtype,
        ).to(low_specxy_.device)
        low_specxy[:, 0: low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # energy compemsation for the length change
        # dom_x=x-low_x

        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy = (low_xy) * torch.sqrt(x_var) + x_mean
        output = xy.permute(0, 2, 1)[:,:,-self.prediction_length:]
        mu = self.encoder_mu(output).permute(0, 2, 1)
        sigma = self.encoder_sigma(output).permute(0, 2, 1)

        return Normal(loc=mu, scale=F.softplus(sigma) + 1e-6)
