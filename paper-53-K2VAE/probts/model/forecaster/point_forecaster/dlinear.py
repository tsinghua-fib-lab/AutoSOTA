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


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(Forecaster):
    def __init__(
            self,
            kernel_size: int,
            individual: bool,
            **kwargs
    ):
        super().__init__(**kwargs)
        if self.input_size != self.target_dim:
            self.enc_linear = nn.Linear(
                in_features=self.input_size, out_features=self.target_dim
            )
        else:
            self.enc_linear = nn.Identity()

        # Decompsition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual



        self.Linear_Seasonal = nn.Linear(self.context_length, self.prediction_length)
        self.Linear_Trend = nn.Linear(self.context_length, self.prediction_length)

        self.Linear_Seasonal_sigma = nn.Linear(self.context_length, self.prediction_length)
        self.Linear_Trend_sigma = nn.Linear(self.context_length, self.prediction_length)

        self.loss_fn = lambda dist, targets: -dist.log_prob(targets)

    def encoder(self, inputs):
        seasonal_init, trend_init = self.decompsition(inputs)

        # [B,C,L]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        seasonal_output_mu = self.Linear_Seasonal(seasonal_init)
        trend_output_mu = self.Linear_Trend(trend_init)

        seasonal_output_sigma = self.Linear_Seasonal_sigma(seasonal_init)
        trend_output_sigma = self.Linear_Trend_sigma(trend_init)

        mu = seasonal_output_mu + trend_output_mu  # [B,C,L]
        sigma = seasonal_output_sigma + trend_output_sigma

        return Normal(loc=mu.permute(0, 2, 1), scale=F.softplus(sigma).permute(0, 2, 1) + 1e-6)

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        dist = self.encoder(inputs)
        targets = batch_data.future_target_cdf
        loss = self.loss_fn(dist, targets)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        dist = self.encoder(inputs)
        return dist.rsample(sample_shape=(num_samples,))
