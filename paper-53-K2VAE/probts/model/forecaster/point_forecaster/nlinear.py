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
import torch.nn.functional as F
from torch.distributions import Normal


class NLinear(Forecaster):
    def __init__(
            self,
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

        self.target_dim = self.target_dim
        self.individual = individual

        self.Linear_mu = nn.Linear(self.context_length, self.prediction_length)
        self.Linear_sigma = nn.Linear(self.context_length, self.prediction_length)

        self.loss_fn = lambda dist, targets: -dist.log_prob(targets)

    def forward(self, inputs):
        seq_last = inputs[:, -1:, :].detach()
        inputs = inputs - seq_last

        mu = self.Linear_mu(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        sigma = self.Linear_sigma(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        mu = mu + seq_last

        return Normal(loc=mu, scale=F.softplus(sigma) + 1e-6)

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        inputs = inputs[:, : self.context_length, ...]
        inputs = self.enc_linear(inputs)
        dist = self(inputs)
        targets = batch_data.future_target_cdf
        loss = self.loss_fn(dist, targets)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        dist = self(inputs)
        return dist.rsample(sample_shape=(num_samples,))
