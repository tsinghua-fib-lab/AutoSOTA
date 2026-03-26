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
from torch.distributions import Normal
import torch.nn.functional as F
from probts.model.forecaster import Forecaster


class LinearForecaster(Forecaster):
    def __init__(
        self,
        individual: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.individual = individual
        
        if self.individual:
            self.linear = nn.ModuleList()
            for i in range(self.input_size):
                self.linear.append(nn.Linear(self.context_length, self.prediction_length))
        else:
            self.linear = nn.Linear(self.context_length, self.prediction_length)
        self.out_linear_mu = nn.Linear(self.input_size, self.target_dim)
        self.out_linear_sigma = nn.Linear(self.input_size, self.target_dim)
        self.loss_fn = lambda dist,targets: -dist.log_prob(targets)

    def forward(self, x):

        outputs = self.linear(x.permute(0,2,1)).permute(0,2,1)
        mu = self.out_linear_mu(outputs)
        sigma = self.out_linear_sigma(outputs)
        return Normal(loc=mu, scale=F.softplus(sigma) + 1e-6)

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        dist = self(inputs)
        return dist.rsample(sample_shape=(num_samples,))

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        dist = self(inputs)
        
        loss = self.loss_fn(dist,batch_data.future_target_cdf)
        return loss.mean()
