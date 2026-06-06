import torch
import torch.nn as nn

from probts.data import ProbTSBatchData
from probts.utils import repeat
from probts.model.forecaster import Forecaster
import torch.nn.functional as F
from torch.distributions import Normal


class GRUForecaster(Forecaster):
    def __init__(
            self,
            num_layers: int = 2,
            f_hidden_size: int = 40,
            dropout: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive = True

        self.model = nn.GRU(
            input_size=self.input_size,
            hidden_size=f_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear_mu = nn.Linear(f_hidden_size, self.target_dim)
        self.linear_sigma = nn.Linear(f_hidden_size, self.target_dim)

        self.loss_fn = lambda dist, targets: -dist.log_prob(targets)

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        outputs, _ = self.model(inputs)
        outputs = outputs[:, -self.prediction_length - 1:-1, ...]
        mu = self.linear_mu(outputs)
        sigma = self.linear_sigma(outputs)
        dist = Normal(loc=mu, scale=F.softplus(sigma) + 1e-6)
        loss = self.loss_fn(dist, batch_data.future_target_cdf)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        forecasts_mu = []
        forecasts_sigma = []
        states = self.encode(batch_data)
        past_target_cdf = batch_data.past_target_cdf

        for k in range(self.prediction_length):
            current_batch_data = ProbTSBatchData({
                'target_dimension_indicator': batch_data.target_dimension_indicator,
                'past_target_cdf': past_target_cdf,
                'future_time_feat': batch_data.future_time_feat[:, k: k + 1:, ...]
            }, device=batch_data.device)

            outputs, states = self.decode(current_batch_data, states)
            mu = self.linear_mu(outputs)
            sigma = self.linear_sigma(outputs)
            forecasts_mu.append(mu)
            forecasts_sigma.append(sigma)
            past_target_cdf = torch.cat(
                (past_target_cdf, mu), dim=1
            )

        forecasts_mu = torch.cat(forecasts_mu, dim=1).reshape(
            -1, self.prediction_length, self.target_dim)
        forecasts_sigma = torch.cat(forecasts_sigma, dim=1).reshape(
            -1, self.prediction_length, self.target_dim)
        dist = Normal(loc=forecasts_mu, scale=F.softplus(forecasts_sigma) + 1e-6)
        return dist.rsample(sample_shape=(num_samples,))

    def encode(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        outputs, states = self.model(inputs)
        return states

    def decode(self, batch_data, states=None):
        inputs = self.get_inputs(batch_data, 'decode')
        outputs, states = self.model(inputs, states)
        return outputs, states
