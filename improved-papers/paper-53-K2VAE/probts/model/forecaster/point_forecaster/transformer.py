import torch
import torch.nn as nn

from probts.data import ProbTSBatchData
from probts.model.forecaster import Forecaster
import torch.nn.functional as F
from torch.distributions import Normal


class TransformerForecaster(Forecaster):
    def __init__(
            self,
            f_hidden_size: int = 32,
            num_heads: int = 8,
            num_encoder_layers: int = 3,
            num_decoder_layers: int = 3,
            dim_feedforward_scale: int = 4,
            dropout: float = 0.1,
            activation: str = 'gelu',
            **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive = True
        self.f_hidden_size = f_hidden_size

        self.enc_linear = nn.Linear(self.input_size, self.f_hidden_size)
        self.dec_linear = nn.Linear(self.input_size, self.f_hidden_size)
        self.model = nn.Transformer(
            d_model=self.f_hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward_scale * self.f_hidden_size,
            dropout=dropout,
            activation=activation
        )

        self.register_buffer(
            "tgt_mask",
            self.model.generate_square_subsequent_mask(self.prediction_length),
        )
        self.linear_mu = nn.Linear(self.f_hidden_size, self.target_dim)
        self.linear_sigma = nn.Linear(self.f_hidden_size, self.target_dim)
        self.loss_fn = lambda dist, target: -dist.log_prob(target)

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')  # [B L D]

        # Encode
        enc_inputs = inputs[:, :self.context_length, ...]
        enc_inputs = self.enc_linear(enc_inputs).permute(1, 0, 2)
        enc_outputs = self.model.encoder(enc_inputs)  # [L_in B H]

        # Decode
        dec_inputs = inputs[:, -self.prediction_length - 1:-1, ...]
        dec_inputs = self.dec_linear(dec_inputs).permute(1, 0, 2)
        dec_outputs = self.model.decoder(
            dec_inputs, enc_outputs, tgt_mask=self.tgt_mask)
        dec_outputs = dec_outputs.permute(1, 0, 2)  # [L_out B D]
        mu = self.linear_mu(dec_outputs)
        sigma = self.linear_sigma(dec_outputs)
        dist = Normal(mu, F.softplus(sigma) + 1e-6)
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
        inputs = self.enc_linear(inputs).permute(1, 0, 2)
        states = self.model.encoder(inputs)
        return states

    def decode(self, batch_data, states=None):
        inputs = self.get_inputs(batch_data, 'decode')
        inputs = self.dec_linear(inputs).permute(1, 0, 2)
        outputs = self.model.decoder(inputs, states, tgt_mask=None)
        return outputs.permute(1, 0, 2), states
