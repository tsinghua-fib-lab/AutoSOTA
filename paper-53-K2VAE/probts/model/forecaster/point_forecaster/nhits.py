# ---------------------------------------------------------------------------------
# Portions of this file are derived from NeuralForecast
# - Source: https://github.com/Nixtla/neuralforecast
# - Paper: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat
from functools import partial
from typing import List, Tuple
from torch.distributions import Normal
from probts.model.forecaster import Forecaster


class StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        layers = [nn.Dropout(p=0.5), nn.Linear(in_features=in_features, out_features=out_features), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or ("cubic" in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(
            self,
            backcast_theta: torch.Tensor,
            forecast_theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = backcast_theta
        knots = forecast_theta

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )  # , align_corners=True)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size: (i + 1) * batch_size], size=self.forecast_size, mode="bicubic"
                )  # , align_corners=True)
                forecast[i * batch_size: (i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast


def init_weights(module, initialization):
    if type(module) == torch.nn.Linear:
        if initialization == "orthogonal":
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == "he_uniform":
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == "he_normal":
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == "glorot_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == "glorot_normal":
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == "lecun_normal":
            pass  # torch.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1 < 0, f"Initialization {initialization} not found"


ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]


class NHiTSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """

    def __init__(
            self,
            context_length: int,
            prediction_length: int,
            output_size: int,
            covariate_size: int,
            static_size: int,
            static_hidden_size: int,
            n_theta: int,
            hidden_size: List[int],
            pooling_sizes: int,
            pooling_mode: str,
            basis: nn.Module,
            n_layers: int,
            batch_normalization: bool,
            dropout: float,
            activation: str,
    ):
        super().__init__()

        assert pooling_mode in ["max", "average"]

        self.context_length_pooled = int(np.ceil(context_length / pooling_sizes))

        if static_size == 0:
            static_hidden_size = 0

        self.context_length = context_length
        self.output_size = [output_size]
        self.n_theta = n_theta
        self.prediction_length = prediction_length
        self.static_size = static_size
        self.static_hidden_size = static_hidden_size
        self.covariate_size = covariate_size
        self.pooling_sizes = pooling_sizes
        self.batch_normalization = batch_normalization
        self.dropout = dropout

        hidden1 = [self.context_length_pooled * len(self.output_size) + (
                    self.context_length + self.prediction_length) * self.covariate_size + self.static_hidden_size]
        self.hidden_size = hidden1 + hidden_size

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        activ = getattr(nn, activation)()

        if pooling_mode == "max":
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)
        elif pooling_mode == "average":
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=self.hidden_size[i], out_features=self.hidden_size[i + 1]))
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=self.hidden_size[i + 1]))

            if self.dropout > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout))

        output_layer = [
            nn.Linear(
                in_features=self.hidden_size[-1],
                out_features=context_length * len(self.output_size) + n_theta * sum(self.output_size),
            )
        ]
        layers = hidden_layers + output_layer

        # static_size is computed with data, static_hidden_size is provided by user, if 0 no statics are used
        if (self.static_size > 0) and (self.static_hidden_size > 0):
            self.static_encoder = StaticFeaturesEncoder(in_features=static_size, out_features=static_hidden_size)
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
            self, encoder_y: torch.Tensor, encoder_x_t: torch.Tensor, decoder_x_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encoder_y)

        encoder_y = encoder_y.transpose(1, 2)
        # Pooling layer to downsample input
        encoder_y = self.pooling_layer(encoder_y)

        encoder_y = encoder_y.transpose(1, 2).reshape(batch_size, -1)

        if self.covariate_size > 0:
            encoder_y = torch.cat(
                (
                    encoder_y,
                    encoder_x_t.reshape(batch_size, -1),
                    decoder_x_t.reshape(batch_size, -1),
                ),
                1,
            )

        # Compute local projection weights and projection
        theta = self.layers(encoder_y)
        backcast_theta = theta[:, : self.context_length * len(self.output_size)].reshape(-1, self.context_length)
        forecast_theta = theta[:, self.context_length * len(self.output_size):].reshape(-1, self.n_theta)
        backcast, forecast = self.basis(backcast_theta, forecast_theta)
        backcast = backcast.reshape(-1, len(self.output_size), self.context_length).transpose(1, 2)
        forecast = forecast.reshape(-1, sum(self.output_size), self.prediction_length).transpose(1, 2)

        return backcast, forecast


class NHiTS(Forecaster):
    def __init__(
            self,
            n_blocks: list,
            pooling_mode,
            interpolation_mode,
            dropout,
            activation,
            initialization,
            batch_normalization,
            shared_weights,
            output_size: int = 1,
            hidden_size: int = 512,
            naive_level: bool = True,
            static_size: int = 0,
            static_hidden_size: int = 0,
            n_layers: int = 2,
            pooling_sizes: list = None,
            downsample_frequencies: list = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        """
        N-HiTS model.

        Parameters
        ----------
        n_time_in: int
            Multiplier to get insample size.
            Insample size = n_time_in * output_size
        n_time_out: int
            Forecast horizon.
        shared_weights: bool
            If True, repeats first block.
        activation: str
            Activation function.
            An item from ['relu', 'softplus', 'tanh', 'selu', 'lrelu', 'prelu', 'sigmoid'].
        initialization: str
            Initialization function.
            An item from ['orthogonal', 'he_uniform', 'glorot_uniform', 'glorot_normal', 'lecun_normal'].
        stack_types: List[str]
            List of stack types.
            Subset from ['identity'].
        n_blocks: List[int]
            Number of blocks for each stack type.
            Note that len(n_blocks) = len(stack_types).
        n_layers: List[int]
            Number of layers for each stack type.
            Note that len(n_layers) = len(stack_types).
        n_theta_hidden: List[List[int]]
            Structure of hidden layers for each stack type.
            Each internal list should contain the number of units of each hidden layer.
            Note that len(n_theta_hidden) = len(stack_types).
        n_pool_kernel_size List[int]:
            Pooling size for input for each stack.
            Note that len(n_pool_kernel_size) = len(stack_types).
        n_freq_downsample List[int]:
            Downsample multiplier of output for each stack.
            Note that len(n_freq_downsample) = len(stack_types).
        batch_normalization: bool
            Whether perform batch normalization.
        dropout_prob_theta: float
            Float between (0, 1).
            Dropout for Nbeats basis.
        """

        n_stacks = len(n_blocks)
        covariate_size = 0
        if self.use_feat_idx_emb:
            covariate_size = covariate_size + self.feat_idx_emb_dim
        if self.use_time_feat:
            covariate_size = covariate_size + self.time_feat_dim
        self.covariate_size = covariate_size
        self.output_size = output_size
        self.naive_level = naive_level

        n_layers = [n_layers] * n_stacks
        hidden_size = n_stacks * [2 * [hidden_size]]

        if pooling_sizes is None:
            pooling_sizes = np.exp2(np.round(np.linspace(0.49, np.log2(self.prediction_length / 2), n_stacks)))
            pooling_sizes = [int(x) for x in pooling_sizes[::-1]]

        if downsample_frequencies is None:
            downsample_frequencies = [min(self.prediction_length, int(np.power(x, 1.5))) for x in pooling_sizes]

        blocks = self.create_stack(
            n_blocks=n_blocks,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            output_size=output_size,
            covariate_size=covariate_size,
            static_size=static_size,
            static_hidden_size=static_hidden_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            batch_normalization=batch_normalization,
            dropout=dropout,
            activation=activation,
            shared_weights=shared_weights,
            initialization=initialization,
        )
        self.blocks = torch.nn.ModuleList(blocks)
        self.loss_fn = lambda dist, targets: -dist.log_prob(targets)
        self.linear_mu = nn.Linear(self.prediction_length, self.prediction_length)
        self.linear_sigma = nn.Linear(self.prediction_length, self.prediction_length)

    def create_stack(
            self,
            n_blocks,
            context_length,
            prediction_length,
            output_size,
            covariate_size,
            static_size,
            static_hidden_size,
            n_layers,
            hidden_size,
            pooling_sizes,
            downsample_frequencies,
            pooling_mode,
            interpolation_mode,
            batch_normalization,
            dropout,
            activation,
            shared_weights,
            initialization,
    ):
        block_list = []

        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):
                # Batch norm only on first block
                if (len(block_list) == 0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    n_theta = max(prediction_length // downsample_frequencies[i], 1)
                    basis = IdentityBasis(
                        backcast_size=context_length,
                        forecast_size=prediction_length,
                        interpolation_mode=interpolation_mode,
                    )

                    nbeats_block = NHiTSBlock(
                        context_length=context_length,
                        prediction_length=prediction_length,
                        output_size=output_size,
                        covariate_size=covariate_size,
                        static_size=static_size,
                        static_hidden_size=static_hidden_size,
                        n_theta=n_theta,
                        hidden_size=hidden_size[i],
                        pooling_sizes=pooling_sizes[i],
                        pooling_mode=pooling_mode,
                        basis=basis,
                        n_layers=n_layers[i],
                        batch_normalization=batch_normalization_block,
                        dropout=dropout,
                        activation=activation,
                    )

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def encoder(self, encoder_y, encoder_x_t, decoder_x_t):
        # encoder_y: [B L D]
        residuals = (encoder_y)
        level = encoder_y[:, -1:].repeat(1, self.prediction_length, 1)  # Level with Naive1
        forecast_level = level.repeat_interleave(torch.tensor(self.output_size, device=level.device), dim=2)

        # level with last available observation
        if self.naive_level:
            block_forecasts = [forecast_level]
            forecast = block_forecasts[0]
        else:
            block_forecasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)

        # forecast by block
        for block in self.blocks:
            block_backcast, block_forecast = block(
                encoder_y=residuals, encoder_x_t=encoder_x_t, decoder_x_t=decoder_x_t
            )
            residuals = (residuals - block_backcast)  # * encoder_mask

            forecast = forecast + block_forecast
        return forecast

    def get_cov(self, inputs):
        if self.use_feat_idx_emb:
            if self.use_time_feat:
                encoder_dim_fea = inputs[:, : self.context_length, self.target_dim:-self.time_feat_dim]  # [B L K*D]
                decoder_dim_fea = inputs[:, -self.prediction_length:, self.target_dim:-self.time_feat_dim]  # [B L K*D]
            else:
                encoder_dim_fea = inputs[:, : self.context_length, self.target_dim:]  # [B L K*D]
                decoder_dim_fea = inputs[:, -self.prediction_length:, self.target_dim:]  # [B L K*D]

            encoder_dim_fea = rearrange(encoder_dim_fea, "b l (k d) -> (b k) l d", k=self.target_dim,
                                        d=self.feat_idx_emb_dim)
            decoder_dim_fea = rearrange(decoder_dim_fea, "b l (k d) -> (b k) l d", k=self.target_dim,
                                        d=self.feat_idx_emb_dim)
        else:
            encoder_dim_fea = []

        if self.time_feat_dim:
            encoder_time_fea = inputs[:, : self.context_length, -self.time_feat_dim:]  # [B L Dt]
            encoder_time_fea = repeat(encoder_time_fea, 'b l d -> (b k) l d', k=self.target_dim)

            decoder_time_fea = inputs[:, -self.prediction_length:, -self.time_feat_dim:]  # [B L Dt]
            decoder_time_fea = repeat(decoder_time_fea, 'b l d -> (b k) l d', k=self.target_dim)

        else:
            encoder_time_fea = []

        if self.use_feat_idx_emb and self.use_time_feat:
            encoder_x_t = torch.cat([encoder_dim_fea, encoder_time_fea], dim=-1)
            decoder_x_t = torch.cat([decoder_dim_fea, decoder_time_fea], dim=-1)
        elif self.use_feat_idx_emb:
            encoder_x_t, decoder_x_t = encoder_dim_fea, decoder_dim_fea
        elif self.use_time_feat:
            encoder_x_t, decoder_x_t = encoder_time_fea, decoder_time_fea
        else:
            encoder_x_t, decoder_x_t = None, None
        return encoder_x_t, decoder_x_t

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')  # [B L D]

        # Encode
        encoder_y = inputs[:, : self.context_length, :self.target_dim]  # [B L K]
        encoder_y = rearrange(encoder_y, "b l k -> (b k) l 1")
        encoder_x_t, decoder_x_t = self.get_cov(inputs)
        outputs = self.encoder(encoder_y, encoder_x_t, decoder_x_t)
        outputs = rearrange(outputs, "(b k) l 1 -> b l k", k=self.target_dim)

        mu = self.linear_mu(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        sigma = self.linear_sigma(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        dist = Normal(loc=mu, scale=F.softplus(sigma) + 1e-6)
        loss = self.loss_fn(dist, batch_data.future_target_cdf)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'all')  # [B L D]
        encoder_y = inputs[:, : self.context_length, :self.target_dim]  # [B L K]
        encoder_y = rearrange(encoder_y, "b l k -> (b k) l 1")
        encoder_x_t, decoder_x_t = self.get_cov(inputs)
        output = self.encoder(encoder_y, encoder_x_t, decoder_x_t)
        outputs = rearrange(output, "(b k) l 1 -> b l k", k=self.target_dim)
        mu = self.linear_mu(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        sigma = self.linear_sigma(outputs.permute(0, 2, 1)).permute(0, 2, 1)
        dist = Normal(loc=mu, scale=F.softplus(sigma) + 1e-6)
        return dist.rsample(sample_shape=(num_samples,))
