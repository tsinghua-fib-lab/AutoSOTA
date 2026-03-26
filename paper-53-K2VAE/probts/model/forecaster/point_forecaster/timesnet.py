# ---------------------------------------------------------------------------------
# Portions of this file are derived from TSLib
# - Source: https://github.com/libts/tslib
# - Paper: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
# - License:  LGPL-2.1

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.TransformerModule.Embed import DataEmbedding
from probts.model.nn.arch.Conv_Blocks import Inception_Block_V1
from torch.distributions import Normal


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, context_length, prediction_length, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = context_length
        self.pred_len = prediction_length
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(Forecaster):
    def __init__(
            self,
            n_layers: int = 2,
            num_kernels: int = 6,
            top_k: int = 5,
            d_ff: int = 32,
            embed: str = 'timeF',
            dropout: float = 0.1,
            f_hidden_size: int = 40,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.seq_len = self.context_length
        self.pred_len = self.prediction_length

        self.model = nn.ModuleList(
            [TimesBlock(self.context_length, self.prediction_length, top_k, f_hidden_size, d_ff, num_kernels)
             for _ in range(n_layers)]
        )
        self.enc_embedding = DataEmbedding(self.target_dim, f_hidden_size, embed, self.freq.lower(), dropout)
        self.layer = n_layers
        self.layer_norm = nn.LayerNorm(f_hidden_size)

        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection_mu = nn.Linear(
            f_hidden_size, self.target_dim, bias=True)
        self.projection_sigma = nn.Linear(
            f_hidden_size, self.target_dim, bias=True)

        if self.input_size != self.target_dim:
            self.enc_linear = nn.Linear(
                in_features=self.input_size, out_features=self.target_dim
            )
        else:
            self.enc_linear = nn.Identity()
        self.loss_fn = lambda dist, targets: -dist.log_prob(targets)

    def forward(self, x_enc, x_mark_enc=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        mu = self.projection_mu(enc_out)
        sigma = self.projection_sigma(enc_out)

        # De-Normalization from Non-stationary Transformer
        mu = mu * \
             (stdev[:, 0, :].unsqueeze(1).repeat(
                 1, self.pred_len + self.seq_len, 1))
        sigma = sigma + \
                (means[:, 0, :].unsqueeze(1).repeat(
                    1, self.pred_len + self.seq_len, 1))
        return Normal(loc=mu[:, -self.pred_len:, :], scale=F.softplus(sigma[:, -self.pred_len:, :]) + 1e-6)  # [B, L, D]

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        inputs = inputs[:, : self.context_length, ...]
        inputs = self.enc_linear(inputs)
        # x: [Batch, Input length, Channel]
        dist = self(inputs)
        targets = batch_data.future_target_cdf
        loss = self.loss_fn(dist, targets)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        inputs = self.get_inputs(batch_data, 'encode')
        inputs = self.enc_linear(inputs)
        dist = self(inputs)
        return dist.rsample(sample_shape=(num_samples,))
