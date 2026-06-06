import torch
import torch.nn as nn

import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted

class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, num_channels, num_features, eps=1e-5):
        super(ChannelwiseLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.eps = eps
        
        # Learnable weights and biases for each channel
        self.weight = nn.Parameter(torch.ones(num_channels, num_features))
        self.bias = nn.Parameter(torch.zeros(num_channels, num_features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, num_channels, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch_size, num_channels, 1)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (batch_size, num_channels, num_features)
        x_out = x_normalized * self.weight + self.bias  # (batch_size, num_channels, num_features)
        return x_out
    
    
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            ChannelwiseLayerNorm(configs.enc_in,configs.d_model),
            nn.Linear(configs.d_model, configs.d_model)
        )
        
        self.temporal2 = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            ChannelwiseLayerNorm(configs.enc_in,configs.d_model),
            nn.Linear(configs.d_model, configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len)

        # self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.enc_in)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # Embedding
        embed_dim = configs.d_model 
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, embed_dim, configs.embed, configs.freq,
                                                    configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None

    def forecast(self, x_enc):
        # x: [B, L, D]
        x_enc = self.rev(x_enc, 'norm')

        B, _, N = x_enc.shape
        # Embedding
        x_enc = self.enc_embedding(x_enc, None)  # [b n d]
        ## Local adaptation
        x_enc = x_enc.transpose(1, 2)  # [b n D] -> [b D n]

        x_enc = x_enc + self.temporal(x_enc.transpose(1, 2)).transpose(1, 2)
        x_enc = x_enc + self.temporal2(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm')

        return pred
        
