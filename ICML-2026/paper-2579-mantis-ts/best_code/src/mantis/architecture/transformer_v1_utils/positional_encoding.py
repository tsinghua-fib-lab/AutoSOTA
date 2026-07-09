import torch
import math

from torch import nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoder.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor of shape ``[seq_len, batch_size, d_model]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
